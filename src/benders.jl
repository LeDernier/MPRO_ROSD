include("struct/tree.jl")
using JuMP
using CPLEX

function benders(x::Matrix{Float64}, clusters::Vector{Cluster}, D::Int64, classes; time_limit=(-1), mu::Float64=(10^(-4)))

    start_time = time()

    clusterCount = length(clusters) # Nombre de clusters de données d'entraînement
    featuresCount = size(clusters[1].x, 2) # Nombre de caractéristiques
    classCount = length(classes) # Nombre de classes différentes
    sepCount = 2^D - 1 # Nombre de séparations de l'arbre
    leavesCount = 2^D # Nombre de feuilles de l'arbre
    dataCount = sum(length(clusters[i].dataIds) for i in 1:clusterCount) # Nombre de données d'entraînement total

    mp = Model(CPLEX.Optimizer)
    set_silent(mp) # Masque les sorties du solveur
    if time_limit != -1
        set_time_limit_sec(mp, time_limit + start_time - time())
    end


    # Déclaration des variables
    @variable(mp, 0 <= r[1:dataCount] <= 1, base_name = "r")
    @variable(mp, z, base_name = "z")

    # Fonction objectif
    @objective(mp, Max, z)

    # Contraintes
    @constraint(mp, [i in 1:clusterCount], sum(r[j] for j in clusters[i].dataIds) == 1) # Un seul représentant par cluster
    @constraint(mp, z <= dataCount) # z borné par le nombre de données

    all_w_i = []
    current_r = zeros(Float64, clusterCount)
    current_z = 0
    updated = true
    while updated && (time_limit == -1 || time() - start_time < time_limit)

        optimize!(mp)

        # Si une solution a été trouvée
        if primal_status(mp) == MOI.FEASIBLE_POINT
            # Si une solution optimale a été trouvée
            if termination_status(mp) == MOI.OPTIMAL
                current_r = value.(r) # Représentants des clusters
                # Si ce n'est pas la première itération
                if !isempty(all_w_i)
                    new_r = zeros(Int, clusterCount)
                    w_c = []
                    # Pour chaque cluster on cherche le représentant parmi ceux qui ont une valeur > 0 qui minimise w_i
                    for c in 1:clusterCount
                        w_c_values = [(w[i], i) for w in all_w_i, i in clusters[c].dataIds if current_r[i] > 0]
                        w_c_index = argmin(w_c_values)
                        push!(w_c, w_c_values[w_c_index])
                        new_r[c] = clusters[c].dataIds[w_c_index] # Nos choix des représentants par rapport a r_i relaché
                    end
                    current_r = new_r
                else
                    current_r = round.(Int, findall(x -> x > 0.5, current_r))
                end
                current_z = value(z) # Valeur de l'objectif
            end
        end


        sp = Model(CPLEX.Optimizer)
        set_silent(sp) # Masque les sorties du solveur
        if time_limit != -1
            set_time_limit_sec(sp, time_limit)
        end

        # Plus petite différence entre deux données pour une caractéristique
        mu_min = 1.0
        # Plus grande différence entre deux données pour une caractéristique
        mu_max = 0.0

        mu_vect = ones(Float64, featuresCount)
        for j in 1:featuresCount
            for i1 in 1:clusterCount
                for i2 in (i1+1):clusterCount
                    if abs(clusters[i1].barycenter[j] - clusters[i2].barycenter[j]) > 1E-4
                        mu_vect[j] = min(mu_vect[j], abs(clusters[i1].barycenter[j] - clusters[i2].barycenter[j]))
                    end
                end
            end
            mu_min = min(mu_min, mu_vect[j])
            mu_max = max(mu_max, mu_vect[j])
        end

        ## Déclaraction des variables
        @variable(sp, a[1:featuresCount, 1:sepCount] >= 0, base_name = "a")
        @variable(sp, b[1:sepCount], base_name = "b_t")
        @variable(sp, c[1:classCount, 1:(sepCount+leavesCount)] >= 0, base_name = "c_{k, t}")
        @variable(sp, u_at[1:clusterCount, 1:(sepCount+leavesCount)] >= 0, base_name = "u^i_{a(t), t}")
        @variable(sp, u_tw[1:clusterCount, 1:(sepCount+leavesCount)] >= 0, base_name = "u^i_{t, w}")

        ## Déclaration des contraintes

        # Contraintes définissant la structure de l'arbre

        @constraint(sp, [t in 1:sepCount], sum(a[j, t] for j in 1:featuresCount) + sum(c[k, t] for k in 1:classCount) == 1) # on s'assure que le noeud applique une règle de branchement OU attribue une classe
        @constraint(sp, [t in 1:sepCount], b[t] <= sum(a[j, t] for j in 1:featuresCount)) # b doit être nul si il n'y a pas de branchement 
        @constraint(sp, [t in 1:sepCount], b[t] >= 0) # b doit être positif
        @constraint(sp, [t in (sepCount+1):(sepCount+leavesCount)], sum(c[k, t] for k in 1:classCount) == 1) # on s'assure qu'on attribue une classe par feuille

        # contraintes de conservation du flot et contraintes de capacité
        @constraint(sp, [i in 1:clusterCount, t in 1:sepCount], u_at[i, t] == u_at[i, t*2] + u_at[i, t*2+1] + u_tw[i, t]) # conservation du flot dans les noeuds de branchement
        @constraint(sp, [i in 1:clusterCount, t in (sepCount+1):(sepCount+leavesCount)], u_at[i, t] == u_tw[i, t]) # conservation du flot dans les feuilles
        @constraint(sp, [i in 1:clusterCount, t in 1:(sepCount+leavesCount)], u_tw[i, t] <= c[findfirst(classes .== clusters[i].class), t]) # contrainte de capacité qui impose le flot a etre nul si la classe de la feuille n'est pas la bonne
        @constraint(sp, con1[i in 1:clusterCount, t in 1:sepCount], sum(a[j, t] * (clusters[i].x[current_r[i], j] + mu_vect[j] - mu_min) for j in 1:featuresCount) + mu_min <= b[t] + (1 + mu_max) * (1 - u_at[i, t*2])) # contrainte de capacité controlant le passage dans le noeud fils gauche
        @constraint(sp, con2[i in 1:clusterCount, t in 1:sepCount], sum(a[j, t] * clusters[i].x[current_r[i], j] for j in 1:featuresCount) >= b[t] - (1 - u_at[i, t*2+1])) # contrainte de capacité controlant le passage dans le noeud fils droit
        @constraint(sp, [i in 1:clusterCount, t in 1:sepCount], u_at[i, t*2+1] <= sum(a[j, t] for j in 1:featuresCount)) # contrainte de capacité empechant les données de passer dans le fils droit d'un noeud n'appliquant pas de règle de branchement
        @constraint(sp, [i in 1:clusterCount], u_at[i, 1] <= 1)

        ## Déclaration de l'objectif
        @objective(sp, Max, sum(length(clusters[i].dataIds) * u_at[i, 1] for i in 1:clusterCount))

        optimize!(sp)

        # Récupération des valeurs des variables duales
        gamma_1_t_C = JuMP.dual.(con1)
        gamma_2_t_C = JuMP.dual.(con2)
        z_bar = JuMP.dual_objective_value(sp)

        w = [sum((1 + mu_max) * gamma_1_t_C[c, t] + gamma_2_t_C[c, t] for t in 1:sepCount) for c in 1:clusterCount]

        # On stocke les valeurs des w_i pour les utiliser dans la prochaine itération
        # On remplit les valeurs manquantes avec Inf
        w_i = [Inf for _ in 1:dataCount]
        for i in 1:clusterCount
            w_i[current_r[i]] = w[i]
        end
        push!(all_w_i, w_i)

        # Si la solution du MP est supérieure ou égale à la solution du SP on ajoute une coupe
        if current_z > z_bar + 10^-4
            updated = true
            println("w = ", w)
            @constraint(mp, z <= z_bar + sum(w[c] * (1 - r[current_r[c]]) for c in 1:clusterCount))
            println("new constraint added")
            #println(mp)
        else
            updated = false
        end

    end


    # On résoud le problème une dernière fois en MILP (sans relacher les variables) pour obtenir la solution finale du problème qui nous donne l'arbre de décision

    sp = Model(CPLEX.Optimizer)
    set_silent(sp) # Masque les sorties du solveur
    if time_limit != -1
        set_time_limit_sec(sp, time_limit)
    end

    # Plus petite différence entre deux données pour une caractéristique
    mu_min = 1.0
    # Plus grande différence entre deux données pour une caractéristique
    mu_max = 0.0

    mu_vect = ones(Float64, featuresCount)
    for j in 1:featuresCount
        for i1 in 1:clusterCount
            for i2 in (i1+1):clusterCount
                if abs(clusters[i1].barycenter[j] - clusters[i2].barycenter[j]) > 1E-4
                    mu_vect[j] = min(mu_vect[j], abs(clusters[i1].barycenter[j] - clusters[i2].barycenter[j]))
                end
                """
                v1 = clusters[i1].lBounds[j] - clusters[i2].uBounds[j]
                v2 = clusters[i2].lBounds[j] - clusters[i1].uBounds[j]

                # Si les clusters n'ont pas des intervalles pour la caractéristique j qui s'intersectent
                if v1 > 0 || v2 > 0
                    vMin = min(abs(v1), abs(v2))
                    mu_vect[j] = min(mu_vect[j], vMin)
                end"""
            end
        end
        mu_min = min(mu_min, mu_vect[j])
        mu_max = max(mu_max, mu_vect[j])
    end

    ## Déclaraction des variables
    @variable(sp, a[1:featuresCount, 1:sepCount], Bin, base_name = "a")
    @variable(sp, b[1:sepCount], base_name = "b_t")
    @variable(sp, c[1:classCount, 1:(sepCount+leavesCount)], Bin, base_name = "c_{k, t}")
    @variable(sp, u_at[1:clusterCount, 1:(sepCount+leavesCount)], Bin, base_name = "u^i_{a(t), t}")
    @variable(sp, u_tw[1:clusterCount, 1:(sepCount+leavesCount)], Bin, base_name = "u^i_{t, w}")

    ## Déclaration des contraintes

    # Contraintes définissant la structure de l'arbre

    @constraint(sp, con1[t in 1:sepCount], sum(a[j, t] for j in 1:featuresCount) + sum(c[k, t] for k in 1:classCount) == 1) # on s'assure que le noeud applique une règle de branchement OU attribue une classe
    @constraint(sp, con2[t in 1:sepCount], b[t] <= sum(a[j, t] for j in 1:featuresCount)) # b doit être nul si il n'y a pas de branchement 
    @constraint(sp, [t in 1:sepCount], b[t] >= 0) # b doit être positif
    @constraint(sp, con3[t in (sepCount+1):(sepCount+leavesCount)], sum(c[k, t] for k in 1:classCount) == 1) # on s'assure qu'on attribue une classe par feuille

    # contraintes de conservation du flot et contraintes de capacité
    @constraint(sp, con4[i in 1:clusterCount, t in 1:sepCount], u_at[i, t] == u_at[i, t*2] + u_at[i, t*2+1] + u_tw[i, t]) # conservation du flot dans les noeuds de branchement
    @constraint(sp, con5[i in 1:clusterCount, t in (sepCount+1):(sepCount+leavesCount)], u_at[i, t] == u_tw[i, t]) # conservation du flot dans les feuilles
    @constraint(sp, con6[i in 1:clusterCount, t in 1:(sepCount+leavesCount)], u_tw[i, t] <= c[findfirst(classes .== clusters[i].class), t]) # contrainte de capacité qui impose le flot a etre nul si la classe de la feuille n'est pas la bonne
    @constraint(sp, con7[i in 1:clusterCount, t in 1:sepCount], sum(a[j, t] * (x[current_r[i], j] + mu_vect[j] - mu_min) for j in 1:featuresCount) + mu_min <= b[t] + (1 + mu_max) * (1 - u_at[i, t*2])) # contrainte de capacité controlant le passage dans le noeud fils gauche
    @constraint(sp, con8[i in 1:clusterCount, t in 1:sepCount], sum(a[j, t] * x[current_r[i], j] for j in 1:featuresCount) >= b[t] - (1 - u_at[i, t*2+1])) # contrainte de capacité controlant le passage dans le noeud fils droit
    @constraint(sp, con9[i in 1:clusterCount, t in 1:sepCount], u_at[i, t*2+1] <= sum(a[j, t] for j in 1:featuresCount)) # contrainte de capacité empechant les données de passer dans le fils droit d'un noeud n'appliquant pas de règle de branchement
    @constraint(sp, con10[i in 1:clusterCount], u_at[i, 1] <= 1)

    ## Déclaration de l'objectif
    @objective(sp, Max, sum(length(clusters[i].dataIds) * u_at[i, 1] for i in 1:clusterCount))

    optimize!(sp)

    gap = -1.0

    # Arbre obtenu (vide si le solveur n'a trouvé aucune solution)
    T = nothing
    objective = -1

    # Si une solution a été trouvée
    if primal_status(sp) == MOI.FEASIBLE_POINT
        # class[t] : classe prédite par le sommet t
        class = Vector{Int64}(undef, sepCount + leavesCount)
        for t in 1:(sepCount+leavesCount)
            k = argmax(value.(c[:, t]))
            if value.(c[k, t]) >= 1.0 - 10^-4
                class[t] = k
            else
                class[t] = -1
            end
        end

        objective = JuMP.objective_value(sp)

        # Si une solution optimale a été trouvée
        if termination_status(sp) == MOI.OPTIMAL
            gap = 0
        else
            # Calcul du gap relatif entre l'objectif de la meilleure solution entière et la borne continue en fin de résolution
            bound = JuMP.objective_bound(sp)
            gap = 100.0 * abs(objective - bound) / (objective + 10^-4) # +10^-4 permet d'éviter de diviser par 0
        end

        # Construction d'une variable de type Tree dans laquelle chaque séparation est recentrée
        T = Tree(D, value.(a), class, round.(Int, value.(u_at)), clusters)
    end

    return T, objective, time() - start_time, gap
end


function iteratively_build_tree_benders(clusters::Vector{Cluster}, D::Int64, x::Matrix{Float64}, y::Vector{}, classes::Vector{}; time_limit::Int64=-1, mu::Float64=10^(-4))

    startingTime = time()

    isExact = true
    finalTime = startingTime + time_limit

    # Define variables used as return values
    # (otherwise they would not be defined outside of the while loop)
    lastObjective = nothing
    lastFeasibleT = nothing
    gap = nothing

    clusterSplit = true
    iterationCount = 0

    useFeS = isExact

    # While cluster are split and the time limit is not reached
    while clusterSplit && (time_limit == -1 || time() < finalTime - 5)

        iterationCount += 1
        remainingTime = round(Int, finalTime - time())

        # Solve with the current clusters
        T, objective, resolution_time, gap = benders(x, clusters, D, classes, time_limit=time_limit == -1 ? -1 : remainingTime)

        # If a solution has been obtained
        if objective != -1

            # List of the clusters for the next iteration
            newClusters = Vector{Cluster}()

            # For each cluster
            for cluster in clusters

                # Split its data according to the leaves of tree T they reach
                newCurrentClusters = getSplitClusters(cluster, T)
                append!(newClusters, newCurrentClusters)
            end

            # If no cluster is split
            if length(clusters) == length(newClusters)
                clusterSplit = false
            else
                clusters = newClusters
            end
            lastFeasibleT = T
            lastObjective = objective
        end
    end

    resolution_time = time() - startingTime
    return lastFeasibleT, lastObjective, resolution_time, gap, iterationCount
end