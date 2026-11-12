def select_parents(population, rng, cost_evaluator, k=2):
    """
    Elite-Protected Stratified Niche Tournament Selection (EP-SNTS).

    Combines Elite Protection with Vidal's k-NN diversity contribution,
    stratified candidate sampling to guarantee non-inbred mating partners,
    and a multi-faceted Goldilocks compatibility score (edge distance,
    route-count consistency, and feasibility synergy).
    """
    n = len(population)
    assert n >= 2, "Population must have at least two solutions."

    # Fast path for minimal population
    if n == 2:
        c0 = cost_evaluator.penalised_cost(population[0])
        c1 = cost_evaluator.penalised_cost(population[1])
        return (
            (population[0], population[1])
            if c0 <= c1
            else (population[1], population[0])
        )

    # 1. Safe extraction of canonical undirected edges (PyVRP 0.14+ compatible)
    def get_solution_edges(solution):
        edges = set()
        for route in solution.routes():
            prev = 0
            for act in route:
                c = getattr(act, "client", act)
                if isinstance(c, int) and c > 0:
                    edges.add((prev, c) if prev < c else (c, prev))
                    prev = c
            if prev > 0:
                edges.add((0, prev))
        return edges

    all_edges = [get_solution_edges(sol) for sol in population]
    costs = [cost_evaluator.penalised_cost(sol) for sol in population]

    # 2. Exact pairwise Jaccard distance matrix
    dist_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        e_i = all_edges[i]
        len_i = len(e_i)
        for j in range(i + 1, n):
            e_j = all_edges[j]
            intersection = len(e_i & e_j)
            union = len_i + len(e_j) - intersection
            d = 1.0 - (intersection / float(union)) if union > 0 else 0.0
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d

    # 3. Vidal's k-Nearest Neighbors Diversity Contribution
    k_close = max(1, min(4, int(0.20 * n)))
    div_contribution = [0.0] * n
    for i in range(n):
        neighbors = [dist_matrix[i][j] for j in range(n) if j != i]
        neighbors.sort()
        div_contribution[i] = sum(neighbors[:k_close]) / float(k_close)

    # 4. Elite-Protected Biased Fitness Ranking
    cost_order = sorted(range(n), key=lambda i: costs[i])
    cost_ranks = [0] * n
    for rank, idx in enumerate(cost_order):
        cost_ranks[idx] = rank

    div_order = sorted(range(n), key=lambda i: div_contribution[i], reverse=True)
    div_ranks = [0] * n
    for rank, idx in enumerate(div_order):
        div_ranks[idx] = rank

    n_elite = max(1, int(0.18 * n))
    div_weight = 1.0 - (float(n_elite) / float(n))

    biased_fitness = [0.0] * n
    for i in range(n):
        c_r = cost_ranks[i]
        if c_r < n_elite:
            # Top elite solutions are strictly evaluated by cost
            biased_fitness[i] = float(c_r)
        else:
            # Non-elites balance cost rank and diversity contribution
            biased_fitness[i] = float(c_r) + div_weight * float(div_ranks[i])

    # Helper: sample `size` distinct items from `pool` without replacement
    def sample_distinct(size, pool):
        m = len(pool)
        if m <= size:
            return list(pool)
        chosen = set()
        result = []
        while len(result) < size:
            idx = pool[rng.randint(m)]
            if idx not in chosen:
                chosen.add(idx)
                result.append(idx)
        return result

    # 5. Tournament Selection for Parent 1 (Backbone parent)
    t1_size = min(5, n)
    candidates1 = sample_distinct(t1_size, list(range(n)))
    parent1_idx = min(candidates1, key=lambda i: biased_fitness[i])
    parent1 = population[parent1_idx]
    p1_routes = parent1.num_routes()
    p1_feas = parent1.is_feasible()

    # 6. Stratified Candidate Sampling for Parent 2 (Donor parent)
    t2_pool = [i for i in range(n) if i != parent1_idx]
    t2_size = min(5, len(t2_pool))

    # Partition by structural compatibility to prevent convergence-induced inbreeding
    compatible_pool = [i for i in t2_pool if dist_matrix[parent1_idx][i] >= 0.12]
    other_pool = [i for i in t2_pool if dist_matrix[parent1_idx][i] < 0.12]

    if len(compatible_pool) >= t2_size:
        candidates2 = sample_distinct(t2_size, compatible_pool)
    elif len(compatible_pool) > 0:
        candidates2 = list(compatible_pool)
        needed = t2_size - len(candidates2)
        candidates2.extend(sample_distinct(needed, other_pool))
    else:
        candidates2 = sample_distinct(t2_size, other_pool)

    # 7. Multi-Faceted Compatibility Scoring
    def mate_score(j):
        d = dist_matrix[parent1_idx][j]

        # Edge distance component
        if d < 1e-6:
            dist_pen = 10000.0  # Exact structural clone: avoid
        elif d < 0.12:
            # Quadratic inbreeding penalty for near-clones
            ratio = (0.12 - d) / 0.12
            dist_pen = (ratio * ratio) * (2.5 * n)
        elif 0.20 <= d <= 0.50:
            # Recombination sweet spot: active synergy bonus
            dist_pen = -0.15 * n * (1.0 - abs(d - 0.35) / 0.15)
        elif d > 0.65:
            # Outbreeding disruption penalty
            dist_pen = ((d - 0.65) / 0.35) * (0.8 * n)
        else:
            dist_pen = 0.0

        # Route-count alignment penalty
        route_diff = abs(p1_routes - population[j].num_routes())
        route_pen = 0.5 * max(0, route_diff - 1)

        # Feasibility synergy adjustment
        j_feas = population[j].is_feasible()
        if not p1_feas and j_feas:
            feas_adj = -0.10 * n  # Feasible donor helps repair infeasible parent
        elif p1_feas and j_feas:
            feas_adj = -0.05 * n
        elif p1_feas and not j_feas:
            feas_adj = 0.05 * n
        else:
            feas_adj = 0.0

        return biased_fitness[j] + dist_pen + route_pen + feas_adj

    best_candidate = min(candidates2, key=mate_score)

    # Fallback check: if tournament selected an exact clone but non-clones exist
    if dist_matrix[parent1_idx][best_candidate] < 1e-6:
        non_exact_pool = [i for i in t2_pool if dist_matrix[parent1_idx][i] > 1e-6]
        if non_exact_pool:
            parent2_idx = min(non_exact_pool, key=mate_score)
        else:
            parent2_idx = best_candidate
    else:
        parent2_idx = best_candidate

    return (parent1, population[parent2_idx])