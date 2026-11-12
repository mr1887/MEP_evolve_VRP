def select_parents(population, rng, cost_evaluator, k=2):
    """
    Dual-Scale k-NN Complementarity Selection (DKCS) for PyVRP 0.14+.

    Innovations:
    1. Vidal's k-NN Local Density Metric: Evaluates diversity contribution via
       the k closest neighbors to aggressively penalize tight clusters and clones.
    2. Champion-Only Protection: Unconditionally protects only the single best
       solution (cost rank 0); all other solutions must balance cost and diversity.
    3. Dual-Scale Structural Distance: Combines canonical edge Jaccard distance
       (micro-level) with route customer-partition overlap (macro-level).
    4. Goldilocks Recombination Zone: Rewards Parent 2 candidates in the optimal
       structural distance window (0.30 - 0.65) to prevent both inbreeding and
       destructive outbreeding.
    5. Trans-Boundary Feasibility Repair: If Parent 1 is infeasible, actively
       steers Parent 2 selection toward feasible partners.
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

    # 1. Safe extraction of canonical edges and route customer sets (PyVRP 0.14+)
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

    def get_route_sets(solution):
        route_sets = []
        for route in solution.routes():
            clients = frozenset(
                getattr(act, "client", act)
                for act in route
                if isinstance(getattr(act, "client", act), int)
                and getattr(act, "client", act) > 0
            )
            if clients:
                route_sets.append(clients)
        return route_sets

    all_edges = [get_solution_edges(sol) for sol in population]
    all_route_sets = [set(get_route_sets(sol)) for sol in population]
    costs = [cost_evaluator.penalised_cost(sol) for sol in population]
    is_feas = [sol.is_feasible() for sol in population]
    num_routes = [sol.num_routes() for sol in population]
    edge_lens = [len(e) for e in all_edges]

    # 2. Pairwise Edge Distance Matrix (exact and fast for population sizes <= 80)
    # Beyond 80, sample a reference set to preserve sub-millisecond execution.
    if n <= 80:
        dist_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            e_i = all_edges[i]
            len_i = edge_lens[i]
            for j in range(i + 1, n):
                inter = len(e_i & all_edges[j])
                union = len_i + edge_lens[j] - inter
                d = 1.0 - (inter / float(union)) if union > 0 else 0.0
                dist_matrix[i][j] = d
                dist_matrix[j][i] = d
    else:
        # Subsample reference pool for distance approximations
        step = max(1, n // 30)
        ref_indices = list(range(0, n, step))[:30]
        dist_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            e_i = all_edges[i]
            len_i = edge_lens[i]
            for j in ref_indices:
                if i == j:
                    continue
                inter = len(e_i & all_edges[j])
                union = len_i + edge_lens[j] - inter
                d = 1.0 - (inter / float(union)) if union > 0 else 0.0
                dist_matrix[i][j] = d
                dist_matrix[j][i] = d

    # 3. Vidal's k-NN Local Density Metric
    # Evaluates crowding as the average distance to the k_close nearest neighbors
    k_close = max(1, min(5, int(0.20 * n)))
    div_contrib = [0.0] * n
    for i in range(n):
        # Gather distances to other individuals
        neighbor_dists = [
            dist_matrix[i][j]
            for j in range(n)
            if j != i and dist_matrix[i][j] > 0.0 or j != i
        ]
        neighbor_dists.sort()
        div_contrib[i] = sum(neighbor_dists[:k_close]) / float(k_close)

    # 4. Biased Fitness with Champion-Only Immunity
    # Cost ranking (tie-broken by feasibility and route compactness)
    cost_order = sorted(
        range(n),
        key=lambda i: (costs[i], not is_feas[i], num_routes[i]),
    )
    cost_ranks = [0] * n
    for rank, idx in enumerate(cost_order):
        cost_ranks[idx] = rank

    # Diversity ranking: 0 is highest diversity (least crowded)
    div_order = sorted(range(n), key=lambda i: div_contrib[i], reverse=True)
    div_ranks = [0] * n
    for rank, idx in enumerate(div_order):
        div_ranks[idx] = rank

    div_weight = 1.0 - (float(max(1, int(0.15 * n))) / float(n))

    biased_fitness = [0.0] * n
    for i in range(n):
        c_r = cost_ranks[i]
        if c_r == 0:
            # Only the champion is immune to diversity penalty
            biased_fitness[i] = 0.0
        else:
            biased_fitness[i] = float(c_r) + div_weight * float(div_ranks[i])

    # Distinct sampling helper using rng.randint
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

    # 5. Tournament Selection for Parent 1
    t1_size = min(4, n)
    candidates1 = sample_distinct(t1_size, list(range(n)))
    parent1_idx = min(
        candidates1,
        key=lambda i: (biased_fitness[i], not is_feas[i], costs[i], num_routes[i]),
    )
    parent1 = population[parent1_idx]
    p1_feas = is_feas[parent1_idx]
    p1_route_set = all_route_sets[parent1_idx]

    # Helper: Dual-scale structural distance to Parent 1
    def dual_scale_dist(idx):
        d_edge = dist_matrix[parent1_idx][idx]
        shared_r = len(p1_route_set & all_route_sets[idx])
        max_r = max(len(p1_route_set), len(all_route_sets[idx]))
        d_route = 1.0 - (shared_r / float(max_r)) if max_r > 0 else 0.0
        return 0.70 * d_edge + 0.30 * d_route

    # Helper: Goldilocks Complementarity Response Function
    def complementarity_bonus(dist):
        if dist < 0.15:
            return (dist / 0.15) * 0.30
        if dist <= 0.65:
            # Recombination sweet spot (linear rise to peak)
            return 0.30 + 0.70 * ((dist - 0.15) / 0.50)
        # Moderate decay beyond 0.65 to penalize destructive outbreeding
        return max(0.0, 1.0 - 1.50 * (dist - 0.65))

    # 6. Tournament Selection for Parent 2
    t2_pool = [i for i in range(n) if i != parent1_idx]
    t2_size = min(6, len(t2_pool))
    candidates2 = sample_distinct(t2_size, t2_pool)

    # Filter out near-clones (dual-scale distance < 0.12)
    eligible = [
        i
        for i in candidates2
        if dual_scale_dist(i) >= 0.12 and population[i] != parent1
    ]

    # Fallbacks if all candidates in tournament are near-clones
    if not eligible:
        eligible = [
            i
            for i in candidates2
            if dual_scale_dist(i) > 0.0 and population[i] != parent1
        ]
    if not eligible:
        eligible = [
            i
            for i in t2_pool
            if dual_scale_dist(i) > 0.0 and population[i] != parent1
        ]
    if not eligible:
        # Full population convergence
        return (parent1, population[(parent1_idx + 1) % n])

    # Trans-boundary repair: prioritize feasible Parent 2 if Parent 1 is infeasible
    if not p1_feas:
        feas_cands = [i for i in eligible if is_feas[i]]
        if feas_cands:
            eligible = feas_cands

    # Select Parent 2: balance Biased Fitness with Goldilocks Complementarity
    parent2_idx = min(
        eligible,
        key=lambda i: (
            biased_fitness[i]
            - 0.35 * div_weight * n * complementarity_bonus(dual_scale_dist(i)),
            not is_feas[i],
            costs[i],
            num_routes[i],
        ),
    )

    return (parent1, population[parent2_idx])