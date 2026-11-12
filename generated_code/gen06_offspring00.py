def select_parents(population, rng, cost_evaluator, k=2):
    """
    Synergistic Sweet-Spot & Adaptive Biased Selection (S3-ABS) for PyVRP 0.14+.

    Innovations:
    1. Safe undirected edge extraction compatible with PyVRP 0.14 ScheduledActivity objects.
    2. Smooth Adaptive Biased Fitness (SABF): Replaces the discontinuous elite cliff
       with a continuous diversity-weighting function that maintains strong cost pressure
       while actively preventing clone stagnation among top individuals.
    3. Tournament selection for Parent 1 (Anchor) with moderate pressure (k=4).
    4. Sweet-Spot Complementarity for Parent 2 (Donor):
       - Strict anti-incest filter (pruning near-clones with Jaccard distance < 0.12).
       - Penalizes structural deviation from the crossover 'Goldilocks zone' (d* ≈ 0.35).
       - Feasibility harmony: prioritizes feasible donor routes to repair infeasible anchors.
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

    # 1. Safe canonical undirected edge extraction for PyVRP 0.14+
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

    # 2. Global diversity: average Jaccard distance to representative pool
    if n <= 35:
        ref_indices = list(range(n))
    else:
        step = max(1, n // 25)
        ref_indices = list(range(0, n, step))[:25]

    avg_dist = [0.0] * n
    for i in range(n):
        e_i = all_edges[i]
        len_i = len(e_i)
        dist_sum = 0.0
        comparisons = 0
        for j in ref_indices:
            if i == j:
                continue
            e_j = all_edges[j]
            inter = len(e_i & e_j)
            union = len_i + len(e_j) - inter
            if union > 0:
                dist_sum += 1.0 - (inter / float(union))
            comparisons += 1
        avg_dist[i] = dist_sum / max(1, comparisons)

    # 3. Smooth Adaptive Biased Fitness (SABF)
    # Cost ranking: penalized cost with feasibility and route count tie-breakers
    cost_order = sorted(
        range(n),
        key=lambda i: (
            costs[i],
            not population[i].is_feasible(),
            population[i].num_routes(),
        ),
    )
    cost_ranks = [0] * n
    for rank, idx in enumerate(cost_order):
        cost_ranks[idx] = rank

    # Diversity ranking: highest average distance gets rank 0
    div_order = sorted(range(n), key=lambda i: avg_dist[i], reverse=True)
    div_ranks = [0] * n
    for rank, idx in enumerate(div_order):
        div_ranks[idx] = rank

    n_elite = max(1, int(0.20 * n))
    base_div_weight = 1.0 - (float(n_elite) / float(n))

    biased_fitness = [0.0] * n
    for i in range(n):
        c_r = cost_ranks[i]
        d_r = div_ranks[i]
        # Smooth weight transition: elites prioritize cost, but diversity still breaks ties
        adaptive_weight = base_div_weight * (
            0.40 + 0.60 * min(1.0, float(c_r) / float(n_elite))
        )
        biased_fitness[i] = float(c_r) + adaptive_weight * float(d_r)

    # Helper: Distinct index sampling without replacement
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

    # 4. Tournament Selection for Parent 1 (Anchor Parent)
    t1_size = min(4, n)
    candidates1 = sample_distinct(t1_size, list(range(n)))
    parent1_idx = min(
        candidates1,
        key=lambda i: (
            biased_fitness[i],
            costs[i],
            not population[i].is_feasible(),
            population[i].num_routes(),
        ),
    )
    parent1 = population[parent1_idx]
    p1_edges = all_edges[parent1_idx]
    len_p1 = len(p1_edges)
    p1_feas = parent1.is_feasible()

    # Pairwise Jaccard distance to Parent 1
    def dist_to_p1(idx):
        e_i = all_edges[idx]
        inter = len(p1_edges & e_i)
        union = len_p1 + len(e_i) - inter
        return 1.0 - (inter / float(union)) if union > 0 else 0.0

    # 5. Tournament Selection for Parent 2 (Donor Parent)
    t2_pool = [i for i in range(n) if i != parent1_idx]
    t2_size = min(5, len(t2_pool))
    candidates2 = sample_distinct(t2_size, t2_pool)

    # Anti-incest: filter near-clones (d < 0.12) to ensure meaningful crossover
    eligible = [
        i
        for i in candidates2
        if population[i] != parent1 and dist_to_p1(i) >= 0.12
    ]

    # Progressive fallback cascade under high population convergence
    if not eligible:
        eligible = [
            i
            for i in candidates2
            if population[i] != parent1 and dist_to_p1(i) > 0.0
        ]
    if not eligible:
        eligible = [
            i
            for i in t2_pool
            if population[i] != parent1 and dist_to_p1(i) >= 0.12
        ]
    if not eligible:
        eligible = [
            i
            for i in t2_pool
            if population[i] != parent1 and dist_to_p1(i) > 0.0
        ]
    if not eligible:
        # Population has fully converged to duplicates
        return (parent1, population[(parent1_idx + 1) % n])

    # Donor scoring: balance Biased Fitness, Recombination Sweet-Spot (d* = 0.35),
    # and feasibility synergy
    def donor_score(idx):
        d_p1 = dist_to_p1(idx)
        # Goldilocks sweet spot penalty (|d - 0.35|)
        sweet_spot_penalty = abs(d_p1 - 0.35) * (0.45 * float(n))
        # Feasibility synergy: strong incentive to supply feasible donor routes
        # when Parent 1 is infeasible; mild incentive when Parent 1 is feasible
        cand_feas = population[idx].is_feasible()
        if not p1_feas:
            feas_penalty = 0.0 if cand_feas else (0.30 * float(n))
        else:
            feas_penalty = 0.0 if cand_feas else (0.12 * float(n))

        return biased_fitness[idx] + sweet_spot_penalty + feas_penalty

    parent2_idx = min(
        eligible,
        key=lambda i: (
            donor_score(i),
            costs[i],
            not population[i].is_feasible(),
            population[i].num_routes(),
        ),
    )

    return (parent1, population[parent2_idx])