def select_parents(population, rng, cost_evaluator, k=2):
    """
    Complementarity-Guided Elite Biased Selection (CEBS) for PyVRP 0.14+.

    Combines:
    1. Safe canonical undirected edge extraction for ScheduledActivity objects.
    2. Global Biased Fitness with 20% elite protection (pure cost for elites,
       cost + diversity for the rest).
    3. Selection of Parent 1 via distinct size-5 tournament on Biased Fitness,
       broken by feasibility and route compactness.
    4. Trans-Boundary Crossover & Complementarity Selection for Parent 2:
       - Prunes near-clones (Jaccard edge distance < 0.10) to avoid wasted crossovers.
       - If Parent 1 is infeasible, prioritizes feasible candidates for route repair.
       - Awards a complementarity bonus to Parent 2 candidates with distinct
         building blocks relative to Parent 1.
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

    # 1. Safe extraction of canonical undirected edges for PyVRP 0.14+
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

    # 2. Compute global diversity (average Jaccard distance to reference pool)
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
            intersection_size = len(e_i & e_j)
            union_size = len_i + len(e_j) - intersection_size
            if union_size > 0:
                dist_sum += 1.0 - (intersection_size / float(union_size))
            comparisons += 1
        avg_dist[i] = dist_sum / max(1, comparisons)

    # 3. Compute Biased Fitness with Elite Protection
    # Cost rank with feasibility and route count tie-breakers
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

    # Diversity rank: 0 is highest average distance (most diverse)
    div_order = sorted(range(n), key=lambda i: avg_dist[i], reverse=True)
    div_ranks = [0] * n
    for rank, idx in enumerate(div_order):
        div_ranks[idx] = rank

    n_elite = max(1, int(0.20 * n))
    div_weight = 1.0 - (float(n_elite) / float(n))

    biased_fitness = [0.0] * n
    for i in range(n):
        c_r = cost_ranks[i]
        if c_r < n_elite:
            biased_fitness[i] = float(c_r)
        else:
            biased_fitness[i] = float(c_r) + div_weight * float(div_ranks[i])

    # Distinct sampling helper
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

    # 4. Tournament Selection for Parent 1
    t1_size = min(5, n)
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
    p1_feas = parent1.is_feasible()
    len_p1 = len(p1_edges)

    # Helper: Jaccard distance from Parent 1
    def dist_to_p1(idx):
        e_i = all_edges[idx]
        inter = len(p1_edges & e_i)
        union = len_p1 + len(e_i) - inter
        return 1.0 - (inter / float(union)) if union > 0 else 0.0

    # 5. Tournament Selection for Parent 2
    t2_pool = [i for i in range(n) if i != parent1_idx]
    t2_size = min(6, len(t2_pool))
    candidates2 = sample_distinct(t2_size, t2_pool)

    # Prune near-clones (distance < 0.10) to guarantee recombination potential
    eligible = [
        i
        for i in candidates2
        if dist_to_p1(i) >= 0.10 and population[i] != parent1
    ]

    # Fallback to non-identical edges if all candidates are near-clones
    if not eligible:
        eligible = [
            i
            for i in candidates2
            if dist_to_p1(i) > 0.0 and population[i] != parent1
        ]
    if not eligible:
        eligible = [
            i
            for i in t2_pool
            if dist_to_p1(i) > 0.0 and population[i] != parent1
        ]
    if not eligible:
        # Population is completely converged
        return (parent1, population[(parent1_idx + 1) % n])

    # Trans-boundary repair: if Parent 1 is infeasible, prioritize feasible Parent 2
    if not p1_feas:
        feas_cands = [i for i in eligible if population[i].is_feasible()]
        if feas_cands:
            eligible = feas_cands

    # Select Parent 2: balance Biased Fitness with pairwise complementarity
    # Candidate score awards a bonus for structural diversity relative to Parent 1
    parent2_idx = min(
        eligible,
        key=lambda i: (
            biased_fitness[i] - 0.25 * div_weight * n * dist_to_p1(i),
            costs[i],
            not population[i].is_feasible(),
            population[i].num_routes(),
        ),
    )

    return (parent1, population[parent2_idx])