def select_parents(population, rng, cost_evaluator, k=2):
    """
    Dynamic Complementary Biased Selection (DCBS) for PyVRP 0.14+.

    Innovations:
    1. Continuous Elite Diversity Scaling: eliminates the clone-immunity bug in top solutions
       by smoothly scaling the diversity rank weight rather than zeroing it out.
    2. k-Nearest-Neighbor Crowding Metric: sharply penalizes local clusters and duplicate
       building blocks.
    3. Pairwise Complementarity Scoring for Parent 2: balances candidate quality, structural
       edge divergence from Parent 1, route count compatibility, and feasibility repair synergy.
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
    feasibility = [sol.is_feasible() for sol in population]
    num_routes = [sol.num_routes() for sol in population]

    # Helper: Sample `size` distinct indices from `pool` without replacement
    def sample_distinct(size, pool):
        m = len(pool)
        if m <= size:
            return list(pool)
        chosen = set()
        result = []
        attempts = 0
        max_attempts = size * 5
        while len(result) < size and attempts < max_attempts:
            idx = pool[rng.randint(m)]
            if idx not in chosen:
                chosen.add(idx)
                result.append(idx)
            attempts += 1
        if len(result) < size:
            for idx in pool:
                if idx not in chosen:
                    chosen.add(idx)
                    result.append(idx)
                    if len(result) >= size:
                        break
        return result

    # 2. Pairwise Jaccard Distance Matrix / Reference Pool
    if n <= 65:
        dist_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            e_i = all_edges[i]
            len_i = len(e_i)
            for j in range(i + 1, n):
                e_j = all_edges[j]
                inter = len(e_i & e_j)
                union = len_i + len(e_j) - inter
                d = 1.0 - (inter / float(union)) if union > 0 else 0.0
                dist_matrix[i][j] = d
                dist_matrix[j][i] = d

        def get_dist(i, j):
            return dist_matrix[i][j]

        ref_indices = list(range(n))
    else:
        dist_matrix = None

        def get_dist(i, j):
            e_i = all_edges[i]
            e_j = all_edges[j]
            inter = len(e_i & e_j)
            union = len(e_i) + len(e_j) - inter
            return 1.0 - (inter / float(union)) if union > 0 else 0.0

        step = max(1, n // 30)
        ref_indices = list(range(0, n, step))[:30]

    # 3. Vidal's k-Nearest-Neighbor Diversity Contribution
    k_close = max(1, min(4, int(0.15 * n)))
    div_contribution = [0.0] * n
    for i in range(n):
        dists = [get_dist(i, j) for j in ref_indices if j != i]
        dists.sort()
        div_contribution[i] = sum(dists[:k_close]) / float(max(1, len(dists[:k_close])))

    # 4. Continuous Biased Fitness with Dynamic Elite Scaling
    cost_order = sorted(range(n), key=lambda i: costs[i])
    cost_ranks = [0] * n
    for rank, idx in enumerate(cost_order):
        cost_ranks[idx] = rank

    div_order = sorted(range(n), key=lambda i: div_contribution[i], reverse=True)
    div_ranks = [0] * n
    for rank, idx in enumerate(div_order):
        div_ranks[idx] = rank

    n_elite = max(1, int(0.20 * n))
    base_div_weight = 1.0 - (float(n_elite) / float(n))

    biased_fitness = [0.0] * n
    for i in range(n):
        c_r = cost_ranks[i]
        # Continuous scaling: elite solutions gradually ramp up diversity pressure,
        # ensuring duplicates of elite solutions are strongly penalized.
        div_factor = min(1.0, (float(c_r) + 1.0) / float(n_elite))
        biased_fitness[i] = float(c_r) + base_div_weight * div_factor * float(div_ranks[i])

    # 5. Tournament Selection for Parent 1 (tournament size min(4, n))
    t1_size = min(4, n)
    candidates1 = sample_distinct(t1_size, list(range(n)))
    parent1_idx = min(
        candidates1,
        key=lambda i: (
            biased_fitness[i],
            0 if feasibility[i] else 1,
            costs[i],
        ),
    )
    parent1 = population[parent1_idx]
    p1_feasible = feasibility[parent1_idx]
    p1_routes = num_routes[parent1_idx]

    # 6. Complementary Tournament Selection for Parent 2
    t2_pool = [i for i in range(n) if i != parent1_idx]
    t2_size = min(6, len(t2_pool))
    candidates2 = sample_distinct(t2_size, t2_pool)

    # Filter near-identical clones of Parent 1 (Jaccard distance < 0.08)
    def is_clone(idx):
        return population[idx] == parent1 or get_dist(parent1_idx, idx) < 0.08

    eligible = [idx for idx in candidates2 if not is_clone(idx)]

    # Fallback if sampled tournament candidates were all near-clones
    if not eligible:
        all_eligible = [idx for idx in t2_pool if not is_clone(idx)]
        eligible = (
            sample_distinct(min(6, len(all_eligible)), all_eligible)
            if all_eligible
            else t2_pool
        )

    if len(eligible) == 1:
        parent2_idx = eligible[0]
    else:
        # Rank eligible candidates along quality (BF) and pairwise distance to Parent 1
        cand_bf_sorted = sorted(eligible, key=lambda idx: biased_fitness[idx])
        rank_bf = {idx: r for r, idx in enumerate(cand_bf_sorted)}

        cand_dist_sorted = sorted(eligible, key=lambda idx: get_dist(parent1_idx, idx), reverse=True)
        rank_dist = {idx: r for r, idx in enumerate(cand_dist_sorted)}

        def parent2_score(idx):
            # Feasibility synergy bonus
            if not p1_feasible and feasibility[idx]:
                feas_bonus = 1.0  # Heavily reward feasible mates when Parent 1 is infeasible
            elif p1_feasible and feasibility[idx]:
                feas_bonus = 0.25
            else:
                feas_bonus = 0.0

            # Route count compatibility bonus (smoother crossover exchanges)
            route_bonus = 0.5 if abs(p1_routes - num_routes[idx]) <= 1 else 0.0

            # Primary: candidate quality; Secondary: distance from Parent 1; Bonuses: repair & alignment
            return rank_bf[idx] + 0.6 * rank_dist[idx] - feas_bonus - route_bonus

        parent2_idx = min(
            eligible,
            key=lambda idx: (
                parent2_score(idx),
                biased_fitness[idx],
                costs[idx],
            ),
        )

    return (parent1, population[parent2_idx])