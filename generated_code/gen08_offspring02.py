def select_parents(population, rng, cost_evaluator, k=2):
    """
    Synergistic Sweet-Spot Tournament Selection (S3-TS) for PyVRP 0.14+.

    Innovations:
    1. Unbroken Vidal Biased Fitness: Applies k-NN crowding penalty across all solutions,
       properly penalizing elite clone clusters to prevent premature convergence.
    2. Sweet-Spot Complementarity Affinity: Steers Parent 2 selection toward the optimal
       Jaccard distance zone (0.30 - 0.65) where crossover recombination is most effective.
    3. Directional Feasibility Repair: Synergistically biases Parent 2 selection toward
       feasible individuals when Parent 1 is infeasible to catalyze constraint repair.
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

    # 1. Safe extraction of canonical undirected edges for PyVRP 0.14+ ScheduledActivity
    def get_solution_edges(solution):
        edges = set()
        for route in solution.routes():
            clients = [getattr(act, "client", act) for act in route]
            clients = [c for c in clients if isinstance(c, int) and c > 0]
            if not clients:
                continue
            edges.add((0, clients[0]))
            for i in range(len(clients) - 1):
                u, v = clients[i], clients[i + 1]
                edges.add((min(u, v), max(u, v)))
            edges.add((0, clients[-1]))
        return edges

    all_edges = [get_solution_edges(sol) for sol in population]
    costs = [cost_evaluator.penalised_cost(sol) for sol in population]
    feasibility = [sol.is_feasible() for sol in population]

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

    # 2. Pairwise Jaccard Distance Matrix
    # PyVRP populations are typically 25-60; full matrix is fast (< 5ms)
    use_full_matrix = n <= 60
    if use_full_matrix:
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
        def get_dist(i, j):
            e_i = all_edges[i]
            e_j = all_edges[j]
            inter = len(e_i & e_j)
            union = len(e_i) + len(e_j) - inter
            return 1.0 - (inter / float(union)) if union > 0 else 0.0

        step = max(1, n // 30)
        ref_indices = list(range(0, n, step))[:30]

    # 3. Vidal's k-Nearest-Neighbor (k-NN) local diversity contribution
    k_close = max(1, min(5, int(0.20 * n)))
    div_contribution = [0.0] * n
    for i in range(n):
        dists = [get_dist(i, j) for j in ref_indices if j != i]
        dists.sort()
        k_dists = dists[:k_close]
        div_contribution[i] = sum(k_dists) / float(max(1, len(k_dists)))

    # 4. Biased Fitness (Vidal 2012) without elite clone exemption
    cost_order = sorted(range(n), key=lambda i: costs[i])
    cost_ranks = [0] * n
    for rank, idx in enumerate(cost_order):
        cost_ranks[idx] = rank

    div_order = sorted(range(n), key=lambda i: div_contribution[i], reverse=True)
    div_ranks = [0] * n
    for rank, idx in enumerate(div_order):
        div_ranks[idx] = rank

    n_elite = max(1, int(0.20 * n))
    div_weight = 1.0 - (float(n_elite) / float(n))

    # Infeasible solutions carry a small soft penalty (+0.5 rank equivalent)
    biased_fitness = [
        float(cost_ranks[i])
        + div_weight * float(div_ranks[i])
        + (0.0 if feasibility[i] else 0.5)
        for i in range(n)
    ]

    # 5. Tournament Selection for Parent 1 (Backbone Parent)
    t1_size = min(5, n)
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

    # 6. Synergistic Tournament Selection for Parent 2 (Donor Parent)
    t2_pool = [i for i in range(n) if i != parent1_idx]
    t2_size = min(6, len(t2_pool))
    candidates2 = sample_distinct(t2_size, t2_pool)

    # Filter near-identical clones of Parent 1 (Jaccard dist < 0.08)
    def is_clone(idx):
        return population[idx] == parent1 or get_dist(parent1_idx, idx) < 0.08

    eligible = [idx for idx in candidates2 if not is_clone(idx)]
    if not eligible:
        all_eligible = [idx for idx in t2_pool if not is_clone(idx)]
        if all_eligible:
            eligible = sample_distinct(min(6, len(all_eligible)), all_eligible)
        else:
            eligible = t2_pool

    if len(eligible) == 1:
        parent2_idx = eligible[0]
    else:
        # Candidate Quality Rank (Biased Fitness)
        cand_bf_sorted = sorted(eligible, key=lambda idx: biased_fitness[idx])
        rank_bf = {idx: r for r, idx in enumerate(cand_bf_sorted)}

        # Sweet-Spot Affinity: Target Jaccard distance ~0.45 (peaks in [0.30, 0.65])
        def sweet_spot_affinity(idx):
            d = get_dist(parent1_idx, idx)
            return 1.0 - min(1.0, abs(d - 0.45) / 0.45)

        cand_comp_sorted = sorted(eligible, key=lambda idx: sweet_spot_affinity(idx), reverse=True)
        rank_comp = {idx: r for r, idx in enumerate(cand_comp_sorted)}

        def mate_score(idx):
            # Feasibility repair synergy
            if not p1_feasible and feasibility[idx]:
                feas_adj = -1.5  # Strong incentive to repair infeasible Parent 1
            elif p1_feasible and feasibility[idx]:
                feas_adj = -0.4  # Preference to keep feasible lineage
            elif not p1_feasible and not feasibility[idx]:
                feas_adj = 1.0   # Penalize compounding infeasibilities
            else:
                feas_adj = 0.0

            return rank_bf[idx] + 0.65 * rank_comp[idx] + feas_adj

        parent2_idx = min(
            eligible,
            key=lambda idx: (
                mate_score(idx),
                biased_fitness[idx],
                costs[idx],
            ),
        )

    return (parent1, population[parent2_idx])