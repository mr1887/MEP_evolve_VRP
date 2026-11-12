def select_parents(population, rng, cost_evaluator, k=2):
    """
    Synergistic Complementarity Biased Fitness Selection (SC-BFS) for PyVRP 0.14+.

    1. Extracts canonical undirected edges safely for PyVRP 0.14 ScheduledActivity objects.
    2. Computes global population diversity (average Jaccard distance against reference pool).
    3. Calculates Vidal's Biased Fitness:
       - Top 20% elite individuals are ranked purely by penalized cost.
       - Non-elite individuals balance penalized cost rank and global diversity rank.
    4. Selects Parent 1 via distinct-candidate tournament of size min(5, n) on Biased Fitness.
    5. Evaluates Parent 2 tournament candidates (size min(6, n - 1)):
       - Multi-tier filtering eliminates unproductive near-clones (Jaccard similarity >= 0.88).
       - Selects Parent 2 using a synergistic rank score combining Biased Fitness rank with
         direct structural complementarity (lower Jaccard similarity to Parent 1).
       - Breaks ties in favor of feasible solutions and lowest penalized cost.
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

    # 1. Safe extraction of undirected canonical edges across PyVRP 0.14+
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
                edges.add((u, v) if u < v else (v, u))
            edges.add((0, clients[-1]))
        return edges

    # Precompute edge sets and penalized costs for all individuals
    all_edges = [get_solution_edges(sol) for sol in population]
    costs = [cost_evaluator.penalised_cost(sol) for sol in population]

    # 2. Compute global diversity (average Jaccard distance to reference pool)
    if n <= 40:
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

    # 3. Compute Biased Fitness (Vidal et al. 2012)
    cost_order = sorted(range(n), key=lambda i: costs[i])
    cost_ranks = [0] * n
    for rank, idx in enumerate(cost_order):
        cost_ranks[idx] = rank

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
                    if len(result) == size:
                        break
        return result

    # 4. Tournament Selection for Parent 1
    t1_size = min(5, n)
    candidates1 = sample_distinct(t1_size, list(range(n)))
    parent1_idx = min(
        candidates1,
        key=lambda i: (
            biased_fitness[i],
            not population[i].is_feasible(),
            costs[i],
        ),
    )
    parent1 = population[parent1_idx]
    p1_edges = all_edges[parent1_idx]
    len_p1 = len(p1_edges)

    # 5. Tournament Selection for Parent 2
    t2_pool = [i for i in range(n) if i != parent1_idx]
    t2_size = min(6, len(t2_pool))
    candidates2 = sample_distinct(t2_size, t2_pool)

    # Compute pairwise Jaccard similarity directly between Parent 1 and candidates
    sims = {}
    for idx in candidates2:
        e_cand = all_edges[idx]
        inter = len(p1_edges & e_cand)
        union = len_p1 + len(e_cand) - inter
        sims[idx] = inter / float(union) if union > 0 else 1.0

    # Multi-tier filtering to remove unproductive near-clones
    # Tier 1: Filter near-clones with similarity >= 0.88
    viable = [i for i in candidates2 if sims[i] < 0.88]

    # Tier 2: If none pass, filter candidates with similarity >= 0.95
    if not viable:
        viable = [i for i in candidates2 if sims[i] < 0.95]

    # Tier 3: If still none, filter exact clones
    if not viable:
        viable = [
            i for i in candidates2
            if population[i] != parent1 and all_edges[i] != p1_edges
        ]

    # Tier 4: Fully converged tournament fallback
    if not viable:
        viable = candidates2

    # Synergistic Parent 2 Selection
    if len(viable) == 1:
        parent2_idx = viable[0]
    else:
        # Rank viable candidates by Biased Fitness (0 = best fitness)
        bf_sorted = sorted(viable, key=lambda i: (biased_fitness[i], costs[i]))
        bf_ranks = {idx: r for r, idx in enumerate(bf_sorted)}

        # Rank viable candidates by similarity to Parent 1 (0 = most complementary)
        sim_sorted = sorted(viable, key=lambda i: (sims[i], costs[i]))
        sim_ranks = {idx: r for r, idx in enumerate(sim_sorted)}

        # Weighting: 0.65 allows a 1-rank difference in BF to be overtaken by
        # significantly superior complementarity, while preserving high quality
        comp_weight = 0.65
        parent2_idx = min(
            viable,
            key=lambda i: (
                bf_ranks[i] + comp_weight * sim_ranks[i],
                not population[i].is_feasible(),
                biased_fitness[i],
                costs[i],
            ),
        )

    return (parent1, population[parent2_idx])