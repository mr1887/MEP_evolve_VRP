def select_parents(population, rng, cost_evaluator, k=2):
    """
    Elite Building-Block & Calibrated-Distance Tournament Selection (EBD-TS).

    Innovations:
    1. PyVRP 0.14+ safe canonical undirected edge extraction handling ScheduledActivity objects.
    2. Local k-NN crowding metric to sharply penalize duplicate clusters, combined with Vidal's
       Biased Fitness (protecting top 15% elites from diversity penalties).
    3. Consensus Elite Edge Mining: Identifies high-frequency edges among top-quartile solutions
       to detect missing high-value building blocks in Parent 1.
    4. Goldilocks Mating Distance: Targets an optimal structural distance (~0.40) to prevent
       both stagnation (near-clones) and lethal crossover disruption (overly distant parents).
    5. Parent 2 Multi-Objective Synergy: Ranks candidates on Biased Fitness, missing elite edge
       donation count, target distance calibration, and feasibility repair bonus.
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

    # Helper: Jaccard distance between two edge sets
    def jaccard_dist(e1, e2):
        inter = len(e1 & e2)
        union = len(e1) + len(e2) - inter
        return 1.0 - (inter / float(union)) if union > 0 else 0.0

    # 2. Pairwise distance computation (matrix for n <= 45; sampled reference for large n)
    if n <= 45:
        dist_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            e_i = all_edges[i]
            for j in range(i + 1, n):
                d = jaccard_dist(e_i, all_edges[j])
                dist_matrix[i][j] = d
                dist_matrix[j][i] = d

        def get_dist(i, j):
            return dist_matrix[i][j]

        ref_indices = list(range(n))
    else:
        def get_dist(i, j):
            return jaccard_dist(all_edges[i], all_edges[j])

        step = max(1, n // 25)
        ref_indices = list(range(0, n, step))[:25]

    # 3. Vidal k-NN diversity contribution (k_close = 2 to 4 nearest neighbors)
    k_close = max(1, min(3, int(0.15 * n)))
    div_contrib = [0.0] * n
    for i in range(n):
        dists = [get_dist(i, j) for j in ref_indices if j != i]
        dists.sort()
        div_contrib[i] = sum(dists[:k_close]) / float(k_close)

    # 4. Vidal Biased Fitness with Elite Protection (top 15% protected)
    cost_order = sorted(range(n), key=lambda i: costs[i])
    cost_ranks = [0] * n
    for rank, idx in enumerate(cost_order):
        cost_ranks[idx] = rank

    div_order = sorted(range(n), key=lambda i: div_contrib[i], reverse=True)
    div_ranks = [0] * n
    for rank, idx in enumerate(div_order):
        div_ranks[idx] = rank

    n_elite = max(1, int(0.15 * n))
    div_weight = 1.0 - (float(n_elite) / float(n))

    biased_fitness = [0.0] * n
    for i in range(n):
        c_r = cost_ranks[i]
        if c_r < n_elite:
            biased_fitness[i] = float(c_r)
        else:
            biased_fitness[i] = float(c_r) + div_weight * float(div_ranks[i])

    # 5. Mine Consensus Elite Building Blocks (frequent edges among top quartile)
    m_elite = max(2, min(8, int(0.25 * n)))
    top_elite_indices = cost_order[:m_elite]
    edge_counts = {}
    for idx in top_elite_indices:
        for edge in all_edges[idx]:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    threshold = 2 if m_elite >= 3 else 1
    elite_edges = {e for e, count in edge_counts.items() if count >= threshold}

    # Distinct random candidate sampling helper
    def sample_distinct(size, pool):
        m = len(pool)
        if m <= size:
            return list(pool)
        chosen = set()
        result = []
        attempts = 0
        max_attempts = size * 4
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

    # 6. Tournament Selection for Parent 1 (size 4: balanced selection pressure)
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
    p1_edges = all_edges[parent1_idx]
    p1_feasible = feasibility[parent1_idx]

    # Elite edges missing in Parent 1 that Parent 2 could potentially donate
    missing_elite_edges = elite_edges - p1_edges

    # 7. Tournament Selection for Parent 2
    # Exclude near-identical solutions (distance < 0.08)
    t2_pool = [
        i for i in range(n)
        if i != parent1_idx and get_dist(parent1_idx, i) >= 0.08 and population[i] != parent1
    ]

    # Fallback if population is heavily converged
    if not t2_pool:
        t2_pool = [i for i in range(n) if i != parent1_idx]
        if not t2_pool:
            return (parent1, population[(parent1_idx + 1) % n])

    t2_size = min(5, len(t2_pool))
    candidates2 = sample_distinct(t2_size, t2_pool)

    if len(candidates2) == 1:
        parent2_idx = candidates2[0]
    else:
        # Multi-attribute scoring for Parent 2:
        # A. Quality: Biased fitness rank among candidates
        cand_bf_sorted = sorted(candidates2, key=lambda idx: biased_fitness[idx])
        rank_bf = {idx: r for r, idx in enumerate(cand_bf_sorted)}

        # B. Building-block donation: rank by count of missing elite edges offered (descending)
        donor_counts = {idx: len(all_edges[idx] & missing_elite_edges) for idx in candidates2}
        cand_donor_sorted = sorted(candidates2, key=lambda idx: donor_counts[idx], reverse=True)
        rank_donor = {idx: r for r, idx in enumerate(cand_donor_sorted)}

        # C. Distance calibration: ideal mating distance ~0.40 (Goldilocks zone)
        # Penalize solutions that are either too close (< 0.20) or discordant (> 0.65)
        ideal_dist = 0.40
        cand_dist_sorted = sorted(
            candidates2,
            key=lambda idx: abs(get_dist(parent1_idx, idx) - ideal_dist),
        )
        rank_dist = {idx: r for r, idx in enumerate(cand_dist_sorted)}

        def p2_score(idx):
            # Feasibility repair synergy
            if not p1_feasible and feasibility[idx]:
                feas_mod = -0.75  # Strongly incentivize repair
            elif p1_feasible and feasibility[idx]:
                feas_mod = -0.20  # Mild preference to preserve feasibility
            else:
                feas_mod = 0.0

            # Combined score: Quality (1.0) + Elite Donor (0.6) + Distance calibration (0.4) + Feas
            return rank_bf[idx] + 0.6 * rank_donor[idx] + 0.4 * rank_dist[idx] + feas_mod

        parent2_idx = min(
            candidates2,
            key=lambda idx: (
                p2_score(idx),
                biased_fitness[idx],
                costs[idx],
            ),
        )

    return (parent1, population[parent2_idx])