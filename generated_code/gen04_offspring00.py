def select_parents(population, rng, cost_evaluator, k=2):
    """
    Targeted Complementary Biased Selection (TCBS) for PyVRP 0.14+.

    1. Computes canonical undirected edge sets safely for ScheduledActivity objects.
    2. Calculates Vidal's Biased Fitness with Elite Protection (top 20% cost-protected).
    3. Selects Parent 1 via distinct-candidate tournament on Biased Fitness.
    4. Selects Parent 2 via a targeted tournament that:
       - Strictly suppresses near-clones (>88% edge overlap with Parent 1).
       - Enforces feasibility repair synergy if Parent 1 is infeasible.
       - Selects within a soft elite band (top quality candidates) the individual
         maximizing structural edge distance to Parent 1 for optimal crossover synergy.
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
    is_feas = [sol.is_feasible() for sol in population]

    # 2. Compute Global Diversity (average Jaccard distance to reference pool)
    if n <= 50:
        ref_indices = list(range(n))
    else:
        step = max(1, n // 30)
        ref_indices = list(range(0, n, step))[:30]

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
            dist_sum += 1.0 - (inter / float(union)) if union > 0 else 0.0
            comparisons += 1
        avg_dist[i] = dist_sum / max(1, comparisons)

    # 3. Biased Fitness Calculation (Vidal et al. with Elite Protection)
    cost_order = sorted(range(n), key=lambda i: (costs[i], 0 if is_feas[i] else 1))
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

    # Distinct random index sampling helper
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
        key=lambda i: (biased_fitness[i], 0 if is_feas[i] else 1),
    )
    parent1 = population[parent1_idx]
    p1_edges = all_edges[parent1_idx]
    len_p1_edges = len(p1_edges)
    p1_feasible = is_feas[parent1_idx]

    # Helper: compute Jaccard distance of candidate to Parent 1
    def dist_to_p1(idx):
        e_cand = all_edges[idx]
        inter = len(p1_edges & e_cand)
        union = len_p1_edges + len(e_cand) - inter
        return 1.0 - (inter / float(union)) if union > 0 else 0.0

    # 5. Targeted Tournament Selection for Parent 2
    t2_pool = [i for i in range(n) if i != parent1_idx]
    t2_size = min(6, len(t2_pool))
    candidates2 = sample_distinct(t2_size, t2_pool)

    # Filter out near-clones (>88% edge similarity or identical cost & high similarity)
    non_clones = [i for i in candidates2 if dist_to_p1(i) >= 0.12]
    eligible = non_clones if non_clones else candidates2

    # Feasibility synergy: If Parent 1 is infeasible, prioritize near-best feasible mates
    if not p1_feasible:
        feasible_candidates = [i for i in eligible if is_feas[i]]
        if feasible_candidates:
            min_cost_eligible = min(costs[i] for i in eligible)
            min_cost_feas = min(costs[i] for i in feasible_candidates)
            if min_cost_feas <= min_cost_eligible * 1.05:
                eligible = feasible_candidates

    # Form soft elite band: candidates within 3% of best cost OR top 2 by biased fitness
    min_elig_cost = min(costs[i] for i in eligible)
    sorted_by_bf = sorted(eligible, key=lambda i: biased_fitness[i])
    top_bf_cutoff = biased_fitness[sorted_by_bf[min(1, len(sorted_by_bf) - 1)]]

    denom_cost = max(1.0, abs(min_elig_cost))
    elite_band = [
        i for i in eligible
        if (costs[i] - min_elig_cost) / denom_cost <= 0.03 or biased_fitness[i] <= top_bf_cutoff
    ]

    # Select the candidate in the elite band with maximum structural distance to Parent 1
    parent2_idx = max(
        elite_band,
        key=lambda i: (dist_to_p1(i), -biased_fitness[i]),
    )

    return (parent1, population[parent2_idx])