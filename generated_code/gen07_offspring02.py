def select_parents(population, rng, cost_evaluator, k=2):
    """
    Consensus Elite Backbone & Synergistic Selection (CEB-SS) for PyVRP 0.14+.

    1. Extracts canonical undirected edges safely across ScheduledActivity objects.
    2. Calculates Biased Fitness with Elite Protection (top 20% cost-protected).
    3. Mines Consensus Elite Backbone edges from top-performing individuals.
    4. Selects Parent 1 via distinct-candidate tournament on Biased Fitness.
    5. Selects Parent 2 by evaluating candidates within a high-quality elite band,
       scoring each on a combination of structural edge diversity and missing
       elite backbone contribution (synergy) to Parent 1.
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
    if n <= 40:
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
            # Top elite solutions are protected from diversity penalty
            biased_fitness[i] = float(c_r)
        else:
            biased_fitness[i] = float(c_r) + div_weight * float(div_ranks[i])

    # 4. Consensus Elite Backbone Mining
    # Identify recurring edges among top-quality individuals
    k_backbone = max(3, min(8, int(0.25 * n)))
    elite_pool = cost_order[:k_backbone]
    edge_counts = {}
    for idx in elite_pool:
        for e in all_edges[idx]:
            edge_counts[e] = edge_counts.get(e, 0) + 1

    min_recurrence = max(2, int(0.40 * k_backbone))
    consensus_backbone = {
        e for e, count in edge_counts.items() if count >= min_recurrence
    }

    # Helper: Distinct random index sampling
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

    # Consensus backbone edges missing in Parent 1
    missing_backbone = consensus_backbone - p1_edges
    len_missing_bb = len(missing_backbone)

    # Helper: compute Jaccard distance of candidate to Parent 1
    def dist_to_p1(idx):
        e_cand = all_edges[idx]
        inter = len(p1_edges & e_cand)
        union = len_p1_edges + len(e_cand) - inter
        return 1.0 - (inter / float(union)) if union > 0 else 0.0

    # 6. Tournament Selection for Parent 2 with Synergy Scoring
    t2_pool = [i for i in range(n) if i != parent1_idx]
    t2_size = min(7, len(t2_pool))
    candidates2 = sample_distinct(t2_size, t2_pool)

    # Filter out near-clones (< 10% Jaccard distance)
    non_clones = [i for i in candidates2 if dist_to_p1(i) >= 0.10]
    eligible = non_clones if non_clones else candidates2

    # Feasibility repair synergy: if Parent 1 is infeasible, bias toward feasible mates
    if not p1_feasible:
        feasible_candidates = [i for i in eligible if is_feas[i]]
        if feasible_candidates:
            min_cost_eligible = min(costs[i] for i in eligible)
            min_cost_feas = min(costs[i] for i in feasible_candidates)
            if min_cost_feas <= min_cost_eligible * 1.06:
                eligible = feasible_candidates

    # Form soft elite band: within 4% cost of best in eligible OR top 2 by biased fitness
    min_elig_cost = min(costs[i] for i in eligible)
    sorted_by_bf = sorted(eligible, key=lambda i: biased_fitness[i])
    top_bf_cutoff = biased_fitness[sorted_by_bf[min(1, len(sorted_by_bf) - 1)]]

    denom_cost = max(1.0, abs(min_elig_cost))
    elite_band = [
        i for i in eligible
        if (costs[i] - min_elig_cost) / denom_cost <= 0.04 or biased_fitness[i] <= top_bf_cutoff
    ]

    # Synergy function: structural distance + contribution of missing consensus backbone edges
    def synergy_score(idx):
        d = dist_to_p1(idx)
        if len_missing_bb > 0:
            bb_gain = len(all_edges[idx] & missing_backbone) / float(len_missing_bb)
        else:
            bb_gain = 0.0
        return d + 1.2 * bb_gain

    parent2_idx = max(
        elite_band,
        key=lambda i: (synergy_score(i), -biased_fitness[i], 1 if is_feas[i] else 0),
    )

    return (parent1, population[parent2_idx])