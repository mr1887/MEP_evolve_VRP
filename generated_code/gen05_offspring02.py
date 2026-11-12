def select_parents(population, rng, cost_evaluator, k=2):
    """
    Synergistic Elite-Protected Biased Fitness Selection (SEP-BFS) for PyVRP 0.14+.

    1. Extracts canonical undirected edges safely for PyVRP 0.14 ScheduledActivity objects.
    2. Computes smooth global population diversity via Jaccard distance to a deterministic reference pool.
    3. Calculates Vidal's Biased Fitness with Elite Protection:
       - Top 20% elite solutions are strictly ranked by penalized cost.
       - Remaining solutions are ranked by cost rank + diversity weight * diversity rank.
    4. Parent 1 is selected via distinct tournament (size min(5, n)) on Biased Fitness.
    5. Parent 2 is selected via Synergistic Mating Selection (size min(7, n - 1)):
       - Filters out near-clones of Parent 1 (Jaccard distance < 0.12) to avoid sterile crossover.
       - Ranks valid candidates by a Mating Score combining Biased Fitness with pairwise
         structural distance to Parent 1, maximizing recombination potential.
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
            prev = 0
            for act in route:
                c = getattr(act, "client", act)
                if isinstance(c, int) and c > 0:
                    edges.add((prev, c) if prev < c else (c, prev))
                    prev = c
            if prev > 0:
                edges.add((0, prev))
        return edges

    # Precompute edge sets and penalized costs for all individuals
    all_edges = [get_solution_edges(sol) for sol in population]
    costs = [cost_evaluator.penalised_cost(sol) for sol in population]

    # 2. Compute global population diversity (average Jaccard distance to population)
    # Stride sample up to 25 reference individuals for O(n) scaling on large populations
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

    # 3. Compute Vidal's Biased Fitness with Elite Protection
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
        while len(result) < size:
            idx = pool[rng.randint(m)]
            if idx not in chosen:
                chosen.add(idx)
                result.append(idx)
        return result

    # 4. Tournament Selection for Parent 1 (Exploitation of High Quality & Global Uniqueness)
    t1_size = min(5, n)
    candidates1 = sample_distinct(t1_size, list(range(n)))
    parent1_idx = min(candidates1, key=lambda i: biased_fitness[i])
    parent1 = population[parent1_idx]
    p1_edges = all_edges[parent1_idx]
    len_p1 = len(p1_edges)

    # 5. Synergistic Complementary Tournament for Parent 2
    t2_pool = [i for i in range(n) if i != parent1_idx]
    t2_size = min(7, len(t2_pool))
    candidates2 = sample_distinct(t2_size, t2_pool)

    # Helper: Pairwise Jaccard distance to Parent 1
    def dist_to_p1(idx):
        e_cand = all_edges[idx]
        inter = len(e_cand & p1_edges)
        union = len_cand + len_p1 - inter if (len_cand := len(e_cand)) else len_p1
        return 1.0 - (inter / float(union)) if union > 0 else 0.0

    cand_dists = {idx: dist_to_p1(idx) for idx in candidates2}

    # Anti-Incest Filtering:
    # Tier 1: Truly structurally distinct candidates (distance >= 0.12, avoiding 1-2 move near-clones)
    tier1 = [i for i in candidates2 if cand_dists[i] >= 0.12]
    # Tier 2: Non-identical candidates (distance >= 0.02)
    tier2 = [i for i in candidates2 if cand_dists[i] >= 0.02]

    if tier1:
        eligible_pool = tier1
    elif tier2:
        eligible_pool = tier2
    else:
        # Fallback to scanning population if tournament sample was overly homogeneous
        pop_tier1 = [
            i for i in t2_pool
            if dist_to_p1(i) >= 0.12
        ]
        if pop_tier1:
            eligible_pool = sample_distinct(min(5, len(pop_tier1)), pop_tier1)
            for i in eligible_pool:
                if i not in cand_dists:
                    cand_dists[i] = dist_to_p1(i)
        else:
            eligible_pool = candidates2

    # Score Parent 2 candidates: Biased Fitness rewarded by structural complementarity to Parent 1
    alpha = 0.35
    def mating_score(idx):
        return biased_fitness[idx] - (alpha * float(n) * cand_dists.get(idx, 0.0))

    parent2_idx = min(eligible_pool, key=mating_score)
    return (parent1, population[parent2_idx])