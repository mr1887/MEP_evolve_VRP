"""
Best heuristic h* found by LLM-Driven Evolution
Fitness: 28,128
Generation: 3
Date: 2026-11-11T16:45:54.070965
"""

def select_parents(population, rng, cost_evaluator, k=2):
    """
    Elite-Protected Biased Fitness Tournament Selection (EP-BFTS) for PyVRP 0.14+.

    Combines the selection pressure of Parent 1 with Vidal's HGS Biased Fitness:
    1. Extracts canonical undirected edges safely for PyVRP 0.14 ScheduledActivity objects.
    2. Computes global population diversity (average Jaccard distance to reference pool).
    3. Calculates Biased Fitness:
       - Top 20% elite solutions are strictly ranked by penalized cost (no diversity penalty).
       - Remaining solutions are ranked by cost rank + diversity weight * diversity rank.
    4. Selects Parent 1 via distinct-candidate tournament of size min(5, n) on Biased Fitness.
    5. Selects Parent 2 via distinct-candidate tournament of size min(5, n - 1), strictly
       filtering out exact structural clones of Parent 1 while preserving solutions with
       high building-block synergy.
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

    # 2. Compute global diversity (average Jaccard distance to population)
    # If population is large, sample up to 25 reference individuals for efficiency
    if n <= 35:
        ref_indices = list(range(n))
    else:
        # Deterministic stride sample across population to capture spread
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
            else:
                dist_sum += 0.0
            comparisons += 1
        avg_dist[i] = dist_sum / max(1, comparisons)

    # 3. Compute Biased Fitness (Vidal et al. 2012)
    # Cost rank: 0 is lowest cost (best)
    cost_order = sorted(range(n), key=lambda i: costs[i])
    cost_ranks = [0] * n
    for rank, idx in enumerate(cost_order):
        cost_ranks[idx] = rank

    # Diversity rank: 0 is highest average distance (most unique)
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
            # Elite individuals are protected: pure cost ranking
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

    # 4. Tournament Selection for Parent 1
    t1_size = min(5, n)
    candidates1 = sample_distinct(t1_size, list(range(n)))
    parent1_idx = min(candidates1, key=lambda i: biased_fitness[i])
    parent1 = population[parent1_idx]
    p1_edges = all_edges[parent1_idx]

    # 5. Tournament Selection for Parent 2 (avoiding exact clones)
    t2_pool = [i for i in range(n) if i != parent1_idx]
    t2_size = min(5, len(t2_pool))
    candidates2 = sample_distinct(t2_size, t2_pool)

    # Filter exact clones of Parent 1 (identical edges or identical solution)
    non_clones = [
        i for i in candidates2
        if population[i] != parent1 and all_edges[i] != p1_edges
    ]

    if non_clones:
        parent2_idx = min(non_clones, key=lambda i: biased_fitness[i])
    else:
        # Fallback: scan full population for the best non-clone
        all_non_clones = [
            i for i in t2_pool
            if population[i] != parent1 and all_edges[i] != p1_edges
        ]
        if all_non_clones:
            parent2_idx = min(all_non_clones, key=lambda i: biased_fitness[i])
        else:
            # Population has completely converged to identical solutions
            parent2_idx = (parent1_idx + 1) % n

    return (parent1, population[parent2_idx])