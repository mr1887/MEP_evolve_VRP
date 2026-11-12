def select_parents(population, rng, cost_evaluator, k=2):
    """
    Quality-Preserving Complementarity Tournament Selection (QPCTS) for PyVRP 0.14+.

    1. Fast path for small population (n == 2).
    2. Parent 1 is selected via an elitist tournament of size min(5, n) using
       pure penalized cost.
    3. Parent 2 is selected from a distinct tournament of size min(6, n - 1).
    4. Candidates with >= 90% edge similarity (near-clones) to Parent 1 are
       filtered out to prevent redundant crossovers that waste local search evaluations.
    5. Quality is strictly preserved: among viable candidates, if one is clearly
       superior (> 0.5% better than peers), it is selected directly. If multiple
       candidates are within 0.5% of the best cost, the tie is broken in favor
       of the candidate offering greater structural complementarity (lower Jaccard similarity).
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

    # Safe extraction of undirected canonical edges across PyVRP 0.14+
    def get_solution_edges(solution):
        edges = set()
        for route in solution.routes():
            clients = []
            for act in route:
                c = getattr(act, "client", act)
                if isinstance(c, int) and c > 0:
                    clients.append(c)
            if not clients:
                continue
            edges.add((0, clients[0]))
            for i in range(len(clients) - 1):
                u, v = clients[i], clients[i + 1]
                edges.add((u, v) if u < v else (v, u))
            edges.add((0, clients[-1]))
        return edges

    # Helper: Sample up to `size` distinct indices from [0, n), excluding `exclude`
    def sample_distinct(size, exclude=None):
        target = min(size, n if exclude is None else n - 1)
        if target <= 0:
            return []
        chosen = set()
        if exclude is not None:
            chosen.add(exclude)
        result = []
        attempts = 0
        max_attempts = target * 4
        while len(result) < target and attempts < max_attempts:
            idx = rng.randint(n)
            if idx not in chosen:
                chosen.add(idx)
                result.append(idx)
            attempts += 1
        if len(result) < target:
            for idx in range(n):
                if idx not in chosen:
                    chosen.add(idx)
                    result.append(idx)
                    if len(result) >= target:
                        break
        return result

    # Step 1: Select Parent 1 via quality tournament (size min(5, n))
    t1_size = min(5, n)
    candidates1 = sample_distinct(t1_size)
    parent1_idx = min(
        candidates1, key=lambda i: cost_evaluator.penalised_cost(population[i])
    )
    parent1 = population[parent1_idx]

    # Step 2: Tournament candidates for Parent 2 (size min(6, n - 1))
    t2_size = min(6, n - 1)
    candidates2 = sample_distinct(t2_size, exclude=parent1_idx)
    if not candidates2:
        parent2_idx = (parent1_idx + 1) % n
        return (parent1, population[parent2_idx])

    # Precompute costs for candidates in T2
    costs = {
        idx: cost_evaluator.penalised_cost(population[idx])
        for idx in candidates2
    }

    # Extract Parent 1 edges and calculate Jaccard similarity for candidates
    p1_edges = get_solution_edges(parent1)
    similarities = {}
    for idx in candidates2:
        cand = population[idx]
        if cand == parent1:
            similarities[idx] = 1.0
            continue
        cand_edges = get_solution_edges(cand)
        union_size = len(p1_edges | cand_edges)
        if union_size == 0:
            similarities[idx] = 1.0
        else:
            similarities[idx] = len(p1_edges & cand_edges) / float(union_size)

    # Step 3: Multi-stage filtering to eliminate unproductive near-clones
    # Non-near-clones have Jaccard similarity < 0.90
    viable = [idx for idx in candidates2 if similarities[idx] < 0.90]

    # Graceful fallback 1: if all are near-clones, filter exact clones only
    if not viable:
        viable = [idx for idx in candidates2 if similarities[idx] < 0.999]

    # Graceful fallback 2: return to all candidates if population is completely converged
    if not viable:
        viable = candidates2

    # Step 4: Quality-Preserving Complementarity Selection
    if len(viable) == 1:
        parent2_idx = viable[0]
    else:
        min_cost = min(costs[idx] for idx in viable)
        # 0.5% tolerance band around the best tournament cost
        cost_threshold = min_cost * 1.005 if min_cost >= 0 else min_cost + 1e-4
        elite_pool = [idx for idx in viable if costs[idx] <= cost_threshold]

        # In the elite band, pick the candidate with lowest similarity (most complementary)
        # Ties broken by lowest cost
        parent2_idx = min(
            elite_pool, key=lambda idx: (similarities[idx], costs[idx])
        )

    return (parent1, population[parent2_idx])