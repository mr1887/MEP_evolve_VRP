def select_parents(population, rng, cost_evaluator, k=2):
    """
    Bi-Criterion Tournament Selection (BCTS) for PyVRP 0.14+.

    1. Parent 1 is selected via an elitist tournament (size min(5, n)) targeting
       the highest quality (minimum penalized cost).
    2. Parent 2 is selected from a tournament pool (size min(6, n - 1)) evaluated
       on both quality and structural complementarity to Parent 1.
    3. Structural distance is measured via edge Jaccard distance to Parent 1.
       Candidates with >= 95% edge overlap (near-clones) are filtered out.
    4. Candidates are evaluated using scale-invariant Biased Fitness Ranks:
       score(i) = 2 * cost_rank(i) + 1 * dist_rank(i)
       This favors high-quality candidates while rewarding structural diversity,
       ensuring crossover operators (e.g., SREX) receive complementary building blocks.
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
        max_attempts = target * 5
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

    # Helper: Extract canonical undirected edges safely across PyVRP 0.14+
    def get_edges(solution):
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

    # Precompute penalized costs for candidates in T2
    costs = {
        idx: cost_evaluator.penalised_cost(population[idx])
        for idx in candidates2
    }

    # Extract Parent 1 edges and calculate Jaccard distance for each candidate
    p1_edges = get_edges(parent1)
    jaccard_dist = {}
    for idx in candidates2:
        cand_edges = get_edges(population[idx])
        union_size = len(p1_edges | cand_edges)
        if union_size == 0:
            jaccard_dist[idx] = 0.0
        else:
            jaccard_dist[idx] = 1.0 - (
                len(p1_edges & cand_edges) / float(union_size)
            )

    # Step 3: Filter near-identical clones (overlap >= 95%, i.e., dist < 0.05)
    non_clones = [idx for idx in candidates2 if jaccard_dist[idx] >= 0.05]
    pool = non_clones if non_clones else candidates2

    # If only 1 candidate remains, select it immediately
    if len(pool) == 1:
        return (parent1, population[pool[0]])

    # Step 4: Bi-Criterion Ranking (Cost Rank & Diversity Rank)
    # Sort by cost ascending (rank 0 = lowest cost)
    by_cost = sorted(pool, key=lambda idx: costs[idx])
    cost_rank = {idx: r for r, idx in enumerate(by_cost)}

    # Sort by distance descending (rank 0 = maximum edge distance from Parent 1)
    by_dist = sorted(pool, key=lambda idx: -jaccard_dist[idx])
    dist_rank = {idx: r for r, idx in enumerate(by_dist)}

    # Composite Biased Fitness score: Quality (weight 2) + Diversity (weight 1)
    # Ties broken by lowest penalized cost
    def biased_score(idx):
        return (2 * cost_rank[idx] + dist_rank[idx], costs[idx])

    parent2_idx = min(pool, key=biased_score)
    return (parent1, population[parent2_idx])