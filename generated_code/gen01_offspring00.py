def select_parents(population, rng, cost_evaluator, k=2):
    """
    Quality-and-Diversity Biased Selection for HGS.
    Parent 1 is selected for high solution quality via tournament selection.
    Parent 2 is selected from a candidate tournament pool balancing solution quality
    and structural distance (edge divergence) relative to Parent 1 to promote
    meaningful crossover exploration.
    """
    n = len(population)
    assert n >= 2, "Population must have at least two solutions."

    # Helper function to extract undirected edges safely across PyVRP versions
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
                edges.add((min(u, v), max(u, v)))
            edges.add((0, clients[-1]))
        return edges

    # Tournament selection for Parent 1 (focus on solution quality)
    t1_size = min(4, n)
    candidates1 = [rng.randint(n) for _ in range(t1_size)]
    parent1_idx = min(candidates1, key=lambda idx: cost_evaluator.penalised_cost(population[idx]))

    # If population has only 2 individuals, Parent 2 is uniquely determined
    if n == 2:
        parent2_idx = 1 - parent1_idx
        return (population[parent1_idx], population[parent2_idx])

    # Sample a tournament pool for Parent 2 distinct from Parent 1
    t2_size = min(6, n - 1)
    candidates2 = []
    attempts = 0
    max_attempts = t2_size * 4
    while len(candidates2) < t2_size and attempts < max_attempts:
        idx = rng.randint(n)
        if idx != parent1_idx and idx not in candidates2:
            candidates2.append(idx)
        attempts += 1

    # Fallback if sampling did not yield enough unique candidates
    if len(candidates2) < t2_size:
        for idx in range(n):
            if idx != parent1_idx and idx not in candidates2:
                candidates2.append(idx)
                if len(candidates2) >= t2_size:
                    break

    # Extract edges for Parent 1
    p1_edges = get_edges(population[parent1_idx])

    # Evaluate cost and structural distance for each Parent 2 candidate
    costs = {}
    distances = {}
    for idx in candidates2:
        sol = population[idx]
        costs[idx] = cost_evaluator.penalised_cost(sol)
        c_edges = get_edges(sol)

        # Normalized edge difference (Jaccard distance)
        union_size = len(p1_edges | c_edges)
        if union_size == 0:
            distances[idx] = 0.0
        else:
            intersection_size = len(p1_edges & c_edges)
            distances[idx] = 1.0 - (intersection_size / union_size)

    # Rank candidates by cost (lower rank is better cost)
    sorted_by_cost = sorted(candidates2, key=lambda idx: costs[idx])
    cost_rank = {idx: rank for rank, idx in enumerate(sorted_by_cost)}

    # Rank candidates by distance to Parent 1 (lower rank is higher distance)
    sorted_by_dist = sorted(candidates2, key=lambda idx: distances[idx], reverse=True)
    dist_rank = {idx: rank for rank, idx in enumerate(sorted_by_dist)}

    # Composite ranking balancing quality (exploitation) and diversity (exploration)
    # Weight of 1.2 on diversity ensures we don't pick identical/near-identical solutions
    parent2_idx = min(candidates2, key=lambda idx: cost_rank[idx] + 1.2 * dist_rank[idx])

    return (population[parent1_idx], population[parent2_idx])