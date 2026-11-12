def select_parents(population, rng, cost_evaluator, k=2):
    """
    Elite-Protected Route-Complementary Selection (EP-RCS) for PyVRP 0.14+.

    1. Extracts undirected edges and intact route sets in a single pass.
    2. Computes Vidal's k-NN diversity contribution to quantify crowding.
    3. Calculates Elite-Protected Biased Fitness: top 15% elite are ranked strictly
       by penalized cost, while non-elites balance cost and crowding rank.
    4. Selects Parent 1 (Backbone) via tournament on Biased Fitness.
    5. Selects Parent 2 (Donor) via Directed Mate Selection: filters out near-clones,
       evaluates distance compatibility (Goldilocks zone [0.20, 0.65]), and rewards
       novel intact routes available for donation.
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

    # 1. Single-pass extraction of canonical undirected edges & route client sets
    def extract_solution_data(solution):
        edges = set()
        routes = []
        for route in solution.routes():
            clients = []
            prev = 0
            for act in route:
                c = getattr(act, "client", act)
                if isinstance(c, int) and c > 0:
                    edges.add((prev, c) if prev < c else (c, prev))
                    clients.append(c)
                    prev = c
            if prev > 0:
                edges.add((0, prev))
            if clients:
                routes.append(frozenset(clients))
        return edges, routes

    all_data = [extract_solution_data(sol) for sol in population]
    all_edges = [d[0] for d in all_data]
    all_routes = [d[1] for d in all_data]
    costs = [cost_evaluator.penalised_cost(sol) for sol in population]

    # 2. Pairwise Jaccard distance matrix
    dist_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        e_i = all_edges[i]
        len_i = len(e_i)
        for j in range(i + 1, n):
            e_j = all_edges[j]
            intersection = len(e_i & e_j)
            union = len_i + len(e_j) - intersection
            d = 1.0 - (intersection / float(union)) if union > 0 else 0.0
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d

    # 3. Vidal's k-Nearest Neighbors Crowding Metric
    k_close = max(1, min(4, int(0.20 * n)))
    div_contribution = [0.0] * n
    for i in range(n):
        neighbors = [dist_matrix[i][j] for j in range(n) if j != i]
        neighbors.sort()
        div_contribution[i] = sum(neighbors[:k_close]) / float(k_close)

    # 4. Elite-Protected Biased Fitness
    cost_order = sorted(range(n), key=lambda i: costs[i])
    cost_ranks = [0] * n
    for rank, idx in enumerate(cost_order):
        cost_ranks[idx] = rank

    div_order = sorted(range(n), key=lambda i: div_contribution[i], reverse=True)
    div_ranks = [0] * n
    for rank, idx in enumerate(div_order):
        div_ranks[idx] = rank

    n_elite = max(1, int(0.15 * n))
    div_weight = 1.0 - (float(n_elite) / float(n))

    biased_fitness = [0.0] * n
    for i in range(n):
        c_r = cost_ranks[i]
        if c_r < n_elite:
            # Top elite solutions are protected from crowding penalties
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

    # 5. Tournament Selection for Parent 1 (Backbone Parent)
    t1_size = min(4, n)
    candidates1 = sample_distinct(t1_size, list(range(n)))
    parent1_idx = min(candidates1, key=lambda i: biased_fitness[i])
    parent1 = population[parent1_idx]
    p1_route_set = set(all_routes[parent1_idx])

    # 6. Directed Mate Selection for Parent 2 (Donor Parent)
    # Filter out near-clones (< 5% distance)
    candidate2_pool = [
        i for i in range(n)
        if i != parent1_idx and dist_matrix[parent1_idx][i] >= 0.05
    ]
    if not candidate2_pool:
        # Fallback if population is completely converged
        candidate2_pool = [i for i in range(n) if i != parent1_idx]

    t2_size = min(6, len(candidate2_pool))
    candidates2 = sample_distinct(t2_size, candidate2_pool)

    def mate_suitability(j):
        d = dist_matrix[parent1_idx][j]
        # Inbreeding penalty (< 20% distance)
        if d < 0.20:
            dist_pen = ((0.20 - d) / 0.20) * (1.5 * n)
        # Outbreeding disruption penalty (> 65% distance)
        elif d > 0.65:
            dist_pen = ((d - 0.65) / 0.35) * (0.75 * n)
        else:
            # Goldilocks sweet spot
            dist_pen = 0.0

        # Route novelty bonus: count donor routes not already in Parent 1
        novel_routes = sum(1 for r in all_routes[j] if r not in p1_route_set)
        novelty_bonus = -0.5 * min(novel_routes, 3)

        return biased_fitness[j] + dist_pen + novelty_bonus

    parent2_idx = min(candidates2, key=mate_suitability)
    return (parent1, population[parent2_idx])