def select_parents(population, rng, cost_evaluator, k=2):
    """
    Goldilocks Niche-Biased Tournament Selection (GN-BTS) for PyVRP 0.14+.

    1. Extracts canonical undirected edges safely for PyVRP 0.14 ScheduledActivity objects.
    2. Builds an exact pairwise Jaccard distance matrix.
    3. Calculates Vidal's k-nearest-neighbors Diversity Contribution (crowding metric).
    4. Computes continuous Biased Fitness balancing penalized cost rank and diversity rank.
    5. Selects Parent 1 via tournament on Biased Fitness.
    6. Selects Parent 2 via tournament applying a Goldilocks mating compatibility penalty
       to prevent both inbreeding (< 10% distance) and outbreeding disruption (> 80% distance).
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

    # 1. Safe extraction of canonical undirected edges
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

    # 2. Exact pairwise Jaccard distance matrix
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

    # 3. Vidal's k-Nearest Neighbors Diversity Contribution
    k_close = max(1, min(4, int(0.20 * n)))
    div_contribution = [0.0] * n
    for i in range(n):
        neighbors = [dist_matrix[i][j] for j in range(n) if j != i]
        neighbors.sort()
        div_contribution[i] = sum(neighbors[:k_close]) / float(k_close)

    # 4. Continuous Biased Fitness
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

    biased_fitness = [
        float(cost_ranks[i]) + div_weight * float(div_ranks[i]) for i in range(n)
    ]

    # Helper: sample `size` distinct indices without replacement
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

    # 5. Tournament Selection for Parent 1 (Backbone parent)
    t1_size = min(5, n)
    candidates1 = sample_distinct(t1_size, list(range(n)))
    parent1_idx = min(candidates1, key=lambda i: biased_fitness[i])
    parent1 = population[parent1_idx]

    # 6. Tournament Selection for Parent 2 (Donor parent with Goldilocks mating)
    t2_pool = [i for i in range(n) if i != parent1_idx]
    t2_size = min(5, len(t2_pool))
    candidates2 = sample_distinct(t2_size, t2_pool)

    def mate_score(j):
        d = dist_matrix[parent1_idx][j]
        # Inbreeding penalty: heavily penalize near-clones (< 10% distance)
        if d < 0.10:
            dist_pen = ((0.10 - d) / 0.10) * (2.0 * n)
        # Outbreeding penalty: mildly penalize excessive divergence (> 80% distance)
        elif d > 0.80:
            dist_pen = ((d - 0.80) / 0.20) * (0.5 * n)
        else:
            # Goldilocks sweet spot [0.10, 0.80]: optimal recombination compatibility
            dist_pen = 0.0
        return biased_fitness[j] + dist_pen

    parent2_idx = min(candidates2, key=mate_score)
    return (parent1, population[parent2_idx])