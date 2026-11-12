def select_parents(population, rng, cost_evaluator, k=2):
    """
    Synergistic Elite Tournament Selection with Bounded Structural Diversity.

    Parent 1 is selected via tournament selection for high solution quality.
    Parent 2 is selected from a tournament pool balancing high solution quality
    with complementary structural edge diversity relative to Parent 1, with an
    explicit anti-clone penalty to prevent unproductive self-mating.
    """
    n = len(population)
    assert n >= 2, "Population must have at least two solutions."

    # Helper: Sample up to `size` unique indices from [0, n), excluding `exclude`
    def sample_distinct(size, exclude=None):
        target = min(size, n if exclude is None else n - 1)
        if target <= 0:
            return []
        chosen = set()
        attempts = 0
        max_attempts = target * 5
        while len(chosen) < target and attempts < max_attempts:
            idx = rng.randint(n)
            if idx != exclude:
                chosen.add(idx)
            attempts += 1
        if len(chosen) < target:
            for idx in range(n):
                if idx != exclude and idx not in chosen:
                    chosen.add(idx)
                    if len(chosen) >= target:
                        break
        return list(chosen)

    # Helper: Safely extract canonical undirected edges across PyVRP versions
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
    t1_size = min(5, n)
    candidates1 = sample_distinct(t1_size)
    parent1_idx = min(
        candidates1, key=lambda idx: cost_evaluator.penalised_cost(population[idx])
    )

    # Fast path for 2-individual population
    if n == 2:
        parent2_idx = 1 - parent1_idx
        return (population[parent1_idx], population[parent2_idx])

    # Sample distinct tournament candidates for Parent 2
    t2_size = min(6, n - 1)
    candidates2 = sample_distinct(t2_size, exclude=parent1_idx)
    if not candidates2:
        parent2_idx = (parent1_idx + 1) % n
        return (population[parent1_idx], population[parent2_idx])

    # Evaluate penalised costs for Parent 2 candidates
    costs = {idx: cost_evaluator.penalised_cost(population[idx]) for idx in candidates2}
    min_cost = min(costs.values())
    denom = max(1.0, abs(min_cost))

    # Extract edges of Parent 1 for diversity comparison
    p1_edges = get_edges(population[parent1_idx])

    # Score candidates balancing quality and complementary structure
    def score_candidate(idx):
        sol = population[idx]
        c_edges = get_edges(sol)

        # Normalized Jaccard edge distance to Parent 1
        union_size = len(p1_edges | c_edges)
        if union_size == 0:
            dist = 0.0
        else:
            dist = 1.0 - (len(p1_edges & c_edges) / union_size)

        # Relative cost excess over the best tournament candidate
        rel_cost = (costs[idx] - min_cost) / denom

        # Composite score: minimize cost while rewarding distance
        score = rel_cost - 0.035 * dist

        # Anti-clone penalty: discourage mating with near-identical solutions
        if dist < 0.02:
            score += 0.06

        # Slight tie-breaker preference for feasible solutions
        if sol.is_feasible():
            score -= 0.002

        return score

    parent2_idx = min(candidates2, key=score_candidate)

    return (population[parent1_idx], population[parent2_idx])