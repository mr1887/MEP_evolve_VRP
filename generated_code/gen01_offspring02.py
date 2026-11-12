def select_parents(population, rng, cost_evaluator, k=2):
    """
    Elite-Band Complementary Tournament Selection.

    Combines strong elitist tournament pressure with bounded structural diversity:
    1. Parent 1 is selected via tournament (size min(5, n)) targeting elite quality.
    2. Parent 2 filters out clone individuals of Parent 1 to avoid wasted crossovers.
    3. If Parent 1 is infeasible, feasible tournament candidates are prioritized to
       encourage constraint repair through crossover.
    4. Within a tight 'Elite Band' (within 0.3% of the best candidate's cost), the
       individual with maximum structural edge diversity relative to Parent 1 is
       selected to maximize crossover recombination synergy.
    """
    n = len(population)
    assert n >= 2, "Population must have at least two solutions."

    # Fast path for minimal population
    if n == 2:
        c0 = cost_evaluator.penalised_cost(population[0])
        c1 = cost_evaluator.penalised_cost(population[1])
        return (population[0], population[1]) if c0 <= c1 else (population[1], population[0])

    # Helper: Sample up to `size` distinct indices from [0, n), excluding `exclude`
    def sample_distinct(size, exclude=None):
        target = min(size, n if exclude is None else n - 1)
        if target <= 0:
            return []
        chosen = []
        seen = set()
        if exclude is not None:
            seen.add(exclude)
        attempts = 0
        max_attempts = target * 4
        while len(chosen) < target and attempts < max_attempts:
            idx = rng.randint(n)
            if idx not in seen:
                seen.add(idx)
                chosen.append(idx)
            attempts += 1
        if len(chosen) < target:
            for idx in range(n):
                if idx not in seen:
                    seen.add(idx)
                    chosen.append(idx)
                    if len(chosen) >= target:
                        break
        return chosen

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
                edges.add((min(u, v), max(u, v)))
            edges.add((0, clients[-1]))
        return edges

    # Tournament selection for Parent 1
    t1_size = min(5, n)
    candidates1 = sample_distinct(t1_size)
    parent1_idx = min(
        candidates1,
        key=lambda i: (
            cost_evaluator.penalised_cost(population[i]),
            0 if population[i].is_feasible() else 1,
        ),
    )
    parent1 = population[parent1_idx]
    p1_cost = cost_evaluator.penalised_cost(parent1)
    p1_feasible = parent1.is_feasible()

    # Tournament candidates for Parent 2
    t2_size = min(5, n - 1)
    candidates2 = sample_distinct(t2_size, exclude=parent1_idx)
    if not candidates2:
        parent2_idx = (parent1_idx + 1) % n
        return (parent1, population[parent2_idx])

    # Cache penalized costs for candidates2
    costs = {idx: cost_evaluator.penalised_cost(population[idx]) for idx in candidates2}

    # Lazy extraction of Parent 1 edges
    p1_edges = None

    def get_p1_edges():
        nonlocal p1_edges
        if p1_edges is None:
            p1_edges = get_edges(parent1)
        return p1_edges

    # Filter out exact or near-identical clones of Parent 1
    non_clones = []
    denom_p1 = max(1.0, abs(p1_cost))
    for idx in candidates2:
        cand_cost = costs[idx]
        # Check if cost is virtually identical to Parent 1
        if abs(cand_cost - p1_cost) / denom_p1 < 1e-5:
            cand_edges = get_edges(population[idx])
            p1_e = get_p1_edges()
            union_e = len(p1_e | cand_edges)
            if union_e > 0 and (len(p1_e & cand_edges) / union_e) > 0.98:
                continue  # Skip clone
        non_clones.append(idx)

    eligible = non_clones if non_clones else candidates2

    # Feasibility synergy: If Parent 1 is infeasible, prioritize near-best feasible mates
    if not p1_feasible:
        feasible_candidates = [idx for idx in eligible if population[idx].is_feasible()]
        if feasible_candidates:
            min_all_cost = min(costs[idx] for idx in eligible)
            min_feas_cost = min(costs[idx] for idx in feasible_candidates)
            # Prioritize feasible if within 2% of the overall tournament minimum
            if min_feas_cost <= min_all_cost * 1.02:
                eligible = feasible_candidates

    # Find the lowest cost among eligible candidates
    best_eligible_cost = min(costs[idx] for idx in eligible)
    denom_elig = max(1.0, abs(best_eligible_cost))

    # Elite Band: candidates within 0.3% of the best eligible cost
    elite_band = [
        idx for idx in eligible if (costs[idx] - best_eligible_cost) / denom_elig <= 0.003
    ]

    if len(elite_band) == 1:
        parent2_idx = elite_band[0]
    else:
        # Multiple elite candidates: select the one with maximum edge distance to Parent 1
        p1_e = get_p1_edges()

        def diversity_score(idx):
            c_edges = get_edges(population[idx])
            union_size = len(p1_e | c_edges)
            dist = 1.0 - (len(p1_e & c_edges) / union_size) if union_size > 0 else 0.0
            # Maximize distance; break ties by penalized cost
            return (-dist, costs[idx])

        parent2_idx = min(elite_band, key=diversity_score)

    return (parent1, population[parent2_idx])