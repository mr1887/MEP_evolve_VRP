def select_parents(population, rng, cost_evaluator, k=2):
    """
    High-Pressure Tournament Selection with Lazy Micro-Diversity and Clone Pruning.

    Maintains strong elitist selection pressure while preventing wasted crossovers:
    1. Parent 1 is selected via size-5 tournament on penalized cost.
    2. Candidate costs are cached to eliminate redundant C++ evaluations.
    3. Exact clones of Parent 1 are pruned from Parent 2 candidates via a fast
       multi-stage check (cost -> route count -> edge identity).
    4. When the top two Parent 2 candidates are within a 0.5% cost band, the tie
       is broken in favor of the candidate sharing fewer canonical edges with
       Parent 1, providing SREX with complementary building blocks without
       sacrificing solution quality.
    """
    n = len(population)
    assert n >= 2, "Population must have at least two solutions."

    # Cache penalized costs to avoid redundant evaluations
    cost_cache = {}

    def get_cost(idx):
        if idx not in cost_cache:
            cost_cache[idx] = cost_evaluator.penalised_cost(population[idx])
        return cost_cache[idx]

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

    # Tournament Selection for Parent 1
    t_size = min(5, n)
    candidates1 = [rng.randint(n) for _ in range(t_size)]
    parent1_idx = min(candidates1, key=get_cost)
    p1_cost = get_cost(parent1_idx)

    # Tournament Selection for Parent 2 (sample candidates distinct in index from Parent 1)
    candidates2 = []
    max_attempts = t_size * 3
    for _ in range(max_attempts):
        idx = rng.randint(n)
        if idx != parent1_idx:
            candidates2.append(idx)
        if len(candidates2) >= t_size:
            break

    if not candidates2:
        parent2_idx = (parent1_idx + 1) % n
        return (population[parent1_idx], population[parent2_idx])

    # Lazy Parent 1 edges cache
    p1_edges = None

    def get_p1_edges():
        nonlocal p1_edges
        if p1_edges is None:
            p1_edges = get_edges(population[parent1_idx])
        return p1_edges

    # Fast multi-stage clone check against Parent 1
    def is_clone_of_p1(cand_idx):
        cand_cost = get_cost(cand_idx)
        denom = max(1.0, abs(p1_cost))
        if abs(cand_cost - p1_cost) / denom > 1e-7:
            return False
        if (
            population[cand_idx].num_routes()
            != population[parent1_idx].num_routes()
        ):
            return False
        return get_edges(population[cand_idx]) == get_p1_edges()

    # Sort unique tournament candidates by penalized cost
    unique_cands = sorted(set(candidates2), key=get_cost)

    # Filter out exact structural clones of Parent 1
    eligible = [idx for idx in unique_cands if not is_clone_of_p1(idx)]
    if not eligible:
        eligible = unique_cands

    # Select Parent 2: if top 2 are within 0.5% cost band, favor edge complementarity
    best_idx = eligible[0]
    if len(eligible) >= 2:
        second_idx = eligible[1]
        best_cost = get_cost(best_idx)
        second_cost = get_cost(second_idx)
        denom = max(1.0, abs(best_cost))

        # Check if the runner-up is within 0.5% of the best candidate
        if (second_cost - best_cost) / denom <= 0.005:
            p1_e = get_p1_edges()
            best_e = get_edges(population[best_idx])
            second_e = get_edges(population[second_idx])

            shared_best = len(p1_e & best_e)
            shared_second = len(p1_e & second_e)

            # Prefer the second candidate if it brings strictly more structural diversity
            if shared_second < shared_best:
                parent2_idx = second_idx
            else:
                parent2_idx = best_idx
        else:
            parent2_idx = best_idx
    else:
        parent2_idx = best_idx

    return (population[parent1_idx], population[parent2_idx])