def select_parents(population, rng, cost_evaluator, k=2):
    """
    Synergy-Driven Assortative Tournament (SDAT) Selection for PyVRP 0.14+.

    Innovations:
    1. Quadratic Elite Ramp (QER): Smoothly phases in diversity weighting for top elite
       individuals, guaranteeing preservation of the global best without an artificial rank cliff.
    2. k-Nearest-Neighbor Diversity Contribution: Sharply penalizes crowded clusters.
    3. Tri-Phase Crossover Synergy: Evaluates Parent 2 candidates based on non-linear
       crossover potential (penalizing clones < 0.10, peaking in the sweet spot 0.25-0.65,
       and dampening hyper-disruptive mates > 0.70).
    4. Adaptive Feasibility Repair Synergy: Actively pairs infeasible donor parents with
       feasible receiver mates to promote repair during SREX.
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

    # 1. Safe extraction of canonical undirected edges for PyVRP 0.14+
    def get_solution_edges(solution):
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

    all_edges = [get_solution_edges(sol) for sol in population]
    costs = [cost_evaluator.penalised_cost(sol) for sol in population]
    feasibility = [sol.is_feasible() for sol in population]

    # Helper: Sample `size` distinct indices from `pool` without replacement
    def sample_distinct(size, pool):
        m = len(pool)
        if m <= size:
            return list(pool)
        chosen = set()
        result = []
        attempts = 0
        max_attempts = size * 5
        while len(result) < size and attempts < max_attempts:
            idx = pool[rng.randint(m)]
            if idx not in chosen:
                chosen.add(idx)
                result.append(idx)
            attempts += 1
        if len(result) < size:
            for idx in pool:
                if idx not in chosen:
                    chosen.add(idx)
                    result.append(idx)
                    if len(result) >= size:
                        break
        return result

    # 2. Memoized on-demand Jaccard distance calculation
    dist_cache = {}

    def get_dist(i, j):
        if i == j:
            return 0.0
        key = (i, j) if i < j else (j, i)
        if key not in dist_cache:
            e_i = all_edges[i]
            e_j = all_edges[j]
            inter = len(e_i & e_j)
            union = len(e_i) + len(e_j) - inter
            dist_cache[key] = 1.0 - (inter / float(union)) if union > 0 else 0.0
        return dist_cache[key]

    # Reference sample for global diversity contribution
    if n <= 35:
        ref_indices = list(range(n))
    else:
        step = max(1, n // 25)
        ref_indices = list(range(0, n, step))[:25]

    # 3. Vidal's k-Nearest-Neighbor Diversity Contribution
    k_close = max(1, min(4, int(0.15 * n + 1)))
    div_contribution = [0.0] * n
    for i in range(n):
        dists = [get_dist(i, j) for j in ref_indices if j != i]
        dists.sort()
        div_contribution[i] = sum(dists[:k_close]) / float(max(1, len(dists[:k_close])))

    # 4. Quadratic Elite Ramp Biased Fitness
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
            # Smooth quadratic scaling: rank 0 has alpha = 0.0 (absolute protection),
            # while near-elite solutions smoothly factor in diversity to avoid clone dominance.
            alpha = ((float(c_r) / float(n_elite)) ** 2) * 0.5
            biased_fitness[i] = float(c_r) + alpha * div_weight * float(div_ranks[i])
        else:
            biased_fitness[i] = float(c_r) + div_weight * float(div_ranks[i])

    # 5. Tournament Selection for Parent 1 (Quality & Feasibility Anchor)
    t1_size = min(4, n)
    candidates1 = sample_distinct(t1_size, list(range(n)))
    parent1_idx = min(
        candidates1,
        key=lambda i: (
            biased_fitness[i],
            0 if feasibility[i] else 1,
            costs[i],
        ),
    )
    parent1 = population[parent1_idx]
    p1_feasible = feasibility[parent1_idx]

    # 6. Tri-Phase Synergy Evaluation for Parent 2 Candidates
    def calc_synergy(cand_idx):
        d = get_dist(parent1_idx, cand_idx)
        # Structural component: peak reward in sweet spot [0.25, 0.65]
        if population[cand_idx] == parent1 or d < 0.10:
            s_dist = -2.0  # Stagnation penalty against clones
        elif d <= 0.45:
            s_dist = (d - 0.10) / 0.35  # Linear ramp to peak
        elif d <= 0.70:
            s_dist = 1.0 - 0.60 * ((d - 0.45) / 0.25)  # Gentle decline
        else:
            s_dist = max(0.0, 0.40 - 0.40 * ((d - 0.70) / 0.30))  # Disruption decay

        # Feasibility synergy component
        if not p1_feasible and feasibility[cand_idx]:
            s_feas = 1.5  # Critical repair bonus
        elif p1_feasible and feasibility[cand_idx]:
            s_feas = 0.5  # Feasibility maintenance bonus
        else:
            s_feas = 0.0

        return s_dist, s_feas

    # 7. Parent 2 Selection via Synergy-Driven Tournament
    t2_pool = [i for i in range(n) if i != parent1_idx]
    t2_size = min(5, len(t2_pool))
    candidates2 = sample_distinct(t2_size, t2_pool)

    # If all drawn candidates are clones, fallback to scanning for any non-clone
    if all(get_dist(parent1_idx, i) < 0.10 for i in candidates2):
        non_clones = [i for i in t2_pool if get_dist(parent1_idx, i) >= 0.10]
        if non_clones:
            candidates2 = sample_distinct(min(5, len(non_clones)), non_clones)

    def parent2_combined_score(idx):
        s_dist, s_feas = calc_synergy(idx)
        # Normalized biased fitness (lower is better) offset by synergy bonuses
        norm_bf = biased_fitness[idx] / float(n)
        return norm_bf - 0.45 * s_dist - 0.20 * s_feas

    parent2_idx = min(
        candidates2,
        key=lambda idx: (
            parent2_combined_score(idx),
            biased_fitness[idx],
            costs[idx],
        ),
    )

    return (parent1, population[parent2_idx])