def select_parents(population, rng, cost_evaluator, k=2):
    """
    Backbone-Synergistic Elite Biased Selection (BSEB) for PyVRP 0.14+.

    Innovations:
    1. Safe canonical undirected edge extraction for PyVRP 0.14 ScheduledActivity objects.
    2. Global Biased Fitness with 20% elite protection (pure cost rank for elites,
       cost + diversity ranking for the remainder).
    3. Consensus Elite Backbone Mining: Identifies high-confidence subpaths shared
       among top-quartile solutions.
    4. Parent 1 Selection: Tournament on Biased Fitness with feasibility and route
       compactness tie-breakers.
    5. Parent 2 Synergistic Complementarity:
       - Prunes structural near-clones (Jaccard distance < 0.08) to avoid barren crossovers.
       - Enforces trans-boundary repair: prioritizes feasible mates if Parent 1 is infeasible.
       - Awards building-block synergy to candidates that carry consensus backbone edges
         missing from Parent 1.
       - Rewards diversity in the optimal SREX sweet spot (0.15 - 0.60 distance).
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

    all_edges = [get_solution_edges(sol) for sol in population]
    costs = [cost_evaluator.penalised_cost(sol) for sol in population]

    # 2. Compute global population diversity (average Jaccard distance to reference pool)
    if n <= 35:
        ref_indices = list(range(n))
    else:
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
            inter = len(e_i & e_j)
            union = len_i + len(e_j) - inter
            dist_sum += 1.0 - (inter / float(union)) if union > 0 else 0.0
            comparisons += 1
        avg_dist[i] = dist_sum / max(1, comparisons)

    # 3. Compute Biased Fitness with Elite Protection (Vidal et al. 2012)
    cost_order = sorted(
        range(n),
        key=lambda i: (
            costs[i],
            not population[i].is_feasible(),
            population[i].num_routes(),
        ),
    )
    cost_ranks = [0] * n
    for rank, idx in enumerate(cost_order):
        cost_ranks[idx] = rank

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

    # 4. Tournament Selection for Parent 1
    t1_size = min(5, n)
    candidates1 = sample_distinct(t1_size, list(range(n)))
    parent1_idx = min(
        candidates1,
        key=lambda i: (
            biased_fitness[i],
            costs[i],
            not population[i].is_feasible(),
            population[i].num_routes(),
        ),
    )
    parent1 = population[parent1_idx]
    p1_edges = all_edges[parent1_idx]
    p1_feas = parent1.is_feasible()
    len_p1 = len(p1_edges)

    # 5. Extract Consensus Elite Backbone edges (recurring edges among top solutions)
    top_m_size = max(2, min(6, n // 3))
    elite_pool = cost_order[:top_m_size]
    edge_counts = {}
    for idx in elite_pool:
        for edge in all_edges[idx]:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    consensus_thresh = max(2, (top_m_size + 1) // 2)
    consensus_edges = {
        e for e, count in edge_counts.items() if count >= consensus_thresh
    }
    missing_backbone = consensus_edges - p1_edges
    num_missing = len(missing_backbone)

    # Helper: Jaccard distance from Parent 1
    def dist_to_p1(idx):
        e_i = all_edges[idx]
        inter = len(p1_edges & e_i)
        union = len_p1 + len(e_i) - inter
        return 1.0 - (inter / float(union)) if union > 0 else 0.0

    # 6. Tournament Selection for Parent 2 with Synergistic Complementarity
    t2_pool = [i for i in range(n) if i != parent1_idx]
    t2_size = min(7, len(t2_pool))
    candidates2 = sample_distinct(t2_size, t2_pool)

    # Filter near-clones to ensure crossover recombination potential
    eligible = [
        i
        for i in candidates2
        if dist_to_p1(i) >= 0.08 and population[i] != parent1
    ]

    # Robust fallback: relax distance filter if candidate pool lacks diversity
    if not eligible:
        eligible = [
            i
            for i in candidates2
            if dist_to_p1(i) > 0.0 and population[i] != parent1
        ]
    if not eligible:
        eligible = [
            i
            for i in t2_pool
            if dist_to_p1(i) > 0.0 and population[i] != parent1
        ]
    if not eligible:
        # Full population convergence fallback
        return (parent1, population[(parent1_idx + 1) % n])

    # Trans-boundary repair: prioritize feasible mates when Parent 1 is infeasible
    if not p1_feas:
        feas_cands = [i for i in eligible if population[i].is_feasible()]
        if feas_cands:
            eligible = feas_cands

    # Score Parent 2 candidates:
    # - Base: Biased Fitness (quality + global diversity)
    # - Missing backbone bonus: reward candidates bringing consensus edges Parent 1 lacks
    # - Sweet-spot distance bonus: reward Jaccard distance centered in [0.20, 0.55]
    def candidate_score(i):
        d = dist_to_p1(i)
        # Optimal SREX crossover affinity (peaks around 0.35 distance)
        dist_affinity = max(0.0, 1.0 - 2.0 * abs(d - 0.35))

        synergy = 0.0
        if num_missing > 0:
            donated = len(all_edges[i] & missing_backbone)
            synergy = donated / float(num_missing)

        # Scale bonuses relative to population size (aligned with Biased Fitness scale)
        backbone_bonus = 0.20 * float(n) * synergy
        sweet_spot_bonus = 0.20 * div_weight * float(n) * dist_affinity

        return biased_fitness[i] - backbone_bonus - sweet_spot_bonus

    parent2_idx = min(
        eligible,
        key=lambda i: (
            candidate_score(i),
            costs[i],
            not population[i].is_feasible(),
            population[i].num_routes(),
        ),
    )

    return (parent1, population[parent2_idx])