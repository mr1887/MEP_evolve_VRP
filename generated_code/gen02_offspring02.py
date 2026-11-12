def select_parents(population, rng, cost_evaluator, k=2):
    """
    High-Pressure Trans-Boundary Tournament Selection with Fast Clone Pruning.

    Key Innovations:
    1. Distinct Candidate Tournaments: Ensures full size-5 selection pressure
       by sampling candidate indices without replacement.
    2. Zero-Overhead Clone Pruning: Filters exact clones of Parent 1 using cost
       pre-filtering and PyVRP's native C++ solution equality.
    3. Trans-Boundary Crossover Prioritization:
       - If Parent 1 is infeasible, Parent 2 prioritizes feasible candidates
         to guarantee a viable route backbone and eliminate wasted Infeasible x Infeasible matings.
       - If Parent 1 is feasible, Parent 2 is guided by penalized cost, enabling
         both Feasible x Feasible refinement and Feasible x Infeasible shortcuts.
    4. Lexicographical Tie-Breaking: Breaks cost ties in favor of feasibility
       and vehicle route compactness (num_routes).
    """
    n = len(population)
    assert n >= 2, "Population must have at least two solutions."

    # Cache penalized costs to prevent redundant C++ evaluations
    cost_cache = {}

    def get_cost(idx):
        if idx not in cost_cache:
            cost_cache[idx] = cost_evaluator.penalised_cost(population[idx])
        return cost_cache[idx]

    # --- Step 1: Select Parent 1 via distinct size-5 tournament ---
    t_size1 = min(5, n)
    candidates1 = []
    while len(candidates1) < t_size1:
        idx = rng.randint(n)
        if idx not in candidates1:
            candidates1.append(idx)

    # Parent 1: Best penalized cost, breaking ties by feasibility then route compactness
    parent1_idx = min(
        candidates1,
        key=lambda i: (
            get_cost(i),
            not population[i].is_feasible(),
            population[i].num_routes(),
        ),
    )
    p1_sol = population[parent1_idx]
    p1_cost = get_cost(parent1_idx)
    p1_feas = p1_sol.is_feasible()

    # --- Step 2: Select Parent 2 candidates distinct from Parent 1 ---
    t_size2 = min(5, n - 1)
    candidates2 = []
    attempts = 0
    max_attempts = t_size2 * 4
    while len(candidates2) < t_size2 and attempts < max_attempts:
        attempts += 1
        idx = rng.randint(n)
        if idx != parent1_idx and idx not in candidates2:
            candidates2.append(idx)

    if not candidates2:
        parent2_idx = (parent1_idx + 1) % n
        return (p1_sol, population[parent2_idx])

    # --- Step 3: Fast Clone Pruning against Parent 1 ---
    def is_clone_of_p1(idx):
        cand_cost = get_cost(idx)
        # Fast cost check before any C++ solution comparison
        if abs(cand_cost - p1_cost) > 1e-9:
            return False
        return population[idx] == p1_sol

    eligible = [idx for idx in candidates2 if not is_clone_of_p1(idx)]
    if not eligible:
        eligible = candidates2

    # --- Step 4: Trans-Boundary Selection for Parent 2 ---
    if not p1_feas:
        # Parent 1 is infeasible: prioritize feasible candidates to repair the child
        # and prevent disastrous Infeasible x Infeasible matings
        parent2_idx = min(
            eligible,
            key=lambda i: (
                not population[i].is_feasible(),
                get_cost(i),
                population[i].num_routes(),
            ),
        )
    else:
        # Parent 1 is feasible: follow penalized cost, breaking ties by feasibility and compactness
        parent2_idx = min(
            eligible,
            key=lambda i: (
                get_cost(i),
                not population[i].is_feasible(),
                population[i].num_routes(),
            ),
        )

    return (p1_sol, population[parent2_idx])