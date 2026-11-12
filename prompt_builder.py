"""
prompt_builder.py — Xây dựng Structured Prompt cho LLM.

Đây chính là hàm p(h₁, h₂, ε) trong sơ đồ framework:
- Nhận 2 heuristic cha mẹ (h₁, h₂) kèm điểm fitness
- Thêm ngữ cảnh bổ sung (ε): kết quả trước đó, gợi ý cải thiện
- Trả về prompt theo chuỗi REASON → ACT → REFLECT
"""


def build_evolution_prompt(
    parent1_code: str,
    parent1_fitness: float,
    parent2_code: str,
    parent2_fitness: float,
    generation: int,
    best_fitness_so_far: float,
    context_notes: str = "",
) -> str:
    """
    Xây dựng prompt có cấu trúc gửi cho LLM.

    Tham số:
        parent1_code:        Code Python của heuristic cha mẹ 1 (h₁)
        parent1_fitness:     Điểm fitness của h₁ (chi phí VRP trung bình — càng thấp càng tốt)
        parent2_code:        Code Python của heuristic cha mẹ 2 (h₂)
        parent2_fitness:     Điểm fitness của h₂
        generation:          Số thế hệ hiện tại (t)
        best_fitness_so_far: Fitness tốt nhất từng đạt được
        context_notes:       Ghi chú bổ sung (ε) — lỗi trước đó, gợi ý, v.v.

    Trả về:
        Chuỗi prompt hoàn chỉnh sẵn sàng gửi cho LLM
    """

    
    # Metacognitive Mandate (M)
    # Giải thích cho LLM biết nó đang làm gì
    metacognitive_mandate = """You are an expert algorithm designer specializing in Vehicle Routing Problems (VRP).
You are proposing variants of a `select_parents` heuristic for a VRP evolutionary-search prototype.
IMPORTANT: this repository currently evaluates candidates in a custom GA + PyVRP LocalSearch evaluator,
not in the original HGS implementation from the PyVRP+ paper. Do not claim otherwise.

YOUR TASK: Propose a testable variant of `select_parents` by studying two parent heuristics.
Prefer a small, motivated change over adding complexity without evidence.

DOMAIN-AWARE KNOWLEDGE (structured guidance; examples, not a claim these are verbatim paper prompts):
- Kp — Common pitfalls: excessive selection pressure can collapse population diversity; purely random selection can ignore solution quality; repeated selection of the same individual can reduce exploration.
- Ks — Mitigation strategies: balance quality and diversity; avoid selecting the same parent twice when population size permits; use bounded sampling/rank-aware selection; handle infeasible solutions consistently through the supplied cost evaluator.
- Kt — Problem-specific traps: VRP solutions are structured route sets, not arbitrary scalar genomes; penalised cost can rank infeasible and feasible solutions differently; PyVRP route APIs differ by version. Do not mutate solutions or assume route iteration yields plain integers.
These items guide reasoning only. The LLM's explanation is not evidence of improvement; evaluate every candidate experimentally.

IMPORTANT CONSTRAINTS:
- The function MUST have this exact signature:
  def select_parents(population, rng, cost_evaluator, k=2):
- Parameters:
  - population: list[Solution] — a list of VRP solutions
  - rng: RandomNumberGenerator — use rng.randint(n) for random int in [0, n), rng.rand() for float in [0, 1)
  - cost_evaluator: CostEvaluator — use cost_evaluator.penalised_cost(sol) to get cost
  - k: int = 2 — always 2
- The function MUST return a tuple of exactly 2 Solution objects: (parent1, parent2)
- You may use: sol.is_feasible(), sol.routes(), sol.num_routes()
- CRITICAL FOR PyVRP 0.14+ (PREVENT TYPE ERROR):
  * When iterating or extracting clients from a route, DO NOT assume elements are integers! In PyVRP 0.14, iterating a Route yields `ScheduledActivity` objects.
  * To get client integer IDs correctly, ALWAYS use this pattern:
    ```python
    # Extract client IDs safely from a route:
    def get_route_clients(route):
        return [getattr(act, "client", act) for act in route if getattr(act, "client", act) > 0]
    ```
  * Or extract edges safely:
    ```python
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
            edges.add((clients[-1], 0))
        return edges
    ```
  * NEVER compare `node > 0` directly on route elements without `getattr(act, "client", act)`!
- Do NOT import any external libraries. Only use built-in Python (set, list, dict, min, max, sum, abs, etc.)
- The function must be efficient and not cause timeouts (< 0.1 seconds per call typically)
"""

    
    # parent heuristics (h₁ và h₂)
    # Cho LLM xem 2 đoạn code hiện có
    parent_section = f"""
=== PARENT HEURISTIC 1 (Fitness: {parent1_fitness:.2f} — lower is better) ===
```python
{parent1_code.strip()}
```

=== PARENT HEURISTIC 2 (Fitness: {parent2_fitness:.2f} — lower is better) ===
```python
{parent2_code.strip()}
```
"""

    # context bổ sung (ε)
    # Thông tin bổ sung: thế hệ, fitness tốt nhất, ghi chú
    context_section = f"""
=== EVOLUTION CONTEXT ===
- Current generation: {generation}
- Best fitness achieved so far: {best_fitness_so_far:.2f}
- Parent 1 fitness: {parent1_fitness:.2f}
- Parent 2 fitness: {parent2_fitness:.2f}
"""
    if context_notes:
        context_section += f"- Additional notes: {context_notes}\n"

    # Reason -> Act -> Reflect
    reasoning_instruction = """
=== YOUR TASK: REASON → ACT → REFLECT ===

**STEP 1 — REASON (Diagnosis & Hypothesis):**
Analyze both parent heuristics:
- What are the strengths and weaknesses of each?
- Why does the better one achieve lower cost?
- What selection strategies are missing? (e.g., diversity pressure, feasibility bias,
  stratification, structural distance, adaptive mechanisms)
- What hypothesis do you have for improving the fitness?

**STEP 2 — ACT (Generate Code):**
Write a complete, new `select_parents` function that:
- Combines the best ideas from both parents
- Adds at least ONE novel improvement or strategy
- Handles edge cases (small population, all infeasible, etc.)
- Is efficient (avoid O(n²) operations on large populations)

**STEP 3 — REFLECT (Self-Evaluation):**
Briefly explain:
- What specific improvement you made and why
- What risks or limitations your new version has
- What you would try next if this doesn't improve fitness

=== OUTPUT FORMAT ===
Provide your response in this exact format:

REASONING:
[Your analysis here]

CODE:
```python
def select_parents(population, rng, cost_evaluator, k=2):
    # Your improved implementation here
    ...
    return (parent1, parent2)
```

REFLECTION:
[Your self-evaluation here]
"""

    # Ghép toàn bộ thành prompt hoàn chỉnh: p(h₁, h₂, ε)
    full_prompt = (
        metacognitive_mandate
        + parent_section
        + context_section
        + reasoning_instruction
    )
    return full_prompt
