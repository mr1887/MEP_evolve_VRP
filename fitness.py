"""
fitness.py — Đánh giá fitness của một heuristic trên các instance VRP.

Đây chính là hàm: f(h, D) = Σᵢ Performance(h, dᵢ)

PyVRP 0.14 dùng Iterated Local Search (ILS), KHÔNG có Genetic Algorithm (GA)
nên không có tham số select_parents trong pyvrp.solve().

Giải pháp: Ta tự xây dựng một vòng lặp GA đơn giản:
  1. Dùng PyVRP's LocalSearch để cải thiện mỗi nghiệm
  2. Dùng hàm select_parents (do LLM sinh ra) để chọn cha mẹ
  3. Dùng Order Crossover (OX) tự viết để tạo con cái
  4. Dùng cơ chế cạnh tranh sống sót giữa con cái và cha mẹ

Quy trình:
1. Với mỗi instance VRP dᵢ trong tập D:
   a. Tạo quần thể ban đầu → cải thiện bằng LocalSearch
   b. Chạy GA loop dùng heuristic h làm toán tử chọn cha mẹ
   c. Ghi nhận chi phí lời giải tốt nhất
2. Trả về chi phí trung bình (fitness) — CÀNG THẤP CÀNG TỐT
"""

from __future__ import annotations

import os
import time
import random
import statistics
from typing import Dict, List, Optional, Tuple

import pyvrp
from pyvrp import SolveParams, PenaltyManager
from pyvrp.search import LocalSearch, PerturbationManager, compute_neighbours
from pyvrp._pyvrp import CostEvaluator, RandomNumberGenerator, Solution
from pyvrp.stop import MaxRuntime

import config


def load_vrp_instance(filename: str):
    """
    Đọc một file instance VRP (.vrp) và trả về ProblemData.
    """
    filepath = os.path.join(config.DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file VRP: {filepath}")

    data = pyvrp.read(filepath, round_func="trunc")
    return data


def extract_routes_as_lists(solution: Solution) -> List[List[int]]:
    """
    Trích xuất lộ trình từ Solution thành danh sách list[int].
    Tương thích với cả PyVRP cũ và mới.
    """
    routes = []
    for route in solution.routes():
        if hasattr(route, "visits"):
            clients = list(route.visits())
        else:
            clients = []
            for act in route:
                if isinstance(act, int):
                    if act > 0:
                        clients.append(act)
                else:
                    cid = getattr(act, "client", 0)
                    if cid > 0:
                        clients.append(cid)
        if clients:
            routes.append(clients)
    return routes


def order_crossover(parent1_routes, parent2_routes, num_clients, py_rng):
    """
    Order Crossover (OX) đơn giản giữa 2 nghiệm VRP.

    Cách hoạt động:
    1. Trải phẳng cả 2 lộ trình cha mẹ thành danh sách khách hàng
    2. Chọn ngẫu nhiên một đoạn từ parent1
    3. Điền phần còn lại theo thứ tự của parent2
    4. Cắt lại thành các tuyến có kích thước tương tự parent1

    Trả về:
        Danh sách các tuyến (list[list[int]]) cho con cái
    """
    # Trải phẳng thành chuỗi khách hàng
    flat1 = []
    for r in parent1_routes:
        flat1.extend(r)

    flat2 = []
    for r in parent2_routes:
        flat2.extend(r)

    if not flat1 or not flat2:
        return parent1_routes if parent1_routes else parent2_routes

    # Đảm bảo cả 2 có cùng tập khách hàng
    all_clients = set(flat1) | set(flat2)

    # Thêm khách hàng thiếu vào cuối mỗi flat
    missing_in_1 = list(all_clients - set(flat1))
    missing_in_2 = list(all_clients - set(flat2))
    flat1.extend(missing_in_1)
    flat2.extend(missing_in_2)

    n = len(flat1)
    if n <= 2:
        return parent1_routes

    # Chọn đoạn cắt từ parent1
    cut1 = py_rng.randint(0, n - 2)
    cut2 = py_rng.randint(cut1 + 1, n)

    # Phần giữ lại từ parent1
    child = [None] * n
    segment = flat1[cut1:cut2]
    segment_set = set(segment)
    for i, val in enumerate(segment):
        child[cut1 + i] = val

    # Điền phần còn lại theo thứ tự parent2
    remaining = [c for c in flat2 if c not in segment_set]
    r_idx = 0
    for i in range(n):
        if child[i] is None:
            if r_idx < len(remaining):
                child[i] = remaining[r_idx]
                r_idx += 1

    # Loại bỏ None còn sót (nếu có)
    child = [c for c in child if c is not None]

    # Cắt lại thành tuyến theo kích thước gần giống parent1
    result_routes = []
    idx = 0
    for r in parent1_routes:
        route_len = len(r)
        route = child[idx:idx + route_len]
        if route:
            result_routes.append(route)
        idx += route_len

    # Thêm phần dư (nếu có)
    if idx < len(child):
        result_routes.append(child[idx:])

    return result_routes


def build_solution_from_routes(data, routes: List[List[int]]) -> Solution:
    """
    Tạo Solution object từ danh sách tuyến đường.
    Dùng PyVRP Model để xây dựng nghiệm hợp lệ.
    """
    try:
        # Tạo Solution trực tiếp từ routes
        # Mỗi route là list client IDs đã thăm
        sol = Solution(data, routes)
        return sol
    except Exception:
        # Nếu không tạo được, trả về nghiệm ngẫu nhiên
        rng = RandomNumberGenerator(seed=12345)
        return Solution.make_random(data, rng)


def evaluate_single_run(
    select_parents_func,
    data,
    seed: int,
    time_limit: float,
) -> float:
    """
    Chạy một vòng GA tùy chỉnh sử dụng hàm select_parents và trả về chi phí.

    Đây chính là Performance(h, dᵢ) cho MỘT lần chạy.

    Quy trình GA:
    1. Tạo quần thể ban đầu (20 cá thể ngẫu nhiên + LocalSearch)
    2. Vòng lặp GA:
       a. Gọi select_parents(population, rng, cost_eval) → (p1, p2)
       b. Tạo con cái bằng OX crossover
       c. Cải thiện con cái bằng LocalSearch
       d. Thay thế cá thể kém nhất nếu con cái tốt hơn
    3. Trả về chi phí nghiệm tốt nhất

    Tham số:
        select_parents_func: hàm select_parents đã compile
        data:                ProblemData của instance VRP
        seed:                Seed cho bộ sinh ngẫu nhiên
        time_limit:          Thời gian chạy tối đa (giây)

    Trả về:
        Chi phí lời giải tốt nhất (float)
    """
    try:
        rng = RandomNumberGenerator(seed=seed)
        py_rng = random.Random(seed)
        params = SolveParams()

        # === Setup PyVRP LocalSearch ===
        neighbours = compute_neighbours(data, params.neighbourhood)
        perturbation = PerturbationManager(params.perturbation)
        ls = LocalSearch(data, rng, neighbours, perturbation)
        for op in params.operators:
            if op.supports(data):
                ls.add_operator(op(data))

        # === Setup PenaltyManager & CostEvaluator ===
        penalties = params.penalty.midpoint_penalties(data)
        pm = PenaltyManager(penalties, params.penalty)
        cost_eval = pm.max_cost_evaluator()

        # === 1. Tạo quần thể ban đầu ===
        pop_size = 20
        population = []
        for _ in range(pop_size):
            sol = Solution.make_random(data, rng)
            improved = ls(sol, cost_eval, exhaustive=True)
            population.append(improved)

        # Sắp xếp theo chi phí
        population.sort(key=lambda s: cost_eval.penalised_cost(s))
        best_cost = cost_eval.penalised_cost(population[0])

        # === 2. Vòng lặp GA ===
        start_time = time.time()
        iteration = 0

        while time.time() - start_time < time_limit:
            iteration += 1

            # (a) Gọi hàm select_parents do LLM sinh ra
            try:
                p1, p2 = select_parents_func(population, rng, cost_eval)
            except Exception:
                # Fallback nếu select_parents crash
                p1 = population[0]
                p2 = population[min(1, len(population) - 1)]

            # (b) Crossover: tạo con cái
            routes1 = extract_routes_as_lists(p1)
            routes2 = extract_routes_as_lists(p2)

            if routes1 and routes2:
                child_routes = order_crossover(
                    routes1, routes2, data.num_clients, py_rng
                )
                child = build_solution_from_routes(data, child_routes)
            else:
                child = Solution.make_random(data, rng)

            # (c) Cải thiện con cái bằng LocalSearch
            child_improved = ls(child, cost_eval, exhaustive=False)

            # (d) Thay thế cá thể kém nhất nếu con cái tốt hơn
            child_cost = cost_eval.penalised_cost(child_improved)
            worst_cost = cost_eval.penalised_cost(population[-1])

            if child_cost < worst_cost:
                population[-1] = child_improved
                population.sort(key=lambda s: cost_eval.penalised_cost(s))

                if child_cost < best_cost:
                    best_cost = child_cost

        # === 3. Trả về chi phí tốt nhất ===
        # Ưu tiên nghiệm khả thi
        feasible_sols = [s for s in population if s.is_feasible()]
        if feasible_sols:
            return min(cost_eval.penalised_cost(s) for s in feasible_sols)
        else:
            return cost_eval.penalised_cost(population[0]) * 1.5

    except Exception as e:
        print(f"    [Fitness] Lỗi khi chạy GA: {e}")
        import traceback
        traceback.print_exc()
        return float("inf")


def evaluate_heuristic(
    select_parents_func,
    instances: Optional[List[str]] = None,
) -> Tuple[float, Dict]:
    """
    Đánh giá toàn bộ fitness: f(h, D) = trung bình chi phí trên tất cả instances.

    Tham số:
        select_parents_func: hàm select_parents đã compile
        instances:           danh sách tên file VRP (mặc định dùng config)

    Trả về:
        (fitness, details):
        - fitness: chi phí trung bình (float) — CÀNG THẤP CÀNG TỐT
        - details: dict chứa chi tiết từng instance
    """
    if instances is None:
        instances = config.VRP_INSTANCES

    all_costs = []
    details = {}

    for inst_name in instances:
        print(f"    [Fitness] Đang đánh giá trên {inst_name}...")
        data = load_vrp_instance(inst_name)

        inst_costs = []
        for run_idx in range(config.NUM_EVAL_RUNS):
            seed = 42 + run_idx * 7
            cost = evaluate_single_run(
                select_parents_func, data, seed, config.HGS_TIME_LIMIT
            )
            inst_costs.append(cost)
            print(f"      Run {run_idx + 1}: cost = {cost:,.0f}")

        avg_cost = statistics.mean(inst_costs)
        all_costs.append(avg_cost)

        details[inst_name] = {
            "costs": inst_costs,
            "mean": avg_cost,
            "std": statistics.stdev(inst_costs) if len(inst_costs) > 1 else 0,
        }
        print(f"    [Fitness]   → {inst_name}: trung bình = {avg_cost:,.0f}")

    overall_fitness = statistics.mean(all_costs) if all_costs else float("inf")
    return overall_fitness, details
