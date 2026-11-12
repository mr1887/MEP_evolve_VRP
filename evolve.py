"""
evolve.py — Vòng lặp tiến hóa chính (Main Evolution Loop).

Đây là file trung tâm kết nối tất cả các module lại:

  Phase I  → Khởi tạo (base_heuristic, data)
  Phase II → Vòng lặp t = 1..T:
              1. Chọn 2 cha mẹ h₁, h₂ từ quần thể
              2. Xây prompt p(h₁, h₂, ε)
              3. Gọi LLM → nhận code mới h'
              4. Validate & compile h'
              5. Đánh giá fitness f(h', D)
              6. Cập nhật quần thể: SelectSurvivors(P^t ∪ O^t; N)
  Phase III → Chọn heuristic có chi phí đánh giá thấp nhất
"""

import os
import sys
import time
import datetime

# Fix encoding cho Windows console để tránh UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo chạy được từ thư mục vrp_evolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from base_heuristic import BASE_HEURISTIC_CODE, BASE_HEURISTIC_DESCRIPTION
from prompt_builder import build_evolution_prompt
from llm_client import generate_heuristic
from code_sandbox import compile_heuristic, quick_smoke_test
from fitness import evaluate_heuristic, load_vrp_instance
from population import Population, Heuristic

from pyvrp._pyvrp import CostEvaluator, RandomNumberGenerator, Solution


def save_code_to_file(code: str, generation: int, index: int):
    """Lưu code được sinh ra vào file để kiểm tra sau."""
    os.makedirs(config.GENERATED_CODE_DIR, exist_ok=True)
    filepath = os.path.join(
        config.GENERATED_CODE_DIR, f"gen{generation:02d}_offspring{index:02d}.py"
    )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    return filepath


def print_banner():
    print("""
+==============================================================+
|   LLM-DRIVEN HEURISTIC EVOLUTION PROTOTYPE FOR VRP      |
|                                                              |
|   Phase I  : Domain-Aware Initialization                     |
|   Phase II : LLM Generation (Evolve Loop)                    |
|   Phase III: Solution Selection (Best-of-N)                  |
+==============================================================+
""")


def run_evolution():
    """
    Hàm chính chạy toàn bộ framework tiến hóa.
    """
    print_banner()
    start_time = time.time()

    # ==================================================================
    # PHASE I: DOMAIN-AWARE INITIALIZATION
    # ==================================================================
    print("=" * 60)
    print("  PHASE I: DOMAIN-AWARE INITIALIZATION")
    print("=" * 60)

    # 1. Kiểm tra API key
    if not config.GEMINI_API_KEY:
        print("\n❌ LỖI: Bạn chưa nhập API key!")
        print("   Đặt biến môi trường GEMINI_API_KEY trước khi chạy.")
        print("   Lấy API key miễn phí tại: https://aistudio.google.com/apikey")
        return

    # 2. Kiểm tra dữ liệu VRP (D = {d₁, d₂, ...})
    print("\n[Phase I] Kiểm tra dữ liệu VRP (D)...")
    for inst_name in config.VRP_INSTANCES:
        data = load_vrp_instance(inst_name)
        print(f"  ✓ {inst_name}: {data.num_clients} khách hàng, {data.num_vehicles} xe")

    # Lưu data cho smoke test (dùng instance đầu tiên)
    smoke_test_data = load_vrp_instance(config.VRP_INSTANCES[0])

    # 3. Khởi tạo heuristic gốc h₀
    print("\n[Phase I] Khởi tạo heuristic gốc (h₀)...")
    h0_func, h0_error = compile_heuristic(BASE_HEURISTIC_CODE)
    if h0_func is None:
        print(f"  ❌ Lỗi compile h₀: {h0_error}")
        return
    print("  ✓ h₀ compile thành công")

    # Smoke test h₀
    passed, err = quick_smoke_test(h0_func, smoke_test_data, RandomNumberGenerator, CostEvaluator, Solution)
    if not passed:
        print(f"  ❌ h₀ smoke test thất bại: {err}")
        return
    print("  ✓ h₀ smoke test thành công")

    # 4. Đánh giá fitness h₀
    print("\n[Phase I] Đánh giá fitness h₀ trên tập dữ liệu D...")
    h0_fitness, h0_details = evaluate_heuristic(h0_func)
    print(f"  ✓ Fitness h₀ = {h0_fitness:,.0f}")

    # 5. Khởi tạo quần thể P⁰ với h₀
    pop = Population(max_size=config.POPULATION_SIZE)
    h0 = Heuristic(
        code=BASE_HEURISTIC_CODE,
        fitness=h0_fitness,
        generation_born=0,
        description=BASE_HEURISTIC_DESCRIPTION,
        eval_details=h0_details,
    )
    pop.add(h0)
    print(f"\n[Phase I] Quần thể khởi tạo xong: {pop.size()} cá thể")
    print(f"  ★ Fitness tốt nhất: {pop.get_best_fitness():,.0f}")

    # ==================================================================
    # PHASE II: LLM GENERATION — VÒNG LẶP TIẾN HÓA (t = 1..T)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  PHASE II: LLM GENERATION — VÒNG LẶP TIẾN HÓA")
    print("=" * 60)
    print(f"  Cấu hình: {config.NUM_GENERATIONS} thế hệ × "
          f"{config.OFFSPRING_PER_GENERATION} offspring/thế hệ")
    print(f"  Quần thể tối đa: {config.POPULATION_SIZE} cá thể")
    print(f"  Thời gian evaluator GA/instance: {config.HGS_TIME_LIMIT}s × "
          f"{config.NUM_EVAL_RUNS} lần × {len(config.VRP_INSTANCES)} instances")

    context_notes = ""  # ε — thông tin ngữ cảnh bổ sung

    for gen in range(1, config.NUM_GENERATIONS + 1):
        gen_start = time.time()
        print(f"\n{'─'*60}")
        print(f"  THẾ HỆ {gen}/{config.NUM_GENERATIONS}")
        print(f"{'─'*60}")

        offspring_added = 0

        for offspring_idx in range(config.OFFSPRING_PER_GENERATION):
            print(f"\n  [Gen {gen}] Offspring {offspring_idx + 1}/{config.OFFSPRING_PER_GENERATION}")

            # ─── Bước 1: Chọn 2 cha mẹ h₁, h₂ ───
            if pop.size() < 2:
                # Nếu quần thể chỉ có 1 cá thể, dùng h₀ cho cả 2
                parent1 = pop.individuals[0]
                parent2 = pop.individuals[0]
            else:
                parent1, parent2 = pop.select_two_parents(rng_seed=gen * 100 + offspring_idx)

            print(f"    Parent 1: fitness = {parent1.fitness:,.0f}")
            print(f"    Parent 2: fitness = {parent2.fitness:,.0f}")

            # ─── Bước 2: Xây Structured Prompt p(h₁, h₂, ε) ───
            prompt = build_evolution_prompt(
                parent1_code=parent1.code,
                parent1_fitness=parent1.fitness,
                parent2_code=parent2.code,
                parent2_fitness=parent2.fitness,
                generation=gen,
                best_fitness_so_far=pop.get_best_fitness(),
                context_notes=context_notes,
            )

            # ─── Bước 3: Gọi LLM → h' = LLM(p(h₁, h₂, ε); Θ) ───
            code, reflection = generate_heuristic(prompt)

            if code is None:
                print(f"    ✗ LLM không sinh được code, bỏ qua offspring này")
                context_notes = "Previous LLM call failed to produce valid code."
                continue

            # Lưu code ra file để kiểm tra
            filepath = save_code_to_file(code, gen, offspring_idx)
            print(f"    → Code lưu tại: {filepath}")

            # ─── Bước 4: Validate & Compile h' ───
            func, error = compile_heuristic(code)
            if func is None:
                print(f"    ✗ Compile thất bại: {error}")
                context_notes = f"Previous offspring failed to compile: {error}"
                continue
            print(f"    ✓ Compile thành công")

            # Smoke test
            passed, err = quick_smoke_test(func, smoke_test_data, RandomNumberGenerator, CostEvaluator, Solution)
            if not passed:
                print(f"    ✗ Smoke test thất bại: {err}")
                context_notes = f"Previous offspring crashed during testing: {err}"
                continue
            print(f"    ✓ Smoke test thành công")

            # ─── Bước 5: Đánh giá fitness f(h', D) ───
            print(f"    Đang đánh giá fitness...")
            try:
                fitness, details = evaluate_heuristic(func)
            except Exception as e:
                print(f"    ✗ Đánh giá thất bại: {e}")
                context_notes = f"Previous offspring crashed during fitness evaluation: {e}"
                continue

            print(f"    ★ Fitness = {fitness:,.0f}")

            # ─── Bước 6: Thêm offspring vào quần thể ───
            new_h = Heuristic(
                code=code,
                fitness=fitness,
                generation_born=gen,
                description=reflection[:200] if reflection else f"Gen {gen} offspring {offspring_idx}",
                eval_details=details,
            )
            pop.add(new_h)
            offspring_added += 1

            # Cập nhật context cho offspring tiếp theo
            if fitness < pop.get_best_fitness():
                context_notes = (
                    f"Great improvement! New best fitness: {fitness:,.0f}. "
                    f"Keep exploring this direction. Reflection: {reflection[:200]}"
                )
            else:
                context_notes = (
                    f"Last offspring fitness ({fitness:,.0f}) did not beat best ({pop.get_best_fitness():,.0f}). "
                    f"Try a different strategy. Reflection: {reflection[:200]}"
                )

        # ─── SelectSurvivors: P^{t+1} ← SelectSurvivors(P^t ∪ O^t; N) ───
        if offspring_added > 0:
            print(f"\n  [Gen {gen}] SelectSurvivors: {pop.size()} cá thể → giữ {pop.max_size}")
            pop.select_survivors()

        gen_elapsed = time.time() - gen_start
        print(pop.summary())
        print(f"  ⏱ Thế hệ {gen} hoàn thành trong {gen_elapsed:.1f}s")

        # Lưu checkpoint
        checkpoint_path = os.path.join(config.RESULTS_DIR, "checkpoint.json")
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        pop.save_to_file(checkpoint_path)

    # ==================================================================
    # PHASE III: SOLUTION SELECTION — h* = argmax f(h)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  PHASE III: BEST HEURISTIC FOUND")
    print("=" * 60)

    best = pop.get_best()
    total_elapsed = time.time() - start_time

    print(f"\n  ★★★ HEURISTIC TỐT NHẤT: h* ★★★")
    print(f"  Fitness     : {best.fitness:,.0f}")
    print(f"  Sinh ra ở   : thế hệ {best.generation_born}")
    print(f"  Mô tả       : {best.description[:100]}")
    print(f"  Tổng thời gian: {total_elapsed:.0f}s ({total_elapsed/60:.1f} phút)")

    # Lưu kết quả cuối cùng
    best_code_path = os.path.join(config.RESULTS_DIR, "best_select_parents.py")
    with open(best_code_path, "w", encoding="utf-8") as f:
        f.write(f'"""\nBest heuristic h* found by LLM-Driven Evolution\n'
                f'Fitness: {best.fitness:,.0f}\n'
                f'Generation: {best.generation_born}\n'
                f'Date: {datetime.datetime.now().isoformat()}\n"""\n\n')
        f.write(best.code)
    print(f"\n  → Code h* đã lưu tại: {best_code_path}")

    pop.save_to_file(os.path.join(config.RESULTS_DIR, "final_population.json"))
    print(f"  → Quần thể cuối cùng đã lưu tại: {config.RESULTS_DIR}/final_population.json")

    print(f"\n{'='*60}")
    print(f"  ✅ HOÀN TẤT! Framework chạy xong sau {total_elapsed:.0f}s")
    print(f"{'='*60}\n")

    return best


# ==================================================================
# ENTRY POINT
# ==================================================================
if __name__ == "__main__":
    best_heuristic = run_evolution()
