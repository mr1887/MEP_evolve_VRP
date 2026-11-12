"""
Heuristic gốc (h₀).

Đây là phiên bản đầu tiên, đơn giản nhất của hàm select_parents.
LLM sẽ dùng đây làm điểm xuất phát để cải thiện dần qua các thế hệ.
"""

# Đoạn code h₀ dưới dạng chuỗi string — vì LLM sẽ đọc và chỉnh sửa nó
BASE_HEURISTIC_CODE = '''
def select_parents(population, rng, cost_evaluator, k=2):
    """
    Base heuristic h0: Chọn cha mẹ bằng Tournament Selection đơn giản.
    Chọn 2 cá thể bằng cách bốc ngẫu nhiên một vài ứng viên,
    rồi chọn cái có chi phí thấp nhất.
    """
    n = len(population)
    assert n >= 2, "Population must have at least two solutions."

    # Tournament Selection cho Parent 1: bốc 5 ứng viên, chọn tốt nhất
    t_size = min(5, n)
    candidates1 = [rng.randint(n) for _ in range(t_size)]
    parent1_idx = min(candidates1, key=lambda i: cost_evaluator.penalised_cost(population[i]))

    # Tournament Selection cho Parent 2: bốc 5 ứng viên (khác parent1), chọn tốt nhất
    candidates2 = []
    for _ in range(t_size * 2):
        idx = rng.randint(n)
        if idx != parent1_idx:
            candidates2.append(idx)
        if len(candidates2) >= t_size:
            break

    if not candidates2:
        # Fallback: chọn bất kỳ cá thể nào khác parent1
        parent2_idx = (parent1_idx + 1) % n
    else:
        parent2_idx = min(candidates2, key=lambda i: cost_evaluator.penalised_cost(population[i]))

    return (population[parent1_idx], population[parent2_idx])
'''

# Mô tả ngắn gọn về heuristic này (dùng trong prompt)
BASE_HEURISTIC_DESCRIPTION = (
    "Simple tournament selection with tournament size 5. "
    "Selects two parents independently using tournament selection, "
    "choosing the one with the lowest penalised cost from random candidates."
)
