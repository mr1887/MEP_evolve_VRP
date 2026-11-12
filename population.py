"""
population.py — Quản lý quần thể heuristic (Population Management).

Quản lý tập P^t gồm N cá thể heuristic, bao gồm:
- Lưu trữ code + fitness + metadata cho mỗi heuristic
- Chọn 2 cha mẹ (h₁, h₂) để lai ghép
- Cập nhật quần thể: P^{t+1} ← SelectSurvivors(P^t ∪ O^t; N)
"""

import json
import os
from dataclasses import dataclass, field

import config


@dataclass
class Heuristic:
    """
    Một cá thể heuristic trong quần thể.

    Tương đương với h ∈ H trong sơ đồ framework.
    """
    code: str                              # Code Python của hàm select_parents
    fitness: float = float("inf")          # Điểm fitness (chi phí VRP trung bình)
    generation_born: int = 0               # Thế hệ được sinh ra
    description: str = ""                  # Mô tả ngắn (từ LLM reflection)
    eval_details: dict = field(default_factory=dict)  # Chi tiết đánh giá
    parent_ids: tuple[int, int] = (-1, -1) # ID của 2 cha mẹ


class Population:
    """
    Quần thể heuristic — tương ứng P^t trong sơ đồ.

    Quản lý N cá thể, hỗ trợ:
    - Thêm cá thể mới (offspring)
    - Chọn 2 cha mẹ để lai ghép
    - Chọn lọc sống sót (SelectSurvivors)
    """

    def __init__(self, max_size: int = None):
        """
        Tham số:
            max_size: N — kích thước quần thể tối đa
        """
        self.max_size = max_size or config.POPULATION_SIZE
        self.individuals: list[Heuristic] = []
        self._next_id = 0

    def add(self, heuristic: Heuristic) -> int:
        """
        Thêm một heuristic vào quần thể.

        Trả về:
            ID (chỉ số) của heuristic trong quần thể
        """
        self.individuals.append(heuristic)
        idx = self._next_id
        self._next_id += 1
        return idx

    def size(self) -> int:
        return len(self.individuals)

    def get_best(self) -> Heuristic:
        """Trả về heuristic có fitness tốt nhất (thấp nhất)."""
        return min(self.individuals, key=lambda h: h.fitness)

    def get_best_fitness(self) -> float:
        """Trả về giá trị fitness tốt nhất."""
        if not self.individuals:
            return float("inf")
        return min(h.fitness for h in self.individuals)

    def select_two_parents(self, rng_seed: int = 42) -> tuple[Heuristic, Heuristic]:
        """
        Chọn 2 heuristic cha mẹ (h₁, h₂) để lai ghép.

        Chiến lược:
        - h₁: Luôn chọn cá thể TỐT NHẤT (fitness thấp nhất) — exploitation
        - h₂: Chọn cá thể KHÁC h₁ có fitness tốt thứ 2 hoặc đa dạng — exploration

        Trả về:
            (parent1, parent2): 2 Heuristic objects
        """
        n = len(self.individuals)
        assert n >= 2, "Cần ít nhất 2 cá thể trong quần thể"

        # Sắp xếp theo fitness (thấp → cao = tốt → kém)
        sorted_pop = sorted(self.individuals, key=lambda h: h.fitness)

        # Parent 1: cá thể tốt nhất
        parent1 = sorted_pop[0]

        # Parent 2: chọn từ nửa sau để tăng đa dạng (nếu có đủ cá thể)
        if n <= 3:
            parent2 = sorted_pop[1]
        else:
            # Chọn cá thể ở vị trí giữa hoặc cuối để tạo sự đa dạng
            import random
            random.seed(rng_seed)
            # Ưu tiên nửa dưới nhưng vẫn có cơ hội chọn nửa trên
            mid = max(1, n // 3)
            parent2 = random.choice(sorted_pop[mid:])

        return parent1, parent2

    def select_survivors(self):
        """
        Chọn lọc sống sót: P^{t+1} ← SelectSurvivors(P^t ∪ O^t; N)

        Giữ lại tối đa N cá thể tốt nhất, loại bỏ phần còn lại.
        """
        if len(self.individuals) <= self.max_size:
            return  # Không cần loại bỏ

        # Sắp xếp theo fitness (thấp nhất = tốt nhất → giữ lại)
        self.individuals.sort(key=lambda h: h.fitness)

        # Giữ lại N cá thể tốt nhất
        removed = self.individuals[self.max_size:]
        self.individuals = self.individuals[:self.max_size]

        print(f"  [Population] Loại bỏ {len(removed)} cá thể kém, giữ lại {len(self.individuals)}")
        for r in removed:
            print(f"    - Loại: fitness = {r.fitness:,.0f} (gen {r.generation_born})")

    def summary(self) -> str:
        """In tóm tắt quần thể hiện tại."""
        lines = [f"\n{'='*60}", f"  QUẦN THỂ HEURISTIC ({len(self.individuals)}/{self.max_size} cá thể)", f"{'='*60}"]
        sorted_pop = sorted(self.individuals, key=lambda h: h.fitness)
        for i, h in enumerate(sorted_pop):
            marker = " ★ BEST" if i == 0 else ""
            lines.append(
                f"  #{i+1}: fitness = {h.fitness:>12,.0f} | gen = {h.generation_born} | "
                f"{h.description[:50]}{marker}"
            )
        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def save_to_file(self, filepath: str):
        """Lưu quần thể ra file JSON."""
        data = []
        for h in self.individuals:
            data.append({
                "code": h.code,
                "fitness": h.fitness,
                "generation_born": h.generation_born,
                "description": h.description,
                "parent_ids": list(h.parent_ids),
            })
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_file(self, filepath: str):
        """Tải quần thể từ file JSON."""
        if not os.path.exists(filepath):
            return
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            h = Heuristic(
                code=item["code"],
                fitness=item.get("fitness", float("inf")),
                generation_born=item.get("generation_born", 0),
                description=item.get("description", ""),
                parent_ids=tuple(item.get("parent_ids", [-1, -1])),
            )
            self.add(h)
