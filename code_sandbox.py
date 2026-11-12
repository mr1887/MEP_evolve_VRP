"""
code_sandbox.py — Kiểm tra và thực thi code được LLM sinh ra một cách an toàn.

Bước này tương ứng với phần "valid. check" trong giai đoạn ACT của sơ đồ.
- Kiểm tra cú pháp (syntax check)
- Kiểm tra chữ ký hàm (signature check)
- Biên dịch và tạo hàm thực thi được (callable)
"""

from __future__ import annotations

import ast
import traceback
from typing import Optional, Tuple

# Cài đặt các phép so sánh cho ScheduledActivity để sử dụng cho các phiển bản cũ hoặc mới mà không bị lỗi
try:
    from pyvrp._pyvrp import ScheduledActivity

    def _act_val(obj):
        return obj.idx if hasattr(obj, "idx") else obj

    ScheduledActivity.__lt__ = lambda self, other: self.idx < _act_val(other)
    ScheduledActivity.__gt__ = lambda self, other: self.idx > _act_val(other)
    ScheduledActivity.__le__ = lambda self, other: self.idx <= _act_val(other)
    ScheduledActivity.__ge__ = lambda self, other: self.idx >= _act_val(other)
    ScheduledActivity.__eq__ = lambda self, other: self.idx == _act_val(other)
    ScheduledActivity.__int__ = lambda self: self.idx
    ScheduledActivity.__index__ = lambda self: self.idx
    ScheduledActivity.__hash__ = lambda self: hash(self.idx)
except ImportError:
    pass



def validate_code(code: str) -> Tuple[bool, str]:
    """
    Kiểm tra tính hợp lệ của code Python được LLM sinh ra.

    Kiểm tra:
    1. Cú pháp Python hợp lệ (parseable)
    2. Có chứa hàm select_parents
    3. Hàm có đúng tham số bắt buộc

    Trả về:
        (is_valid, error_message): True nếu hợp lệ, kèm thông báo lỗi nếu không
    """
    # --- Bước 1: Kiểm tra cú pháp ---
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Lỗi cú pháp dòng {e.lineno}: {e.msg}"

    # --- Bước 2: Tìm hàm select_parents ---
    func_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "select_parents":
            func_found = True

            # --- Bước 3: Kiểm tra tham số ---
            args = node.args
            arg_names = [arg.arg for arg in args.args]

            required_args = ["population", "rng", "cost_evaluator"]
            for req in required_args:
                if req not in arg_names:
                    return False, f"Thiếu tham số bắt buộc: '{req}'"

            break

    if not func_found:
        return False, "Không tìm thấy hàm 'select_parents' trong code"

    # --- Bước 4: Kiểm tra import nguy hiểm ---
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Chỉ cho phép một số module an toàn
            if isinstance(node, ast.Import):
                module_names = [alias.name for alias in node.names]
            else:
                module_names = [node.module] if node.module else []

            dangerous = {"os", "sys", "subprocess", "shutil", "pathlib"}
            for mod in module_names:
                if mod and mod.split(".")[0] in dangerous:
                    return False, f"Import không được phép: '{mod}'"

    return True, "OK"


def compile_heuristic(code: str) -> Tuple[Optional[object], str]:
    """
    Biên dịch code thành hàm Python có thể gọi được.

    Quy trình:
    1. validate_code() → kiểm tra tính hợp lệ
    2. exec() → thực thi code trong namespace cách ly
    3. Trích xuất hàm select_parents từ namespace

    Trả về:
        (func, error): func là hàm callable hoặc None, error là thông báo lỗi
    """
    # Bước 1: Validate
    is_valid, error = validate_code(code)
    if not is_valid:
        return None, error

    # Bước 2: Compile & Execute trong namespace cách ly
    try:
        namespace = {}
        exec(code, namespace)
    except Exception as e:
        return None, f"Lỗi khi thực thi code: {traceback.format_exc()}"

    # Bước 3: Lấy hàm ra
    func = namespace.get("select_parents")
    if func is None:
        return None, "Không tìm thấy hàm 'select_parents' sau khi exec"

    if not callable(func):
        return None, "'select_parents' không phải hàm callable"

    return func, "OK"


def quick_smoke_test(func, data, rng_class, cost_eval_class, solution_class) -> Tuple[bool, str]:
    """
    Chạy thử nhanh (smoke test) hàm select_parents trên dữ liệu nhỏ.

    Kiểm tra:
    - Hàm không crash
    - Trả về tuple gồm 2 phần tử
    - Mỗi phần tử là Solution

    Trả về:
        (passed, error): True nếu qua, kèm thông báo lỗi nếu không
    """
    try:
        rng = rng_class(seed=999)
        cost_eval = cost_eval_class(
            load_penalties=[50.0],
            tw_penalty=50.0,
            dist_penalty=0.0,
        )

        # Tạo quần thể nhỏ 10 cá thể để test nhanh
        population = [solution_class.make_random(data, rng) for _ in range(10)]

        # Gọi hàm
        result = func(population, rng, cost_eval)

        # Kiểm tra kết quả
        if not isinstance(result, tuple) or len(result) != 2:
            return False, f"Hàm phải trả về tuple 2 phần tử, nhận được: {type(result)}"

        p1, p2 = result
        # Kiểm tra p1, p2 có phải Solution không (dùng duck typing)
        if not hasattr(p1, "is_feasible") or not hasattr(p2, "is_feasible"):
            return False, "Kết quả không phải Solution objects"

        return True, "OK"

    except Exception as e:
        return False, f"Smoke test thất bại: {traceback.format_exc()}"
