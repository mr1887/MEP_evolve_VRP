"""
llm_client.py — Gọi LLM (Gemini API) để sinh code mới.

Đây chính là bước: h' = LLM(p(h₁, h₂, ε); Θ)
- Gửi structured prompt cho Gemini
- Nhận về response chứa code Python
- Trích xuất đoạn code từ response
"""

from __future__ import annotations

import re
import time
from typing import Optional, Tuple

from google import genai
import config


def create_client():
    """Tạo client kết nối tới Gemini API."""
    return genai.Client(api_key=config.GEMINI_API_KEY)


def call_llm(prompt: str, max_retries: int = 3) -> Optional[str]:
    """
    Gửi prompt cho LLM và nhận response.

    Tham số:
        prompt:      Chuỗi prompt hoàn chỉnh (đầu ra của prompt_builder)
        max_retries: Số lần thử lại nếu gọi API thất bại

    Trả về:
        Chuỗi response text từ LLM, hoặc None nếu thất bại
    """
    client = create_client()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
            )
            if response and response.text:
                return response.text
            else:
                print(f"  [LLM] Response rỗng ở lần thử {attempt + 1}")

        except Exception as e:
            print(f"  [LLM] Lỗi API lần thử {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  [LLM] Đợi {wait}s rồi thử lại...")
                time.sleep(wait)

    return None


def extract_code_from_response(response_text: str) -> Optional[str]:
    """
    Trích xuất đoạn code Python từ response của LLM.

    LLM trả về text dạng:
        REASONING: ...
        CODE:
        ```python
        def select_parents(...):
            ...
        ```
        REFLECTION: ...

    Hàm này tìm và lấy ra đoạn code nằm giữa ```python ... ```

    Trả về:
        Chuỗi code Python, hoặc None nếu không tìm thấy
    """
    # Tìm tất cả các khối ```python ... ```
    code_blocks = re.findall(
        r"```python\s*\n(.*?)```",
        response_text,
        re.DOTALL,
    )

    if not code_blocks:
        # Thử tìm khối ``` ... ``` không có nhãn ngôn ngữ
        code_blocks = re.findall(
            r"```\s*\n(.*?)```",
            response_text,
            re.DOTALL,
        )

    if not code_blocks:
        return None

    # Tìm khối code chứa "def select_parents"
    for block in code_blocks:
        if "def select_parents" in block:
            return block.strip()

    # Nếu không khối nào chứa tên hàm, trả về khối lớn nhất
    return max(code_blocks, key=len).strip()


def extract_reflection(response_text: str) -> str:
    """
    Trích xuất phần REFLECTION từ response (dùng làm context_notes cho thế hệ sau).
    """
    match = re.search(
        r"REFLECTION:\s*\n(.*?)(?:\n```|\Z)",
        response_text,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()[:500]  # Giới hạn 500 ký tự
    return ""


def generate_heuristic(prompt: str) -> Tuple[Optional[str], str]:
    """
    Pipeline hoàn chỉnh: gọi LLM → trích xuất code → trích xuất reflection.

    Trả về:
        (code, reflection): code là chuỗi Python hoặc None, reflection là ghi chú
    """
    print("  [LLM] Đang gọi Gemini API...")
    response_text = call_llm(prompt)

    if response_text is None:
        return None, "LLM API call failed"

    code = extract_code_from_response(response_text)
    reflection = extract_reflection(response_text)

    if code is None:
        print("  [LLM] Không tìm thấy code trong response!")
        print(f"  [LLM] Response (500 ký tự đầu): {response_text[:500]}")
        return None, "Could not extract code from LLM response"

    print(f"  [LLM] Đã trích xuất code ({len(code)} ký tự)")
    return code, reflection
