# Hướng dẫn cài đặt và chạy MEP_evolve_VRP

## 1. Phạm vi

Đây là prototype tiến hóa hàm `select_parents` bằng Gemini. **Evaluator hiện tại là GA tự cài đặt + PyVRP LocalSearch + OX, không phải HGS gốc của bài báo PyVRP+.** Xem README.md để biết đầy đủ giới hạn nghiên cứu.

## 2. Cài đặt

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
$env:GEMINI_API_KEY="YOUR_API_KEY"
python evolve.py
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="YOUR_API_KEY"
python evolve.py
```

Lấy Gemini API key tại https://aistudio.google.com/apikey. Không commit API key lên GitHub. Có thể đổi model qua biến môi trường `GEMINI_MODEL`.

## 3. Tham số mặc định

`config.py` mặc định dùng 10 thế hệ, 10 offspring mỗi thế hệ, population heuristic tối đa 5, 5 lần đánh giá mỗi instance và 10 giây cho mỗi lần chạy evaluator. Tham số thời gian có tên legacy `HGS_TIME_LIMIT`, nhưng hiện áp dụng cho GA tùy chỉnh chứ không phải HGS.

Để test nhanh:
```powershell
$env:NUM_GENERATIONS="1"
$env:OFFSPRING_PER_GENERATION="1"
$env:NUM_EVAL_RUNS="2"
$env:EVALUATION_TIME_LIMIT="2"
python evolve.py
```

## 4. Kết quả

- `generated_code/`: code LLM sinh.
- `results/checkpoint.json`: checkpoint quần thể.
- `results/best_select_parents.py`: heuristic tốt nhất tìm được theo evaluator hiện tại.
- `results/final_population.json`: quần thể cuối.

## 5. Lưu ý

- `NUM_EVAL_RUNS` là số lần đánh giá từng heuristic, không phải số lần chạy độc lập toàn bộ MEP.
- Chỉ có benchmark được liệt kê trong `config.VRP_INSTANCES` mới được đánh giá.
- `code_sandbox.py` dùng `exec()`; AST validation không phải cơ chế cách ly bảo mật. Không chạy code không đáng tin cậy trên máy chứa dữ liệu nhạy cảm.
- Không diễn giải fitness hiện tại thành kết quả HGS của bài báo. Muốn tái hiện HGS, cần thay evaluator bằng đúng implementation HGS mà bài báo sử dụng.
