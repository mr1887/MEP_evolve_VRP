# MEP_evolve_VRP

Prototype tiến hóa heuristic `select_parents` cho Vehicle Routing Problem (VRP), sử dụng LLM (Gemini) để đề xuất mã heuristic mới theo chu trình Reason–Act–Reflect.

> **Phạm vi và giới hạn quan trọng:** Repository hiện tiến hóa riêng hàm `select_parents`. Bộ đánh giá trong `fitness.py` là **GA tùy chỉnh + PyVRP LocalSearch + Order Crossover (OX)**, không phải Hybrid Genetic Search (HGS) nguyên bản của PyVRP+ paper. Vì vậy, kết quả hiện tại chỉ đánh giá heuristic trong evaluator tùy chỉnh này; không được diễn giải là tái lập trực tiếp kết quả HGS trong bài báo.

## 1. Cấu trúc

```text
MEP_evolve_VRP/
├── config.py             # Tham số, đường dẫn và cấu hình Gemini
├── base_heuristic.py     # Heuristic select_parents ban đầu h0
├── prompt_builder.py     # Prompt Reason–Act–Reflect + tri thức miền Kp/Ks/Kt
├── llm_client.py         # Gọi Gemini, trích code và reflection
├── code_sandbox.py       # AST, compile và smoke test (không phải sandbox bảo mật)
├── fitness.py            # Evaluator GA tùy chỉnh, OX và PyVRP LocalSearch
├── population.py         # Quần thể các chương trình heuristic
├── evolve.py             # Điều phối tiến hóa
├── requirements.txt      # Dependencies
├── .env.example          # Ví dụ biến môi trường
├── data/                 # VRP benchmark instances
├── generated_code/       # Code do LLM sinh
└── results/              # Checkpoint, quần thể cuối và heuristic tốt nhất
```

## 2. Hai cấp độ "parent" cần phân biệt

1. **Heuristic parents:** hai chương trình `select_parents` được `Population.select_two_parents()` chọn để đưa cho LLM. Đây là hai cá thể trong quần thể tiến hóa MEP.
2. **Solution parents:** hai nghiệm VRP mà hàm `select_parents(population, rng, cost_evaluator)` do LLM tạo ra chọn trong population nghiệm của evaluator.

Hai thao tác có tên gần giống nhưng hoạt động ở hai tầng khác nhau.

## 3. Luồng chương trình

1. Đọc cấu hình và benchmark instance.
2. Compile, smoke-test và đánh giá heuristic gốc `h0`.
3. Khởi tạo quần thể chương trình heuristic.
4. Với mỗi thế hệ:
   - Chọn hai heuristic cha mẹ.
   - Tạo structured prompt, cung cấp code/fitness và ghi chú thế hệ trước.
   - Gemini thực hiện Reason → Act → Reflect và sinh code mới.
   - AST/compile/smoke-test code.
   - Đánh giá heuristic trong evaluator tùy chỉnh.
   - Thêm offspring hợp lệ và giữ tối đa N heuristic có fitness thấp nhất.
5. Lưu checkpoint và heuristic có fitness tốt nhất trong quần thể.

## 4. Domain-aware prompt

Prompt cung cấp ba nhóm tri thức định hướng:

- **Kp — Common pitfalls:** các hạn chế thường gặp, ví dụ áp lực chọn lọc quá mạnh làm giảm đa dạng.
- **Ks — Mitigation strategies:** hướng giảm thiểu, ví dụ cân bằng chất lượng và đa dạng.
- **Kt — Problem-specific traps:** đặc thù VRP/PyVRP, ví dụ nghiệm là tập tuyến chứ không phải vector số tùy ý, và API route có thể thay đổi theo phiên bản.

Các mục này trong repo là tri thức hướng dẫn do nhóm biên soạn để mô phỏng tinh thần Domain-Aware Initialization; không nên tuyên bố chúng là bản sao nguyên văn prompt/knowledge base của tác giả nếu chưa đối chiếu phụ lục và mã nguồn gốc.

## 5. Cấu hình mặc định

Các tham số tiến hóa được đặt gần với ngân sách được mô tả trong bài báo: 10 thế hệ, 10 offspring mỗi thế hệ, quần thể sống sót tối đa 5 heuristic. Số lần đánh giá mỗi instance mặc định là 5 để đo biến thiên; đây là số lần chạy evaluator cho từng heuristic, **không đồng nghĩa với 5 lần chạy độc lập toàn bộ quá trình tiến hóa MEP**.

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `NUM_GENERATIONS` | 10 | Số thế hệ |
| `POPULATION_SIZE` | 5 | Kích thước quần thể heuristic tối đa |
| `OFFSPRING_PER_GENERATION` | 10 | Số heuristic mới yêu cầu LLM sinh mỗi thế hệ |
| `HGS_TIME_LIMIT` | 10 giây | Tên biến legacy; thực tế là thời gian chạy evaluator GA tùy chỉnh trên mỗi instance/run |
| `NUM_EVAL_RUNS` | 5 | Số seed đánh giá mỗi heuristic trên mỗi instance |
| `VRP_INSTANCES` | X-n101-k25.vrp | Benchmark hiện được cấu hình mặc định |

Có thể override các giá trị bằng biến môi trường. Ví dụ:

```bash
NUM_GENERATIONS=2 OFFSPRING_PER_GENERATION=2 NUM_EVAL_RUNS=2 EVALUATION_TIME_LIMIT=3 python evolve.py
```

## 6. Cài đặt và chạy

Yêu cầu Python tương thích với PyVRP 0.14.x.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Đặt API key Gemini trong môi trường, không ghi key vào Git:

**Windows PowerShell**
```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
python evolve.py
```

**macOS/Linux**
```bash
export GEMINI_API_KEY="YOUR_API_KEY"
python evolve.py
```

Có thể đặt model bằng biến môi trường `GEMINI_MODEL`. File `.env.example` chỉ là mẫu; code hiện tại không tự đọc file `.env`.

## 7. Đầu ra

- `generated_code/`: code heuristic sinh theo từng thế hệ.
- `results/checkpoint.json`: quần thể sau mỗi thế hệ.
- `results/best_select_parents.py`: heuristic có fitness thấp nhất được tìm thấy.
- `results/final_population.json`: quần thể cuối.

## 8. Những điểm đã được gia cố trong prototype

- Thêm `config.py` và lấy API key từ biến môi trường.
- Đặt mặc định 10 thế hệ, 10 offspring/thế hệ, population tối đa 5.
- Crossover OX kiểm tra hai parent có cùng hoán vị khách hàng và điền gene theo thứ tự OX chuẩn.
- Không âm thầm thay offspring lỗi bằng nghiệm ngẫu nhiên.
- Không âm thầm thay heuristic lỗi bằng hai nghiệm tốt nhất.
- Không gán chi phí giả bằng hệ số phạt tùy ý khi không có nghiệm khả thi.
- Smoke test kiểm tra hai parent thuộc population và khác nhau khi có thể.
- Không thay đổi global random seed khi chọn heuristic parents.

## 9. Hạn chế còn tồn tại — không được bỏ qua khi báo cáo

1. **Evaluator không phải HGS gốc.** Đây là khác biệt lớn nhất so với bài báo. Muốn tái hiện kết quả PyVRP+, cần tích hợp hook `select_parents` vào đúng HGS implementation/fork mà bài báo sử dụng, cùng crossover, local search, survivor management và penalty adaptation tương ứng.
2. **Benchmark hiện chỉ có một instance được cấu hình.** Một instance CVRP không đủ để kết luận khả năng tổng quát hóa trên nhiều biến thể VRP.
3. **Chưa tự động chạy nhiều lần tiến hóa MEP độc lập.** `NUM_EVAL_RUNS` chỉ lặp đánh giá fitness của mỗi heuristic.
4. **Khởi tạo quần thể chỉ có h0.** Các offspring đầu có thể được tạo từ cùng một heuristic cha mẹ cho đến khi có thêm cá thể hợp lệ.
5. **Code LLM vẫn được thực thi bằng `exec()`.** AST check và namespace riêng không tạo thành sandbox an toàn. Chỉ chạy với code đáng tin cậy; nghiên cứu nghiêm túc cần process/container cô lập và giới hạn tài nguyên.
6. **Fitness theo thời gian có thể nhiễu.** Cần giữ cùng benchmark, seed, giới hạn thời gian và phiên bản dependency cho mọi heuristic; lưu log từng lần chạy.
7. **Cần kiểm thử tích hợp trên môi trường sạch.** README không khẳng định dự án đã được chạy end-to-end trên máy mới.

## 10. Để tiến gần hơn tới bài báo

- Xác định và dùng đúng mã HGS/PyVRP+ gốc mà bài báo dùng; không thay bằng GA tùy chỉnh nếu muốn so sánh trực tiếp.
- Đối chiếu chính xác heuristic gốc và heuristic tiến hóa trong phụ lục/mã nguồn bài báo.
- Dùng đúng benchmark/biến thể VRP, ngân sách đánh giá, số seed và tiêu chí fitness.
- Tách dữ liệu tiến hóa và dữ liệu kiểm thử độc lập.
- Báo cáo từng seed, mean, standard deviation và mức cải thiện so với baseline.
- Ghi lại prompt, model/version, API calls, code từng thế hệ và lỗi validation để đảm bảo khả năng tái lập.

