# GIÁO TRÌNH HỌC TẬP: GIẢI THÍCH CHI TIẾT TỪNG DÒNG CODE (LINE-BY-LINE)
## FRAMEWORK TIẾN HÓA MÃ NGUỒN TỰ ĐỘNG BẰNG LLM CHO BÀI TOÁN VRP

> Tài liệu này được biên soạn dưới dạng giáo trình tự học, giải thích **từng dòng code, từng câu lệnh, từng biến số** trong toàn bộ 8 tệp mã nguồn của dự án `vrp_evolve`.

---

# MỤC LỤC
1. [config.py — Cấu hình hệ thống](#1-configpy)
2. [base_heuristic.py — Đoạn mã khởi nguyên h0](#2-base_heuristicpy)
3. [prompt_builder.py — Xây dựng câu lệnh cho AI](#3-prompt_builderpy)
4. [llm_client.py — Kết nối và xử lý phản hồi từ Gemini API](#4-llm_clientpy)
5. [code_sandbox.py — Hộp cát kiểm tra an toàn & Polyfill](#5-code_sandboxpy)
6. [fitness.py — Đánh giá chất lượng thuật toán trên bài toán VRP](#6-fitnesspy)
7. [population.py — Quản lý quần thể và chọn lọc tự nhiên](#7-populationpy)
8. [evolve.py — Nhạc trưởng điều phối 3 Phase](#8-evolvepy)

---

# 1. config.py
Tệp này lưu trữ các hằng số điều khiển toàn bộ quá trình thực thi.

```python
1: """
2: config.py — Cấu hình toàn bộ framework.
3: """
4: 
5: # 1. CẤU HÌNH LLM (Gemini API)
6: GEMINI_API_KEY = "AQ.Ab8RN6Kf04XBprC2lqlS8kR_VNPaVENBarowzz2lv9ILafWT_Q"
```
- **Dòng 6 (`GEMINI_API_KEY`)**: Chuỗi khóa bí mật (Secret Key) do Google cấp, xác thực danh tính để chương trình gửi yêu cầu đến máy chủ AI của Google.

```python
7: 
8: # Model Gemini sử dụng
9: GEMINI_MODEL = "gemini-3.8-flash"
```
- **Dòng 9 (`GEMINI_MODEL`)**: Tên định danh của mô hình trí tuệ nhân tạo sẽ sinh code. Dòng model Flash có tốc độ phản hồi cực nhanh và khả năng viết code logic tốt.

```python
10: 
11: # 2. CẤU HÌNH TIẾN HÓA
12: # Số thế hệ tiến hóa tối đa T (số lượng vòng lặp)
13: NUM_GENERATIONS = 10
```
- **Dòng 13 (`NUM_GENERATIONS`)**: Tương ứng với biến $T$ trong bài báo. Đại diện cho số lần vòng lặp thế hệ được lặp lại.

```python
14: 
15: # Kích thước quần thể heuristic N
16: POPULATION_SIZE = 6
```
- **Dòng 16 (`POPULATION_SIZE`)**: Tương ứng với biến $N$ trong bài báo. Quần thể chỉ giữ tối đa 6 thuật toán tốt nhất tại một thời điểm để tiết kiệm bộ nhớ và duy trì áp lực chọn lọc cao.

```python
17: 
18: # Số con được sinh ra mỗi thế hệ
19: OFFSPRING_PER_GENERATION = 3
```
- **Dòng 19 (`OFFSPRING_PER_GENERATION`)**: Số lượng đoạn mã mới mà LLM cần viết trong mỗi thế hệ ($|O^t| = 3$).

```python
20: 
21: # 3. CẤU HÌNH ĐÁNH GIÁ FITNESS
22: # Đường dẫn tới thư mục chứa file VRP (.vrp)
23: DATA_DIR = "data"
```
- **Dòng 23 (`DATA_DIR`)**: Tên thư mục cục bộ chứa các file dữ liệu bài toán VRP đầu vào.

```python
24: 
25: # Danh sách file instance VRP để đánh giá (D = {d1, d2, ...})
26: VRP_INSTANCES = [
27:     "X-n101-k25.vrp",
28: ]
```
- **Dòng 26-28 (`VRP_INSTANCES`)**: Tương ứng với tập dữ liệu $D = \{d_1, d_2, \dots\}$. Ở đây ta dùng file `X-n101-k25.vrp` (100 khách hàng, 25 xe) từ bộ dữ liệu chuẩn CVRPLIB.

```python
29: 
30: # Thời gian chạy HGS tối đa cho mỗi instance (giây)
31: HGS_TIME_LIMIT = 30
```
- **Dòng 31 (`HGS_TIME_LIMIT`)**: Thời gian cấp phép cho mỗi thuật toán chạy giải bài toán VRP. 30 giây là thời gian đủ dài để thuật toán hội tụ và bộc lộ rõ chất lượng.

```python
32: 
33: # Số lần chạy lặp lại cho mỗi instance (để giảm nhiễu do tính ngẫu nhiên)
34: NUM_EVAL_RUNS = 2
```
- **Dòng 34 (`NUM_EVAL_RUNS`)**: Số lần chạy lại trên cùng 1 bài toán (với các hạt giống ngẫu nhiên seed khác nhau) rồi lấy trung bình cộng, giúp kết quả đánh giá khách quan, loại bỏ may rủi.

```python
35: 
36: # 4. CẤU HÌNH LƯU TRỮ
37: # Thư mục lưu code được sinh ra
38: GENERATED_CODE_DIR = "generated_code"
39: 
40: # Thư mục lưu kết quả
41: RESULTS_DIR = "results"
```
- **Dòng 38-41**: Tên 2 thư mục để tự động lưu lại các file `.py` do AI viết ra và file kết quả cuối cùng.

---

# 2. base_heuristic.py
Tệp này chứa thuật toán gốc $h_0$ được viết dưới dạng chuỗi văn bản (string) để sẵn sàng đưa vào prompt của LLM.

```python
1: """
2: base_heuristic.py — Heuristic gốc (h₀).
3: """
4: 
5: BASE_HEURISTIC_CODE = '''
6: def select_parents(population, rng, cost_evaluator, k=2):
```
- **Dòng 6**: Khai báo hàm `select_parents`. 
  - `population`: danh sách các nghiệm VRP hiện có trong thuật toán HGS.
  - `rng`: bộ tạo số ngẫu nhiên của PyVRP.
  - `cost_evaluator`: đối tượng tính chi phí đường đi và mức phạt vi phạm.
  - `k=2`: số lượng cha mẹ cần chọn (luôn bằng 2).

```python
7:     """
8:     Base heuristic h0: Chọn cha mẹ bằng Tournament Selection đơn giản.
9:     """
10:     n = len(population)
11:     assert n >= 2, "Population must have at least two solutions."
```
- **Dòng 10-11**: Lấy kích thước quần thể `n` và kiểm tra an toàn xem có đủ ít nhất 2 cá thể để chọn hay không.

```python
12: 
13:     # Tournament Selection cho Parent 1: bốc 5 ứng viên, chọn tốt nhất
14:     t_size = min(5, n)
15:     candidates1 = [rng.randint(n) for _ in range(t_size)]
16:     parent1_idx = min(candidates1, key=lambda i: cost_evaluator.penalised_cost(population[i]))
```
- **Dòng 14**: Thiết lập kích thước giải đấu `t_size = 5` (nếu quần thể nhỏ hơn 5 thì lấy `n`).
- **Dòng 15**: Dùng `rng.randint(n)` bốc ngẫu nhiên 5 chỉ số cá thể trong quần thể đưa vào danh sách `candidates1`.
- **Dòng 16**: Dùng hàm `min` kèm `key` để tìm ra chỉ số `i` có chi phí `penalised_cost` thấp nhất trong 5 ứng viên làm `parent1_idx`.

```python
17: 
18:     # Tournament Selection cho Parent 2: bốc 5 ứng viên (khác parent1), chọn tốt nhất
19:     candidates2 = []
20:     for _ in range(t_size * 2):
21:         idx = rng.randint(n)
22:         if idx != parent1_idx:
23:             candidates2.append(idx)
24:         if len(candidates2) >= t_size:
25:             break
```
- **Dòng 19-25**: Chọn `parent2` cũng bằng giải đấu, nhưng lặp tối đa `t_size * 2` lần để tìm các ứng viên khác với `parent1_idx`, tránh việc chọn trùng lặp một cá thể làm cả bố lẫn mẹ.

```python
26: 
27:     if not candidates2:
28:         parent2_idx = (parent1_idx + 1) % n
29:     else:
30:         parent2_idx = min(candidates2, key=lambda i: cost_evaluator.penalised_cost(population[i]))
31: 
32:     return (population[parent1_idx], population[parent2_idx])
33: '''
```
- **Dòng 27-28**: Nếu bốc thăm ngẫu nhiên không ra ứng viên khác, fallback lấy ngay cá thể kế tiếp `(parent1_idx + 1) % n`.
- **Dòng 30**: Chọn ra ứng viên tốt nhất trong nhóm 2 làm `parent2_idx`.
- **Dòng 32**: Trả về một tuple chứa 2 đối tượng nghiệm `(Solution, Solution)`.

---

# 3. prompt_builder.py
Đây là hàm $p(h_1, h_2, \varepsilon)$ chuyển hóa ngữ cảnh và code cũ thành câu lệnh gửi cho AI.

```python
1: """
2: prompt_builder.py — Xây dựng Structured Prompt cho LLM.
3: """
4: 
5: def build_evolution_prompt(
6:     parent1_code: str,
7:     parent1_fitness: float,
8:     parent2_code: str,
9:     parent2_fitness: float,
10:     generation: int,
11:     best_fitness_so_far: float,
12:     context_notes: str = "",
13: ) -> str:
```
- **Dòng 5-13**: Hàm nhận vào mã nguồn của 2 thuật toán cha mẹ (`parent1_code`, `parent2_code`), điểm số hiệu năng thực tế của chúng (`parent1_fitness`, `parent2_fitness`), số thế hệ hiện tại, điểm kỷ lục tốt nhất từ trước đến nay, và các ghi chú phản hồi lỗi/gợi ý (`context_notes`).

```python
40:     metacognitive_mandate = """You are an expert algorithm designer specializing in Vehicle Routing Problems (VRP).
41: You are evolving the `select_parents` function used inside the Hybrid Genetic Search (HGS) algorithm
42: implemented by the PyVRP library (version 0.14.x).
```
- **Dòng 40-42**: Thiết lập vai trò chuyên gia (Role-prompting) và bối cảnh cụ thể của bài toán tối ưu lộ trình xe cộ.

```python
47: IMPORTANT CONSTRAINTS:
48: - The function MUST have this exact signature:
49:   def select_parents(population, rng, cost_evaluator, k=2):
...
55: - The function MUST return a tuple of exactly 2 Solution objects: (parent1, parent2)
56: - You may use: sol.is_feasible(), sol.routes(), sol.num_routes()
```
- **Dòng 47-56**: Đưa ra các ràng buộc bất di bất dịch về chữ ký hàm và giá trị trả về để đảm bảo code sinh ra tương thích tuyệt đối với PyVRP.

```python
57: - CRITICAL FOR PyVRP 0.14+ (PREVENT TYPE ERROR):
58:   * When iterating or extracting clients from a route, DO NOT assume elements are integers! In PyVRP 0.14, iterating a Route yields `ScheduledActivity` objects.
59:   * To get client integer IDs correctly, ALWAYS use this pattern:
60:     ```python
61:     # Extract client IDs safely from a route:
62:     def get_route_clients(route):
63:         return [getattr(act, "client", act) for act in route if getattr(act, "client", act) > 0]
64:     ```
```
- **Dòng 57-64 (Cực kỳ quan trọng)**: Đây là phần phòng ngừa lỗi kiểu dữ liệu. Hướng dẫn trực tiếp cho LLM rằng trong PyVRP 0.14, một tuyến đường không chứa trực tiếp số nguyên `int` mà chứa `ScheduledActivity`, bắt buộc phải dùng hàm bảo vệ để lấy ra ID nguyên.

```python
83: === PARENT HEURISTIC 1 (Fitness: {parent1_fitness:.2f} — lower is better) ===
84: ```python
85: {parent1_code.strip()}
86: ```
87: === PARENT HEURISTIC 2 (Fitness: {parent2_fitness:.2f} — lower is better) ===
88: ```python
89: {parent2_code.strip()}
90: ```
```
- **Dòng 83-90**: Nhúng toàn bộ mã nguồn của 2 thuật toán cha mẹ kèm điểm số đo đạc được vào prompt để AI phân tích.

```python
116: === YOUR TASK: REASON -> ACT -> REFLECT ===
117: **STEP 1 — REASON (Diagnosis & Hypothesis):**
118: Analyze both parent heuristics...
124: **STEP 2 — ACT (Generate Code):**
125: Write a complete, new `select_parents` function...
130: **STEP 3 — REFLECT (Self-Evaluation):**
131: Briefly explain...
```
- **Dòng 116-131**: Ép AI tư duy theo chuỗi CoT (Chain-of-Thought) có cấu trúc:
  1. Phân tích nguyên nhân tại sao thuật toán cũ chạy tốt/kém và đưa ra giả thuyết cải tiến.
  2. Viết mã nguồn hoàn chỉnh hiện thực hóa giả thuyết.
  3. Tự phản biện lại những rủi ro của đoạn mã mới vừa viết.

---

# 4. llm_client.py
Tệp này giao tiếp với máy chủ Google AI và bóc tách code sạch.

```python
16: def create_client():
17:     """Tạo client kết nối tới Gemini API."""
18:     return genai.Client(api_key=config.GEMINI_API_KEY)
```
- **Dòng 16-18**: Khởi tạo kết nối với Google GenAI SDK bằng API Key đã khai báo trong config.

```python
21: def call_llm(prompt: str, max_retries: int = 3) -> Optional[str]:
...
27:     client = create_client()
28:     for attempt in range(max_retries):
29:         try:
30:             response = client.models.generate_content(
31:                 model=config.GEMINI_MODEL,
32:                 contents=prompt,
33:             )
34:             if response and response.text:
35:                 return response.text
...
37:         except Exception as e:
38:             wait = 5 * (attempt + 1)
39:             time.sleep(wait)
```
- **Dòng 21-39**: Vòng lặp gửi yêu cầu tới Gemini. Nếu xảy ra nghẽn mạng hoặc vượt hạn ngạch tạm thời, hàm sẽ bắt ngoại lệ (`except`), tự động ngủ tăng dần 5s, 10s (`time.sleep`) rồi thử lại tối đa 3 lần (`max_retries`).

```python
55: def extract_code_from_response(response_text: str) -> Optional[str]:
...
74:     code_blocks = re.findall(
75:         r"```python\s*\n(.*?)```",
76:         response_text,
77:         re.DOTALL,
78:     )
```
- **Dòng 74-78**: Sử dụng Regex `re.findall` quét toàn bộ văn bản phản hồi để tìm khối code nằm giữa ````python ... ````. Cờ `re.DOTALL` giúp dấu chấm `.` khớp với cả ký tự xuống dòng `\n`.

```python
92:     for block in code_blocks:
93:         if "def select_parents" in block:
94:             return block.strip()
```
- **Dòng 92-94**: Lọc qua các khối mã trích xuất được, khối nào có chứa định nghĩa `def select_parents` thì đó chính là hàm thuật toán cần lấy.

---

# 5. code_sandbox.py
Hộp cát kiểm thử tính an toàn và tương thích trước khi đưa code vào môi trường chạy thật.

```python
17: # POLYFILL CHO PYVRP 0.14+:
18: try:
19:     from pyvrp._pyvrp import ScheduledActivity
20: 
21:     def _act_val(obj):
22:         return obj.idx if hasattr(obj, "idx") else obj
23: 
24:     ScheduledActivity.__lt__ = lambda self, other: self.idx < _act_val(other)
25:     ScheduledActivity.__gt__ = lambda self, other: self.idx > _act_val(other)
26:     ScheduledActivity.__le__ = lambda self, other: self.idx <= _act_val(other)
27:     ScheduledActivity.__ge__ = lambda self, other: self.idx >= _act_val(other)
28:     ScheduledActivity.__eq__ = lambda self, other: self.idx == _act_val(other)
29:     ScheduledActivity.__int__ = lambda self: self.idx
30:     ScheduledActivity.__index__ = lambda self: self.idx
31:     ScheduledActivity.__hash__ = lambda self: hash(self.idx)
32: except ImportError:
33:     pass
```
- **Dòng 17-33 (Bảo bối ngăn chặn lỗi `TypeError`)**: Đây là kỹ thuật **Monkey-Patching (Polyfill)**. 
  - Lớp `ScheduledActivity` của C++ binding vốn dĩ không cho phép so sánh nhỏ hơn/lớn hơn với số nguyên `int`.
  - Đoạn code này gán các hàm ẩn danh (`lambda`) vào các phương thức ma thuật (`__lt__` cho phép `<`, `__gt__` cho phép `>`, `__hash__` cho phép đưa vào `set/dict`). 
  - Nhờ vậy, bất kỳ đoạn code nào do AI sinh ra lỡ viết so sánh `node > 0` hay `min(u, v)` thì Python tự động lấy `self.idx` để so sánh, triệt tiêu 100% nguy cơ văng lỗi kiểu dữ liệu.

```python
45: def validate_code(code: str) -> Tuple[bool, str]:
46:     try:
47:         tree = ast.parse(code)
48:     except SyntaxError as e:
49:         return False, f"Lỗi cú pháp dòng {e.lineno}: {e.msg}"
```
- **Dòng 45-49**: Sử dụng thư viện `ast` (Abstract Syntax Tree) để biên dịch thử mã nguồn ở dạng cây cú pháp. Nếu AI viết thiếu ngoặc, sai thụt lề, hàm sẽ phát hiện và chặn lại ngay lập tức mà không gây crash hệ thống.

```python
73:     dangerous = {"os", "sys", "subprocess", "shutil", "pathlib"}
74:     for mod in module_names:
75:         if mod and mod.split(".")[0] in dangerous:
76:             return False, f"Import không được phép: '{mod}'"
```
- **Dòng 73-76**: Kiểm tra an toàn bảo mật (Security check). Chặn mọi nỗ lực của AI muốn import các thư viện thao tác file hoặc can thiệp hệ điều hành.

```python
83: def compile_heuristic(code: str) -> Tuple[Optional[object], str]:
...
93:         namespace = {}
94:         exec(code, namespace)
95:         func = namespace.get("select_parents")
```
- **Dòng 93-95**: Dùng lệnh `exec()` để thực thi mã nguồn chuỗi trong một từ điển cô lập `namespace = {}`. Sau đó lấy hàm `select_parents` ra ngoài dưới dạng một đối tượng Python hàm (`callable`) có thể gọi được.

```python
107: def quick_smoke_test(func, data, rng_class, cost_eval_class, solution_class) -> Tuple[bool, str]:
...
127:         population = [solution_class.make_random(data, rng) for _ in range(10)]
128:         result = func(population, rng, cost_eval)
129:         if not isinstance(result, tuple) or len(result) != 2:
130:             return False, "Hàm phải trả về tuple 2 phần tử"
```
- **Dòng 107-130**: Chạy thử hàm với 10 cá thể nghiệm ngẫu nhiên trong vài mili-giây. Kiểm tra xem khi thực thi thực tế hàm có bị vòng lặp vô tận, chia cho 0, hay trả về sai cấu trúc `(parent1, parent2)` hay không.

---

# 6. fitness.py
File này đo lường giá trị thực tiễn của đoạn mã do AI viết thông qua việc giải thử bài toán VRP.

```python
75: def order_crossover(parent1_routes, parent2_routes, num_clients, py_rng):
```
- **Dòng 75**: Hàm lai ghép thứ tự (Order Crossover - OX) chuẩn cho bài toán hoán vị đường đi:
  1. Trải phẳng các tuyến đường của cha và mẹ thành dãy số khách hàng.
  2. Cắt một đoạn ngẫu nhiên từ cha mẹ 1 giữ nguyên vị trí.
  3. Lấy các khách hàng còn lại từ cha mẹ 2 điền vào các vị trí trống theo đúng thứ tự xuất hiện.
  4. Cắt dãy số con cái vừa tạo thành các tuyến đường mới cho đội xe.

```python
182: def evaluate_single_run(select_parents_func, data, seed: int, time_limit: float) -> float:
```
- **Dòng 182**: Đây chính là hàm thực thi $\text{Performance}(h, d_i)$ cho 1 lượt chạy:

```python
205:         pop_size = 20
206:         population = []
207:         for _ in range(pop_size):
208:             sol = Solution.make_random(data, rng)
209:             improved = ls(sol, cost_eval, exhaustive=True)
210:             population.append(improved)
```
- **Dòng 205-210**: Tạo 20 nghiệm khởi đầu ngẫu nhiên, sau đó lập tức dùng thuật toán tìm kiếm cục bộ chuyên sâu của PyVRP (`LocalSearch`) để tối ưu hóa, đưa các nghiệm về trạng thái tối ưu cục bộ.

```python
216:         while time.time() - start_time < time_limit:
217:             iteration += 1
218:             # (a) Gọi hàm select_parents do LLM sinh ra
219:             p1, p2 = select_parents_func(population, rng, cost_eval)
```
- **Dòng 216-219**: Vòng lặp tối ưu hóa chạy liên tục trong đúng 30 giây (`time_limit`). Trong mỗi vòng lặp, hệ thống **trao quyền cho hàm `select_parents` của AI** lựa chọn ra 2 cá thể bố mẹ $p_1, p_2$.

```python
229:             child_routes = order_crossover(routes1, routes2, data.num_clients, py_rng)
230:             child = build_solution_from_routes(data, child_routes)
231:             child_improved = ls(child, cost_eval, exhaustive=False)
```
- **Dòng 229-231**: Lai ghép 2 cha mẹ vừa chọn bằng OX để sinh ra nghiệm con `child`, sau đó dùng `LocalSearch` tinh chỉnh nghiệm con.

```python
234:             child_cost = cost_eval.penalised_cost(child_improved)
235:             worst_cost = cost_eval.penalised_cost(population[-1])
236:             if child_cost < worst_cost:
237:                 population[-1] = child_improved
238:                 population.sort(key=lambda s: cost_eval.penalised_cost(s))
```
- **Dòng 234-238**: Cơ chế đào thải: Nếu nghiệm con có chi phí thấp hơn nghiệm tệ nhất trong quần thể (`worst_cost`), lập tức thay thế cá thể tệ nhất và sắp xếp lại quần thể.
- **Ý nghĩa cốt lõi**: Nếu hàm `select_parents` của AI chọn bố mẹ khéo léo (vừa tốt vừa đa dạng cấu trúc), con cái sinh ra sẽ đột phá chi phí rất nhanh. Ngược lại, nếu chọn bố mẹ tồi, con cái sinh ra sẽ vô dụng và không thể cải thiện quần thể.

---

# 7. population.py
File này quản lý quần thể các thuật toán ($P^t$).

```python
17: @dataclass
18: class Heuristic:
19:     code: str                              # Mã nguồn Python của hàm
20:     fitness: float = float("inf")          # Chi phí VRP (càng thấp càng tốt)
21:     generation_born: int = 0               # Thế hệ sinh ra
22:     description: str = ""                  # Phản biện của AI về thuật toán
```
- **Dòng 17-22**: Định nghĩa một cá thể thuật toán trong không gian tiến hóa mã nguồn.

```python
65:     def select_two_parents(self, rng_seed: int = 42) -> tuple[Heuristic, Heuristic]:
...
76:         sorted_pop = sorted(self.individuals, key=lambda h: h.fitness)
77:         parent1 = sorted_pop[0]
...
84:         parent2 = random.choice(sorted_pop[mid:])
85:         return parent1, parent2
```
- **Dòng 65-85**: Lựa chọn 2 thuật toán cha mẹ để gửi vào prompt cho AI thế hệ kế tiếp:
  - `parent1`: Luôn là thuật toán có `fitness` thấp nhất (tốt nhất) hiện tại $\rightarrow$ Tận dụng tri thức ưu tú (Exploitation).
  - `parent2`: Chọn ngẫu nhiên từ nửa dưới của quần thể $\rightarrow$ Giữ lại sự phong phú, đa dạng ý tưởng giải thuật (Exploration).

```python
87:     def select_survivors(self):
...
94:         self.individuals.sort(key=lambda h: h.fitness)
95:         removed = self.individuals[self.max_size:]
96:         self.individuals = self.individuals[:self.max_size]
```
- **Dòng 87-96**: Hiện thực hóa công thức $P^{t+1} \leftarrow \text{SelectSurvivors}(P^t \cup O^t; N)$. 
  - Gộp tất cả thuật toán cũ và mới lại.
  - Sắp xếp theo thứ tự chi phí tăng dần.
  - Cắt lấy đúng $N$ cá thể tốt nhất (`[:self.max_size]`), xóa bỏ các cá thể kém cỏi.

---

# 8. evolve.py
Tệp điều khiển trung tâm, kết nối toàn bộ hệ thống từ đầu đến cuối.

```python
18: import os
19: import sys
20: import time
21: import datetime
22: 
23: # Fix encoding cho Windows console để tránh UnicodeEncodeError
24: if hasattr(sys.stdout, "reconfigure"):
25:     try:
26:         sys.stdout.reconfigure(encoding="utf-8")
27:     except Exception:
28:         pass
```
- **Dòng 23-28**: Cấu hình luồng xuất chuẩn của Windows PowerShell/CMD sang chuẩn UTF-8, đảm bảo các ký tự tiếng Việt, dấu mũi tên, hộp đồ họa không bao giờ bị lỗi `UnicodeEncodeError`.

```python
74:     # 1. Kiểm tra API key
75:     if config.GEMINI_API_KEY == "YOUR_API_KEY_HERE":
76:         print("\n❌ LỖI: Bạn chưa nhập API key!")
77:         return
```
- **Dòng 74-77**: Chốt chặn cảnh báo nếu người dùng quên chưa điền key API.

```python
90:     # ==================================================================
91:     # PHASE I: DOMAIN-AWARE INITIALIZATION
92:     # ==================================================================
93:     h0_func, h0_error = compile_heuristic(BASE_HEURISTIC_CODE)
94:     h0_fitness, h0_details = evaluate_heuristic(h0_func)
95:     pop = Population(max_size=config.POPULATION_SIZE)
96:     pop.add(Heuristic(code=BASE_HEURISTIC_CODE, fitness=h0_fitness, generation_born=0))
```
- **Dòng 90-96**: **Thực hiện Phase I trong bài báo**. 
  - Biên dịch đoạn mã gốc $h_0$.
  - Đánh giá benchmark điểm số ban đầu của $h_0$ trên bài toán VRP ($f(h_0, D)$).
  - Khởi tạo quần thể thế hệ số 0 ($P^0$) nạp $h_0$ làm hạt giống đầu tiên.

```python
136:     # ==================================================================
137:     # PHASE II: LLM GENERATION — VÒNG LẶP TIẾN HÓA (t = 1..T)
138:     # ==================================================================
139:     for gen in range(1, config.NUM_GENERATIONS + 1):
```
- **Dòng 139**: Vòng lặp lớn của Phase II, tiến hóa qua $T$ thế hệ (`NUM_GENERATIONS = 10`).

```python
144:         for offspring_idx in range(config.OFFSPRING_PER_GENERATION):
145:             # Bước 1: Chọn 2 cha mẹ h1, h2
146:             parent1, parent2 = pop.select_two_parents(...)
```
- **Dòng 144-146**: Với mỗi thế hệ, lặp lại việc sinh ra các cá thể con `offspring`. Chọn 2 thuật toán cha mẹ từ quần thể hiện tại.

```python
159:             # Bước 2: Xây Structured Prompt p(h1, h2, ε)
160:             prompt = build_evolution_prompt(...)
161: 
162:             # Bước 3: Gọi LLM -> h' = LLM(p; Θ)
163:             code, reflection = generate_heuristic(prompt)
```
- **Dòng 159-163**: Đóng gói mã nguồn 2 cha mẹ thành prompt chuẩn rồi gửi cho Gemini. Gemini phân tích và trả về mã nguồn mới $h'$.

```python
182:             # Bước 4: Validate & Compile h'
183:             func, error = compile_heuristic(code)
184:             passed, err = quick_smoke_test(func, ...)
```
- **Dòng 182-184**: Đưa đoạn mã do AI viết vào hộp cát kiểm tra cú pháp và smoke test nhanh. Nếu code bị lỗi cú pháp, vòng lặp tự động bỏ qua và nạp lỗi vào `context_notes` để thế hệ sau AI rút kinh nghiệm.

```python
198:             # Bước 5: Đánh giá fitness f(h', D)
199:             fitness, details = evaluate_heuristic(func)
```
- **Dòng 198-199**: Chạy giải thực tế bài toán VRP trong 30 giây bằng hàm mới của AI để đo chi phí thực tế.

```python
210:             # Bước 6: Thêm offspring vào quần thể
211:             pop.add(Heuristic(code=code, fitness=fitness, generation_born=gen))
```
- **Dòng 210-211**: Nạp thuật toán mới vào tập con cái thế hệ $t$ ($O^t$).

```python
233:         # SelectSurvivors: P^{t+1} <- SelectSurvivors(P^t ∪ O^t; N)
234:         pop.select_survivors()
```
- **Dòng 233-234**: Kết thúc một thế hệ, thanh lọc toàn bộ các thuật toán kém, chỉ giữ lại đúng $N$ thuật toán có chi phí thấp nhất để làm cha mẹ cho thế hệ kế tiếp.

```python
252:     # ==================================================================
253:     # PHASE III: SOLUTION SELECTION — Best-of-N
254:     # ==================================================================
255:     best = pop.get_best()
256:     best_code_path = os.path.join(config.RESULTS_DIR, "best_select_parents.py")
257:     with open(best_code_path, "w", encoding="utf-8") as f:
258:         f.write(best.code)
```
- **Dòng 252-258**: **Thực hiện Phase III trong bài báo**. 
  - Sau khi trải qua tất cả các thế hệ, lấy ra thuật toán có điểm số kỷ lục thấp nhất ($h^* = \arg\max f(h)$).
  - Xuất ra file mã nguồn hoàn chỉnh tại `results/best_select_parents.py`.

---

## TỔNG KẾT BÀI HỌC
Khi xem xét toàn bộ hệ thống, bạn có thể thấy đây là một chu trình khép kín:
$$\text{Code thô } (h_0) \xrightarrow[\text{Thế hệ 1}]{\text{LLM phân tích}} \text{Code lai } (h_1) \xrightarrow[\text{Thế hệ 2}]{\text{LLM tối ưu}} \dots \xrightarrow[\text{Thế hệ T}]{\text{Chọn lọc tự nhiên}} \text{Code hoàn hảo } (h^*)$$

Chính nhờ chu trình tự động này mà các chiến lược tối ưu phức tạp như **phân tầng Elite/Mid/Tail, đo độ tương đồng Jaccard, kiểm soát đa dạng cấu trúc** đã được AI tự động "nghĩ ra" và hoàn thiện từng bước!
