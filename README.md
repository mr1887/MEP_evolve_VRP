# MEP_evolve_VRP

Khung thử nghiệm tiến hóa heuristic `select_parents` cho bài toán Vehicle Routing Problem (VRP), sử dụng LLM để đề xuất các phiên bản mã Python mới và đánh giá chúng bằng một bộ giải VRP tự xây dựng trên nền các thành phần tìm kiếm của PyVRP.

> **Trạng thái tái hiện:** Đây là một prototype nghiên cứu theo ý tưởng LLM-driven heuristic evolution. Chưa đủ căn cứ để xem repository là bản tái hiện trung thực/toàn bộ phương pháp MEP trong bài báo PyVRP+. Đặc biệt, `fitness.py` hiện đánh giá heuristic trong một GA tùy chỉnh có OX crossover, không phải trong nguyên bản Hybrid Genetic Search (HGS) của PyVRP+.

## 1. Mục tiêu

Repository tập trung vào một thành phần của HGS:

- `select_parents`: chọn hai nghiệm VRP làm cha mẹ để tạo nghiệm con.
- LLM (Gemini) nhận mã của hai heuristic cha mẹ, phân tích và sinh một phiên bản `select_parents` mới.
- Mã mới được kiểm tra cú pháp, biên dịch, smoke-test và đánh giá trên các instance VRP.
- Các heuristic có fitness tốt được giữ trong quần thể để tiếp tục tiến hóa.

Phạm vi này chỉ nhắm tới `select_parents`; nó không tái hiện đồng thời các thành phần `select_survivors` và `update_penalties` của PyVRP+.

## 2. Kiến trúc tổng thể

```text
base_heuristic.py
      |
      v
population.py -------------------------------+
      | chọn hai heuristic cha mẹ            |
      v                                      |
prompt_builder.py                            |
      | tạo prompt Reason-Act-Reflect        |
      v                                      |
llm_client.py                                |
      | gọi Gemini, trích code và reflection |
      v                                      |
code_sandbox.py                              |
      | validate, compile, smoke test        |
      v                                      |
fitness.py                                   |
      | đánh giá select_parents trong        |
      | GA tùy chỉnh + PyVRP LocalSearch     |
      v                                      |
population.py                                |
      | thêm offspring, giữ top-N            |
      +--------------------------------------+
      |
      v
evolve.py: lặp qua các thế hệ và lưu heuristic tốt nhất
```

## 3. Các module

| File | Trách nhiệm |
|---|---|
| `base_heuristic.py` | Khai báo heuristic khởi tạo `h0`: tournament selection với tournament size tối đa 5, xếp hạng theo penalised cost. |
| `prompt_builder.py` | Tạo prompt cho Gemini, đưa vào hai heuristic cha, fitness, thế hệ hiện tại, best fitness và ghi chú ngữ cảnh; yêu cầu đầu ra theo Reason → Act → Reflect. |
| `llm_client.py` | Gọi Gemini API, thử lại khi lỗi, trích xuất khối code Python và phần reflection từ phản hồi. |
| `code_sandbox.py` | Phân tích cú pháp AST, kiểm tra sự tồn tại/chữ ký cơ bản của `select_parents`, compile và chạy smoke test. |
| `fitness.py` | Đọc instance VRP; tạo quần thể nghiệm; gọi heuristic chọn cha mẹ; tạo con bằng Order Crossover (OX); cải thiện bằng PyVRP LocalSearch; tính chi phí trung bình. |
| `population.py` | Lưu các heuristic cùng code, fitness và metadata; chọn hai heuristic cha mẹ; giữ tối đa `max_size` heuristic có fitness thấp nhất. |
| `evolve.py` | Điều phối các phase: khởi tạo, sinh offspring qua LLM, kiểm tra, đánh giá, cập nhật quần thể, checkpoint và lưu heuristic tốt nhất. |
| `data/X-n101-k25.vrp` | Instance CVRP X-n101-k25 được commit trong repository. |

Các module trên import `config` để lấy API key, đường dẫn dữ liệu, danh sách instance và tham số chạy. **Tại thời điểm viết README, file `config.py` không có trong nhánh `main` mà connector đọc được.** Vì vậy cấu hình và hướng dẫn chạy end-to-end chưa thể được xác minh từ repository hiện tại.

## 4. Luồng hoạt động chi tiết

### Phase I — Khởi tạo

1. Kiểm tra Gemini API key trong cấu hình.
2. Đọc các instance VRP được cấu hình.
3. Compile heuristic gốc `h0`.
4. Chạy smoke test.
5. Đánh giá fitness của `h0`.
6. Tạo quần thể heuristic ban đầu.

### Phase II — Tiến hóa

Với mỗi thế hệ và mỗi offspring:

1. `Population.select_two_parents()` chọn hai **heuristic cha mẹ** (hai chương trình `select_parents`).
2. `build_evolution_prompt()` ghép mã cha mẹ và ngữ cảnh thành prompt.
3. Gemini thực hiện nhiệm vụ Reason–Act–Reflect và sinh code heuristic mới.
4. `compile_heuristic()` kiểm tra và biên dịch code.
5. `quick_smoke_test()` kiểm tra khả năng gọi hàm và dạng kết quả cơ bản.
6. `evaluate_heuristic()` đo fitness bằng cách chạy bộ đánh giá trong `fitness.py`.
7. Offspring hợp lệ được thêm vào quần thể.
8. `select_survivors()` giữ lại tối đa N heuristic có chi phí trung bình thấp nhất.

### Phase III — Chọn kết quả

Chọn heuristic có fitness thấp nhất trong quần thể cuối, lưu code và lưu quần thể/checkpoint.

**Chú ý hai cấp độ chọn cha mẹ:**

- `Population.select_two_parents()`: chọn hai *chương trình heuristic* để LLM tham khảo khi tạo chương trình mới.
- Hàm `select_parents()` do LLM sinh: chọn hai *nghiệm VRP* trong quần thể nghiệm của bộ đánh giá.

Đây là hai thao tác khác nhau.

## 5. Cách tính fitness hiện tại

Trong `fitness.py`, fitness được tính là chi phí trung bình trên các instance và các lần chạy; **giá trị thấp hơn được xem là tốt hơn**.

Bộ đánh giá hiện tại không gọi một API `pyvrp.solve()` có thể cắm trực tiếp `select_parents`. Thay vào đó, code tự xây dựng vòng lặp GA:

1. Tạo 20 nghiệm ngẫu nhiên và cải thiện bằng PyVRP LocalSearch.
2. Gọi heuristic `select_parents` để lấy hai nghiệm cha mẹ.
3. Dùng Order Crossover (OX) tự cài đặt trên danh sách khách hàng đã làm phẳng.
4. Cải thiện nghiệm con bằng LocalSearch.
5. Nếu con tốt hơn cá thể tệ nhất, thay thế cá thể đó.

Do đó, fitness hiện đo hiệu quả của `select_parents` **trong GA tùy chỉnh này**. Không nên diễn giải kết quả đó thành hiệu quả của heuristic trong HGS gốc nếu chưa thay bộ đánh giá bằng đúng HGS/crossover và cơ chế liên quan của bài báo.

## 6. Cài đặt và chạy

Repository hiện thiếu `config.py` trong nhánh `main`, trong khi các module chính import file này. Vì vậy, hướng dẫn chạy hoàn chỉnh chưa thể cung cấp một cách đáng tin cậy.

Các dependency thể hiện trực tiếp qua imports gồm:

- Python
- `pyvrp`
- `google-genai` (module `google.genai`)
- Các thư viện chuẩn Python

Trước khi chạy cần khôi phục/bổ sung `config.py` với ít nhất các cấu hình mà code đang tham chiếu:

- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `DATA_DIR`
- `VRP_INSTANCES`
- `POPULATION_SIZE`
- `NUM_GENERATIONS`
- `OFFSPRING_PER_GENERATION`
- `HGS_TIME_LIMIT`
- `NUM_EVAL_RUNS`
- `GENERATED_CODE_DIR`
- `RESULTS_DIR`

Không commit API key trực tiếp lên GitHub. Nên đọc key từ biến môi trường hoặc file `.env` không được commit.

Sau khi bổ sung cấu hình và xác nhận dependency tương thích, entry point dự kiến là:

```bash
python evolve.py
```

Lệnh này là entry point được khai báo trong `evolve.py`; chưa được xác nhận chạy thành công trên một môi trường sạch.

## 7. Hạn chế kỹ thuật cần xử lý

1. **Chưa dùng HGS gốc làm evaluator.** `fitness.py` sử dụng GA tùy chỉnh và OX crossover; kết quả không thể so sánh trực tiếp với kết quả HGS trong bài báo.
2. **Chưa có cấu hình chạy được trong repository hiện tại.** `config.py` được import nhưng không có trong nhánh `main` mà README này kiểm tra.
3. **Smoke test còn yếu.** Nó kiểm tra tuple hai phần tử và một vài thuộc tính, nhưng chưa xác nhận hai parent thuộc population, không trùng nhau khi có thể, hay kiểm tra các invariant của nghiệm.
4. **`code_sandbox.py` chưa phải sandbox an toàn.** `exec(code, namespace)` vẫn chạy với builtins mặc định; chặn một số import bằng AST không đủ để cô lập mã không tin cậy. Chỉ chạy code LLM trong môi trường cô lập thực sự.
5. **OX có thể che giấu lỗi tạo nghiệm.** `build_solution_from_routes()` bắt mọi exception và thay bằng nghiệm ngẫu nhiên, khiến lỗi crossover/solution construction có thể không được phản ánh rõ trong kết quả.
6. **Thiết kế thí nghiệm cần được kiểm soát.** Cần báo cáo nhiều seed, mean/std hoặc khoảng tin cậy, cùng benchmark train/validation/test tách biệt; đồng thời đảm bảo các heuristic được so sánh dùng cùng ngân sách thời gian và điều kiện chạy.
7. **Quản lý quần thể heuristic khác với chọn cha mẹ nghiệm VRP.** Cần mô tả rõ hai tầng lựa chọn này trong báo cáo và slide.

## 8. Những bước nên làm tiếp theo để tăng độ trung thực với bài báo

- Khôi phục cấu hình và khóa phiên bản dependency để dự án chạy tái lập được.
- Tích hợp heuristic vào đúng HGS evaluator mà bài báo sử dụng; nếu không thể, ghi rõ đây là evaluator thay thế và không tuyên bố tái lập kết quả HGS.
- Xác minh cách tính fitness và selection/survivor policy so với mô tả gốc của MEP.
- Bổ sung kiểm thử đơn vị cho các trường hợp population nhỏ, nghiệm trùng, nghiệm không khả thi và lỗi crossover.
- Chạy baseline và heuristic mới trên cùng instance, cùng seed, cùng thời gian; lưu kết quả từng lần chạy.
- Tách tập tiến hóa và tập kiểm thử độc lập để kiểm tra khả năng tổng quát hóa.

## 9. Trạng thái xác minh

README này được viết dựa trên mã nguồn đọc được ở nhánh `main`. Chưa thực thi repository end-to-end trong môi trường sạch; do thiếu `config.py`, không thể xác nhận lệnh chạy thành công hoặc xác nhận các con số kết quả thực nghiệm.

