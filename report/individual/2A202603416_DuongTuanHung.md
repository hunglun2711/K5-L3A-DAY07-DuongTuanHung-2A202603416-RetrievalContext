# Báo cáo Day 7 — Agentic RAG (Retrieval & Context Engineering)

- **Sinh viên thực hiện:** Dương Tuấn Hưng
- **MSSV:** `2A202603416`
- **Lớp:** `K5-L3A`
- **Tài liệu báo cáo:** Kết nối hệ sinh thái Day 6 (OpenStax Python), đánh giá Retrieval & Multi-Tool Agent phần Cơ bản (`data/day7/`, `results/core/`), và hoàn thiện Thử thách nâng cao tài liệu 100 trang đa phương thức (`advanced/`).

---

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Bộ tài liệu kế thừa từ Day 6 là 4 trích đoạn của giáo trình mở **OpenStax — Introduction to Python Programming** (Rice University, năm `2024`), phát hành dưới giấy phép **Creative Commons Attribution 4.0 International (CC BY 4.0)**, URL gốc: `https://assets.openstax.org/oscms-prodcms/media/documents/Introduction_to_Python_Programming-WEB.pdf`.
  - Gồm 4 phần (`openstax-py-ch01`, `openstax-py-ch03`, `openstax-py-ch04`, `openstax-py-ch05`) lưu bản chuẩn hóa tại [data/day7/manifest.json](../../data/day7/manifest.json) và [data/day7/chunks.json](../../data/day7/chunks.json): tổng cộng `44` trang PDF vật lý, sau khi loại `3` trang không mang nội dung học thuật ở Chương 1 (trang 1 trống `0` ký tự, trang 2 bìa `199` ký tự, trang 3 mục lục `720` ký tự) còn **`41` trang nội dung hợp lệ**, **`54,015` ký tự chuẩn hóa** và **`95` chunks**.

- **Vector store, embedding model, phiên bản và số chiều:**
  - Vector store: **Qdrant local** tại `starter/data/qdrant_storage`, collection `day6_chunks`, filter `strategy="B"`, độ đo khoảng cách **`Cosine`** (`starter/data/qdrant_storage/meta.json`).
  - Embedding model: **`openai/text-embedding-3-small`** gọi qua OpenRouter API, kích thước **`1536` chiều**, có lưu cache cục bộ tại `starter/data/.embedding_cache.json` để tái lập chính xác và tiết kiệm chi phí truy vấn.

- **Agent model, cách chạy và phân chia dev/test:**
  - Chat / Controller / Reranker model phần cơ bản: **`openai/gpt-4o-mini`** qua OpenRouter API (tập cấu hình tại `src/retrievers.py`, `src/agent_system.py`).
  - Quy trình tách biệt nghiêm ngặt `dev` và `test` qua CLI:
    - Chuẩn bị dữ liệu: `python src/prepare_day7_data.py`
    - Retrieval dev & test: `python src/run_retrieval_experiments.py --stage dev` -> chốt cấu hình -> `python src/run_retrieval_experiments.py --stage test --chosen improve2_hybrid_llm_rerank_adj2200`
    - Agent dev & test: `python src/run_agent_experiments.py --stage dev` -> `python src/run_agent_experiments.py --stage test`
    - Tổng hợp & audit: `python src/audit_core_results.py`
  - Phân chia benchmark phần cơ bản tại [data/day7/cases.json](../../data/day7/cases.json) (tổng **`29` câu**, vượt mức tối thiểu 24 câu):
    - **`dev` (`21` câu):** `18` câu có đáp án (`q01`–`q18`, gồm cả câu `single_hop` và `multi_evidence`) + `3` câu thiếu bằng chứng (`u01`–`u03`).
    - **`test` (`8` câu):** `6` câu có đáp án (`q19`–`q24`) + `2` câu thiếu bằng chứng (`u04`–`u05`).

- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - **Câu có đáp án** ([data/day7/questions.json](../../data/day7/questions.json)): mỗi câu đi kèm danh sách `gold` gồm `source_id`, số trang `page` và trích đoạn nguyên văn `evidence` được kiểm tra khớp chính xác trên chunk thật bằng `starter/baseline.py` (`hit`, `validate`). Với câu multi-evidence (`multi_evidence: true`), chỉ tính `Context Hit` khi tất cả bằng chứng trong `gold` đều có mặt trong context cuối cùng (`compute_context_hit` trong [src/run_retrieval_experiments.py](../../src/run_retrieval_experiments.py)).
  - **Câu thiếu bằng chứng** ([data/day7/unanswerable.json](../../data/day7/unanswerable.json) và [data/day7/unanswerable_verification.json](../../data/day7/unanswerable_verification.json)): được thiết kế xoay quanh các chủ đề Python nâng cao không xuất hiện trong 4 chương trích đoạn (`match/case` structural pattern matching, dictionary hash table & `O(1)` complexity, `async def`/`await` `asyncio`, `yield` generator & `StopIteration`, `C3 linearization` / `MRO`). Toàn bộ `95` chunks đã được quét regex toàn văn bản để xác nhận `confirmed_absent: true`.
  - **Tiêu chí chấm:**
    - Retrieval chấm tự động `Hit@1`, `Evidence Hit@3`, `Context Evidence Rate` trên câu có đáp án; đồng thời trên `test` có **chấm thủ công Context thang `0/1/2`** tại [results/core/retrieval-test/context_manual_audit.json](../../results/core/retrieval-test/context_manual_audit.json).
    - Agent được đánh giá thủ công từng câu theo thang `0/1/2` cho `Grounded Answer`, kiểm tra `citation_support` (`true`/`false`) từng citation và xác nhận `abstention_correct` trên câu thiếu bằng chứng, sau đó tổng hợp qua `starter/agent_metrics.py`. Lỗi hạ tầng/API (`status="error"`) được tách riêng thành `provider_errors`, không gộp làm sai lệch chất lượng retrieval/generation.

---

## Retrieval và thử nghiệm

Thử nghiệm gồm **4 phương pháp retrieval khác nhau** (`TF-IDF`, `Dense Day 6`, `BM25`, `Hybrid RRF`) kết hợp **2 bước cải thiện liên tiếp** trên tập `dev` (`18` câu có đáp án), rồi chốt cấu hình tốt nhất `improve2_hybrid_llm_rerank_adj2200` để đánh giá độc lập trên `test` (`6` câu có đáp án). Lưu ý `improve1` và `improve2` vừa là mở rộng kiến trúc (thêm metadata/query-inferred section filter, LLM listwise reranking, adjacent chunk expansion) vừa cải thiện trực tiếp cách chọn context (`pack_context_adaptive`).

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline (`baseline_tfidf`) | TF-IDF lexical (`starter/baseline.py`), `top_k=3`, không filter, `budget=800` chars | Điểm chuẩn lexical tối giản | dev | `14/18` (`77.78%`) | `17/18` (`94.44%`) | `10/18` (`55.56%`) | median `0.287 ms` / `$0.00` |
| Phương pháp 2 (`method2_dense_day6`) | Dense Cosine trên Qdrant Day 6 (`text-embedding-3-small`), `top_k=3`, không filter, `budget=800` | Thử semantic search từ Day 6; dùng chung budget chặt để so công bằng | dev | `11/18` (`61.11%`) | `16/18` (`88.89%`) | `7/18` (`38.89%`) | median `5.074 ms` / `$0.00` (cache) |
| Phương pháp 3 (`method3_bm25`) | Okapi BM25 với `tokenize_code_aware`, `top_k=3`, không filter, `budget=800` | Tách token giữ nguyên ký hiệu code (`id()`, `int()`, `while`) để tăng khớp chính xác | dev | `15/18` (`83.33%`) | `18/18` (`100.00%`) | `9/18` (`50.00%`) | median `0.289 ms` / `$0.00` |
| Phương pháp 4 (`method4_hybrid_rrf`) | Hybrid RRF (Dense Day 6 + BM25, `rrf_k=60`, `candidate_k=12`), `top_k=3`, `budget=800` | Kết hợp semantic + lexical nhưng giữ nguyên budget `800` để tách tác động của thuật toán rank | dev | `14/18` (`77.78%`) | `17/18` (`94.44%`) | `8/18` (`44.44%`) | median `5.809 ms` / `$0.00` (cache) |
| Cải thiện 1 (`improve1_hybrid_filter_budget1800`) | Hybrid RRF + section filter (từ user / suy luận rule theo từ khóa) + `budget=1800` chars | Mở rộng budget và khoanh đúng chương mục giúp giữ đủ các chunk evidence cho câu multi-evidence | dev | `14/18` (`77.78%`) | `17/18` (`94.44%`) | `17/18` (`94.44%`) | median `6.088 ms` / `$0.00` (cache) |
| Cải thiện 2 — Đã chọn (`improve2_hybrid_llm_rerank_adj2200`) | Hybrid RRF `cand=8` + `gpt-4o-mini` listwise rerank + section filter + ghép chunk kề (`include_next_adjacent=True`) + `budget=2200` | Rerank ngữ nghĩa đưa đúng chunk chứa giải thích lên Rank 1 và nạp kèm chunk kế tiếp | dev | **`16/18` (`88.89%`)** | `17/18` (`94.44%`) | **`17/18` (`94.44%`)** | median `1621.517 ms` (`30.696 s` total) / `$0.004415` |
| Cấu hình đã chọn (`improve2_hybrid_llm_rerank_adj2200`) | Khóa nguyên cấu hình tốt nhất từ `dev` | Kiểm chứng độc lập trên tập giữ lại | test | **`4/6` (`66.67%`)** | **`6/6` (`100.00%`)** | **`6/6` (`100.00%`)** (audit tay `2.0/2.0`) | median `1687.189 ms` / `$0.001475` |
| Baseline (`baseline_tfidf`) | TF-IDF lexical, `top_k=3`, không filter, `budget=800` | Đối chứng trên cùng tập `test` | test | `3/6` (`50.00%`) | `5/6` (`83.33%`) | `1/6` (`16.67%`) (audit tay `0.5/2.0`) | median `0.365 ms` / `$0.00` |

**Nguồn số liệu:** [results/core/retrieval-dev/summary.json](../../results/core/retrieval-dev/summary.json), [results/core/retrieval-test/summary.json](../../results/core/retrieval-test/summary.json), [results/core/retrieval-test/context_manual_audit.json](../../results/core/retrieval-test/context_manual_audit.json).

### Hai lỗi tiêu biểu trên `dev`, nguyên nhân, cách thử và lý do chọn cấu hình

1. **Lỗi cắt cụt bằng chứng do `budget=800` ở các cấu hình đầu (`q02`, `q05`, `q06`, `q07`, `q11`, `q12`, `q14`):**
   - **Hiện tượng:** Dù `Hit@3` đạt `16/18` (`method2_dense_day6`), `17/18` (`baseline_tfidf`, `method4_hybrid_rrf`) và `18/18` (`method3_bm25`), `Context Evidence Rate` lại tụt xuống chỉ còn `7/18`–`10/18` (`38.9%`–`55.6%`).
   - **Nguyên nhân:** Hàm `pack_context` chỉ giữ `800` ký tự (~1.3 chunk trung bình), khiến chunk #2/#3 hoặc phần bằng chứng thứ hai của câu multi-evidence bị loại bỏ ngay lúc đóng gói context.
   - **Cách thử & khắc phục:** Trong `improve1_hybrid_filter_budget1800`, nâng `budget=1800` kết hợp filter theo chương mục (`PERMITTED_FILTER_KEYS` + `infer_section_filter_from_query` trong [src/retrievers.py](../../src/retrievers.py)). `Context Evidence Rate` nhảy vọt từ `8/18 (44.44%)` lên **`17/18 (94.44%)`**, trung bình context `1450.33` ký tự với chi phí tăng thêm `$0.00`.

2. **Lỗi lệch thứ hạng & thiếu đoạn văn liền kề (`q08`, `q10`, `q18`):**
   - **Hiện tượng:** Với `improve1`, `Hit@1` dừng ở `14/18` (`77.78%`), một số câu hỏi liên quan trực tiếp đoạn giải thích ngay sau đoạn khái niệm hoặc chứa hàm/cú pháp Python (`q08` về object identity/type, `q10`, `q18`) bị xếp sau các đoạn bài tập/ví dụ có nhiều từ khóa trùng lặp.
   - **Cách thử & khắc phục:** Trong `improve2_hybrid_llm_rerank_adj2200`, áp dụng `gpt-4o-mini` listwise reranking trên top-8 Hybrid RRF và bật `include_next_adjacent=True` (`budget=2200`). `Hit@1` tăng lên **`16/18` (`88.89%`)**, đồng thời câu `q18` có evidence đầy đủ ngay trong top-1/adjacent context. Với `q08`, chunk liền kề nằm ở phần sau chưa lọt vào `top_k=3` tĩnh — đây cũng là động lực chuyển sang Agent nhiều lượt gọi (`get_adjacent_chunks`) ở phần sau.
   - **Lý do chốt `improve2_hybrid_llm_rerank_adj2200`:** Cân bằng tốt nhất giữa độ chính xác (`16/18` Hit@1, `17/18` Context Hit trên `dev`), tốc độ (`~1.62 s/query`) và chi phí rất thấp (`$0.00025/query`). Khi chạy khóa trên `test`, cấu hình giữ vững **`6/6` (`100%`) Context Hit** và điểm audit tay tuyệt đối **`2.0/2.0`** (so với `1/6 = 16.67%` và `0.5/2.0` của `baseline_tfidf`).

---

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - **Cấu hình đơn công cụ (`baseline_tfidf`, `dense_day6`):** dùng vòng lặp `StarterLLMController` + `run_agent` (`starter/agentic.py`), chỉ có 1 tool tìm kiếm theo cấu hình (`tfidf` với `max_searches=2, budget=800`; `dense_day6` với `max_searches=3, budget=1800`).
  - **Cấu hình đa công cụ đã chọn (`chosen_multitool`, `MultiToolAgentRunner` trong [src/agent_system.py](../../src/agent_system.py)):** cấp cho `openai/gpt-4o-mini` **3 tool**:
    1. `hybrid_search`: gọi `LLMRerankedHybridRetriever` (Hybrid RRF + rerank `top_k=3`).
    2. `keyword_search`: gọi `BM25Retriever` cho ký hiệu, từ khóa code, tên hàm cụ thể.
    3. `get_adjacent_chunks`: lấy chunk liền trước/sau (`offset -1, 0, +1`) cùng trang để kiểm chứng đoạn nối tiếp.
  - **Điều kiện dừng & budget:** Tối đa `max_searches=4` lượt gọi tool (`với vòng lặp tối đa max_searches + 2 bước`), sau khi đạt trần sẽ khóa tool (`tools=None`) và ép model xuất JSON trả lời cuối cùng hoặc từ chối (`"Không đủ bằng chứng trong tài liệu để trả lời câu hỏi này."`). Context được cộng dồn theo ưu tiên kết quả mới nhất kết hợp kết quả cũ và chunk kề (`pack_context_adaptive(..., budget=2400, include_next_adjacent=True)`).
  - **Kiểm tra citation & xử lý lỗi:** Toàn bộ `citations` do model sinh ra được đối chiếu ngược với `final_context_ids` (loại bỏ/abstain nếu citation không có trong context thật). Ngoại lệ từ OpenRouter được ghi nhận riêng thành `status="error"` (`results/core/agent-test/u05_http402_incident.json` ghi nhận lỗi `HTTP 402` hết in-flight credit khi chạy song song, sau đó đã chạy lại tuần tự thành công với `0` lỗi hệ thống và bảo lưu file sự cố để minh bạch).

### Kết quả tổng hợp Agent (`dev` n=21 và `test` n=8)

| Cấu hình / split | Tool use | Số lượt gọi (searches / model calls) | Grounded answer `0/1/2` và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| `baseline_tfidf` (`dev`, n=21) | `21/21` | `24` searches / `45` model calls | `7` / `2` / `9` (N=18), TB **`1.1111`** | `9/11` | `11/11` | `3/3` | `7/18` |
| `dense_day6` (`dev`, n=21) | `21/21` | `21` searches / `42` model calls | `0` / `4` / `14` (N=18), TB **`1.7778`** | `16/20` | `18/18` | `3/3` | `0/18` |
| `chosen_multitool_v1` trước khi sửa (`dev`, n=21) | `21/21` | — | `4` / `0` / `14` (N=18), TB **`1.5556`** | `20/20` | `14/14` | `3/3` | `4/18` (`q01, q10, q11, q17`) |
| `chosen_multitool` đã chọn (`dev`, n=21) | `21/21` | `46` searches / `45` model calls | `0` / `1` / `17` (N=18), TB **`1.9444`** | **`24/25`** | **`18/18`** | **`3/3`** | **`0/18`** |
| `baseline_tfidf` (`test`, n=8) | `8/8` | `9` searches / `17` model calls | `3` / `2` / `1` (N=6), TB **`0.6667`** | `2/3` | `3/3` | `2/2` | `3/6` |
| **`chosen_multitool` đã chọn (`test`, n=8)** | **`8/8`** | **`20` searches / `20` model calls** | **`0` / `0` / `6` (N=6), TB `2.0000`** | **`9/9`** | **`6/6`** | **`2/2`** | **`0/6`** |

*(Toàn bộ 6 lần chạy đều ghi nhận `provider_errors: 0`.)*

### Chi tiết đánh giá từng câu tập `test` (`chosen_multitool`)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q19` | Có đáp án (Ch.1) | 2 lượt `hybrid_search` ("Python library Standard Library", "data analysis library") -> lấy đúng `p06_para_01..03` | Giải thích đầy đủ thư viện Python, Standard Library và các thư viện phổ biến (`[openstax-py-ch01_p06_para_01]`) | **`answer_grade=2`**, citation hợp lệ `1/1`: khớp nguyên văn trang 6 Ch.1 |
| `q20` | Có đáp án (Ch.3, multi-evidence) | 2 lượt `hybrid_search` ("container object", "character indexes numbered in string") -> lấy `p01_para_01`, `p01_para_02` | Trả lời đủ định nghĩa container object và đánh số chỉ mục ký tự từ 0 (`[openstax-py-ch03_p01_para_01, openstax-py-ch03_p01_para_02]`) | **`answer_grade=2`**, citation hợp lệ `2/2`: đủ cả 2 ý |
| `q21` | Có đáp án (Ch.4) | 2 lượt `hybrid_search` ("bool data type binary values", "Boolean keywords capitalization") -> lấy `p02_para_00` | Nêu đúng kiểu `bool`, từ khóa `True`/`False` viết hoa chữ cái đầu (`[openstax-py-ch04_p02_para_00]`) | **`answer_grade=2`**, citation hợp lệ `1/1` |
| `q22` | Có đáp án (Ch.5) | `hybrid_search` ("while loop") + `keyword_search` ("while") -> lấy `p02_para_00` | Trình bày chính xác điều kiện lặp khi biểu thức đúng và thoát khi sai (`[openstax-py-ch05_p02_para_00]`) | **`answer_grade=2`**, citation hợp lệ `1/1` |
| `q23` | Có đáp án (Ch.3, multi-evidence) | 2 lượt `keyword_search` lấy đủ cụm `p10_para_02`, `p11_para_00`, `p11_para_01` | Nêu đúng `id()`, `type()` và bản chất gán lại biến chỉ đổi tham chiếu đối tượng (`[p10_para_02, p11_para_00, p11_para_01]`) | **`answer_grade=2`**, citation hợp lệ `3/3`: chính xác hoàn toàn |
| `q24` | Có đáp án (Ch.5) | 2 lượt `hybrid_search` ("for loop iterate over containers", "string considered iterable") -> lấy `p05_para_00` | Giải thích đúng `for` duyệt container gồm `range`, `string`, `list` (`[openstax-py-ch05_p05_para_00]`) | **`answer_grade=2`**, citation hợp lệ `1/1` |
| `u04` | Thiếu bằng chứng (`yield` / `StopIteration`) | 4 lượt search (2 `keyword_search` + 2 `hybrid_search`) xác nhận không có chủ đề generator/StopIteration | `"Không đủ bằng chứng trong tài liệu để trả lời câu hỏi này."` (`citations=[]`) | **`abstention_correct=true`**: từ chối chính xác sau 4 lượt tìm |
| `u05` | Thiếu bằng chứng (`C3` / `MRO`) | 4 lượt search (3 `keyword_search` + 1 `hybrid_search`) xác nhận không có đa kế thừa / C3 | `"Không đủ bằng chứng trong tài liệu để trả lời câu hỏi này."` (`citations=[]`) | **`abstention_correct=true`**: từ chối chính xác |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Traces & nhãn `dev`: `results/core/agent-dev/traces_*.json`, `reviews_*.json`, `metrics_*.json`, `run_summary.json`.
  - Traces & nhãn `test`: [results/core/agent-test/traces_chosen_multitool.json](../../results/core/agent-test/traces_chosen_multitool.json), [reviews_chosen_multitool.json](../../results/core/agent-test/reviews_chosen_multitool.json), [metrics_chosen_multitool.json](../../results/core/agent-test/metrics_chosen_multitool.json), [run_summary.json](../../results/core/agent-test/run_summary.json).
  - Lệnh chạy: `python src/run_agent_experiments.py --stage dev` và `python src/run_agent_experiments.py --stage test` + `python src/audit_core_results.py`.
  - Thời gian & chi phí:
    - `dev`: `baseline_tfidf` (`82.04 s`, `$0.003660`), `dense_day6` (`120.35 s`, `$0.004685`), `chosen_multitool` (`147.53 s`, `$0.014090`).
    - `test`: `baseline_tfidf` (`32.85 s`, `$0.001347`), `chosen_multitool` (`88.08 s`, `$0.007453`).

- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - **Trước/sau sửa trên `dev`:** Phiên bản đầu `chosen_multitool_v1` bị **từ chối nhầm `4/18` câu** (`q01`, `q10`, `q11`, `q17`) khiến trung bình Grounded chỉ đạt `1.5556`. Nguyên nhân chính là chunk đầu trong `context` chưa chứa hết bằng chứng nhưng model ngại gọi tool tiếp hoặc gặp budget chật khi hợp nhất ứng viên; sau khi nới `budget=2400`, giữ `get_adjacent_chunks` và tối ưu hợp nhất kết quả theo lượt mới nhất, `false_abstention` giảm về **`0/18`**, điểm trung bình tăng lên **`1.9444`** trên `dev` và đạt tuyệt đối **`2.0000`** trên `test` (so với `0.6667` của `baseline_tfidf`, nơi TF-IDF bị false abstention tới `3/6` câu `q22`, `q23`, `q24` và đoán từ tham số ở `q21`).
  - **Ví dụ lỗi còn lại trên `dev` (`q08` — điểm `1`):** Câu trả lời trên `q08` trích đúng `openstax-py-ch03_p10_para_02` và `p11_para_01` nhưng diễn đạt tự mâu thuẫn (`"identity and type of the variable change, but only ... which object it refers to changes"`), làm giảm điểm grounded từ `2` xuống `1` và tính `1` citation chưa hậu thuẫn hoàn toàn (`24/25`).
  - **Đánh đổi & giới hạn:** `chosen_multitool` tiêu tốn chi phí gấp ~`5.5x` (`$0.00745` vs `$0.00135` trên test) và thời gian ~`2.7x` (`88.08 s` vs `32.85 s`) so với baseline đơn công cụ do LLM reranking + gọi nhiều tool, nhưng triệt tiêu hoàn toàn hiện tượng từ chối nhầm trên tập test. Bộ corpus OpenStax không có bảng biểu/đồ thị phức tạp nên năng lực đa phương thức được kiểm chứng tiếp ở Thử thách nâng cao.

---

## Thử thách nâng cao (100 trang đa phương thức)

- **Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:**
  - Tài liệu: **OpenAI — GPT-4 Technical Report** (`arXiv:2303.08774v6`), URL gốc: `https://arxiv.org/pdf/2303.08774.pdf`, quyền sử dụng học thuật công khai arXiv (bản local dùng tái lập không push file PDF nặng nhờ `.gitignore`).
  - Thông số xác thực tại [advanced/data/source_manifest.json](../../advanced/data/source_manifest.json): **`100` trang PDF vật lý**, dung lượng `5,245,564` bytes, SHA256 `c33a66dadca2388d7b172d6293b00dc32b71110c6f38fafe0d41112e61be7774`, tổng `283,983` ký tự văn bản trích xuất.

- **Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:**
  - Toàn bộ **`100/100` trang** được index bởi [advanced/scripts/build_multimodal_index.py](../../advanced/scripts/build_multimodal_index.py) vào Qdrant local (`advanced/data/qdrant_storage`, collection `day7_advanced_100p`, `openai/text-embedding-3-small`, `1536` chiều, `12` API calls, `97,780` tokens, chi phí `$0.001956`) với tổng cộng **`359` chunks** ([advanced/data/index_manifest.json](../../advanced/data/index_manifest.json)):
    - **`text` (`331` chunks):** phân đoạn theo trang và tọa độ bounding box, giữ nguyên tiêu đề mục (Main Paper vs System Card & Appendices).
    - **`table` (`9` chunks), `chart` (`13` chunks), `image_diagram` (`6` chunks):** render vùng tọa độ độ phân giải cao thành **`28` ảnh crop PNG (`200 DPI`)** tại `advanced/crops/`, trích xuất text PDF gốc kết hợp mô tả cấu trúc trực quan đã kiểm chứng (bảng Markdown, trục/chú giải/giá trị biểu đồ, sơ đồ kiến trúc RLHF & sơ đồ vật lý Ecole Polytechnique) lưu trong `advanced/data/advanced_chunks.json` và `advanced/data/.vision_cache.json`.
  - **Phạm vi và phần ít text / hạn chế:** `100%` trang có mặt trong chỉ mục; `3` trang phân cách / ảnh thuần ít ký tự (`p.40` với `79` chars, `p.79` với `8` chars, `p.98` với `78` chars — nơi chứa ma trận System Card Figure 11 đã được bù đắp hoàn chỉnh bằng chart crop `gpt4_p098_sc_fig11_chart.png`). Hạn chế về tọa độ crop (`gpt4_p008_fig05_chart.png` bị cắt hụt phần đỉnh trục X của Figure 5 trang 8) được ghi nhận rõ trong phần kiểm chứng thủ công bên dưới.

- **Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:**
  - Benchmark thiết kế tại [advanced/benchmark/questions.json](../../advanced/benchmark/questions.json) (bản sạch không lộ gold tại [cases.json](../../advanced/benchmark/cases.json)), gồm **`48` câu** chia tách hoàn toàn:
    - **`dev` (`18` câu):** `15` câu có đáp án + `3` câu thiếu bằng chứng. So sánh 2 cấu hình trên `dev` ([advanced/results/dev_summary.json](../../advanced/results/dev_summary.json)):
      - `text_only_dense` (`openai/gpt-4o-mini`, chỉ truy xuất `modality="text"`): đạt `13/15` câu có đáp án + `3/3` từ chối (`16/18`), từ chối nhầm 2 câu cần bảng/nhiều bảng (`adv_dev_t02`, `adv_dev_m03`), median `4546.12 ms`, `$0.00416`.
      - **`multimodal_hybrid_agent` (đã chọn):** đạt **`15/15` (`100%`)** câu có đáp án + **`3/3` (`100%`)** từ chối = **`18/18` (`100%`)**, median `5754.5 ms`, `$0.014339` (`79` tool calls trải đều cả `table: 36, text: 69, image_diagram: 4, chart: 8`).
    - **`test` (`30` câu):** `24` câu có đáp án (`table: 6`, `image_diagram: 5`, `chart: 5`, `multi_hop: 4`, `text: 4`) + `6` câu thiếu bằng chứng (`unanswerable: 6`).
  - **Cấu hình đã chốt chạy trên `test`:** `multimodal_hybrid_agent` trong [advanced/scripts/run_advanced_eval.py](../../advanced/scripts/run_advanced_eval.py) — lập kế hoạch đa công cụ 3 bước (Step 1a `multimodal_hybrid_search`, Step 1b phân rã sub-query cho tên `Table N`/`Figure N` với bộ lọc ranh giới từ `\b(table|figure)\s+\d+\b` và `+0.25` RRF boost, Step 2 `get_page_context` đồng hành cả bảng/biểu đồ và văn bản cùng trang, Step 3 `keyword_search` BM25) + ép kiểm tra citation & từ chối nghiêm ngặt. Khi chạy `test`, do quota trả phí OpenRouter đã gần cạn sau quá trình chạy trước đó, hệ thống sử dụng model miễn phí **`inclusionai/ling-3.2-flash:free`** (`max_tokens=2400`, `reasoning={"effort": "low"}`) để tránh cắt cụt output JSON khi model sinh token suy luận.

### Kết quả kiểm chứng thủ công trên tập `test` (30 câu đối chiếu trực tiếp PDF gốc)

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng (`table`) | `6 / 6` | `100.0%` | Đạt đủ Table 1, Table 2, Table 8, Table 10, Table 13 (`t01`–`t06`), trích đúng số liệu và độ lệch |
| Ảnh, sơ đồ (`image_diagram`) | `5 / 5` | `100.0%` | Đọc đúng sơ đồ bolometer nhiệt p.35 (`i01`, `i05`), 3 bước RLHF Fig. 2 p.37 (`i02`), comic Moar Layers (`i03`), adapter ELECOM p.9 (`i04`) |
| Biểu đồ (`chart`) | `5 / 5` | `100.0%` | Trích đúng biểu đồ tiêu thụ thịt p.34 (`c01`), Figure 5 MMLU (`c02`), Fig 9 + SC Fig 7 (`c03`), Fig 11 matrix p.98 (`c04`), TruthfulQA (`c05`) |
| Multi-hop (`multi_hop`) | `3 / 4` | `75.0%` | Đạt `m01`, `m03`, `m04`; lỗi **`adv_test_m02`** (tìm thấy Table 6/7 p.13 và Fig 9 p.14 nhưng thiếu con số exact `"82%"` trong văn bản p.62/p.13 nên ước lượng `~80%+ relative`) |
| Text (`text`) | `3 / 4` | `75.0%` | Đạt `x01`, `x03`, `x04`; lỗi **`adv_test_x02`** (từ chối nhầm do regex phạm vi trang `"61-63"` chỉ tách trang `61` và `"Figure 6"` kéo ưu tiên trang `10`, bỏ sót RBRM p.62–63) |
| Thiếu bằng chứng (`unanswerable`) | `6 / 6` | `100.0%` | Từ chối sạch cả 6 câu hỏi ngoài báo cáo (`u01` GPU/FLOPs bị ẩn, `u02` Putnam/IMO, `u03` Claude 2/Llama 2, `u04` năm 2020, `u05` thông số số bolometer, `u06` MT-Bench/BFCL) |
| **Toàn bộ** | **`28 / 30`** | **`93.33%`** | **Vượt ngưỡng `>= 90%` (không làm tròn); `0` lỗi provider/hệ thống** |

- **Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:**
  - Đường dẫn trace & audit đầy đủ 30 câu `test`: [advanced/results/test_traces_multimodal_hybrid_agent.json](../../advanced/results/test_traces_multimodal_hybrid_agent.json), [test_audit_reviews.json](../../advanced/results/test_audit_reviews.json), [test_summary.json](../../advanced/results/test_summary.json) (sinh bởi [advanced/scripts/audit_advanced_test.py](../../advanced/scripts/audit_advanced_test.py)).
  - Toàn bộ `30` câu đạt **`provider_errors: 0`**, tổng **`30` LLM calls**, **`148` tool calls** (phân bố `text: 261`, `chart: 74`, `table: 43`, `image_diagram: 14` lượt chunk được truy xuất), độ trễ trung vị **`44,432.69 ms`** (~`44.4 s/câu` do model `:free` có pha reasoning), tổng chi phí OpenRouter LLM+query: **`$0.00`** (embedding truy vấn tận dụng cache; chi phí dựng index 359 chunks ban đầu là `$0.001956`).
  - **Phát hiện quan trọng khi audit trực tiếp trên PDF gốc (`adv_test_c02`):** Khi mở nguyên trang 8 PDF để kiểm tra Figure 5 (*MMLU 3-shot accuracy across languages*), phát hiện vùng ảnh crop `advanced/crops/gpt4_p008_fig05_chart.png` (`y=195..526`) đã vô tình cắt hụt 4 cột ngôn ngữ cao nhất trên đầu biểu đồ (`Italian 84.1%`, `Afrikaans 84.1%`, `Spanish 84.0%`, `German 83.7%`, khiến đỉnh crop trông như bắt đầu từ `French 83.6%`). Tuy nhiên nhờ agent truy xuất kèm `gpt4_p008_text_00` chứa đầy đủ bảng/chữ của trang 8, câu trả lời cuối cùng vẫn đạt chính xác `Italian (84.1%), Afrikaans (84.1%), Spanish (84.0%)` và đáy `Marathi (66.7%), Telugu (62.0%)` — cho thấy giá trị thực tế của việc luôn ghép ngữ cảnh text trang đồng hành và audit bằng mắt trên PDF gốc.

- **Giới hạn, cách tái lập và đường dẫn benchmark:**
  - **Giới hạn còn tồn tại:**
    1. Regex bắt số trang (`pages?\s+(\d+)`) chưa phân tách phạm vi dạng hyphen `"page 61-63"` (gây mất trang 62–63 ở `adv_test_x02`).
    2. Câu hỏi nhiều nhảy (`adv_test_m02`) kéo về trang 13/14 nhưng chunk văn bản chứa đúng số liệu tổng hợp `"82%"` không được xếp đủ cao trong budget token.
    3. Độ trễ của model reasoning miễn phí `inclusionai/ling-3.2-flash:free` cao (`median ~44.4 s`) so với `gpt-4o-mini` trên `dev` (`median ~5.75 s`).
  - **Đường dẫn & lệnh tái lập đầy đủ:**
    - Dữ liệu & benchmark: [advanced/data/source_manifest.json](../../advanced/data/source_manifest.json), [index_manifest.json](../../advanced/data/index_manifest.json), [advanced/benchmark/questions.json](../../advanced/benchmark/questions.json), [cases.json](../../advanced/benchmark/cases.json).
    - Lệnh tái lập:
      ```bash
      python -m unittest discover -s tests -v
      python advanced/scripts/prepare_advanced_corpus.py
      python advanced/scripts/build_multimodal_index.py
      python advanced/scripts/build_advanced_benchmark.py
      python advanced/scripts/run_advanced_eval.py --split dev --config all
      python advanced/scripts/run_advanced_eval.py --split test --config multimodal_hybrid_agent --chat-model inclusionai/ling-3.2-flash:free
      python advanced/scripts/audit_advanced_test.py
      ```
