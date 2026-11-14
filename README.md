# Day 7 — Agentic RAG (Retrieval & Context Engineering)

**Bài nộp cá nhân · Sinh viên: Dương Tuấn Hưng · MSSV: `2A202603416` · Lớp: `K5-L3A` · Python 3.11+ · tiếp nối Day 6**

> **Báo cáo chi tiết:** [report/individual/2A202603416_DuongTuanHung.md](report/individual/2A202603416_DuongTuanHung.md)  
> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

---

## Tóm tắt kết quả thực nghiệm (đã kiểm chứng)

- **Unit Tests (`tests/`):** **`23 / 23` passed** (`python -m unittest discover -s tests -v`), bao gồm kiểm tra unit retriever/packing, guard JSON/citation và kiểm tra hợp đồng dữ liệu/audit Core + Advanced (`tests/test_day7_pipeline.py`).
- **Phần Cơ bản (Core — OpenStax Python 4 chương, 41 trang nội dung hợp lệ, `95` chunks, Qdrant `day6_chunks` + `text-embedding-3-small` 1536-dim):**
  - **Retrieval (`dev` `n=18` -> `test` `n=6`):** Cấu hình chọn `improve2_hybrid_llm_rerank_adj2200` (Hybrid RRF + section filter + `gpt-4o-mini` listwise rerank + adjacent chunk + `budget=2200`) đạt **`16/18` Hit@1 / `17/18` Context Hit** trên `dev`, và **`6/6` (`100%`) Context Hit**, điểm audit Context **`2.0/2.0`** trên `test` (vượt `baseline_tfidf` ở mức `1/6 = 16.67%` và `0.5/2.0`).
  - **Multi-Tool Agent (`dev` `n=21` -> `test` `n=8`):** Cấu hình `chosen_multitool` (`hybrid_search`, `keyword_search`, `get_adjacent_chunks`, `max_searches=4`, `budget=2400`) đạt trung bình Grounded **`1.9444` (`17/18` điểm 2, `0` từ chối nhầm)** trên `dev` và **`2.0000` (`6/6` điểm 2, `9/9` citation hợp lệ, `2/2` abstention đúng, `0/6` từ chối nhầm, `0` provider errors)** trên `test`.
- **Thử thách Nâng cao (+15 điểm — *GPT-4 Technical Report*, `arXiv:2303.08774v6`, `100` trang PDF, `359` chunks đa phương thức `text/table/chart/image_diagram` + `28` PNG crops):**
  - **`dev` (`n=18`):** `multimodal_hybrid_agent` đạt **`15/15` có đáp án + `3/3` thiếu bằng chứng (`18/18`)**, vượt `text_only_dense` (`13/15` + `3/3`).
  - **`test` (`n=30`, khóa đề qua `cases.json`, chạy model `inclusionai/ling-3.2-flash:free`, audit trực tiếp từng câu trên PDF gốc):** đạt **`28 / 30` (`93.33%` >= `90%`)**, `0` lỗi provider: Bảng **`6/6`**, Ảnh/sơ đồ **`5/5`**, Biểu đồ **`5/5`**, Multi-hop **`3/4`**, Text **`3/4`**, Thiếu bằng chứng **`6/6`** ([advanced/results/test_summary.json](advanced/results/test_summary.json)).

---

## Cấu trúc thư mục

- `src/` — Mã nguồn pipeline Phần Cơ bản:
  - `src/prepare_day7_data.py` — Chuyển đổi dữ liệu từ Day 6 (`starter/data/qdrant_storage`), kiểm chứng `questions.json` & `unanswerable.json`.
  - `src/retrievers.py` — Cài đặt `TfidfRetriever`, `DenseDay6Retriever`, `BM25Retriever`, `HybridRRFRetriever`, `LLMRerankedHybridRetriever` và `pack_context_adaptive`.
  - `src/agent_system.py` — Agent đơn công cụ (`StarterLLMController`) và đa công cụ (`MultiToolAgentRunner` với 3 tools).
  - `src/run_retrieval_experiments.py`, `src/run_agent_experiments.py`, `src/audit_core_results.py` — Script chạy `dev`/`test` và chấm kiểm chứng Core.
- `data/day7/` — Bộ dữ liệu & benchmark Phần Cơ bản (`chunks.json`, `questions.json`, `unanswerable.json`, `cases.json`, `manifest.json`, `unanswerable_verification.json`).
- `results/core/` — Toàn bộ trace, tổng hợp và kết quả chấm tay cho `retrieval-dev`, `retrieval-test`, `agent-dev`, `agent-test` (kèm bản ghi sự cố [u05_http402_incident.json](results/core/agent-test/u05_http402_incident.json)).
- `advanced/` — Pipeline Thử thách Nâng cao 100 trang:
  - `advanced/scripts/` — `prepare_advanced_corpus.py`, `build_multimodal_index.py`, `build_advanced_benchmark.py`, `run_advanced_eval.py`, `audit_advanced_test.py`.
  - `advanced/crops/` — `28` ảnh crop độ phân giải cao các bảng, biểu đồ và sơ đồ trong PDF.
  - `advanced/benchmark/` — `questions.json` và `cases.json` (`18 dev + 30 test`).
  - `advanced/results/` — `dev_summary.json`, `test_summary.json`, `test_audit_reviews.json`, và traces đầy đủ.
- `report/individual/2A202603416_DuongTuanHung.md` — Báo cáo cá nhân hoàn chỉnh theo [mẫu báo cáo](REPORT_TEMPLATE.md).
- `tests/` — Suite kiểm thử (`test_day7_pipeline.py` cùng các test starter).

---

## Hướng dẫn cài đặt và chạy tái lập

### 1. Cấu hình môi trường

1. Tạo môi trường Python 3.11+ và cài thư viện cần thiết (ví dụ `qdrant-client`, `pymupdf`, `httpx`, `python-dotenv`).
2. Sao chép `.env.example` thành `.env` và điền `OPENROUTER_API_KEY` (không bao giờ commit `.env`):

```env
OPENROUTER_API_KEY=<your_openrouter_key>
EMBEDDING_MODEL=openai/text-embedding-3-small
PROVIDER=openrouter
MODEL=openai/gpt-4o-mini
DAY6_REPO_PATH=../K5-L3A-DAY06-DuongTuanHung-2A202603416-DataIngestion
```

### 2. Chạy Unit Tests

```bash
python -m unittest discover -s tests -v
```

### 3. Tái lập Phần Cơ bản (Core)

```bash
python src/prepare_day7_data.py
python src/run_retrieval_experiments.py --stage dev
python src/run_retrieval_experiments.py --stage test --chosen improve2_hybrid_llm_rerank_adj2200
python src/run_agent_experiments.py --stage dev
python src/run_agent_experiments.py --stage test
python src/audit_core_results.py
```

### 4. Tái lập Thử thách Nâng cao (100 trang GPT-4 Technical Report)

```bash
python advanced/scripts/prepare_advanced_corpus.py
python advanced/scripts/build_multimodal_index.py
python advanced/scripts/build_advanced_benchmark.py
python advanced/scripts/run_advanced_eval.py --split dev --config all
python advanced/scripts/run_advanced_eval.py --split test --config multimodal_hybrid_agent --chat-model inclusionai/ling-3.2-flash:free
python advanced/scripts/audit_advanced_test.py
```

*(Lưu ý: `.gitignore` đã loại trừ `.env`, `advanced/data/*.pdf`, `advanced/data/qdrant_storage/`, `advanced/data/.embedding_cache.json` và `starter/data/` để tránh đẩy dữ liệu nhị phân nặng và khóa bí mật lên Git; các script trên cho phép tái tạo đầy đủ index cục bộ.)*

---

## Yêu cầu đề bài tham chiếu

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

- Tài liệu đề bài gốc: [GUIDE.md](GUIDE.md), [BENCHMARK.md](BENCHMARK.md), [RUBRIC.md](RUBRIC.md), [RULES.md](RULES.md), [ADVANCED.md](ADVANCED.md), [SUBMISSION.md](SUBMISSION.md), [CHECKPOINTS.md](CHECKPOINTS.md), [starter/README.md](starter/README.md).
