"""End-to-end verification unit tests for Core Day 7 and Advanced 100-page Multimodal pipelines."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "advanced" / "scripts"))

from retrievers import (  # noqa: E402
    BM25Retriever,
    TfidfRetriever,
    infer_section_filter_from_query,
    pack_context_adaptive,
    tokenize_code_aware,
)
from run_advanced_eval import _parse_json_answer  # noqa: E402


class TestCoreDay7DatasetsAndMetrics(unittest.TestCase):
    def test_core_dataset_counts_and_splits(self) -> None:
        chunks = json.loads((REPO_ROOT / "data" / "day7" / "chunks.json").read_text(encoding="utf-8"))
        chunk_pairs = {(c["source_id"], c["page"]) for c in chunks}
        questions = json.loads((REPO_ROOT / "data" / "day7" / "questions.json").read_text(encoding="utf-8"))
        unans = json.loads((REPO_ROOT / "data" / "day7" / "unanswerable.json").read_text(encoding="utf-8"))
        cases = json.loads((REPO_ROOT / "data" / "day7" / "cases.json").read_text(encoding="utf-8"))

        self.assertGreaterEqual(len(chunks), 30)
        self.assertEqual(len(questions), 24)
        self.assertEqual(len(unans), 5)
        self.assertEqual(len(cases), 29)

        dev_q = [q for q in questions if q["split"] == "dev"]
        test_q = [q for q in questions if q["split"] == "test"]
        dev_u = [u for u in unans if u["split"] == "dev"]
        test_u = [u for u in unans if u["split"] == "test"]
        self.assertEqual(len(dev_q), 18)
        self.assertEqual(len(test_q), 6)
        self.assertEqual(len(dev_u), 3)
        self.assertEqual(len(test_u), 2)
        self.assertEqual({q["question"] for q in dev_q} & {q["question"] for q in test_q}, set())

        for q in questions:
            self.assertTrue(q["gold"])
            for g in q["gold"]:
                self.assertIn((g["source_id"], g["page"]), chunk_pairs)

    def test_core_retrieval_audit_and_agent_metrics(self) -> None:
        ret_test = json.loads(
            (REPO_ROOT / "results" / "core" / "retrieval-test" / "summary.json").read_text(encoding="utf-8")
        )
        by_cfg = {r["config"]: r for r in ret_test}
        self.assertEqual(by_cfg["improve2_hybrid_llm_rerank_adj2200"]["context_hit"], 6)
        self.assertEqual(by_cfg["improve2_hybrid_llm_rerank_adj2200"]["n"], 6)

        ctx_audit = json.loads(
            (REPO_ROOT / "results" / "core" / "retrieval-test" / "context_manual_audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(ctx_audit["summary"]["improve2_hybrid_llm_rerank_adj2200"]["mean_context_grade"], 2.0)

        agent_test = json.loads(
            (REPO_ROOT / "results" / "core" / "agent-test" / "metrics_chosen_multitool.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(agent_test["provider_errors"], 0)
        self.assertEqual(agent_test["grounded_answer"]["mean"], 2.0)
        self.assertEqual(agent_test["citation_validity"], {"n": 9, "N": 9})
        self.assertEqual(agent_test["abstention"], {"n": 2, "N": 2})
        self.assertEqual(agent_test["false_abstention"], {"n": 0, "N": 6})


class TestRetrieversAndPackingUnits(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_chunks = [
            {
                "id": "doc_p01_c00",
                "source_id": "book",
                "source": "book.pdf",
                "page": 1,
                "text": "The while loop repeats while condition holds. Counter variable tracks iterations.",
                "metadata": {"section": "ch05_loops", "chunk_index": 0},
            },
            {
                "id": "doc_p01_c01",
                "source_id": "book",
                "source": "book.pdf",
                "page": 1,
                "text": "Infinite loop occurs if counter is never updated; use == carefully.",
                "metadata": {"section": "ch05_loops", "chunk_index": 1},
            },
            {
                "id": "doc_p02_c00",
                "source_id": "book",
                "source": "book.pdf",
                "page": 2,
                "text": "PEP 8 recommends using 'is' instead of == when comparing to None in an if statement.",
                "metadata": {"section": "ch04_branching", "chunk_index": 0},
            },
        ]

    def test_code_aware_tokenization_and_section_inference(self) -> None:
        toks = tokenize_code_aware("f'{val:08.4f}' == True")
        self.assertIn("08.4f", toks)
        self.assertIn("==", toks)
        self.assertEqual(infer_section_filter_from_query("Explain while loop counter variable"), {"section": "ch05_loops"})
        self.assertEqual(infer_section_filter_from_query("What does PEP 8 say about if statement?"), {"section": "ch04_branching"})
        self.assertEqual(infer_section_filter_from_query("How does ord() relate to Unicode?"), {"section": "ch03_data_types"})
        self.assertEqual(infer_section_filter_from_query("How does print() work?"), {"section": "ch01_statements_variables"})
        self.assertIsNone(infer_section_filter_from_query("Explain general algorithm design"))

    def test_filter_validation_and_bm25_tfidf(self) -> None:
        bm25 = BM25Retriever(self.sample_chunks)
        tfidf = TfidfRetriever(self.sample_chunks)

        res_bm25 = bm25.search("while loop counter", filters={"section": "ch05_loops"}, top_k=3)
        self.assertTrue(res_bm25)
        self.assertTrue(all(r["metadata"]["section"] == "ch05_loops" for r in res_bm25))

        res_tfidf = tfidf.search("PEP 8 None", filters={"section": "ch04_branching"}, top_k=3)
        self.assertEqual(res_tfidf[0]["id"], "doc_p02_c00")

        with self.assertRaises(ValueError):
            bm25.search("test", filters={"unsupported_key": "x"})
        with self.assertRaises(ValueError):
            tfidf.search("test", filters={"bad": 1})

    def test_pack_context_adaptive_adjacent_and_dedup(self) -> None:
        hits = [self.sample_chunks[0], self.sample_chunks[0]]  # deliberate duplicate
        packed, ids = pack_context_adaptive(
            hits,
            budget=2000,
            all_chunks=self.sample_chunks,
            include_next_adjacent=True,
        )
        self.assertEqual(ids, ["doc_p01_c00", "doc_p01_c01"])
        self.assertIn("Infinite loop occurs", packed)

    def test_parse_json_answer_strict_guards(self) -> None:
        valid_ids = {"gpt4_p005_tab01_table"}
        xml_bad = _parse_json_answer("<tool_call><function=keyword_search></tool_call>", valid_ids)
        self.assertEqual(xml_bad["status"], "abstain")
        self.assertEqual(xml_bad["citations"], [])

        hallucinated_cit = _parse_json_answer(
            '{"status": "answered", "answer": "Score is 99% [fake_id].", "citations": ["fake_id"]}',
            valid_ids,
        )
        self.assertEqual(hallucinated_cit["status"], "abstain")
        self.assertEqual(hallucinated_cit["citations"], [])

        normalized = _parse_json_answer(
            '{"evidence_check": "Table 1 found", "status": "answered", "answer": "Passed 31/41 [chunk_id: gpt4_p005_tab01_table].", "citations": ["gpt4_p005_tab01_table"]}',
            valid_ids,
        )
        self.assertEqual(normalized["status"], "answered")
        self.assertEqual(normalized["citations"], ["gpt4_p005_tab01_table"])
        self.assertIn("[gpt4_p005_tab01_table]", normalized["answer"])


class TestAdvanced100PageContract(unittest.TestCase):
    def test_index_and_benchmark_contract(self) -> None:
        src = json.loads((REPO_ROOT / "advanced" / "data" / "source_manifest.json").read_text(encoding="utf-8"))
        idx = json.loads((REPO_ROOT / "advanced" / "data" / "index_manifest.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(src["page_count"], 100)
        self.assertEqual(idx["total_chunks"], idx["qdrant_stored_points"])
        self.assertEqual(idx["total_chunks"], 359)
        self.assertEqual(set(idx["chunks_by_modality"].keys()), {"text", "table", "chart", "image_diagram"})

        cases = json.loads((REPO_ROOT / "advanced" / "benchmark" / "cases.json").read_text(encoding="utf-8"))
        questions = json.loads((REPO_ROOT / "advanced" / "benchmark" / "questions.json").read_text(encoding="utf-8"))
        self.assertEqual(len(cases), 48)
        self.assertEqual(len(questions), 48)
        for c in cases:
            self.assertNotIn("expected_answer", c)
            self.assertNotIn("gold_evidence", c)

        dev_qs = [q for q in questions if q["split"] == "dev"]
        test_qs = [q for q in questions if q["split"] == "test"]
        self.assertEqual(len(dev_qs), 18)
        self.assertEqual(len(test_qs), 30)
        self.assertEqual({q["question"] for q in dev_qs} & {q["question"] for q in test_qs}, set())

        counts: dict[str, int] = {}
        for q in test_qs:
            counts[q["primary_group"]] = counts.get(q["primary_group"], 0) + 1
        self.assertGreaterEqual(counts["table"], 6)
        self.assertGreaterEqual(counts["image_diagram"], 5)
        self.assertGreaterEqual(counts["chart"], 5)
        self.assertGreaterEqual(counts["multi_hop"], 4)
        self.assertGreaterEqual(counts["text"], 4)
        self.assertGreaterEqual(counts["unanswerable"], 6)

    def test_advanced_audited_test_results_pass_threshold(self) -> None:
        summary = json.loads((REPO_ROOT / "advanced" / "results" / "test_summary.json").read_text(encoding="utf-8"))
        reviews = json.loads(
            (REPO_ROOT / "advanced" / "results" / "test_audit_reviews.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["total_cases"], 30)
        self.assertEqual(len(reviews), 30)
        self.assertEqual(summary["provider_errors"], 0)
        self.assertTrue(summary["meets_90pct_threshold"])
        self.assertGreaterEqual(summary["total_passed"], 27)
        self.assertTrue(all(r["citations_in_retrieved_and_corpus"] for r in reviews))


if __name__ == "__main__":
    unittest.main()
