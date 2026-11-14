"""Run real Agentic RAG experiments on dev and test splits (OpenRouter gpt-4o-mini + Qdrant Day 6).

Usage:
  python src/run_agent_experiments.py --stage dev
  python src/run_agent_experiments.py --stage test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "starter"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic import run_agent  # noqa: E402
from agent_system import MultiToolAgentRunner, StarterLLMController  # noqa: E402
from retrievers import (  # noqa: E402
    BM25Retriever,
    DenseDay6Retriever,
    HybridRRFRetriever,
    LLMRerankedHybridRetriever,
    OpenRouterEmbedder,
    TfidfRetriever,
)


def run_starter_loop_config(
    config_name: str,
    cases: list[dict[str, Any]],
    retriever: Any,
    *,
    embedder: OpenRouterEmbedder | None = None,
    max_searches: int = 2,
    top_k: int = 3,
    budget: int = 800,
    pass_case_filter: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    controller = StarterLLMController()
    traces: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    for case in cases:
        controller.reset_counters()
        if embedder:
            embedder.reset_counters()
        t0 = time.perf_counter()
        eff_filter = case.get("filter") if pass_case_filter else None
        try:
            res = run_agent(
                case["question"],
                controller,
                retriever,
                filters=eff_filter,
                max_searches=max_searches,
                top_k=top_k,
                budget=budget,
            )
            num_searches = sum(1 for e in res["events"] if e.get("event") == "search")
            last_search = next(
                (e for e in reversed(res["events"]) if e.get("event") == "search"), {}
            )
            trace = {
                "query_id": case["id"],
                "split": case["split"],
                "answerable": case["answerable"],
                "question": case["question"],
                "status": res["status"],
                "answer": res["answer"],
                "citations": res["citations"],
                "final_context": last_search.get("context", ""),
                "final_context_ids": last_search.get("context_ids", []),
                "events": [
                    {**e, "tool_name": e.get("tool_name", retriever.__class__.__name__)}
                    if e.get("event") == "search"
                    else e
                    for e in res["events"]
                ],
                "num_searches": num_searches,
                "model_calls": controller.model_calls,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "prompt_tokens": controller.prompt_tokens,
                "completion_tokens": controller.completion_tokens,
                "embed_tokens": embedder.billed_tokens if embedder else 0,
                "service_cost_usd": round(
                    controller.cost_usd + (embedder.billed_cost_usd if embedder else 0.0), 8
                ),
            }
        except Exception as exc:
            trace = {
                "query_id": case["id"],
                "split": case["split"],
                "answerable": case["answerable"],
                "question": case["question"],
                "status": "error",
                "error": str(exc),
                "answer": "",
                "citations": [],
                "final_context": "",
                "final_context_ids": [],
                "events": [],
                "num_searches": 0,
                "model_calls": controller.model_calls,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "prompt_tokens": controller.prompt_tokens,
                "completion_tokens": controller.completion_tokens,
                "embed_tokens": embedder.billed_tokens if embedder else 0,
                "service_cost_usd": round(
                    controller.cost_usd + (embedder.billed_cost_usd if embedder else 0.0), 8
                ),
            }
        traces.append(trace)
        print(
            f"  [{config_name}] {case['id']} ({case['split']} ans={case['answerable']}): "
            f"status={trace['status']} searches={trace['num_searches']} cits={trace['citations']} "
            f"ms={trace['elapsed_ms']}"
        )

    summary = {
        "config": config_name,
        "split": cases[0]["split"] if cases else "",
        "total_cases": len(traces),
        "answered": sum(1 for t in traces if t["status"] == "answered"),
        "abstained": sum(1 for t in traces if t["status"] == "abstain"),
        "errors": sum(1 for t in traces if t["status"] == "error"),
        "total_searches": sum(t["num_searches"] for t in traces),
        "total_model_calls": sum(t["model_calls"] for t in traces),
        "total_prompt_tokens": sum(t["prompt_tokens"] for t in traces),
        "total_completion_tokens": sum(t["completion_tokens"] for t in traces),
        "total_seconds": round(time.perf_counter() - t_all, 2),
        "total_cost_usd": round(sum(t["service_cost_usd"] for t in traces), 8),
    }
    return traces, summary


def run_multitool_config(
    config_name: str,
    cases: list[dict[str, Any]],
    runner: MultiToolAgentRunner,
    *,
    max_searches: int = 4,
    top_k: int = 3,
    budget: int = 2400,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    for case in cases:
        try:
            trace = runner.run_case(
                case, max_searches=max_searches, top_k=top_k, budget=budget
            )
        except Exception as exc:
            trace = {
                "query_id": case["id"],
                "split": case["split"],
                "answerable": case["answerable"],
                "question": case["question"],
                "status": "error",
                "error": str(exc),
                "answer": "",
                "citations": [],
                "final_context": "",
                "final_context_ids": [],
                "events": [],
                "num_searches": 0,
                "model_calls": 0,
                "elapsed_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "embed_tokens": 0,
                "service_cost_usd": 0.0,
            }
        traces.append(trace)
        tools_used = [e.get("tool_name") for e in trace["events"] if e.get("event") == "search"]
        print(
            f"  [{config_name}] {case['id']} ({case['split']} ans={case['answerable']}): "
            f"status={trace['status']} tools={tools_used} cits={trace['citations']} "
            f"ms={trace['elapsed_ms']}"
        )

    summary = {
        "config": config_name,
        "split": cases[0]["split"] if cases else "",
        "total_cases": len(traces),
        "answered": sum(1 for t in traces if t["status"] == "answered"),
        "abstained": sum(1 for t in traces if t["status"] == "abstain"),
        "errors": sum(1 for t in traces if t["status"] == "error"),
        "total_searches": sum(t["num_searches"] for t in traces),
        "total_model_calls": sum(t["model_calls"] for t in traces),
        "total_prompt_tokens": sum(t["prompt_tokens"] for t in traces),
        "total_completion_tokens": sum(t["completion_tokens"] for t in traces),
        "total_seconds": round(time.perf_counter() - t_all, 2),
        "total_cost_usd": round(sum(t["service_cost_usd"] for t in traces), 8),
    }
    return traces, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real Agentic RAG experiments")
    parser.add_argument("--stage", choices=("dev", "test"), required=True)
    args = parser.parse_args()

    chunks = json.loads((REPO_ROOT / "data" / "day7" / "chunks.json").read_text(encoding="utf-8"))
    cases = json.loads((REPO_ROOT / "data" / "day7" / "cases.json").read_text(encoding="utf-8"))
    selected = [c for c in cases if c["split"] == args.stage]

    embedder = OpenRouterEmbedder()
    dense = DenseDay6Retriever(embedder=embedder)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRRFRetriever(dense=dense, bm25=bm25, rrf_k=60, candidate_k=12)
    reranker = LLMRerankedHybridRetriever(
        hybrid=hybrid, candidate_k=8, use_query_section_inference=True
    )
    tfidf = TfidfRetriever(chunks)
    multitool_runner = MultiToolAgentRunner(
        chunks=chunks,
        hybrid_retriever=reranker,
        bm25_retriever=bm25,
        embedder=embedder,
        reranker=reranker,
    )

    out_root = REPO_ROOT / "results" / "core" / f"agent-{args.stage}"
    out_root.mkdir(parents=True, exist_ok=True)

    run_summaries: list[dict[str, Any]] = []

    print(f"=== Running Agent Baseline (TF-IDF, max_searches=2, budget=800) on {args.stage} ===")
    b_traces, b_sum = run_starter_loop_config(
        "agent_baseline_tfidf",
        selected,
        tfidf,
        embedder=None,
        max_searches=2,
        top_k=3,
        budget=800,
        pass_case_filter=False,
    )
    (out_root / "traces_baseline_tfidf.json").write_text(
        json.dumps(b_traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    run_summaries.append(b_sum)

    if args.stage == "dev":
        print(f"=== Running Agent Dense Day 6 (max_searches=3, budget=1800) on {args.stage} ===")
        d_traces, d_sum = run_starter_loop_config(
            "agent_dense_day6",
            selected,
            dense,
            embedder=embedder,
            max_searches=3,
            top_k=3,
            budget=1800,
            pass_case_filter=True,
        )
        (out_root / "traces_dense_day6.json").write_text(
            json.dumps(d_traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        run_summaries.append(d_sum)

    print(
        f"=== Running Agent Chosen (Multi-Tool Hybrid+Rerank+Adj, max_searches=4, budget=2400) on {args.stage} ==="
    )
    c_traces, c_sum = run_multitool_config(
        "agent_chosen_multitool",
        selected,
        multitool_runner,
        max_searches=4,
        top_k=3,
        budget=2400,
    )
    (out_root / "traces_chosen_multitool.json").write_text(
        json.dumps(c_traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    run_summaries.append(c_sum)

    (out_root / "run_summary.json").write_text(
        json.dumps(run_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dense.close()
    print(json.dumps(run_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
