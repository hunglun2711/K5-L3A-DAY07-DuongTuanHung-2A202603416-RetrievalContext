"""Run Day 7 retrieval experiments cleanly separated into dev tuning and held-out test.

Usage:
  python src/run_retrieval_experiments.py --stage dev
  python src/run_retrieval_experiments.py --stage test --chosen <config_name>
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "starter"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from baseline import hit  # noqa: E402
from retrievers import (  # noqa: E402
    BM25Retriever,
    DenseDay6Retriever,
    HybridRRFRetriever,
    LLMRerankedHybridRetriever,
    OpenRouterEmbedder,
    TfidfRetriever,
    infer_section_filter_from_query,
    pack_context_adaptive,
)


def compute_context_hit(
    question: dict[str, Any],
    context_ids: list[str],
    chunk_by_id: dict[str, dict[str, Any]],
) -> bool:
    """Check whether final packed context covers required evidence.

    For single-evidence questions, at least one gold alternative must be in context_ids.
    For multi-evidence questions (question['multi_evidence'] == True), ALL items in
    question['gold'] must be covered across chunks in context_ids.
    """
    ctx_chunks = [chunk_by_id[cid] for cid in context_ids if cid in chunk_by_id]
    if question.get("multi_evidence"):
        return all(any(hit(c, [g]) for c in ctx_chunks) for g in question["gold"])
    return any(hit(c, question["gold"]) for c in ctx_chunks)


def evaluate_config(
    name: str,
    description: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    chunk_by_id: dict[str, dict[str, Any]],
    all_chunks: list[dict[str, Any]],
    *,
    top_k: int = 3,
    budget: int = 800,
    use_filter: bool = False,
    infer_filter: bool = False,
    include_next_adjacent: bool = False,
    embedder: OpenRouterEmbedder | None = None,
    reranker: LLMRerankedHybridRetriever | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if embedder:
        embedder.reset_counters()
    if reranker:
        reranker.reset_counters()

    traces: list[dict[str, Any]] = []
    t_total_0 = time.perf_counter()

    for q in questions:
        applied_filter = None
        filter_source = "none"
        if use_filter:
            if q.get("filter"):
                applied_filter = q["filter"]
                filter_source = "user_provided"
            elif infer_filter:
                inferred = infer_section_filter_from_query(q["question"])
                if inferred:
                    applied_filter = inferred
                    filter_source = "query_rule_inferred"

        t0 = time.perf_counter()
        results = retriever.search(q["question"], filters=applied_filter, top_k=top_k)
        context, ids = pack_context_adaptive(
            results,
            budget=budget,
            all_chunks=all_chunks,
            include_next_adjacent=include_next_adjacent,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        marked = [{**r, "hit": hit(r, q["gold"])} for r in results[:3]]
        hit1 = bool(marked and marked[0]["hit"])
        hit3 = any(r["hit"] for r in marked)
        ctx_hit = compute_context_hit(q, ids, chunk_by_id)

        traces.append(
            {
                "config": name,
                "query_id": q["id"],
                "split": q["split"],
                "type": q.get("type", "direct"),
                "multi_evidence": bool(q.get("multi_evidence", False)),
                "question": q["question"],
                "applied_filter": applied_filter,
                "filter_source": filter_source,
                "top_k": top_k,
                "budget_chars": budget,
                "include_next_adjacent": include_next_adjacent,
                "results": marked,
                "context": context,
                "context_ids": ids,
                "context_chars": len(context),
                "elapsed_ms": round(elapsed_ms, 3),
                "hit1": hit1,
                "hit3": hit3,
                "context_hit": ctx_hit,
            }
        )

    total_sec = round(time.perf_counter() - t_total_0, 3)
    n = len(traces)
    h1 = sum(t["hit1"] for t in traces)
    h3 = sum(t["hit3"] for t in traces)
    ch = sum(t["context_hit"] for t in traces)

    embed_cost = embedder.billed_cost_usd if embedder else 0.0
    embed_toks = embedder.billed_tokens if embedder else 0
    rerank_cost = reranker.rerank_cost_usd if reranker else 0.0
    rerank_ptok = reranker.rerank_prompt_tokens if reranker else 0
    rerank_ctok = reranker.rerank_completion_tokens if reranker else 0

    summary = {
        "config": name,
        "description": description,
        "split": questions[0]["split"] if questions else "",
        "n": n,
        "top_k": top_k,
        "budget_chars": budget,
        "use_filter": use_filter,
        "infer_filter": infer_filter,
        "include_next_adjacent": include_next_adjacent,
        "hit1": h1,
        "hit1_str": f"{h1}/{n}",
        "hit1_rate": round(h1 / n, 4) if n else 0.0,
        "hit3": h3,
        "hit3_str": f"{h3}/{n}",
        "hit3_rate": round(h3 / n, 4) if n else 0.0,
        "context_hit": ch,
        "context_hit_str": f"{ch}/{n}",
        "context_hit_rate": round(ch / n, 4) if n else 0.0,
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 2),
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 3),
        "total_seconds": total_sec,
        "embed_api_calls": embedder.api_calls if embedder else 0,
        "embed_cache_hits": embedder.cache_hits if embedder else 0,
        "embed_billed_tokens": embed_toks,
        "rerank_calls": reranker.rerank_calls if reranker else 0,
        "rerank_prompt_tokens": rerank_ptok,
        "rerank_completion_tokens": rerank_ctok,
        "service_cost_usd": round(embed_cost + rerank_cost, 8),
    }
    return summary, traces


def build_configs(
    chunks: list[dict[str, Any]],
    embedder: OpenRouterEmbedder,
    dense: DenseDay6Retriever,
    bm25: BM25Retriever,
    hybrid: HybridRRFRetriever,
    reranker: LLMRerankedHybridRetriever,
) -> dict[str, dict[str, Any]]:
    tfidf = TfidfRetriever(chunks)
    return {
        "baseline_tfidf": {
            "description": "Lexical TF-IDF baseline (starter/baseline.py), top_k=3, no filter, budget=800",
            "retriever": tfidf,
            "top_k": 3,
            "budget": 800,
            "use_filter": False,
            "infer_filter": False,
            "include_next_adjacent": False,
            "embedder": None,
            "reranker": None,
        },
        "method2_dense_day6": {
            "description": "Dense retrieval on Day 6 Qdrant (text-embedding-3-small, Cosine), top_k=3, no filter, budget=800",
            "retriever": dense,
            "top_k": 3,
            "budget": 800,
            "use_filter": False,
            "infer_filter": False,
            "include_next_adjacent": False,
            "embedder": embedder,
            "reranker": None,
        },
        "method3_bm25": {
            "description": "Okapi BM25 lexical retrieval with code-aware tokenizer, top_k=3, no filter, budget=800",
            "retriever": bm25,
            "top_k": 3,
            "budget": 800,
            "use_filter": False,
            "infer_filter": False,
            "include_next_adjacent": False,
            "embedder": None,
            "reranker": None,
        },
        "method4_hybrid_rrf": {
            "description": "Hybrid RRF (Dense Day 6 + BM25, k=60, cand=12), top_k=3, no filter, budget=800",
            "retriever": hybrid,
            "top_k": 3,
            "budget": 800,
            "use_filter": False,
            "infer_filter": False,
            "include_next_adjacent": False,
            "embedder": embedder,
            "reranker": None,
        },
        "improve1_hybrid_filter_budget1800": {
            "description": "Improvement 1: Hybrid RRF + user/query-inferred section filter + budget=1800 chars",
            "retriever": hybrid,
            "top_k": 3,
            "budget": 1800,
            "use_filter": True,
            "infer_filter": True,
            "include_next_adjacent": False,
            "embedder": embedder,
            "reranker": None,
        },
        "improve2_hybrid_llm_rerank_adj2200": {
            "description": "Improvement 2: Hybrid RRF top-8 + gpt-4o-mini listwise rerank + section filter + adjacent chunk + budget=2200",
            "retriever": reranker,
            "top_k": 3,
            "budget": 2200,
            "use_filter": True,
            "infer_filter": True,
            "include_next_adjacent": True,
            "embedder": embedder,
            "reranker": reranker,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 7 retrieval benchmarks")
    parser.add_argument("--stage", choices=("dev", "test"), required=True)
    parser.add_argument("--chosen", type=str, default=None, help="Chosen config name for --stage test")
    args = parser.parse_args()

    chunks = json.loads((REPO_ROOT / "data" / "day7" / "chunks.json").read_text(encoding="utf-8"))
    questions = json.loads((REPO_ROOT / "data" / "day7" / "questions.json").read_text(encoding="utf-8"))
    chunk_by_id = {c["id"]: c for c in chunks}
    selected = [q for q in questions if q["split"] == args.stage]

    embedder = OpenRouterEmbedder()
    dense = DenseDay6Retriever(embedder=embedder)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRRFRetriever(dense=dense, bm25=bm25, rrf_k=60, candidate_k=12)
    reranker = LLMRerankedHybridRetriever(hybrid=hybrid, candidate_k=8)

    configs = build_configs(chunks, embedder, dense, bm25, hybrid, reranker)

    out_dir = REPO_ROOT / "results" / "core" / f"retrieval-{args.stage}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "dev":
        run_names = list(configs.keys())
    else:
        if not args.chosen or args.chosen not in configs:
            raise ValueError(f"Must specify valid --chosen from {list(configs.keys())}")
        run_names = ["baseline_tfidf", args.chosen]

    summaries = []
    for cname in run_names:
        cfg = configs[cname]
        summary, traces = evaluate_config(
            cname,
            cfg["description"],
            cfg["retriever"],
            selected,
            chunk_by_id,
            chunks,
            top_k=cfg["top_k"],
            budget=cfg["budget"],
            use_filter=cfg["use_filter"],
            infer_filter=cfg["infer_filter"],
            include_next_adjacent=cfg["include_next_adjacent"],
            embedder=cfg["embedder"],
            reranker=cfg["reranker"],
        )
        summaries.append(summary)
        (out_dir / f"traces_{cname}.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[{args.stage}] {cname:36s} | Hit@1={summary['hit1_str']} ({summary['hit1_rate']:.1%}) | "
            f"Hit@3={summary['hit3_str']} ({summary['hit3_rate']:.1%}) | "
            f"Ctx={summary['context_hit_str']} ({summary['context_hit_rate']:.1%}) | "
            f"AvgChars={summary['avg_context_chars']:.0f} | med={summary['median_ms']:.1f}ms | cost=${summary['service_cost_usd']:.6f}"
        )

    (out_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dense.close()


if __name__ == "__main__":
    main()
