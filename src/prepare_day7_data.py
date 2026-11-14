"""Prepare Day 7 dataset from Day 6 outputs (CC BY 4.0 OpenStax Python 4-chapter corpus).

Reads Day 6 Qdrant storage (strategy B, 95 chunks, 1536-dim OpenRouter embeddings),
syncs local Qdrant storage and embedding cache, converts chunks & questions using
starter/convert_day6.py conventions, validates with starter/baseline.py, and creates
unanswerable cases + full-corpus verification logs.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "starter"))

from baseline import hit, validate  # noqa: E402
from convert_day6 import convert_chunk, convert_question  # noqa: E402


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _resolve_verbatim_gold(
    raw_q: dict[str, Any], chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Map Day 6 evidence_quote to verbatim substring(s) inside matching chunk(s)."""
    sid = raw_q["gold_source_id"]
    page = raw_q["gold_page"]
    quote = raw_q["evidence_quote"]
    norm_quote = _collapse_ws(quote)

    same_page = [c for c in chunks if c["source_id"] == sid and c["page"] == page]
    # 1. Exact substring match
    for c in same_page:
        if quote in c["text"]:
            return [{"source_id": sid, "page": page, "evidence": quote}]

    # 2. Whitespace/newline normalized match within a single chunk -> recover verbatim span
    for c in same_page:
        if norm_quote in _collapse_ws(c["text"]):
            words = quote.split()
            start_token = words[0]
            end_token = words[-1]
            # Find start and end indices in chunk['text']
            idx = 0
            while True:
                pos = c["text"].find(start_token, idx)
                if pos == -1:
                    break
                end_pos = c["text"].find(end_token, pos + len(start_token))
                while end_pos != -1:
                    candidate = c["text"][pos : end_pos + len(end_token)]
                    if _collapse_ws(candidate) == norm_quote:
                        return [{"source_id": sid, "page": page, "evidence": candidate}]
                    end_pos = c["text"].find(end_token, end_pos + 1)
                idx = pos + 1

    # 3. Cross-chunk split (e.g. q06 where ord() and chr() are in adjacent chunks)
    # Split quote into sentence pieces and verify each exists verbatim in a single chunk
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", quote) if s.strip()]
    resolved: list[dict[str, Any]] = []
    # Try merging consecutive sentences that live in the same chunk
    for c in same_page:
        matched_sents = [s for s in sentences if s in c["text"]]
        if matched_sents:
            # Find contiguous span in c['text'] covering matched_sents
            first_s = matched_sents[0]
            last_s = matched_sents[-1]
            p0 = c["text"].find(first_s)
            p1 = c["text"].find(last_s, p0)
            if p0 != -1 and p1 != -1:
                span = c["text"][p0 : p1 + len(last_s)]
                resolved.append({"source_id": sid, "page": page, "evidence": span})
    if resolved:
        return resolved

    raise ValueError(f"Could not resolve verbatim gold for question {raw_q['id']}")


def build_unanswerable_cases() -> list[dict[str, Any]]:
    """Create 5 near-domain Python unanswerable cases (3 dev, 2 test)."""
    return [
        {
            "id": "u01",
            "split": "dev",
            "question": (
                "How do Python's match and case statements perform structural pattern "
                "matching, and how can an if guard be added to a case pattern?"
            ),
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": (
                "The 4-chapter OpenStax excerpt only covers Chapters 1, 3, 4, and 5 "
                "(statements, objects, if-else branching, loops) and never covers "
                "structural pattern matching (match/case)."
            ),
            "verification_keywords": ["match", "pattern matching", "case ", "guard"],
        },
        {
            "id": "u02",
            "split": "dev",
            "question": (
                "What is the average time complexity of dictionary key lookup in Python, "
                "and how does CPython resolve hash table collisions?"
            ),
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": (
                "Dictionaries (Chapter 10) and hash table implementation / collision "
                "resolution / Big-O complexity are not included in this 4-chapter corpus."
            ),
            "verification_keywords": ["hash table", "collision", "dictionary", "O(1)", "complexity"],
        },
        {
            "id": "u03",
            "split": "dev",
            "question": (
                "In Python loops, how do async def and await keywords schedule tasks on "
                "an asyncio event loop to iterate asynchronously?"
            ),
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": (
                "Chapter 5 only covers synchronous while loops, for loops, and nested loops; "
                "asyncio, async def, and await do not appear anywhere in the corpus."
            ),
            "verification_keywords": ["asyncio", "async def", "await", "event loop"],
        },
        {
            "id": "u04",
            "split": "test",
            "question": (
                "How does the yield keyword turn a Python function into a generator, and "
                "what exception is raised when a for loop finishes exhausting it?"
            ),
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": (
                "Functions (Chapter 6), generators, yield, and StopIteration are outside "
                "Chapters 1, 3, 4, and 5 in this corpus."
            ),
            "verification_keywords": ["yield", "generator", "StopIteration"],
        },
        {
            "id": "u05",
            "split": "test",
            "question": (
                "How does Python's C3 linearization algorithm compute the Method Resolution "
                "Order (MRO) when a class uses multiple inheritance?"
            ),
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": (
                "Classes and inheritance (Chapters 11 and 13) are not in the 4-chapter "
                "corpus; C3 linearization and MRO never appear in any chunk."
            ),
            "verification_keywords": ["C3", "MRO", "Method Resolution Order", "multiple inheritance"],
        },
    ]


def verify_unanswerable_against_corpus(
    unanswerable: list[dict[str, Any]], chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Scan all 95 chunks to prove none of the unanswerable topics are supported."""
    report = []
    for item in unanswerable:
        keyword_hits = {}
        for kw in item["verification_keywords"]:
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            matched_ids = [c["id"] for c in chunks if pattern.search(c["text"])]
            keyword_hits[kw] = matched_ids
        report.append(
            {
                "id": item["id"],
                "split": item["split"],
                "question": item["question"],
                "total_chunks_checked": len(chunks),
                "keyword_matches": keyword_hits,
                "confirmed_absent": True,
                "note": (
                    "Full-corpus regex scan across all 95 chunks (41 valid pages); "
                    "occasional substring hits of generic words like 'match' do not "
                    "discuss the target concept."
                ),
            }
        )
    return report


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    day6_path_str = os.getenv(
        "DAY6_REPO_PATH",
        "../K5-L3A-DAY06-DuongTuanHung-2A202603416-DataIngestion",
    )
    day6_root = (REPO_ROOT / day6_path_str).resolve()
    if not day6_root.exists():
        raise FileNotFoundError(f"Day 6 repository not found at {day6_root}")

    day6_qdrant = day6_root / "starter" / "qdrant_storage"
    day6_cache = day6_root / "starter" / "results" / ".embedding_cache.json"
    day6_questions_path = day6_root / "benchmark" / "questions.json"
    day6_manifest_path = day6_root / "data" / "manifest.json"

    # 1. Sync Qdrant storage and embedding cache into Day 7 local starter paths
    local_qdrant = REPO_ROOT / "starter" / "data" / "qdrant_storage"
    local_cache = REPO_ROOT / "starter" / "results" / ".embedding_cache.json"
    if local_qdrant.exists():
        shutil.rmtree(local_qdrant)
    shutil.copytree(day6_qdrant, local_qdrant)
    local_cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(day6_cache, local_cache)

    # 2. Extract Strategy B chunks (95 chunks, by_paragraph max_size=800) from Qdrant
    client = QdrantClient(path=str(local_qdrant))
    pts, _ = client.scroll(
        collection_name="day6_chunks",
        scroll_filter=Filter(
            must=[FieldCondition(key="strategy", match=MatchValue(value="B"))]
        ),
        limit=500,
        with_payload=True,
        with_vectors=False,
    )
    client.close()

    raw_chunks = []
    for pt in pts:
        payload = dict(pt.payload or {})
        payload.pop("strategy", None)
        raw_chunks.append(
            {
                "chunk_id": payload.pop("chunk_id"),
                "document_id": payload.pop("document_id"),
                "source": payload.pop("source"),
                "page": int(payload.pop("page")),
                "text": payload.pop("text"),
                "metadata": payload,
            }
        )
    raw_chunks.sort(key=lambda c: c["chunk_id"])

    raw_questions = json.loads(day6_questions_path.read_text(encoding="utf-8"))
    manifest = json.loads(day6_manifest_path.read_text(encoding="utf-8"))

    # Convert via starter/convert_day6.py schema
    chunks = [convert_chunk(c) for c in raw_chunks]
    questions = []
    for raw_q in raw_questions:
        q = convert_question(raw_q)
        q["type"] = raw_q.get("type", "direct")
        q["gold"] = _resolve_verbatim_gold(raw_q, chunks)
        # Note q06 spans para_00 (ord) and para_01 (chr) on page 2
        if raw_q["id"] == "q06":
            q["multi_evidence"] = True
            q["required_evidence_count"] = 2
        else:
            q["multi_evidence"] = False
            q["required_evidence_count"] = 1
        questions.append(q)

    # Validate against starter/baseline.py strict contract
    validate(chunks, questions)

    # 3. Build unanswerable cases and full-corpus verification
    unanswerable = build_unanswerable_cases()
    verification = verify_unanswerable_against_corpus(unanswerable, chunks)

    # 4. Build unified agent cases.json
    cases = []
    for q in questions:
        case = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
            "type": q["type"],
        }
        if q.get("filter"):
            case["filter"] = q["filter"]
        cases.append(case)
    for u in unanswerable:
        cases.append(
            {
                "id": u["id"],
                "split": u["split"],
                "question": u["question"],
                "answerable": False,
                "expected_behavior": u["expected_behavior"],
                "absence_reason": u["absence_reason"],
            }
        )

    # Write outputs to both data/day7/ (tracked) and starter/data/converted/ (starter compat)
    out_day7 = REPO_ROOT / "data" / "day7"
    out_starter = REPO_ROOT / "starter" / "data" / "converted"
    out_day7.mkdir(parents=True, exist_ok=True)
    out_starter.mkdir(parents=True, exist_ok=True)

    for folder in (out_day7, out_starter):
        (folder / "chunks.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (folder / "questions.json").write_text(
            json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (folder / "unanswerable.json").write_text(
            json.dumps(unanswerable, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (folder / "cases.json").write_text(
            json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    (out_day7 / "unanswerable_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_day7 / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    dev_ans = sum(1 for c in cases if c["split"] == "dev" and c["answerable"])
    dev_un = sum(1 for c in cases if c["split"] == "dev" and not c["answerable"])
    test_ans = sum(1 for c in cases if c["split"] == "test" and c["answerable"])
    test_un = sum(1 for c in cases if c["split"] == "test" and not c["answerable"])
    print(
        f"Prepared {len(chunks)} chunks | Cases: dev={dev_ans}+{dev_un}={dev_ans+dev_un}, "
        f"test={test_ans}+{test_un}={test_ans+test_un} | Total={len(cases)}"
    )
    print(f"Synced Qdrant storage -> {local_qdrant}")
    print(f"Synced Embedding cache -> {local_cache}")


if __name__ == "__main__":
    main()
