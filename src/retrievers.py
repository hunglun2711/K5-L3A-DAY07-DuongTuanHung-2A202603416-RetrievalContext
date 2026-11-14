"""Retrieval methods and reranking/context strategies for Day 7.

Implements:
1. TfidfRetriever (wrapper around starter/baseline.py TfidfIndex)
2. DenseDay6Retriever (OpenRouter text-embedding-3-small + Day 6 QdrantVectorStore)
3. BM25Retriever (Okapi BM25 with code/symbol-aware tokenization)
4. HybridRRFRetriever (Dense Day 6 + BM25 combined via Reciprocal Rank Fusion)
5. RerankedHybridRetriever (top-k candidates from Hybrid RRF reranked by gpt-4o-mini or lexical-structural scorer)
6. Context packing utilities with budget, deduplication, and optional adjacent-chunk enrichment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "starter"))

from baseline import TfidfIndex, hit, pack_context  # noqa: E402

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
PERMITTED_FILTER_KEYS = frozenset(
    {"section", "chapter", "year", "document_type", "category", "language", "license"}
)


def tokenize_code_aware(text: str) -> list[str]:
    """Tokenize text while preserving Python identifiers, numbers/formats, and operators."""
    norm = unicodedata.normalize("NFC", text or "").casefold()
    # Keep words, format specs like 08.4f, operators like ==, !=, <=, >=, //, **, \\
    tokens = re.findall(r"[a-z0-9_]+(?:\.[a-z0-9_]+)?|==|!=|<=|>=|//|\*\*|\\\\|\\t|\\n|=", norm)
    return tokens


class OpenRouterEmbedder:
    """Real OpenRouter embedding client compatible with Day 6 index (1536-dim)."""

    def __init__(
        self,
        model_name: str | None = None,
        cache_path: Path | None = None,
        env_path: Path | None = None,
    ) -> None:
        load_dotenv(env_path or (REPO_ROOT / ".env"))
        self.model_name = (
            model_name or os.getenv("EMBEDDING_MODEL") or "openai/text-embedding-3-small"
        ).strip()
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing in .env")
        self.dimensions = 3072 if "large" in self.model_name else 1536
        self.cache_path = cache_path or (REPO_ROOT / "starter" / "results" / ".embedding_cache.json")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = self._load_cache()
        self.api_calls = 0
        self.cache_hits = 0
        self.billed_tokens = 0
        self.billed_cost_usd = 0.0

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_cache(self) -> None:
        self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}::{text}".encode("utf-8")).hexdigest()

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)
        missing_idx: list[int] = []
        missing_texts: list[str] = []
        for idx, text in enumerate(texts):
            entry = self._cache.get(self._cache_key(text))
            if entry and len(entry.get("embedding", [])) == self.dimensions:
                results[idx] = entry["embedding"]
                self.cache_hits += 1
            else:
                missing_idx.append(idx)
                missing_texts.append(text)

        for start in range(0, len(missing_texts), batch_size):
            batch = missing_texts[start : start + batch_size]
            batch_pos = missing_idx[start : start + batch_size]
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {"model": self.model_name, "input": batch}
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(OPENROUTER_EMBEDDINGS_URL, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"OpenRouter embeddings error HTTP {resp.status_code}: {resp.text[:300]}"
                )
            body = resp.json()
            items = sorted(body.get("data", []), key=lambda x: x["index"])
            if len(items) != len(batch):
                raise RuntimeError("Embedding count mismatch")
            usage = body.get("usage", {})
            tokens = int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
            cost = float(usage.get("cost") or (tokens * 0.02 / 1_000_000))
            self.api_calls += 1
            self.billed_tokens += tokens
            self.billed_cost_usd += cost
            per_tok = tokens / max(len(batch), 1)
            per_cost = cost / max(len(batch), 1)
            for local_i, item in enumerate(items):
                vec = item["embedding"]
                if len(vec) != self.dimensions:
                    raise RuntimeError(f"Dimension mismatch: {len(vec)} != {self.dimensions}")
                orig_i = batch_pos[local_i]
                self._cache[self._cache_key(texts[orig_i])] = {
                    "embedding": vec,
                    "approx_tokens": per_tok,
                    "approx_cost_usd": per_cost,
                }
                results[orig_i] = vec
            self._save_cache()

        return [v for v in results if v is not None]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

    def reset_counters(self) -> None:
        self.api_calls = 0
        self.cache_hits = 0
        self.billed_tokens = 0
        self.billed_cost_usd = 0.0


def _validate_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    if not filters:
        return {}
    clean = {}
    for k, v in filters.items():
        if k not in PERMITTED_FILTER_KEYS:
            raise ValueError(f"Unsupported filter key: {k}")
        if v is not None:
            clean[k] = v
    return clean


def _matches_filters(chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
    if not filters:
        return True
    meta = chunk.get("metadata", {})
    for k, v in filters.items():
        if meta.get(k) != v:
            return False
    return True


def infer_section_filter_from_query(query: str) -> dict[str, Any] | None:
    """Rule-based high-precision section inference from query terms ONLY (never uses gold)."""
    q = query.casefold()
    # Chapter 5: loops
    if any(k in q for k in ("while loop", "for loop", "nested loop", "range()", "step size", "counter variable")):
        return {"section": "ch05_loops"}
    # Chapter 4: branching / booleans
    if any(k in q for k in ("bool()", "if-else", "if statement", "pep 8", "logical operator", "boolean", "== ", "instead of ==")):
        return {"section": "ch04_branching"}
    # Chapter 3: objects / strings / unicode / f-string
    if any(k in q for k in ("ord()", "chr()", "f-string", "08.4f", "ascii", "unicode", "escape sequence", "container object", "id()", "type()")):
        return {"section": "ch03_data_types"}
    # Chapter 1: statements / print / input / variable naming
    if any(k in q for k in ("print()", "input(", "variable name", "string literal", "standard library", "pandas", "hello, world")):
        return {"section": "ch01_statements_variables"}
    return None


class TfidfRetriever:
    """Adapter around starter/baseline.py TfidfIndex with optional filter support."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.index = TfidfIndex(chunks)

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        clean = _validate_filters(filters)
        if not clean:
            rows = self.index.search(query, top_k=top_k)
        else:
            all_rows = self.index.search(query, top_k=len(self.chunks))
            rows = [r for r in all_rows if _matches_filters(r, clean)][:top_k]
        return [
            {**r, "rank": idx + 1, "strategy": "tfidf"}
            for idx, r in enumerate(rows[:top_k])
        ]


class DenseDay6Retriever:
    """Dense vector retrieval on the persisted Day 6 Qdrant storage."""

    def __init__(
        self,
        storage_path: Path | None = None,
        collection_name: str = "day6_chunks",
        qdrant_strategy: str = "B",
        embedder: OpenRouterEmbedder | None = None,
    ) -> None:
        self.storage_path = storage_path or (REPO_ROOT / "starter" / "data" / "qdrant_storage")
        self.collection_name = collection_name
        self.qdrant_strategy = qdrant_strategy
        self.embedder = embedder or OpenRouterEmbedder()
        self.client = QdrantClient(path=str(self.storage_path))

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        clean = _validate_filters(filters)
        qvec = self.embedder.embed_query(query)
        conditions = [FieldCondition(key="strategy", match=MatchValue(value=self.qdrant_strategy))]
        for key, val in clean.items():
            conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))

        resp = self.client.query_points(
            collection_name=self.collection_name,
            query=qvec,
            query_filter=Filter(must=conditions),
            limit=top_k,
            with_payload=True,
        )
        results: list[dict[str, Any]] = []
        for rank, pt in enumerate(resp.points, start=1):
            payload = dict(pt.payload or {})
            cid = payload.pop("chunk_id", str(pt.id))
            doc_id = payload.pop("document_id", "")
            src = payload.pop("source", "")
            page = int(payload.pop("page", 0))
            txt = payload.pop("text", "")
            payload.pop("strategy", None)
            results.append(
                {
                    "id": cid,
                    "source_id": doc_id,
                    "source": src,
                    "page": page,
                    "text": txt,
                    "metadata": payload,
                    "score": float(pt.score),
                    "rank": rank,
                    "strategy": "dense_day6",
                }
            )
        return results

    def close(self) -> None:
        self.client.close()


class BM25Retriever:
    """Okapi BM25 lexical retrieval with code/operator-aware tokenization."""

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize_code_aware(c["text"]) for c in chunks]
        self.doc_tf = [Counter(toks) for toks in self.doc_tokens]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        df = Counter(t for toks in self.doc_tf for t in toks)
        n_docs = len(chunks)
        self.idf = {
            t: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for t, freq in df.items()
        }

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        clean = _validate_filters(filters)
        q_tokens = tokenize_code_aware(query)
        scored: list[dict[str, Any]] = []
        for idx, chunk in enumerate(self.chunks):
            if not _matches_filters(chunk, clean):
                continue
            tf_map = self.doc_tf[idx]
            dl = self.doc_len[idx] or 1
            score = 0.0
            for qt in q_tokens:
                if qt not in tf_map:
                    continue
                tf = tf_map[qt]
                idf = self.idf.get(qt, 0.0)
                num = tf * (self.k1 + 1.0)
                den = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                score += idf * (num / den)
            if score > 0:
                scored.append({**chunk, "score": score, "strategy": "bm25"})
        scored.sort(key=lambda c: (-c["score"], c["id"]))
        return [{**r, "rank": i + 1} for i, r in enumerate(scored[:top_k])]


class HybridRRFRetriever:
    """Hybrid retriever fusing Dense Day 6 + BM25 via Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense: DenseDay6Retriever,
        bm25: BM25Retriever,
        rrf_k: int = 60,
        candidate_k: int = 12,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        dense_hits = self.dense.search(query, filters=filters, top_k=self.candidate_k)
        bm25_hits = self.bm25.search(query, filters=filters, top_k=self.candidate_k)

        by_id: dict[str, dict[str, Any]] = {}
        scores: dict[str, float] = {}
        sub_ranks: dict[str, dict[str, int | float | None]] = {}

        for rank, item in enumerate(dense_hits, start=1):
            cid = item["id"]
            by_id[cid] = item
            scores[cid] = scores.get(cid, 0.0) + self.dense_weight / (self.rrf_k + rank)
            sub_ranks.setdefault(cid, {})["dense_rank"] = rank
            sub_ranks[cid]["dense_score"] = round(item["score"], 6)

        for rank, item in enumerate(bm25_hits, start=1):
            cid = item["id"]
            if cid not in by_id:
                by_id[cid] = item
            scores[cid] = scores.get(cid, 0.0) + self.bm25_weight / (self.rrf_k + rank)
            sub_ranks.setdefault(cid, {})["bm25_rank"] = rank
            sub_ranks[cid]["bm25_score"] = round(item["score"], 6)

        fused = []
        for cid, rrf_score in scores.items():
            item = dict(by_id[cid])
            item["score"] = float(rrf_score)
            item["strategy"] = "hybrid_rrf"
            item["fusion_details"] = sub_ranks[cid]
            fused.append(item)

        fused.sort(key=lambda c: (-c["score"], c["id"]))
        return [{**r, "rank": i + 1} for i, r in enumerate(fused[:top_k])]


class LLMRerankedHybridRetriever:
    """Two-stage retrieval: Hybrid RRF candidate generation + OpenRouter LLM listwise reranking."""

    def __init__(
        self,
        hybrid: HybridRRFRetriever,
        rerank_model: str | None = None,
        candidate_k: int = 8,
        use_query_section_inference: bool = False,
        env_path: Path | None = None,
    ) -> None:
        load_dotenv(env_path or (REPO_ROOT / ".env"))
        self.hybrid = hybrid
        self.rerank_model = (rerank_model or os.getenv("MODEL") or "openai/gpt-4o-mini").strip()
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing for LLMRerankedHybridRetriever")
        self.candidate_k = candidate_k
        self.use_query_section_inference = use_query_section_inference
        self.rerank_calls = 0
        self.rerank_prompt_tokens = 0
        self.rerank_completion_tokens = 0
        self.rerank_cost_usd = 0.0
        self.last_inferred_filter: dict[str, Any] | None = None

    def reset_counters(self) -> None:
        self.rerank_calls = 0
        self.rerank_prompt_tokens = 0
        self.rerank_completion_tokens = 0
        self.rerank_cost_usd = 0.0

    def _llm_rerank(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if len(candidates) <= 1:
            return candidates
        passages = "\n\n".join(
            f"[{i}] (id={c['id']}, page={c['page']}, rrf={c['score']:.4f})\n{c['text'][:650]}"
            for i, c in enumerate(candidates)
        )
        prompt = (
            "You are a precision passage reranker for a technical Python textbook.\n"
            "Given a user question and candidate passages [0..N-1], rank the passages from MOST "
            "helpful (contains direct definitions, rules, syntax, or explanations answering ALL "
            "parts of the question) to LEAST helpful (e.g., short quiz questions without answers).\n"
            "Return ONLY a JSON object: {\"ranking\": [int, ...]} containing every candidate index once.\n\n"
            f"Question: {query}\n\nCandidates:\n{passages}"
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.rerank_model,
            "temperature": 0.0,
            "max_tokens": 180,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(OPENROUTER_CHAT_URL, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"OpenRouter rerank error HTTP {resp.status_code}: {resp.text[:300]}"
                )
            body = resp.json()
            usage = body.get("usage", {})
            self.rerank_calls += 1
            self.rerank_prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.rerank_completion_tokens += int(usage.get("completion_tokens") or 0)
            self.rerank_cost_usd += float(usage.get("cost") or 0.0)

            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            order = parsed.get("ranking", [])
        except Exception as exc:
            # Graceful fallback to RRF ranking if rerank hits HTTP 402 / transient error
            self.last_rerank_fallback = str(exc)[:200]
            fallback = []
            for rank_i, c in enumerate(candidates, start=1):
                item = dict(c)
                item["pre_rerank_rank"] = c.get("rank")
                item["pre_rerank_score"] = c.get("score")
                item["rank"] = rank_i
                item["score"] = round(1.0 / rank_i, 6)
                item["strategy"] = "hybrid_rrf_fallback"
                fallback.append(item)
            return fallback

        self.last_rerank_fallback = None
        seen = set()
        reordered: list[dict[str, Any]] = []
        for idx in order:
            if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                seen.add(idx)
                reordered.append(dict(candidates[idx]))
        for idx, c in enumerate(candidates):
            if idx not in seen:
                reordered.append(dict(c))
        for new_rank, c in enumerate(reordered, start=1):
            c["pre_rerank_rank"] = c.get("rank")
            c["pre_rerank_score"] = c.get("score")
            c["rank"] = new_rank
            # Normalize rerank score descending for trace readability
            c["score"] = round(1.0 / new_rank, 6)
            c["strategy"] = "hybrid_llm_rerank"
        return reordered

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        effective_filter = filters
        self.last_inferred_filter = None
        if effective_filter is None and self.use_query_section_inference:
            inferred = infer_section_filter_from_query(query)
            if inferred:
                effective_filter = inferred
                self.last_inferred_filter = inferred

        candidates = self.hybrid.search(query, filters=effective_filter, top_k=self.candidate_k)
        # Fallback if strict inferred filter yields too few hits
        if not candidates and self.last_inferred_filter is not None and filters is None:
            candidates = self.hybrid.search(query, filters=None, top_k=self.candidate_k)
        reranked = self._llm_rerank(query, candidates)
        return reranked[:top_k]


def pack_context_adaptive(
    results: list[dict[str, Any]],
    budget: int = 1800,
    all_chunks: list[dict[str, Any]] | None = None,
    include_next_adjacent: bool = False,
) -> tuple[str, list[str]]:
    """Pack deduplicated chunks within character budget; optionally append same-page next chunk."""
    blocks: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()

    expanded: list[dict[str, Any]] = []
    chunk_lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    if include_next_adjacent and all_chunks:
        for ch in all_chunks:
            idx = ch.get("metadata", {}).get("chunk_index")
            if isinstance(idx, int):
                chunk_lookup[(ch["source_id"], ch["page"], idx)] = ch

    for c in results:
        expanded.append(c)
        if include_next_adjacent and chunk_lookup:
            idx = c.get("metadata", {}).get("chunk_index")
            if isinstance(idx, int):
                nxt = chunk_lookup.get((c["source_id"], c["page"], idx + 1))
                if nxt:
                    expanded.append(nxt)

    for c in expanded:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        block = f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"
        candidate_text = "\n\n".join(blocks + [block])
        if len(candidate_text) <= budget:
            blocks.append(block)
            ids.append(c["id"])
    return "\n\n".join(blocks), ids
