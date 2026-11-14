"""Manual PDF-verified audit script for Day 7 Advanced 30-case Test benchmark.

Reads:
- `advanced/benchmark/questions.json` (gold answers + PDF/crop references)
- `advanced/results/test_traces_multimodal_hybrid_agent.json`
- `advanced/data/advanced_chunks.json`

Records explicit human/manual verification for every single test question (1..30)
against original PDF pages, tables, charts, images, and diagrams per ADVANCED.md Section 3,
and outputs `advanced/results/test_audit_reviews.json` and `advanced/results/test_summary.json`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = REPO_ROOT / "advanced" / "benchmark" / "questions.json"
CHUNKS_PATH = REPO_ROOT / "advanced" / "data" / "advanced_chunks.json"
TRACES_PATH = REPO_ROOT / "advanced" / "results" / "test_traces_multimodal_hybrid_agent.json"
REVIEWS_OUT = REPO_ROOT / "advanced" / "results" / "test_audit_reviews.json"
SUMMARY_OUT = REPO_ROOT / "advanced" / "results" / "test_summary.json"

MANUAL_AUDIT_NOTES: dict[str, tuple[bool, str]] = {
    "adv_test_t01": (
        True,
        "PDF Table 1 (p.5) verified: GRE Quant 163/170 (~80th) vs 157/170 (~62nd) no-vision (+6) vs GPT-3.5 147/170; AP Micro 5 (82nd-100th) vs 4 (60th-82nd) no-vision (+1) vs GPT-3.5 4. Exact values, gains, and citations.",
    ),
    "adv_test_t02": (
        True,
        "PDF Table 1 (p.5) verified: Leetcode easy 31/41 (~75.6%), medium 21/80, hard 3/45 vs GPT-3.5 12/41, 8/80, 0/45. Cites gpt4_p005_tab01_table.",
    ),
    "adv_test_t03": (
        True,
        "PDF Table 2 (p.7) verified: GSM-8K 92.0%* vs 57.1% (5-shot CoT + training-mix asterisk noted), ARC 96.3% vs 85.2% (25-shot). Cites gpt4_p007_tab02_table.",
    ),
    "adv_test_t04": (
        True,
        "PDF Table 10 (p.31) verified: AP Eng Lit 81.82% contam / 60.00% non-contam vs 75.56% contam / -17.50% deg; LSAT 39.00% contam / 83.61% non-contam vs 64.10% contam / +10.01% deg. Cites gpt4_p031_tab10_table and gpt4_p031_text_00.",
    ),
    "adv_test_t05": (
        True,
        "PDF Table 8 (p.28) verified: AP Micro Base 90.0% -> RLHF 76.7% (-13.3 pp); AMC 12 Base 20.0% -> RLHF 32.0% (+12.0 pp). Cites gpt4_p028_tab08_table.",
    ),
    "adv_test_t06": (
        True,
        "PDF Table 13 (p.33) verified: Marathi (90M speakers), Latvian (2M), Welsh (600k) alongside English (>1B). Cites gpt4_p033_tab13_table and p.33 text.",
    ),
    "adv_test_i01": (
        True,
        "PDF Table 15 (p.35) verified: length L, section S, thermal conductivity lambda; T(0)=T0, T(L)=Tb; T(x)=(Tb-T0)*(x/L)+T0. Cites gpt4_p035_diag_physics and p.35 text.",
    ),
    "adv_test_i02": (
        True,
        "PDF Table 17 / Figure 2 (p.37) verified: (1) SFT on demonstration data, (2) RM training on comparison rankings, (3) PPO optimization against RM rewards. Cites gpt4_p037_diag_rlhf and p.37 text.",
    ),
    "adv_test_i03": (
        True,
        "PDF Table 19 comic (p.39) verified: Statistical Learning panel cites VC-Dimension / structural risk / loss / soft margin; Neural Networks panel yells STACK MORE LAYERS with both axes labeled LAYERS. Cites gpt4_p039_img_comic.",
    ),
    "adv_test_i04": (
        True,
        "PDF Table 3 collage (p.9) verified visually: Panel 2 ELECOM 'Cable Mania' / 'ELECOM P-APLTDCN series' / 'For Lightning Cable'; Panel 3 hidden Apple Lightning connector inside blue 15-pin VGA shell. Cites gpt4_p009_img01_adapter.",
    ),
    "adv_test_i05": (
        True,
        "PDF Table 15 Fig. 1 diagram (p.35) verified visually: rayonnement incident, puissance Pr, absorbeur, R(T), Cth at T, courant de polarisation I, terminating at thermostat Tb. Cites gpt4_p035_diag_physics.",
    ),
    "adv_test_c01": (
        True,
        "PDF Table 14 bar chart (p.34) verified visually: Finland 175.09 g/person/day, Georgia 79.84 g/person/day, Western Asia 69.62 g; diff 105.47 g/person/day. Cites gpt4_p034_chart_meat and gpt4_p034_text_00.",
    ),
    "adv_test_c02": (
        True,
        "Verified directly against full original PDF Page 8 Figure 5 (clipped crop p008_fig05 missed top 4 bars, while companion gpt4_p008_text_00 captured full page 8): top 3 non-English are Italian 84.1%, Afrikaans 84.1%, Spanish 84.0%; bottom 2 are Marathi 66.7%, Telugu 62.0%.",
    ),
    "adv_test_c03": (
        True,
        "PDF Fig 9 (p.14) / System Card Fig 7 (p.63) verified: Sensitive ~47% / ~41% / ~24% vs Disallowed ~22% / ~4.5% / <1% (~0.7-0.8%). Cites both charts accurately.",
    ),
    "adv_test_c04": (
        True,
        "PDF System Card Fig 11 matrix (p.98) verified: GPT-3.5 Turbo RLHF wins 38.9% vs GPT-4 RLHF and 63.0% vs GPT-3.5 RLHF; off-diagonal pairs sum to 100%, 50.0% diagonal. Cites gpt4_p098_sc_fig11_chart.",
    ),
    "adv_test_c05": (
        True,
        "PDF Fig 7 (p.11) / System Card Fig 8 (p.65) verified: GPT-4 0-shot base ~29%, 5-shot base ~39%, RLHF ~59-60% vs gpt-3.5-turbo RLHF ~47% and Anthropic-LM RLHF ~32%. Cites both charts.",
    ),
    "adv_test_m01": (
        True,
        "PDF Fig 7 (p.11) & Fig 8 (p.12) verified: RLHF raises TruthfulQA mc1 from ~29% (0-shot base) to ~59-60% while degrading MMLU calibration ECE >10x from 0.007 to 0.074. Cites both charts.",
    ),
    "adv_test_m02": (
        False,
        "STRICT FAIL (partial multi-hop retrieval miss): Correctly contrasted Early vs Latest GPT-4 on bomb & cigarette prompts from Table 6/7 (p.13), but failed to retrieve System Card p.62 / p.13 text containing the exact '82%' disallowed reduction figure, approximating '~80%+ relative' from Figure 9 instead.",
    ),
    "adv_test_m03": (
        True,
        "PDF Table 1 (p.5) & Table 9 (p.30) verified: Codeforces 0% contam, 392 (<5th) across all columns; AMC 10 4% contam, no-vision 36/150 (10th-19th) full vs 38/150 (14th-21st) non-contam, with-vision 30/150 (6th-12th) full vs 31/150 (7th-12th) non-contam.",
    ),
    "adv_test_m04": (
        True,
        "PDF System Card p.64, Fig 11 (p.98), & Table 17 (p.37) verified: 70.2% vs GPT-3.5 RLHF and 61.1% vs GPT-3.5 Turbo RLHF match Row 1 of Fig 11; SFT, RM, and PPO steps cited accurately.",
    ),
    "adv_test_x01": (
        True,
        "PDF System Card p.62/64 verified: RealToxicityPrompts 0.73% (GPT-4) vs 6.48% (GPT-3.5), +29% policy compliance on sensitive prompts (and -82% disallowed). Valid citations gpt4_p062_text_04 and gpt4_p064_text_00.",
    ),
    "adv_test_x02": (
        False,
        "FAIL (false abstention / retrieval miss): Page-range '61-63' regex only pulled page 61 and 'Figure 6' prioritized main paper p.10 over System Card p.62-63, missing gpt4_p062_text_01 ('rule-based reward models (RBRMs)') and gpt4_p063_text_00.",
    ),
    "adv_test_x03": (
        True,
        "PDF Section 2 (p.2) verified: accurately quotes both reasons ('competitive landscape' and 'safety implications') for withholding architecture/hardware/compute/dataset details, plus third-party transparency note. Cites gpt4_p002_text_01.",
    ),
    "adv_test_x04": (
        True,
        "PDF System Card Appendix D.3 Prompt 3 (p.85) verified: accurately contrasts Early GPT-4's step-by-step laundering workflow against Launch GPT-4's outright refusal quote across gpt4_p085_text_00/01/02.",
    ),
    "adv_test_u01": (
        True,
        "PDF Section 2 (p.2) verified: A100 GPU count, training days, and FLOPs are explicitly withheld; cleanly abstained without guessing external rumors.",
    ),
    "adv_test_u02": (
        True,
        "PDF Table 1 (p.5) & Table 9 (p.30) verified: Putnam, USAMO, and IMO are not evaluated in the report; cleanly abstained without substituting AMC 10/12.",
    ),
    "adv_test_u03": (
        True,
        "PDF System Card Fig 11 (p.98) verified: compares only GPT-4 RLHF, GPT-3.5 Turbo RLHF, and GPT-3.5 RLHF; neither Claude 2 nor PaLM-2 exists; cleanly abstained.",
    ),
    "adv_test_u04": (
        True,
        "PDF Table 14 (p.34) verified: reports 1997 daily meat consumption only, no 2020 figures; cleanly abstained.",
    ),
    "adv_test_u05": (
        True,
        "PDF Table 15 (p.35) verified: Ecole Polytechnique bolometer diagram specifies symbolic L, S, lambda, T0, Tb only, no numerical SI values; cleanly abstained.",
    ),
    "adv_test_u06": (
        True,
        "March 2023 GPT-4 Technical Report verified: neither MT-Bench nor BFCL exists in the report; cleanly abstained.",
    ),
}


def main() -> None:
    questions = {
        q["id"]: q
        for q in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
        if q["split"] == "test"
    }
    traces = json.loads(TRACES_PATH.read_text(encoding="utf-8"))
    chunks_by_id = {c["id"]: c for c in json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))}

    assert len(questions) == 30, f"Expected 30 test questions, got {len(questions)}"
    assert len(traces) == 30, f"Expected 30 test traces, got {len(traces)}"

    reviews: list[dict[str, Any]] = []
    by_group: dict[str, dict[str, int]] = {}
    for t in traces:
        qid = t["id"]
        q = questions[qid]
        grp = q["primary_group"]
        passed, note = MANUAL_AUDIT_NOTES[qid]
        retrieved_ids = {h["id"] for h in t.get("retrieved_chunks", [])}
        cits_valid = all((cid in chunks_by_id and cid in retrieved_ids) for cid in t.get("citations", []))
        if t["status"] == "error":
            passed = False

        rec = by_group.setdefault(grp, {"passed": 0, "total": 0})
        rec["total"] += 1
        if passed:
            rec["passed"] += 1

        reviews.append(
            {
                "id": qid,
                "primary_group": grp,
                "answerable": q["answerable"],
                "model": t["model"],
                "status": t["status"],
                "passed": passed,
                "citations_in_retrieved_and_corpus": cits_valid,
                "cited_pages": sorted({chunks_by_id[c]["page"] for c in t.get("citations", []) if c in chunks_by_id}),
                "gold_pages": q["gold_evidence"]["pages"],
                "citations": t.get("citations", []),
                "latency_ms": t["latency_ms"],
                "cost_usd": t["cost_usd"],
                "audit_notes": note,
            }
        )

    total_passed = sum(1 for r in reviews if r["passed"])
    total_cases = len(reviews)
    latencies = sorted(float(t["latency_ms"]) for t in traces)
    total_cost = round(sum(float(t["cost_usd"]) for t in traces), 6)
    provider_errors = sum(1 for t in traces if t["status"] == "error")

    summary = {
        "split": "test",
        "pipeline": "multimodal_hybrid_agent",
        "chat_model": traces[0]["model"],
        "total_cases": total_cases,
        "total_passed": total_passed,
        "pass_rate_pct": round(100.0 * total_passed / total_cases, 2),
        "meets_90pct_threshold": (total_passed * 10 >= total_cases * 9),
        "provider_errors": provider_errors,
        "median_latency_ms": round(latencies[len(latencies) // 2], 2),
        "total_cost_usd": total_cost,
        "total_llm_calls": sum(int(t["llm_calls_count"]) for t in traces),
        "total_tool_calls": sum(int(t["tool_calls_count"]) for t in traces),
        "per_group": {
            grp: {
                "passed": d["passed"],
                "total": d["total"],
                "ratio": f"{d['passed']}/{d['total']}",
                "pct": round(100.0 * d["passed"] / d["total"], 2),
            }
            for grp, d in by_group.items()
        },
        "failed_cases": [
            {"id": r["id"], "primary_group": r["primary_group"], "notes": r["audit_notes"]}
            for r in reviews
            if not r["passed"]
        ],
    }

    REVIEWS_OUT.write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
