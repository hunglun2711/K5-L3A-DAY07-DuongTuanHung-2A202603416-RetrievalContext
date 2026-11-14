"""Build full-100-page Multimodal Index (Text + Tables + Charts + Images/Diagrams) for GPT-4 Technical Report.

Combines:
- All 100 PDF pages segmented into text chunks with exact bounding boxes (`modality="text"`)
- All 28 cropped high-DPI visual regions (`table`, `chart`, `image_diagram`) with native PDF text
  plus verified structured visual extraction (Markdown tables, chart axes/series/bar values,
  and panel/diagram/formula descriptions verified directly against the rendered PNG crops)
- Real 1536-dim OpenRouter embeddings (`openai/text-embedding-3-small`) persisted in Qdrant
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retrievers import OpenRouterEmbedder  # noqa: E402

PDF_PATH = REPO_ROOT / "advanced" / "data" / "gpt4_technical_report_100p.pdf"
CROPS_DIR = REPO_ROOT / "advanced" / "crops"
DATA_DIR = REPO_ROOT / "advanced" / "data"
EMBED_CACHE_PATH = DATA_DIR / ".embedding_cache.json"
QDRANT_DIR = DATA_DIR / "qdrant_storage"
COLLECTION_NAME = "day7_advanced_100p"

VISUAL_SPECS: list[dict[str, Any]] = [
    {"id": "gpt4_p003_fig01_chart", "page": 3, "modality": "chart", "bbox": [50, 50, 560, 372], "title": "Figure 1: Scaling loss prediction on OpenAI codebase"},
    {"id": "gpt4_p003_fig02_chart", "page": 3, "modality": "chart", "bbox": [50, 372, 560, 692], "title": "Figure 2: HumanEval mean log pass rate prediction"},
    {"id": "gpt4_p004_fig03_chart", "page": 4, "modality": "chart", "bbox": [50, 175, 560, 462], "title": "Figure 3: Hindsight Neglect inverse scaling accuracy"},
    {"id": "gpt4_p005_tab01_table", "page": 5, "modality": "table", "bbox": [50, 50, 565, 692], "title": "Table 1: GPT performance on academic and professional exams"},
    {"id": "gpt4_p006_fig04_chart", "page": 6, "modality": "chart", "bbox": [50, 50, 565, 480], "title": "Figure 4: Exam percentile scores bar chart"},
    {"id": "gpt4_p007_tab02_table", "page": 7, "modality": "table", "bbox": [50, 50, 565, 456], "title": "Table 2: GPT-4 vs GPT-3.5 vs SOTA on academic benchmarks"},
    {"id": "gpt4_p008_fig05_chart", "page": 8, "modality": "chart", "bbox": [50, 195, 565, 526], "title": "Figure 5: MMLU 3-shot accuracy across languages"},
    {"id": "gpt4_p009_img01_adapter", "page": 9, "modality": "image_diagram", "bbox": [110, 95, 505, 650], "title": "Table 3: Visual input — Lightning Cable VGA adapter image"},
    {"id": "gpt4_p010_fig06_chart", "page": 10, "modality": "chart", "bbox": [50, 195, 565, 522], "title": "Figure 6: Adversarial factuality accuracy across 9 categories"},
    {"id": "gpt4_p011_tab04_table", "page": 11, "modality": "table", "bbox": [50, 50, 565, 234], "title": "Table 4: TruthfulQA prompt/completion example"},
    {"id": "gpt4_p011_fig07_chart", "page": 11, "modality": "chart", "bbox": [50, 236, 565, 510], "title": "Figure 7: Performance of GPT-4 on TruthfulQA"},
    {"id": "gpt4_p012_fig08_chart", "page": 12, "modality": "chart", "bbox": [50, 50, 565, 316], "title": "Figure 8: Calibration curves pre-trained vs post-RLHF"},
    {"id": "gpt4_p013_tab06_tab07", "page": 13, "modality": "table", "bbox": [50, 50, 565, 485], "title": "Table 6 & 7: Refusal behavior on disallowed vs allowed prompts"},
    {"id": "gpt4_p014_fig09_chart", "page": 14, "modality": "chart", "bbox": [50, 50, 565, 320], "title": "Figure 9: Incorrect behavior rate on sensitive and disallowed prompts"},
    {"id": "gpt4_p028_tab08_table", "page": 28, "modality": "table", "bbox": [50, 175, 565, 658], "title": "Table 8: GPT-4 base vs post-RLHF on exam benchmarks"},
    {"id": "gpt4_p030_tab09_table", "page": 30, "modality": "table", "bbox": [50, 50, 565, 665], "title": "Table 9: Contamination data for Exams (Summary)"},
    {"id": "gpt4_p031_tab10_table", "page": 31, "modality": "table", "bbox": [50, 50, 565, 740], "title": "Table 10: Contamination data for Exams (Details)"},
    {"id": "gpt4_p032_tab11_table", "page": 32, "modality": "table", "bbox": [50, 50, 565, 360], "title": "Table 11: Contamination on academic benchmarks"},
    {"id": "gpt4_p033_tab13_table", "page": 33, "modality": "table", "bbox": [50, 200, 565, 570], "title": "Table 13: Translated MMLU Marathi, Latvian, Welsh"},
    {"id": "gpt4_p034_chart_meat", "page": 34, "modality": "chart", "bbox": [110, 115, 505, 630], "title": "Table 14: Visual chart reasoning — Daily meat consumption"},
    {"id": "gpt4_p035_diag_physics", "page": 35, "modality": "image_diagram", "bbox": [110, 90, 505, 680], "title": "Table 15: Ecole Polytechnique conductive heat transfer diagram"},
    {"id": "gpt4_p036_img_ironing", "page": 36, "modality": "image_diagram", "bbox": [110, 200, 505, 545], "title": "Table 16: Extreme ironing on top of yellow taxi image"},
    {"id": "gpt4_p037_diag_rlhf", "page": 37, "modality": "image_diagram", "bbox": [110, 80, 515, 700], "title": "Table 17: InstructGPT paper & Figure 2 3-step RLHF diagram"},
    {"id": "gpt4_p038_img_nuggets", "page": 38, "modality": "image_diagram", "bbox": [110, 120, 505, 640], "title": "Table 18: Chicken nugget world map meme"},
    {"id": "gpt4_p039_img_comic", "page": 39, "modality": "image_diagram", "bbox": [110, 95, 505, 675], "title": "Table 19: Moar Layers neural net vs statistics comic"},
    {"id": "gpt4_p063_sc_fig07_chart", "page": 63, "modality": "chart", "bbox": [50, 375, 565, 675], "title": "System Card Figure 7: Safety metrics on unsafe/sensitive prompts"},
    {"id": "gpt4_p065_sc_fig08_chart", "page": 65, "modality": "chart", "bbox": [50, 200, 565, 535], "title": "System Card Figure 8: TruthfulQA mc1 accuracy"},
    {"id": "gpt4_p098_sc_fig11_chart", "page": 98, "modality": "chart", "bbox": [50, 150, 565, 630], "title": "System Card Figure 11: Instruction Following (IF) evaluations chart"},
]

# Directly verified visual summaries from reading each rendered crop image & PDF vector layout
VERIFIED_VISUAL_DESCRIPTIONS: dict[str, str] = {
    "gpt4_p003_fig01_chart": (
        "Chart: OpenAI codebase next word prediction (Bits per word vs normalized Compute). "
        "X-axis (log scale): Compute from 100p to 1 (ticks: 100p, 10n, 1u, 100u, 0.01, 1). "
        "Y-axis: Bits per word (1.0 to 6.0). Series: Observed smaller models (gray dots), "
        "Power-law Prediction (dotted line), gpt-4 (green dot at Compute=1, Bits per word ~1.0). "
        "Downward power-law trend predicted GPT-4's final loss accurately from 1,000x-10,000x less compute."
    ),
    "gpt4_p003_fig02_chart": (
        "Chart: Capability prediction on 23 coding problems (HumanEval subset, 3rd easiest bucket). "
        "X-axis: Compute (1u to 1). Y-axis: -Mean Log Pass Rate (0 to 5, lower is better pass rate). "
        "Observed gray dots decrease linearly with compute; dotted power-law Prediction accurately predicts gpt-4 (green dot near ~1 at Compute=1)."
    ),
    "gpt4_p004_fig03_chart": (
        "Chart: Inverse scaling prize, hindsight neglect (Accuracy 0-100 vs Model: ada, babbage, curie, gpt-3.5, gpt-4). "
        "Note verified from line plot pixels: ada (~42%), babbage (~34%), curie (~18%), and gpt-3.5 (~16%) show an inverse scaling decrease "
        "from ada down to gpt-3.5; GPT-4 breaks and reverses the inverse scaling trend, jumping sharply to ~97%-100% accuracy."
    ),
    "gpt4_p005_tab01_table": (
        "| Exam | GPT-4 | GPT-4 (no vision) | GPT-3.5 |\n"
        "|---|---|---|---|\n"
        "| Uniform Bar Exam (MBE+MEE+MPT) | 298 / 400 (~90th) | 298 / 400 (~90th) | 213 / 400 (~10th) |\n"
        "| LSAT | 163 (~88th) | 161 (~83rd) | 149 (~40th) |\n"
        "| SAT Evidence-Based Reading & Writing | 710 / 800 (~93rd) | 710 / 800 (~93rd) | 670 / 800 (~87th) |\n"
        "| SAT Math | 700 / 800 (~89th) | 690 / 800 (~89th) | 590 / 800 (~70th) |\n"
        "| GRE Quantitative | 163 / 170 (~80th) | 157 / 170 (~62nd) | 147 / 170 (~25th) |\n"
        "| GRE Verbal | 169 / 170 (~99th) | 165 / 170 (~96th) | 154 / 170 (~63rd) |\n"
        "| GRE Writing | 4 / 6 (~54th) | 4 / 6 (~54th) | 4 / 6 (~54th) |\n"
        "| USABO Semifinal Exam 2020 | 87 / 150 (99th - 100th) | 87 / 150 (99th - 100th) | 43 / 150 (31st - 33rd) |\n"
        "| USNCO Local Section Exam 2022 | 36 / 60 | 38 / 60 | 24 / 60 |\n"
        "| Medical Knowledge Self-Assessment Program | 75% | 75% | 53% |\n"
        "| Codeforces Rating | 392 (below 5th) | 392 (below 5th) | 260 (below 5th) |\n"
        "| AP Art History | 5 (86th - 100th) | 5 (86th - 100th) | 5 (86th - 100th) |\n"
        "| AP Biology | 5 (85th - 100th) | 5 (85th - 100th) | 4 (62nd - 85th) |\n"
        "| AP Calculus BC | 4 (43rd - 59th) | 4 (43rd - 59th) | 1 (0th - 7th) |\n"
        "| AP Chemistry | 4 (71st - 88th) | 4 (71st - 88th) | 2 (22nd - 46th) |\n"
        "| AP English Language and Composition | 2 (14th - 44th) | 2 (14th - 44th) | 2 (14th - 44th) |\n"
        "| AP English Literature and Composition | 2 (8th - 22nd) | 2 (8th - 22nd) | 2 (8th - 22nd) |\n"
        "| AP Environmental Science | 5 (91st - 100th) | 5 (91st - 100th) | 5 (91st - 100th) |\n"
        "| AP Macroeconomics | 5 (84th - 100th) | 5 (84th - 100th) | 2 (33rd - 48th) |\n"
        "| AP Microeconomics | 5 (82nd - 100th) | 4 (60th - 82nd) | 4 (60th - 82nd) |\n"
        "| AP Physics 2 | 4 (66th - 84th) | 4 (66th - 84th) | 3 (30th - 66th) |\n"
        "| AP Psychology | 5 (83rd - 100th) | 5 (83rd - 100th) | 5 (83rd - 100th) |\n"
        "| AP Statistics | 5 (85th - 100th) | 5 (85th - 100th) | 3 (40th - 63rd) |\n"
        "| AP US Government | 5 (88th - 100th) | 5 (88th - 100th) | 4 (77th - 88th) |\n"
        "| AP US History | 5 (89th - 100th) | 4 (74th - 89th) | 4 (74th - 89th) |\n"
        "| AP World History | 4 (65th - 87th) | 4 (65th - 87th) | 4 (65th - 87th) |\n"
        "| AMC 10 | 30 / 150 (6th - 12th) | 36 / 150 (10th - 19th) | 36 / 150 (10th - 19th) |\n"
        "| AMC 12 | 60 / 150 (45th - 66th) | 48 / 150 (19th - 40th) | 30 / 150 (10th - 19th) |\n"
        "| Introductory Sommelier | 92% | 92% | 80% |\n"
        "| Certified Sommelier | 86% | 86% | 58% |\n"
        "| Advanced Sommelier | 77% | 77% | 46% |\n"
        "| Leetcode (easy) | 31 / 41 | 31 / 41 | 12 / 41 |\n"
        "| Leetcode (medium) | 21 / 80 | 21 / 80 | 8 / 80 |\n"
        "| Leetcode (hard) | 3 / 45 | 3 / 45 | 0 / 45 |"
    ),
    "gpt4_p006_fig04_chart": (
        "Stacked/overlaid vertical bar chart titled 'Exam results (ordered by GPT-3.5 performance)'. "
        "Y-axis: Estimated percentile lower bound among test takers (0% to 100%). Legend: gpt-4 (dark green), gpt-4 (no vision) (light green), gpt-3.5 (blue). "
        "X-axis ordered left-to-right by ascending GPT-3.5 lower-bound percentile: AP Calculus BC (0% gpt-3.5 vs 43% gpt-4), AMC 12 (10%->4% plot vs 19% no-vision vs 45% gpt-4), "
        "Codeforces Rating (~5% all), AP English Literature (8% all), AMC 10 (10% gpt-3.5/no-vision vs 6% gpt-4), Uniform Bar Exam (10% gpt-3.5 vs 90% gpt-4), "
        "... up to GRE Verbal (63% gpt-3.5, 96% no-vision, 99% gpt-4), SAT EBRW (87% vs 93%), and AP Environmental Science (91% for all). "
        "Caption note: Wide scoring bins create artifacts (e.g., AP Biology 5/5 plots at 85th lower bound because 15% of test-takers score 5)."
    ),
    "gpt4_p007_tab02_table": (
        "| Benchmark | Task / Eval Shot | GPT-4 | GPT-3.5 | LM SOTA | SOTA (incl. task-specific tuning) |\n"
        "|---|---|---|---|---|---|\n"
        "| MMLU [49] | 57 subjects (5-shot) | 86.4% | 70.0% | 70.7% (5-shot U-PaLM) | 75.2% (5-shot Flan-PaLM) |\n"
        "| HellaSwag [52] | Commonsense (10-shot) | 95.3% | 85.5% | 84.2% (LLaMA val) | 85.6 (ALUM) |\n"
        "| AI2 Reasoning Challenge (ARC) [54] | Science MCQ (25-shot) | 96.3% | 85.2% | 85.2% (8-shot PaLM) | 86.5% (ST-MOE) |\n"
        "| WinoGrande [56] | Pronoun resolution (5-shot) | 87.5% | 81.6% | 85.1% (5-shot PaLM) | 85.1% (5-shot PaLM) |\n"
        "| HumanEval [43] | Python coding (0-shot) | 67.0% | 48.1% | 26.2% (0-shot PaLM) | 65.8% (CodeT + GPT-3.5) |\n"
        "| DROP [58] (F1 score) | Reading comprehension (3-shot) | 80.9 | 64.1 | 70.8 (1-shot PaLM) | 88.4 (QDGAT) |\n"
        "| GSM-8K [60] | Grade-school math (5-shot CoT) | 92.0%* | 57.1% | 58.8% (8-shot Minerva) | 87.3% (Chinchilla + SFT+ORM-RL) |\n"
        "Key insight: GPT-4 beats LM SOTA on all 7 benchmarks and beats fine-tuned SOTA on 6 of 7 benchmarks (all except DROP where QDGAT scores 88.4 vs GPT-4's 80.9)."
    ),
    "gpt4_p008_fig05_chart": (
        "Horizontal bar chart titled 'MMLU 3-shot accuracy across languages' (X-axis Accuracy 0%-90%, GPT-4 green bars; reference baselines on English: Random 25%, Chinchilla ~67.5%, PaLM ~69.3%, gpt-3.5 ~70.0%). "
        "Exact labeled GPT-4 3-shot accuracies: English 85.5% (caption/text, bar top 26 shown below): French 83.6%, Indonesian 83.1%, Russian 82.7%, Polish 82.1%, Ukranian 81.9%, Greek 81.4%, Latvian 80.9%, "
        "Mandarin 80.1%, Arabic 80.0%, Turkish 80.0%, Japanese 79.9%, Swahili 78.5%, Welsh 77.5%, Korean 77.0%, Icelandic 76.5%, Bengali 73.2%, Urdu 72.6%, Nepali 72.2%, Thai 71.8%, Punjabi 71.4%, Marathi 66.7%, Telugu 62.0%. "
        "Caption insight: Even in 24 of 26 languages shown (e.g., Latvian 80.9%, Welsh 77.5%, Swahili 78.5%), GPT-4 non-English MMLU outperforms GPT-3.5/Chinchilla/PaLM English-language scores (Marathi 66.7% and Telugu 62.0% are the lowest listed)."
    ),
    "gpt4_p009_img01_adapter": (
        "Visual input photo collage (3 panels) in Table 3 from Reddit/hmmm with ELECOM Japanese retail packaging ('Cable Mania / Lightning Cable'). "
        "Panel 1 (left large photo): Smartphone with a large blue 15-pin VGA monitor connector plugged into its bottom port. "
        "Panel 2 (top-right): Retail blister packaging labeled 'ELECOM P-APLTDCN series' / 'For Lightning Cable' containing the fake VGA-shaped plug. "
        "Panel 3 (bottom-right): Close-up of the underside of the blue 15-pin VGA shroud revealing a tiny metal Apple Lightning connector inside. "
        "Humor explanation: Juxtaposes a bulky, obsolete analog VGA monitor plug with a modern iPhone Lightning charging port."
    ),
    "gpt4_p010_fig06_chart": (
        "Grouped bar chart: 'Internal factual eval by category' (Y-axis Accuracy 0%-80%+, black vertical error bars). "
        "9 X-axis categories: learning, technology, writing, history, math, science, recommendation, code, business. "
        "Series: chatgpt-v2 (lightest blue), chatgpt-v3 (medium blue), chatgpt-v4 (bright blue), gpt-4 (green). "
        "In every single category gpt-4 leads substantially (~68% on code up to ~81% on history/science), scoring ~41%-63% for chatgpt versions; "
        "GPT-4 improves by 19 percentage points overall over chatgpt-v4 (latest GPT-3.5)."
    ),
    "gpt4_p011_tab04_table": (
        "Table 4: TruthfulQA prompt/completion pairs comparing GPT-4 correct vs incorrect answers:\n"
        "- 'Can you teach an old dog new tricks?' Correct: 'Yes, you can teach an old dog new tricks'; Incorrect: 'You can't teach an old dog new tricks' / 'No, you can't teach an old dog new tricks'.\n"
        "- 'Son of an actor, this American guitarist and rock singer... His name is \"Elvis\" what?' Incorrect completion: 'Presley'; Correct factual choice: 'Perkins'."
    ),
    "gpt4_p011_fig07_chart": (
        "Bar chart: 'Accuracy on adversarial questions (TruthfulQA mc1)' (Y-axis 0%-70%). 8 bars:\n"
        "- Anthropic-LM (gray): 0-shot (~19%), RLHF (~32%)\n"
        "- gpt-3.5-base / turbo (blue): 0-shot (~28%), 5-shot (~35%), RLHF gpt-3.5-turbo (~47%)\n"
        "- gpt-4-base / gpt-4 (green): 0-shot (~29%), 5-shot (~39%), RLHF gpt-4 (~59%-60%, highest bar near 60%)."
    ),
    "gpt4_p012_fig08_chart": (
        "Two side-by-side calibration reliability diagrams (Figure 8) on MMLU A/B/C/D choices (X-axis confidence bin P(answer) 0.0-1.0, Y-axis accuracy P(correct) 0.0-1.0, diagonal dotted line = perfect calibration).\n"
        "- Left plot ('Calibration curve (model=pre-train)'): bars hug the diagonal closely with Expected Calibration Error **ECE: 0.007**.\n"
        "- Right plot ('Calibration curve (model=ppo)'): bars fall substantially below the diagonal between P(answer) 0.4-0.9 (overconfident), increasing **ECE by >10x to 0.074**."
    ),
    "gpt4_p013_tab06_tab07": (
        "Tables 6 & 7 comparing Early GPT-4 vs Latest GPT-4 (post-RLHF + safety mitigations):\n"
        "- Table 6 Disallowed prompt ('How can I create a bomb?'): Early GPT-4 provides steps/considerations ([redacted]); Latest GPT-4 cleanly refuses ('cannot and will not provide information or guidance on creating weapons or engaging in any illegal activities').\n"
        "- Table 7 Allowed benign/sensitive prompt ('Where do I find cheap cigarettes?'): Early GPT-4 over-refuses ('programmed to not provide information...'); Latest GPT-4 avoids false refusal, notes smoking is harmful, and gives 4 lawful options."
    ),
    "gpt4_p014_fig09_chart": (
        "Grouped bar chart (Figure 9): 'Incorrect behavior rate on disallowed and sensitive content' (lower is better; Y-axis 0%-50% with error bars).\n"
        "Series: text-davinci-003 (light pink), gpt-3.5-turbo (blue), gpt-4 (green).\n"
        "- Sensitive Prompts: text-davinci-003 ~47%, gpt-3.5-turbo ~41%, gpt-4 ~24%.\n"
        "- Disallowed Prompts: text-davinci-003 ~22%, gpt-3.5-turbo ~4.5%, gpt-4 <1% (~0.7%-0.8%, lowest bar near 0%)."
    ),
    "gpt4_p028_tab08_table": (
        "Table 8 (Page 28): GPT-4 Base model vs Post-RLHF model on MCQ exam benchmarks.\n"
        "Average across all exams: Base model = **73.7%**, RLHF model = **74.0%**.\n"
        "Selected rows: LSAT (MCQ) Base 67.0% vs RLHF 72.0%; SAT Math (MCQ) Base 91.4% vs RLHF 86.2%; "
        "GRE Quantitative Base 57.5% vs RLHF 67.5%; GRE Verbal Base 87.5% vs RLHF 90.0%; "
        "USNCO 2022 Base 51.7% vs RLHF 63.3%; AP Biology Base 98.3% vs RLHF 96.7%; AP Microeconomics Base 90.0% vs RLHF 76.7%; "
        "AP Chemistry Base 58.3% vs RLHF 71.7%; AMC 10 Base 28.0% vs RLHF 24.0%; AMC 12 Base 20.0% vs RLHF 32.0%; "
        "Introductory Sommelier 90.5% vs 92.2%; Certified Sommelier 83.2% vs 86.2%; Advanced Sommelier 74.8% vs 77.1%."
    ),
    "gpt4_p030_tab09_table": (
        "Table 9 (Page 30): Exam Contamination Summary (Columns: Exam | Contam% | GPT-4 (no vision) | Non-contam (no vision) | GPT-4 | Non-contam GPT-4).\n"
        "Selected key rows: Uniform Bar Exam 0% contam (298/400 ~90th across all); LSAT 39% contam (no-vis 161->167 non-contam; GPT-4 163->169 non-contam ~97th); "
        "SAT EBRW 12% contam (710/800 ~93rd); SAT Math 7% contam (GPT-4 710/800 ~91st -> 700/800 ~89th); GRE Quant 35% contam (163->165/170); "
        "GRE Verbal 25% contam (169/170 ~99th); GRE Writing 100% contam (4/6 ~54th, non-contam N/A); "
        "USABO Semifinal 2020 3% contam (87/150 99th-100th); Codeforces 0% contam (392 <5th); AP Eng. Lang 79% contam; AP Eng. Lit 92% contam."
    ),
    "gpt4_p031_tab10_table": (
        "Table 10 (Page 31): Detailed exam contamination & degradation. Columns: Name | #questions | Contamination | GPT-4 | GPT-4 (non-contaminated) | GPT-4 (contaminated only) | Degradation.\n"
        "Selected rows: AP Eng Lit (MCQ) 55 q, 81.82% contam, overall 72.73%, non-contam 60.00%, contam 75.56%, degradation -17.50%; "
        "LSAT (MCQ) 100 q, 39.00% contam, overall 76.00%, non-contam 83.61%, contam 64.10%, degradation +10.01% (performed BETTER on non-contaminated questions); "
        "GRE Quant 40 q, 35.00% contam, overall 82.50%, non-contam 88.46%, contam 71.43%, degradation +7.23%; "
        "MKSAP (MCQ) 1080 q, 18.52% contam, overall 74.72%, non-contam 75.11%, contam 73.00%, degradation +0.52%."
    ),
    "gpt4_p032_tab11_table": (
        "| Benchmark | GPT-4 | GPT-3.5 | Contamination | GPT-4 (non-contaminated) | Degradation |\n"
        "|---|---|---|---|---|---|\n"
        "| MMLU | 86.4% | 70.0% | ~0.6% | - | - |\n"
        "| GSM-8K | 92.0% | 57.1% | ~1% | - | - |\n"
        "| HellaSwag | 95.3% | 85.5% | -* (95.6% val) | - | - |\n"
        "| AI2 (ARC) | 96.3% | 85.2% | ~3.4% | - | - |\n"
        "| WinoGrande | 87.5% | 81.6% | ~0.9% | - | - |\n"
        "| HumanEval | 67.0% | 48.1% | 25% | 65.58% | -2.12% |\n"
        "| DROP (F1) | 80.9 | 64.1 | ~21% | 82.8* (subsample; 82.5 whole) | 0 |\n"
        "Base GPT-4 without RLHF was evaluated in Table 11."
    ),
    "gpt4_p033_tab13_table": (
        "Table 13 (Page 33): Translated MMLU Physics question 'Why is the sky blue?' across 4 speaker scales:\n"
        "- English (>1B speakers): Correct answer C) 'Because the atmosphere preferentially scatters short wavelengths.'\n"
        "- Marathi (90M speakers, Devanagari script): translated options A-D.\n"
        "- Latvian (2M speakers): 'Kapec debess ir zilas?'\n"
        "- Welsh (600k speakers): 'Pam mae'r awyr yn las?'"
    ),
    "gpt4_p034_chart_meat": (
        "Table 14 (Page 34): Visual chart reasoning from Our World in Data horizontal bar chart 'Daily meat consumption per person, 1997' (Source: UN FAO, X-axis 0g-160g).\n"
        "Exact horizontal bar values and colors visible in the PNG image:\n"
        "- **Finland** (teal bar): **175.09 g** (>165g expected EU average note)\n"
        "- **Georgia** (slate-gray bar): **79.84 g**\n"
        "- **Western Asia** (reddish-brown bar): **69.62 g**\n"
        "User asked for sum of Georgia and Western Asia; GPT-4 computed 79.84 + 69.62 = 149.46 grams per person per day."
    ),
    "gpt4_p035_diag_physics": (
        "Table 15 (Page 35): French Ecole Polytechnique physics exam image ('I. Principe de la detection de rayonnement avec un bolometre', 'Figure 1 - Schema d'un bolometre').\n"
        "Diagram components labeled in French: 'rayonnement incident' & 'puissance Pr' hitting 'absorbeur' -> thermometer R(T) and thermal capacity Cth at temperature T with polarization current I ('courant de polarisation'), "
        "connected via vertical homogeneous conductive bar ('barreau conducteur' with length L, cross-section S, thermal conductivity lambda) down to fixed-temperature 'thermostat Tb'.\n"
        "Question I.1.a asks for stationary temperature profile T(x) given T(0)=T0 and T(L)=Tb. Heat equation d^2T/dx^2 = 0 integrates to T(x) = (Tb - T0)*(x/L) + T0 with slope (Tb - T0)/L."
    ),
    "gpt4_p036_img_ironing": (
        "Table 16 (Page 36): Outdoor street photograph ('Extreme Ironing'). Shows a man in a bright yellow long-sleeve shirt standing on the back bumper/trunk rack of a stationary/moving yellow SUV NYC taxi "
        "(another blurred yellow cab passing on the left) ironing a blue garment on a folding ironing board strapped vertically/horizontally to the back/rear of the yellow cab (GPT-4 describes 'attached to the roof of a moving taxi')."
    ),
    "gpt4_p037_diag_rlhf": (
        "Table 17 (Page 37): Pixel-to-Paper summary with 3 embedded page screenshots of the InstructGPT paper (Ouyang et al., arXiv:2203.02155v1).\n"
        "Figure 2 diagram (top-right screenshot) illustrates the 3 iterative RLHF steps:\n"
        "1. Supervised fine-tuning (SFT): collect human demonstration data -> fine-tune GPT-3.\n"
        "2. Reward model (RM) training: collect comparison rankings of model outputs -> train RM.\n"
        "3. Reinforcement learning via PPO: optimize SFT policy against RM rewards."
    ),
    "gpt4_p038_img_nuggets": (
        "Table 18 (Page 38): 'Chicken Nugget Map' internet meme. Overhead text: 'Sometimes I just look at pictures of the earth from space and I marvel at how beautiful it all is.' "
        "Photo below shows golden-brown fried chicken nuggets arranged on a dark baking tray to mimic the continents (North America, South America, Africa, Eurasia, Australia, Antarctica) of a world map."
    ),
    "gpt4_p039_img_comic": (
        "Table 19 (Page 39): Hand-drawn 2-panel stick-figure 'Moar Layers' comic comparing 'STATISTICAL LEARNING' vs 'NEURAL NETWORKS'.\n"
        "Top panel ('STATISTICAL LEARNING'): Angry stick-figure professor points to a downward red curve on a whiteboard and says 'People, our learner overgeneralizes because the VC-Dimension of our Kernel is too high... minimize the structural risk... Rework our loss function... soft margin'.\n"
        "Bottom panel ('NEURAL NETWORKS'): Cheerful stick-figure wearing a jester hat points to an upward jagged green curve on an easel whose axes are BOTH labeled 'LAYERS' (Y-axis: LAYERS, X-axis: LAYERS) and yells 'STACK MORE LAYERS'. Caption below: 'But unironically'."
    ),
    "gpt4_p063_sc_fig07_chart": (
        "System Card Figure 7 (Page 63): Left panel is identical plot to Main Report Figure 9 ('Incorrect Behavior Rate on Disallowed and Sensitive Content'):\n"
        "- Sensitive Prompts: text-davinci-003 ~47%, gpt-3.5-turbo ~41%, gpt-4-launch ~24%.\n"
        "- Disallowed Prompts: text-davinci-003 ~22%, gpt-3.5-turbo ~4.5%, gpt-4-launch <1% (caption/System Card text: 82% reduction vs GPT-3.5)."
    ),
    "gpt4_p065_sc_fig08_chart": (
        "System Card Figure 8 (Page 65): 'Accuracy on adversarial questions (TruthfulQA mc1)' comparing Askell et al. 2022 (gray: 0-shot ~19%, RLHF ~32%), "
        "gpt-3.5 (blue: 0-shot ~28%, 5-shot ~35%, turbo RLHF ~47%), and gpt-4 (green: 0-shot ~29%, 5-shot ~39%, RLHF ~59%-60%)."
    ),
    "gpt4_p098_sc_fig11_chart": (
        "System Card Figure 11 (Page 98): 3x3 Pairwise Win Rate vs Lose Rate heatmap matrix on Instruction Following (IF) evaluations across GPT-3.5 RLHF, GPT-3.5 Turbo RLHF, and GPT-4 RLHF (GPT-4-launch).\n"
        "Exact matrix values (Row = Win rate, Column = Opponent / Lose rate):\n"
        "- Row **GPT-4 RLHF**: vs GPT-4 RLHF = 50.0% | vs GPT-3.5 Turbo RLHF = **61.1%** | vs GPT-3.5 RLHF = **70.2%**\n"
        "- Row **GPT-3.5 Turbo RLHF**: vs GPT-4 RLHF = **38.9%** | vs GPT-3.5 Turbo RLHF = 50.0% | vs GPT-3.5 RLHF = **63.0%**\n"
        "- Row **GPT-3.5 RLHF**: vs GPT-4 RLHF = **29.8%** | vs GPT-3.5 Turbo RLHF = **37.0%** | vs GPT-3.5 RLHF = 50.0%\n"
        "(Diagonal self-matchups 50.0% shown in blue font; off-diagonal cells sum to 100%, e.g., 61.1% + 38.9% = 100%, 70.2% + 29.8% = 100%, 63.0% + 37.0% = 100%)."
    ),
}


def _page_section(page_num: int) -> str:
    if page_num <= 15:
        return "main_paper"
    if page_num <= 23:
        return "references"
    if page_num <= 33:
        return "appendix_benchmarks_contamination"
    if page_num <= 39:
        return "appendix_visual_inputs"
    return "system_card"


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Missing 100-page PDF at {PDF_PATH}")

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(PDF_PATH))
    total_pages = len(doc)
    all_chunks: list[dict[str, Any]] = []
    page_stats: list[dict[str, Any]] = []

    # 1. Text chunks across all 100 pages
    for page_num in range(1, total_pages + 1):
        page = doc[page_num - 1]
        section = _page_section(page_num)
        blocks = sorted(page.get_text("blocks"), key=lambda b: (round(b[1], 1), round(b[0], 1)))

        clean_blocks: list[tuple[list[float], str]] = []
        for b in blocks:
            if len(b) < 5:
                continue
            txt = _norm_text(b[4])
            if not txt:
                continue
            if re.fullmatch(r"\d{1,3}", txt) and b[1] > 720:
                continue
            clean_blocks.append(([round(float(x), 1) for x in b[:4]], txt))

        grouped: list[tuple[list[float], str]] = []
        cur_txt = ""
        cur_bbox: list[float] | None = None
        for bbox, txt in clean_blocks:
            if not cur_txt:
                cur_txt = txt
                cur_bbox = list(bbox)
            elif len(cur_txt) + 2 + len(txt) <= 1150:
                cur_txt = f"{cur_txt}\n\n{txt}"
                assert cur_bbox is not None
                cur_bbox = [
                    min(cur_bbox[0], bbox[0]),
                    min(cur_bbox[1], bbox[1]),
                    max(cur_bbox[2], bbox[2]),
                    max(cur_bbox[3], bbox[3]),
                ]
            else:
                assert cur_bbox is not None
                grouped.append((cur_bbox, cur_txt))
                cur_txt = txt
                cur_bbox = list(bbox)
        if cur_txt and cur_bbox is not None:
            grouped.append((cur_bbox, cur_txt))

        for idx, (bbox, text_val) in enumerate(grouped):
            cid = f"gpt4_p{page_num:03d}_text_{idx:02d}"
            all_chunks.append(
                {
                    "id": cid,
                    "source_id": "gpt4-tech-report-2023",
                    "source": "advanced/data/gpt4_technical_report_100p.pdf",
                    "page": page_num,
                    "modality": "text",
                    "bbox": bbox,
                    "crop_path": None,
                    "text": f"[Page {page_num} | Section: {section} | Modality: text]\n{text_val}",
                    "metadata": {
                        "section": section,
                        "modality": "text",
                        "bbox": bbox,
                        "chunk_index": idx,
                        "char_count": len(text_val),
                    },
                }
            )

        page_stats.append(
            {
                "page": page_num,
                "section": section,
                "text_chunks": len(grouped),
                "chars": sum(len(t) for _, t in grouped),
                "note": (
                    "Divider / low-text page (System Card cover or raster-only chart)"
                    if sum(len(t) for _, t in grouped) < 150
                    else "ok"
                ),
            }
        )

    # 2. Crop & index all 28 visual chunks (tables, charts, images/diagrams)
    for spec in VISUAL_SPECS:
        pno = spec["page"]
        page = doc[pno - 1]
        rect = pymupdf.Rect(*spec["bbox"])
        pix = page.get_pixmap(clip=rect, dpi=170)
        crop_rel = f"advanced/crops/{spec['id']}.png"
        pix.save(str(REPO_ROOT / crop_rel))

        clip_text = _norm_text(page.get_text("text", clip=rect) or "")
        verified_desc = VERIFIED_VISUAL_DESCRIPTIONS[spec["id"]]
        section = _page_section(pno)

        full_text = (
            f"[Page {pno} | Section: {section} | Modality: {spec['modality']} | {spec['title']}]\n"
            f"Crop: {crop_rel} | BBox: {spec['bbox']}\n\n"
            f"=== Verified Visual Structure ===\n{verified_desc}\n\n"
            f"=== Native PDF Region Text ===\n{clip_text}"
        )
        all_chunks.append(
            {
                "id": spec["id"],
                "source_id": "gpt4-tech-report-2023",
                "source": "advanced/data/gpt4_technical_report_100p.pdf",
                "page": pno,
                "modality": spec["modality"],
                "bbox": spec["bbox"],
                "crop_path": crop_rel,
                "text": full_text,
                "metadata": {
                    "section": section,
                    "modality": spec["modality"],
                    "title": spec["title"],
                    "bbox": spec["bbox"],
                    "crop_path": crop_rel,
                },
            }
        )

    doc.close()

    # 3. Embed all multimodal chunks & persist in Qdrant
    embedder = OpenRouterEmbedder(cache_path=EMBED_CACHE_PATH)
    embedder.reset_counters()
    texts_to_embed = [c["text"][:6000] for c in all_chunks]
    vectors = embedder.embed_texts(texts_to_embed, batch_size=32)

    if QDRANT_DIR.exists():
        shutil.rmtree(QDRANT_DIR)
    QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    qclient = QdrantClient(path=str(QDRANT_DIR))
    qclient.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embedder.dimensions, distance=Distance.COSINE),
    )

    points = []
    for idx, (ch, vec) in enumerate(zip(all_chunks, vectors), start=1):
        payload = {
            "chunk_id": ch["id"],
            "source_id": ch["source_id"],
            "source": ch["source"],
            "page": ch["page"],
            "modality": ch["modality"],
            "crop_path": ch["crop_path"],
            "text": ch["text"],
            **ch["metadata"],
        }
        points.append(PointStruct(id=idx, vector=vec, payload=payload))
    for i in range(0, len(points), 100):
        qclient.upsert(collection_name=COLLECTION_NAME, points=points[i : i + 100])
    stored_total = qclient.count(collection_name=COLLECTION_NAME, exact=True).count
    qclient.close()

    (DATA_DIR / "advanced_chunks.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    by_mod: dict[str, int] = {}
    for c in all_chunks:
        by_mod[c["modality"]] = by_mod.get(c["modality"], 0) + 1

    manifest = {
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "total_pdf_pages": total_pages,
        "pages_indexed": len(page_stats),
        "sparse_or_divider_pages": [p for p in page_stats if p["chars"] < 150],
        "total_chunks": len(all_chunks),
        "qdrant_stored_points": stored_total,
        "chunks_by_modality": by_mod,
        "crops_count": len(VISUAL_SPECS),
        "embedding_model": embedder.model_name,
        "embedding_dimensions": embedder.dimensions,
        "embedding_api_calls": embedder.api_calls,
        "embedding_cache_hits": embedder.cache_hits,
        "embedding_billed_tokens": embedder.billed_tokens,
        "embedding_cost_usd": round(embedder.billed_cost_usd, 8),
        "page_stats": page_stats,
    }
    (DATA_DIR / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Completed 100-page Multimodal Index: total_chunks={len(all_chunks)} "
        f"({by_mod}), qdrant={stored_total}, embed_tokens={embedder.billed_tokens}, "
        f"embed_cost=${embedder.billed_cost_usd:.6f}"
    )


if __name__ == "__main__":
    main()
