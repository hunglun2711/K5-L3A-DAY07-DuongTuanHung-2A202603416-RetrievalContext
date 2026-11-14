"""Download/verify the 100-page GPT-4 Technical Report (arXiv:2303.08774) for Advanced Challenge."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import fitz
import httpx

SOURCE_URL = "https://arxiv.org/pdf/2303.08774.pdf"
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "advanced" / "data"
PDF_PATH = OUT_DIR / "gpt4_technical_report_100p.pdf"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_pdf = Path(tempfile.gettempdir()) / "gpt4_tech_report.pdf"

    if not PDF_PATH.exists():
        if tmp_pdf.exists() and tmp_pdf.stat().st_size > 1_000_000:
            shutil.copy2(tmp_pdf, PDF_PATH)
        else:
            print(f"Downloading {SOURCE_URL} ...")
            resp = httpx.get(SOURCE_URL, follow_redirects=True, timeout=120.0)
            resp.raise_for_status()
            PDF_PATH.write_bytes(resp.content)

    doc = fitz.open(str(PDF_PATH))
    page_count = len(doc)
    if page_count < 100:
        raise RuntimeError(f"Expected >= 100 pages, got {page_count}")

    sha256 = hashlib.sha256(PDF_PATH.read_bytes()).hexdigest()
    total_chars = sum(len(page.get_text("text") or "") for page in doc)
    doc.close()

    manifest = {
        "title": "GPT-4 Technical Report",
        "publisher": "OpenAI",
        "arxiv_id": "2303.08774v6",
        "source_url": SOURCE_URL,
        "abs_url": "https://arxiv.org/abs/2303.08774",
        "local_pdf_path": "advanced/data/gpt4_technical_report_100p.pdf",
        "access_and_license": (
            "Publicly accessible OpenAI technical report hosted on arXiv under arXiv's "
            "non-exclusive distribution license; excluded from Git via .gitignore and "
            "reproducible via python advanced/scripts/prepare_advanced_corpus.py."
        ),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "page_count": page_count,
        "pdf_bytes": PDF_PATH.stat().st_size,
        "sha256": sha256,
        "total_extracted_chars": total_chars,
    }
    (OUT_DIR / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verified 100-page PDF: {page_count} pages, {total_chars} chars, sha256={sha256[:12]}...")


if __name__ == "__main__":
    main()
