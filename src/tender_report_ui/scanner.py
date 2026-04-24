from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _safe_read_text(path: Path, *, max_chars: int) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except OSError:
        return ""


def extract_tender_name_from_markdown(md_path: Path) -> str:
    text = _safe_read_text(md_path, max_chars=6000)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            # Examples:
            # "# Tender Intelligence Report: XYZ"
            # "# Tender report: 000057"
            line = line.lstrip("#").strip()
            line = re.sub(r"^(Tender Intelligence Report|Tender report)\s*:\s*", "", line, flags=re.I).strip()
            return line or md_path.stem
        break
    return md_path.stem


def _safe_load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def extract_tender_name_from_json(report_json: Path) -> str:
    obj = _safe_load_json(report_json)
    pi = obj.get("project_intel") if isinstance(obj, dict) else None
    if isinstance(pi, dict):
        title = str(pi.get("title_or_work") or "").strip()
        if title:
            return title
    da = obj.get("document_analysis") if isinstance(obj, dict) else None
    if isinstance(da, dict):
        details = da.get("details")
        if isinstance(details, list) and details:
            first = details[0]
            if isinstance(first, str) and first.strip():
                return first.strip()[:120]
    return report_json.stem


def extract_tender_name_from_document_json(doc_json: Path) -> str:
    obj = _safe_load_json(doc_json)
    overview = obj.get("tender_overview") if isinstance(obj, dict) else None
    if isinstance(overview, dict):
        title = str(overview.get("tender_title") or "").strip()
        if title:
            return title
    return ""


def scan_reports_root(output_root: Path) -> list[dict[str, str | None]]:
    """
    Returns entries:
      { tender_id, tender_name, md_path, json_path, html_path }
    """
    out: list[dict[str, str | None]] = []
    if not output_root.exists():
        return out
    for tender_dir in sorted(output_root.iterdir(), key=lambda p: p.name):
        if not tender_dir.is_dir():
            continue
        tender_id = tender_dir.name
        md = tender_dir / f"{tender_id}_report.md"
        js = tender_dir / f"{tender_id}_report.json"
        html = tender_dir / f"{tender_id}_report.html"
        doc1 = tender_dir / f"{tender_id}_document_1.json"
        has_document_json = any(tender_dir.glob(f"{tender_id}_document_*.json"))
        has_bundle = (tender_dir / f"{tender_id}_pipeline_bundle.json").exists()
        if not md.exists() and not js.exists() and not html.exists() and not has_document_json and not has_bundle:
            continue

        tender_name = ""
        if js.exists():
            tender_name = extract_tender_name_from_json(js)
        if not tender_name and doc1.exists():
            tender_name = extract_tender_name_from_document_json(doc1)
        if not tender_name and md.exists():
            tender_name = extract_tender_name_from_markdown(md)
        if not tender_name:
            tender_name = f"Tender {tender_id}"

        out.append(
            {
                "tender_id": tender_id,
                "tender_name": tender_name,
                "md_path": str(md) if md.exists() else None,
                "json_path": str(js) if js.exists() else None,
                "html_path": str(html) if html.exists() else None,
            }
        )
    return out

