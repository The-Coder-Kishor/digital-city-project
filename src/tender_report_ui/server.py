from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt

from . import db as dbmod
from .scanner import scan_reports_root


def _workspace_root() -> Path:
    # src/tender_report_ui/server.py -> workspace root
    return Path(__file__).resolve().parents[2]


def _default_paths() -> tuple[Path, Path]:
    root = _workspace_root()
    outputs = root / "outputs" / "tender_reports"
    db_path = outputs / "tender_reports_ui.sqlite3"
    return outputs, db_path


def _read_text(path: Path, *, max_chars: int = 250_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except OSError:
        return ""


def _safe_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _safe_json_any(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_vendor_name(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def _document_label(doc_no: int) -> str:
    if doc_no == 1:
        return "Tender details"
    if doc_no == 2:
        return "Search results"
    if doc_no == 3:
        return "Vendor details"
    return f"Document {doc_no}"


def _extract_tender_title_from_doc(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    overview = payload.get("tender_overview")
    if isinstance(overview, dict):
        return str(overview.get("tender_title") or "").strip()
    return ""


def _extract_vendor_name_from_doc(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    direct = str(payload.get("vendor_name") or "").strip()
    if direct:
        return direct
    award = payload.get("award_information")
    if isinstance(award, dict):
        awarded_to = str(award.get("awarded_to") or "").strip()
        if awarded_to:
            return awarded_to
    return ""


def _collect_tender_documents(tender_dir: Path, tender_id: str) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in sorted(tender_dir.glob(f"{tender_id}_document_*.json"), key=lambda p: p.name):
        m = re.search(r"_document_(\d+)\.json$", path.name)
        if not m:
            continue
        doc_no = int(m.group(1))
        payload = _safe_json_any(path)
        docs.append(
            {
                "doc_no": doc_no,
                "label": _document_label(doc_no),
                "file_name": path.name,
                "payload": payload,
                "tender_title": _extract_tender_title_from_doc(payload),
                "vendor_name": _extract_vendor_name_from_doc(payload),
            }
        )
    return docs


def _build_vendor_index(output_root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    if not output_root.exists():
        return index

    for tdir in output_root.iterdir():
        if not tdir.is_dir():
            continue
        tender_id = tdir.name
        doc3 = tdir / f"{tender_id}_document_3.json"
        doc1 = tdir / f"{tender_id}_document_1.json"
        if not doc3.exists() and not doc1.exists():
            continue

        vendor_payload = _safe_json_any(doc3) if doc3.exists() else None
        tender_payload = _safe_json_any(doc1) if doc1.exists() else None

        vendor_name = _extract_vendor_name_from_doc(vendor_payload)
        if not vendor_name:
            vendor_name = _extract_vendor_name_from_doc(tender_payload)
        if not vendor_name:
            continue

        vendor_key = _normalize_vendor_name(vendor_name)
        if not vendor_key:
            continue

        tender_title = _extract_tender_title_from_doc(tender_payload) or f"Tender {tender_id}"
        record = index.setdefault(
            vendor_key,
            {
                "vendor_name": vendor_name,
                "tender_ids": [],
                "tenders": [],
            },
        )
        if tender_id not in record["tender_ids"]:
            record["tender_ids"].append(tender_id)
            record["tenders"].append({"tender_id": tender_id, "tender_title": tender_title})

    for record in index.values():
        record["tender_ids"].sort(reverse=True)
        record["tenders"].sort(key=lambda x: x["tender_id"], reverse=True)
        record["tender_count"] = len(record["tender_ids"])
    return index


def _unique_by(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        v = str(it.get(key) or "").strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(it)
    return out


def _tender_dir(output_root: Path, tender_id: str) -> Path:
    return output_root / tender_id


def _uploads_dir(output_root: Path, tender_id: str) -> Path:
    return _tender_dir(output_root, tender_id) / "user_uploads"


def _user_sources_path(output_root: Path, tender_id: str) -> Path:
    return _tender_dir(output_root, tender_id) / "user_sources.json"


def _append_user_source(output_root: Path, tender_id: str, source: dict[str, Any]) -> None:
    path = _user_sources_path(output_root, tender_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                existing = [x for x in payload if isinstance(x, dict)]
        except Exception:
            existing = []
    existing.insert(0, source)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_pipeline(tender_id: str, *, output: str) -> tuple[int, str]:
    root = _workspace_root()
    cmd = [
        "uv",
        "run",
        "python",
        str(root / "scripts" / "agentic_tender_report_pipeline.py"),
        "--tender-id",
        tender_id,
        "--output",
        output,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root))
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, combined[-12000:]


def _is_valid_tender_id(value: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", (value or "").strip()))


def create_app() -> FastAPI:
    output_root, db_path = _default_paths()
    con = dbmod.connect(db_path)

    app = FastAPI(title="Tender Report UI", version="0.1.0")

    # Serve tender-specific uploads and artifacts.
    if output_root.exists():
        app.mount("/tfiles", StaticFiles(directory=str(output_root)), name="tfiles")

    static_dir = Path(__file__).resolve().parent / "static"
    templates_dir = Path(__file__).resolve().parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        cache_size=0,
    )
    md = MarkdownIt("commonmark", {"html": False, "linkify": True, "typographer": True}).enable(
        ["table", "strikethrough"]
    )

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def render(name: str, context: dict[str, Any]) -> HTMLResponse:
        tpl = env.get_template(name)
        return HTMLResponse(tpl.render(**context))

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, q: str = "") -> HTMLResponse:
        tenders = dbmod.list_tenders(con, q=q)
        vendor_index = _build_vendor_index(output_root)
        vendor_count = len(vendor_index)
        return render(
            "index.html",
            {
                "request": request,
                "q": q,
                "tenders": tenders,
                "vendor_count": vendor_count,
            },
        )

    @app.get("/vendors", response_class=HTMLResponse)
    def vendors_page(request: Request, q: str = "") -> HTMLResponse:
        q_norm = _normalize_vendor_name(q)
        vendor_index = _build_vendor_index(output_root)
        entries = list(vendor_index.values())
        if q_norm:
            entries = [e for e in entries if q_norm in _normalize_vendor_name(str(e.get("vendor_name") or ""))]
        entries.sort(key=lambda e: (-int(e.get("tender_count") or 0), str(e.get("vendor_name") or "")))
        return render(
            "vendors.html",
            {
                "request": request,
                "q": q,
                "vendors": entries,
            },
        )

    @app.get("/vendor/{vendor_name:path}", response_class=HTMLResponse)
    def vendor_detail_page(request: Request, vendor_name: str) -> HTMLResponse:
        vendor_index = _build_vendor_index(output_root)
        key = _normalize_vendor_name(vendor_name)
        entry = vendor_index.get(key)
        if not entry:
            raise HTTPException(status_code=404, detail="Vendor not found")
        return render(
            "vendor_detail.html",
            {
                "request": request,
                "vendor": entry,
            },
        )

    @app.post("/refresh")
    def refresh() -> RedirectResponse:
        rows = scan_reports_root(output_root)
        for r in rows:
            dbmod.upsert_tender(
                con,
                tender_id=str(r["tender_id"]),
                tender_name=str(r["tender_name"]),
                report_md_path=r["md_path"],
                report_json_path=r["json_path"],
                report_html_path=r["html_path"],
            )
        return RedirectResponse("/", status_code=303)

    @app.post("/run_new")
    def run_new_tender(
        tender_id: str = Form(...),
        output: str = Form("markdown"),
    ) -> RedirectResponse:
        tender_id = (tender_id or "").strip()
        if not _is_valid_tender_id(tender_id):
            raise HTTPException(status_code=400, detail="tender_id must be exactly 6 digits")
        if output not in ("markdown", "full"):
            raise HTTPException(status_code=400, detail="Invalid output mode")

        code, tail = _run_pipeline(tender_id, output=output)
        _append_user_source(
            output_root,
            tender_id,
            {
                "kind": "pipeline_run",
                "title": f"Pipeline run ({output})",
                "url": "",
                "notes": f"exit_code={code}\n\n{tail}",
                "added_at": int(time.time()),
            },
        )

        # Refresh DB (so Name-<id> becomes searchable immediately)
        rows = scan_reports_root(output_root)
        for r in rows:
            dbmod.upsert_tender(
                con,
                tender_id=str(r["tender_id"]),
                tender_name=str(r["tender_name"]),
                report_md_path=r["md_path"],
                report_json_path=r["json_path"],
                report_html_path=r["html_path"],
            )

        return RedirectResponse(f"/t/{tender_id}", status_code=303)

    @app.get("/t/{tender_id}", response_class=HTMLResponse)
    def tender_page(request: Request, tender_id: str) -> HTMLResponse:
        t = dbmod.get_tender(con, tender_id)
        if not t:
            # Lazy index for direct deep links when DB is stale.
            tdir = _tender_dir(output_root, tender_id)
            if not tdir.exists():
                raise HTTPException(status_code=404, detail="Tender not found")
            rows = scan_reports_root(output_root)
            for r in rows:
                if str(r["tender_id"]) != tender_id:
                    continue
                dbmod.upsert_tender(
                    con,
                    tender_id=str(r["tender_id"]),
                    tender_name=str(r["tender_name"]),
                    report_md_path=r["md_path"],
                    report_json_path=r["json_path"],
                    report_html_path=r["html_path"],
                )
                break
            t = dbmod.get_tender(con, tender_id)
            if not t:
                raise HTTPException(status_code=404, detail="Tender not indexed")

        tender_dir = _tender_dir(output_root, tender_id)
        documents = _collect_tender_documents(tender_dir, tender_id)

        current_vendor = ""
        for doc in documents:
            if int(doc.get("doc_no") or 0) == 3 and str(doc.get("vendor_name") or "").strip():
                current_vendor = str(doc.get("vendor_name") or "").strip()
                break
        if not current_vendor:
            for doc in documents:
                if str(doc.get("vendor_name") or "").strip():
                    current_vendor = str(doc.get("vendor_name") or "").strip()
                    break
        md_path = Path(t.report_md_path) if t.report_md_path else (tender_dir / f"{tender_id}_report.md")
        report_md = _read_text(md_path) if md_path.exists() else ""
        report_html = md.render(report_md) if report_md.strip() else ""

        js_path = Path(t.report_json_path) if t.report_json_path else (tender_dir / f"{tender_id}_report.json")
        report_json = _safe_json(js_path) if js_path.exists() else {}

        collection = _as_dict(report_json.get("collection"))
        portal_data = _as_dict(collection.get("portal_data"))
        endpoints = _as_dict(portal_data.get("endpoints"))
        download_status = _as_dict(collection.get("download_status"))
        source_dirs = _as_list(collection.get("source_dirs"))
        extraction_issues = _as_list(report_json.get("extraction_issues"))
        document_analysis = _as_dict(report_json.get("document_analysis"))
        project_intel = _as_dict(report_json.get("project_intel")) or _as_dict(document_analysis.get("project_intel"))

        # File sources
        sources_used = _as_list(document_analysis.get("sources_used"))

        # Web sources
        web_search_results = _as_list(report_json.get("web_search_results"))
        web_crawled_pages = _as_list(report_json.get("web_crawled_pages"))
        deepresearch = _as_dict(report_json.get("deepresearch"))
        deep_web_pages = _as_list(deepresearch.get("web_pages"))
        all_web_pages = _unique_by(
            [
                *[p for p in web_crawled_pages if isinstance(p, dict)],
                *[p for p in deep_web_pages if isinstance(p, dict)],
            ],
            "url",
        )

        # DeepResearch evidence
        dr_evidence = _as_dict(deepresearch.get("evidence"))
        dr_file_quotes = [q for q in _as_list(dr_evidence.get("file_quotes")) if isinstance(q, dict)]
        dr_web_quotes = [q for q in _as_list(dr_evidence.get("web_quotes")) if isinstance(q, dict)]
        dr_entities = _as_dict(dr_evidence.get("entities"))

        # Geotags
        geotags = [g for g in _as_list(report_json.get("geotags")) if isinstance(g, dict)]

        workflow = {
            "has_inspect": (tender_dir / "inspect").exists() or (Path(report_json.get("collection", {}).get("inspect_dir", "")) if report_json else Path("")).exists(),
            "has_extracted": any(tender_dir.glob("*_extracted")) or (Path(report_json.get("collection", {}).get("download_status", {}).get("extracted_dir", "")) if report_json else Path("")).exists(),
            "has_web": bool((report_json.get("web_crawled_pages") or report_json.get("deepresearch", {}).get("web_pages")) if isinstance(report_json, dict) else False),
            "has_geotags": bool(report_json.get("geotags")) if isinstance(report_json, dict) else False,
            "has_report": bool(report_md.strip()),
        }

        sources = dbmod.list_sources(con, tender_id)
        user_sources_path = _user_sources_path(output_root, tender_id)
        user_sources: list[dict[str, Any]] = []
        if user_sources_path.exists():
            try:
                payload = json.loads(user_sources_path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    user_sources = [x for x in payload if isinstance(x, dict)]
            except Exception:
                user_sources = []

        same_vendor_tenders: list[dict[str, str]] = []
        if current_vendor:
            vendor_index = _build_vendor_index(output_root)
            ventry = vendor_index.get(_normalize_vendor_name(current_vendor))
            if ventry:
                for t in ventry.get("tenders", []):
                    if str(t.get("tender_id") or "") == tender_id:
                        continue
                    same_vendor_tenders.append(
                        {
                            "tender_id": str(t.get("tender_id") or ""),
                            "tender_title": str(t.get("tender_title") or ""),
                        }
                    )

        return render(
            "tender.html",
            {
                "request": request,
                "tender": t,
                "workflow": workflow,
                "report_md": report_md,
                "report_html": report_html,
                "report_json": report_json,
                "collection": collection,
                "portal_data": portal_data,
                "endpoints": endpoints,
                "download_status": download_status,
                "source_dirs": source_dirs,
                "extraction_issues": extraction_issues,
                "project_intel": project_intel,
                "document_analysis": document_analysis,
                "sources_used": sources_used,
                "web_search_results": web_search_results,
                "all_web_pages": all_web_pages,
                "deepresearch": deepresearch,
                "dr_file_quotes": dr_file_quotes,
                "dr_web_quotes": dr_web_quotes,
                "dr_entities": dr_entities,
                "geotags": geotags,
                "sources": sources,
                "user_sources": user_sources,
                "documents": documents,
                "current_vendor": current_vendor,
                "same_vendor_tenders": same_vendor_tenders,
            },
        )

    @app.post("/t/{tender_id}/run")
    def run_pipeline(tender_id: str, output: str = Form("markdown")) -> RedirectResponse:
        if output not in ("markdown", "full"):
            raise HTTPException(status_code=400, detail="Invalid output mode")
        code, tail = _run_pipeline(tender_id, output=output)
        # Always refresh index after run attempt.
        _append_user_source(
            output_root,
            tender_id,
            {
                "kind": "pipeline_run",
                "title": f"Pipeline run ({output})",
                "url": "",
                "notes": f"exit_code={code}\n\n{tail}",
                "added_at": int(time.time()),
            },
        )
        return RedirectResponse(f"/t/{tender_id}", status_code=303)

    @app.post("/t/{tender_id}/sources/url")
    def add_url_source(
        tender_id: str,
        kind: str = Form(...),
        title: str = Form(""),
        url: str = Form(""),
        notes: str = Form(""),
    ) -> RedirectResponse:
        if not url.strip():
            raise HTTPException(status_code=400, detail="url is required")
        kind = (kind or "url").strip()[:40]
        title = (title or "").strip()[:200] or None
        url = url.strip()[:2000]
        notes = (notes or "").strip()[:6000] or None
        dbmod.add_source(con, tender_id=tender_id, kind=kind, title=title, url=url, notes=notes)
        _append_user_source(
            output_root,
            tender_id,
            {
                "kind": kind,
                "title": title or "",
                "url": url,
                "notes": notes or "",
                "added_at": int(time.time()),
            },
        )
        return RedirectResponse(f"/t/{tender_id}", status_code=303)

    @app.post("/t/{tender_id}/sources/upload")
    async def upload_source(
        tender_id: str,
        kind: str = Form("upload"),
        title: str = Form(""),
        notes: str = Form(""),
        file: UploadFile = File(...),
    ) -> RedirectResponse:
        kind = (kind or "upload").strip()[:40]
        title_v = (title or "").strip()[:200] or file.filename or "upload"
        notes_v = (notes or "").strip()[:6000]

        up_dir = _uploads_dir(output_root, tender_id)
        up_dir.mkdir(parents=True, exist_ok=True)
        safe_name = (file.filename or "upload").replace("\\", "_").replace("/", "_")
        stamp = int(time.time())
        dest = up_dir / f"{stamp}_{safe_name}"

        data = await file.read()
        dest.write_bytes(data)

        rel_path = str(dest.relative_to(_tender_dir(output_root, tender_id))).replace("\\", "/")
        dbmod.add_source(con, tender_id=tender_id, kind=kind, title=title_v, file_path=rel_path, notes=notes_v or None)
        _append_user_source(
            output_root,
            tender_id,
            {
                "kind": kind,
                "title": title_v,
                "file_path": rel_path,
                "notes": notes_v,
                "added_at": int(time.time()),
            },
        )
        return RedirectResponse(f"/t/{tender_id}", status_code=303)

    @app.get("/t/{tender_id}/file/{rel_path:path}")
    def serve_tender_file(tender_id: str, rel_path: str) -> RedirectResponse:
        # Use a redirect to static file handler? Keep simple: file:// isn't allowed.
        # Users can open the file from disk; this endpoint is mainly for uploaded assets.
        # We serve via StaticFiles mount under /tfiles.
        raise HTTPException(status_code=404, detail="Direct file serving not enabled")

    return app


app = create_app()

