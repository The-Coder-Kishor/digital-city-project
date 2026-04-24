from __future__ import annotations

import argparse
import html as pyhtml
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from lxml import etree, html as lxml_html

from telangana_tenders import TelanganaTenderClient, parse_html_to_json, recursive_unzip
from telangana_tenders.client import validate_zip

TEXT_EXTENSIONS = {
    ".md",
    ".json",
}

SKIP_BINARY_EXTENSIONS = {
    ".zip",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".mp4",
    ".mp3",
}


@dataclass(slots=True)
class ExtractionIssue:
    file: str
    extension: str
    reason: str


@dataclass(slots=True)
class SearchResult:
    title: str
    href: str
    snippet: str
    source_query: str


class OpenRouterChatClient:
    def __init__(
        self,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key = (os.getenv("OPENROUTER_API_KEY") or "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").strip()
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required to call OpenRouter")
        self._session = requests.Session()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        num_ctx: int | None = None,
    ) -> str:
        _ = num_ctx
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        resp = self._session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        body = resp.json() or {}
        choices = body.get("choices") or []
        content: Any = None
        if isinstance(choices, list) and choices:
            content = ((choices[0] or {}).get("message") or {}).get("content")
        if isinstance(content, list):
            content = "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
        if not isinstance(content, str) or not content.strip():
            raise ValueError("OpenRouter returned empty content")
        return content


class DocumentReader:
    def __init__(self, *, max_files: int = 200, max_chars_per_file: int = 12000) -> None:
        self.max_files = max_files
        self.max_chars_per_file = max_chars_per_file

    def read_documents(self, source_dirs: list[Path]) -> tuple[list[dict[str, str]], list[ExtractionIssue]]:
        docs: list[dict[str, str]] = []
        issues: list[ExtractionIssue] = []

        files: list[Path] = []
        for source_dir in source_dirs:
            if source_dir.exists():
                files.extend([p for p in source_dir.rglob("*") if p.is_file()])

        files.sort(key=lambda p: str(p).lower())
        files = files[: self.max_files]

        for path in files:
            text, handled, reason = self._extract_text(path)
            normalized = self._normalize_text(text)

            if reason:
                issues.append(
                    ExtractionIssue(
                        file=str(path).replace("\\", "/"),
                        extension=path.suffix.lower(),
                        reason=reason,
                    )
                )

            if not handled or not normalized:
                continue

            docs.append(
                {
                    "path": str(path).replace("\\", "/"),
                    "text": normalized[: self.max_chars_per_file],
                }
            )

        return docs, issues

    def _extract_text(self, path: Path) -> tuple[str, bool, str | None]:
        suffix = path.suffix.lower()

        if suffix in TEXT_EXTENSIONS:
            try:
                return path.read_text(encoding="utf-8", errors="ignore"), True, None
            except OSError:
                return "", True, f"Failed to read text file: {path.name}"

        return "", False, f"Skipped unsupported extension: {suffix or '<none>'}"

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        if "<html" in text.lower() or "<table" in text.lower():
            try:
                tree = lxml_html.fromstring(text)
                text = tree.text_content()
            except Exception:
                pass
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())


class LocalNewsCrawler:
    def __init__(self, *, timeout_seconds: float = 15.0, max_workers: int = 6) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
                )
            }
        )

    def search(self, queries: list[str], *, max_results_per_query: int = 10) -> list[SearchResult]:
        DDGS = None
        try:
            from ddgs import DDGS as DDGSClient  # type: ignore

            DDGS = DDGSClient
        except Exception:
            try:
                # Backward compatibility for older environments.
                from duckduckgo_search import DDGS as DDGSClient  # type: ignore

                DDGS = DDGSClient
            except Exception as exc:
                raise RuntimeError(
                    "ddgs is required for local web crawling. Install dependencies with `uv sync`."
                ) from exc

        assert DDGS is not None

        def _safe_text_search(ddgs_client: Any, query: str) -> list[dict[str, Any]]:
            try:
                news_rows = []
                try:
                    news_rows = ddgs_client.news(query, max_results=max_results_per_query) or []
                except Exception:
                    pass
                
                text_rows = []
                try:
                    text_rows = ddgs_client.text(query, max_results=max_results_per_query, timelimit="m") or []
                    if not text_rows:
                        text_rows = ddgs_client.text(query, max_results=max_results_per_query) or []
                except Exception:
                    try:
                        text_rows = ddgs_client.text(query, max_results=max_results_per_query) or []
                    except Exception:
                        pass
                        
                combined = []
                for r in news_rows:
                    if isinstance(r, dict): combined.append(r)
                for r in text_rows:
                    if isinstance(r, dict): combined.append(r)
                return combined
            except Exception:
                return []

        seen: set[str] = set()
        results: list[SearchResult] = []
        with DDGS() as ddgs:
            for query in queries:
                rows = _safe_text_search(ddgs, query)
                for row in rows:
                    href = str(row.get("href") or row.get("url") or "").strip()
                    if not href or href in seen:
                        continue
                    seen.add(href)
                    results.append(
                        SearchResult(
                            title=str(row.get("title") or "").strip(),
                            href=href,
                            snippet=str(row.get("body") or "").strip(),
                            source_query=query,
                        )
                    )

        # Fallback broadening when highly specific queries return no usable results.
        if not results:
            fallback_queries = []
            for query in queries:
                q = query.replace('"', "").strip()
                if q:
                    fallback_queries.append(q)
            with DDGS() as ddgs:
                for query in fallback_queries[:6]:
                    rows = _safe_text_search(ddgs, query)
                    for row in rows:
                        href = str(row.get("href") or row.get("url") or "").strip()
                        if not href or href in seen:
                            continue
                        seen.add(href)
                        results.append(
                            SearchResult(
                                title=str(row.get("title") or "").strip(),
                                href=href,
                                snippet=str(row.get("body") or "").strip(),
                                source_query=query,
                            )
                        )
        return results

    def crawl(self, results: list[SearchResult], *, max_pages: int = 10, max_chars: int = 14000) -> list[dict[str, str]]:
        selected = results[:max_pages]
        pages: list[dict[str, str]] = []

        def fetch_one(item: SearchResult) -> dict[str, str]:
            try:
                resp = self.session.get(item.href, timeout=self.timeout_seconds)
                resp.raise_for_status()
                content_type = (resp.headers.get("content-type") or "").lower()
                if "text/html" not in content_type and "xml" not in content_type:
                    return {
                        "url": item.href,
                        "title": item.title,
                        "query": item.source_query,
                        "snippet": item.snippet,
                        "content": "",
                        "error": f"Skipped non-HTML content type: {content_type}",
                    }

                tree = lxml_html.fromstring(resp.text)
                for bad in tree.xpath("//script|//style|//noscript"):
                    bad.getparent().remove(bad)
                text = " ".join(tree.text_content().split())
                return {
                    "url": item.href,
                    "title": item.title,
                    "query": item.source_query,
                    "snippet": item.snippet,
                    "content": text[:max_chars],
                    "error": "",
                }
            except Exception as exc:
                return {
                    "url": item.href,
                    "title": item.title,
                    "query": item.source_query,
                    "snippet": item.snippet,
                    "content": "",
                    "error": f"crawl failed: {exc}",
                }

        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as executor:
            futures = [executor.submit(fetch_one, item) for item in selected]
            for future in as_completed(futures):
                pages.append(future.result())

        return pages


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z0-9_\-]*", "", candidate).strip()
        if candidate.endswith("```"):
            candidate = candidate[:-3].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_local_source_collection(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"source root is not a directory: {source_root}")

    portal_data: dict[str, Any] = {
        "mode": "local_root",
        "source_root": str(source_root),
        "endpoints": {},
        "bootstrap_error": "",
        "local_scan": {
            "file_count": sum(1 for p in source_root.rglob("*") if p.is_file()),
        },
    }

    return {
        "tender_dir": str(source_root),
        "inspect_dir": str(source_root),
        "portal_data": portal_data,
        "download_status": {
            "zip_path": "",
            "extracted_dir": str(source_root),
            "downloaded": False,
            "source_root": str(source_root),
        },
        "source_dirs": [str(source_root)],
    }


def analyze_documents(
    llm: OpenRouterChatClient,
    tender_id: str,
    docs: list[dict[str, str]],
    portal_data: dict[str, Any],
) -> dict[str, Any]:
    manifest = [{"path": d["path"], "chars": len(d["text"])} for d in docs]
    snippets = [{"path": d["path"], "snippet": d["text"][:3000]} for d in docs[:22]]

    doc1_prompt = """
You are a tender-analysis assistant designed to help ordinary citizens understand government tenders.

Your job is to read a tender document and extract the most important information in a citizen-friendly way. Focus on public impact, money, timing, location, accountability, and practical relevance. Avoid procurement jargon where possible.

For every tender, produce output with the following sections:

Tender overview
Tender title
Tender ID / reference number
Issuing authority / department
Publication date
Closing date and time
Status: open, upcoming, amended, cancelled, awarded, or unknown

What is being procured
Plain-English summary of the project
Category: construction, road, IT, health, education, goods, services, consultancy, etc.
Scope of work
Deliverables / outputs
Quantity or scale
Location(s) affected
Whether it is a one-time project or ongoing contract

Why citizens should care
Public benefit or service impact
Who benefits directly
Possible disruptions or inconveniences
Whether it affects daily life, taxes, transport, health, schools, utilities, jobs, or the environment

Money and timeline
Estimated contract value / budget
Funding source, if stated
Expected duration
Start and end dates, if available

Award information
Who the contract was awarded to
Awarded amount / contract value at award, if available
Date of award, if available
If not yet awarded, explicitly state "not stated"

Bidding and eligibility
Who can bid
Required qualifications, experience, or certifications
Bid security / deposit / earnest money
Pre-bid meeting date, if any
Site visit requirement, if any
Submission method and deadline
Selection process and evaluation criteria, if stated

Risks, watch-outs, and accountability
Key deadlines
Unusual or restrictive conditions
Short bidding windows
Vague scope
Single-bid risk
Amendments or extensions
Complaint / grievance channel
Link or reference to original documents, if present

Citizen-friendly summary
Write 3–10 short bullet points explaining the tender in plain language for a general audience. Ensure that it is a balanced summary giving the pros and cons of the development project while keeping in mind the benefit of the citizens.

Red flags
List any potentially concerning issues, such as:
unusually short deadline
unclear requirements
narrow eligibility
repeated extensions
missing budget information
excessive restrictions
high-value contract with low transparency

Rules:
Be accurate and only use information supported by the tender text.
If a field is missing, write “not stated.”
Translate technical language into plain English.
Keep the summary concise and useful for non-experts.
If the tender contains numbers, dates, places, or eligibility rules, extract them exactly.
Do not speculate. If something is unclear, say so.
Source hierarchy: Primary NIT page > tender tables > annexures/instructions/repeated excerpts.
If sources disagree, report disagreement instead of forcing one value.
Internal consistency checks before finalizing: estimated value vs EMD, submission dates across sections, duration consistency, mandatory docs consistency, authority/location/completion contradictions. Ensure that all information is accurate and complete.

Return JSON matching this schema:

{
"type": "object",
"additionalProperties": false,
"required": [
"tender_overview",
"what_is_being_procured",
"why_citizens_should_care",
"money_and_timeline",
"bidding_and_eligibility",
"award_information",
"risks_watchouts_and_accountability",
"citizen_friendly_summary",
"red_flags"
],
"properties": {
"tender_overview": {
"type": "object",
"additionalProperties": false,
"required": [
"tender_title",
"tender_id",
"issuing_authority",
"publication_date",
"closing_date_time",
"status"
],
"properties": {
"tender_title": { "type": "string" },
"tender_id": { "type": "string" },
"issuing_authority": { "type": "string" },
"publication_date": { "type": "string" },
"closing_date_time": { "type": "string" },
"status": {
"type": "string",
"enum": ["open", "upcoming", "amended", "cancelled", "awarded", "unknown"]
}
}
},

"what_is_being_procured": {
"type": "object",
"additionalProperties": false,
"required": [
"plain_english_summary",
"category",
"scope_of_work",
"deliverables",
"quantity_or_scale",
"locations_affected",
"contract_type"
],
"properties": {
"plain_english_summary": { "type": "string" },
"category": { "type": "string" },
"scope_of_work": { "type": "string" },
"deliverables": { "type": "string" },
"quantity_or_scale": { "type": "string" },
"locations_affected": { "type": "string" },
"contract_type": { "type": "string" }
}
},

"why_citizens_should_care": {
"type": "object",
"additionalProperties": false,
"required": [
"public_benefit_or_service_impact",
"direct_beneficiaries",
"possible_disruptions",
"daily_life_impact"
],
"properties": {
"public_benefit_or_service_impact": { "type": "string" },
"direct_beneficiaries": { "type": "string" },
"possible_disruptions": { "type": "string" },
"daily_life_impact": { "type": "string" }
}
},

"money_and_timeline": {
"type": "object",
"additionalProperties": false,
"required": [
"estimated_contract_value",
"funding_source",
"expected_duration",
"start_date",
"end_date"
],
"properties": {
"estimated_contract_value": { "type": "string" },
"funding_source": { "type": "string" },
"expected_duration": { "type": "string" },
"start_date": { "type": "string" },
"end_date": { "type": "string" }
}
},

"bidding_and_eligibility": {
"type": "object",
"additionalProperties": false,
"required": [
"who_can_bid",
"required_qualifications",
"bid_security",
"pre_bid_meeting",
"site_visit_requirement",
"submission_method",
"selection_process",
"evaluation_criteria"
],
"properties": {
"who_can_bid": { "type": "string" },
"required_qualifications": { "type": "string" },
"bid_security": { "type": "string" },
"pre_bid_meeting": { "type": "string" },
"site_visit_requirement": { "type": "string" },
"submission_method": { "type": "string" },
"selection_process": { "type": "string" },
"evaluation_criteria": { "type": "string" }
}
},

"award_information": {
"type": "object",
"additionalProperties": false,
"required": [
"awarded_to",
"awarded_amount",
"award_date"
],
"properties": {
"awarded_to": { "type": "string" },
"awarded_amount": { "type": "string" },
"award_date": { "type": "string" }
}
},

"risks_watchouts_and_accountability": {
"type": "object",
"additionalProperties": false,
"required": [
"key_deadlines",
"unusual_or_restrictive_conditions",
"short_bidding_window",
"vague_scope",
"single_bid_risk",
"amendments_or_extensions",
"grievance_channel",
"original_document_reference"
],
"properties": {
"key_deadlines": { "type": "string" },
"unusual_or_restrictive_conditions": { "type": "string" },
"short_bidding_window": { "type": "string" },
"vague_scope": { "type": "string" },
"single_bid_risk": { "type": "string" },
"amendments_or_extensions": { "type": "string" },
"grievance_channel": { "type": "string" },
"misc": { "type": "string" },
"original_document_reference": { "type": "string" }
}
},

"citizen_friendly_summary": {
"type": "array",
"items": { "type": "string" },
"minItems": 3,
"maxItems": 10
},

"red_flags": {
"type": "array",
"items": { "type": "string" }
}

}
}
""".strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a tender-analysis assistant for government procurement documents. "
                "Use only directly supported facts. Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Tender ID: {tender_id}\n\n"
                f"{doc1_prompt}\n\n"
                f"Portal data summary:\n{json.dumps(portal_data, ensure_ascii=True)[:22000]}\n\n"
                f"Document manifest:\n{json.dumps(manifest, ensure_ascii=True)}\n\n"
                f"Document snippets:\n{json.dumps(snippets, ensure_ascii=True)}"
            ),
        },
    ]

    try:
        output = llm.chat(messages, temperature=0.0, max_tokens=3200, num_ctx=32768)
        payload = extract_json_object(output)
        if not isinstance(payload, dict):
            raise ValueError("document analysis returned non-object JSON")
    except Exception as exc:
        payload = {
            "tender_overview": {
                "tender_title": "not stated",
                "tender_id": tender_id,
                "issuing_authority": "not stated",
                "publication_date": "not stated",
                "closing_date_time": "not stated",
                "status": "unknown",
            },
            "what_is_being_procured": {
                "plain_english_summary": "not stated",
                "category": "not stated",
                "scope_of_work": "not stated",
                "deliverables": "not stated",
                "quantity_or_scale": "not stated",
                "locations_affected": "not stated",
                "contract_type": "not stated",
            },
            "why_citizens_should_care": {
                "public_benefit_or_service_impact": "not stated",
                "direct_beneficiaries": "not stated",
                "possible_disruptions": "not stated",
                "daily_life_impact": "not stated",
            },
            "money_and_timeline": {
                "estimated_contract_value": "not stated",
                "funding_source": "not stated",
                "expected_duration": "not stated",
                "start_date": "not stated",
                "end_date": "not stated",
            },
            "bidding_and_eligibility": {
                "who_can_bid": "not stated",
                "required_qualifications": "not stated",
                "bid_security": "not stated",
                "pre_bid_meeting": "not stated",
                "site_visit_requirement": "not stated",
                "submission_method": "not stated",
                "selection_process": "not stated",
                "evaluation_criteria": "not stated",
            },
            "award_information": {
                "awarded_to": "not stated",
                "awarded_amount": "not stated",
                "award_date": "not stated",
            },
            "risks_watchouts_and_accountability": {
                "key_deadlines": "not stated",
                "unusual_or_restrictive_conditions": "not stated",
                "short_bidding_window": "not stated",
                "vague_scope": "not stated",
                "single_bid_risk": "not stated",
                "amendments_or_extensions": "not stated",
                "grievance_channel": "not stated",
                "misc": f"Document analysis failed: {exc}",
                "original_document_reference": "not stated",
            },
            "citizen_friendly_summary": [
                "Tender extraction failed; key values are marked as not stated.",
                "Verify source tender files and rerun the pipeline.",
                "Use official tender portal details for final validation.",
            ],
            "red_flags": [f"Document analysis failed: {exc}"],
        }

    payload.setdefault("tender_overview", {})
    if isinstance(payload.get("tender_overview"), dict):
        payload["tender_overview"].setdefault("tender_id", tender_id)

    inferred_award = infer_award_information_from_docs(docs)
    award_info = payload.get("award_information")
    if not isinstance(award_info, dict):
        award_info = {
            "awarded_to": "not stated",
            "awarded_amount": "not stated",
            "award_date": "not stated",
        }
    for key in ("awarded_to", "awarded_amount", "award_date"):
        cur = str(award_info.get(key) or "").strip().lower()
        if not cur or cur in {"not stated", "unknown", "unclear", "not found"}:
            award_info[key] = inferred_award.get(key, "not stated")
    payload["award_information"] = award_info

    return payload


def _safe_get(d: dict[str, Any], path: list[str], default: str = "") -> str:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return str(cur).strip() if cur is not None else default


def infer_award_information_from_docs(docs: list[dict[str, str]]) -> dict[str, str]:
    merged = "\n".join((d.get("text") or "")[:8000] for d in docs[:40])
    compact = " ".join(merged.split())

    awarded_to = "not stated"
    awarded_amount = "not stated"
    award_date = "not stated"

    awarded_to_patterns = [
        r"(?:awarded to|awarded in favour of|work awarded to|successful bidder|l1 bidder)\s*[:\-]?\s*([A-Za-z0-9&.,()'\-/ ]{3,140})",
    ]
    for pattern in awarded_to_patterns:
        m = re.search(pattern, compact, flags=re.IGNORECASE)
        if m:
            candidate = re.split(r"(?:\.|;|,\s+at\s+|,\s+for\s+)", m.group(1).strip())[0].strip(" -:")
            if len(candidate) >= 3:
                awarded_to = candidate
                break

    amount_patterns = [
        r"(?:award(?:ed)? amount|contract value|accepted value|awarded value|work value)\s*[:\-]?\s*(Rs\.?\s*[\d,]+(?:\.\d+)?(?:\s*(?:crore|lakh))?)",
        r"\b(Rs\.?\s*[\d,]{4,}(?:\.\d+)?(?:\s*(?:crore|lakh))?)\b",
    ]
    for pattern in amount_patterns:
        m = re.search(pattern, compact, flags=re.IGNORECASE)
        if m:
            awarded_amount = m.group(1).strip()
            break

    date_patterns = [
        r"(?:date of award|awarded on|work order date|loa date|letter of acceptance date)\s*[:\-]?\s*([0-3]?\d[\-/][01]?\d[\-/]\d{2,4})",
        r"(?:date of award|awarded on|work order date|loa date|letter of acceptance date)\s*[:\-]?\s*([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})",
    ]
    for pattern in date_patterns:
        m = re.search(pattern, compact, flags=re.IGNORECASE)
        if m:
            award_date = m.group(1).strip()
            break

    return {
        "awarded_to": awarded_to,
        "awarded_amount": awarded_amount,
        "award_date": award_date,
    }


def extract_identifiers_from_text(text: str) -> dict[str, list[str]]:
    content = (text or "").upper()
    gstin_re = r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b"
    cin_re = r"\b[L|U]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b"
    pan_re = r"\b[A-Z]{5}\d{4}[A-Z]\b"

    gstins = sorted(set(re.findall(gstin_re, content)))
    cins = sorted(set(re.findall(cin_re, content)))
    pans = sorted(set(re.findall(pan_re, content)))
    return {"gstin": gstins, "cin": cins, "pan": pans}


def merge_identifier_map(base: dict[str, Any], extra: dict[str, list[str]]) -> dict[str, list[str]]:
    out = {
        "gstin": [],
        "cin": [],
        "pan": [],
    }
    for key in out:
        base_values = base.get(key) if isinstance(base, dict) else []
        merged = []
        seen: set[str] = set()
        for value in list(base_values or []) + list(extra.get(key) or []):
            v = str(value).strip().upper()
            if not v or v in seen:
                continue
            seen.add(v)
            merged.append(v)
        out[key] = merged
    return out


def fetch_gst_lookup_data(entity_name: str, *, timeout_seconds: float = 30.0) -> dict[str, Any]:
    lookup_url = "https://www.knowyourgst.com/gst-number-search/by-name-pan/"
    clean_name = re.sub(r"\s+", " ", (entity_name or "")).strip()
    if not clean_name:
        return {
            "lookup_url": lookup_url,
            "entity_query": "",
            "http_status": None,
            "matched_rows": [],
            "raw_text_excerpt": "",
            "error": "Empty entity name for GST lookup",
        }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    try:
        response = requests.post(
            lookup_url,
            data={"gstnum": clean_name},
            headers=headers,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except Exception as exc:
        return {
            "lookup_url": lookup_url,
            "entity_query": clean_name,
            "http_status": None,
            "matched_rows": [],
            "raw_text_excerpt": "",
            "error": f"GST lookup failed: {exc}",
        }

    matched_rows: list[dict[str, str]] = []
    raw_text_excerpt = ""
    parse_error = ""
    try:
        doc = lxml_html.fromstring(response.text)
        for tr in doc.xpath("//table//tr")[:80]:
            cells = [" ".join((td.text_content() or "").split()) for td in tr.xpath("./th|./td")]
            cells = [c for c in cells if c]
            if not cells:
                continue
            row: dict[str, str]
            if len(cells) >= 2:
                row = {"field": cells[0], "value": " | ".join(cells[1:])}
            else:
                row = {"field": "row", "value": cells[0]}
            matched_rows.append(row)
            if len(matched_rows) >= 80:
                break

        raw_text = " ".join((doc.text_content() or "").split())
        raw_text_excerpt = raw_text[:5000]
    except Exception as exc:
        parse_error = f"GST HTML parse failed: {exc}"

    return {
        "lookup_url": lookup_url,
        "entity_query": clean_name,
        "http_status": response.status_code,
        "matched_rows": matched_rows,
        "raw_text_excerpt": raw_text_excerpt,
        "error": parse_error,
    }


def derive_step2_queries(llm: OpenRouterChatClient, tender_id: str, document1: dict[str, Any], *, max_queries: int = 15) -> list[str]:
    summary_old = document1.get("tender_summary") if isinstance(document1.get("tender_summary"), dict) else {}
    scope_old = document1.get("scope") if isinstance(document1.get("scope"), dict) else {}
    overview_new = document1.get("tender_overview") if isinstance(document1.get("tender_overview"), dict) else {}
    procured_new = document1.get("what_is_being_procured") if isinstance(document1.get("what_is_being_procured"), dict) else {}

    title = _safe_get(summary_old, ["title"]) or _safe_get(overview_new, ["tender_title"])
    department = _safe_get(summary_old, ["department"]) or _safe_get(overview_new, ["issuing_authority"])
    notice_number = _safe_get(summary_old, ["notice_number"]) or _safe_get(overview_new, ["tender_id"])
    category = _safe_get(summary_old, ["category"]) or _safe_get(procured_new, ["category"])
    location = _safe_get(scope_old, ["location"]) or _safe_get(procured_new, ["locations_affected"])

    try:
        ctx_str = f"Title: {title}\nDepartment: {department}\nLocation: {location}\nCategory: {category}"
        msgs = [
            {"role": "system", "content": "You are an OSINT researcher. Generate a list of precise DuckDuckGo search queries (strings) to find news about the following tender. Output ONLY a valid JSON array of strings."},
            {"role": "user", "content": f"Tender context:\n{ctx_str}\n\nTender ID: {tender_id}\nTarget topics: Construction delays, civic complaints, contractor scams, status updates.\nGenerate up to {max_queries} varied and specific queries. Return JSON array of strings ONLY."}
        ]
        resp = llm.chat(msgs, max_tokens=1000)
        resp = resp.strip()
        if resp.startswith("```json"): resp = resp[7:]
        if resp.startswith("```"): resp = resp[3:]
        if resp.endswith("```"): resp = resp[:-3]
        parsed = json.loads(resp.strip())
        if isinstance(parsed, list):
            return [str(x) for x in parsed[:max_queries]]
    except Exception:
        pass

    awards = document1.get("award_information", {})
    if not isinstance(awards, dict): awards = {}
    awarded_entity = str(awards.get("awarded_to") or "").replace("not stated", "").strip()

    words = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', title) if w.lower() not in {"this", "that", "with", "from", "under", "period", "months", "days", "work", "maintenance"}]
    title_keywords = " ".join(words[:5])

    dept_short = " ".join(re.findall(r'\b[a-zA-Z]{3,}\b', department)[:3])
    loc_short = " ".join(re.findall(r'\b[a-zA-Z]{3,}\b', location)[:3])

    seeds = [
        f'"{tender_id}" {dept_short} tender',
        f'{title_keywords} {loc_short} construction delay OR issue OR problem',
        f'{title_keywords} {loc_short} civic complain OR news',
        f'{loc_short} {category} {dept_short} contractor scam OR blacklist OR irregularities',
    ]

    if len(awarded_entity) > 3 and awarded_entity.lower() != "unknown":
        comp_name = " ".join(re.findall(r'\b[a-zA-Z0-9]{3,}\b', awarded_entity)[:4])
        if comp_name:
            seeds.append(f'"{comp_name}" contractor reviews OR scam OR issue')
            seeds.append(f'"{comp_name}" {dept_short} contract')

    seeds.append(f'site:tender.telangana.gov.in {tender_id}')

    seen: set[str] = set()
    out: list[str] = []
    for q in seeds:
        qq = re.sub(r"\s+", " ", q).strip()
        if not qq or len(qq) < 5 or qq in seen:
            continue
        seen.add(qq)
        out.append(qq)
        if len(out) >= max_queries:
            break
    return out


def build_document2(
    llm: OpenRouterChatClient,
    tender_id: str,
    document1: dict[str, Any],
    crawler: LocalNewsCrawler,
    *,
    search_results_per_query: int,
    crawl_max_pages: int,
    crawl_max_chars: int,
) -> tuple[dict[str, Any], list[SearchResult], list[dict[str, str]]]:
    queries = derive_step2_queries(llm, tender_id, document1)
    results = crawler.search(queries, max_results_per_query=search_results_per_query)
    pages = crawler.crawl(results, max_pages=max(1, crawl_max_pages), max_chars=max(2000, crawl_max_chars))

    compact_pages = [
        {
            "url": page.get("url", ""),
            "title": page.get("title", ""),
            "query": page.get("query", ""),
            "snippet": page.get("snippet", ""),
            "content": (page.get("content", "") or "")[:3000],
            "error": page.get("error", ""),
        }
        for page in pages
    ]

    messages = [
        {"role": "system", "content": "You are a procurement news researcher. Respond in English only and return strict JSON only."},
        {
            "role": "user",
            "content": (
                f"Tender ID: {tender_id}\n"
                "Review the tender data and crawled news pages.\n"
                "Identify and summarize the news that is relevant to this tender (work type, department, location, timeline, vendors).\n"
                "Ignore irrelevant pages.\n"
                "List the sources used.\n\n"
                "Important constraints:\n"
                "1) Output must be in English only.\n"
                "2) Use only the provided crawled pages as source evidence.\n"
                "3) If nothing relevant is found, say so clearly in English and keep sources empty.\n\n"
                "Return strict JSON matching this schema:\n"
                "{\n"
                "  \"overall_summary\": \"High level overview of news and public information retrieved related to the project context.\",\n"
                "  \"tender_specific_news\": \"Any mentions of this specific tender ID, process, delays, or status directly.\",\n"
                "  \"market_and_location_context\": \"Context on the contracting department, vendor, geographic project location, or similar historical projects from the news.\",\n"
                "  \"sources\": [{\"title\":\"\",\"url\":\"\",\"relevance\":\"high|medium|low\"}]\n"
                "}\n\n"
                f"Document 1 (Tender Data):\n{json.dumps(document1, ensure_ascii=True)[:22000]}\n\n"
                f"Crawled News Pages:\n{json.dumps(compact_pages, ensure_ascii=True)[:120000]}"
            ),
        },
    ]

    def _normalize_payload(raw: dict[str, Any]) -> dict[str, Any]:
        overall = raw.get("overall_summary")
        if not isinstance(overall, str) or not overall.strip():
            overall = "No clearly relevant overall news could be confirmed from crawled sources."
            
        specific = raw.get("tender_specific_news", "No specific news found.")
        market = raw.get("market_and_location_context", "No market context found.")

        normalized_sources: list[dict[str, str]] = []
        sources = raw.get("sources")
        if isinstance(sources, list):
            for src in sources:
                if not isinstance(src, dict):
                    continue
                title = str(src.get("title") or "").strip()
                url = str(src.get("url") or "").strip()
                relevance = str(src.get("relevance") or "low").strip().lower()
                if relevance not in {"high", "medium", "low"}:
                    relevance = "low"
                if not title and not url:
                    continue
                normalized_sources.append({"title": title, "url": url, "relevance": relevance})
                if len(normalized_sources) >= 20:
                    break

        return {
            "overall_summary": overall.strip(),
            "tender_specific_news": specific if isinstance(specific, str) else str(specific),
            "market_and_location_context": market if isinstance(market, str) else str(market),
            "sources": normalized_sources
        }

    def _summary_unusable(summary: str) -> bool:
        s = (summary or "").strip()
        if not s:
            return True
        if re.search(r"[\u4e00-\u9fff]", s):
            return True
        lowered = s.lower()
        blocked_phrases = [
            "unable to provide",
            "cannot provide",
            "i cannot",
            "\u65e0\u6cd5",
            "\u4f60\u597d",
        ]
        if any(p in lowered for p in blocked_phrases):
            return True
        return False

    def _keyword_tokens() -> list[str]:
        summary_old = document1.get("tender_summary") if isinstance(document1.get("tender_summary"), dict) else {}
        scope_old = document1.get("scope") if isinstance(document1.get("scope"), dict) else {}
        overview_new = document1.get("tender_overview") if isinstance(document1.get("tender_overview"), dict) else {}
        procured_new = document1.get("what_is_being_procured") if isinstance(document1.get("what_is_being_procured"), dict) else {}
        raw_parts = [
            tender_id,
            _safe_get(summary_old, ["title"]) or _safe_get(overview_new, ["tender_title"]),
            _safe_get(summary_old, ["work_name"]) or _safe_get(procured_new, ["plain_english_summary"]),
            _safe_get(summary_old, ["department"]) or _safe_get(overview_new, ["issuing_authority"]),
            _safe_get(summary_old, ["notice_number"]) or _safe_get(overview_new, ["tender_id"]),
            _safe_get(scope_old, ["location"]) or _safe_get(procured_new, ["locations_affected"]),
        ]
        tokens: list[str] = []
        seen: set[str] = set()
        for part in raw_parts:
            for tok in re.split(r"[^a-zA-Z0-9]+", (part or "").lower()):
                if len(tok) < 3:
                    continue
                if tok in seen:
                    continue
                seen.add(tok)
                tokens.append(tok)
        return tokens

    def _deterministic_fallback() -> dict[str, Any]:
        tokens = _keyword_tokens()
        scored: list[tuple[int, dict[str, str]]] = []
        for page in compact_pages:
            text = " ".join(
                [
                    str(page.get("title") or ""),
                    str(page.get("snippet") or ""),
                    str(page.get("query") or ""),
                    str(page.get("content") or ""),
                ]
            ).lower()
            score = 0
            for tok in tokens:
                if tok in text:
                    score += 1
            source = {
                "title": str(page.get("title") or "").strip(),
                "url": str(page.get("url") or "").strip(),
                "relevance": "high" if score >= 3 else ("medium" if score >= 1 else "low"),
            }
            scored.append((score, source))

        scored.sort(key=lambda item: item[0], reverse=True)
        picked = [src for score, src in scored if score > 0][:10]
        if not picked:
            picked = [{"title": r.title, "url": r.href, "relevance": "low"} for r in results[:10]]

        high = sum(1 for s in picked if s["relevance"] == "high")
        medium = sum(1 for s in picked if s["relevance"] == "medium")
        summary_text = (
            f"Relevant tender news summary generated from crawled pages. "
            f"High relevance sources: {high}, medium relevance sources: {medium}, total cited sources: {len(picked)}."
        )
        return {
            "overall_summary": summary_text,
            "tender_specific_news": "No specific news identified.",
            "market_and_location_context": "No market context identified.",
            "sources": picked
        }

    raw_response = ""
    payload: dict[str, Any]
    try:
        raw_response = llm.chat(messages, temperature=0.0, max_tokens=2500, num_ctx=32768)
        payload = _normalize_payload(extract_json_object(raw_response))
        if _summary_unusable(payload.get("overall_summary", "")):
            raise ValueError("document 2 summary unusable")
    except Exception:
        try:
            source_candidates = [
                {"title": str(p.get("title") or ""), "url": str(p.get("url") or "")}
                for p in compact_pages[:20]
            ]
            repair_messages = [
                {"role": "system", "content": "Convert the input into strict JSON. Return JSON only."},
                {
                    "role": "user",
                    "content": (
                        "Convert this model output to strict JSON with schema:\n"
                        '{"overall_summary":"", "tender_specific_news":"", "market_and_location_context":"", "sources":[{"title":"","url":"","relevance":"high|medium|low"}]}.\n'
                        "If information is missing, infer minimally and keep it concise.\n\n"
                        f"Model output:\n{raw_response[:12000]}\n\n"
                        f"Source candidates:\n{json.dumps(source_candidates, ensure_ascii=True)}"
                    ),
                },
            ]
            repaired = llm.chat(repair_messages, temperature=0.0, max_tokens=1000, num_ctx=16384)
            payload = _normalize_payload(extract_json_object(repaired))
            if _summary_unusable(payload.get("overall_summary", "")):
                raise ValueError("document 2 repaired summary unusable")
        except Exception:
            payload = _deterministic_fallback()

    if not payload.get("sources"):
        payload = _deterministic_fallback()

    return payload, results, pages


def build_document3(
    llm: OpenRouterChatClient,
    tender_id: str,
    document1: dict[str, Any],
    document2: dict[str, Any],
    crawler: LocalNewsCrawler,
    *,
    search_results_per_query: int,
    crawl_max_pages: int,
) -> dict[str, Any]:
    awarded_entity = _safe_get(document1, ["award_details", "awarded_entity", "value"], "")
    if not awarded_entity:
        award_info = document1.get("award_information")
        if isinstance(award_info, dict):
            awarded_entity = str(
                award_info.get("awarded_to")
                or award_info.get("awarded_entity")
                or award_info.get("winner")
                or ""
            ).strip()
        elif isinstance(award_info, str):
            awarded_entity = award_info.strip()
    if not awarded_entity or awarded_entity.lower() in {"not found", "unclear"}:
        awarded_entity = "unknown vendor"

    queries = []
    try:
        msgs_d3 = [
            {"role": "system", "content": "You are an OSINT researcher. Generate a list of precise DuckDuckGo search queries (strings) to find background details about a specific company. Output ONLY a valid JSON array of strings."},
            {"role": "user", "content": f"Target company/vendor: {awarded_entity}\nGenerate up to 8 queries looking for their GSTIN, CIN, address, scam/blacklist history, and director info. Return JSON array of strings ONLY."}
        ]
        resp = llm.chat(msgs_d3, max_tokens=500).strip()
        if resp.startswith("```json"): resp = resp[7:]
        if resp.startswith("```"): resp = resp[3:]
        if resp.endswith("```"): resp = resp[:-3]
        parsed = json.loads(resp.strip())
        if isinstance(parsed, list):
            queries = [str(x) for x in parsed[:10]]
    except Exception:
        pass

    if not queries:
        queries = [
            f'"{awarded_entity}" GSTIN',
            f'"{awarded_entity}" CIN',
            f'"{awarded_entity}" company profile',
            f'"{awarded_entity}" address',
            f'"{awarded_entity}" scam OR blacklist'
        ]
    results = crawler.search(queries, max_results_per_query=max(6, search_results_per_query // 2))
    pages = crawler.crawl(results, max_pages=max(1, min(crawl_max_pages, 24)), max_chars=10000)
    gst_lookup_data = fetch_gst_lookup_data(awarded_entity)

    compact_pages = [
        {
            "url": page.get("url", ""),
            "title": page.get("title", ""),
            "query": page.get("query", ""),
            "content": (page.get("content", "") or "")[:2500],
            "error": page.get("error", ""),
        }
        for page in pages
    ]

    deterministic_text_parts: list[str] = [
        json.dumps(gst_lookup_data, ensure_ascii=True),
        " ".join((p.get("content") or "")[:4000] for p in compact_pages[:12]),
        " ".join((p.get("title") or "") for p in compact_pages[:20]),
    ]
    deterministic_identifiers = extract_identifiers_from_text("\n".join(deterministic_text_parts))

    messages = [
        {"role": "system", "content": "You are a vendor due-diligence assistant. Return JSON only."},
        {
            "role": "user",
            "content": (
                f"Tender ID: {tender_id}\n"
                f"Target awarded vendor: {awarded_entity}\n"
                "Create document 3 with vendor details, especially GSTIN and identifiers.\n"
                "Return JSON schema only:\n"
                "{"
                "\"vendor_name\":\"\","
                "\"identifiers\":{\"gstin\":[],\"cin\":[],\"pan\":[]},"
                "\"business_profile\":{\"registered_name\":\"\",\"status\":\"\",\"addresses\":[],\"directors_or_proprietor\":[]},"
                "\"risk_flags\":[],"
                "\"source_map\":[{\"field\":\"\",\"value\":\"\",\"source\":\"\"}],"
                "\"confidence\":0.0"
                "}\n\n"
                f"Document 1:\n{json.dumps(document1, ensure_ascii=True)[:18000]}\n\n"
                f"Document 2:\n{json.dumps(document2, ensure_ascii=True)[:18000]}\n\n"
                f"GST lookup parsed data for awarded entity:\n{json.dumps(gst_lookup_data, ensure_ascii=True)[:18000]}\n\n"
                f"Crawled vendor pages:\n{json.dumps(compact_pages, ensure_ascii=True)[:100000]}"
            ),
        },
    ]

    try:
        payload = extract_json_object(llm.chat(messages, temperature=0.0, max_tokens=2200, num_ctx=32768))
        if not isinstance(payload, dict):
            raise ValueError("document 3 returned non-object JSON")
    except Exception as exc:
        payload = {
            "vendor_name": awarded_entity,
            "identifiers": {"gstin": [], "cin": [], "pan": []},
            "business_profile": {"registered_name": "", "status": "", "addresses": [], "directors_or_proprietor": []},
            "risk_flags": [f"Document 3 generation failed: {exc}"],
            "source_map": [],
            "confidence": 0.0,
        }

    payload.setdefault("vendor_name", awarded_entity)
    merged_identifiers = merge_identifier_map(
        payload.get("identifiers") if isinstance(payload.get("identifiers"), dict) else {},
        deterministic_identifiers,
    )
    payload["identifiers"] = merged_identifiers

    payload.setdefault("business_profile", {})
    if isinstance(payload.get("business_profile"), dict):
        payload["business_profile"].setdefault("registered_name", awarded_entity)
        if not str(payload["business_profile"].get("registered_name") or "").strip():
            payload["business_profile"]["registered_name"] = awarded_entity

    payload.setdefault("source_map", [])
    if isinstance(payload.get("source_map"), list):
        if merged_identifiers.get("gstin"):
            for gstin in merged_identifiers["gstin"]:
                payload["source_map"].append({"field": "identifiers.gstin", "value": gstin, "source": "gst_lookup_and_crawled_pages"})
        if merged_identifiers.get("cin"):
            for cin in merged_identifiers["cin"]:
                payload["source_map"].append({"field": "identifiers.cin", "value": cin, "source": "gst_lookup_and_crawled_pages"})
        if merged_identifiers.get("pan"):
            for pan in merged_identifiers["pan"]:
                payload["source_map"].append({"field": "identifiers.pan", "value": pan, "source": "gst_lookup_and_crawled_pages"})

    if not isinstance(payload.get("confidence"), (int, float)):
        payload["confidence"] = 0.0
    if any(payload["identifiers"].get(k) for k in ("gstin", "cin", "pan")):
        payload["confidence"] = max(float(payload.get("confidence") or 0.0), 0.55)

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Agentic tender pipeline: read source_dir/<tender_id>, research, and write a detailed report."
        )
    )
    parser.add_argument("--tender-id", required=True, help="6-digit tender ID")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tender_id = args.tender_id.strip()
    if not re.fullmatch(r"\d{6}", tender_id):
        print("tender-id must be exactly 6 digits")
        return 2

    source_root = (Path("downloads") / tender_id).resolve()
    output_root = Path("outputs/tender_reports").resolve()
    openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "tencent/hy3-preview:free")
    llm_timeout_seconds = 600.0
    max_doc_files = 200
    max_doc_chars = 12000
    search_results_per_query = 14
    crawl_max_pages = 36
    crawl_max_chars = 16000
    crawl_workers = 8

    tender_output_dir = output_root / tender_id
    tender_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading tender data from {source_root}")
    try:
        collected = build_local_source_collection(source_root)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(str(exc))
        return 2

    source_dirs = [Path(p) for p in collected["source_dirs"]]
    print(f"[2/6] Reading documents from {len(source_dirs)} source directories")
    reader = DocumentReader(max_files=max_doc_files, max_chars_per_file=max_doc_chars)
    docs, issues = reader.read_documents(source_dirs)
    print(f"      Readable docs: {len(docs)} | Issues: {len(issues)}")

    llm = OpenRouterChatClient(
        base_url=openrouter_base_url,
        model=openrouter_model,
        timeout_seconds=llm_timeout_seconds,
    )
    crawler = LocalNewsCrawler(max_workers=crawl_workers)

    print("[3/5] Step 1: Extracting structured tender facts (document 1)")
    document1 = analyze_documents(llm, tender_id, docs, collected["portal_data"])
    doc1_path = tender_output_dir / f"{tender_id}_document_1.json"
    write_json(doc1_path, document1)
    print(f"Document 1: {doc1_path}")

    print("[4/5] Step 2: Building search strings, crawling, and summarizing (document 2)")
    document2, step2_results, step2_pages = build_document2(
        llm,
        tender_id,
        document1,
        crawler,
        search_results_per_query=search_results_per_query,
        crawl_max_pages=crawl_max_pages,
        crawl_max_chars=crawl_max_chars,
    )
    doc2_path = tender_output_dir / f"{tender_id}_document_2.json"
    write_json(doc2_path, document2)
    print(f"Document 2: {doc2_path} | Search results: {len(step2_results)} | Crawled pages: {len(step2_pages)}")

    print("[5/5] Step 3: Vendor enrichment (GSTIN and identifiers) (document 3)")
    document3 = build_document3(
        llm,
        tender_id,
        document1,
        document2,
        crawler,
        search_results_per_query=search_results_per_query,
        crawl_max_pages=crawl_max_pages,
    )
    doc3_path = tender_output_dir / f"{tender_id}_document_3.json"
    write_json(doc3_path, document3)
    print(f"Document 3: {doc3_path}")

    pipeline_bundle = {
        "tender_id": tender_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "source_root": str(source_root),
            "output_root": str(output_root),
            "openrouter_model": openrouter_model,
            "openrouter_base_url": openrouter_base_url,
        },
        "collection": collected,
        "document_count": len(docs),
        "extraction_issues": [asdict(issue) for issue in issues],
        "document_1": document1,
        "document_2": document2,
        "document_3": document3,
    }
    bundle_path = tender_output_dir / f"{tender_id}_pipeline_bundle.json"
    write_json(bundle_path, pipeline_bundle)
    print(f"Pipeline bundle: {bundle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
