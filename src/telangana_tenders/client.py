from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from lxml import html


BASE_URL = "https://tender.telangana.gov.in"
EXCLUDED_RECURSIVE_UNZIP_EXTENSIONS = {".docx", ".xlsx", ".pptx"}


@dataclass(slots=True)
class TenderPage:
    tender_id: str
    url: str
    html_text: str

    @property
    def text(self) -> str:
        tree = html.fromstring(self.html_text)
        return " ".join(tree.text_content().split())

    def to_json(self) -> dict[str, object]:
        return parse_html_to_json(self.html_text, self.url, self.tender_id)


class TelanganaTenderClient:
    def __init__(
        self,
        base_url: str = BASE_URL,
        *,
        rate_limit_seconds: float = 2.0,
        timeout_seconds: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.rate_limit_seconds = rate_limit_seconds
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self._last_request_at: float | None = None

    def bootstrap_cookies(self) -> dict[str, str]:
        response = self._request("GET", "/")
        response.raise_for_status()
        return self.get_cookie_values()

    def get_cookie_values(self) -> dict[str, str]:
        cookies = self.session.cookies.get_dict()
        required = {"SRVID", "JSESSIONID"}
        missing = sorted(required - set(cookies.keys()))
        if missing:
            raise RuntimeError(
                f"Missing required cookies: {', '.join(missing)}. "
                "Try bootstrapping cookies again."
            )
        return {name: cookies[name] for name in sorted(required)}

    def fetch_general_info(self, tender_id: str) -> TenderPage:
        return self._fetch_page(
            tender_id=tender_id,
            endpoint="/ViewTender.html#",
            payload={"hdnProcurementID": tender_id, "popUPRequestParameter": "popUPRequestParameter"},
        )

    def fetch_documents_info(self, tender_id: str) -> TenderPage:
        return self._fetch_page(
            tender_id=tender_id,
            endpoint="/ViewTenderDocuments.html#",
            payload={"hdnProcurementID": tender_id, "popUPRequestParameter": "popUPRequestParameter"},
        )

    def fetch_award_details(self, tender_id: str) -> TenderPage:
        return self._fetch_page(
            tender_id=tender_id,
            endpoint="/viewitemawardeddetails.html#",
            payload={"hdnProcurementID": tender_id, "popUPRequestParameter": "popUPRequestParameter"},
        )

    def fetch_award_documents(self, tender_id: str) -> TenderPage:
        return self._fetch_page(
            tender_id=tender_id,
            endpoint="/viewitemawarddocuments.html#",
            payload={"hdnProcurementID": tender_id, "popUPRequestParameter": "popUPRequestParameter"},
        )

    def download_documents_zip(self, tender_id: str, destination: Path) -> Path:
        response = self._request(
            "POST",
            "/BulkDownLoad.html",
            data={"hdnProcurementID": tender_id},
            stream=True,
        )
        response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
        return destination

    def _fetch_page(self, *, tender_id: str, endpoint: str, payload: dict[str, str]) -> TenderPage:
        response = self._request("POST", endpoint, data=payload)
        response.raise_for_status()
        return TenderPage(
            tender_id=tender_id,
            url=response.url,
            html_text=response.text,
        )

    def _request(self, method: str, endpoint: str, **kwargs: object) -> requests.Response:
        self._wait_for_rate_limit()
        url = self._absolute_url(endpoint)
        response = self.session.request(method, url, timeout=self.timeout_seconds, **kwargs)
        self._last_request_at = time.monotonic()
        return response

    def _wait_for_rate_limit(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.rate_limit_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _absolute_url(self, endpoint: str) -> str:
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        return f"{self.base_url}{endpoint}"


def recursive_unzip(
    zip_path: Path,
    output_dir: Path,
    *,
    excluded_extensions: Iterable[str] = EXCLUDED_RECURSIVE_UNZIP_EXTENSIONS,
) -> list[Path]:
    excluded = {ext.lower() for ext in excluded_extensions}
    extracted_archives: list[Path] = []
    queue: list[Path] = [zip_path]
    seen: set[Path] = set()

    while queue:
        current_zip = queue.pop(0).resolve()
        if current_zip in seen:
            continue
        seen.add(current_zip)

        with zipfile.ZipFile(current_zip) as archive:
            _safe_extract(archive, output_dir)
            extracted_archives.append(current_zip)

        for candidate in output_dir.rglob("*"):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() in excluded:
                continue
            if zipfile.is_zipfile(candidate):
                queue.append(candidate)

    return extracted_archives


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        member_path = destination.joinpath(member.filename).resolve()
        if not str(member_path).startswith(str(destination)):
            raise ValueError(f"Unsafe zip member path detected: {member.filename}")
    archive.extractall(destination)


def is_zip_binary(path: Path) -> bool:
    with path.open("rb") as handle:
        head = handle.read(4)
    return head == b"PK\x03\x04"


def validate_zip(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if not is_zip_binary(path):
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def parse_html_to_json(html_text: str, source_url: str, tender_id: str) -> dict[str, object]:
    tree = html.fromstring(html_text)
    title_nodes = tree.xpath("//title/text()")
    title = title_nodes[0].strip() if title_nodes else ""

    tables: list[dict[str, object]] = []
    for idx, table in enumerate(tree.xpath("//table"), start=1):
        headers = [_clean_text(" ".join(node.itertext())) for node in table.xpath(".//tr[1]/th")]
        panel_header = _infer_panel_header(table)
        rows = []
        for tr in table.xpath(".//tr"):
            cells = [_clean_text(" ".join(node.itertext())) for node in tr.xpath("./th|./td")]
            cells = [c for c in cells if c]
            if cells:
                rows.append(cells)
        if not rows:
            continue
        tables.append(
            {
                "index": idx,
                "panel_header": panel_header,
                "headers": headers,
                "rows": rows,
                "row_count": len(rows),
            }
        )

    key_values: dict[str, str] = {}
    for tr in tree.xpath("//tr[count(td)=2]"):
        first = _clean_text(" ".join(tr.xpath("./td[1]//text()")))
        second = _clean_text(" ".join(tr.xpath("./td[2]//text()")))
        if first and second:
            key_values[first] = second

    form_inputs = []
    for form in tree.xpath("//form"):
        for inp in form.xpath(".//input|.//select|.//textarea"):
            name = inp.get("name") or inp.get("id")
            if not name:
                continue
            value = inp.get("value") or _clean_text(" ".join(inp.itertext()))
            form_inputs.append({"name": name, "value": value})

    return {
        "source_url": source_url,
        "tender_id": tender_id,
        "title": title,
        "key_values": key_values,
        "tables": tables,
        "form_inputs": form_inputs,
        "text_preview": " ".join(tree.text_content().split())[:1200],
    }


def _clean_text(value: str) -> str:
    return " ".join(value.split()).strip()


def _infer_panel_header(table: html.HtmlElement) -> str:
    # Common pattern: a panel heading div appears before the table.
    heading_nodes = table.xpath(
        (
            "ancestor::*[contains(concat(' ', normalize-space(@class), ' '), ' panel ')][1]"
            "//*[contains(concat(' ', normalize-space(@class), ' '), ' panel-heading ')][1]"
        )
    )
    if heading_nodes:
        return _clean_text(" ".join(heading_nodes[0].itertext()))

    candidates = table.xpath(
        (
            "preceding::*[self::div or self::h1 or self::h2 or self::h3 or self::h4]"
            "[contains(translate(@class, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'panel-heading')"
            " or contains(translate(@class, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'card-header')"
            " or self::h1 or self::h2 or self::h3 or self::h4][1]"
        )
    )
    if candidates:
        return _clean_text(" ".join(candidates[0].itertext()))

    return ""
