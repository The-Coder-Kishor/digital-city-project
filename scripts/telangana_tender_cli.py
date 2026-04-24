from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import requests

from telangana_tenders import TelanganaTenderClient, recursive_unzip
from telangana_tenders.client import validate_zip

OFFICE_EXTENSIONS = {
    ".doc",
    ".docx",
    ".docm",
    ".xls",
    ".xlsx",
    ".xlsm",
    ".ppt",
    ".pptx",
    ".odt",
    ".ods",
    ".odp",
}
LEGACY_OFFICE_EXTENSIONS = {".doc", ".xls", ".ppt"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Telangana tender pages and download related documents "
            "with request rate-limiting."
        )
    )
    parser.add_argument("--tender-id", required=True, help="Procurement ID (hdnProcurementID).")
    parser.add_argument(
        "--rate-limit-seconds",
        type=float,
        default=2.0,
        help="Minimum delay between requests in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        default="downloads",
        help="Directory for downloaded ZIP and extracted contents.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download tender documents ZIP using BulkDownLoad.html.",
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        help="Recursively unzip downloaded files (except docx/xlsx/pptx).",
    )
    parser.add_argument(
        "--write-html",
        action="store_true",
        help="Write raw HTML responses to files in output-dir/inspect.",
    )
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Write parsed JSON for each fetched endpoint.",
    )
    parser.add_argument(
        "--skip-general",
        action="store_true",
        help="Skip general tender details fetch.",
    )
    parser.add_argument(
        "--skip-docs-page",
        action="store_true",
        help="Skip documents listing page fetch.",
    )
    parser.add_argument(
        "--skip-award",
        action="store_true",
        help="Skip awarded-details fetch.",
    )
    parser.add_argument(
        "--skip-award-docs",
        action="store_true",
        help="Skip awarded-documents fetch.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = TelanganaTenderClient(rate_limit_seconds=args.rate_limit_seconds)
    cookies = client.bootstrap_cookies()
    print("Cookies:", json.dumps(cookies))

    if not args.skip_general:
        general = client.fetch_general_info(args.tender_id)
        general_json = general.to_json()
        print(f"\n[General] URL: {general.url}")
        if args.write_html:
            write_text(output_dir / "inspect" / "general.html", general.html_text)
        if args.write_json:
            write_json(output_dir / "inspect" / "general.json", general_json)

    if not args.skip_docs_page:
        docs = client.fetch_documents_info(args.tender_id)
        docs_json = docs.to_json()
        print(f"\n[Documents] URL: {docs.url}")
        if args.write_html:
            write_text(output_dir / "inspect" / "documents.html", docs.html_text)
        if args.write_json:
            write_json(output_dir / "inspect" / "documents.json", docs_json)

    if not args.skip_award:
        try:
            award = client.fetch_award_details(args.tender_id)
            award_json = award.to_json()
            print(f"\n[Award] URL: {award.url}")
            if args.write_html:
                write_text(output_dir / "inspect" / "award.html", award.html_text)
            if args.write_json:
                write_json(output_dir / "inspect" / "award.json", award_json)
        except requests.RequestException as exc:
            print(
                "\n[Award] Skipped: this tender may not be awarded yet "
                f"or award details are unavailable ({exc})."
            )

    if not args.skip_award_docs:
        try:
            award_docs = client.fetch_award_documents(args.tender_id)
            award_docs_json = award_docs.to_json()
            print(f"\n[AwardDocuments] URL: {award_docs.url}")
            if args.write_html:
                write_text(output_dir / "inspect" / "award_documents.html", award_docs.html_text)
            if args.write_json:
                write_json(output_dir / "inspect" / "award_documents.json", award_docs_json)
        except requests.RequestException as exc:
            print(
                "\n[AwardDocuments] Skipped: this tender may not be awarded yet "
                f"or award documents are unavailable ({exc})."
            )

    if args.download:
        zip_path = output_dir / f"{args.tender_id}_documents.zip"
        client.download_documents_zip(args.tender_id, zip_path)
        print(f"\nDownloaded ZIP: {zip_path}")
        if validate_zip(zip_path):
            print("ZIP validation: OK")
        else:
            print("ZIP validation: failed (download may not be a real archive)")
        if args.unzip and validate_zip(zip_path):
            unzip_dir = output_dir / f"{args.tender_id}_extracted"
            unzip_dir.mkdir(parents=True, exist_ok=True)
            archives = recursive_unzip(zip_path, unzip_dir)
            print(f"Extracted archives: {len(archives)}")
            print(f"Extraction directory: {unzip_dir}")
            conversion_summary = convert_office_files_to_markdown(unzip_dir)
            print(
                "Office->MD conversion: "
                f"{conversion_summary['converted']} converted, "
                f"{conversion_summary['failed']} failed, "
                f"{conversion_summary['skipped']} skipped"
            )
        elif args.unzip:
            print("Skipping unzip because the downloaded file is not a valid ZIP.")
    elif args.unzip:
        print("--unzip ignored because --download was not set.")

    return 0


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Wrote: {path}")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote: {path}")


def convert_office_files_to_markdown(root: Path) -> dict[str, int]:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        print("Office->MD conversion skipped: `pandoc` not found in PATH.")
        return {"converted": 0, "failed": 0, "skipped": 0}

    converted = 0
    failed = 0
    skipped = 0
    office_files = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in OFFICE_EXTENSIONS])
    for source in office_files:
        target = Path(f"{source}.md")
        if target.exists():
            skipped += 1
            continue

        markdown = office_to_markdown(source, pandoc_bin=pandoc)
        if not markdown:
            failed += 1
            continue

        target.write_text(markdown, encoding="utf-8", errors="ignore")
        converted += 1

    return {"converted": converted, "failed": failed, "skipped": skipped}


def office_to_markdown(source: Path, *, pandoc_bin: str) -> str:
    suffix = source.suffix.lower()
    soffice_bin = shutil.which("libreoffice") or shutil.which("soffice")
    with tempfile.TemporaryDirectory(prefix="office_md_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        input_path = source

        if suffix in LEGACY_OFFICE_EXTENSIONS:
            if soffice_bin:
                converted = convert_with_libreoffice(source, tmp_root, "odt", soffice_bin=soffice_bin)
                if converted is not None:
                    input_path = converted
        elif suffix in {".xls", ".xlsx", ".xlsm", ".ppt", ".pptx", ".ods", ".odp"}:
            if soffice_bin:
                converted = convert_with_libreoffice(source, tmp_root, "odt", soffice_bin=soffice_bin)
                if converted is not None:
                    input_path = converted

        output_path = tmp_root / "converted.md"
        try:
            completed = subprocess.run(
                [pandoc_bin, str(input_path), "--to=gfm", "--output", str(output_path)],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
        except OSError:
            return ""
        if completed.returncode != 0 or not output_path.exists():
            return ""
        try:
            return output_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""


def convert_with_libreoffice(source: Path, out_dir: Path, target_ext: str, *, soffice_bin: str) -> Path | None:
    try:
        completed = subprocess.run(
            [soffice_bin, "--headless", "--convert-to", target_ext, "--outdir", str(out_dir), str(source)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    exact = sorted(out_dir.glob(f"{source.stem}.{target_ext}"))
    if exact:
        return exact[0]
    any_match = sorted(out_dir.glob(f"*.{target_ext}"))
    return any_match[0] if any_match else None


if __name__ == "__main__":
    raise SystemExit(main())