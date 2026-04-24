from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS tenders (
  tender_id TEXT PRIMARY KEY,
  tender_name TEXT NOT NULL,
  name_id TEXT NOT NULL,
  report_md_path TEXT,
  report_json_path TEXT,
  report_html_path TEXT,
  updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tenders_name_id ON tenders(name_id);

CREATE TABLE IF NOT EXISTS sources (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tender_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  title TEXT,
  url TEXT,
  file_path TEXT,
  notes TEXT,
  added_at INTEGER NOT NULL,
  FOREIGN KEY (tender_id) REFERENCES tenders(tender_id)
);

CREATE INDEX IF NOT EXISTS idx_sources_tender ON sources(tender_id);
"""


@dataclass(frozen=True, slots=True)
class TenderRow:
    tender_id: str
    tender_name: str
    name_id: str
    report_md_path: str | None
    report_json_path: str | None
    report_html_path: str | None
    updated_at: int


@dataclass(frozen=True, slots=True)
class SourceRow:
    id: int
    tender_id: str
    kind: str
    title: str | None
    url: str | None
    file_path: str | None
    notes: str | None
    added_at: int


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def now_ts() -> int:
    return int(time.time())


def upsert_tender(
    con: sqlite3.Connection,
    *,
    tender_id: str,
    tender_name: str,
    report_md_path: str | None,
    report_json_path: str | None,
    report_html_path: str | None,
) -> None:
    name = (tender_name or "").strip() or f"Tender {tender_id}"
    name_id = f"{name}-{tender_id}"
    ts = now_ts()
    con.execute(
        """
        INSERT INTO tenders(tender_id, tender_name, name_id, report_md_path, report_json_path, report_html_path, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(tender_id) DO UPDATE SET
          tender_name=excluded.tender_name,
          name_id=excluded.name_id,
          report_md_path=excluded.report_md_path,
          report_json_path=excluded.report_json_path,
          report_html_path=excluded.report_html_path,
          updated_at=excluded.updated_at
        """,
        (tender_id, name, name_id, report_md_path, report_json_path, report_html_path, ts),
    )
    con.commit()


def list_tenders(con: sqlite3.Connection, *, q: str = "", limit: int = 200) -> list[TenderRow]:
    q = (q or "").strip()
    if q:
        rows = con.execute(
            """
            SELECT * FROM tenders
            WHERE name_id LIKE ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (f"%{q}%", max(1, limit)),
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT * FROM tenders ORDER BY updated_at DESC LIMIT ?",
            (max(1, limit),),
        ).fetchall()
    return [
        TenderRow(
            tender_id=str(r["tender_id"]),
            tender_name=str(r["tender_name"]),
            name_id=str(r["name_id"]),
            report_md_path=r["report_md_path"],
            report_json_path=r["report_json_path"],
            report_html_path=r["report_html_path"],
            updated_at=int(r["updated_at"]),
        )
        for r in rows
    ]


def get_tender(con: sqlite3.Connection, tender_id: str) -> TenderRow | None:
    r = con.execute("SELECT * FROM tenders WHERE tender_id=?", (tender_id,)).fetchone()
    if not r:
        return None
    return TenderRow(
        tender_id=str(r["tender_id"]),
        tender_name=str(r["tender_name"]),
        name_id=str(r["name_id"]),
        report_md_path=r["report_md_path"],
        report_json_path=r["report_json_path"],
        report_html_path=r["report_html_path"],
        updated_at=int(r["updated_at"]),
    )


def add_source(
    con: sqlite3.Connection,
    *,
    tender_id: str,
    kind: str,
    title: str | None = None,
    url: str | None = None,
    file_path: str | None = None,
    notes: str | None = None,
) -> int:
    ts = now_ts()
    cur = con.execute(
        """
        INSERT INTO sources(tender_id, kind, title, url, file_path, notes, added_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (tender_id, kind, title, url, file_path, notes, ts),
    )
    con.commit()
    return int(cur.lastrowid)


def list_sources(con: sqlite3.Connection, tender_id: str, *, limit: int = 500) -> list[SourceRow]:
    rows = con.execute(
        "SELECT * FROM sources WHERE tender_id=? ORDER BY added_at DESC LIMIT ?",
        (tender_id, max(1, limit)),
    ).fetchall()
    return [
        SourceRow(
            id=int(r["id"]),
            tender_id=str(r["tender_id"]),
            kind=str(r["kind"]),
            title=r["title"],
            url=r["url"],
            file_path=r["file_path"],
            notes=r["notes"],
            added_at=int(r["added_at"]),
        )
        for r in rows
    ]


def delete_source(con: sqlite3.Connection, source_id: int) -> None:
    con.execute("DELETE FROM sources WHERE id=?", (int(source_id),))
    con.commit()


def as_jsonable(row: Any) -> dict[str, Any]:
    if hasattr(row, "__dict__"):
        return dict(row.__dict__)
    if hasattr(row, "_asdict"):
        return row._asdict()
    raise TypeError("Unsupported row type")

