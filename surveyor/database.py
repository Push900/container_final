"""
surveyor.database — persistent inspection store + searchable audit trail.

A self-contained SQLite database (zero external services) that records every
inspection so users can search history by container number, seal number, date,
shipping line, result or damage category, and so each inspection keeps a
tamper-evident audit trail.

Tables
------
inspections   one row per completed inspection (indexed for search) + a JSON
              blob holding the full record.
audit_log     append-only event log (created, viewed, report_generated, …),
              each row chained by hash to the previous for tamper-evidence.

The DB is safe for the Gradio app's threaded use (check_same_thread=False + a
process-level lock).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/inspections.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id                 TEXT PRIMARY KEY,
    created_at         TEXT NOT NULL,
    container_number   TEXT,
    seal_number        TEXT,
    seal_status        TEXT,
    container_type     TEXT,
    size_type_code     TEXT,
    shipping_line      TEXT,
    result             TEXT,           -- PASS / FAIL / REVIEW
    inspection_score   REAL,
    worst_severity     TEXT,
    damage_count       INTEGER,
    repair_priority    TEXT,
    gps                TEXT,
    inspector          TEXT,
    report_path        TEXT,
    signature_hash     TEXT,
    record_json        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_container ON inspections(container_number);
CREATE INDEX IF NOT EXISTS idx_seal      ON inspections(seal_number);
CREATE INDEX IF NOT EXISTS idx_created   ON inspections(created_at);
CREATE INDEX IF NOT EXISTS idx_line      ON inspections(shipping_line);
CREATE INDEX IF NOT EXISTS idx_result    ON inspections(result);

CREATE TABLE IF NOT EXISTS audit_log (
    seq           INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id TEXT,
    ts            TEXT NOT NULL,
    event         TEXT NOT NULL,
    detail        TEXT,
    prev_hash     TEXT,
    entry_hash    TEXT NOT NULL
);
"""

_SEARCH_COLUMNS = [
    "id", "created_at", "container_number", "seal_number", "seal_status",
    "container_type", "size_type_code", "shipping_line", "result",
    "inspection_score", "worst_severity", "damage_count", "repair_priority",
    "gps", "inspector", "report_path", "signature_hash",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class InspectionDB:
    def __init__(self, path: Path = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ── write ───────────────────────────────────────────────────────────────
    def save(self, record: dict) -> str:
        """Persist an inspection record dict; returns the inspection id.

        The record should contain an ``id`` (else one is generated) and the
        fields used for search; the entire dict is stored as JSON too.
        """
        rid = record.get("id") or self._gen_id(record)
        record["id"] = rid
        row = {
            "id": rid,
            "created_at": record.get("created_at") or _now(),
            "container_number": record.get("container_number"),
            "seal_number": record.get("seal_number"),
            "seal_status": record.get("seal_status"),
            "container_type": record.get("container_type"),
            "size_type_code": record.get("size_type_code"),
            "shipping_line": record.get("shipping_line"),
            "result": record.get("result"),
            "inspection_score": record.get("inspection_score"),
            "worst_severity": record.get("worst_severity"),
            "damage_count": record.get("damage_count"),
            "repair_priority": record.get("repair_priority"),
            "gps": record.get("gps"),
            "inspector": record.get("inspector", "AI Surveyor"),
            "report_path": record.get("report_path"),
            "signature_hash": record.get("signature_hash"),
            "record_json": json.dumps(record, default=str),
        }
        cols = ", ".join(row)
        ph = ", ".join("?" for _ in row)
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO inspections ({cols}) VALUES ({ph})",
                list(row.values()),
            )
            self._conn.commit()
        self.audit(rid, "created", f"result={row['result']} score={row['inspection_score']}")
        return rid

    def update_report_path(self, rid: str, report_path: str, signature_hash: str | None = None):
        with self._lock:
            self._conn.execute(
                "UPDATE inspections SET report_path=?, signature_hash=COALESCE(?, signature_hash) WHERE id=?",
                (report_path, signature_hash, rid),
            )
            self._conn.commit()
        self.audit(rid, "report_generated", report_path)

    # ── read / search ────────────────────────────────────────────────────────
    def get(self, rid: str) -> dict | None:
        with self._lock:
            cur = self._conn.execute("SELECT record_json FROM inspections WHERE id=?", (rid,))
            r = cur.fetchone()
        return json.loads(r["record_json"]) if r else None

    def search(
        self,
        container_number: str | None = None,
        seal_number: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        shipping_line: str | None = None,
        result: str | None = None,
        damage_category: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Search inspections; all filters are AND-combined and optional."""
        clauses, params = [], []
        if container_number:
            clauses.append("container_number LIKE ?"); params.append(f"%{container_number.upper()}%")
        if seal_number:
            clauses.append("seal_number LIKE ?"); params.append(f"%{seal_number.upper()}%")
        if date_from:
            clauses.append("created_at >= ?"); params.append(date_from)
        if date_to:
            clauses.append("created_at <= ?"); params.append(date_to)
        if shipping_line:
            clauses.append("shipping_line LIKE ?"); params.append(f"%{shipping_line}%")
        if result:
            clauses.append("result = ?"); params.append(result.upper())
        if damage_category:
            clauses.append("(worst_severity = ? OR record_json LIKE ?)")
            params += [damage_category.lower(), f"%{damage_category.lower()}%"]
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (f"SELECT {', '.join(_SEARCH_COLUMNS)} FROM inspections{where} "
               f"ORDER BY created_at DESC LIMIT ?")
        params.append(limit)
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
        return rows

    def recent(self, limit: int = 25) -> list[dict]:
        return self.search(limit=limit)

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) c FROM inspections").fetchone()["c"]

    # ── audit trail (hash-chained) ───────────────────────────────────────────
    def audit(self, inspection_id: str | None, event: str, detail: str = "") -> str:
        ts = _now()
        with self._lock:
            prev = self._conn.execute(
                "SELECT entry_hash FROM audit_log ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            prev_hash = prev["entry_hash"] if prev else ""
            payload = f"{prev_hash}|{inspection_id}|{ts}|{event}|{detail}"
            entry_hash = hashlib.sha256(payload.encode()).hexdigest()
            self._conn.execute(
                "INSERT INTO audit_log (inspection_id, ts, event, detail, prev_hash, entry_hash)"
                " VALUES (?,?,?,?,?,?)",
                (inspection_id, ts, event, detail, prev_hash, entry_hash),
            )
            self._conn.commit()
        return entry_hash

    def audit_trail(self, inspection_id: str | None = None) -> list[dict]:
        with self._lock:
            if inspection_id:
                cur = self._conn.execute(
                    "SELECT * FROM audit_log WHERE inspection_id=? ORDER BY seq", (inspection_id,))
            else:
                cur = self._conn.execute("SELECT * FROM audit_log ORDER BY seq")
            return [dict(r) for r in cur.fetchall()]

    def verify_audit_chain(self) -> bool:
        """Recompute the hash chain to detect tampering."""
        rows = self.audit_trail()
        prev = ""
        for r in rows:
            payload = f"{prev}|{r['inspection_id']}|{r['ts']}|{r['event']}|{r['detail']}"
            if hashlib.sha256(payload.encode()).hexdigest() != r["entry_hash"]:
                return False
            prev = r["entry_hash"]
        return True

    @staticmethod
    def _gen_id(record: dict) -> str:
        base = f"{record.get('container_number','UNK')}-{_now()}"
        return "INS-" + hashlib.sha1(base.encode()).hexdigest()[:12].upper()

    def close(self):
        with self._lock:
            self._conn.close()


# module-level singleton for convenience
_DB: InspectionDB | None = None


def get_db(path: Path = DB_PATH) -> InspectionDB:
    global _DB
    if _DB is None:
        _DB = InspectionDB(path)
    return _DB
