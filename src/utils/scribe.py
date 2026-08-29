from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
GENESIS_HASH = "0" * 64
_HASH_DOMAIN = b"llm-router-audit-v1\0"


class AuditError(RuntimeError):
    """Base exception for audit ledger failures."""


class AuditIntegrityError(AuditError):
    """Raised when an audit ledger fails hash-chain verification."""


@dataclass(frozen=True)
class AuditRecord:
    schema_version: int
    sequence: int
    event_id: str
    occurred_at: str
    event_type: str
    request_id: str
    decision_id: str | None
    payload: dict[str, Any]
    previous_hash: str
    record_hash: str


@dataclass(frozen=True)
class AuditReceipt:
    sequence: int
    event_id: str
    record_hash: str


class Scribe:
    """Append-only SQLite audit ledger protected by a SHA-256 hash chain."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._id_factory = id_factory or (lambda: str(uuid.uuid4()))
        self._initialize()
        self.verify()

    def append(
        self,
        event_type: str,
        *,
        request_id: str,
        payload: Mapping[str, object],
        decision_id: str | None = None,
    ) -> AuditReceipt:
        if not event_type:
            raise ValueError("event_type must not be empty.")
        if not request_id:
            raise ValueError("request_id must not be empty.")

        payload_json = _canonical_json(dict(payload))
        occurred_at = self._clock().astimezone(timezone.utc).isoformat()
        event_id = self._id_factory()

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                records = self._verify_connection(connection)
                sequence = records[-1].sequence + 1 if records else 1
                previous_hash = records[-1].record_hash if records else GENESIS_HASH
                envelope = _record_envelope(
                    schema_version=SCHEMA_VERSION,
                    sequence=sequence,
                    event_id=event_id,
                    occurred_at=occurred_at,
                    event_type=event_type,
                    request_id=request_id,
                    decision_id=decision_id,
                    payload_json=payload_json,
                    previous_hash=previous_hash,
                )
                record_hash = _record_hash(previous_hash, envelope)
                connection.execute(
                    """
                    INSERT INTO audit_records (
                        schema_version,
                        sequence,
                        event_id,
                        occurred_at,
                        event_type,
                        request_id,
                        decision_id,
                        payload_json,
                        previous_hash,
                        record_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        SCHEMA_VERSION,
                        sequence,
                        event_id,
                        occurred_at,
                        event_type,
                        request_id,
                        decision_id,
                        payload_json,
                        previous_hash,
                        record_hash,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

        return AuditReceipt(
            sequence=sequence,
            event_id=event_id,
            record_hash=record_hash,
        )

    def verify(self) -> tuple[AuditRecord, ...]:
        with self._connect() as connection:
            connection.execute("BEGIN")
            try:
                records = self._verify_connection(connection)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return records

    def records(self) -> Iterator[AuditRecord]:
        yield from self.verify()

    def new_id(self) -> str:
        """Create an identifier using the ledger's injectable ID source."""
        return self._id_factory()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_records (
                    schema_version INTEGER NOT NULL,
                    sequence INTEGER PRIMARY KEY CHECK (sequence > 0),
                    event_id TEXT NOT NULL UNIQUE,
                    occurred_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    decision_id TEXT,
                    payload_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    record_hash TEXT NOT NULL UNIQUE
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _verify_connection(
        self,
        connection: sqlite3.Connection,
    ) -> tuple[AuditRecord, ...]:
        rows = connection.execute(
            """
            SELECT
                schema_version,
                sequence,
                event_id,
                occurred_at,
                event_type,
                request_id,
                decision_id,
                payload_json,
                previous_hash,
                record_hash
            FROM audit_records
            ORDER BY sequence
            """
        ).fetchall()

        records: list[AuditRecord] = []
        expected_previous_hash = GENESIS_HASH
        for expected_sequence, row in enumerate(rows, start=1):
            (
                schema_version,
                sequence,
                event_id,
                occurred_at,
                event_type,
                request_id,
                decision_id,
                payload_json,
                previous_hash,
                stored_hash,
            ) = row
            if schema_version != SCHEMA_VERSION:
                raise AuditIntegrityError(
                    f"Unsupported schema version at audit record {sequence}."
                )
            if sequence != expected_sequence:
                raise AuditIntegrityError(
                    f"Audit sequence gap at record {expected_sequence}."
                )
            if previous_hash != expected_previous_hash:
                raise AuditIntegrityError(
                    f"Invalid previous hash at audit record {sequence}."
                )

            try:
                payload = json.loads(payload_json)
            except (TypeError, json.JSONDecodeError) as exc:
                raise AuditIntegrityError(
                    f"Invalid payload JSON at audit record {sequence}."
                ) from exc
            if not isinstance(payload, dict):
                raise AuditIntegrityError(
                    f"Audit payload at record {sequence} is not an object."
                )
            if _canonical_json(payload) != payload_json:
                raise AuditIntegrityError(
                    f"Non-canonical payload at audit record {sequence}."
                )

            envelope = _record_envelope(
                schema_version=schema_version,
                sequence=sequence,
                event_id=event_id,
                occurred_at=occurred_at,
                event_type=event_type,
                request_id=request_id,
                decision_id=decision_id,
                payload_json=payload_json,
                previous_hash=previous_hash,
            )
            calculated_hash = _record_hash(previous_hash, envelope)
            if calculated_hash != stored_hash:
                raise AuditIntegrityError(
                    f"Invalid hash at audit record {sequence}."
                )

            record = AuditRecord(
                schema_version=schema_version,
                sequence=sequence,
                event_id=event_id,
                occurred_at=occurred_at,
                event_type=event_type,
                request_id=request_id,
                decision_id=decision_id,
                payload=payload,
                previous_hash=previous_hash,
                record_hash=stored_hash,
            )
            records.append(record)
            expected_previous_hash = stored_hash

        return tuple(records)


def _record_envelope(
    *,
    schema_version: int,
    sequence: int,
    event_id: str,
    occurred_at: str,
    event_type: str,
    request_id: str,
    decision_id: str | None,
    payload_json: str,
    previous_hash: str,
) -> bytes:
    payload = json.loads(payload_json)
    return _canonical_json(
        {
            "schema_version": schema_version,
            "sequence": sequence,
            "event_id": event_id,
            "occurred_at": occurred_at,
            "event_type": event_type,
            "request_id": request_id,
            "decision_id": decision_id,
            "payload": payload,
            "previous_hash": previous_hash,
        }
    ).encode("utf-8")


def _record_hash(previous_hash: str, envelope: bytes) -> str:
    try:
        previous_hash_bytes = bytes.fromhex(previous_hash)
    except ValueError as exc:
        raise AuditIntegrityError("Audit record contains a malformed hash.") from exc
    if len(previous_hash_bytes) != hashlib.sha256().digest_size:
        raise AuditIntegrityError("Audit record contains a malformed hash.")
    return hashlib.sha256(_HASH_DOMAIN + previous_hash_bytes + envelope).hexdigest()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Audit payload must contain finite JSON values.") from exc
