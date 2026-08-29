from __future__ import annotations

import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

from utils.scribe import AuditIntegrityError, GENESIS_HASH, Scribe


def fixed_clock() -> datetime:
    return datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


def append_from_process(arguments: tuple[str, int]) -> None:
    path, number = arguments
    Scribe(path).append(
        "event",
        request_id=f"request-{number}",
        payload={"number": number},
    )


def test_appends_and_verifies_a_hash_chain(tmp_path: Path) -> None:
    ids = iter(("event-1", "event-2"))
    scribe = Scribe(
        tmp_path / "audit.db",
        clock=fixed_clock,
        id_factory=lambda: next(ids),
    )

    first = scribe.append(
        "routing_decision",
        request_id="request-1",
        decision_id="decision-1",
        payload={"query": "hello", "selected_provider_id": "provider-a"},
    )
    second = scribe.append(
        "provider_attempt_started",
        request_id="request-1",
        decision_id="decision-1",
        payload={"attempt": 1, "provider_id": "provider-a"},
    )

    records = scribe.verify()
    assert first.sequence == 1
    assert second.sequence == 2
    assert records[0].previous_hash == GENESIS_HASH
    assert records[1].previous_hash == records[0].record_hash
    assert records[0].payload["query"] == "hello"


@pytest.mark.parametrize(
    ("column", "replacement"),
    (
        ("event_type", "changed"),
        ("request_id", "changed"),
        ("payload_json", '{"query":"changed"}'),
        ("previous_hash", "f" * 64),
        ("record_hash", "f" * 64),
    ),
)
def test_detects_modified_record_fields(
    tmp_path: Path,
    column: str,
    replacement: str,
) -> None:
    path = tmp_path / "audit.db"
    scribe = Scribe(path, clock=fixed_clock, id_factory=lambda: "event-1")
    scribe.append(
        "routing_decision",
        request_id="request-1",
        payload={"query": "original"},
    )

    with sqlite3.connect(path) as connection:
        connection.execute(
            f"UPDATE audit_records SET {column} = ? WHERE sequence = 1",
            (replacement,),
        )

    with pytest.raises(AuditIntegrityError):
        scribe.verify()
    with pytest.raises(AuditIntegrityError):
        scribe.append("another_event", request_id="request-1", payload={})


def test_detects_deleted_record_from_the_middle(tmp_path: Path) -> None:
    path = tmp_path / "audit.db"
    ids = iter(("event-1", "event-2", "event-3"))
    scribe = Scribe(path, clock=fixed_clock, id_factory=lambda: next(ids))
    for number in range(3):
        scribe.append("event", request_id="request-1", payload={"number": number})

    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM audit_records WHERE sequence = 2")

    with pytest.raises(AuditIntegrityError, match="sequence gap"):
        scribe.verify()


def test_detects_inserted_and_reordered_records(tmp_path: Path) -> None:
    path = tmp_path / "audit.db"
    scribe = Scribe(path)
    scribe.append("first", request_id="request-1", payload={})
    scribe.append("second", request_id="request-2", payload={})

    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE audit_records SET sequence = 99 WHERE sequence = 1")
        connection.execute("UPDATE audit_records SET sequence = 1 WHERE sequence = 2")
        connection.execute("UPDATE audit_records SET sequence = 2 WHERE sequence = 99")

    with pytest.raises(AuditIntegrityError):
        scribe.verify()

    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM audit_records")
        connection.execute(
            """
            INSERT INTO audit_records VALUES (
                1, 1, 'inserted', '2026-08-29T12:00:00+00:00', 'event',
                'request', NULL, '{}', ?, ?
            )
            """,
            (GENESIS_HASH, "f" * 64),
        )

    with pytest.raises(AuditIntegrityError):
        scribe.verify()


def test_reopens_and_continues_an_existing_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.db"
    Scribe(path, clock=fixed_clock, id_factory=lambda: "event-1").append(
        "event",
        request_id="request-1",
        payload={"number": 1},
    )

    reopened = Scribe(path, clock=fixed_clock, id_factory=lambda: "event-2")
    receipt = reopened.append(
        "event",
        request_id="request-2",
        payload={"number": 2},
    )

    assert receipt.sequence == 2
    assert len(reopened.verify()) == 2


def test_serializes_concurrent_appends(tmp_path: Path) -> None:
    path = tmp_path / "audit.db"
    Scribe(path)

    def append(number: int) -> None:
        Scribe(path).append(
            "event",
            request_id=f"request-{number}",
            payload={"number": number},
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(append, range(12)))

    records = Scribe(path).verify()
    assert [record.sequence for record in records] == list(range(1, 13))
    assert {record.payload["number"] for record in records} == set(range(12))


def test_serializes_concurrent_process_appends(tmp_path: Path) -> None:
    path = tmp_path / "audit.db"
    Scribe(path)

    with ProcessPoolExecutor(max_workers=3) as executor:
        list(executor.map(append_from_process, [(str(path), number) for number in range(6)]))

    records = Scribe(path).verify()
    assert [record.sequence for record in records] == list(range(1, 7))
    assert {record.payload["number"] for record in records} == set(range(6))


def test_rejects_non_json_and_non_finite_payloads(tmp_path: Path) -> None:
    scribe = Scribe(tmp_path / "audit.db")

    with pytest.raises(ValueError, match="finite JSON"):
        scribe.append("event", request_id="request", payload={"value": object()})
    with pytest.raises(ValueError, match="finite JSON"):
        scribe.append("event", request_id="request", payload={"value": float("nan")})


def test_tail_deletion_is_a_documented_hash_chain_limitation(tmp_path: Path) -> None:
    path = tmp_path / "audit.db"
    scribe = Scribe(path)
    scribe.append("event", request_id="request-1", payload={})
    scribe.append("event", request_id="request-2", payload={})

    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM audit_records WHERE sequence = 2")

    # Without an external trusted chain head, the remaining prefix is valid.
    assert len(scribe.verify()) == 1
