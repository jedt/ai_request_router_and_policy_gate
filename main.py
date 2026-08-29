from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_router.defaults import (
    DEFAULT_POLICY,
    DEFAULT_PROVIDERS,
    FAILOVER_TEST_PROVIDERS,
)
from llm_router.factory import build_router
from llm_router.models import QueryProfile
from llm_router.provider_engine import CompletionProvider
from service.mock_api_service_provider import (
    MockAPIServiceProvider,
    MockGeminiService,
    MockGeminiLLM,
    MockOpenAIService,
)
from utils.scribe import AuditIntegrityError, AuditRecord, Scribe


AUDIT_LOG_PATH = Path("llm-router-audit.db")


def select_test_case(test_case: int) -> tuple[str, QueryProfile]:
    match test_case:
        case 1:
            return (
                "What is the capital of France?",
                QueryProfile(
                    reasoning_depth=0.05,
                    latency_sensitivity=0.95,
                    cost_sensitivity=0.95,
                ),
            )
        case 2:
            return (
                "Compare REST and GraphQL for a small e-commerce API.",
                QueryProfile(
                    reasoning_depth=0.50,
                    latency_sensitivity=0.60,
                    cost_sensitivity=0.60,
                ),
            )
        case 3:
            return (
                "Design a zero-downtime migration plan for a payment system including rollback and data consistency strategies.",
                QueryProfile(
                    reasoning_depth=0.95,
                    latency_sensitivity=0.10,
                    cost_sensitivity=0.10,
                ),
            )
        case 4:
            return (
                "Who wrote the book Do Androids Dream of Electric Sheep?",
                QueryProfile(
                    reasoning_depth=0.10,
                    latency_sensitivity=0.90,
                    cost_sensitivity=0.80,
                ),
            )
        case _:
            raise ValueError(f"Unsupported test case: {test_case}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--test-case",
    type=click.IntRange(1, 4),
    default=1,
    show_default=True,
    help="Mock classification test case to run.",
)
@click.option(
    "--logs",
    is_flag=True,
    help="Verify and display the tamper-evident audit trail, then exit.",
)
def main(test_case: int, logs: bool) -> None:
    if logs:
        display_audit_logs()
        return

    user_request, classification_profile = select_test_case(test_case)
    providers = FAILOVER_TEST_PROVIDERS if test_case == 4 else DEFAULT_PROVIDERS
    failing_models = (
        frozenset({FAILOVER_TEST_PROVIDERS[0].model})
        if test_case == 4
        else frozenset()
    )
    openai_service = MockOpenAIService(
        classification_profile,
        failing_models=failing_models,
        failure="budget_exhausted",
    )
    gemini_service = MockGeminiService()
    mock_services = MockAPIServiceProvider(openai_service, gemini_service)

    with httpx.Client(
        transport=httpx.MockTransport(mock_services.handle),
    ) as http_client:
        provider_llms: dict[str, CompletionProvider] | None = (
            {
                "gemini-flash": MockGeminiLLM(
                    http_client,
                    FAILOVER_TEST_PROVIDERS[1].model,
                )
            }
            if test_case == 4
            else None
        )
        router, selector = build_router(
            http_client=http_client,
            providers=providers,
            policy=DEFAULT_POLICY,
            api_key="fake-key",
            scribe=Scribe(AUDIT_LOG_PATH),
            sleep=lambda _: None,
            provider_llms=provider_llms,
        )
        response = router.query(user_request)

    decision = selector.last_decision
    if decision is None:
        raise click.ClickException("The router did not produce a routing decision.")

    table = Table(title="LLM Routing Result", show_header=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    table.add_row("Test case", str(test_case))
    table.add_row("User query", user_request)
    table.add_row("Reasoning depth", f"{decision.profile.reasoning_depth:.2f}")
    table.add_row("Latency sensitivity", f"{decision.profile.latency_sensitivity:.2f}")
    table.add_row("Cost sensitivity", f"{decision.profile.cost_sensitivity:.2f}")
    table.add_row("Eligible providers", ", ".join(decision.eligible_provider_ids))
    table.add_row("Selected provider", decision.provider.id)
    table.add_row("Selected model", decision.provider.model)
    table.add_row("Routing score", f"{decision.score:.4f}")
    if response.metadata and response.metadata.get("fallback_reason"):
        table.add_row("Fallback reason", str(response.metadata["fallback_reason"]))
        table.add_row(
            "Serving provider", str(response.metadata.get("provider_id", "unknown"))
        )
        table.add_row("Serving model", str(response.metadata.get("model", "unknown")))
        table.add_row("Retry count", str(response.metadata["retry_count"]))
    table.add_row("Response", str(response))
    table.add_row("Response metadata", str(response.metadata))
    Console().print(table)


def display_audit_logs() -> None:
    try:
        records = Scribe(AUDIT_LOG_PATH).verify()
    except AuditIntegrityError as exc:
        raise click.ClickException(
            f"Audit integrity verification failed: {exc}"
        ) from exc

    if not records:
        click.echo("Audit trail is empty.")
        return

    table = Table(title="Tamper-Evident Audit Trail", show_lines=True)
    table.add_column("Sequence", style="bold cyan", no_wrap=True)
    table.add_column("Field", style="bold")
    table.add_column("Value", overflow="fold")

    for record in records:
        for index, (field, value) in enumerate(_audit_record_fields(record)):
            table.add_row(
                str(record.sequence) if index == 0 else "",
                field,
                value,
            )

    Console().print(table)


def _audit_record_fields(record: AuditRecord) -> tuple[tuple[str, str], ...]:
    return (
        ("Timestamp", record.occurred_at),
        ("Event", record.event_type),
        ("Event ID", record.event_id),
        ("Request ID", record.request_id),
        ("Decision ID", record.decision_id or "-"),
        (
            "Payload",
            json.dumps(record.payload, indent=2, sort_keys=True, ensure_ascii=True),
        ),
        ("Previous hash", record.previous_hash),
        ("Record hash", record.record_hash),
    )


if __name__ == "__main__":
    main()
