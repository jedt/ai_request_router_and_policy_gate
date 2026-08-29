from __future__ import annotations

import sys
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.table import Table
from llm_router.models import QueryProfile

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_router.defaults import DEFAULT_POLICY, DEFAULT_PROVIDERS
from llm_router.factory import build_router
from service.mock_openai import MockOpenAIService

MOCK_CLASSIFICATION_CASES: dict[str, QueryProfile] = {
    "test-case-1": QueryProfile(
        reasoning_depth=0.05,
        latency_sensitivity=0.95,
        cost_sensitivity=0.95,
    ),
    "test-case-2": QueryProfile(
        reasoning_depth=0.50,
        latency_sensitivity=0.60,
        cost_sensitivity=0.60,
    ),
    "test-case-3": QueryProfile(
        reasoning_depth=0.95,
        latency_sensitivity=0.10,
        cost_sensitivity=0.10,
    ),
}


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--test-case",
    type=click.IntRange(1, len(MOCK_CLASSIFICATION_CASES)),
    default=1,
    show_default=True,
    help="Mock classification test case to run.",
)
def main(test_case: int) -> None:
    """Run one semantic-routing test case against the mock OpenAI service."""
    user_request = f"test-case-{test_case}"
    mock_service = MockOpenAIService(MOCK_CLASSIFICATION_CASES)

    with httpx.Client(
        transport=httpx.MockTransport(mock_service.handle),
    ) as http_client:
        router, selector = build_router(
            http_client=http_client,
            providers=DEFAULT_PROVIDERS,
            policy=DEFAULT_POLICY,
            api_key="fake-key",
        )
        response = router.query(user_request)

    decision = selector.last_decision
    if decision is None:
        raise click.ClickException("The router did not produce a routing decision.")

    table = Table(title="LLM Routing Result", show_header=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    table.add_row("User query", user_request)
    table.add_row("Reasoning depth", f"{decision.profile.reasoning_depth:.2f}")
    table.add_row("Latency sensitivity", f"{decision.profile.latency_sensitivity:.2f}")
    table.add_row("Cost sensitivity", f"{decision.profile.cost_sensitivity:.2f}")
    table.add_row("Eligible providers", ", ".join(decision.eligible_provider_ids))
    table.add_row("Selected provider", decision.provider.id)
    table.add_row("Selected model", decision.provider.model)
    table.add_row("Routing score", f"{decision.score:.4f}")
    table.add_row("Response", str(response))
    table.add_row("Response metadata", str(response.metadata))
    Console().print(table)


if __name__ == "__main__":
    main()
