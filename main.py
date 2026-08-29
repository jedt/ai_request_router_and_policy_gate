from __future__ import annotations

import sys
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_router.defaults import DEFAULT_POLICY, DEFAULT_PROVIDERS
from llm_router.factory import build_router
from llm_router.models import QueryProfile
from service.classification_simulations import (
    simulate_high_reasoning_user_query_classification,
    simulate_low_reasoning_user_query_classification,
    simulate_medium_reasoning_user_query_classification,
)
from service.mock_openai import MockOpenAIService


def select_test_case(test_case: int) -> tuple[str, QueryProfile]:
    match test_case:
        case 1:
            return (
                "What is the capital of France?",
                simulate_low_reasoning_user_query_classification(),
            )
        case 2:
            return (
                "Compare REST and GraphQL for a small e-commerce API.",
                simulate_medium_reasoning_user_query_classification(),
            )
        case 3:
            return (
                "Design a zero-downtime migration plan for a payment system including rollback and data consistency strategies.",
                simulate_high_reasoning_user_query_classification(),
            )
        case _:
            raise ValueError(f"Unsupported test case: {test_case}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--test-case",
    type=click.IntRange(1, 3),
    default=1,
    show_default=True,
    help="Mock classification test case to run.",
)
def main(test_case: int) -> None:
    """Run one semantic-routing test case against the mock OpenAI service."""
    user_request, classification_profile = select_test_case(test_case)
    mock_service = MockOpenAIService(classification_profile)

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
    table.add_row("Test case", str(test_case))
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
