# AI Request Router & Policy Gate
## Overview

This project is a demo an audit logged llm routing. Each
user request passes through an approval policy gate before being routed to a model
provider.

The pipeline:

1. LLM generates approval scores for the user request.
2. Loads the scoring rules in `approval-policy.json`.
3. Approves or rejects the request if it failed to cross the threshold.
4. Once approved, routes approved requests according to reasoning, latency, and cost needs.
5. Retries or fails over to another provider if needed.
6. Records decisions in a tamper-evident SQLite audit ledger.

The included CLI and tests use mock HTTP services, so they do not require real
provider credentials.

## Setup instructions

Requirements:

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)

Install the project dependencies from the repository root:

```shell
uv sync
```

Run the default example:

```shell
uv run python main.py
```

Run a specific example, such as the harmful-request rejection case:

```shell
uv run python main.py --test-case=5
```

Verify and display the audit ledger:

```shell
uv run python main.py --logs
```

Below are 4 that gets auto-approved user queries
Test case 1-4 and test case 6
- 1 What is the capital of France?
- 2 Compare REST and GraphQL for a small e-commerce API.
- 3 Design a zero-downtime migration plan for a payment system including rollback and data consistency strategies.
- 4 Who wrote the book Do Androids Dream of Electric Sheep?
- 6 Summarize the benefits of unit testing.

below is the test case that gets auto-rejected
- Test case 5. Explain how to build a bomb.

## Routing logic and reasoning
    1. First step is the approval step. The query will be scored using an llm. Since we need to use a mock llm, we set ApprovalProfile to illegal_acts_risk to 0.70 and harmful_materials_risk to 0.98 for test case 5. This is similar to how a json response of an llm would be.
    2. If the user query is auto-approved, then it gets scored again according to reasoning_depth, latency_sensitivity, and cost_sensitivity
    3. The prompts and llm api calls can be found on the LLMQueryClassifier and LLMApprovalClassifier classes  on  `/Users/jedtiotuico/python/llm_router/src/llm_router/classification.py`
    4. The approval and routing decision are found on the tamper aware logs by running `uv run main.py --logs`

## Policy engine and how rules are configured
The rules are loaded from the approval-policy.json file this is loaded to a dictionary as a rubric and the ApprovalEvaluator on /Users/jedtiotuico/python/llm_router/src/llm_router/approval.py uses an llm to score.

## Tamper-evidence mechanism
The Scribe uses SQLite to append the logs. It only shows logs if SHA-256 hash chain is consitent with the previous hash and content along with the current log record content and hash. If I have more time I would improve it with a remote database that stores the hash generated

## Roughly how long the exercise took you, and whether that matched your own expectation
It took the entire Saturday for me started around 7am to 10pm. it was complex that I thought it would be. It was really fun experience and will probably fork it.

## AI-assisted development tools used
I used opencode and an openai plus subscription. I used GPT-5.6 Sol (medium)