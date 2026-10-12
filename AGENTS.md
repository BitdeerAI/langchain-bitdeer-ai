# AGENTS.md

Guidance for coding agents working in this repository.

## Project

`langchain-bitdeer-ai` is the LangChain integration for [Bitdeer AI](https://www.bitdeer.ai). It is a thin
layer over `langchain-openai`, because the Bitdeer AI API is OpenAI-compatible:

- `langchain_bitdeer_ai/chat_models.py`: `ChatBitdeerAI`, a subclass of `BaseChatOpenAI`.
- `langchain_bitdeer_ai/embeddings.py`: `BitdeerAIEmbeddings`, a subclass of `OpenAIEmbeddings`.
- `langchain_bitdeer_ai/data/_profiles.py`: model profiles (context window, modalities, tool support),
  maintained by hand.
- `README.md`: the user documentation. It follows the layout of LangChain's integration pages.

Keep the code small. Override a base class method only when Bitdeer AI behaves differently from OpenAI, and
explain the difference in a comment.

Scope and conventions:

- Chat models come first, then embeddings. LangChain discourages text-completion LLMs, document loaders and
  similar components, so do not add them unless asked.
- Build on existing LangChain building blocks (`langchain-openai`) instead of writing a client from scratch.
  Be ready to explain what this package adds over the generic OpenAI-compatible route.
- Follow the layout and test structure of existing partner packages (for example `langchain-deepseek`,
  `langchain-xai`). Check one of them before inventing a convention.
- Only list or default to models Bitdeer AI sells today. Check model IDs against the `/v1/models` endpoint
  (needs an API key, returns IDs only) and take prices and context windows from another official Bitdeer AI
  source, never from memory or old docs.
- LangChain's APIs and contribution requirements change. When a decision depends on them (base classes,
  standard-test properties, packaging or listing rules), check the current docs or source and cite what you
  found.

## Working style

These bias toward caution over speed; for trivial tasks, use judgment.

- **Think before coding.** State assumptions. If a request has several readings, lay them out instead of
  picking silently. If something is unclear, ask.
- **Simplicity first.** Write the minimum code that solves the problem: no unrequested features,
  configurability, single-use abstractions, or error handling for impossible cases.
- **Surgical changes.** Every changed line should trace back to the request. Do not reformat or refactor
  adjacent code; match the existing style. Remove imports and code that your change orphaned, but only
  mention pre-existing dead code.
- **Goal-driven execution.** Turn tasks into verifiable goals ("fix the bug" becomes "write a test that
  reproduces it, then make it pass"). For multi-step work, state a brief plan with a check per step.

## Commands

Use `uv` and `make`. Do not use `pip`.

```bash
uv sync --all-groups
make format             # ruff format + ruff check --fix
make lint               # import checks, ruff, mypy
make test               # unit tests, network disabled
make test TEST_FILE=tests/unit_tests/test_chat_models.py
make integration_tests  # live API, needs BITDEER_API_KEY
```

Run `make format lint test` before you finish a change, and `make check_imports` (every module imports
cleanly) before you commit. Commits that only touch Markdown can skip the checks.

## Invariants

These are easy to break by accident. Each one exists for a reason, so do not "simplify" it away.

- **Never fall back to `OPENAI_*` settings.** Both classes replace the parent `validate_environment` so that
  only `api_key`/`BITDEER_API_KEY` and `base_url`/`BITDEER_API_BASE` are read. The `openai` client reads
  `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` by itself, so they are cleared after the client is built. A missing
  key raises `ValueError`; it must never silently pick up `OPENAI_API_KEY`, because that would send an
  OpenAI key to Bitdeer AI.
- **Chat Completions only.** `_use_responses_api` always returns `False` and raises `ValueError` for
  parameters only the Responses API accepts (`reasoning`, `include`, `truncation`, ...).
- **Give Pydantic validators unique names.** A validator with the same name as one in the parent class
  replaces it instead of running after it (see `_set_bitdeer_ai_version`).
- **Provider name is `bitdeer_ai`** in `ls_provider` and `response_metadata["model_provider"]`.
- **Reasoning output** goes to `additional_kwargs["reasoning_content"]`, and reasoning tokens go to
  `usage_metadata["output_token_details"]["reasoning"]`, both for `invoke` and for `stream`.
- **Model profiles** are keyed by the model ID that `/v1/models` returns. Take values only from official
  Bitdeer AI sources, and leave a key out when it is not known. Do not guess.

## Naming

- Display name: **Bitdeer AI**, never just "Bitdeer" (on its own that names the bitcoin-mining business).
  If an upstream maintainer asks for a different spelling, ask the user before changing anything.
- Hyphen form where hyphens are allowed (`langchain-bitdeer-ai`, `bitdeer-ai`), underscore form only where
  they are not (`langchain_bitdeer_ai`, `bitdeer_ai`). Never a bare `bitdeer` identifier.
- Classes: `ChatBitdeerAI`, `BitdeerAIEmbeddings`. Environment variables: `BITDEER_API_KEY`,
  `BITDEER_API_BASE`. Do not rename them.

## Code style

- Python 3.10+. Type hints on everything (`mypy` runs with `disallow_untyped_defs`).
- Ruff with `select = ["ALL"]`: fix the code instead of adding `noqa`, unless the rule is wrong for that
  line, and then say why.
- Google-style docstrings. Types go in signatures, not docstrings. American English.
- Absolute imports only. Do not import from `langchain` (except `langchain.agents` and `langchain.tools`)
  or `langchain_experimental`; `make lint` checks this.
- Error messages go in a `msg` variable before `raise`.
- Public API is what `langchain_bitdeer_ai/__init__.py` exports. Do not change existing signatures; add new
  parameters as keyword arguments with defaults.

## Tests

- `tests/unit_tests/`: no network (pytest-socket blocks it). Every behavior change needs a unit test that
  fails without the change.
- `tests/integration_tests/`: call the live API. They are skipped when `BITDEER_API_KEY` is unset, so they
  must never be the only test for a change. Share the module-level `rate_limiter` to stay within the test
  account's limits.
- `test_standard.py` in both folders runs the `langchain-tests` standard suites. When a standard test
  cannot pass, mark it `xfail(strict=True)` with the reason instead of deleting it.
- Snapshot files under `__snapshots__/` change only together with the behavior they record.

## Documentation

When behavior a user can see changes, update `README.md` in the same commit: the feature table, the
examples, or "Known issues". Keep every README example runnable.

## Commits

- [Conventional Commits](https://www.conventionalcommits.org/): `feat(chat): ...`, `fix(embeddings): ...`,
  `test: ...`, `docs: ...`, `chore: ...`. Scopes in use: `chat`, `embeddings`.
- Write an imperative summary line and a body that explains what changed and why. Keep messages
  self-contained, with no issue, PR or MR numbers and no links to non-public systems.
- History is linear: rebase instead of merging.
- Everything committed is public, including commit messages. Never commit API keys, hostnames, IP addresses,
  people's names, mentions of internal tooling or processes, or other credentials, including in tests and
  examples.
- Do not run integration tests unless asked.
- Releases are tag-driven and published with PyPI Trusted Publishing. Never add a long-lived PyPI token, and
  do not tag or publish unless asked.
