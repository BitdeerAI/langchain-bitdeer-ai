# langchain-bitdeer-ai

[LangChain](https://www.langchain.com/) integration for [Bitdeer AI](https://www.bitdeer.ai).

> **Status: under development.** Not yet published to PyPI.

## Installation

```bash
pip install -U langchain-bitdeer-ai
```

## Authentication

Get an API key from Bitdeer AI and set it as an environment variable:

```bash
export BITDEER_API_KEY="your-api-key"
```

You can also pass it directly with `ChatBitdeerAI(api_key="...")`.

## Chat models

`ChatBitdeerAI` gives you Bitdeer AI's chat models through LangChain's standard chat model interface.
The `model` argument is required. Pass any chat model ID from the Bitdeer AI model catalog.

```python
from langchain_bitdeer_ai import ChatBitdeerAI

model = ChatBitdeerAI(model="zai-org/GLM-5.3-Flash")

response = model.invoke("What is the capital of France?")
print(response.content)
```

Streaming, async, batching, tool calling and structured output work as they do for any LangChain chat
model:

```python
for chunk in model.stream("Write a haiku about the sea."):
    print(chunk.text, end="")
```

### Reasoning output

Bitdeer AI models return their reasoning along with the answer. `ChatBitdeerAI` keeps it in
`additional_kwargs["reasoning_content"]` and in a standard `reasoning` content block:

```python
response = model.invoke("What is 2 + 3?")

print(response.additional_kwargs["reasoning_content"])

for block in response.content_blocks:
    if block["type"] == "reasoning":
        print(block["reasoning"])
```

When you stream, each chunk carries its part of the reasoning, and adding the chunks together gives the
full text.

`ChatBitdeerAI` uses the Chat Completions API. Parameters that only OpenAI's Responses API accepts, such as
`reasoning`, `include` and `truncation`, raise a `ValueError`. The Chat Completions parameter that
corresponds to `reasoning={"effort": ...}` is `reasoning_effort`:

```python
model = ChatBitdeerAI(model="zai-org/GLM-5.3-Flash", reasoning_effort="low")
```

### Token usage

`response.usage_metadata` reports input, output and total tokens, and reasoning tokens under
`output_token_details.reasoning`. Token usage is also reported when streaming. Pass `stream_usage=False`
to turn it off.

### Why not `ChatOpenAI(base_url=...)`?

Bitdeer AI's API is OpenAI-compatible, so `ChatOpenAI` can reach it. `ChatBitdeerAI` builds on the same
`langchain-openai` code, and adds:

- **Reasoning output.** `ChatOpenAI` drops the `reasoning_content` field that Bitdeer AI models return.
- **Complete token usage.** Reasoning tokens are reported, and streamed responses include usage by default.
- **Separate credentials.** It reads only `BITDEER_API_KEY` and `BITDEER_API_BASE`. It never sends your
  `OPENAI_API_KEY` to Bitdeer AI, and `OPENAI_BASE_URL` can't redirect it.
- **Correct tracing.** Runs show up in LangSmith under the `bitdeer_ai` provider instead of `openai`.

### Known issues

- With `DeepSeek-V4.1-Flash`, forcing a tool call (as `with_structured_output` does by default) can return
  the tool call as plain text. Use `method="json_schema"` for structured output with this model:

  ```python
  from pydantic import BaseModel


  class Person(BaseModel):
      name: str
      age: int


  model = ChatBitdeerAI(model="deepseek-ai/DeepSeek-V4.1-Flash")
  structured_model = model.with_structured_output(Person, method="json_schema")
  structured_model.invoke("Ada is 36 years old.")
  ```

## Configuration

| Argument   | Environment variable | Default                                |
| ---------- | -------------------- | -------------------------------------- |
| `api_key`  | `BITDEER_API_KEY`    | none (required)                        |
| `base_url` | `BITDEER_API_BASE`   | `https://api-inference.bitdeer.ai/v1`  |

Other arguments such as `temperature`, `max_tokens`, `timeout` and `max_retries` work as they do in
`langchain-openai`.

## Contributing

Issues and pull requests are welcome.

To set up a development environment, install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync --all-groups
make lint   # ruff and mypy
make test   # unit tests, no network access
```

Integration tests call the live API and need `BITDEER_API_KEY`. Without it they are skipped:

```bash
make integration_tests
```
