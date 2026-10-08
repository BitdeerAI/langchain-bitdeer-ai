# langchain-bitdeer-ai

[LangChain](https://www.langchain.com/) integration for [Bitdeer AI](https://www.bitdeer.ai).

> **Status: under development.** Not yet published to PyPI.

This guide gives a quick overview of getting started with Bitdeer AI
[chat models](https://docs.langchain.com/oss/python/langchain/models) in LangChain.

## Overview

### Details

| Class           | Package                | Serializable | JS support |
| :-------------- | :--------------------- | :----------: | :--------: |
| `ChatBitdeerAI` | `langchain-bitdeer-ai` |     beta     |     ❌     |

### Features

| [Tool calling](https://docs.langchain.com/oss/python/langchain/tools) | [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output) | [Image input](https://docs.langchain.com/oss/python/langchain/messages#multimodal) | Audio input | Video input | [Token-level streaming](https://docs.langchain.com/oss/python/langchain/streaming) | Native async | [Token usage](https://docs.langchain.com/oss/python/langchain/models#token-usage) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |

Image input depends on the model. The [Bitdeer AI model catalog](https://developers.bitdeer.ai/docs/api/models-catalog)
lists which models accept images.

## Setup

To access Bitdeer AI models, you need a Bitdeer AI account and an API key, and you need to install the
`langchain-bitdeer-ai` integration package.

### Credentials

Get an API key from Bitdeer AI and set it as the `BITDEER_API_KEY` environment variable:

```python
import getpass
import os

if "BITDEER_API_KEY" not in os.environ:
    os.environ["BITDEER_API_KEY"] = getpass.getpass("Enter your Bitdeer AI API key: ")
```

You can also pass it directly with `ChatBitdeerAI(api_key="...")`.

To trace your model calls automatically, set your [LangSmith](https://docs.langchain.com/langsmith/observability)
API key:

```python
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
os.environ["LANGSMITH_TRACING"] = "true"
```

Runs show up in LangSmith under the `bitdeer_ai` provider.

### Installation

The LangChain Bitdeer AI integration lives in the `langchain-bitdeer-ai` package:

```bash
pip install -U langchain-bitdeer-ai
```

```bash
uv add langchain-bitdeer-ai
```

## Instantiation

The `model` argument is required. Pass any chat model ID from the
[Bitdeer AI model catalog](https://developers.bitdeer.ai/docs/api/models-catalog):

```python
from langchain_bitdeer_ai import ChatBitdeerAI

model = ChatBitdeerAI(
    model="zai-org/GLM-5.3-Flash",
    temperature=0,
    timeout=None,
    max_tokens=1024,
    max_retries=2,
)
```

These and other arguments work as they do in `langchain-openai`.

## Invocation

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]
response = model.invoke(messages)

print(response.text)
print(response.content_blocks)
```

Streaming, async, batching, tool calling and structured output work as they do for any LangChain chat
model:

```python
for chunk in model.stream("Write a haiku about the sea."):
    print(chunk.text, end="")
```

## Reasoning output

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

## Token usage

`response.usage_metadata` reports input, output and total tokens, and reasoning tokens under
`output_token_details.reasoning`. Token usage is also reported when streaming. Pass `stream_usage=False`
to turn it off.

## Custom base URL

| Argument   | Environment variable | Default                                |
| ---------- | -------------------- | -------------------------------------- |
| `api_key`  | `BITDEER_API_KEY`    | none (required)                        |
| `base_url` | `BITDEER_API_BASE`   | `https://api-inference.bitdeer.ai/v1`  |

`ChatBitdeerAI` reads only these variables. It never falls back to `OPENAI_API_KEY` or `OPENAI_BASE_URL`,
so an OpenAI key in your environment is never sent to Bitdeer AI.

## Known issues

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

- Structured output with `method="json_schema"` and JSON mode can print a Pydantic serializer warning
  about a `metadata` field (`PydanticSerializationUnexpectedValue`). The API returns `metadata` in a shape
  the `openai` SDK doesn't expect. The warning is harmless, and the output is correct.

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
