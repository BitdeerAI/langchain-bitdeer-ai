"""Behavior tests across every Bitdeer AI chat model."""

import base64
import struct
import zlib
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_bitdeer_ai import ChatBitdeerAI

# The standard tests cover `zai-org/GLM-5.3-Flash` in depth; these run a smaller
# set of scenarios on every chat model.
MODELS = [
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "deepseek-ai/DeepSeek-V4-Flash",
    "zai-org/GLM-5.3",
    "zai-org/GLM-5.3-Flash",
    "Qwen/Qwen3.8-27B",
]
IMAGE_MODELS = [
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "zai-org/GLM-5.3-Flash",
    "Qwen/Qwen3.8-27B",
]
# With a forced `tool_choice`, this model returns the call as plain text in
# `content` instead of `tool_calls`.
FORCED_TOOL_BROKEN = "deepseek-ai/DeepSeek-V4.1-Flash"

PROMPT = "What is 2 + 3? Answer with the number only."

rate_limiter = InMemoryRateLimiter(requests_per_second=0.5)


def _model(name: str, **kwargs: Any) -> ChatBitdeerAI:
    return ChatBitdeerAI(
        model=name, max_tokens=2048, rate_limiter=rate_limiter, **kwargs
    )


def _skip_forced_tool_broken(models: list[str]) -> list[object]:
    return [
        pytest.param(
            m,
            marks=pytest.mark.skip(
                reason="forced tool calls are returned as plain text"
            ),
        )
        if m == FORCED_TOOL_BROKEN
        else m
        for m in models
    ]


def _solid_png(rgb: tuple[int, int, int], size: int = 32) -> str:
    """Return a base64 PNG of one color, built without extra dependencies."""
    row = b"\x00" + bytes(rgb) * size

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is sunny in {city}."


class Person(BaseModel):
    """A person."""

    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age in years")


@pytest.mark.parametrize("model_name", MODELS)
def test_invoke(model_name: str) -> None:
    """`invoke` returns an answer, reasoning output and token usage."""
    result = _model(model_name).invoke(PROMPT)
    assert result.content
    assert result.additional_kwargs.get("reasoning_content")
    assert result.usage_metadata is not None
    assert result.usage_metadata["output_tokens"] > 0


@pytest.mark.parametrize("model_name", MODELS)
def test_stream(model_name: str) -> None:
    """Summed stream chunks hold an answer, reasoning output and token usage."""
    full: AIMessageChunk | None = None
    for chunk in _model(model_name).stream(PROMPT):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.content
    assert full.additional_kwargs.get("reasoning_content")
    assert full.usage_metadata is not None
    assert full.usage_metadata["output_tokens"] > 0


@pytest.mark.parametrize("model_name", MODELS)
def test_tool_calling(model_name: str) -> None:
    """The model calls a bound tool when the choice is left to it."""
    llm = _model(model_name).bind_tools([get_weather])
    result = llm.invoke("What is the weather in Paris? Use the tool.")
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "get_weather"
    assert "city" in result.tool_calls[0]["args"]


@pytest.mark.parametrize("model_name", _skip_forced_tool_broken(MODELS))
def test_forced_tool_choice(model_name: str) -> None:
    """`tool_choice` forces a tool call."""
    llm = _model(model_name).bind_tools([get_weather], tool_choice="get_weather")
    result = llm.invoke("Hello")
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "get_weather"


@pytest.mark.parametrize("model_name", _skip_forced_tool_broken(MODELS))
def test_structured_output_function_calling(model_name: str) -> None:
    """`with_structured_output` works through forced tool calling (the default)."""
    llm = _model(model_name).with_structured_output(Person)
    result = llm.invoke("Alice is 30 years old.")
    assert result == Person(name="Alice", age=30)


@pytest.mark.parametrize("model_name", MODELS)
def test_structured_output_json_schema(model_name: str) -> None:
    """`with_structured_output(method="json_schema")` works on every model."""
    llm = _model(model_name).with_structured_output(Person, method="json_schema")
    result = llm.invoke("Alice is 30 years old.")
    assert result == Person(name="Alice", age=30)


@pytest.mark.parametrize("model_name", IMAGE_MODELS)
def test_image_input(model_name: str) -> None:
    """Models with vision describe a base64 `image_url` correctly."""
    message = HumanMessage(
        content=[
            {"type": "text", "text": "What color is this image? One word."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_solid_png((255, 0, 0))}"
                },
            },
        ]
    )
    result = _model(model_name).invoke([message])
    assert "red" in str(result.content).lower()


@pytest.mark.parametrize("model_name", MODELS)
def test_stream_v1_has_tool_call_block(model_name: str) -> None:
    """With `output_version="v1"`, a streamed tool call gets its own block."""
    llm = _model(model_name, output_version="v1").bind_tools([get_weather])
    full: Any = None
    for chunk in llm.stream("What is the weather in Paris? Use the tool."):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.tool_calls
    assert any(b["type"] == "tool_call" for b in full.content_blocks)
