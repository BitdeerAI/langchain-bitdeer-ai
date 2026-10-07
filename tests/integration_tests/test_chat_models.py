"""Smoke tests across Bitdeer AI model families."""

import pytest
from langchain_core.messages import AIMessageChunk

from langchain_bitdeer_ai import ChatBitdeerAI

# The standard tests cover `zai-org/GLM-5.3-Flash`; these cover one model from
# each other family.
MODELS = ["deepseek-ai/DeepSeek-V4-Flash", "Qwen/Qwen3.8-27B"]

PROMPT = "What is 2 + 3? Answer with the number only."


@pytest.mark.parametrize("model_name", MODELS)
def test_invoke(model_name: str) -> None:
    """`invoke` returns an answer, reasoning output and token usage."""
    llm = ChatBitdeerAI(model=model_name, max_tokens=512)
    result = llm.invoke(PROMPT)
    assert result.content
    assert result.additional_kwargs.get("reasoning_content")
    assert result.usage_metadata is not None
    assert result.usage_metadata["output_tokens"] > 0


@pytest.mark.parametrize("model_name", MODELS)
def test_stream(model_name: str) -> None:
    """Summed stream chunks hold an answer, reasoning output and token usage."""
    llm = ChatBitdeerAI(model=model_name, max_tokens=512)
    full: AIMessageChunk | None = None
    for chunk in llm.stream(PROMPT):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.content
    assert full.additional_kwargs.get("reasoning_content")
    assert full.usage_metadata is not None
    assert full.usage_metadata["output_tokens"] > 0
