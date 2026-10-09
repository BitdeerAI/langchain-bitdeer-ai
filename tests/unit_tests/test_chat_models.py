"""Unit tests for `ChatBitdeerAI`."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.language_models import ModelProfile
from langchain_core.load import dumpd, dumps, load
from langchain_core.messages import AIMessage, AIMessageChunk
from openai.types.chat import ChatCompletion
from pydantic import SecretStr, ValidationError

from langchain_bitdeer_ai import ChatBitdeerAI
from langchain_bitdeer_ai.chat_models import DEFAULT_API_BASE
from langchain_bitdeer_ai.data._profiles import _PROFILES

MODEL_NAME = "zai-org/GLM-5.3-Flash"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test without Bitdeer AI or OpenAI settings in the environment."""
    for var in (
        "BITDEER_API_KEY",
        "BITDEER_API_BASE",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_model_is_required() -> None:
    """Construction fails without a model, naming the missing field."""
    with pytest.raises(ValidationError, match="model"):
        ChatBitdeerAI(api_key=SecretStr("test"))  # type: ignore[call-arg]


def test_model_is_sent() -> None:
    """The model ID goes into the request payload."""
    llm = ChatBitdeerAI(model=MODEL_NAME, api_key=SecretStr("test"))
    assert llm._get_request_payload("hi")["model"] == MODEL_NAME


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The key is read from `BITDEER_API_KEY` when not passed."""
    monkeypatch.setenv("BITDEER_API_KEY", "env-key")
    llm = ChatBitdeerAI(model=MODEL_NAME)
    assert llm.root_client.api_key == "env-key"
    assert llm.root_async_client.api_key == "env-key"


def test_explicit_api_key_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A passed `api_key` takes precedence over `BITDEER_API_KEY`."""
    monkeypatch.setenv("BITDEER_API_KEY", "env-key")
    llm = ChatBitdeerAI(model=MODEL_NAME, api_key=SecretStr("arg-key"))
    assert llm.root_client.api_key == "arg-key"
    assert llm.root_async_client.api_key == "arg-key"


def test_api_key_is_not_leaked() -> None:
    """The key does not appear in repr, str or serialized output."""
    llm = ChatBitdeerAI(model=MODEL_NAME, api_key=SecretStr("secret-value"))
    assert "secret-value" not in repr(llm)
    assert "secret-value" not in str(llm)
    assert "secret-value" not in dumps(llm)


def test_missing_api_key() -> None:
    """Construction without a key fails with an error naming `BITDEER_API_KEY`."""
    with pytest.raises(ValueError, match="BITDEER_API_KEY"):
        ChatBitdeerAI(model=MODEL_NAME)


def test_openai_api_key_is_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """`OPENAI_API_KEY` is neither a fallback nor sent alongside the Bitdeer AI key."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    with pytest.raises(ValueError, match="BITDEER_API_KEY"):
        ChatBitdeerAI(model=MODEL_NAME)

    monkeypatch.setenv("BITDEER_API_KEY", "env-key")
    llm = ChatBitdeerAI(model=MODEL_NAME)
    assert llm.root_client.api_key == "env-key"
    assert llm.root_async_client.api_key == "env-key"


@pytest.mark.parametrize("var", ["OPENAI_BASE_URL", "OPENAI_API_BASE"])
def test_openai_base_url_is_not_used(monkeypatch: pytest.MonkeyPatch, var: str) -> None:
    """OpenAI base URL settings do not redirect requests."""
    monkeypatch.setenv(var, "https://other.example.com/v1")
    llm = ChatBitdeerAI(model=MODEL_NAME, api_key=SecretStr("test"))
    assert str(llm.root_client.base_url) == f"{DEFAULT_API_BASE}/"
    assert str(llm.root_async_client.base_url) == f"{DEFAULT_API_BASE}/"


def test_openai_org_and_project_are_not_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """`OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` do not become request headers."""
    monkeypatch.setenv("OPENAI_ORG_ID", "org-openai")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-openai")
    requests: list[httpx.Request] = []
    _model_with_transport(requests).invoke("hi")
    assert "openai-organization" not in requests[0].headers
    assert "openai-project" not in requests[0].headers


def test_default_base_url() -> None:
    """Requests go to the Bitdeer AI endpoint by default."""
    llm = ChatBitdeerAI(model=MODEL_NAME, api_key=SecretStr("test"))
    assert llm.bitdeer_ai_api_base == DEFAULT_API_BASE
    assert str(llm.root_client.base_url) == f"{DEFAULT_API_BASE}/"


def test_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """`BITDEER_API_BASE` overrides the default endpoint."""
    monkeypatch.setenv("BITDEER_API_BASE", "https://env.example.com/v1")
    llm = ChatBitdeerAI(model=MODEL_NAME, api_key=SecretStr("test"))
    assert str(llm.root_client.base_url) == "https://env.example.com/v1/"
    assert str(llm.root_async_client.base_url) == "https://env.example.com/v1/"


def test_base_url_argument_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A passed `base_url` takes precedence over `BITDEER_API_BASE`."""
    monkeypatch.setenv("BITDEER_API_BASE", "https://env.example.com/v1")
    llm = ChatBitdeerAI(
        model=MODEL_NAME,
        api_key=SecretStr("test"),
        base_url="https://proxy.example.com/v1",
    )
    assert str(llm.root_client.base_url) == "https://proxy.example.com/v1/"
    assert str(llm.root_async_client.base_url) == "https://proxy.example.com/v1/"


def test_client_options_are_passed_through() -> None:
    """Client options reach both the sync and async `openai` clients."""
    timeout = 12.0
    max_retries = 5
    llm = ChatBitdeerAI(
        model=MODEL_NAME,
        api_key=SecretStr("test"),
        timeout=timeout,
        max_retries=max_retries,
        default_headers={"X-Test": "1"},
        default_query={"q": "1"},
    )
    for client in (llm.root_client, llm.root_async_client):
        assert client.timeout == timeout
        assert client.max_retries == max_retries
        assert client.default_headers["X-Test"] == "1"
        assert client.default_query == {"q": "1"}


def test_stream_usage_on_by_default() -> None:
    """Streaming asks for token usage unless the user opts out."""
    llm = ChatBitdeerAI(model=MODEL_NAME, api_key=SecretStr("test"))
    assert llm.stream_usage is True
    assert llm._should_stream_usage() is True


def test_stream_usage_opt_out() -> None:
    """`stream_usage=False` turns off streamed token usage."""
    llm = ChatBitdeerAI(model=MODEL_NAME, api_key=SecretStr("test"), stream_usage=False)
    assert llm.stream_usage is False
    assert llm._should_stream_usage() is False


def test_ls_params() -> None:
    """Tracing reports the `bitdeer_ai` provider and the model ID."""
    llm = ChatBitdeerAI(model=MODEL_NAME, api_key=SecretStr("test"))
    ls_params = llm._get_ls_params()
    assert ls_params["ls_provider"] == "bitdeer_ai"
    assert ls_params["ls_model_name"] == MODEL_NAME


def test_serialization_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model serializes under its own namespace and loads back."""
    monkeypatch.setenv("BITDEER_API_KEY", "env-key")
    llm = ChatBitdeerAI(model=MODEL_NAME)
    serialized = dumpd(llm)
    assert serialized["id"] == ["langchain_bitdeer_ai", "chat_models", "ChatBitdeerAI"]
    assert serialized["kwargs"]["bitdeer_ai_api_key"] == {
        "lc": 1,
        "type": "secret",
        "id": ["BITDEER_API_KEY"],
    }

    loaded = load(
        serialized,
        valid_namespaces=["langchain_bitdeer_ai"],
        allowed_objects=[ChatBitdeerAI],
        secrets_from_env=True,
    )
    assert isinstance(loaded, ChatBitdeerAI)
    assert loaded.model_name == MODEL_NAME
    assert loaded.root_client.api_key == "env-key"


# Response shapes below follow what the Bitdeer AI API returns: reasoning in
# `reasoning_content`, and reasoning tokens at the top level of `usage`.
USAGE = {
    "prompt_tokens": 40,
    "completion_tokens": 21,
    "total_tokens": 61,
    "reasoning_tokens": 19,
    "prompt_tokens_details": {"cached_tokens": 8},
}


def _completion(message: dict) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": USAGE,
    }


def _chunk(delta: dict | None, finish_reason: str | None = None) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL_NAME,
        "choices": (
            []
            if delta is None
            else [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
        ),
        **({"usage": USAGE} if delta is None else {}),
    }


COMPLETION_WITH_REASONING = _completion(
    {"role": "assistant", "content": "5", "reasoning_content": "2+3=5"}
)

STREAM_CHUNKS = [
    _chunk({"role": "assistant", "content": "", "reasoning_content": "2+3"}),
    _chunk({"reasoning_content": "=5"}),
    _chunk({"content": "5"}),
    _chunk({}, finish_reason="stop"),
    _chunk(None),  # trailing usage chunk, no choices
]


def _sse(chunks: list[dict]) -> bytes:
    events = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    return "".join([*events, "data: [DONE]\n\n"]).encode()


def _model_with_transport(
    requests: list[httpx.Request],
    *,
    completion: dict = COMPLETION_WITH_REASONING,
    stream_chunks: list[dict] = STREAM_CHUNKS,
    **kwargs: Any,
) -> ChatBitdeerAI:
    """Return a model whose HTTP calls are answered locally and recorded."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if json.loads(request.content).get("stream"):
            return httpx.Response(
                200,
                content=_sse(stream_chunks),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=completion)

    transport = httpx.MockTransport(handler)
    return ChatBitdeerAI(
        model=MODEL_NAME,
        api_key=SecretStr("test"),
        http_client=httpx.Client(transport=transport),
        http_async_client=httpx.AsyncClient(transport=transport),
        **kwargs,
    )


def test_always_uses_chat_completions() -> None:
    """`use_responses_api=True` still sends requests to Chat Completions."""
    requests: list[httpx.Request] = []
    llm = _model_with_transport(requests, use_responses_api=True)
    llm.invoke("hi")
    list(llm.stream("hi"))
    assert [request.url.path for request in requests] == [
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]


REASONING = {"reasoning": {"effort": "low"}}


@pytest.mark.parametrize(
    ("init_kwargs", "call_kwargs"),
    [(REASONING, {}), ({}, REASONING)],
    ids=["at_init", "at_call"],
)
def test_reasoning_param_is_rejected(init_kwargs: dict, call_kwargs: dict) -> None:
    """`reasoning=` raises a clear error for invoke and stream, and sends nothing."""
    requests: list[httpx.Request] = []
    llm = _model_with_transport(requests, **init_kwargs)
    with pytest.raises(ValueError, match=r"`reasoning`.*`reasoning_effort`"):
        llm.invoke("hi", **call_kwargs)
    with pytest.raises(ValueError, match=r"`reasoning`.*`reasoning_effort`"):
        list(llm.stream("hi", **call_kwargs))
    assert requests == []


@pytest.mark.parametrize(
    ("param", "value"),
    [("include", ["reasoning.encrypted_content"]), ("truncation", "auto")],
)
def test_responses_only_params_are_rejected(param: str, value: Any) -> None:
    """Other Responses-only parameters raise an error that names them."""
    requests: list[httpx.Request] = []
    llm = _model_with_transport(requests, **{param: value})
    with pytest.raises(ValueError, match=f"`{param}`"):
        llm.invoke("hi")
    assert requests == []


def test_reasoning_effort_is_sent() -> None:
    """`reasoning_effort` is the Chat Completions parameter and reaches the API."""
    requests: list[httpx.Request] = []
    _model_with_transport(requests, reasoning_effort="low").invoke("hi")
    assert requests[0].url.path == "/v1/chat/completions"
    assert json.loads(requests[0].content)["reasoning_effort"] == "low"


def test_create_chat_result_reasoning() -> None:
    """Reasoning output and Bitdeer AI usage fields are mapped onto the message."""
    llm = ChatBitdeerAI(model=MODEL_NAME, api_key=SecretStr("test"))
    response = ChatCompletion.model_validate(COMPLETION_WITH_REASONING)
    message = llm._create_chat_result(response).generations[0].message

    assert isinstance(message, AIMessage)
    assert message.content == "5"
    assert message.additional_kwargs["reasoning_content"] == "2+3=5"
    assert message.response_metadata["model_provider"] == "bitdeer_ai"
    assert {"type": "reasoning", "reasoning": "2+3=5"}.items() <= next(
        block for block in message.content_blocks if block["type"] == "reasoning"
    ).items()
    assert message.usage_metadata == {
        "input_tokens": 40,
        "output_tokens": 21,
        "total_tokens": 61,
        "input_token_details": {"cache_read": 8},
        "output_token_details": {"reasoning": 19},
    }


def test_create_chat_result_from_dict() -> None:
    """Dict responses are handled the same way as `openai` response objects."""
    llm = ChatBitdeerAI(model=MODEL_NAME, api_key=SecretStr("test"))
    message = llm._create_chat_result(COMPLETION_WITH_REASONING).generations[0].message
    assert message.additional_kwargs["reasoning_content"] == "2+3=5"
    assert message.response_metadata["model_provider"] == "bitdeer_ai"


def test_create_chat_result_without_reasoning() -> None:
    """Without `reasoning_content` there is no reasoning key or block."""
    llm = ChatBitdeerAI(model=MODEL_NAME, api_key=SecretStr("test"))
    response = ChatCompletion.model_validate(
        _completion({"role": "assistant", "content": "5"})
    )
    message = llm._create_chat_result(response).generations[0].message
    assert "reasoning_content" not in message.additional_kwargs
    assert all(block["type"] != "reasoning" for block in message.content_blocks)


def test_invoke_reasoning() -> None:
    """`invoke` returns reasoning output and the `bitdeer_ai` provider."""
    llm = _model_with_transport([])
    message = llm.invoke("What is 2 + 3?")
    assert message.additional_kwargs["reasoning_content"] == "2+3=5"
    assert message.response_metadata["model_provider"] == "bitdeer_ai"
    assert any(block["type"] == "reasoning" for block in message.content_blocks)


def _check_stream(chunks: list[AIMessageChunk]) -> None:
    reasoning = [chunk.additional_kwargs.get("reasoning_content") for chunk in chunks]
    assert reasoning[:3] == ["2+3", "=5", None]
    # LangChain closes every stream with an empty chunk of its own.
    assert all(
        chunk.response_metadata["model_provider"] == "bitdeer_ai"
        for chunk in chunks
        if chunk.chunk_position != "last"
    )

    full = chunks[0]
    for chunk in chunks[1:]:
        full += chunk
    assert full.content == "5"
    assert full.additional_kwargs["reasoning_content"] == "2+3=5"
    assert full.response_metadata["model_provider"] == "bitdeer_ai"
    assert any(block["type"] == "reasoning" for block in full.content_blocks)
    assert full.usage_metadata == {
        "input_tokens": 40,
        "output_tokens": 21,
        "total_tokens": 61,
        "input_token_details": {"cache_read": 8},
        "output_token_details": {"reasoning": 19},
    }


def test_stream_reasoning_and_usage() -> None:
    """Streamed chunks carry reasoning pieces, and the sum carries usage."""
    requests: list[httpx.Request] = []
    llm = _model_with_transport(requests)
    _check_stream(list(llm.stream("What is 2 + 3?")))
    assert json.loads(requests[0].content)["stream_options"] == {"include_usage": True}


async def test_astream_reasoning_and_usage() -> None:
    """Async streaming behaves like sync streaming."""
    requests: list[httpx.Request] = []
    llm = _model_with_transport(requests)
    _check_stream([chunk async for chunk in llm.astream("What is 2 + 3?")])
    assert json.loads(requests[0].content)["stream_options"] == {"include_usage": True}


def test_stream_usage_opt_out_request() -> None:
    """With `stream_usage=False` the request does not ask for usage."""
    requests: list[httpx.Request] = []
    llm = _model_with_transport(requests, stream_usage=False)
    list(llm.stream("hi"))
    assert "stream_options" not in json.loads(requests[0].content)


TOOL_CALL_STREAM_CHUNKS = [
    _chunk({"role": "assistant", "content": "", "reasoning_content": "Use the"}),
    _chunk({"reasoning_content": " tool."}),
    _chunk(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "magic_function", "arguments": ""},
                }
            ]
        }
    ),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"input": '}}]}),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": "3}"}}]}),
    _chunk({}, finish_reason="tool_calls"),
    _chunk(None),
]


def test_stream_v1_reasoning_then_tool_call() -> None:
    """In v1 output, a streamed tool call stays separate from the reasoning block."""
    llm = _model_with_transport(
        [], stream_chunks=TOOL_CALL_STREAM_CHUNKS, output_version="v1"
    )
    chunks = list(llm.stream("What is magic_function(3)?"))
    full = chunks[0]
    for chunk in chunks[1:]:
        full += chunk

    assert [block["type"] for block in full.content_blocks] == [
        "reasoning",
        "tool_call",
    ]
    assert {"type": "reasoning", "reasoning": "Use the tool."}.items() <= (
        full.content_blocks[0].items()
    )
    assert {
        "type": "tool_call",
        "id": "call_1",
        "name": "magic_function",
        "args": {"input": 3},
    }.items() <= full.content_blocks[1].items()


def test_profile_is_none_for_unknown_model() -> None:
    """A model with no bundled profile gets `None`, not OpenAI's profile."""
    model = ChatBitdeerAI(model="not-a-real-model", api_key=SecretStr("k"))
    assert model.profile is None


def test_profile_comes_from_bundled_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """The profile is a copy of the bundled entry, so edits don't leak back."""
    limit = 1000
    monkeypatch.setitem(_PROFILES, MODEL_NAME, {"max_input_tokens": limit})
    model = ChatBitdeerAI(model=MODEL_NAME, api_key=SecretStr("k"))
    assert model.profile == {"max_input_tokens": limit}
    model.profile["max_input_tokens"] = 1  # type: ignore[index]
    assert _PROFILES[MODEL_NAME]["max_input_tokens"] == limit


def test_real_model_id_resolves_to_bundled_profile() -> None:
    """A model ID that Bitdeer AI serves gets its bundled profile."""
    model = ChatBitdeerAI(model="zai-org/GLM-5.3-Flash", api_key=SecretStr("k"))
    assert model.profile is not None
    assert model.profile == _PROFILES["zai-org/GLM-5.3-Flash"]


def test_bundled_profiles_use_known_keys() -> None:
    """Every bundled profile uses only keys defined by `ModelProfile`."""
    known = set(ModelProfile.__annotations__)
    for name, profile in _PROFILES.items():
        assert set(profile) <= known, name
