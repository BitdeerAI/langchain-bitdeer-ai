"""Unit tests for `BitdeerAIEmbeddings`."""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.embeddings import Embeddings
from langchain_tests.unit_tests import EmbeddingsUnitTests
from pydantic import SecretStr, ValidationError

from langchain_bitdeer_ai import BitdeerAIEmbeddings
from langchain_bitdeer_ai.chat_models import DEFAULT_API_BASE

MODEL_NAME = "BAAI/bge-m3"

_OPENAI_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT_ID",
    "OPENAI_PROXY",
    "OPENAI_API_TYPE",
    "OPENAI_API_VERSION",
)


class TestBitdeerAIEmbeddingsStandard(EmbeddingsUnitTests):
    """Run LangChain's standard embeddings unit tests."""

    @property
    def embeddings_class(self) -> type[Embeddings]:
        """Embeddings class under test."""
        return BitdeerAIEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        """Parameters for constructing the embeddings."""
        return {"model": MODEL_NAME, "api_key": "test"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Environment variables, init args and the attributes they should set."""
        return (
            {
                "BITDEER_API_KEY": "api_key",
                "BITDEER_API_BASE": DEFAULT_API_BASE,
            },
            {"model": MODEL_NAME},
            {
                "bitdeer_ai_api_key": "api_key",
                "bitdeer_ai_api_base": DEFAULT_API_BASE,
            },
        )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test without Bitdeer AI or OpenAI settings in the environment."""
    for var in ("BITDEER_API_KEY", "BITDEER_API_BASE", *_OPENAI_VARS):
        monkeypatch.delenv(var, raising=False)


def _embeddings_with_recorder(
    requests: list[httpx.Request], **kwargs: object
) -> BitdeerAIEmbeddings:
    """Build embeddings whose sync client records requests instead of sending them."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        n = len(json.loads(request.content)["input"])
        data = [
            {"object": "embedding", "index": i, "embedding": [0.0, 1.0]}
            for i in range(n)
        ]
        body = {
            "object": "list",
            "model": MODEL_NAME,
            "data": data,
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
        return httpx.Response(200, json=body)

    return BitdeerAIEmbeddings(
        model=MODEL_NAME,
        api_key=SecretStr("test"),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,  # type: ignore[arg-type]
    )


def test_model_is_required() -> None:
    """Construction fails without a model, naming the missing field."""
    with pytest.raises(ValidationError, match="model"):
        BitdeerAIEmbeddings(api_key=SecretStr("test"))  # type: ignore[call-arg]


def test_model_is_sent() -> None:
    """The model ID goes into the request body."""
    requests: list[httpx.Request] = []
    _embeddings_with_recorder(requests).embed_query("hi")
    assert json.loads(requests[0].content)["model"] == MODEL_NAME


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The key is read from `BITDEER_API_KEY` when not passed."""
    monkeypatch.setenv("BITDEER_API_KEY", "env-key")
    embed = BitdeerAIEmbeddings(model=MODEL_NAME)
    assert embed.client._client.api_key == "env-key"
    assert embed.async_client._client.api_key == "env-key"


def test_explicit_api_key_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A passed `api_key` takes precedence over `BITDEER_API_KEY`."""
    monkeypatch.setenv("BITDEER_API_KEY", "env-key")
    embed = BitdeerAIEmbeddings(model=MODEL_NAME, api_key=SecretStr("arg-key"))
    assert embed.client._client.api_key == "arg-key"
    assert embed.async_client._client.api_key == "arg-key"


def test_api_key_is_not_leaked() -> None:
    """The key does not appear in repr or str."""
    embed = BitdeerAIEmbeddings(model=MODEL_NAME, api_key=SecretStr("secret-value"))
    assert "secret-value" not in repr(embed)
    assert "secret-value" not in str(embed)


def test_missing_api_key() -> None:
    """Construction without a key fails with an error naming `BITDEER_API_KEY`."""
    with pytest.raises(ValueError, match="BITDEER_API_KEY"):
        BitdeerAIEmbeddings(model=MODEL_NAME)


def test_openai_api_key_is_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """`OPENAI_API_KEY` is neither a fallback nor sent alongside the Bitdeer AI key."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    with pytest.raises(ValueError, match="BITDEER_API_KEY"):
        BitdeerAIEmbeddings(model=MODEL_NAME)

    monkeypatch.setenv("BITDEER_API_KEY", "env-key")
    embed = BitdeerAIEmbeddings(model=MODEL_NAME)
    assert embed.client._client.api_key == "env-key"


def test_default_base_url() -> None:
    """Requests go to the Bitdeer AI endpoint by default."""
    embed = BitdeerAIEmbeddings(model=MODEL_NAME, api_key=SecretStr("test"))
    assert embed.bitdeer_ai_api_base == DEFAULT_API_BASE
    assert str(embed.client._client.base_url) == f"{DEFAULT_API_BASE}/"
    assert str(embed.async_client._client.base_url) == f"{DEFAULT_API_BASE}/"


def test_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """`BITDEER_API_BASE` overrides the default endpoint."""
    monkeypatch.setenv("BITDEER_API_BASE", "https://env.example.com/v1")
    embed = BitdeerAIEmbeddings(model=MODEL_NAME, api_key=SecretStr("test"))
    assert str(embed.client._client.base_url) == "https://env.example.com/v1/"


def test_base_url_argument_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A passed `base_url` takes precedence over `BITDEER_API_BASE`."""
    monkeypatch.setenv("BITDEER_API_BASE", "https://env.example.com/v1")
    embed = BitdeerAIEmbeddings(
        model=MODEL_NAME,
        api_key=SecretStr("test"),
        base_url="https://proxy.example.com/v1",
    )
    assert str(embed.client._client.base_url) == "https://proxy.example.com/v1/"
    assert str(embed.async_client._client.base_url) == "https://proxy.example.com/v1/"


def test_openai_settings_are_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI settings neither redirect requests nor add OpenAI headers."""
    monkeypatch.setenv("OPENAI_API_BASE", "https://other.example.com/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://other.example.com/v1")
    monkeypatch.setenv("OPENAI_ORG_ID", "org-openai")
    monkeypatch.setenv("OPENAI_ORGANIZATION", "org-openai")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-openai")
    monkeypatch.setenv("OPENAI_API_TYPE", "azure")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-01-01")
    monkeypatch.setenv("OPENAI_PROXY", "http://proxy.example.com")
    requests: list[httpx.Request] = []
    embed = _embeddings_with_recorder(requests)
    embed.embed_query("hi")
    assert str(embed.client._client.base_url) == f"{DEFAULT_API_BASE}/"
    assert requests[0].url.host == "api-inference.bitdeer.ai"
    assert "openai-organization" not in requests[0].headers
    assert "openai-project" not in requests[0].headers


def test_documents_are_sent_as_text() -> None:
    """Documents reach the endpoint as the original strings, one vector each."""
    requests: list[httpx.Request] = []
    embed = _embeddings_with_recorder(requests)
    vectors = embed.embed_documents(["a", "b"])
    assert json.loads(requests[0].content)["input"] == ["a", "b"]
    assert len(vectors) == 2  # noqa: PLR2004


def test_documents_are_batched_by_default_chunk_size() -> None:
    """The default `chunk_size` is 1000 texts per request."""
    requests: list[httpx.Request] = []
    embed = _embeddings_with_recorder(requests)
    assert embed.chunk_size == 1000  # noqa: PLR2004
    vectors = embed.embed_documents(["x"] * 1001)
    sizes = [len(json.loads(r.content)["input"]) for r in requests]
    assert sizes == [1000, 1]
    assert len(vectors) == 1001  # noqa: PLR2004


def test_query_is_sent_as_text() -> None:
    """A query reaches the endpoint as the original string."""
    requests: list[httpx.Request] = []
    embed = _embeddings_with_recorder(requests)
    vector = embed.embed_query("hello")
    assert "hello" in json.loads(requests[0].content)["input"]
    assert vector == [0.0, 1.0]


def test_dimensions_are_rejected() -> None:
    """`dimensions` raises, because the endpoint ignores it."""
    with pytest.raises(ValueError, match="dimensions"):
        BitdeerAIEmbeddings(model=MODEL_NAME, api_key=SecretStr("test"), dimensions=256)


def test_check_embedding_ctx_length_is_rejected() -> None:
    """`check_embedding_ctx_length=True` raises, because it sends OpenAI token IDs."""
    with pytest.raises(ValueError, match="check_embedding_ctx_length"):
        BitdeerAIEmbeddings(
            model=MODEL_NAME,
            api_key=SecretStr("test"),
            check_embedding_ctx_length=True,
        )


def _proxy_url(client: httpx.Client | httpx.AsyncClient) -> str | None:
    mounts = client._mounts
    transport = next((t for t in mounts.values() if t is not None), None)
    pool = getattr(transport, "_pool", None)
    return str(pool._proxy_url) if pool is not None else None


def test_openai_proxy_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """`openai_proxy` installs a proxy transport on both clients."""
    monkeypatch.setenv("BITDEER_API_KEY", "env-key")
    obj = BitdeerAIEmbeddings(
        model="BAAI/bge-m3", openai_proxy="http://proxy.example:3128"
    )
    assert "proxy.example" in str(_proxy_url(obj.client._client._client))
    assert "proxy.example" in str(_proxy_url(obj.async_client._client._client))


def test_openai_proxy_conflicts_with_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proxy cannot be combined with a caller-supplied client."""
    monkeypatch.setenv("BITDEER_API_KEY", "env-key")
    with pytest.raises(ValueError, match="openai_proxy"):
        BitdeerAIEmbeddings(
            model="BAAI/bge-m3",
            openai_proxy="http://proxy.example:3128",
            http_client=httpx.Client(),
        )


def test_openai_proxy_env_is_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """`OPENAI_PROXY` is not read."""
    monkeypatch.setenv("BITDEER_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_PROXY", "http://proxy.example:3128")
    assert BitdeerAIEmbeddings(model="BAAI/bge-m3").openai_proxy is None
