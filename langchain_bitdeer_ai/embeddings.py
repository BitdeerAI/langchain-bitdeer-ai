"""Bitdeer AI embedding models."""

from __future__ import annotations

import openai
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_bitdeer_ai.chat_models import DEFAULT_API_BASE


class BitdeerAIEmbeddings(OpenAIEmbeddings):
    """Bitdeer AI embedding model integration.

    Uses Bitdeer AI's OpenAI-compatible embeddings API.

    Setup:
        Install `langchain-bitdeer-ai` and set environment variable `BITDEER_API_KEY`.

        ```bash
        pip install -U langchain-bitdeer-ai
        export BITDEER_API_KEY="your-api-key"
        ```

    Key init args — embedding params:
        model:
            Bitdeer AI embedding model ID, e.g. `'BAAI/bge-m3'`. Required.

    Key init args — client params:
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        api_key:
            Bitdeer AI API key. If not passed in will be read from env var
            `BITDEER_API_KEY`.
        base_url:
            Base URL for API requests. If not passed in will be read from env var
            `BITDEER_API_BASE`, defaulting to `https://api-inference.bitdeer.ai/v1`.

    Instantiate:
        ```python
        from langchain_bitdeer_ai import BitdeerAIEmbeddings

        embed = BitdeerAIEmbeddings(
            model="BAAI/bge-m3",
            # api_key="...",
            # other params...
        )
        ```

    Embed single text:
        ```python
        vector = embed.embed_query("The quick brown fox jumps over the lazy dog.")
        len(vector)
        ```

    Embed multiple texts:
        ```python
        vectors = embed.embed_documents(["Hello", "World"])
        len(vectors)
        ```

    Async:
        ```python
        await embed.aembed_query("Hello")
        await embed.aembed_documents(["Hello", "World"])
        ```

    Input and output:
        Each text can be at most 8192 tokens. The length is not checked before
        sending, so longer input fails with an HTTP 400 from the API. `embed_documents`
        sends `chunk_size` texts (default 1000) per request.
        Texts are sent to the API as they are. They are not tokenized with an
        OpenAI tokenizer, so leave `check_embedding_ctx_length` at `False`.
        The endpoint ignores `dimensions`, so passing it raises a `ValueError`.
    """

    model: str = Field(alias="model")
    """Embedding model ID to use, for example `'BAAI/bge-m3'`."""

    bitdeer_ai_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env("BITDEER_API_KEY", default=None),
    )
    """Bitdeer AI API key.

    Automatically read from env variable `BITDEER_API_KEY` if not provided.
    """

    bitdeer_ai_api_base: str = Field(
        alias="base_url",
        default_factory=from_env("BITDEER_API_BASE", default=DEFAULT_API_BASE),
    )
    """Base URL for API requests.

    Automatically read from env variable `BITDEER_API_BASE` if not provided.
    """

    # The parent reads these from `OPENAI_*` variables. Redeclaring them without
    # an environment lookup keeps OpenAI settings out of Bitdeer AI requests.
    openai_api_key: SecretStr | None = None  # type: ignore[assignment]

    openai_api_base: str | None = None

    openai_organization: str | None = None

    openai_proxy: str | None = None

    openai_api_type: str | None = None

    openai_api_version: str | None = None

    check_embedding_ctx_length: bool = False
    """Send raw text instead of OpenAI `cl100k_base` token IDs."""

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Build the clients from the Bitdeer AI settings only.

        Replaces `OpenAIEmbeddings.validate_environment`, which would fall back to
        `OPENAI_API_KEY`, `OPENAI_API_BASE` and the other `OPENAI_*` settings.
        """
        if self.dimensions is not None:
            msg = "Bitdeer AI embeddings ignore `dimensions`, so it is not supported."
            raise ValueError(msg)

        # Without a key the `openai` client would read `OPENAI_API_KEY` instead.
        if not (self.bitdeer_ai_api_key and self.bitdeer_ai_api_key.get_secret_value()):
            msg = (
                "Bitdeer AI API key is not set. Pass `api_key` or set the "
                "BITDEER_API_KEY environment variable."
            )
            raise ValueError(msg)

        client_params: dict = {
            "api_key": self.bitdeer_ai_api_key.get_secret_value(),
            "base_url": self.bitdeer_ai_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        # The `openai` client reads `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` itself
        # and turns them into request headers, so clear them after construction.
        if not self.client:
            root_client = openai.OpenAI(**client_params, http_client=self.http_client)
            root_client.organization = root_client.project = None
            self.client = root_client.embeddings
        if not self.async_client:
            root_async_client = openai.AsyncOpenAI(
                **client_params, http_client=self.http_async_client
            )
            root_async_client.organization = root_async_client.project = None
            self.async_client = root_async_client.embeddings
        return self
