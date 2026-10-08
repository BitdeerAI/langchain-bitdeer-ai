"""Bitdeer AI chat models."""

from __future__ import annotations

from typing import Any, cast

import openai
from langchain_core.language_models import LangSmithParams, ModelProfile
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import OutputTokenDetails, UsageMetadata
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_openai.chat_models.base import (
    _use_responses_api as _requires_responses_api,
)
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_bitdeer_ai._version import __version__

DEFAULT_API_BASE = "https://api-inference.bitdeer.ai/v1"

_PROVIDER = "bitdeer_ai"

# Named in the error message. Whether a request needs the Responses API is decided
# by `langchain-openai`'s own rule (`_requires_responses_api`), which also covers
# OpenAI built-in tools.
_RESPONSES_ONLY_PARAMS = {
    "context_management",
    "include",
    "previous_response_id",
    "reasoning",
    "text",
    "truncation",
}


def _get(obj: Any, key: str) -> Any:
    """Read `key` from a response dict or an `openai` response object."""
    if isinstance(obj, dict):
        return obj.get(key)
    # Fields Bitdeer AI adds beyond OpenAI's schema (`reasoning_content`,
    # `usage.reasoning_tokens`) are kept as extras, which `getattr` reads.
    return getattr(obj, key, None)


def _add_reasoning_tokens(message: BaseMessage, usage: Any) -> None:
    """Record Bitdeer AI's top-level `usage.reasoning_tokens` in usage metadata.

    OpenAI nests the count in `completion_tokens_details`, which is what the base
    class reads. Bitdeer AI reports it at the top level of `usage`, and counts it
    in `completion_tokens`, so `output_tokens` needs no adjustment. A count the
    base class already set is left untouched.
    """
    if not isinstance(message, AIMessage) or message.usage_metadata is None:
        return
    reasoning_tokens = _get(usage, "reasoning_tokens")
    if not isinstance(reasoning_tokens, int):
        return
    output_token_details = message.usage_metadata.get("output_token_details") or {}
    if "reasoning" in output_token_details:
        return
    usage_metadata: dict[str, Any] = dict(message.usage_metadata)
    usage_metadata["output_token_details"] = cast(
        "OutputTokenDetails",
        {**output_token_details, "reasoning": reasoning_tokens},
    )
    message.usage_metadata = cast("UsageMetadata", usage_metadata)


class ChatBitdeerAI(BaseChatOpenAI):
    """Bitdeer AI chat model integration.

    Uses Bitdeer AI's OpenAI-compatible Chat Completions API.

    Setup:
        Install `langchain-bitdeer-ai` and set environment variable `BITDEER_API_KEY`.

        ```bash
        pip install -U langchain-bitdeer-ai
        export BITDEER_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model:
            Bitdeer AI model ID, e.g. `'zai-org/GLM-5.3-Flash'`. Required.
        temperature:
            Sampling temperature.
        max_tokens:
            Max number of tokens to generate.

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

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        ```python
        from langchain_bitdeer_ai import ChatBitdeerAI

        model = ChatBitdeerAI(
            model="zai-org/GLM-5.3-Flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # api_key="...",
            # other params...
        )
        ```

    Invoke:
        ```python
        messages = [
            ("system", "You are a helpful translator. Translate the user sentence to French."),
            ("human", "I love programming."),
        ]
        model.invoke(messages)
        ```

    Stream:
        ```python
        for chunk in model.stream(messages):
            print(chunk.text, end="")
        ```
        ```python
        stream = model.stream(messages)
        full = next(stream)
        for chunk in stream:
            full += chunk
        full
        ```

    Async:
        ```python
        await model.ainvoke(messages)

        # stream:
        # async for chunk in model.astream(messages)

        # batch:
        # await model.abatch([messages])
        ```

    Tool calling:
        ```python
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        model_with_tools = model.bind_tools([GetWeather])
        ai_msg = model_with_tools.invoke("What's the weather in LA?")
        ai_msg.tool_calls
        ```

        See `ChatBitdeerAI.bind_tools()` method for more.

    Structured output:
        ```python
        from pydantic import BaseModel, Field


        class Joke(BaseModel):
            '''Joke to tell user.'''

            setup: str = Field(description="The setup of the joke")
            punchline: str = Field(description="The punchline to the joke")


        structured_model = model.with_structured_output(Joke, method="json_schema")
        structured_model.invoke("Tell me a joke about cats")
        ```

        `method` can be `'function_calling'` (the default), `'json_schema'` or
        `'json_mode'`. See `ChatBitdeerAI.with_structured_output()` for more.

    Reasoning output:
        Bitdeer AI models return their reasoning alongside the answer. It is
        exposed in `additional_kwargs` and as a standard `reasoning` content block.

        ```python
        ai_msg = model.invoke("What is 2 + 3?")
        ai_msg.additional_kwargs["reasoning_content"]
        [block for block in ai_msg.content_blocks if block["type"] == "reasoning"]
        ```

        When streaming, each chunk carries its piece of the reasoning, and the
        summed chunks hold the full text.

    Token usage:
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.usage_metadata
        ```
        ```python
        {
            "input_tokens": 40,
            "output_tokens": 21,
            "total_tokens": 61,
            "input_token_details": {},
            "output_token_details": {"reasoning": 19},
        }
        ```

        Reasoning tokens are part of `output_tokens`. Token usage is also included
        when streaming; set `stream_usage=False` to turn it off.

    Response metadata:
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.response_metadata
        ```
    """  # noqa: E501

    model_name: str = Field(alias="model")
    """Model ID to use, for example `'zai-org/GLM-5.3-Flash'`."""

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

    openai_api_key: SecretStr | None = None

    openai_api_base: str | None = None

    model_config = ConfigDict(populate_by_name=True)

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"bitdeer_ai_api_key": "BITDEER_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain_bitdeer_ai", "chat_models"]`
        """
        return ["langchain_bitdeer_ai", "chat_models"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by LangChain."""
        return True

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-bitdeer-ai"

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = _PROVIDER
        return ls_params

    @model_validator(mode="after")
    def _set_bitdeer_ai_version(self) -> Self:
        """Set package version in metadata.

        Named uniquely to avoid shadowing `BaseChatOpenAI._set_openai_chat_version`;
        Pydantic replaces same-named validators rather than chaining them.
        """
        self._add_version("langchain-bitdeer-ai", __version__)
        return self

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Build the clients from the Bitdeer AI settings only.

        Replaces `BaseChatOpenAI.validate_environment`, which would fall back to
        `OPENAI_API_KEY`, `OPENAI_API_BASE` and LangSmith gateway settings.
        """
        if self.n is not None and self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.n is not None and self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
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
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        if not self.client:
            self.root_client = openai.OpenAI(
                **client_params, http_client=self.http_client
            )
            self.client = self.root_client.chat.completions
        if not self.async_client:
            self.root_async_client = openai.AsyncOpenAI(
                **client_params, http_client=self.http_async_client
            )
            self.async_client = self.root_async_client.chat.completions

        if self.stream_usage is not False:
            self.stream_usage = True
        return self

    def _resolve_model_profile(self) -> ModelProfile | None:
        # Profiles will come from models.dev; the base class would look the model
        # up in OpenAI's profiles.
        return None

    def _use_responses_api(self, payload: dict) -> bool:
        # Always use Chat Completions. The base class would send some calls to the
        # Responses API (for example when `reasoning` is set) but only for
        # `invoke`, not `stream`. Parameters that only the Responses API accepts
        # would make the `openai` client fail with an unclear `TypeError`, so
        # reject them here, using upstream's own rule for what they are.
        if _requires_responses_api(payload):
            names = [
                f"`{name}`" for name in sorted(_RESPONSES_ONLY_PARAMS & payload.keys())
            ]
            msg = (
                "ChatBitdeerAI uses the Chat Completions API, which doesn't accept "
                f"{', '.join(names) or 'OpenAI built-in tools'}."
            )
            if "reasoning" in payload:
                msg += " Use `reasoning_effort` instead of `reasoning`."
            raise ValueError(msg)
        return False

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        rtn = super()._create_chat_result(response, generation_info)
        if rtn.llm_output is not None:
            rtn.llm_output["model_provider"] = _PROVIDER

        choices = _get(response, "choices") or []
        usage = _get(response, "usage")
        for generation, choice in zip(rtn.generations, choices, strict=False):
            message = generation.message
            message.response_metadata["model_provider"] = _PROVIDER
            if reasoning_content := _get(_get(choice, "message"), "reasoning_content"):
                message.additional_kwargs["reasoning_content"] = reasoning_content
            if usage is not None:
                _add_reasoning_tokens(message, usage)
        return rtn

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if generation_chunk is None:
            return None

        message = generation_chunk.message
        # Also covers the trailing usage chunk, which has no choices.
        message.response_metadata["model_provider"] = _PROVIDER
        if choices := chunk.get("choices"):
            delta = choices[0].get("delta") or {}
            if reasoning_content := delta.get("reasoning_content"):
                message.additional_kwargs["reasoning_content"] = reasoning_content
        if usage := chunk.get("usage"):
            _add_reasoning_tokens(message, usage)
        return generation_chunk
