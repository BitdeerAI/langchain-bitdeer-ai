"""Standard LangChain interface tests."""

from typing import Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_bitdeer_ai import ChatBitdeerAI

MODEL_NAME = "zai-org/GLM-5.3-Flash"

# Shared across tests to stay within the test account's rate limits.
rate_limiter = InMemoryRateLimiter(requests_per_second=0.5)


class TestChatBitdeerAIStandard(ChatModelIntegrationTests):
    """Run LangChain's standard chat model integration tests."""

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        """Chat model class under test."""
        return ChatBitdeerAI

    @property
    def chat_model_params(self) -> dict:
        """Parameters for constructing the chat model."""
        return {
            "model": MODEL_NAME,
            "temperature": 0,
            "rate_limiter": rate_limiter,
        }

    @property
    def has_tool_calling(self) -> bool:
        """Bitdeer AI supports tool calling."""
        return True

    @property
    def has_tool_choice(self) -> bool:
        """Bitdeer AI supports forcing a tool call."""
        return True

    @property
    def has_structured_output(self) -> bool:
        """Bitdeer AI supports structured output."""
        return True

    @property
    def supports_image_inputs(self) -> bool:
        """Bitdeer AI accepts base64 `image_url` content."""
        return True

    @property
    def supports_image_urls(self) -> bool:
        """Bitdeer AI accepts image URLs."""
        return True

    @property
    def supports_json_mode(self) -> bool:
        """Bitdeer AI supports `response_format={"type": "json_object"}`."""
        return True

    @property
    def supports_pdf_inputs(self) -> bool:
        """Bitdeer AI does not accept PDF inputs."""
        return False

    @property
    def supports_audio_inputs(self) -> bool:
        """Bitdeer AI does not accept audio inputs."""
        return False

    @property
    def supports_video_inputs(self) -> bool:
        """Bitdeer AI does not accept video inputs."""
        return False

    @property
    def supported_usage_metadata_details(
        self,
    ) -> dict[
        Literal["invoke", "stream"],
        list[
            Literal[
                "audio_input",
                "audio_output",
                "reasoning_output",
                "cache_read_input",
                "cache_creation_input",
            ]
        ],
    ]:
        """Bitdeer AI reports reasoning tokens."""
        return {"invoke": ["reasoning_output"], "stream": ["reasoning_output"]}

    def invoke_with_reasoning_output(self, *, stream: bool = False) -> AIMessage:
        """Invoke with a prompt that makes the model reason."""
        model = ChatBitdeerAI(**self.chat_model_params)
        prompt = "Is 391 a prime number? Answer yes or no."
        if not stream:
            return cast("AIMessage", model.invoke(prompt))
        full: AIMessageChunk | None = None
        for chunk in model.stream(prompt):
            full = chunk if full is None else full + chunk
        return cast("AIMessage", full)
