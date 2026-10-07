"""Standard LangChain interface tests."""

from typing import Literal

from langchain_core.language_models import BaseChatModel
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
