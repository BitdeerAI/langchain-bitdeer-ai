"""Standard LangChain interface tests."""

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests
from syrupy.assertion import SnapshotAssertion
from typing_extensions import override

from langchain_bitdeer_ai import ChatBitdeerAI
from langchain_bitdeer_ai.chat_models import DEFAULT_API_BASE

MODEL_NAME = "zai-org/GLM-5.3-Flash"


class TestChatBitdeerAIStandard(ChatModelUnitTests):
    """Run LangChain's standard chat model unit tests."""

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        """Chat model class under test."""
        return ChatBitdeerAI

    @property
    def chat_model_params(self) -> dict:
        """Parameters for constructing the chat model."""
        return {"model": MODEL_NAME}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Environment variables, init args and the attributes they should set."""
        return (
            {
                "BITDEER_API_KEY": "api_key",
                "BITDEER_API_BASE": DEFAULT_API_BASE,
            },
            {
                "model": MODEL_NAME,
            },
            {
                "bitdeer_ai_api_key": "api_key",
                "bitdeer_ai_api_base": DEFAULT_API_BASE,
            },
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Loads with allowed_objects='all', which only admits classes listed in "
            "langchain_core.load.mapping; ChatBitdeerAI is not listed there. "
            "test_chat_models.py::test_serialization_round_trip covers loading."
        ),
    )
    @override
    def test_serdes(self, model: BaseChatModel, snapshot: SnapshotAssertion) -> None:
        super().test_serdes(model, snapshot)
