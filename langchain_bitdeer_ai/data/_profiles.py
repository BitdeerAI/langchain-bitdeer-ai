"""Model profiles for Bitdeer AI models.

Maintained by hand, keyed by model ID as served by the `/v1/models` endpoint.
Values come from official Bitdeer AI sources. Leave a key out when it isn't known.
"""

from langchain_core.language_models import ModelProfile

_PROFILES: dict[str, ModelProfile] = {
    "deepseek-ai/DeepSeek-V4-Flash": {
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 384_000,
        "text_inputs": True,
        "image_inputs": False,
        "text_outputs": True,
        "reasoning_output": True,
        "reasoning_effort_levels": ["none", "low", "high", "max"],
        "tool_calling": True,
        "tool_choice": True,
        "tool_call_streaming": True,
        "structured_output": True,
    },
    "deepseek-ai/DeepSeek-V4.1-Flash": {
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 390_000,
        "text_inputs": True,
        "image_inputs": True,
        "image_url_inputs": True,
        "text_outputs": True,
        "reasoning_output": True,
        "reasoning_effort_levels": ["none", "low", "high", "max"],
        "tool_calling": True,
        "tool_call_streaming": True,
        "structured_output": True,
    },
    "Qwen/Qwen3.8-27B": {
        "max_input_tokens": 260_000,
        "max_output_tokens": 130_000,
        "text_inputs": True,
        "image_inputs": True,
        "image_url_inputs": True,
        "text_outputs": True,
        "reasoning_output": True,
        "reasoning_effort_levels": ["none", "low", "medium", "xhigh"],
        "tool_calling": True,
        "tool_choice": True,
        "tool_call_streaming": True,
        "structured_output": True,
    },
    "zai-org/GLM-5.3": {
        "max_input_tokens": 260_000,
        "max_output_tokens": 130_000,
        "text_inputs": True,
        "image_inputs": False,
        "text_outputs": True,
        "reasoning_output": True,
        "reasoning_effort_levels": ["low", "high", "max"],
        "tool_calling": True,
        "tool_choice": True,
        "tool_call_streaming": True,
        "structured_output": True,
    },
    "zai-org/GLM-5.3-Flash": {
        "max_input_tokens": 260_000,
        "max_output_tokens": 130_000,
        "text_inputs": True,
        "image_inputs": True,
        "image_url_inputs": True,
        "text_outputs": True,
        "reasoning_output": True,
        "reasoning_effort_levels": ["low", "high", "max"],
        "tool_calling": True,
        "tool_choice": True,
        "tool_call_streaming": True,
        "structured_output": True,
    },
}
