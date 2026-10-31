"""Model profiles for Bitdeer AI models.

Maintained by hand, keyed by the model ID that API requests use.
Values come from official Bitdeer AI sources. Leave a key out when it isn't known.
"""

from langchain_core.language_models import ModelProfile

_PROFILES: dict[str, ModelProfile] = {
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": {
        "max_input_tokens": 1_048_576,
        "text_inputs": True,
        "image_inputs": True,
        "image_url_inputs": True,
        "text_outputs": True,
        "reasoning_output": True,
        "reasoning_effort_levels": ["none", "low", "medium", "high", "xhigh", "max"],
        "tool_calling": True,
        "tool_choice": True,
        "tool_call_streaming": True,
        "structured_output": True,
    },
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B": {
        "max_input_tokens": 262_144,
        "text_inputs": True,
        "image_inputs": True,
        "image_url_inputs": True,
        "text_outputs": True,
        "reasoning_output": True,
        "reasoning_effort_levels": ["none", "low", "medium", "high", "xhigh", "max"],
        "tool_calling": True,
        "tool_choice": True,
        "tool_call_streaming": True,
        "structured_output": True,
    },
    "Qwen/Qwen3.8-27B": {
        "max_input_tokens": 262_144,
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
        "max_input_tokens": 1_048_576,
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
        "max_input_tokens": 1_048_576,
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
