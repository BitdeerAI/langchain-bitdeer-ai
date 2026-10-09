"""Model profiles for Bitdeer AI models.

Maintained by hand, keyed by model ID as served by the `/v1/models` endpoint.
Values come from official Bitdeer AI sources. Leave a key out when it isn't known.
"""

from langchain_core.language_models import ModelProfile

_PROFILES: dict[str, ModelProfile] = {}
