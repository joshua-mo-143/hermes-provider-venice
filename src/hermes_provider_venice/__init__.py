"""Venice AI model provider profile for Hermes Agent.

Venice is an OpenAI-compatible inference API
(https://api.venice.ai/api/v1). See https://docs.venice.ai.

Hermes can load this module from a Python package entry point or from
``$HERMES_HOME/plugins/model-providers/venice/``.
"""

from __future__ import annotations

import os
import time
from typing import Any

__version__ = "0.1.0"

_CATALOG_TTL_SECONDS = 300.0
_catalog_cache: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {}

# Venice appends its own default system prompts unless disabled. Hermes
# already ships a full agent system prompt, so we turn Venice's off by
# default to avoid conflicting instructions and wasted tokens.
_DEFAULT_VENICE_PARAMETERS: dict[str, Any] = {
    "include_venice_system_prompt": False,
}

_REASONING_EFFORT_ORDER = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


def _catalog_key(base_url: str) -> str:
    return base_url.strip().rstrip("/")


def _parse_catalog(payload: Any) -> dict[str, dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return {}

    catalog: dict[str, dict[str, Any]] = {}
    for raw_model in data:
        if not isinstance(raw_model, dict):
            continue
        model_id = raw_model.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue

        model_type = raw_model.get("type")
        if model_type and model_type != "text":
            continue

        model_spec = raw_model.get("model_spec")
        if not isinstance(model_spec, dict):
            model_spec = {}
        if model_spec.get("offline"):
            continue

        capabilities = model_spec.get("capabilities")
        if not isinstance(capabilities, dict):
            capabilities = {}

        metadata = {
            "capabilities": capabilities,
            "context_length": raw_model.get("context_length")
            or model_spec.get("availableContextTokens"),
            "max_completion_tokens": model_spec.get("maxCompletionTokens"),
        }
        normalized_id = model_id.strip()
        catalog[normalized_id] = metadata

    return catalog


def _clamp_reasoning_effort(effort: str, options: Any) -> str | None:
    normalized = effort.strip().lower()
    if normalized == "ultra":
        normalized = "max"
    if normalized not in _REASONING_EFFORT_ORDER:
        return None

    supported = (
        [
            str(option).strip().lower()
            for option in options
            if str(option).strip().lower() in _REASONING_EFFORT_ORDER
        ]
        if isinstance(options, list)
        else []
    )
    if not supported or normalized in supported:
        return normalized

    requested_index = _REASONING_EFFORT_ORDER.index(normalized)
    return min(
        supported,
        key=lambda option: (
            abs(_REASONING_EFFORT_ORDER.index(option) - requested_index),
            _REASONING_EFFORT_ORDER.index(option) > requested_index,
        ),
    )


venice: Any | None = None

try:
    from providers import register_provider
    from providers.base import ProviderProfile
except ImportError:
    # Keep package metadata and the compatibility installer importable when
    # this distribution is inspected outside the Hermes Python environment.
    pass
else:

    class VeniceProfile(ProviderProfile):
        """Venice AI — OpenAI-compat chat with venice_parameters extras."""

        def _fetch_catalog(
            self,
            *,
            api_key: str | None = None,
            base_url: str | None = None,
            timeout: float = 8.0,
        ) -> dict[str, dict[str, Any]] | None:
            effective_base = _catalog_key(base_url or self.base_url or "")
            if not effective_base:
                return None

            cached = _catalog_cache.get(effective_base)
            now = time.monotonic()
            if (
                cached
                and now - cached[0] < _CATALOG_TTL_SECONDS
                and (cached[1] or not api_key)
            ):
                return cached[1]

            import json
            import urllib.request

            from hermes_cli.urllib_security import open_credentialed_url

            req = urllib.request.Request(f"{effective_base}/models?type=text")
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            req.add_header("Accept", "application/json")
            for key, value in (self.default_headers or {}).items():
                req.add_header(key, value)

            try:
                with open_credentialed_url(req, timeout=timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except Exception:
                _catalog_cache[effective_base] = (now, {})
                return None

            catalog = _parse_catalog(payload)
            _catalog_cache[effective_base] = (now, catalog)
            return catalog or None

        def _model_metadata(
            self,
            model: str | None,
            *,
            base_url: str | None = None,
        ) -> dict[str, Any] | None:
            if not model:
                return None
            catalog = self._fetch_catalog(
                api_key=os.environ.get("VENICE_API_KEY"),
                base_url=base_url,
            )
            if not catalog:
                return None
            return catalog.get(model) or catalog.get(model.split("/", 1)[-1])

        def build_extra_body(
            self, *, session_id: str | None = None, **context
        ) -> dict[str, Any]:
            body: dict[str, Any] = {
                "venice_parameters": dict(_DEFAULT_VENICE_PARAMETERS),
            }
            # Venice's prompt_cache_key routes multi-turn traffic to the same
            # backend to improve cache hit rates.
            if session_id:
                body["prompt_cache_key"] = session_id

            return body

        def build_api_kwargs_extras(
            self,
            *,
            reasoning_config: dict | None = None,
            model: str | None = None,
            base_url: str | None = None,
            **context: Any,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            if not isinstance(reasoning_config, dict):
                return {}, {}

            effort = str(reasoning_config.get("effort") or "").strip().lower()
            if reasoning_config.get("enabled") is False or effort == "none":
                return {"reasoning": {"enabled": False}}, {}
            if not effort:
                return {}, {}

            metadata = self._model_metadata(model, base_url=base_url)
            capabilities = metadata.get("capabilities", {}) if metadata else {}
            if capabilities.get("supportsReasoningEffort") is not True:
                return {}, {}

            mapped_effort = _clamp_reasoning_effort(
                effort,
                capabilities.get("reasoningEffortOptions"),
            )
            if not mapped_effort:
                return {}, {}
            return {}, {"reasoning_effort": mapped_effort}

        def fetch_models(
            self,
            *,
            api_key: str | None = None,
            base_url: str | None = None,
            timeout: float = 8.0,
        ) -> list[str] | None:
            """Return online text models that can support an agent tool loop."""
            catalog = self._fetch_catalog(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
            )
            if not catalog:
                return None

            return [
                model_id
                for model_id, metadata in catalog.items()
                if metadata.get("capabilities", {}).get("supportsFunctionCalling")
                is True
            ] or None

    venice = VeniceProfile(
        name="venice",
        aliases=("venice-ai", "veniceai", "venice.ai", "api.venice.ai"),
        display_name="Venice AI",
        description="Venice AI — private, uncensored OpenAI-compatible inference",
        signup_url="https://venice.ai/settings/api",
        env_vars=("VENICE_API_KEY", "VENICE_BASE_URL"),
        base_url="https://api.venice.ai/api/v1",
        models_url="https://api.venice.ai/api/v1/models?type=text",
        auth_type="api_key",
        # Cheap tool-capable default for compression / titles / vision-aux.
        # Catalog refreshed 2026-07-16 from GET /models?type=text.
        default_aux_model="deepseek-v4-flash",
        fallback_models=(
            # Flagship private / current generation
            "zai-org-glm-5-2",
            "zai-org-glm-5",
            "grok-4-5",
            "qwen-3-7-max",
            "qwen-3-7-plus",
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "minimax-m3-preview",
            "kimi-k2-7-code",
            "xiaomi-mimo-v2-5",
            # Anonymized frontier (OpenAI / Anthropic / Google via Venice)
            "openai-gpt-56-luna",
            "openai-gpt-56-terra",
            "claude-opus-4-8",
            "claude-sonnet-5",
            "gemini-3-5-flash",
            # Venice uncensored + specialist
            "venice-uncensored-1-2",
            "qwen3-coder-480b-a35b-instruct-turbo",
            "qwen3-vl-235b-a22b",
            # Still listed as Venice trait defaults (stable aliases)
            "zai-org-glm-4.7",
            "zai-org-glm-4.7-flash",
            "qwen3-235b-a22b-thinking-2507",
        ),
    )

    # Directory plugins are registered by importing their __init__.py.
    register_provider(venice)


def register(_context: object | None = None) -> None:
    """Register the Venice profile with the host Hermes process.

    ``_context`` keeps this callable compatible with Hermes's general plugin
    entry-point loader. The dedicated model-provider loader calls it without
    arguments.
    """
    if venice is None:
        raise RuntimeError(
            "Hermes Agent is not importable. Install this package in the "
            "same Python environment as Hermes Agent."
        )
    register_provider(venice)


__all__ = ("__version__", "register", "venice")
