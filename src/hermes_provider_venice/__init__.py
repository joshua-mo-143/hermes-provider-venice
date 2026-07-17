"""Venice AI model provider profile for Hermes Agent.

Venice is an OpenAI-compatible inference API
(https://api.venice.ai/api/v1). See https://docs.venice.ai.

Hermes can load this module from a Python package entry point or from
``$HERMES_HOME/plugins/model-providers/venice/``.

Hermes integration
------------------
Most provider behaviour is declared through :class:`VeniceProfile`, which
subclasses Hermes's ``ProviderProfile`` and overrides the documented request
hooks (``build_extra_body``, ``build_api_kwargs_extras``, ``fetch_models``).
That covers everything Hermes routes *through the provider object*.

The remaining integration points — context-window, cost, model-picker pricing,
and capability lookups — are resolved by Hermes through provider-agnostic,
module-level functions (``agent.models_dev``, ``agent.usage_pricing``,
``hermes_cli.models``) that key off a static catalog (models.dev) and never
consult the provider plugin. Venice is not in that catalog, and its ``/models``
response uses shapes those resolvers cannot parse (per-million ``{usd, diem}``
pricing objects, ``model_spec`` capability flags). Rather than fork Hermes, we
wrap those four functions so Venice routes resolve from Venice's own live
catalog.

Every ``_install_*_patch`` helper follows the same contract, so the patching
stays predictable despite touching Hermes internals:

* **Import-guarded** — a missing Hermes module is a no-op, keeping the package
  importable outside a Hermes environment (metadata, the compatibility
  installer, and tests all rely on this).
* **Idempotent** — the wrapper is tagged with a sentinel attribute and skipped
  if already applied, so repeated registration (import + ``register()``) wraps
  each target exactly once.
* **Venice-scoped and transparent** — the wrapper only intercepts Venice
  routes (see :func:`_is_venice_provider` / :func:`_is_venice_endpoint`) and
  delegates to the original function for every other provider and for any
  Venice model it cannot resolve.

The patches are installed at import time and again from :func:`register`.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import wraps
from typing import Any
from urllib.parse import urlparse

__version__ = "0.1.0"

_CATALOG_TTL_SECONDS = 300.0
_catalog_cache: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {}

# Venice reports token prices per million tokens; Hermes stores catalog
# pricing per token, so this factor converts between the two.
_ONE_MILLION = Decimal(1_000_000)

# Provider names/aliases that resolve to Venice inside Hermes.
_VENICE_PROVIDER_NAMES = frozenset(
    {"venice", "venice-ai", "veniceai", "venice.ai", "api.venice.ai"}
)

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
    """Normalise a base URL into the key used for the catalog cache."""
    return base_url.strip().rstrip("/")


def _is_venice_endpoint(base_url: str) -> bool:
    """Return ``True`` when *base_url* points at Venice.

    Matches the user's ``VENICE_BASE_URL`` override exactly, otherwise falls
    back to a ``venice.ai`` hostname check so custom base URLs still resolve.
    """
    normalized = _catalog_key(base_url)
    configured = _catalog_key(os.environ.get("VENICE_BASE_URL", ""))
    if configured and normalized == configured:
        return True
    try:
        hostname = (urlparse(normalized).hostname or "").lower()
    except ValueError:
        return False
    return hostname == "venice.ai" or hostname.endswith(".venice.ai")


def _is_venice_provider(provider: str | None) -> bool:
    """Return ``True`` for the Venice provider id or any of its aliases."""
    return (provider or "").strip().lower() in _VENICE_PROVIDER_NAMES


def _price_per_token(value: Any) -> str | None:
    """Convert a Venice per-million price into a per-token decimal string.

    Accepts either a bare number or Venice's ``{"usd": ..., "diem": ...}``
    price object. Returns ``None`` for missing, boolean, or unparseable
    values so callers can simply omit the field.
    """
    if isinstance(value, dict):
        value = value.get("usd")
    if value is None or isinstance(value, bool):
        return None
    try:
        price = Decimal(str(value)) / Decimal(1_000_000)
    except (InvalidOperation, TypeError, ValueError):
        return None
    return format(price, "f")


def _parse_pricing(model_spec: dict[str, Any]) -> dict[str, str]:
    """Extract per-token pricing from a model's ``model_spec.pricing`` block.

    Maps Venice's price keys onto the ``prompt`` / ``completion`` /
    ``cache_read`` / ``cache_write`` names the rest of this module (and
    Hermes) expect. Returns an empty dict when no pricing is present.
    """
    raw_pricing = model_spec.get("pricing")
    if not isinstance(raw_pricing, dict):
        return {}

    key_map = {
        "input": "prompt",
        "output": "completion",
        "cache_input": "cache_read",
        "cache_write": "cache_write",
        "cache_output": "cache_write",
    }
    pricing: dict[str, str] = {}
    for source, target in key_map.items():
        value = _price_per_token(raw_pricing.get(source))
        if value is not None:
            pricing[target] = value
    return pricing


def _parse_catalog(payload: Any) -> dict[str, dict[str, Any]]:
    """Normalise a ``GET /models`` response into a catalog keyed by model id.

    Skips non-text and offline models and flattens the fields the profile and
    the Hermes patches consume (capabilities, context length, max completion
    tokens, per-token pricing). Malformed payloads yield an empty dict.
    """
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
            "pricing": _parse_pricing(model_spec),
        }
        normalized_id = model_id.strip()
        catalog[normalized_id] = metadata

    return catalog


def _clamp_reasoning_effort(effort: str, options: Any) -> str | None:
    """Resolve a requested effort against a model's supported effort levels.

    Aliases ``ultra`` to ``max``, then returns the requested level when it is
    supported (or when the supported set is unknown). Otherwise picks the
    nearest supported level by distance along :data:`_REASONING_EFFORT_ORDER`,
    breaking ties toward the lower (cheaper) level. Returns ``None`` for an
    unrecognised effort.
    """
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
            force_refresh: bool = False,
        ) -> dict[str, dict[str, Any]] | None:
            """Return the parsed Venice model catalog, cached per base URL.

            Serves a cached catalog within :data:`_CATALOG_TTL_SECONDS` (an
            unauthenticated cached result is reused for authenticated callers,
            but not vice versa) and otherwise fetches ``GET /models?type=text``.
            Network or parse failures are cached as empty and surface as
            ``None`` so callers fall back to their defaults.
            """
            effective_base = _catalog_key(base_url or self.base_url or "")
            if not effective_base:
                return None

            cached = _catalog_cache.get(effective_base)
            now = time.monotonic()
            if (
                cached
                and now - cached[0] < _CATALOG_TTL_SECONDS
                and (cached[1] or not api_key)
                and not force_refresh
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
            """Return one model's catalog metadata, or ``None`` if not found.

            Tries the exact id first, then the bare id after a ``vendor/``
            prefix, so callers can pass either form.
            """
            if not model:
                return None
            catalog = self._fetch_catalog(
                api_key=os.environ.get("VENICE_API_KEY"),
                base_url=base_url,
            )
            if not catalog:
                return None
            return catalog.get(model) or catalog.get(model.split("/", 1)[-1])

        def _live_pricing_entry(
            self,
            model: str | None,
            *,
            base_url: str | None = None,
            api_key: str | None = None,
        ) -> Any | None:
            """Build a Hermes pricing entry from Venice's live per-million rates.

            Venice publishes per-model token prices on ``GET /models``; the
            parsed catalog stores them per token. Returns ``None`` when the
            model is unknown or carries no usable token pricing, so callers can
            fall back to Hermes's default pricing resolution.
            """
            if not model:
                return None
            catalog = self._fetch_catalog(
                api_key=api_key or os.environ.get("VENICE_API_KEY"),
                base_url=base_url,
            )
            if not catalog:
                return None
            metadata = catalog.get(model) or catalog.get(model.split("/", 1)[-1])
            pricing = metadata.get("pricing") if metadata else None
            if not pricing:
                return None

            from agent.usage_pricing import PricingEntry

            def _per_million(value: str | None) -> Decimal | None:
                if value is None:
                    return None
                try:
                    return Decimal(value) * _ONE_MILLION
                except (InvalidOperation, TypeError, ValueError):
                    return None

            input_cost = _per_million(pricing.get("prompt"))
            output_cost = _per_million(pricing.get("completion"))
            if input_cost is None and output_cost is None:
                return None

            effective_base = _catalog_key(base_url or self.base_url or "")
            return PricingEntry(
                input_cost_per_million=input_cost,
                output_cost_per_million=output_cost,
                cache_read_cost_per_million=_per_million(pricing.get("cache_read")),
                cache_write_cost_per_million=_per_million(pricing.get("cache_write")),
                source="provider_models_api",
                source_url=f"{effective_base}/models" if effective_base else None,
                pricing_version="venice-models-api",
                fetched_at=datetime.now(timezone.utc),
            )

        def _catalog_pricing_map(
            self,
            *,
            base_url: str | None = None,
            api_key: str | None = None,
            force_refresh: bool = False,
        ) -> dict[str, dict[str, str]]:
            """Return per-token pricing keyed by model, for Hermes model pickers.

            Hermes's picker (CLI + desktop) renders pricing from a
            ``{model: {prompt, completion, input_cache_read}}`` map of
            per-token price strings, which is the shape Venice's catalog is
            already parsed into.
            """
            catalog = self._fetch_catalog(
                api_key=api_key or os.environ.get("VENICE_API_KEY"),
                base_url=base_url or os.environ.get("VENICE_BASE_URL"),
                force_refresh=force_refresh,
            )
            if not catalog:
                return {}

            result: dict[str, dict[str, str]] = {}
            for model_id, metadata in catalog.items():
                pricing = metadata.get("pricing") or {}
                row: dict[str, str] = {}
                if pricing.get("prompt") is not None:
                    row["prompt"] = pricing["prompt"]
                if pricing.get("completion") is not None:
                    row["completion"] = pricing["completion"]
                if pricing.get("cache_read") is not None:
                    row["input_cache_read"] = pricing["cache_read"]
                if row:
                    result[model_id] = row
            return result

        def _hermes_model_capabilities(
            self,
            model: str | None,
            *,
            base_url: str | None = None,
        ) -> Any | None:
            """Map Venice's live capability flags to Hermes's ModelCapabilities.

            Lets Hermes surfaces (the desktop model picker's reasoning toggle,
            model-switch validation) reflect Venice's real per-model reasoning,
            tool, vision, and context data instead of defaulting to unknown.
            """
            metadata = self._model_metadata(model, base_url=base_url)
            if not metadata:
                return None

            from agent.models_dev import ModelCapabilities

            capabilities = metadata.get("capabilities") or {}
            kwargs: dict[str, Any] = {
                "supports_tools": bool(
                    capabilities.get("supportsFunctionCalling", True)
                ),
                "supports_vision": bool(capabilities.get("supportsVision")),
                "supports_reasoning": bool(capabilities.get("supportsReasoning")),
            }
            context_length = metadata.get("context_length")
            if (
                isinstance(context_length, int)
                and not isinstance(context_length, bool)
                and context_length > 0
            ):
                kwargs["context_window"] = context_length
            max_output = metadata.get("max_completion_tokens")
            if (
                isinstance(max_output, int)
                and not isinstance(max_output, bool)
                and max_output > 0
            ):
                kwargs["max_output_tokens"] = max_output
            return ModelCapabilities(**kwargs)

        def build_extra_body(
            self, *, session_id: str | None = None, **context
        ) -> dict[str, Any]:
            """Return Venice-specific ``extra_body`` fields for a request.

            Disables Venice's default system prompt (Hermes ships its own) and,
            when a session id is present, sets ``prompt_cache_key`` so a
            conversation's turns route to the same backend for better cache
            hit rates.
            """
            body: dict[str, Any] = {
                "venice_parameters": dict(_DEFAULT_VENICE_PARAMETERS),
            }
            if session_id:
                body["prompt_cache_key"] = session_id

            return body

        def reasoning_effort_options(
            self,
            model: str | None,
            *,
            base_url: str | None = None,
        ) -> list[str]:
            """Return the reasoning-effort levels a Venice model accepts.

            Reads the live catalog's ``supportsReasoningEffort`` and
            ``reasoningEffortOptions`` capability fields. Returns an empty list
            for models that either lack reasoning entirely or reason with a
            fixed, non-configurable budget — those 400 when sent any effort
            parameter, so callers must not pass one.
            """
            metadata = self._model_metadata(model, base_url=base_url)
            capabilities = metadata.get("capabilities", {}) if metadata else {}
            if capabilities.get("supportsReasoningEffort") is not True:
                return []

            options = capabilities.get("reasoningEffortOptions")
            if not isinstance(options, list):
                return []

            normalized: list[str] = []
            for option in options:
                value = str(option).strip().lower()
                if value in _REASONING_EFFORT_ORDER and value not in normalized:
                    normalized.append(value)
            return normalized

        def build_api_kwargs_extras(
            self,
            *,
            reasoning_config: dict | None = None,
            model: str | None = None,
            base_url: str | None = None,
            **context: Any,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            """Translate Hermes reasoning controls into Venice request kwargs.

            Returns a ``(extra_body, top_level_kwargs)`` pair. Disabling
            reasoning (or an explicit ``none`` effort) uses Venice's
            recommended ``reasoning.enabled: false`` toggle. An enabled effort
            is emitted as top-level ``reasoning_effort`` only when the model
            advertises effort support, clamped to a level it accepts;
            otherwise no reasoning kwargs are sent so fixed-budget and
            non-reasoning models do not 400.
            """
            if not isinstance(reasoning_config, dict):
                return {}, {}

            effort = str(reasoning_config.get("effort") or "").strip().lower()
            if reasoning_config.get("enabled") is False or effort == "none":
                return {"reasoning": {"enabled": False}}, {}
            if not effort:
                return {}, {}

            options = self.reasoning_effort_options(model, base_url=base_url)
            if not options:
                return {}, {}

            mapped_effort = _clamp_reasoning_effort(effort, options)
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

    def _install_context_lookup_patch() -> None:
        """Resolve Venice context windows via ``lookup_models_dev_context``.

        Hermes sizes context budgets from ``agent.models_dev`` (models.dev),
        which does not list Venice. This wraps ``lookup_models_dev_context`` so
        Venice models report their real context window from the live catalog.
        See the module docstring for the shared patch contract.
        """
        try:
            import agent.models_dev as models_dev
        except ImportError:
            return

        original = getattr(models_dev, "lookup_models_dev_context", None)
        if not callable(original):
            return
        if getattr(original, "_hermes_venice_context_patch", False):
            return

        @wraps(original)
        def lookup_models_dev_context(provider: str, model: str) -> int | None:
            if _is_venice_provider(provider):
                metadata = venice._model_metadata(
                    model,
                    base_url=os.environ.get("VENICE_BASE_URL") or venice.base_url,
                )
                context_length = metadata.get("context_length") if metadata else None
                if (
                    isinstance(context_length, int)
                    and not isinstance(context_length, bool)
                    and context_length > 0
                ):
                    return context_length
            return original(provider, model)

        lookup_models_dev_context._hermes_venice_context_patch = True
        models_dev.lookup_models_dev_context = lookup_models_dev_context

    def _install_pricing_lookup_patch() -> None:
        """Teach current Hermes releases to price Venice models from live rates.

        Venice returns per-million token prices nested under
        ``model_spec.pricing`` as ``{usd, diem}`` objects, a shape Hermes's
        generic OpenAI-compatible pricing extractor cannot read. This wraps
        ``agent.usage_pricing.get_pricing_entry`` so Venice routes report
        accurate cost (session totals, the expensive-model guard, usage
        reports) from the live catalog. See the module docstring for the
        shared patch contract.
        """
        try:
            import agent.usage_pricing as usage_pricing
        except ImportError:
            return

        original = getattr(usage_pricing, "get_pricing_entry", None)
        if not callable(original):
            return
        if getattr(original, "_hermes_venice_pricing_patch", False):
            return

        @wraps(original)
        def get_pricing_entry(
            model_name: str,
            provider: str | None = None,
            base_url: str | None = None,
            api_key: str | None = None,
        ) -> Any | None:
            if _is_venice_provider(provider) or _is_venice_endpoint(base_url or ""):
                entry = venice._live_pricing_entry(
                    model_name,
                    base_url=(
                        base_url or os.environ.get("VENICE_BASE_URL") or venice.base_url
                    ),
                    api_key=api_key,
                )
                if entry is not None:
                    return entry
            return original(
                model_name,
                provider=provider,
                base_url=base_url,
                api_key=api_key,
            )

        get_pricing_entry._hermes_venice_pricing_patch = True
        usage_pricing.get_pricing_entry = get_pricing_entry

    def _install_picker_pricing_patch() -> None:
        """Show Venice's live per-model pricing in Hermes model pickers.

        The CLI and desktop pickers read pricing from
        ``hermes_cli.models.get_pricing_for_provider``, which natively covers
        only a few aggregators (not Venice). This wraps it to serve Venice's
        catalog pricing so per-model rates render in the picker UI. This is a
        separate code path from cost tracking (see
        :func:`_install_pricing_lookup_patch`). See the module docstring for
        the shared patch contract.
        """
        try:
            import hermes_cli.models as cli_models
        except ImportError:
            return

        original = getattr(cli_models, "get_pricing_for_provider", None)
        if not callable(original):
            return
        if getattr(original, "_hermes_venice_picker_pricing_patch", False):
            return

        @wraps(original)
        def get_pricing_for_provider(
            provider: str, *, force_refresh: bool = False
        ) -> dict[str, dict[str, str]]:
            if _is_venice_provider(provider):
                pricing = venice._catalog_pricing_map(force_refresh=force_refresh)
                if pricing:
                    return pricing
            return original(provider, force_refresh=force_refresh)

        get_pricing_for_provider._hermes_venice_picker_pricing_patch = True
        cli_models.get_pricing_for_provider = get_pricing_for_provider

    def _install_capabilities_patch() -> None:
        """Teach Hermes to resolve Venice model capabilities from the catalog.

        Wraps ``agent.models_dev.get_model_capabilities`` so the desktop
        picker's per-model reasoning toggle and model-switch validation reflect
        Venice's live ``supportsReasoning`` / tool / vision / context data
        instead of the models.dev catalog, which does not list Venice. See the
        module docstring for the shared patch contract.
        """
        try:
            import agent.models_dev as models_dev
        except ImportError:
            return

        original = getattr(models_dev, "get_model_capabilities", None)
        if not callable(original):
            return
        if getattr(original, "_hermes_venice_capabilities_patch", False):
            return

        @wraps(original)
        def get_model_capabilities(provider: str, model: str) -> Any | None:
            if _is_venice_provider(provider):
                caps = venice._hermes_model_capabilities(
                    model,
                    base_url=os.environ.get("VENICE_BASE_URL") or venice.base_url,
                )
                if caps is not None:
                    return caps
            return original(provider, model)

        get_model_capabilities._hermes_venice_capabilities_patch = True
        models_dev.get_model_capabilities = get_model_capabilities

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
    _install_context_lookup_patch()
    _install_pricing_lookup_patch()
    _install_picker_pricing_patch()
    _install_capabilities_patch()


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
    _install_context_lookup_patch()
    _install_pricing_lookup_patch()
    _install_picker_pricing_patch()
    _install_capabilities_patch()


__all__ = ("__version__", "register", "venice")
