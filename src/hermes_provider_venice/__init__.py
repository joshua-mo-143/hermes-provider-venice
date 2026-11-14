"""Venice AI model provider profile for Hermes Agent.

Venice is an OpenAI-compatible inference API
(https://api.venice.ai/api/v1). See https://docs.venice.ai.

Hermes loads this module from a plugin directory (a ``hermes plugins install``
clone, or ``$HERMES_HOME/plugins/model-providers/venice/``) or from a Python
package entry point.

Hermes integration
------------------
The plugin extends Hermes only through documented ``ProviderProfile`` hooks
and never patches Hermes modules, which the Hermes plugin catalog requires:

* ``build_extra_body`` and ``build_api_kwargs_extras`` shape each request.
* ``fetch_models`` lists online, tool-capable text models.
* ``live_context_length`` reports each model's context window.
* ``get_usage_cost`` prices usage from Venice's per-model token rates.
* ``model_capabilities`` carries per-model tool, vision and reasoning flags.
* ``invalidate_catalog_cache`` supports ``/model --refresh``.

Model metadata comes from Venice's live ``GET /models`` catalog, cached in
memory per base URL. Hooks that Hermes calls on every request read only that
cache: when it is cold or stale they start a background refresh and answer
from what is cached, so a slow or failing catalog never delays a request.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, NamedTuple
from urllib.parse import urlparse

__version__ = "0.2.2"

logger = logging.getLogger(__name__)

_CATALOG_TTL_SECONDS = 300.0
# After a failed fetch, wait this long before asking Venice again.
_CATALOG_RETRY_SECONDS = 60.0
_CATALOG_TIMEOUT_SECONDS = 8.0
# The live text catalog is under 200 KB; anything far larger is not a catalog.
_MAX_CATALOG_BYTES = 8 * 1024 * 1024

_SUBCENT = Decimal("0.01")

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


class _Catalog(NamedTuple):
    models: dict[str, dict[str, Any]]
    fetched_at: float  # time.monotonic()
    fetched_on: datetime


_cache_lock = threading.Lock()
_catalogs: dict[str, _Catalog] = {}
# Base URL -> (time.monotonic() of the failure, whether the request had a key).
_failures: dict[str, tuple[float, bool]] = {}
_refreshing: set[str] = set()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # Never follow redirects, so the API key is never forwarded to another host.
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _catalog_key(base_url: str) -> str:
    """Normalise a base URL into the key used for the catalog cache."""
    return base_url.strip().rstrip("/")


def _can_send_credentials(base_url: str) -> bool:
    """Return ``True`` when an API key may be sent to *base_url*.

    Requires HTTPS, except for loopback hosts, so a plain ``http://`` base URL
    never carries the key across the network unencrypted.
    """
    try:
        parsed = urlparse(base_url)
    except ValueError:
        return False
    if parsed.scheme == "https":
        return True
    if parsed.scheme != "http":
        return False
    hostname = (parsed.hostname or "").lower()
    return (
        hostname in ("localhost", "::1")
        or hostname.endswith(".localhost")
        or hostname.startswith("127.")
    )


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _retry_blocked(
    failure: tuple[float, bool] | None, *, has_key: bool, now: float
) -> bool:
    """Return ``True`` when a recent failed fetch rules out asking again yet.

    A failure without a key does not block a request that has one.
    """
    if failure is None:
        return False
    failed_at, failed_with_key = failure
    return now - failed_at < _CATALOG_RETRY_SECONDS and (failed_with_key or not has_key)


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

    Maps Venice's price keys onto ``prompt`` / ``completion`` /
    ``cache_read`` / ``cache_write``. Returns an empty dict when no pricing
    is present.
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

    Skips non-text and offline models and flattens the fields the profile
    consumes (capabilities, context length, per-token pricing). Malformed
    payloads yield an empty dict.
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
            "pricing": _parse_pricing(model_spec),
        }
        normalized_id = model_id.strip()
        catalog[normalized_id] = metadata

    return catalog


def _lookup(
    models: dict[str, dict[str, Any]] | None, model: str | None
) -> dict[str, Any] | None:
    """Return one model's metadata, or ``None`` if not found.

    Tries the exact id first, then the bare id after a ``vendor/`` prefix, so
    callers can pass either form.
    """
    if not models or not model:
        return None
    return models.get(model) or models.get(model.split("/", 1)[-1])


def _effort_options(metadata: dict[str, Any] | None) -> list[str]:
    """Return the reasoning-effort levels a model accepts.

    Reads the catalog's ``supportsReasoningEffort`` and
    ``reasoningEffortOptions`` capability fields. Returns an empty list for
    unknown models and for models that either lack reasoning entirely or
    reason with a fixed, non-configurable budget — those 400 when sent any
    effort parameter, so callers must not pass one.
    """
    capabilities = (metadata or {}).get("capabilities") or {}
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


def _model_capabilities(
    models: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map catalog entries onto Hermes's ``model_overrides`` capability schema.

    Flags the catalog does not report are left out, so Hermes treats them as
    unknown rather than false.
    """
    flags = (
        ("supportsFunctionCalling", "supports_tools"),
        ("supportsVision", "supports_vision"),
        ("supportsReasoning", "supports_reasoning"),
    )
    result: dict[str, dict[str, Any]] = {}
    for model_id, metadata in models.items():
        capabilities = metadata.get("capabilities") or {}
        entry: dict[str, Any] = {
            target: capabilities[source]
            for source, target in flags
            if isinstance(capabilities.get(source), bool)
        }
        if _is_positive_int(metadata.get("context_length")):
            entry["context_window"] = metadata["context_length"]
        if entry:
            result[model_id] = entry
    return result


def _usage_cost_usd(pricing: dict[str, str], usage: Any) -> Decimal | None:
    """Price one response's token usage from per-token rates, as Hermes does.

    Returns ``None`` when tokens of some kind were used but the model has no
    rate for them, so the cost reads as unknown rather than too low.
    """
    amount = Decimal(0)
    for tokens_field, rate_key in (
        ("input_tokens", "prompt"),
        ("output_tokens", "completion"),
        ("cache_read_tokens", "cache_read"),
        ("cache_write_tokens", "cache_write"),
    ):
        tokens = getattr(usage, tokens_field, 0) or 0
        rate = pricing.get(rate_key)
        if rate is None:
            if tokens:
                return None
            continue
        try:
            amount += Decimal(tokens) * Decimal(rate)
        except (InvalidOperation, TypeError, ValueError):
            return None
    return amount


def _cost_label(amount: Decimal) -> str:
    """Format a cost the way Hermes does: ``$0.00``, ``~$0.0046`` or ``~$1.23``."""
    if amount == 0:
        return "$0.00"
    if amount < _SUBCENT:
        label = f"~${amount:.4f}"
        return label if label != "~$0.0000" else "~$<0.0001"
    return f"~${amount:.2f}"


def _fetch(
    base_url: str,
    api_key: str | None,
    headers: Mapping[str, str] | None,
    timeout: float,
) -> dict[str, dict[str, Any]] | None:
    """Fetch and parse ``GET {base_url}/models?type=text``.

    Sends the API key only to HTTPS or loopback base URLs. Returns ``None`` on
    any network or parse failure, or for responses over
    :data:`_MAX_CATALOG_BYTES`.
    """
    request = urllib.request.Request(f"{base_url}/models?type=text")
    if api_key and _can_send_credentials(base_url):
        request.add_header("Authorization", f"Bearer {api_key}")
    request.add_header("Accept", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)

    try:
        with _OPENER.open(request, timeout=timeout) as response:
            raw = response.read(_MAX_CATALOG_BYTES + 1)
        if len(raw) > _MAX_CATALOG_BYTES:
            raise ValueError("Venice catalog response is too large")
        return _parse_catalog(json.loads(raw.decode("utf-8")))
    except Exception as exc:
        logger.debug("Venice catalog fetch from %s failed: %s", base_url, exc)
        return None


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

        def _base(self, base_url: str | None) -> str:
            return _catalog_key(base_url or self.base_url or "")

        def _api_key(self) -> str | None:
            """Return the Venice API key for the active Hermes profile, if any."""
            try:
                from agent.secret_scope import UnscopedSecretError, get_secret
            except ImportError:
                return None
            try:
                return get_secret("VENICE_API_KEY") or None
            except UnscopedSecretError:
                return None

        def _fetch_catalog(
            self,
            *,
            api_key: str | None = None,
            base_url: str | None = None,
            timeout: float = _CATALOG_TIMEOUT_SECONDS,
            force_refresh: bool = False,
        ) -> dict[str, dict[str, Any]] | None:
            """Return the catalog for *base_url*, fetching it when stale.

            Blocks on the network, so only hooks that may do I/O call it.
            Serves a cached catalog younger than :data:`_CATALOG_TTL_SECONDS`.
            After a failed fetch it waits :data:`_CATALOG_RETRY_SECONDS` before
            asking again, and meanwhile returns the last good catalog, if any.
            """
            base = self._base(base_url)
            if not base:
                return None
            if api_key and not _can_send_credentials(base):
                api_key = None

            with _cache_lock:
                cached = _catalogs.get(base)
                failure = _failures.get(base)
            now = time.monotonic()
            if not force_refresh:
                if cached and now - cached.fetched_at < _CATALOG_TTL_SECONDS:
                    return cached.models
                if _retry_blocked(failure, has_key=bool(api_key), now=now):
                    return cached.models if cached else None

            models = _fetch(base, api_key, self.default_headers, timeout)
            with _cache_lock:
                if models is None:
                    _failures[base] = (time.monotonic(), bool(api_key))
                else:
                    _catalogs[base] = _Catalog(
                        models, time.monotonic(), datetime.now(UTC)
                    )
                    _failures.pop(base, None)
            if models is None:
                return cached.models if cached else None

            # Older Hermes releases have no model_capabilities field.
            capabilities = getattr(self, "model_capabilities", None)
            if isinstance(capabilities, dict):
                capabilities.update(_model_capabilities(models))
            return models

        def _cached_catalog(self, base_url: str | None) -> dict[str, dict[str, Any]]:
            """Return the cached catalog for *base_url* without network I/O.

            When the catalog is missing or stale, starts a background refresh
            and returns what is cached (an empty dict when cold).
            """
            base = self._base(base_url)
            if not base:
                return {}
            with _cache_lock:
                cached = _catalogs.get(base)
            if (
                cached is None
                or time.monotonic() - cached.fetched_at >= _CATALOG_TTL_SECONDS
            ):
                self._refresh_in_background(base)
            return cached.models if cached else {}

        def _refresh_in_background(self, base: str) -> None:
            """Fetch *base*'s catalog on a daemon thread, one at a time per URL."""
            api_key = self._api_key()
            if api_key and not _can_send_credentials(base):
                api_key = None
            with _cache_lock:
                if base in _refreshing or _retry_blocked(
                    _failures.get(base), has_key=bool(api_key), now=time.monotonic()
                ):
                    return
                _refreshing.add(base)

            def refresh() -> None:
                try:
                    self._fetch_catalog(
                        api_key=api_key, base_url=base, force_refresh=True
                    )
                finally:
                    with _cache_lock:
                        _refreshing.discard(base)

            try:
                threading.Thread(
                    target=refresh, name="venice-catalog-refresh", daemon=True
                ).start()
            except RuntimeError as exc:
                logger.debug("Could not start the Venice catalog refresh: %s", exc)
                with _cache_lock:
                    _refreshing.discard(base)

        def _any_cached(
            self, model: str | None
        ) -> tuple[dict[str, Any] | None, _Catalog | None]:
            """Find *model* in any cached catalog, the default base URL's first."""
            with _cache_lock:
                catalogs = [_catalogs.get(self._base(None)), *_catalogs.values()]
            for catalog in catalogs:
                metadata = _lookup(catalog.models, model) if catalog else None
                if metadata:
                    return metadata, catalog
            return None, None

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
            is emitted as top-level ``reasoning_effort`` only when the catalog
            says the model supports effort, clamped to a level it accepts;
            otherwise no reasoning kwargs are sent so fixed-budget and
            non-reasoning models do not 400.

            Hermes calls this for every request, so it reads the catalog cache
            only; until the cache is warm, no effort is sent.
            """
            models = self._cached_catalog(base_url)
            if not isinstance(reasoning_config, dict):
                return {}, {}

            effort = str(reasoning_config.get("effort") or "").strip().lower()
            if reasoning_config.get("enabled") is False or effort == "none":
                return {"reasoning": {"enabled": False}}, {}
            if not effort:
                return {}, {}

            options = _effort_options(_lookup(models, model))
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
            timeout: float = _CATALOG_TIMEOUT_SECONDS,
        ) -> list[str] | None:
            """Return online text models that can support an agent tool loop."""
            models = self._fetch_catalog(
                api_key=api_key or self._api_key(),
                base_url=base_url,
                timeout=timeout,
            )
            if not models:
                return None

            return [
                model_id
                for model_id, metadata in models.items()
                if metadata.get("capabilities", {}).get("supportsFunctionCalling")
                is True
            ] or None

        def live_context_length(
            self, model: str, *, api_key: str = "", base_url: str = ""
        ) -> int | None:
            """Return *model*'s context window from Venice's live catalog."""
            models = self._fetch_catalog(
                api_key=api_key or self._api_key(), base_url=base_url or None
            )
            metadata = _lookup(models, model)
            context_length = metadata.get("context_length") if metadata else None
            return context_length if _is_positive_int(context_length) else None

        def get_usage_cost(self, model: str, usage: Any) -> Any | None:
            """Estimate a response's cost from Venice's per-model token rates.

            Hermes calls this after every response, so it reads the catalog
            cache only. Returns ``None`` when the model or a rate it needs is
            unknown, which leaves the cost to Hermes's own pricing lookup.
            """
            metadata, catalog = self._any_cached(model)
            pricing = metadata.get("pricing") if metadata else None
            if not pricing or catalog is None:
                return None
            amount = _usage_cost_usd(pricing, usage)
            if amount is None:
                return None

            from agent.usage_pricing import CostResult

            return CostResult(
                amount_usd=amount,
                status="estimated",
                source="provider_models_api",
                label=_cost_label(amount),
                fetched_at=catalog.fetched_on,
                pricing_version="venice-models-api",
            )

        def invalidate_catalog_cache(self) -> None:
            """Drop cached catalogs so the next lookup fetches a fresh one."""
            with _cache_lock:
                _catalogs.clear()
                _failures.clear()

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
            # Previous generation, still served
            "zai-org-glm-4.7",
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
