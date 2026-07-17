from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
import tomllib
import types
from pathlib import Path

import pytest


@pytest.fixture
def plugin(monkeypatch: pytest.MonkeyPatch):
    registered = []

    class ProviderProfile:
        def __init__(self, **kwargs):
            self.default_headers = {}
            for key, value in kwargs.items():
                setattr(self, key, value)

    providers = types.ModuleType("providers")
    providers.register_provider = registered.append
    base = types.ModuleType("providers.base")
    base.ProviderProfile = ProviderProfile

    monkeypatch.setitem(sys.modules, "providers", providers)
    monkeypatch.setitem(sys.modules, "providers.base", base)
    sys.modules.pop("hermes_provider_venice", None)

    module = importlib.import_module("hermes_provider_venice")
    yield module, registered
    sys.modules.pop("hermes_provider_venice", None)


def test_profile_registers_on_import_and_via_entry_point(plugin) -> None:
    module, registered = plugin

    assert registered == [module.venice]
    assert module.venice.name == "venice"
    assert module.venice.base_url == "https://api.venice.ai/api/v1"
    assert "api.venice.ai" in module.venice.aliases

    module.register(object())
    assert registered == [module.venice, module.venice]


def test_extra_body_maps_session(plugin) -> None:
    module, _ = plugin

    body = module.venice.build_extra_body(
        session_id="session-123",
        reasoning_config={"enabled": False},
    )

    assert body == {
        "prompt_cache_key": "session-123",
        "venice_parameters": {
            "include_venice_system_prompt": False,
        },
    }


def test_fetch_models_filters_non_text_and_offline_entries(
    plugin, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _ = plugin
    payload = {
        "data": [
            {
                "id": "online-text",
                "type": "text",
                "model_spec": {
                    "capabilities": {"supportsFunctionCalling": True},
                    "pricing": {
                        "input": {"usd": 1.5},
                        "output": {"usd": 4.0},
                    },
                },
                "context_length": 1_000_000,
            },
            {
                "id": "offline-text",
                "type": "text",
                "model_spec": {"offline": True},
            },
            {
                "id": "no-tools",
                "type": "text",
                "model_spec": {
                    "capabilities": {"supportsFunctionCalling": False},
                },
            },
            {"id": "image-only", "type": "image"},
            {"id": "implicit-text"},
        ]
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return json.dumps(payload).encode()

    security = types.ModuleType("hermes_cli.urllib_security")
    security.open_credentialed_url = lambda *_args, **_kwargs: Response()
    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.urllib_security = security
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.urllib_security", security)

    assert module.venice.fetch_models(api_key="secret") == ["online-text"]
    assert module.venice._model_metadata("online-text") == {
        "capabilities": {"supportsFunctionCalling": True},
        "context_length": 1_000_000,
        "max_completion_tokens": None,
        "pricing": {
            "prompt": "0.0000015",
            "completion": "0.000004",
        },
    }


def test_reasoning_uses_supported_effort_and_clamps(plugin) -> None:
    module, _ = plugin
    module._catalog_cache[module.venice.base_url] = (
        module.time.monotonic(),
        {
            "effort-model": {
                "capabilities": {
                    "supportsReasoning": True,
                    "supportsReasoningEffort": True,
                    "reasoningEffortOptions": ["low", "medium", "high"],
                }
            },
            "fixed-reasoning-model": {
                "capabilities": {
                    "supportsReasoning": True,
                    "supportsReasoningEffort": False,
                }
            },
        },
    )

    assert module.venice.build_api_kwargs_extras(
        model="effort-model",
        reasoning_config={"enabled": True, "effort": "xhigh"},
    ) == ({}, {"reasoning_effort": "high"})
    assert module.venice.build_api_kwargs_extras(
        model="fixed-reasoning-model",
        reasoning_config={"enabled": True, "effort": "high"},
    ) == ({}, {})


def test_reasoning_disable_uses_venice_recommended_shape(plugin) -> None:
    module, _ = plugin

    assert module.venice.build_api_kwargs_extras(
        model="any-model",
        reasoning_config={"enabled": False, "effort": "high"},
    ) == ({"reasoning": {"enabled": False}}, {})
    assert module.venice.build_api_kwargs_extras(
        model="any-model",
        reasoning_config={"enabled": True, "effort": "none"},
    ) == ({"reasoning": {"enabled": False}}, {})


def test_reasoning_effort_options_reflect_live_catalog(plugin) -> None:
    module, _ = plugin
    module._catalog_cache[module.venice.base_url] = (
        module.time.monotonic(),
        {
            "effort-model": {
                "capabilities": {
                    "supportsReasoning": True,
                    "supportsReasoningEffort": True,
                    "reasoningEffortOptions": ["low", "LOW", "medium", "bogus"],
                }
            },
            "fixed-reasoning-model": {
                "capabilities": {
                    "supportsReasoning": True,
                    "supportsReasoningEffort": False,
                }
            },
        },
    )

    assert module.venice.reasoning_effort_options("effort-model") == [
        "low",
        "medium",
    ]
    assert module.venice.reasoning_effort_options("fixed-reasoning-model") == []
    assert module.venice.reasoning_effort_options("unknown-model") == []


def _install_fake_usage_pricing(monkeypatch: pytest.MonkeyPatch):
    from dataclasses import dataclass
    from decimal import Decimal
    from typing import Optional

    @dataclass(frozen=True)
    class PricingEntry:
        input_cost_per_million: Optional[Decimal] = None
        output_cost_per_million: Optional[Decimal] = None
        cache_read_cost_per_million: Optional[Decimal] = None
        cache_write_cost_per_million: Optional[Decimal] = None
        request_cost: Optional[Decimal] = None
        source: str = "none"
        source_url: Optional[str] = None
        pricing_version: Optional[str] = None
        fetched_at: object = None

    fallback_calls: list[tuple] = []

    def original_get_pricing_entry(
        model_name, provider=None, base_url=None, api_key=None
    ):
        fallback_calls.append((model_name, provider, base_url, api_key))
        return None

    usage_pricing = types.ModuleType("agent.usage_pricing")
    usage_pricing.PricingEntry = PricingEntry
    usage_pricing.get_pricing_entry = original_get_pricing_entry

    agent = sys.modules.get("agent") or types.ModuleType("agent")
    agent.usage_pricing = usage_pricing
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.usage_pricing", usage_pricing)
    return usage_pricing, fallback_calls, Decimal


def test_pricing_patch_injects_live_venice_rates(
    plugin, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _ = plugin
    usage_pricing, fallback_calls, Decimal = _install_fake_usage_pricing(monkeypatch)

    module._catalog_cache[module.venice.base_url] = (
        module.time.monotonic(),
        {
            "priced-model": {
                "capabilities": {"supportsFunctionCalling": True},
                "context_length": 131_072,
                "max_completion_tokens": None,
                # Per-token strings, as stored by _parse_pricing: 0.7 / 2.8 /
                # 0.35 USD per million tokens for input / output / cache read.
                "pricing": {
                    "prompt": "0.0000007",
                    "completion": "0.0000028",
                    "cache_read": "0.00000035",
                },
            }
        },
    )
    module._install_pricing_lookup_patch()

    entry = usage_pricing.get_pricing_entry("priced-model", provider="venice")
    assert entry.input_cost_per_million == Decimal("0.7")
    assert entry.output_cost_per_million == Decimal("2.8")
    assert entry.cache_read_cost_per_million == Decimal("0.35")
    assert entry.cache_write_cost_per_million is None
    assert entry.source == "provider_models_api"
    assert entry.pricing_version == "venice-models-api"
    assert fallback_calls == []


def test_pricing_patch_is_scoped_to_venice(
    plugin, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _ = plugin
    usage_pricing, fallback_calls, _ = _install_fake_usage_pricing(monkeypatch)

    module._catalog_cache[module.venice.base_url] = (
        module.time.monotonic(),
        {
            "priced-model": {
                "capabilities": {"supportsFunctionCalling": True},
                "context_length": 131_072,
                "max_completion_tokens": None,
                "pricing": {"prompt": "0.0000007", "completion": "0.0000028"},
            }
        },
    )
    module._install_pricing_lookup_patch()

    # Unknown model on a Venice route falls through to Hermes' resolver.
    assert usage_pricing.get_pricing_entry("mystery", provider="venice") is None
    # Other providers are never intercepted.
    assert usage_pricing.get_pricing_entry("priced-model", provider="openai") is None
    assert ("mystery", "venice", None, None) in fallback_calls
    assert ("priced-model", "openai", None, None) in fallback_calls


def test_pricing_patch_is_idempotent(plugin, monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = plugin
    usage_pricing, _, _ = _install_fake_usage_pricing(monkeypatch)

    module._install_pricing_lookup_patch()
    patched = usage_pricing.get_pricing_entry
    module._install_pricing_lookup_patch()

    assert usage_pricing.get_pricing_entry is patched


def test_picker_pricing_patch_exposes_catalog_pricing(
    plugin, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _ = plugin

    fallback_calls: list[tuple] = []

    def original_get_pricing_for_provider(provider, *, force_refresh=False):
        fallback_calls.append((provider, force_refresh))
        return {}

    cli_models = types.ModuleType("hermes_cli.models")
    cli_models.get_pricing_for_provider = original_get_pricing_for_provider
    hermes_cli = sys.modules.get("hermes_cli") or types.ModuleType("hermes_cli")
    hermes_cli.models = cli_models
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.models", cli_models)

    module._catalog_cache[module.venice.base_url] = (
        module.time.monotonic(),
        {
            "priced-model": {
                "capabilities": {"supportsFunctionCalling": True},
                "context_length": 131_072,
                "max_completion_tokens": None,
                "pricing": {
                    "prompt": "0.0000007",
                    "completion": "0.0000028",
                    "cache_read": "0.00000035",
                },
            },
            "unpriced-model": {
                "capabilities": {"supportsFunctionCalling": True},
                "context_length": 65_536,
                "max_completion_tokens": None,
                "pricing": {},
            },
        },
    )
    module._install_picker_pricing_patch()

    pricing = cli_models.get_pricing_for_provider("venice")
    assert pricing == {
        "priced-model": {
            "prompt": "0.0000007",
            "completion": "0.0000028",
            "input_cache_read": "0.00000035",
        }
    }
    assert fallback_calls == []

    # Other providers fall through to Hermes' native pricing resolution.
    assert cli_models.get_pricing_for_provider("openrouter") == {}
    assert ("openrouter", False) in fallback_calls


def test_capabilities_patch_maps_venice_reasoning(
    plugin, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _ = plugin
    from dataclasses import dataclass

    @dataclass
    class ModelCapabilities:
        supports_tools: bool = True
        supports_vision: bool = False
        supports_reasoning: bool = False
        context_window: int = 200000
        max_output_tokens: int = 8192
        model_family: str = ""

    fallback_calls: list[tuple] = []

    def original_get_model_capabilities(provider, model):
        fallback_calls.append((provider, model))
        return None

    models_dev = types.ModuleType("agent.models_dev")
    models_dev.ModelCapabilities = ModelCapabilities
    models_dev.get_model_capabilities = original_get_model_capabilities
    agent = sys.modules.get("agent") or types.ModuleType("agent")
    agent.models_dev = models_dev
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.models_dev", models_dev)

    module._catalog_cache[module.venice.base_url] = (
        module.time.monotonic(),
        {
            "reasoner": {
                "capabilities": {
                    "supportsFunctionCalling": True,
                    "supportsReasoning": True,
                    "supportsVision": True,
                },
                "context_length": 262_144,
                "max_completion_tokens": 32_000,
                "pricing": {},
            },
            "non-reasoner": {
                "capabilities": {
                    "supportsFunctionCalling": True,
                    "supportsReasoning": False,
                },
                "context_length": 128_000,
                "max_completion_tokens": None,
                "pricing": {},
            },
        },
    )
    module._install_capabilities_patch()

    reasoner = models_dev.get_model_capabilities("venice", "reasoner")
    assert reasoner.supports_reasoning is True
    assert reasoner.supports_vision is True
    assert reasoner.supports_tools is True
    assert reasoner.context_window == 262_144
    assert reasoner.max_output_tokens == 32_000

    plain = models_dev.get_model_capabilities("venice", "non-reasoner")
    assert plain.supports_reasoning is False
    assert plain.context_window == 128_000
    # Unknown output cap keeps the dataclass default.
    assert plain.max_output_tokens == 8192

    # Non-venice providers still resolve through models.dev.
    assert models_dev.get_model_capabilities("openai", "gpt-4o") is None
    assert ("openai", "gpt-4o") in fallback_calls


def test_context_metadata_patch_is_scoped_to_venice(
    plugin, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _ = plugin
    models_dev = types.ModuleType("agent.models_dev")
    models_dev.lookup_models_dev_context = lambda *_args, **_kwargs: 123_456
    agent = types.ModuleType("agent")
    agent.models_dev = models_dev
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.models_dev", models_dev)

    expected = {
        "test-model": {
            "capabilities": {"supportsFunctionCalling": True},
            "context_length": 500_000,
            "max_completion_tokens": 32_000,
            "pricing": {},
        }
    }
    module._catalog_cache[module.venice.base_url] = (
        module.time.monotonic(),
        expected,
    )
    module._install_context_lookup_patch()

    assert models_dev.lookup_models_dev_context("venice", "test-model") == 500_000
    assert (
        models_dev.lookup_models_dev_context("other-provider", "test-model") == 123_456
    )


def test_distribution_exposes_both_discovery_entry_points() -> None:
    entry_points = {
        (entry_point.group, entry_point.name): entry_point.value
        for entry_point in importlib.metadata.distribution(
            "hermes-provider-venice"
        ).entry_points
    }

    assert entry_points[("hermes_agent.plugins", "venice")] == (
        "hermes_provider_venice"
    )
    assert entry_points[("hermes_agent.model_providers", "venice")] == (
        "hermes_provider_venice:register"
    )


def test_release_versions_stay_synchronized(plugin) -> None:
    module, _ = plugin
    root = Path(__file__).resolve().parents[1]
    project_version = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    manifest_version = next(
        line.partition(":")[2].strip()
        for line in (root / "src/hermes_provider_venice/plugin.yaml")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("version:")
    )

    assert module.__version__ == project_version == manifest_version


def test_compatibility_installer_copies_plugin_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_provider_venice import __main__

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    destination = __main__.install()

    assert destination == tmp_path / "plugins" / "model-providers" / "venice"
    assert (destination / "__init__.py").is_file()
    assert (destination / "plugin.yaml").is_file()
    assert not (destination / "__main__.py").exists()


def test_update_upgrades_via_pip_from_pypi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_provider_venice import __main__

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls: list[list[str]] = []

    def fake_run(command, check):
        calls.append(command)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(__main__.subprocess, "run", fake_run)

    __main__.update()

    assert calls == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "hermes-provider-venice",
        ]
    ]
    # Nothing installed in HERMES_HOME, so no directory copy is created.
    assert not (tmp_path / "plugins").exists()


def test_update_passes_pre_and_source(monkeypatch: pytest.MonkeyPatch) -> None:
    from hermes_provider_venice import __main__

    calls: list[list[str]] = []
    monkeypatch.setattr(
        __main__.subprocess,
        "run",
        lambda command, check: calls.append(command),
    )

    __main__.update(pre=True, source="git+https://example.com/pkg.git@main")

    assert calls == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--pre",
            "git+https://example.com/pkg.git@main",
        ]
    ]


def test_update_refreshes_existing_directory_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_provider_venice import __main__

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    __main__.install()

    monkeypatch.setattr(__main__.subprocess, "run", lambda command, check: None)

    __main__.update()

    destination = tmp_path / "plugins" / "model-providers" / "venice"
    assert (destination / "__init__.py").is_file()


def test_update_reports_pip_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from hermes_provider_venice import __main__

    def fake_run(command, check):
        raise __main__.subprocess.CalledProcessError(returncode=2, cmd=command)

    monkeypatch.setattr(__main__.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="pip failed to upgrade"):
        __main__.update()
