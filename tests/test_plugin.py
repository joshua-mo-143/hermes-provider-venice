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
                },
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
