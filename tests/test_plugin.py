from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
import tomllib
import types
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._body if size < 0 else self._body[:size]


@pytest.fixture
def plugin(monkeypatch: pytest.MonkeyPatch):
    """Import the plugin against fake Hermes modules, a fake HTTP opener and a
    thread starter that queues background refreshes instead of running them.

    Set ``response`` to catalog bytes or an exception; ``requests`` records each
    catalog request; ``run_background()`` runs the queued refreshes.
    """
    registered = []

    class ProviderProfile:
        def __init__(self, **kwargs):
            self.default_headers = {}
            self.model_capabilities = {}
            for key, value in kwargs.items():
                setattr(self, key, value)

    providers = types.ModuleType("providers")
    providers.register_provider = registered.append
    base = types.ModuleType("providers.base")
    base.ProviderProfile = ProviderProfile
    monkeypatch.setitem(sys.modules, "providers", providers)
    monkeypatch.setitem(sys.modules, "providers.base", base)

    harness = types.SimpleNamespace(
        registered=registered, requests=[], pending=[], response=None, secret=None
    )

    class UnscopedSecretError(RuntimeError):
        pass

    secret_scope = types.ModuleType("agent.secret_scope")
    secret_scope.UnscopedSecretError = UnscopedSecretError
    secret_scope.get_secret = lambda name, default=None: (
        harness.secret if name == "VENICE_API_KEY" else default
    )
    agent = types.ModuleType("agent")
    agent.secret_scope = secret_scope
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.secret_scope", secret_scope)

    sys.modules.pop("hermes_provider_venice", None)
    module = importlib.import_module("hermes_provider_venice")
    harness.module = module
    harness.real_opener = module._OPENER

    class Opener:
        def open(self, req: urllib.request.Request, timeout: float) -> _Response:
            harness.requests.append(
                types.SimpleNamespace(
                    url=req.full_url, authorization=req.get_header("Authorization")
                )
            )
            if harness.response is None:
                raise OSError("no catalog response configured")
            if isinstance(harness.response, BaseException):
                raise harness.response
            return _Response(harness.response)

    class Thread:
        def __init__(self, *, target, name: str, daemon: bool) -> None:
            self.target = target

        def start(self) -> None:
            harness.pending.append(self.target)

    def run_background() -> None:
        while harness.pending:
            harness.pending.pop(0)()

    harness.run_background = run_background
    monkeypatch.setattr(module, "_OPENER", Opener())
    monkeypatch.setattr(module, "threading", types.SimpleNamespace(Thread=Thread))
    yield harness
    sys.modules.pop("hermes_provider_venice", None)


@pytest.fixture
def clock(plugin, monkeypatch: pytest.MonkeyPatch):
    now = [1_000.0]
    monkeypatch.setattr(
        plugin.module, "time", types.SimpleNamespace(monotonic=lambda: now[0])
    )

    def advance(seconds: float) -> None:
        now[0] += seconds

    return types.SimpleNamespace(advance=advance)


def _catalog(*models: dict[str, Any]) -> bytes:
    return json.dumps({"data": list(models)}).encode()


def _seed(module: Any, *models: dict[str, Any]) -> None:
    """Put a freshly fetched catalog for the default base URL in the cache."""
    module._catalogs[module.venice.base_url] = module._Catalog(
        module._parse_catalog({"data": list(models)}),
        module.time.monotonic(),
        datetime.now(UTC),
    )


_TOOL_MODEL = {
    "id": "tool-model",
    "type": "text",
    "model_spec": {"capabilities": {"supportsFunctionCalling": True}},
}

_EFFORT_MODEL = {
    "id": "effort-model",
    "type": "text",
    "model_spec": {
        "capabilities": {
            "supportsFunctionCalling": True,
            "supportsReasoning": True,
            "supportsReasoningEffort": True,
            "reasoningEffortOptions": ["low", "medium", "high"],
        },
        "availableContextTokens": 262_144,
        # USD per million tokens, as Venice publishes them.
        "pricing": {
            "input": {"usd": 0.7, "diem": 0.7},
            "output": {"usd": 2.8, "diem": 2.8},
            "cache_input": {"usd": 0.35, "diem": 0.35},
        },
    },
}

_FIXED_REASONING_MODEL = {
    "id": "fixed-reasoning-model",
    "type": "text",
    "model_spec": {
        "capabilities": {"supportsReasoning": True, "supportsReasoningEffort": False}
    },
}


def test_profile_registers_on_import_and_via_entry_point(plugin) -> None:
    module = plugin.module

    assert plugin.registered == [module.venice]
    assert module.venice.name == "venice"
    assert module.venice.base_url == "https://api.venice.ai/api/v1"
    assert "api.venice.ai" in module.venice.aliases

    module.register(object())
    assert plugin.registered == [module.venice, module.venice]


def test_extra_body_maps_session(plugin) -> None:
    body = plugin.module.venice.build_extra_body(
        session_id="session-123",
        reasoning_config={"enabled": False},
    )

    assert body == {
        "prompt_cache_key": "session-123",
        "venice_parameters": {
            "include_venice_system_prompt": False,
        },
    }


def test_fetch_models_filters_non_text_and_offline_entries(plugin) -> None:
    module = plugin.module
    plugin.response = _catalog(
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
        {"id": "offline-text", "type": "text", "model_spec": {"offline": True}},
        {
            "id": "no-tools",
            "type": "text",
            "model_spec": {"capabilities": {"supportsFunctionCalling": False}},
        },
        {"id": "image-only", "type": "image"},
        {"id": "implicit-text"},
    )

    assert module.venice.fetch_models(api_key="secret") == ["online-text"]
    assert module._catalogs[module.venice.base_url].models["online-text"] == {
        "capabilities": {"supportsFunctionCalling": True},
        "context_length": 1_000_000,
        "pricing": {
            "prompt": "0.0000015",
            "completion": "0.000004",
        },
    }


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.venice.ai/api/v1",
        "http://localhost:8080/v1",
        "http://127.0.0.1:8080/v1",
        "http://[::1]:8080/v1",
    ],
)
def test_catalog_fetch_sends_key_over_https_and_loopback(plugin, base_url: str) -> None:
    plugin.response = _catalog(_TOOL_MODEL)

    assert plugin.module.venice.fetch_models(api_key="secret", base_url=base_url) == [
        "tool-model"
    ]
    assert plugin.requests[0].authorization == "Bearer secret"


def test_catalog_fetch_withholds_key_over_plain_http(plugin) -> None:
    plugin.response = _catalog(_TOOL_MODEL)

    assert plugin.module.venice.fetch_models(
        api_key="secret", base_url="http://proxy.example.com/v1"
    ) == ["tool-model"]
    assert plugin.requests[0].url == "http://proxy.example.com/v1/models?type=text"
    assert plugin.requests[0].authorization is None


def test_catalog_fetch_rejects_oversized_response(
    plugin, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin.response = _catalog(_TOOL_MODEL)
    monkeypatch.setattr(plugin.module, "_MAX_CATALOG_BYTES", len(plugin.response) - 1)

    assert plugin.module.venice.fetch_models(api_key="secret") is None


def test_catalog_fetch_never_follows_redirects(plugin) -> None:
    module = plugin.module
    request = urllib.request.Request(
        "https://api.venice.ai/api/v1/models?type=text",
        headers={"Authorization": "Bearer secret"},
    )

    assert any(
        isinstance(handler, module._NoRedirect)
        for handler in plugin.real_opener.handlers
    )
    assert (
        module._NoRedirect().redirect_request(
            request, None, 302, "Found", {}, "https://elsewhere.example/models"
        )
        is None
    )


def test_api_key_comes_from_the_hermes_secret_scope(
    plugin, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin.secret = "scoped-key"
    monkeypatch.setenv("VENICE_API_KEY", "process-env-key")
    plugin.response = _catalog(_TOOL_MODEL)

    assert plugin.module.venice.fetch_models() == ["tool-model"]
    assert plugin.requests[0].authorization == "Bearer scoped-key"


def test_context_window_comes_from_the_live_catalog(plugin) -> None:
    venice = plugin.module.venice
    plugin.secret = "scoped-key"
    plugin.response = _catalog(_EFFORT_MODEL)

    assert venice.live_context_length("effort-model", api_key="hermes-key") == 262_144
    assert venice.live_context_length("vendor/effort-model") == 262_144
    assert venice.live_context_length("unknown-model") is None
    # The key Hermes passes wins, and the catalog is fetched once.
    assert [request.authorization for request in plugin.requests] == [
        "Bearer hermes-key"
    ]


def test_failed_fetch_waits_before_retrying(plugin, clock) -> None:
    venice = plugin.module.venice
    plugin.response = OSError("Venice is unreachable")

    for _ in range(5):
        assert venice.live_context_length("effort-model", api_key="secret") is None
    assert len(plugin.requests) == 1

    clock.advance(plugin.module._CATALOG_RETRY_SECONDS)
    plugin.response = _catalog(_EFFORT_MODEL)
    assert venice.live_context_length("effort-model", api_key="secret") == 262_144
    assert len(plugin.requests) == 2


def test_failure_without_key_does_not_block_a_request_with_one(plugin) -> None:
    venice = plugin.module.venice
    plugin.response = OSError("401 Unauthorized")
    assert venice.fetch_models() is None

    plugin.response = _catalog(_TOOL_MODEL)
    assert venice.fetch_models(api_key="secret") == ["tool-model"]
    assert [request.authorization for request in plugin.requests] == [
        None,
        "Bearer secret",
    ]


def test_last_good_catalog_is_served_while_venice_is_down(plugin, clock) -> None:
    venice = plugin.module.venice
    plugin.response = _catalog(_EFFORT_MODEL)
    assert venice.live_context_length("effort-model") == 262_144

    clock.advance(plugin.module._CATALOG_TTL_SECONDS)
    plugin.response = OSError("Venice is unreachable")
    assert venice.live_context_length("effort-model") == 262_144
    assert len(plugin.requests) == 2


def test_request_hooks_read_the_cache_and_refresh_in_background(plugin) -> None:
    venice = plugin.module.venice
    plugin.response = _catalog(_EFFORT_MODEL)
    reasoning = {"enabled": True, "effort": "xhigh"}

    # Cold cache: nothing is fetched inline, so no effort can be sent yet.
    assert venice.build_api_kwargs_extras(
        model="effort-model", reasoning_config=reasoning
    ) == ({}, {})
    assert plugin.requests == []
    assert len(plugin.pending) == 1

    # Requests made while a refresh is pending do not start another one.
    venice.build_api_kwargs_extras(model="effort-model", reasoning_config=reasoning)
    assert len(plugin.pending) == 1

    plugin.run_background()
    assert venice.build_api_kwargs_extras(
        model="effort-model", reasoning_config=reasoning
    ) == ({}, {"reasoning_effort": "high"})
    assert len(plugin.requests) == 1
    assert plugin.pending == []


def test_reasoning_uses_supported_effort_and_clamps(plugin) -> None:
    module = plugin.module
    _seed(module, _EFFORT_MODEL, _FIXED_REASONING_MODEL)

    assert module.venice.build_api_kwargs_extras(
        model="effort-model",
        reasoning_config={"enabled": True, "effort": "xhigh"},
    ) == ({}, {"reasoning_effort": "high"})
    assert module.venice.build_api_kwargs_extras(
        model="fixed-reasoning-model",
        reasoning_config={"enabled": True, "effort": "high"},
    ) == ({}, {})


def test_reasoning_disable_uses_venice_recommended_shape(plugin) -> None:
    venice = plugin.module.venice

    assert venice.build_api_kwargs_extras(
        model="any-model",
        reasoning_config={"enabled": False, "effort": "high"},
    ) == ({"reasoning": {"enabled": False}}, {})
    assert venice.build_api_kwargs_extras(
        model="any-model",
        reasoning_config={"enabled": True, "effort": "none"},
    ) == ({"reasoning": {"enabled": False}}, {})


def test_effort_options_reflect_the_catalog(plugin) -> None:
    module = plugin.module
    effort = {
        "capabilities": {
            "supportsReasoning": True,
            "supportsReasoningEffort": True,
            "reasoningEffortOptions": ["low", "LOW", "medium", "bogus"],
        }
    }
    fixed = {
        "capabilities": {"supportsReasoning": True, "supportsReasoningEffort": False}
    }

    assert module._effort_options(effort) == ["low", "medium"]
    assert module._effort_options(fixed) == []
    assert module._effort_options(None) == []


def test_catalog_fills_model_capabilities(plugin) -> None:
    venice = plugin.module.venice
    plugin.response = _catalog(
        _EFFORT_MODEL,
        {
            "id": "vision-model",
            "type": "text",
            "model_spec": {
                "capabilities": {
                    "supportsFunctionCalling": False,
                    "supportsVision": True,
                }
            },
        },
    )

    venice.fetch_models()

    # Flags the catalog omits stay out, so Hermes treats them as unknown.
    assert venice.model_capabilities == {
        "effort-model": {
            "supports_tools": True,
            "supports_reasoning": True,
            "context_window": 262_144,
        },
        "vision-model": {"supports_tools": False, "supports_vision": True},
    }


def _install_cost_result(monkeypatch: pytest.MonkeyPatch) -> type:
    @dataclass(frozen=True)
    class CostResult:
        amount_usd: Decimal | None
        status: str
        source: str
        label: str
        fetched_at: datetime | None = None
        pricing_version: str | None = None
        notes: tuple[str, ...] = ()

    usage_pricing = types.ModuleType("agent.usage_pricing")
    usage_pricing.CostResult = CostResult
    sys.modules["agent"].usage_pricing = usage_pricing
    monkeypatch.setitem(sys.modules, "agent.usage_pricing", usage_pricing)
    return CostResult


def test_usage_cost_is_estimated_from_live_rates(
    plugin, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = plugin.module
    cost_result = _install_cost_result(monkeypatch)
    _seed(module, _EFFORT_MODEL)
    usage = types.SimpleNamespace(
        input_tokens=1_000_000,
        output_tokens=500_000,
        cache_read_tokens=2_000_000,
        cache_write_tokens=0,
    )

    cost = module.venice.get_usage_cost("effort-model", usage)

    assert isinstance(cost, cost_result)
    # $0.70 + 0.5 × $2.80 + 2 × $0.35
    assert cost.amount_usd == Decimal("2.8")
    assert cost.status == "estimated"
    assert cost.source == "provider_models_api"
    assert cost.label == "~$2.80"
    assert cost.pricing_version == "venice-models-api"


def test_usage_cost_is_left_to_hermes_when_unknown(
    plugin, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = plugin.module
    _install_cost_result(monkeypatch)
    usage = types.SimpleNamespace(
        input_tokens=10, output_tokens=10, cache_read_tokens=0, cache_write_tokens=5
    )

    assert module.venice.get_usage_cost("effort-model", usage) is None
    _seed(module, _EFFORT_MODEL)
    assert module.venice.get_usage_cost("unknown-model", usage) is None
    # The model has no cache-write rate, so a response with cache writes is
    # unpriced rather than undercounted.
    assert module.venice.get_usage_cost("effort-model", usage) is None
    assert plugin.requests == []


@pytest.mark.parametrize(
    ("amount", "label"),
    [
        ("0", "$0.00"),
        ("0.0046", "~$0.0046"),
        ("0.00001", "~$<0.0001"),
        ("1.234", "~$1.23"),
    ],
)
def test_cost_label_matches_the_hermes_format(plugin, amount: str, label: str) -> None:
    assert plugin.module._cost_label(Decimal(amount)) == label


def test_model_refresh_drops_the_cached_catalog(plugin) -> None:
    venice = plugin.module.venice
    plugin.response = _catalog(_TOOL_MODEL)

    venice.fetch_models()
    venice.invalidate_catalog_cache()
    venice.fetch_models()

    assert len(plugin.requests) == 2


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
    project_version = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    manifest_version = next(
        line.partition(":")[2].strip()
        for line in (ROOT / "src/hermes_provider_venice/plugin.yaml")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("version:")
    )

    assert plugin.module.__version__ == project_version == manifest_version


def test_github_links_use_the_owning_account() -> None:
    sources = [
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / "src/hermes_provider_venice/README.md",
        ROOT / "src/hermes_provider_venice_cli/__init__.py",
    ]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert "github.com/joshua-mo/" not in text, path
    assert "github.com/joshua-mo-143/hermes-provider-venice" in (
        ROOT / "pyproject.toml"
    ).read_text(encoding="utf-8")


def test_compatibility_installer_copies_only_the_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hermes_provider_venice_cli as cli

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    destination = cli.install()

    assert destination == tmp_path / "plugins" / "model-providers" / "venice"
    # The catalog installs this same directory, so it must hold no CLI or
    # updater code.
    assert sorted(path.name for path in destination.iterdir()) == [
        "README.md",
        "__init__.py",
        "plugin.yaml",
    ]


def test_update_upgrades_via_pip_from_pypi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hermes_provider_venice_cli as cli

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls: list[list[str]] = []

    def fake_run(command, check):
        calls.append(command)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    cli.update()

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
    import hermes_provider_venice_cli as cli

    calls: list[list[str]] = []
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command, check: calls.append(command),
    )

    cli.update(pre=True, source="git+https://example.com/pkg.git@main")

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
    import hermes_provider_venice_cli as cli

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cli.install()

    monkeypatch.setattr(cli.subprocess, "run", lambda command, check: None)

    cli.update()

    destination = tmp_path / "plugins" / "model-providers" / "venice"
    assert (destination / "__init__.py").is_file()


def test_update_reports_pip_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import hermes_provider_venice_cli as cli

    def fake_run(command, check):
        raise cli.subprocess.CalledProcessError(returncode=2, cmd=command)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="pip failed to upgrade"):
        cli.update()
