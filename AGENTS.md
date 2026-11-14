# Repository Guidance

## Local Development

See `DEVELOPMENT.md` for instructions on setting up.

## Python

- Target Python 3.12 or newer and use modern type annotations.
- Add `from __future__ import annotations` to Python modules.
- Follow PEP 8 and keep code formatted and lint-clean with Ruff.
- Prefer clear, typed functions, `pathlib.Path`, and standard-library solutions.
- Keep imports at module scope unless a lazy import avoids an optional host
dependency or an import cycle.
- Catch specific exceptions. Broad catches are acceptable only at plugin or
network boundaries where failure must degrade safely.
- Keep the package importable outside Hermes so metadata and the compatibility
installer continue to work.
- Avoid new runtime dependencies unless the standard library is insufficient.
- Add or update pytest tests for behavioral changes. Mock network access in
tests; live API checks must be explicit and must never expose credentials.
- Run `uv run pytest -q`, `uv run ruff check .`, and
`uv run ruff format --check .` before handing off changes.

## Hermes integration

The plugin is built to be listed in the Hermes plugin catalog.
`hermes plugins validate` enforces some of these rules (see `DEVELOPMENT.md`)
and catalog reviewers check the rest.

- Extend Hermes only through documented `ProviderProfile` hooks and fields.
  Never assign to, wrap or replace anything in a Hermes module.
- Import from Hermes only what its public plugin API lists
  (`hermes_cli/data/plugin_api.json` in Hermes), lazily, inside the function
  that needs it.
- Get the API key from the hook argument Hermes passes or from
  `agent.secret_scope.get_secret`; never read `os.environ`.
- Hooks that run for every request (`build_extra_body`,
  `build_api_kwargs_extras`, `get_usage_cost`) must not wait on the network.
  Read the catalog cache and refresh it in the background.
- Keep `src/hermes_provider_venice/` free of CLI, installer and update code; it
  is the directory the catalog installs.
- Disclose any new network call, file access, background work or credential
  use in both READMEs.

## Publishing

Follow [`docs/PUBLISHING.md`](docs/PUBLISHING.md) for every release.

- Use `uv build` and `uv publish`; do not use Twine to publish.
- Keep the version synchronized across `pyproject.toml`, the Python package,
and `plugin.yaml`.
- Run all checks and a dry-run before uploading.
- Never publish unless the user explicitly requests it.
- Never store or expose PyPI credentials.

