# Contributing

Contributions that improve Venice compatibility, Hermes integration,
documentation, or test coverage are welcome.

## Set up the project

This project uses Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
Clone your fork, then install the project and development dependencies:

```bash
cd hermes-provider-venice
uv sync
```

See [`DEVELOPMENT.md`](DEVELOPMENT.md) for local Hermes integration and live
testing instructions.

Create a focused branch and keep each change small enough to review easily.
Do not include API keys, `.env` files, generated distributions, or unrelated
formatting changes.

## Make and test changes

Follow the Python conventions in [`AGENTS.md`](AGENTS.md). Add tests for new
behavior and avoid real network requests in the automated test suite.

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv build --clear
```

For optional end-to-end testing, install the provider into a temporary
`HERMES_HOME` and use your own `VENICE_API_KEY`. Never put credentials in test
output, commits, issues, or pull requests.

## Submit a pull request

Explain what changed, why it is needed, and how it was tested. Update the
README or other documentation when user-facing behavior changes. Do not bump
the package version or publish a release unless a maintainer asks you to.

Maintainers should follow [`docs/PUBLISHING.md`](docs/PUBLISHING.md) when
creating a release.
