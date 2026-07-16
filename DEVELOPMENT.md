# Local development

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A local Hermes Agent checkout for integration testing
- A Venice API key for optional live tests

## Set up

Install the package and development tools:

```bash
uv sync
```

Run the local checks:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Automated tests mock network access and do not require an API key.

## Test with a local Hermes checkout

Set the path to Hermes Agent and create an isolated Hermes home so development
does not alter your normal configuration:

```bash
export HERMES_REPO=../hermes-agent
export TEST_HERMES_HOME="$(mktemp -d)"
trap 'rm -rf "$TEST_HERMES_HOME"' EXIT

HERMES_HOME="$TEST_HERMES_HOME" \
  uv run hermes-provider-venice install

HERMES_HOME="$TEST_HERMES_HOME" \
  uv run --project "$HERMES_REPO" python - <<'PY'
from providers import get_provider_profile

profile = get_provider_profile("venice")
assert profile is not None
print(profile.name, profile.base_url)
PY
```

This exercises Hermes's directory-plugin discovery without installing the
package into Hermes's Python environment.

## Run a live request

Read your key without echoing it or placing it in shell history:

```bash
read -rsp "Venice API key: " VENICE_API_KEY && echo
export VENICE_API_KEY

HERMES_HOME="$TEST_HERMES_HOME" \
VENICE_API_KEY="$VENICE_API_KEY" \
  uv run --project "$HERMES_REPO" hermes \
  -z "Reply with exactly: VENICE_LIVE_OK" \
  --provider venice \
  -m zai-org-glm-5-2
```

The expected response is `VENICE_LIVE_OK`.

## Test native package discovery

Hermes releases that support packaged model providers must have this package
installed in the same Python environment. With a Hermes development virtual
environment already created:

```bash
uv pip install \
  --python "$HERMES_REPO/.venv/bin/python" \
  --editable .

"$HERMES_REPO/.venv/bin/python" - <<'PY'
from providers import get_provider_profile

assert get_provider_profile("venice") is not None
print("native package discovery passed")
PY
```

Use the isolated-directory method above when testing Hermes versions that do
not yet support the dedicated `hermes_agent.model_providers` entry point.

## Build locally

```bash
uv build --clear
```

The wheel and source distribution are written to `dist/`. Release publishing
is documented separately in [`docs/PUBLISHING.md`](docs/PUBLISHING.md).
