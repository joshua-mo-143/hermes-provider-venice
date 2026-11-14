# Publishing to PyPI

1. Choose a new version and update it in:
   - `pyproject.toml`
   - `src/hermes_provider_venice/__init__.py`
   - `src/hermes_provider_venice/plugin.yaml`

2. Refresh the lockfile and run the checks:

   ```bash
   uv lock
   uv run pytest -q
   uv run ruff check .
   uv run ruff format --check .
   ```

3. Commit the reviewed release changes and confirm the release is reproducible
   from a clean working tree:

   ```bash
   git diff --check
   git status --short
   ```

   `git status --short` should produce no output. Do not publish from an
   uncommitted or partially staged tree.

4. Build fresh distribution artifacts:

   ```bash
   uv build --clear
   ```

5. Verify the upload without publishing:

   ```bash
   uv publish --dry-run --trusted-publishing never
   ```

6. Set a PyPI token and publish:

   ```bash
   read -rsp "PyPI token: " UV_PUBLISH_TOKEN && echo
   export UV_PUBLISH_TOKEN
   uv publish
   ```

PyPI does not allow replacing an existing release. If publishing fails because
the version already exists, increment the version, rebuild, and publish again.
Never commit or print the PyPI token.

# Submitting to the Hermes plugin catalog

The [Hermes plugin catalog](https://hermes-agent.nousresearch.com/docs/developer-guide/plugins/catalog-submission)
installs the plugin from GitHub at an exact commit. Catalog users get a release
only after a reviewed pull request to `NousResearch/hermes-agent` pins it.

1. Merge the release to `main` and copy its full 40-character commit SHA.
   Never rewrite history past a pinned commit: the catalog delists entries whose
   commit disappears.

2. Run the catalog check from [`DEVELOPMENT.md`](../DEVELOPMENT.md) on that
   commit and fix every failure and warning.

3. Open a pull request that adds `plugin-catalog/venice.yaml`. Leave `tier` as
   `community`; Nous maintainers set `verified` for plugins published by the
   vendor itself. Keep `requires_hermes` and `version` equal to `plugin.yaml`.

   ```yaml
   name: venice
   title: Venice AI
   repo: https://github.com/<owner>/hermes-provider-venice
   sha: <full commit SHA>
   subdir: src/hermes_provider_venice
   description: "Venice AI — privacy-first, OpenAI-compatible inference. Disclosure — sends your prompts, tool definitions, tool results and VENICE_API_KEY to api.venice.ai (or VENICE_BASE_URL); Venice passes requests for models its catalog marks anonymized, such as Claude, GPT and Gemini, on to the company that serves them. The plugin downloads Venice's model catalog with your key, sometimes on a short-lived background thread, keeps it in memory only, and reads or writes no files."
   maintainer: <owner>
   tier: community
   category: models
   requires_hermes: ">=0.22.6"
   docs_url: https://github.com/<owner>/hermes-provider-venice#readme
   version: "<version>"
   capabilities:
     provides_tools: []
     provides_hooks: []
     provides_middleware: []
     requires_env:
       - VENICE_API_KEY
   ```

4. In the pull request description, say what the plugin does, that it uses only
   `ProviderProfile` hooks (`build_extra_body`, `build_api_kwargs_extras`,
   `fetch_models`, `live_context_length`, `get_usage_cost`,
   `invalidate_catalog_cache` and the `model_capabilities` field), and repeat the
   disclosure.

To ship a later release to catalog users, open a new pull request that changes
`sha` and `version` together.
