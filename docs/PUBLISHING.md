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
