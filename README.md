# hermes-provider-venice

[Venice AI](https://docs.venice.ai) model provider plugin for
[Hermes Agent](https://github.com/NousResearch/hermes-agent).

Venice has a privacy-first OpenAI-compatible inference API. This package
registers a `venice` provider so Hermes can route chat through
`https://api.venice.ai/api/v1`.

## Install

```bash
# From PyPI:
pip install hermes-provider-venice

# From the latest GitHub main branch:
pip install "git+https://github.com/joshua-mo/hermes-provider-venice.git@main"

# Or from this checkout:
pip install .
```

For a reproducible GitHub installation, replace `main` with a release tag.

For Hermes releases without native packaged-provider discovery, or when the
package is installed outside Hermes's environment, run the compatibility
installer:

```bash
hermes-provider-venice install
```

The package exposes both the current `hermes_agent.plugins` entry point and the
dedicated `hermes_agent.model_providers` entry point. Hermes releases with
native packaged-provider discovery load it automatically when it is installed
in the same Python environment as Hermes. The compatibility command copies the
same profile to `$HERMES_HOME/plugins/model-providers/venice`.

Add your API key (create one at [https://venice.ai/settings/api](https://venice.ai/settings/api)):

```bash
# ~/.hermes/.env
VENICE_API_KEY=your-key-here
```

## Use

```bash
hermes -z "hello" --provider venice -m zai-org-glm-5-2
```

Or in `~/.hermes/config.yaml`:

```yaml
model:
  provider: venice
  default: zai-org-glm-5-2
```

The live picker fetches `GET /models?type=text` and filters out offline,
non-text, and non-tool-capable models. If that fails, these curated fallbacks
are used (refreshed 2026-07-16 from the Venice catalog):

| Role                 | Model ID                                                 |
| -------------------- | -------------------------------------------------------- |
| Recommended default  | `zai-org-glm-5-2`                                        |
| GLM 5                | `zai-org-glm-5`                                          |
| Grok                 | `grok-4-5`                                               |
| Qwen 3.7             | `qwen-3-7-max`, `qwen-3-7-plus`                          |
| DeepSeek V4          | `deepseek-v4-pro`, `deepseek-v4-flash`                   |
| GPT-5.6 (anonymized) | `openai-gpt-56-luna`, `openai-gpt-56-terra`              |
| Claude (anonymized)  | `claude-opus-4-8`, `claude-sonnet-5`                     |
| Gemini               | `gemini-3-5-flash`                                       |
| Uncensored           | `venice-uncensored-1-2`                                  |
| Code                 | `qwen3-coder-480b-a35b-instruct-turbo`, `kimi-k2-7-code` |
| Vision               | `qwen3-vl-235b-a22b`                                     |
| Fast / aux           | `deepseek-v4-flash`                                      |

Full catalog: `https://api.venice.ai/api/v1/models?type=text` or
[https://docs.venice.ai](https://docs.venice.ai)

## What the profile does

- `auth_type: api_key` via `VENICE_API_KEY`
- Optional `VENICE_BASE_URL` override
- Sets `venice_parameters.include_venice_system_prompt: false` so Hermes's
  system prompt is not appended to Venice's defaults
- Passes `prompt_cache_key` from the Hermes session id for better cache
  locality
- Filters the live catalog to online, text, tool-capable models
- Maps Hermes reasoning controls to Venice's recommended disable and
  model-supported effort parameters, clamping unsupported effort levels

## Develop

See [`DEVELOPMENT.md`](DEVELOPMENT.md) for setup, checks, local Hermes
integration, and live API testing.

Contributions are described in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Uninstall

```bash
rm -rf ~/.hermes/plugins/model-providers/venice
# optionally: uv tool uninstall hermes-provider-venice
```
