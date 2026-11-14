# Venice AI for Hermes Agent

Adds [Venice AI](https://venice.ai) as a model provider in [Hermes Agent](https://github.com/NousResearch/hermes-agent). Venice has a privacy-first, OpenAI-compatible inference API, and this plugin registers a `venice` provider that sends chat requests to `https://api.venice.ai/api/v1`.

> **Disclaimer:** This is an independent, open source project. It is not officially maintained, endorsed, or supported by Venice AI.

Requires Hermes Agent 0.22.6 or newer.

## Install

```bash
hermes plugins install venice
```

If `VENICE_API_KEY` isn't set, Hermes asks for it during the install. Create a key at [venice.ai/settings/api](https://venice.ai/settings/api). To set it yourself, add it to `~/.hermes/.env`:

```bash
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

To send requests through another endpoint, such as a proxy, set `VENICE_BASE_URL`.

## What it does

- Lists Venice's online, tool-capable text models in the model picker, with a curated fallback list when the catalog can't be fetched.
- Reads each model's context window, token prices, and tool, vision and reasoning support from Venice's live model catalog.
- Estimates the cost of each response from Venice's per-model token prices.
- Maps Hermes's reasoning settings onto Venice's. Turning reasoning off sends `reasoning.enabled: false`. An effort level is moved to the nearest level the model supports, and left out for models with a fixed reasoning budget.
- Turns off Venice's default system prompt, because Hermes sends its own.
- Sends the Hermes session id as `prompt_cache_key`, so a conversation's requests share a prompt cache.

## Privacy and network access

- Hermes sends your prompts, tool definitions, tool results and `VENICE_API_KEY` to `api.venice.ai`, or to `VENICE_BASE_URL` if you set it.
- Venice's catalog marks each model as `private` or `anonymized`. Venice passes requests for anonymized models, such as the Claude, GPT and Gemini models, on to the company that serves them. See the [Venice documentation](https://docs.venice.ai) for details.
- The plugin itself downloads Venice's model catalog (`GET /models?type=text`) with your key when Hermes lists models or looks up a context window. It also downloads it when Hermes sends a request and the cached copy is missing or more than five minutes old; that download runs on a short-lived background thread so the request doesn't wait.
- The catalog is kept in memory only. The key is sent only over HTTPS or to a loopback address, and catalog requests never follow redirects.
- The plugin collects no telemetry and doesn't read or write files.

## Uninstall

```bash
hermes plugins remove venice
```
