# Configuration

By default, Lofi reads `config.toml` from `$XDG_CONFIG_HOME/lofi/` (`~/.config/lofi/config.toml`), and reads shell approval rules from `policy.toml` in the same directory. To point Lofi at custom file paths instead, set `$LOFI_CONFIG` or `$LOFI_POLICY` in the environment.

Runtime state such as session transcripts, model-discovery caches, workspace memory, and external directory grants lives separately under `$XDG_STATE_HOME/lofi/` (`~/.local/state/lofi/` by default). You can relocate that state root by setting `$LOFI_STATE_HOME`, which places state under `$LOFI_STATE_HOME/lofi/`. Lofi creates each state subdirectory lazily when a session first writes to it, while leaving the configuration directory untouched.

## Quick start

You can run Lofi against a direct HTTP model provider, an external CLI agent backend, or both at once. HTTP provider models run inside the native Lofi agent loop. CLI backend models delegate turn generation to an external CLI while Lofi supplies the `exec` tool, transcript, compaction, and policy. Models from both sources appear side by side in `lofi models` and in the `/model` picker.

### Model provider

To configure an HTTP provider, declare a `[providers.<name>]` table and its models in `config.toml`, then export the API key variable in your shell. Unless you override `base_url`, Lofi derives the host and endpoint path from `api_type`:

```toml
default_model = "openai/gpt-4o"

[providers.openai]
api_type = "openai-completions"
env_name = "OPENAI_API_KEY"

[providers.openai.models]
gpt-4o = { name = "GPT-4o", supports_image = true, context_window = 128000 }
```

If `config.toml` does not exist on disk, Lofi falls back to built-in definitions for OpenAI Responses, Anthropic Messages, and Google Generative AI. Exporting `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY` activates the corresponding built-in provider. Once you create a `config.toml` file, its `[providers]` entries replace that built-in catalog.

### Claude Code

To use Claude Code as a backend, install the `claude` CLI and sign in with `claude auth login`, then add a `[backends.<name>]` table to `config.toml`. At runtime, Claude Code uses its own stored login while Lofi manages the transcript, context compaction, MCP `exec` bridge, and `policy.toml` approval rules:

```toml
default_model = "claude/sonnet"

[backends.claude]
type = "claude-code"
model = "sonnet"
models = ["sonnet", "opus", "haiku"]
```

### Run Lofi

Once a provider or backend is configured, list resolved models with `lofi models`, start the TUI with `lofi`, or run `--print`. On any invocation, `--model` selects a qualified model directly, and `--backend` selects a configured CLI backend by name:

```sh
lofi models
lofi
lofi --backend claude --print "Describe this project"
lofi --model claude/sonnet --print "Describe this project"
```

## Model selection

Every model has a qualified name in the form `<provider-or-backend>/<model-id>`, such as `openai/gpt-4o` or `claude/sonnet`. To set a startup default across sessions, place `default_model` or `default_provider` at the top of `config.toml` before any `[table]` header:

```toml
default_model = "claude/sonnet"
```

| Key | Type | Description |
| --- | --- | --- |
| `default_model` | string | Qualified model selected at startup. Overrides `default_provider`. |
| `default_provider` | string | Provider or backend whose first available model is selected at startup. |

For a single run, `--model <name>/<id>` or `--backend <name>` overrides `config.toml` from the command line, and `/model` switches models live in the TUI. In both views, configured CLI backend models are always listed, whereas HTTP provider models appear only when credentials resolve or `no_auth = true` is set. Without a startup default, Lofi picks the first available HTTP provider model in file order. Otherwise, it falls back to the first configured backend model (`model`, then `models`, or `default` when empty).

When you reopen a transcript with `--continue` or `--resume <ID>`, Lofi restores the saved model and tier unless `--model` or `--backend` overrides it. Switching between an HTTP provider and a CLI backend mid-session preserves conversation history, active jobs, subagents, directory grants, and shell policy. Model specs also accept optional thinking and service-tier suffixes formatted as `provider/id[:level][@tier]` (see [Thinking levels](#thinking-levels) and [Service tiers](#service-tiers)).

## Agent backends

Each `[backends.<name>]` table registers an external CLI agent and the model aliases it exposes. Currently, `claude-code` is the supported backend `type`. Switching models within a backend takes effect on the next turn without resetting the session, and aliases such as `sonnet` persist on resume.

### Backend fields

If the `claude` binary is not on `PATH`, or if you want to pin a specific binary, set `command` on the backend table. Alongside `command`, `context_window` sets the initial token budget before the child reports its own window. `frame_idle_timeout_ms` then bounds silence between streamed frames without capping total turn length.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | string | required | Backend driver name (`claude-code`). |
| `command` | string | `claude` | Executable name on `PATH` or path to the binary. |
| `model` | string | none | Default model alias for this backend, also added to the model picker. |
| `models` | string array | empty | Additional model aliases exposed in `lofi models` and `/model`. |
| `context_window` | integer | `200000` | Initial context window in tokens until the child reports its own window. |
| `frame_idle_timeout_ms` | positive integer | `7200000` | Maximum idle gap between child frames in milliseconds before Lofi stops the child and fails the turn. |
| `thinking_level` | string | none | Default thinking effort for this backend when no `:level` suffix is given. Overrides `[agent].thinking_level`. |
| `auto_continue` | table | none | Per-backend overrides for `[agent.auto_continue]` (`intent`, `lost_tool_call`, `byte_cap`). |

```toml
[backends.claude]
type = "claude-code"
command = "/absolute/path/to/claude"
models = ["sonnet", "opus", "haiku"]
```

### Claude Code sessions

Lofi spawns Claude Code in streaming print mode and connects it to `lofi mcp serve`. To keep tool calls inside Lofi, the child runs without built-in Claude tools, slash commands, hooks, Chrome integration, or local Claude configuration. Enterprise managed Claude policies still apply to that child process. At startup, the child must send an `init` frame within 30 seconds that lists only the `exec` tool and the Lofi MCP server. If that frame reports any other tool or server, Lofi aborts the turn with an error naming it. Once `init` succeeds, the turn runs without a total duration limit as long as frames arrive within `frame_idle_timeout_ms`.

In this architecture, Claude Code acts as a turn transport rather than a separate session owner. Lofi records the authoritative transcript and reuses the external Claude session identifier whenever its history matches the active branch. Any unseen items such as `!` shell output travel on the next turn as deltas. If a turn fails or is canceled, Lofi stops the child process group and reopens the CLI session on the next turn from `<session>.backend.json`. In contrast, running `/compact` or changing the system prompt rewrites model context, so Lofi starts a fresh Claude session seeded from the compacted history.

Context compaction also stays under Lofi control on this backend. Lofi passes `--autocompact 1000000` so Claude Code never compacts on its own. Instead, Lofi stops a turn with `ContextPressure` when usage reaches `context_window - reserved_context_tokens`. If you pass `--no-session`, Lofi skips writing its own transcript while Claude Code still writes its local session file.

### Model labels, spend, and subagents

In the TUI header, Lofi displays the model label reported by Claude Code at startup, such as `[1m]` suffixes. If that startup report omits a model label, Lofi takes it from the first response instead. Because Claude Code reports cumulative USD spend for the child process, Lofi stores each finished turn as the spend delta since the previous turn. Canceled turns report no final cost frame, so Lofi records zero USD delta alongside the token usage from the last completed round. For reasoning effort, `/thinking` and `:level` map `low`, `medium`, `high`, `xhigh`, and `max` onto `--effort`, while `off` omits the flag.

Shell approval in `[auto_mode]` works the same way on CLI backends and HTTP providers. Only an `allow` verdict runs silently, whereas asks and evaluator errors open the approval prompt. Likewise, `lofi.subagent` can spawn either a CLI backend child or an HTTP provider child regardless of what the parent session runs on. When targeting a CLI backend, a subagent `model` override must match the backend `model`, an entry in `models`, or `default`. Every child shares the parent job registry, shell policy, and `[truncate]` limits, while isolating its own QuickJS worker, auto-mode context, and mailbox inbox.

## Value resolution

Provider `api_key`, `base_url`, and `headers` strings support dynamic resolution through shell commands and environment variables. By contrast, `env_name` names an environment variable directly: if that variable is unset or empty, the provider stays inactive without raising an error. Explicit `api_key`, `base_url`, and `headers` values are strict, and an empty resolved result counts as unset. If a referenced variable is missing or a `!command` fails, startup fails with an error naming the provider and field.

| Form | Meaning |
| --- | --- |
| `!command` | Run `sh -c command` and use trimmed stdout. |
| `$VAR` or `${VAR}` | Expand an environment variable. |
| `$$` | Literal `$`. |
| `$!` | Literal `!`. |
| Any other string | Literal value as written. |

```toml
[providers.example]
api_type = "openai-completions"
api_key = "!pass show services/example/api-key"

[providers.example.headers]
x-organization = "$EXAMPLE_ORGANIZATION"
x-literal = "$$5"
```

### Credential command timeout

When a provider field uses `!command`, the `[credential]` table bounds how long that shell helper can run during startup. By default, `timeout_ms` allows `30000` milliseconds and caps captured stdout bytes. If the helper times out or exits non-zero, Lofi kills its process group and reports a configuration error:

```toml
[credential]
timeout_ms = 30000
```

### Environment overrides

For one-off runs, export `LOFI__SECTION__KEY=VALUE` to override any key in `config.toml` or `policy.toml` without editing files on disk. Double underscores step into nested TOML tables, keys map to lowercase, and values parse as TOML literals before falling back to strings:

```sh
LOFI__DEFAULT_MODEL=claude/opus lofi
LOFI__RETRY__MAX_RETRIES=0 lofi
```

You can also pass `-e` or `--env` on the command line to inject environment variables into both configuration resolution and `lofi.bash` child processes. Passing a bare `NAME` forwards the current value from the parent shell, whereas `NAME=VALUE` sets an explicit value for that process:

```sh
lofi -e EXAMPLE_API_KEY -e EXAMPLE_BASE_URL="https://api.example.com"
```

## UI theme

The `[ui]` table chooses the initial color theme (`auto`, `light`, or `dark`, defaulting to `auto`), and `/theme` switches it live during a session. When `theme = "auto"`, Lofi enables terminal mode 2031 notifications and queries CSI 997 and OSC 11 at startup and on focus gain. Between the two, CSI 997 replies take priority over OSC 11 luminance, and Lofi falls back to `dark` if the terminal answers neither query:

```toml
[ui]
theme = "auto"
```

Alongside light and dark detection, Lofi queries ANSI palette slots with OSC 4 to detect grayscale and color-cube direction for palettes like Ghostty `palette-harmonious`. It refreshes that palette probe on color-scheme events, focus gain, and `/theme` changes without overwriting terminal colors. Until the first full OSC 4 reply pair arrives, Lofi assumes standard xterm palette ordering.

## Thinking levels

For HTTP provider models, you can set a default `thinking_level` globally under `[agent]`, per provider, or per static model. Valid levels are `off`, `low`, `medium`, `high`, and `xhigh`, and every non-`off` level chosen for a model must appear in its `thinking_levels` list. At runtime, model defaults override provider defaults, provider defaults override `[agent]`, and a `:level` CLI suffix such as `--model openai/o3:high` overrides all three:

```toml
[agent]
thinking_level = "medium"

[providers.openai]
api_type = "openai-responses"
api_key = "$OPENAI_API_KEY"
thinking_level = "high"

[providers.openai.models]
o3 = { reasoning = true, thinking_level = "medium", thinking_levels = ["low", "medium", "high"] }
```

## Automatic turn continuation

In the native HTTP provider loop, `auto_continue` controls automatic recovery when a model stops mid-task. Lofi resolves these booleans field by field from built-in defaults, then `[agent.auto_continue]`, then `[providers.<name>.auto_continue]`, and finally `[providers.<name>.models.<id>.auto_continue]` (while `auto_models` entries inherit from their provider). Across all three triggers, Lofi allows one automatic continuation per user turn and stores the prompt as a `notice` message.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `lost_tool_call` | boolean | `true` | Recover when a provider reports a tool-use stop reason without emitting any structured tool call. |
| `intent` | boolean | `false` | Recover when a clean stop ends with a short paragraph announcing an immediate tool action, such as `"I will run the tests next."` Enable only for templates known to drop tool calls. |
| `byte_cap` | boolean | `true` | Continue once when a streamed round hits the transport reply byte budget. Setting `false` stops the turn after running any tool calls finished before the cut. |

```toml
[agent.auto_continue]
lost_tool_call = true
intent = false
byte_cap = true

[providers.example.auto_continue]
intent = true

[providers.example.models.broken-template.auto_continue]
lost_tool_call = false
```

### Reply byte cap recovery

To bound memory during streaming, the transport reply budget sums streamed text, thinking text, and tool input JSON plus a small per-event cost. Because the model never reads opaque signatures or redacted reasoning back, those blocks bypass the reply budget and share a separate 8 MiB ceiling. When a round reaches the reply budget, Lofi preserves completed blocks and earlier tool calls, and marks the interrupted tool call as cut off. Incomplete reasoning blocks are dropped unless an earlier signature names their replay field.

What happens next depends on what the cut interrupted. If `byte_cap = true` cut a plain reply, Lofi asks once for a briefer answer unless the kept prefix fills the context window. If the cut stopped a tool round instead, Lofi executes the finished tool calls and feeds their results into the next round automatically. A second byte-cap cut within the same user turn ends the turn.

## Service tiers

Service tiers select the request priority class sent to HTTP providers (`auto`, `flex`, `priority`, or custom provider strings). Selecting `auto` leaves the provider default untouched, whereas custom strings travel verbatim in the request body. Precedence runs from model, to provider, to `[agent]`, and `--model openai/o3:high@flex` overrides all three when listed in `service_tiers`:

```toml
[agent]
service_tier = "flex"

[providers.openai]
api_type = "openai-responses"
api_key = "$OPENAI_API_KEY"
service_tier = "priority"

[providers.openai.models]
o3 = { reasoning = true, thinking_levels = ["low", "medium", "high"], service_tier = "flex", service_tiers = ["flex", "priority"] }
```

## Providers

Each `[providers.<name>]` table defines authentication, HTTP protocol routing, stream timeouts, pricing rules, and available models. Four wire protocols serve interactive and print chat turns, while a fifth protocol, `systemone`, serves typed `[auto_mode]` decision queries. Because `systemone` is a decision endpoint rather than a chat transport, its models never appear in `lofi models` or `/model`.

### Provider fields

Most providers only need `api_type` and credentials, since other fields inherit sensible protocol defaults. For network resilience, `response_start_timeout_ms` bounds connection setup and response headers, while `body_idle_timeout_ms` bounds stalls between streamed chunks on both parent and subagent turns.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `api_type` | string | `openai-completions` | Default wire protocol for models on this provider. |
| `base_url` | string | protocol host | Host root URL to which Lofi appends the protocol endpoint path. |
| `api_key` | string | none | API key or value-resolution expression. Overrides `env_name`. |
| `env_name` | string | none | Environment variable holding the API key. |
| `headers` | table | none | Extra HTTP headers (values support value resolution). |
| `no_auth` | boolean | `false` | Allow requests without an API key or auth header. |
| `response_start_timeout_ms` | positive integer | `90000` | Maximum wait in milliseconds for connection and HTTP response headers. |
| `body_idle_timeout_ms` | positive integer | `300000` | Maximum wait in milliseconds between response body chunks before failing as a retryable timeout. |
| `thinking_level` | string | inherited | Provider-level default thinking effort. |
| `thinking_levels` | string array | empty | Provider capability metadata (declare per-model levels on static models). |
| `service_tier` | string | inherited | Provider-level default service tier. |
| `service_tiers` | string array | empty | Provider capability metadata (declare per-model tiers on static models). |
| `pricing_convention` | string | `per_token` | Unit for discovered prices: `per_token` or `per_million`. |
| `pricing_field_mappings` | table | standard `pricing.*` | Dot paths for discovered input, output, cache, and per-request prices. |
| `api_types` | table | empty | Per-protocol endpoint paths and pricing overrides on a multi-protocol host. |
| `auto_models` | table | none | Remote `/models` discovery configuration. |
| `auto_continue` | table | inherited | Provider-level `lost_tool_call`, `intent`, and `byte_cap` overrides. |

| `api_type` | Default host | Default path |
| --- | --- | --- |
| `openai-completions` | `https://api.openai.com` | `/v1/chat/completions` |
| `openai-responses` | `https://api.openai.com` | `/v1/responses` |
| `anthropic-messages` | `https://api.anthropic.com` | `/v1/messages` |
| `google-generative-ai` | `https://generativelanguage.googleapis.com` | `/v1beta` |
| `systemone` | `https://api.typesafe.ai` | `/v1/systemone` |

### Endpoint routing with `api_types`

On a provider table, `base_url` specifies a host root to which Lofi appends the protocol path, whereas a model-level `base_url` specifies a full endpoint URL. Single-protocol providers therefore need only `api_type` and `base_url`. When a single proxy host exposes multiple protocols under custom paths, add `[providers.<name>.api_types.<protocol>]` tables instead. Both static `api_type` overrides and discovered models route through those entries, and any entry that omits `path` uses the default protocol path:

```toml
[providers.proxy]
base_url = "https://proxy.example.com"
api_type = "openai-completions"
api_key = "$PROXY_API_KEY"

[providers.proxy.api_types."openai-completions"]
path = "/v1/chat/completions"

[providers.proxy.api_types."openai-responses"]
path = "/v1/responses"

[providers.proxy.api_types."anthropic-messages"]
path = "/v1/messages"

[providers.proxy.api_types."google-generative-ai"]
path = "/v1beta"

[providers.proxy.models]
chat = { api_type = "openai-completions" }
reasoner = { api_type = "openai-responses", reasoning = true }
claude = { api_type = "anthropic-messages" }
```

## Static models

Static model definitions live as keys under `[providers.<name>.models]`. You can use an empty inline table (`{}`) when default metadata is sufficient, and quote keys that contain dots or colons. In static model entries, all token rates are written in USD per million tokens:

```toml
[providers.openai.models]
"gpt-4o-mini" = {}

[providers.openai.models.gpt-4o]
name = "GPT-4o"
supports_image = true
context_window = 128000
max_tokens = 16384
input_price = 2.50
output_price = 10.00
```

| Field | Type | Description |
| --- | --- | --- |
| `name` | string | Display name (defaults to the model key). |
| `api_type` | string | Per-model wire protocol override. |
| `reasoning` | boolean | Whether the model produces reasoning blocks. |
| `supports_image` | boolean | Whether the model accepts image attachments. |
| `context_window` | integer | Context window size in tokens. |
| `max_tokens` | integer | Maximum output tokens per response. |
| `thinking_level` | string | Default thinking level for this model. |
| `thinking_levels` | string array | Thinking levels supported by this model. |
| `service_tier` | string | Default service tier for this model. |
| `service_tiers` | string array | Service tiers supported by this model. |
| `base_url` | string | Full endpoint URL override for this model. |
| `input_price` | number | USD per million input tokens. |
| `output_price` | number | USD per million output tokens. |
| `cache_read_price` | number | USD per million cache-read tokens (falls back to `input_price` when unset). |
| `cache_write_price` | number | USD per million cache-write tokens (falls back to `input_price` when unset). |
| `per_request_price` | number | Flat USD charge added per HTTP request. |
| `auto_continue` | table | Model-level `lost_tool_call`, `intent`, and `byte_cap` overrides. |

### OpenAI and local server examples

For OpenAI-compatible endpoints, choose `openai-completions` for Chat Completions servers and `openai-responses` for reasoning models on the Responses API. When connecting to a local server such as `llama.cpp` or `vLLM` that expects no `Authorization` header, set `no_auth = true` on that provider. Each of these provider blocks can coexist in the same `config.toml`:

```toml
[providers.openai]
api_type = "openai-completions"
api_key = "$OPENAI_API_KEY"

[providers.openai.models]
gpt-4o = { name = "GPT-4o", supports_image = true, context_window = 128000 }
"gpt-4o-mini" = {}

[providers.openai-reasoning]
api_type = "openai-responses"
api_key = "$OPENAI_API_KEY"

[providers.openai-reasoning.models]
o3 = { reasoning = true, thinking_levels = ["low", "medium", "high"] }

[providers.local]
base_url = "http://127.0.0.1:8000"
api_type = "openai-completions"
no_auth = true

[providers.local.models]
"local-model" = {}
```

### Anthropic and Google examples

Anthropic and Google transports handle provider-specific caching and reasoning metadata automatically. On `anthropic-messages`, Lofi places ephemeral prompt-cache breakpoints on tool definitions, the system prompt, and the latest user message without extra headers. On `google-generative-ai`, Lofi preserves Gemini thought signatures on assistant parts for same-model replay, and also keeps signatures returned under `extra_content.google` or `extra_content.vertex` on `openai-completions` proxies:

```toml
[providers.anthropic]
api_type = "anthropic-messages"
api_key = "$ANTHROPIC_API_KEY"

[providers.anthropic.models]
"claude-opus-4" = { name = "Claude Opus 4", reasoning = true, max_tokens = 4096 }

[providers.google]
api_type = "google-generative-ai"
api_key = "$GEMINI_API_KEY"

[providers.google.models."gemini-3.7-flash"]
reasoning = true
supports_image = true
thinking_levels = ["low", "medium", "high"]
```

## Remote model discovery

Enabling `[providers.<name>.auto_models]` fetches an OpenAI-style model list at startup and merges those entries with `[providers.<name>.models]`. Whenever an identifier appears in both sources, static fields take precedence while omitted fields keep their discovered values. Lofi caches each fetched catalog in `$XDG_STATE_HOME/lofi/discovery.json` and reuses that cache if a later network fetch fails. If you omit `models_url`, Lofi derives a `/models` endpoint from the default protocol path, such as `https://api.openai.com/v1/models`:

```toml
[providers.proxy]
base_url = "https://proxy.example.com"
api_type = "openai-completions"
api_key = "$PROXY_API_KEY"

[providers.proxy.auto_models]
enabled = true
path = "data"
auth = true
ttl_seconds = 300
api_type_field = "preferred_api"
thinking_levels = ["low", "medium", "high"]
thinking_level = "medium"

[providers.proxy.auto_models.api_type_mappings]
chat_completions = "openai-completions"
responses = "openai-responses"
messages = "anthropic-messages"

[providers.proxy.auto_models.field_mappings]
name = "name"
context_window = "context_length"
max_tokens = "top_provider.max_completion_tokens"
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | boolean | `false` | Fetch remote models at startup. |
| `models_url` | string | derived | Complete URL of the remote model list endpoint. |
| `path` | string | `data` | Dot path to the model array inside the JSON response. |
| `auth` | boolean | `true` | Send provider authentication headers on the discovery request. |
| `ttl_seconds` | integer | `300` | Cache freshness lifetime in seconds. |
| `api_type_field` | string | none | JSON field holding the remote protocol string, mapped via `api_type_mappings` (unmapped values fall back to provider `api_type`). |
| `thinking_level` | string | none | Default thinking level assigned to discovered models. |
| `thinking_levels` | string array | empty | Supported thinking levels assigned to discovered models. |
| `service_tier` | string | none | Default service tier assigned to discovered models. |
| `service_tiers` | string array | empty | Supported service tiers assigned to discovered models. |

### Discovery pricing

During discovery, Lofi reads token and request prices from each remote model object using dot-path mappings. Provider-level `pricing_field_mappings` apply across every protocol unless `[providers.<name>.api_types.<protocol>.pricing_field_mappings]` overrides them for a specific protocol. All five fields are optional, and the four `pricing.*` token paths below are the built-in defaults (with `per_request` unset):

```toml
[providers.proxy]
pricing_convention = "per_token" # per_token | per_million

[providers.proxy.pricing_field_mappings]
input = "pricing.prompt"
output = "pricing.completion"
cache_read = "pricing.input_cache_read"
cache_write = "pricing.input_cache_write"
per_request = "pricing.request"

[providers.proxy.api_types."openai-responses".pricing_field_mappings]
input = "responses_pricing.input"
output = "responses_pricing.output"
```

## Compaction

The `[compaction]` section governs context-window compaction identically across interactive sessions, `--print` runs, and subagents on both HTTP providers and CLI backends, and `/compact` triggers the same pass manually. On Claude Code, compacting rewrites the Lofi transcript and opens a new Claude child on the next turn so old context stays unused:

```toml
[compaction]
reserved_context_tokens = 20000
min_messages_between_hard_compacts = 6

[compaction.auto]
enable = true
max_context_tokens = 150000
context_ratio = 0.75

[compaction.edit]
enabled = true
keep_results = 6
keep_thinking = 2
keep_calls = 6
```

### Hard compaction

To leave room for output, `reserved_context_tokens` (default `20000`) reserves headroom at the top of `context_window`, clamped to 25 percent of the window. Lofi measures context fill as `input + output + cache_read + cache_write` tokens, which is the same total shown in the TUI footer gauge. When a round crosses `context_window - reserved_context_tokens`, Lofi pauses the turn, summarizes older messages, and continues the turn from the compacted context. To prevent loops when the retained tail alone fills the window, `min_messages_between_hard_compacts` (default `6`) requires six new messages before another hard compaction. If the hard limit triggers again sooner than that, Lofi fails the turn instead of compacting again.

### Soft compaction

Unlike hard compaction, `[compaction.auto]` waits until an active turn settles before testing whether context fill crossed a soft threshold. If both `max_context_tokens` and `context_ratio` are configured, the lower token threshold wins. When neither soft threshold is set, Lofi runs hard compaction only.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `enable` | boolean | `true` | Enable soft compaction when at least one soft threshold is set. |
| `max_context_tokens` | integer | none | Soft threshold in absolute tokens. |
| `context_ratio` | number | none | Soft threshold as a fraction of `context_window` (for example `0.75`). |

### Tiered tail retention

During compaction, `[compaction.edit]` trims heavy payloads inside the recent messages kept verbatim after the summary. Tool results and `exec` code blocks beyond the retention counts become event-id stubs that the model can reload with `lofi.result(eventId)`. Older thinking blocks are dropped outright, while assistant prose remains untouched.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | boolean | `true` | Trim older tool calls, tool results, and thinking blocks in the kept tail. |
| `keep_results` | integer | `6` | Number of recent tool-result blocks kept verbatim. |
| `keep_thinking` | integer | `2` | Number of recent thinking blocks kept verbatim. |
| `keep_calls` | integer | `6` | Number of recent `exec` code blocks kept verbatim. |

### Workspace memory

Every workspace shares a persistent notes file at `$XDG_STATE_HOME/lofi/states/<workspace-key>/memory.md` without needing any `config.toml` entry. On the first write, Lofi creates the parent directory (`0700`) and file (`0600`). From inside `exec`, `lofi.memory()` returns the current text and path so `lofi.write` or `lofi.edit` can update it without prompting. Whenever compaction runs, Lofi embeds `memory.md` in the summary handoff under `Workspace Memory` (clipped to 3000 characters, with a note when clipped). Because concurrent sessions share this file without locking, rewrite the full file concisely rather than appending indefinitely.

## Bash environment

The `[bash]` table controls which environment variables reach `lofi.bash` and `lofi.jobSpawn` child processes. Any variable forwarded through `pass_env` or `env_file` has its value replaced with `[redacted]` in captured command output, and `env_file` entries override matching keys from `pass_env`. Keep `env_file` outside the workspace root so file tools cannot read raw secret values directly:

```toml
[bash]
strip_env = true
pass_env = ["GITHUB_TOKEN"]
env_file = "~/.config/lofi/secrets.env"
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `strip_env` | boolean | `true` | Reset the child environment to a minimal baseline (`PATH`, `HOME`, locale, and `-e` flags). Setting `false` exposes the full parent environment. |
| `pass_env` | string array | empty | Parent environment variable names copied into the child and replaced with `[redacted]` in captured output. |
| `env_file` | path | none | `KEY=VALUE` file (`~` expanded) loaded into the child environment with values redacted from output. |

## Truncation and images

`[truncate]` sets the inline output ceiling shared by `lofi.read` (head-truncated with paging) and foreground `lofi.bash` (tail-truncated with a temporary log path). Alongside text caps, `[image]` bounds images read through `lofi.read`. Lofi downscales each image to fit `max_width` by `max_height` with aspect ratio intact, then lowers JPEG quality until it fits `max_bytes`. Those image blocks are sent only when the active model sets `supports_image = true`, and are omitted with a warning on text-only models:

```toml
[truncate]
max_lines = 2000
max_bytes = 51200

[image]
max_width = 2000
max_height = 2000
max_bytes = 1048576
```

| Table and field | Type | Default | Description |
| --- | --- | --- | --- |
| `truncate.max_lines` | integer | `2000` | Maximum lines returned inline by `lofi.read` or `lofi.bash`. |
| `truncate.max_bytes` | integer | `51200` | Maximum bytes (50 KiB) returned inline before truncating. |
| `image.max_width` | integer | `2000` | Maximum downscaled image width in pixels. |
| `image.max_height` | integer | `2000` | Maximum downscaled image height in pixels. |
| `image.max_bytes` | integer | `1048576` | Maximum re-encoded JPEG payload size in bytes (1 MiB). |

## Retries

The `[retry]` table configures exponential-backoff retries for native HTTP provider turns, whereas disconnected CLI backend turns stop and resume on the next turn. Under this policy, Lofi retries HTTP 429, HTTP 5xx, overload events, network drops, premature EOF, and `body_idle_timeout_ms` stalls. In contrast, authentication failures, billing or quota rejections, and context overflows fail immediately without retrying:

```toml
[retry]
max_retries = 10
base_delay_ms = 2000
max_delay_ms = 60000
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `max_retries` | integer | `10` | Maximum retry attempts after the initial request (`0` disables retries). |
| `base_delay_ms` | integer | `2000` | Initial backoff delay in milliseconds before retry one. |
| `max_delay_ms` | integer | `60000` | Maximum backoff delay ceiling in milliseconds per retry. |

## Workspace directories

Built-in file tools (`read`, `ls`, `find`, `grep`, `write`, `edit`, and `patch`) resolve relative paths inside the workspace root and reject `..` escapes. For paths outside the workspace, read-only tools automatically allow built-in roots such as `lofi.tmpDir` and skill directories. These directory grants govern file tools only, whereas `lofi.bash` runs on the host under `policy.toml`.

When a file tool requests an unapproved external path, Lofi opens one grant prompt. That prompt offers either the enclosing project root (detected from `.git`, `.jj`, `Cargo.toml`, or `package.json` below `$HOME`) or the requested directory alone. Read operations ask for `read` access, write operations ask for `write` access, and a later write inside a `read` directory asks to widen that grant. Accepted grants are saved per workspace in `$XDG_STATE_HOME/lofi/states/<workspace-key>/state.json`, while declined requests remain unapproved and prompt again if retried.

### Allowed paths

To pre-approve external directories across every workspace, add `[[permissions]]` entries to `config.toml`. In each entry, `access` defaults to `read` so omitting the field never grants write access by mistake. `~` expands to `$HOME`, and paths that do not exist yet resolve through their deepest existing ancestor. Multiple grants combine additively so `write` wins over `read`, and `/paths` views all roots or toggles (`Space`) and deletes (`d`) workspace grants:

```toml
[[permissions]]
path = "/nix/store"
access = "read"

[[permissions]]
path = "~/notes"
access = "write"
```

## Shell policy

Shell approval rules live in `policy.toml` alongside `config.toml`. Before running any `lofi.bash` or `lofi.jobSpawn` command, Lofi parses the syntax into pipelines, subshells, wrappers, redirects, and heredocs. It then matches each segment against `deny`, `ask`, and `allow` rules. Unmatched commands fail closed to `ask`, and because `sh -c` runs unsandboxed on the host, the default policy asks for user approval on every command.

If you set `yolo = true`, Lofi skips prompts for `ask` and unmatched commands while still enforcing explicit `deny` rules. Conversely, a `policy.toml` that still sets the removed `mode` key fails to load at startup so older preset files never drop approvals silently. To test how your rules treat a command without running it, execute `lofi policy explain <COMMAND>`.

### Custom rules

Custom `[[allow]]`, `[[ask]]`, and `[[deny]]` tables layer on top of Lofi built-in denials. Matching a custom `[[ask]]` rule reserves that decision for the user, so `[auto_mode]` never auto-approves it. Both `lofi policy explain` and the TUI approval prompt display the matched rule. Other asks raised by redirects, heredocs, unparseable syntax, or unmatched commands still pass through `[auto_mode]` when automatic approval is enabled:

```toml
[[allow]]
match = "my-tool"
mode = "prefix"

[[ask]]
match = "git commit"
mode = "prefix"

[[deny]]
match = "curl"
mode = "prefix"
```

| `mode` | Behavior |
| --- | --- |
| `exact` | Match the entire normalized command string. |
| `prefix` | Match the start of the command on a token boundary. |
| `substring` | Match a contiguous token sequence anywhere in the command. |
| `args` | Match a command prefix plus required argument tokens anywhere after it. |

### Command wrappers

Before matching command rules, the parser unwraps inner commands passed through launchers such as `sh -c`, `sudo`, `env`, `xargs`, `docker`, and `podman`. When your project uses a custom wrapper script that forwards another command in its arguments, register it under `[[wrappers]]` to extend that built-in list:

```toml
[[wrappers]]
name = "project-shell"
kind = "shell_c"
```

| `kind` | Extracted inner command |
| --- | --- |
| `shell_c` | Command string argument immediately following `-c`. |
| `utility_operand` | First non-option operand and all trailing arguments. |
| `env` | Command following flags and `KEY=VALUE` assignments. |
| `xargs` | Target utility operand and its trailing arguments. |
| `docker_run` | Container command following `run`, `exec`, or `create` (Docker and Podman style). |

### Redirects and heredocs

Output redirects and heredocs are evaluated separately from command rules and default to `ask`. Redirects targeting paths in `safe_targets` skip the redirect action, while `allow_fd_dup = true` permits descriptor duplications such as `2>&1`. Shell backgrounding primitives (`&`, `nohup`, `setsid`, and `disown`) are always denied so background work goes through `lofi.jobSpawn` instead:

```toml
[redirects]
action = "ask" # allow | ask | deny
safe_targets = ["/dev/null"]
allow_fd_dup = false

[heredocs]
action = "ask" # allow | ask | deny
```

### Automatic approval

Enabling `[auto_mode]` in `policy.toml` sends commands that would normally prompt (other than custom `[[ask]]` matches) to an evaluator model first. That evaluator never denies a command outright. Instead, `low` runs immediately, `confirm` opens the approval prompt, and `block` opens the prompt with the evaluator warning attached. Likewise, any network error, timeout, or malformed evaluator response falls back safely to the approval prompt:

```toml
[auto_mode]
enable = true
model = "openai/gpt-4o-mini"
# decision_model = "plexus/clef-flash"
# max_tokens = 1024
# timeout_ms = 30000
# approval_timeout_ms = 120000

# [auto_mode.decision]
# known = 0.30
# destructive = 0.50
# exposure = 0.50
# authorized = 0.75
# wide_scope_mass = 0.40
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enable` | boolean | `false` | Enable automatic shell approval evaluation. |
| `model` | string | required unless `decision_model` is set | Qualified chat model or CLI backend model query (`provider/id` or legacy `provider` + `model`). |
| `decision_model` | string | none | Qualified `systemone` decision model query (`api_type = "systemone"`). |
| `max_tokens` | integer | model default | Output token limit for the chat evaluator stage. |
| `timeout_ms` | positive integer | `30000` | Per-stage evaluation deadline in milliseconds before falling back to the user prompt. |
| `approval_timeout_ms` | integer | `120000` | Milliseconds before an unanswered escalated prompt or custom `[[ask]]` prompt auto-denies (`0` disables). |
| `decision.known` | number | calibrated | Minimum probability that the decision model recognizes the command utilities. |
| `decision.destructive` | number | calibrated | Maximum destructiveness score allowed by the decision gate. |
| `decision.exposure` | number | calibrated | Maximum secret-exposure or remote-code risk allowed by the decision gate. |
| `decision.authorized` | number | calibrated | Minimum user-authorization probability required when scope requires authorization. |
| `decision.wide_scope_mass` | number | calibrated | Maximum wide-scope probability mass allowed without explicit user authorization. |

#### Scope and risk rules

Under the evaluation rubric, local reads (even outside the workspace), public network reads (`GET`, `HEAD`, and GraphQL queries), and read-only `ssh` calls qualify as `low`. Testing whether a secret environment variable is set also stays `low`, provided the command never prints, compares, or traces its value. File deletes and overwrites qualify as `low` only when every resolved target stays inside a temporary directory, which excludes cache directories. If a symlink or redirect leaves the temporary directories, Lofi reverts the command to the approval prompt. Downloads saved to plain data files in the workspace or temporary directories stay `low` as well. By contrast, privilege escalation, writes elsewhere outside the workspace, uploads, and reading credential files always prompt.

#### Trusted and untrusted context

To judge intent, the evaluator receives the command, workspace root, current user request, and `exec` call `description`. Do not construct `-p` or `--print` prompts from untrusted issue or pull-request text in CI. Lofi treats that current user request as trusted authorization for remote writes such as pushing or commenting.

When that request has three words or fewer (such as `"yes"`), the evaluator also reads the prior request and last assistant message from live history. This lets a short `"yes"` authorize one explicit action and target proposed by the assistant. All other context, including earlier turns, assistant prose, `exec` descriptions, and subagent task strings, is untrusted and can only raise risk. Subagents never inherit parent conversation context, except for the root user request captured at spawn time (which clears if the process restarts).

#### Decision gate, two-stage evaluation, and calibration

In `[auto_mode]`, `decision_model` must point to a `systemone` provider (`api_type = "systemone"`), while `model` must point to a chat provider or CLI backend. Swapping the two models or omitting credentials fails startup immediately. Rather than generating prose, the decision gate scores scope, destructiveness, secret or remote-code exposure, user authorization, and utility familiarity (`known`). Commands whose CLI utility or subcommand falls below `known` escalate instead of passing on a guessed scope. When both `decision_model` and `model` are set, commands allowed by `decision_model` run right away. Asks or errors from `decision_model` pass to the chat `model` rubric and show reasons from both stages if still unapproved.

Each decision-gate evaluation appends a JSON line to `$XDG_STATE_HOME/lofi/logs/<session>/auto-mode.jsonl`. Before tuning `[auto_mode.decision]` thresholds, replay saved session logs or probe and grid-sweep labeled `allow: <cmd>` and `ask: <cmd>` corpora with `lofi policy calibrate`:

```sh
lofi policy calibrate
lofi policy calibrate --probe corpus.txt --out records.jsonl
lofi policy calibrate --records records.jsonl --sweep
```

#### Evaluation and prompt timeouts

During evaluation, `timeout_ms` caps each stage independently, covers rate-limit retries on `decision_model`, and honors provider `response_start_timeout_ms` on streaming chat evaluators. Once `[auto_mode]` escalates a command or a custom `[[ask]]` rule matches, `approval_timeout_ms` starts counting immediately, even while queued behind an earlier dialog. If left unanswered, the prompt auto-denies with `approval timed out`, whereas manual-mode asks and directory grant prompts wait without this timer. However, if the enclosing `exec` call sets `timeoutMs`, that deadline closes every pending prompt early (even directory grants and when `approval_timeout_ms = 0`). When `timeoutMs` expires, the tool fails with `the call's deadline passed before approval` and ignores any late user response.
