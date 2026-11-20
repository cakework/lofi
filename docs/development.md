# Development

Lofi is organized as a seven-crate Rust workspace with a strict one-way dependency graph. During development, run `just check` to format the code, run Clippy with warnings denied, and execute all unit and end-to-end tests. When you only want to run the process-level binary tests, run `just e2e` instead (see [End-to-end testing](e2e.md)).

## Workspace architecture

At the base of the workspace, shared data structures and error types live in leaf crates so higher layers never form dependency cycles. Above that base, HTTP model transports (`lofi-providers`) and the TypeScript tool sandbox (`lofi-code`) stay independent of each other and meet inside `lofi-core`. Inside `lofi-core`, the `Backend` trait unifies native HTTP provider turns (`Agent`) and external CLI turns (`ExternalBackend`) under one shared session loop (`turn_session`). At the top of the graph, `lofi-ui` and the `lofi` binary drive interactive TUI sessions, print runs, and CLI subcommands.

| Crate | Responsibility |
| --- | --- |
| `lofi-types` | Configuration structures, model metadata, conversation messages, streaming events, truncation limits, and session transcript schemas. |
| `lofi-error` | Workspace `Error` and `Result` types. |
| `lofi-providers` | Streaming HTTP transports (`IrProvider`) for OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, Google Generative AI, and the `systemone` decision API. |
| `lofi-code` | SWC TypeScript type stripping, the embedded QuickJS `exec` worker, built-in `lofi.*` host tools, shell policy parsing, and auto-mode rubrics. |
| `lofi-core` | `Backend` turn engine, `ToolExecutor`, subagent registry and watchdog, parent-child mailbox, JSON Lines session store, compaction, BM25 recall, and model discovery. |
| `lofi-ui` | `ratatui` terminal user interface, windowed Markdown and thinking rendering, `PickerList` modals, and terminal theme detection. |
| `lofi` | Command-line parser, print mode, `mcp serve` bridge, policy calibration CLI, and the `lofi/e2e/` test suite. |

## Code mode

Instead of advertising separate schemas for every tool, Lofi exposes a single `exec` tool to the model. Each `exec` call carries TypeScript source, a short intent description, optional string bindings, and an optional wall-clock timeout in milliseconds:

```json
{
  "code": "const list = await lofi.ls('src'); return list.entries;",
  "description": "List files under src",
  "strings": { "key": "value" },
  "timeoutMs": 30000
}
```

When an `exec` call arrives, `lofi-code` parses `code` as TypeScript and strips type annotations with SWC. It then wraps the body in an async function so top-level `await` and `return` work, and evaluates the script inside a fresh QuickJS runtime. If the call includes `strings`, the sandbox exposes that map on `lofi_strings` so large string literals do not need source escaping. Inside the sandbox, guest scripts call async methods on the global `lofi` object across six groups:

- File and search: `read`, `ls`, `find`, `grep`, `write`, `edit`, and `patch`.
- Foreground shell: `bash`.
- Background jobs: `jobSpawn`, `jobStatus`, `jobList`, `jobRead`, `jobScreen`, `jobKill`, `jobNotify`, `jobType`, and `jobKeyPress`.
- Sleep and wake-up timers: `sleep`, `notifyAt`, `notifyEvery`, and `notifyCancel`.
- Subagents and mail: `subagent`, `subagentSpawn`, `subagentStatus`, `subagentList`, `subagentUsage`, `subagentWait`, `subagentRead`, `subagentKill`, `subagentNotify`, `subagentMailboxSend`, `subagentMailboxReceive`, and `subagentMailboxList`.
- Context and metadata: `memory`, `recall`, `result`, `models`, `skills`, `skill`, `docs`, `docsSearch`, and `tmpDir`.

Across background jobs, subagents, and timers, Lofi allocates handles from one per-session `IdSpace` of 13-character base-36 strings that sort in creation order. Exact parameter types, return payloads, paging cursors, and safety ceilings are documented in `lofi docs` (`lofi-code/src/docs/api.md`).

### Path grants and shell execution

File tools resolve relative paths against the workspace root and reject `..` escapes or symlink leaves. For external paths, read-only tools also accept built-in read roots such as `lofi.tmpDir` and skill directories without prompting. If a file tool targets an unapproved external path, Lofi prompts the user to grant read or write access to that directory. A later write inside a read-only granted directory prompts again to widen that grant. The one built-in write exception is the workspace memory file (`memory.md`), which the host creates with private permissions without prompting.

Unlike the file tools, foreground `bash` and background jobs execute on the host through `sh -c` with their working directory pinned to the workspace root. Before and during execution, the host enforces shell policy rules, environment stripping, secret redaction, cancellation, and process-group cleanup. Foreground `bash` also refuses bare `&` backgrounding, and steers plain `sleep` or external agent CLI calls toward native tools unless `steer: false` is set. When output grows large, `bash` truncates inline text to `[truncate]` and links a temporary log, while `jobRead` pages background logs incrementally.

### Sandbox results and resource limits

Inside QuickJS, `print(...)` appends to a bounded in-memory log buffer rather than writing to host stdout. When the script finishes, `exec` returns `status` (`completed`, `failed`, `timed_out`, or `cancelled`), optional `logs`, and `value` only if guest code ran `return`. Because of that rule, a script without `return` omits `value` cleanly, whereas an explicit `return null` preserves `value: null`.

Before exiting, the worker also drains any foreground `lofi.*` promises that guest code started without `await`. If any of those calls rejects or returns `ok: false`, the parent `exec` result flips to `ok: false` and records structured `errors`. Throughout execution, QuickJS enforces hard bounds on heap size, stack depth, synchronous CPU time, converted value size, and log bytes. Awaited host tools run outside that guest CPU budget and enforce their own timeouts instead.

## Session transcripts

Lofi saves each persistent session as an append-only JSON Lines file under `$XDG_STATE_HOME/lofi/sessions/<workspace-name>-<path-hash>/` (or `$LOFI_STATE_HOME/lofi/sessions/...`). Appending the path hash keeps workspaces with identical folder names separate, while older slug-only folders remain readable. Inside each file, line one is a `meta` header (`version`, `created`, `cwd`, `model`), and every later line is a `SessionEvent`. At startup, `--no-session` disables persistence, `--continue` reopens the latest workspace transcript, and `--resume <ID>` reopens a transcript by prefix.

Rather than forming a flat list, session events form a tree through `id` and `parent_id` fields. When you roll back with `/tree`, Lofi appends a new branch to an earlier node instead of deleting existing sibling turns. Append-only `cursor` records track the active leaf of that tree. On resume, Lofi walks from that active leaf back to the root so sibling branches stay out of model context. Alongside the `.jsonl` file, CLI backends save external session checkpoints in `<session>.backend.json`, and subagent mail journals to `mailbox-<session>.jsonl`.

| `type` | Main fields | Meaning |
| --- | --- | --- |
| `message` | `role`, `blocks`, `kind` | System, user, assistant, or tool-result message (`kind` separates typed user prompts from injected notices). |
| `context_reported` | `context` | Context window fill reported after a step (also reads legacy `backend_usage` tags). |
| `user_shell` | `command`, `output`, `exit_code`, `signal`, `duration_ms`, `truncated`, `cancelled`, `exclude_from_context` | Direct `!` or `!!` user shell run and its settled output. |
| `tool_timing` | `tool_call_id`, `elapsed_ms` | Wall-clock duration of a finished `exec` tool call. |
| `thinking_timing` | `elapsed_ms` | Wall-clock duration of one finished assistant thinking block. |
| `native_tool` | `parent`, `call_id`, `name`, `args`, `result`, `is_error` | Nested `lofi.*` call executed inside an `exec` block. |
| `round_discarded` | `detail` | Streamed round kept for display but excluded from model context after loop recovery or retry re-roll. |
| `turn_started` | none | Boundary written before a turn starts so replay settles unfinished prior spend. |
| `turn_usage` | `elapsed_ms`, `cost`, `usage` | Per-round checkpointed spend for an in-progress turn, replaced when a terminal turn marker lands. |
| `turn_end` | `model`, `elapsed_ms`, `cost`, `usage`, `stop_reason` | Completed turn marker. |
| `turn_failed` | `model`, `elapsed_ms`, `error`, `cost`, `usage`, `keep_context` | Failed turn marker whose output stays visible in the UI (`keep_context` bounds what stays in model history). |
| `turn_cancelled` | `model`, `elapsed_ms`, `cost`, `usage` | User-canceled turn marker whose partial output stays visible and stays in model context. |
| `cursor` | `leaf_id` | Metadata pointer that selects the active logical head of the transcript tree. |
| `compaction` | `summary`, `first_kept_entry_id`, `summarized_range`, `checkpointed_tail`, `summarized`, `represented`, `kept` | Checkpoint replacing older model history with a structured summary and edited tail. |
| `job_started` / `job_finished` | `job_id`, optional `cmd` | Lineage markers used to reconcile background jobs across restarts and branch rollbacks. |
| `timer_state` | `timers` | Full pending wake-up timer snapshot (`id`, `due_ms`, `every_ms`, `message`, `owner`) re-armed on resume. |
| `subagent_usage` | `usage`, `cost` | Delta token usage and USD spend from settled subagent runs. |
| `run_settled` | `status`, `result`, `error`, `usage`, `cost`, `prior`, `stopped` | Terminal leaf record of a subagent transcript used to restore `SubagentRegistry` entries on restart. |

During a turn, the recorder checkpoints each finished round with `turn_usage` and writes a terminal turn marker once the turn settles. The one exception is a hard context-limit stop, which omits the terminal marker because automatic compaction continues the same logical turn. On replay, discarded rounds and failed turns stay visible in the UI, while model context omits discarded messages and failed messages past the `keep_context` boundary. Canceled turns, by contrast, remain both visible and active in model context so the user can pick up from interrupted work.

At the file format layer, Lofi rejects transcripts whose header version is not `1`. Inside a version-`1` transcript, unknown event types deserialize as ignored entries so newer markers stay forward-compatible. Files without `cursor` records fall back to the last non-cursor event. Likewise, if a crash left a truncated final line, Lofi skips that line so the committed prefix still loads.

## UI theme

In `lofi-ui`, `theme::ThemeManager` owns the selected mode, detected terminal appearance, and palette direction as a pure state machine. Rather than performing terminal I/O itself, it consumes typed `ThemeEvent` values and returns terminal query requests to the caller. When appearance replies arrive, CSI 997 color-scheme reports take priority over OSC 11 background luminance, while forced `light` and `dark` modes override detection.

Terminal I/O lives in `tty_events`, which writes the queries and parses terminal replies through the shared input reader. As those events arrive, `App::update_theme` updates the resolved `Theme` snapshot and clears cached rendered blocks only when resolved colors actually change. During rendering, view widgets read `Theme` from their render context instead of subscribing to theme updates directly. That `Theme` provides shared semantic styles for text emphasis, headings, key hints, markers, badges, and panel backgrounds, while layout attributes stay local to each widget.

## Lint rules

Every crate in the workspace inherits `[lints]` from the root `Cargo.toml`. That shared configuration denies `unwrap_used`, `expect_used`, `todo`, `dbg_macro`, and `print_stdout` across library and binary targets. As a result, production code propagates typed `Result` errors and writes CLI output through `std::io::Write`, while the TUI draws exclusively through `ratatui`. Narrow `#[allow(...)]` attributes document compile-time invariants where needed, and unit test modules allow `unwrap_used` and `expect_used` to keep test assertions readable.
