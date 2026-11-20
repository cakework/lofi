# Lofi

Lofi is a minimal coding-agent harness written in Rust for interactive terminal sessions and non-interactive print runs. Instead of exposing dozens of separate tool schemas, it gives the model a single `exec` tool that runs TypeScript in an embedded QuickJS sandbox. Inside that sandbox, guest scripts call built-in file, search, subagent, and shell helpers on the host. Every model-generated shell command then passes through a structural approval policy before it runs. This shared session, tool, and policy layer drives both direct HTTP model providers and external CLI agents such as Claude Code.

## Build and platform support

Lofi targets Unix-like systems that provide POSIX shell and process-group semantics. Linux is the primary tested platform, while macOS and other Unix targets receive best-effort support, and Windows is not supported. To build Lofi from source, compile the release binary with Cargo and start it inside the workspace directory that you want to edit:

```sh
cargo build --release
target/release/lofi
```

## Test suite

To format the workspace, run Clippy lints, and execute all unit and end-to-end tests together, run `just`:

```sh
just check
```

If you only want to run the process-level suite, run `just e2e` instead. That suite drives the compiled `lofi` binary against a local mock model server in both print mode and a pseudo-terminal (PTY, a virtual terminal device). Because all model traffic stays local, the tests do not need API keys or external network access. See [End-to-end testing](docs/e2e.md) for the test coverage map and [Development](docs/development.md) for workspace architecture notes.

## Command-line usage

Running `lofi` with no flags or subcommand opens the interactive terminal user interface (TUI, a full-screen terminal view) rooted at the current working directory. To run a single non-interactive turn instead, pass `-p` or `--print` to stream the assistant response to stdout and exit. In both modes, command-line flags let you override the model, backend, environment variables, and session behavior for that run.

### CLI flags

You can combine the flags below with interactive sessions or `--print` runs. Flags that only apply to print mode fail immediately at startup if `--print` is missing. To pass multiple environment variables into configuration resolution and `lofi.bash` child processes, repeat `-e` or `--env`.

| Flag | Description |
| --- | --- |
| `-p`, `--print <PROMPT>` | Run one non-interactive turn and stream assistant text to stdout. |
| `--model <SPEC>` | Select the model as `provider/model[:level][@tier]`. |
| `--backend <NAME>` | Run on the named CLI agent backend from `config.toml`. |
| `-e`, `--env <NAME[=VALUE]>` | Set an environment variable for this process. A bare `NAME` forwards the value from the parent shell. |
| `-c`, `--continue` | Resume the most recent session for this workspace. |
| `--resume <ID>` | Resume a saved session by identifier prefix. |
| `--no-session` | Do not write a session transcript. |
| `--session` | With `--print`, save a transcript so a later run can resume it. |
| `--summary <FILE>` | With `--print`, write one JSON summary record of the run to `<FILE>`. |

### Subcommands

Subcommands inspect configured models, manage saved transcripts, query embedded API documentation, test shell policy rules, or serve the internal MCP bridge. Each informational subcommand writes to stdout and exits without starting a model turn. To view all flags for a specific subcommand, run `lofi <subcommand> --help`.

| Command | Description |
| --- | --- |
| `models` | Print configured models with their provider identifier and display name, then exit. |
| `sessions` | Print saved sessions for this workspace and exit. Pass `--all` to list every workspace with its directory. |
| `sessions prune` | Delete session transcripts older than `--delete-older-than <AGE>` (default `7d`). Pass `--dry-run` to preview or `--all` for all workspaces. |
| `docs` | Print the embedded `lofi` sandbox API reference index. |
| `docs search <QUERY>` | Search the embedded `lofi` sandbox API reference by keyword. |
| `policy explain <COMMAND>` | Evaluate a shell command against `policy.toml` without running it. |
| `policy calibrate` | Replay, probe, or sweep thresholds for the `[auto_mode]` decision gate. See [Automatic approval](docs/configuration.md#automatic-approval). |
| `mcp serve` | Speak the internal `exec` bridge protocol on stdio for a spawned CLI backend. |

### Print-mode sessions

By default, `--print` runs in memory and writes no session transcript to disk. To keep a transcript, pass `--session` to start one, `--continue` to resume the latest workspace one, or `--resume <ID>` to resume by prefix. Combining `--no-session` with any of those three flags fails at startup so print mode never ignores an explicit session flag.

When a print run uses a session, Lofi writes `session: <ID>` to stderr before the first round so callers can capture the identifier. After each round finishes, Lofi checkpoints token usage and cost to that transcript. If a run is interrupted mid-turn, resuming the session later keeps both the unfinished conversation history and the spend already recorded:

```sh
lofi --print "Fix the failing test" --session
lofi --print "Now add a regression test" --resume 3f9a2c
```

### Run summary

Passing `--summary <FILE>` alongside `--print` writes one JSON summary object to `<FILE>` when the run ends, while leaving the assistant stream on stdout unchanged. Lofi writes this record even if setup fails before the first model round, which lets batch runners account for every invocation. However, if `SIGINT` or `SIGTERM` terminates the process, Lofi exits by that signal without writing a summary record. If a signal kills the run, do not read `<FILE>`, because it can be missing, truncated, or left over from a previous run.

```sh
lofi --print "Fix the failing test" --summary /tmp/run.json
```

| Field | Description |
| --- | --- |
| `ok` | `true` when `status` is `completed`, and `false` otherwise. |
| `status` | Run outcome: `completed`, `failed`, or `incomplete`. |
| `session` | Session identifier written or resumed by the run, or `null` when no transcript was saved. |
| `model` | Object with `provider`, `id`, `selected`, `thinking`, and `service_tier`. |
| `rounds` | Number of model rounds executed in the turn. |
| `tool_calls` | Number of `exec` tool calls executed in the turn. |
| `retries` | Number of transport retries performed during the turn. |
| `usage` | Accumulated `input_tokens`, `output_tokens`, `cache_read_tokens`, and `cache_write_tokens` across rounds. |
| `cost` | Accumulated USD cost for the parent turn. Add `subagents.cost` for total run spend. |
| `subagents` | Subagent token totals plus `cost` in USD. |
| `stop_reason` | Final provider stop reason when one was reported. |
| `duration_ms` | Wall-clock run duration in milliseconds from process start to exit. |
| `error` | Error message when the run failed, or `null` on completion. |

## Interactive TUI

The TUI splits interaction between prompt editing in Input mode and transcript browsing in Navigate and Select modes. During a session, model-generated shell commands execute on the host through `sh -c`, because QuickJS isolation does not sandbox native child processes. By default, the shell policy asks for approval before running any model-generated command. Read [Shell policy](docs/configuration.md#shell-policy) before enabling a broader approval policy.

### Input keys and user shell commands

Input mode focuses the prompt editor at the bottom of the terminal. Prefixing a line with `!` runs the command through `sh -c` in the workspace and attaches its output to later model context. If you want to run a shell command without sending its output to the model, prefix the line with `!!` instead. Both `!` and `!!` commands run without a wall-clock timeout, persist in the session transcript, and stop when you press `Ctrl+C`.

| Key | Action |
| --- | --- |
| `Enter` | Send the prompt, or queue it while a turn is running. |
| `Alt+Enter` / `Ctrl+J` | Insert a newline in the prompt. |
| `Alt+Up` | Restore the latest queued prompt into the editor. |
| `Up` / `Down` | Move between prompt lines, or recall input history at the top or bottom edge. |
| `PageUp` / `PageDown` | Scroll the transcript by one page. |
| `Tab` | Switch from Input mode to Navigate mode, or accept a slash-command completion. |
| `Esc` | Cancel the active run, or clear the prompt when idle. |
| `Ctrl+C` | Clear the prompt, cancel the active run, or quit on a second press when idle and empty. |
| `Ctrl+D` | Delete the character under the cursor, or cancel or quit when the prompt is empty. |
| `Ctrl+B` | Open the background work dialog when the prompt is empty. |

### Navigate and Select keys

Pressing `Tab` in Input mode switches the TUI into Navigate mode so you can move a cursor across turns, tool calls, and thinking blocks. From Navigate mode, pressing `v` starts a visual highlight in Select mode, where `y` copies the selected range back into the prompt editor. To leave Navigate or Select mode at any time, press `i` or `Tab` to return to Input mode.

| Key | Action |
| --- | --- |
| `j` / `k` or `Up` / `Down` | Scroll or move by line in the transcript or focused detail view. |
| `h` / `l` or `Left` / `Right` | Move left or right by column. |
| `0` / `^` / `$` | Jump to line start, first non-blank character, or line end. |
| `w` / `b` / `e` | Move forward or backward by word. |
| `g` / `G` | Jump to the top or bottom of the transcript. |
| `[` / `]` | Jump to the previous or next turn. |
| `PageUp` / `PageDown` | Move the cursor by one page. |
| `Enter` / `Space` | Open or focus a tool or thinking detail block, or leave a focused detail in Select mode. |
| `v` | Start visual selection in Select mode. |
| `y` | Copy the current line in Navigate mode, or copy the selection to Input in Select mode. |
| `i` / `Tab` / `Esc` | Return from Navigate or Select mode to Input mode. |

### Slash commands

Typing `/` at the start of an empty prompt opens the slash-command completion menu. From that menu, move through candidates with `Up` and `Down`, press `Tab` to accept one, and press `Enter` to run it.

| Command | Action |
| --- | --- |
| `/help` | Show keybindings and slash commands. |
| `/clear` | Clear the visible transcript log. |
| `/compact` | Fold older conversation history into a summary. |
| `/debug` | Toggle resource diagnostics. |
| `/recall [query]` | Search session history and compacted turns. |
| `/new` | Start a new session. |
| `/resume` | Open the saved-session picker. |
| `/tree` | Open the session tree picker to roll back to an earlier turn. |
| `/session` | Show active session details. |
| `/model` | Open the model picker. |
| `/thinking` | Switch the thinking effort level. |
| `/service` | Switch the provider service tier. |
| `/policy` | Switch the shell approval mode for this session. |
| `/paths` | View and edit allowed external directories for file tools. |
| `/theme` | Switch the color scheme (`auto`, `light`, or `dark`) for this session. |
| `/job` or `/jobs` | Open the background dialog on shell jobs. |
| `/subagent` or `/subagents` | Open the background dialog on subagents. |
| `/timer` or `/timers` | Open the background dialog on pending wake-up timers. |
| `/quit` or `/exit` | Exit Lofi. |

## Configuration and paths

Lofi keeps user-edited configuration files separate from runtime state managed by the harness. At runtime, the configuration directory stays read-only, while subdirectories under the state directory are created on demand only when a session writes data. Read [Configuration](docs/configuration.md) for provider setup, Claude Code backends, context compaction, allowed paths, and shell policy rules.

| Path or variable | Purpose |
| --- | --- |
| `$XDG_CONFIG_HOME/lofi/config.toml` | Main configuration file (defaults to `~/.config/lofi/config.toml`). Override the file path with `$LOFI_CONFIG`. |
| `$XDG_CONFIG_HOME/lofi/policy.toml` | Shell approval rules and `[auto_mode]` configuration. Override the file path with `$LOFI_POLICY`. |
| `$XDG_CONFIG_HOME/lofi/AGENTS.md` | Global agent instructions loaded alongside workspace `AGENTS.md` files. |
| `$XDG_CONFIG_HOME/lofi/skills/` | Global skill directories loaded alongside workspace `.lofi/skills/` directories. |
| `$XDG_STATE_HOME/lofi/` | Session transcripts, workspace memory, directory grants, and discovery cache (defaults to `~/.local/state/lofi/`). Override the base state root with `$LOFI_STATE_HOME` so Lofi uses `$LOFI_STATE_HOME/lofi/`. |
| `LOFI__SECTION__KEY` | Per-invocation TOML configuration override, such as `LOFI__DEFAULT_MODEL=claude/opus`. |

## Acknowledgments

Lofi builds on ideas explored in three earlier coding-agent projects. In particular, its single-tool TypeScript sandbox, branching session transcripts, and terminal workflow grow directly out of those designs. Refer to their repositories for background context:

- [pi](https://github.com/earendil-works/pi)
- [pi-fabric](https://github.com/monotykamary/pi-fabric)
- [pi-vcc](https://github.com/monotykamary/pi-vcc)
