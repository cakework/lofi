# End-to-end testing

The process-level test target is declared in `lofi/Cargo.toml` and implemented in `lofi/e2e/`. Each test spawns the compiled `lofi` binary in print mode or inside a pseudo-terminal (PTY, a virtual terminal device). Every child process runs against temporary configuration, policy, state, and workspace directories paired with a local mock HTTP server. Because all model traffic stays on loopback, the suite needs no API keys and makes no external network calls. The target currently contains 297 tests.

## Run the suite

To execute all end-to-end tests at once, run `just`:

```sh
just e2e
```

When working on a single feature, pass a Cargo name filter to run one module or one test case:

```sh
cargo test -p lofi --test e2e --offline providers::
cargo test -p lofi --test e2e --offline responses_stalled_body_times_out_and_retries
```

To list all registered test names without running them, pass `--list` to the test binary:

```sh
cargo test -p lofi --test e2e --offline -- --list
```

Before finishing a change, run `just check` to execute formatting, Clippy lints, unit tests, and the full E2E suite together.

## Test design

Shared helpers in `lofi/e2e/support.rs` build isolated sandboxes for each child `lofi` process. By default, a test fixture serves OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and Google Generative AI endpoints from one local server. At the same time, the fixture strips common provider API keys from child environments and points `LOFI_CONFIG`, `LOFI_POLICY`, and `LOFI_STATE_HOME` at temporary paths. Five helpers in `support.rs` drive most test cases:

- `Fixture` creates and cleans up isolated configuration, policy, state, and workspace directories.
- `MockServer` and `MockResponse` script ordered HTTP JSON and Server-Sent Events (SSE, a streaming HTTP text format) replies.
- Print-mode helpers capture stdout, stderr, exit codes, recorded HTTP requests, and saved transcripts.
- `Tui` drives PTY keystrokes, screen assertions, window resizes, signals, and OSC escape replies.
- Bounded polling helpers wait for asynchronous process, transcript, and background-job state changes.

Using those helpers, tests assert public behavior at process boundaries. Typical assertions inspect HTTP request payloads, terminal output, exit codes, JSON Lines transcripts, process-group cleanup, and state after a restart. In contrast, wire-parser edge cases and QuickJS memory limits stay in crate unit tests where spawning a full binary adds no extra signal.

## Coverage map

The test counts below come directly from `cargo test -p lofi --test e2e --offline -- --list`. Each module owns a distinct user-visible workflow so failures point straight to the affected subsystem. Together, the nine modules cover subcommands, print runs, and interactive PTY sessions.

| Module | Tests | Main coverage |
| --- | ---: | --- |
| `agent` | 25 | System prompts, skills, thinking controls, loop recovery, retries, directory grants, tool errors, cancellation, and durable partial rounds. |
| `backends` | 63 | Fake Claude CLI child processes: handshake validation, reported and aliased models, effort flags, session resume, compaction, cost deltas, jobs, subagents, backend switching, and CLI evaluators. |
| `cli` | 28 | Help and version flags, print mode, informational subcommands, `-e` environment injection, startup validation, session listing and pruning, signals, and reasoning replay. |
| `configuration` | 11 | `[auto_mode]` evaluation, reserved `[[ask]]` rules, rejected legacy `mode` keys, resolved and redacted environment values, model discovery cache fallback, and secret boundaries. |
| `native_tools` | 22 | File, search, docs, skill, memory, recall, result, image, and background-job APIs plus interactive PTY jobs, key sequences, signals, dimensions, and idle detection. |
| `providers` | 19 | All four chat protocols, fragmented SSE chunks, custom headers, `no_auth` providers, premature EOF, long active streams, stalled body timeouts, retry reset after progress, tiers, images, and signed reasoning. |
| `sessions` | 66 | Transcript creation, resume, tree rollback, job reconciliation, failed and canceled turns, compaction checkpoints, truncated final lines, file permissions, temp leases, and large transcript replay. |
| `tools` | 14 | End-to-end `exec` tool cycles across all four protocols, parallel and interleaved calls, malformed tool JSON, thinking signatures, Anthropic cache breakpoints, compaction, and resume. |
| `tui` | 49 | Detail expansion, slash commands, picker modals, approval and path dialogs, queued prompts, `!` and `!!` user shell runs, visual selection and OSC 52 copy, Markdown resize, and terminal cleanup. |
| Total | 297 | Real-binary coverage across print mode, subcommands, and PTY sessions. |

### Provider and stream coverage

Provider tests send real HTTP requests over localhost across all four supported chat protocols. Across those protocols, the suite tests streamed text, reasoning traces, token usage, tool calls, image inputs, error classification, and multi-turn replay. Lower-level transport tests also simulate fragmented socket reads, premature connection drops, retry counter resets after stream progress, and OpenAI Responses lifecycle events. Finally, 31-second Responses streams make sure that active streams have no fixed total-turn deadline, whereas streams that stall past `body_idle_timeout_ms` time out and retry.

### Session and process coverage

Session tests stop and restart the `lofi` binary against saved JSON Lines transcripts on disk. Across restarts, they test active tree branches, `/tree` rollbacks, compaction checkpoints, truncated final lines, missing cursor records, unsupported format versions, file permissions, and model restoration. Alongside transcript state, process tests make sure that process groups exit cleanly, signals report expected exit codes, and terminal modes restore on shutdown. On Linux with glibc, large replay and tree-picker fixtures also assert that freed heap memory returns to the operating system.

### Terminal and job coverage

Terminal tests run `lofi` in a PTY to cover keyboard input, paste, resize, and OSC 2, 7, 11, 52, and 7501 sequences. On top of raw terminal I/O, they test modal priority, transcript cursor movement, expanded details, and live `!` shell streaming. Background-job tests cover both piped processes and interactive PTY jobs. Those cases test `jobType`, `jobKeyPress`, `Ctrl+C`, `Ctrl+D`, terminal dimensions, idle detection, wake-up notices, `jobKill`, and shutdown cleanup.

## Current boundaries

Some narrow or host-dependent behaviors belong in crate unit tests rather than full-process E2E tests. Two heap-release tests require Linux with glibc, whereas the rest of the suite targets POSIX process and PTY semantics without Windows support. Crate-level tests cover the following areas directly:

- QuickJS heap, stack, synchronous CPU, log buffer, value conversion, and interrupt limits.
- Malformed UTF-8 SSE frames, oversized SSE event buffers, and detailed wire-parser permutations.
- Exhaustive configuration precedence matrices, corrupted discovery caches, JPEG quality sweeps, and token price arithmetic.
- Fine-grained Unicode display widths, focus events, and raw terminal escape permutations.
- Concurrent transcript writers and older synthetic transcript migration fixtures.
- Release binaries, Nix flake packaging, and non-Linux Unix platforms.

## Add a test

When adding coverage for a new feature or bug fix, keep the test deterministic and isolated from the host machine. That isolation lets Cargo run E2E tests in parallel without port collisions or shared state leaks. Follow these steps when adding a test:

1. Place the test in the `lofi/e2e/` module that owns the user-facing workflow.
2. Build the test environment with `Fixture`, `MockServer`, `MockResponse`, and `Tui`.
3. Keep all HTTP traffic on `127.0.0.1` and all files inside the fixture directories.
4. Wait for asynchronous state with the bounded polling helpers in `support.rs` instead of fixed `sleep` calls.
5. Assert process-visible outputs and saved files rather than internal Rust types.
6. Run the single test with `cargo test -p lofi --test e2e --offline <name>`, then run `just e2e`.
7. Update the coverage map on this page when module counts or boundaries change.
