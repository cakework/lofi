pub mod agent;
pub mod backend;
mod bash_env;
pub mod calibrate;

pub use lofi_code::docs;
pub(crate) mod compact;
pub mod config_loader;
pub(crate) mod context_edit;
mod eval_log;
pub(crate) mod image;
mod lifecycle;
pub mod models;
pub mod paths;
pub mod recall;
pub(crate) mod retry;
pub mod session;
mod shell;
pub mod state;

pub(crate) use agent::{exec_input_code_and_label, SubagentSupport};
pub use agent::{
    exec_result_display, new_cancel_flag, signal_cancel, Agent, AgentEvent, ConfirmRequest,
    ConfirmResponse, RunControls, SubagentLive, SubagentNotice, SubagentNoticeKind, SubagentOp,
    SubagentRegistry, SubagentRunning, TurnInput, TurnRequest,
};
pub use compact::Compaction;
pub use lifecycle::{
    AgentLifecycle, CompactionAction, HardCompactOutcome, LineageJobReconciliation,
    SettledCompaction,
};
pub(crate) use lofi_code::compact_hook::CodeCompactionHook;
pub use lofi_code::policy::{default_mode as default_approval_mode, PolicyOverride};
pub use lofi_code::tools::{
    JobAttributes, JobColor, JobInfo, JobNotice, JobRegistry, JobScreen, JobScreenLine, JobSpan,
    JobStyle, TimerInfo, TimerRegistry,
};
pub use lofi_code::ConfirmReason;
pub use lofi_error::{Error, Result};
pub use lofi_types::{BashApprovalMode, CompactBlock, CompactionHook, SummarySection};
pub use lofi_types::{Config, SessionEvent};
pub use models::ModelRegistry;
pub use shell::{run_user_shell_command, UserShellResult};
