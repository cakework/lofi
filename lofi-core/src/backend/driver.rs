//! Transport seam between generic external-backend orchestration and a
//! concrete CLI agent driver (such as Claude Code).

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use lofi_error::Result;
use lofi_types::{BackendConfig, ContentBlock, StopReason, ThinkingLevel};

use super::contract::{Completion, CompletionRequest};
use crate::agent::{ToolCallContext, ToolExecutor};

/// Why a multi-round CLI turn was stopped cleanly by the engine consumer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MidTurnStop {
    Preempt,
    LoopDetected,
}

/// How one CLI prompt exchange ended.
#[derive(Debug)]
pub(super) enum PromptOutcome {
    Completed(String),
    /// Stopped cleanly at a round boundary (hard context cap or preemption).
    Stopped,
}

/// Mid-turn controls passed into one CLI prompt exchange.
#[derive(Default)]
pub(super) struct PromptControls {
    pub hard_cap: Option<u64>,
    pub stop_rx: Option<tokio::sync::watch::Receiver<Option<MidTurnStop>>>,
}

/// One live CLI child session owned by the backend worker thread.
#[async_trait]
pub(super) trait CliSession: Send {
    fn session_id(&self) -> &str;
    fn context_window(&self) -> Option<u64>;
    fn stop_reason(&self) -> Option<StopReason>;
    fn is_closed(&mut self) -> bool;
    fn is_initialized(&self) -> bool;
    fn prompt_written(&self) -> bool;
    fn take_cost(&mut self) -> f64;
    async fn set_model(&mut self, model: Option<&str>) -> Result<()>;
    async fn prompt(
        &mut self,
        prompt: &str,
        images: &[ContentBlock],
        executor: ToolExecutor,
        context: ToolCallContext,
        controls: PromptControls,
    ) -> Result<PromptOutcome>;
    async fn close(&mut self) -> Result<()>;
}

/// Factory and stateless capabilities for one CLI agent driver.
#[async_trait]
pub(super) trait CliDriver: Send + Sync {
    fn effort_levels(&self) -> Vec<ThinkingLevel>;

    async fn open_session(
        &self,
        root: &Path,
        resume: Option<&str>,
        model: Option<&str>,
        effort: ThinkingLevel,
        system_prompt: &str,
    ) -> Result<Box<dyn CliSession>>;

    async fn complete(
        &self,
        root: &Path,
        model: Option<&str>,
        request: &CompletionRequest,
    ) -> Result<Completion>;
}

/// Build the concrete driver for a configured backend entry.
#[must_use]
pub(super) fn for_config(config: &BackendConfig) -> Arc<dyn CliDriver> {
    match config {
        BackendConfig::ClaudeCode {
            command,
            frame_idle_timeout_ms,
            ..
        } => Arc::new(super::claude::ClaudeDriver::new(
            command.clone(),
            frame_idle_timeout_ms.map(Duration::from_millis),
        )),
    }
}

/// Supported thinking levels for a configured backend entry.
#[must_use]
pub(super) fn effort_levels_for(config: &BackendConfig) -> Vec<ThinkingLevel> {
    for_config(config).effort_levels()
}
