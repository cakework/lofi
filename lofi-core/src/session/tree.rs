//! Session tree projection over the positional event index. Owns all
//! index-walking (active path, hidden checkpoints, branch detection) and
//! event-derived row labels so the TUI only maps rows onto widgets.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use lofi_types::{ContentBlock, Role, SessionEventKind};

use super::store::{IndexKind, SessionCursor, TreeIndex};

/// A display-agnostic tree row. The TUI maps these onto its widgets; all
/// session reasoning (which events, what labels, which branches) lives here.
#[derive(Debug, Clone)]
pub struct TreeRow {
    pub branch_point: Box<str>,
    pub label: Box<str>,
    /// Shared indentation: a whole-transcript row list repeats the same two
    /// connectors, so each row holds a reference instead of its own string.
    pub prefix: Arc<str>,
    pub prefill: Box<str>,
    pub is_active: bool,
    /// Position in the tree index, which never exceeds `u32::MAX` because one
    /// entry per transcript event is already the addressable limit.
    pub source_index: u32,
    pub source_offset: u64,
    pub source_kind: IndexKind,
    pub hydrated: bool,
}

struct TreeCtx<'a> {
    index: &'a TreeIndex,
    cursor: &'a SessionCursor,
    native_tools: RefCell<HashMap<String, Vec<(String, String)>>>,
    lazy_native_tools: bool,
    hydrate: bool,
    retain_prefill: bool,
    hydrate_only: Option<&'a HashSet<usize>>,
}

/// A set of positions or path slots, sized to the collection it describes.
/// A tree row set spans the whole transcript, where a hash set per entry
/// would cost more than the rows it classifies.
#[derive(Default)]
struct PositionSet {
    words: Vec<u64>,
}

impl PositionSet {
    fn with_len(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(64)],
        }
    }

    fn insert(&mut self, position: usize) {
        if let Some(word) = self.words.get_mut(position / 64) {
            *word |= 1 << (position % 64);
        }
    }

    fn contains(&self, position: usize) -> bool {
        self.words
            .get(position / 64)
            .is_some_and(|word| word & (1 << (position % 64)) != 0)
    }
}

#[must_use]
pub fn build_tree_rows(
    index: &TreeIndex,
    leaf: Option<u32>,
    cursor: &SessionCursor,
) -> Vec<TreeRow> {
    build_tree_rows_inner(index, leaf, cursor, true, true, None)
}

#[must_use]
pub fn build_tree_row_skeletons(
    index: &TreeIndex,
    leaf: Option<u32>,
    cursor: &SessionCursor,
) -> Vec<TreeRow> {
    build_tree_rows_inner(index, leaf, cursor, false, false, None)
}

pub fn hydrate_tree_rows(
    index: &TreeIndex,
    cursor: &SessionCursor,
    requested: &[(usize, TreeRow)],
    mut emit: impl FnMut(Vec<(usize, TreeRow)>) -> bool,
    cancelled: impl Fn() -> bool,
) {
    if requested.is_empty() || cancelled() {
        return;
    }
    let ctx = TreeCtx {
        index,
        cursor,
        native_tools: RefCell::new(HashMap::new()),
        lazy_native_tools: true,
        hydrate: true,
        retain_prefill: false,
        hydrate_only: None,
    };
    let mut rows = Vec::with_capacity(requested.len());
    for (display_index, skeleton) in requested {
        if cancelled() {
            return;
        }
        let mut entry = Vec::with_capacity(1);
        push_tree_row(
            &ctx,
            skeleton.source_index as usize,
            Arc::clone(&skeleton.prefix),
            skeleton.is_active,
            &mut entry,
        );
        if let Some(entry) = entry.pop() {
            rows.push((*display_index, entry));
        }
    }
    if !rows.is_empty() {
        let _ = emit(rows);
    }
}

pub fn hydrate_tree_window(
    index: &TreeIndex,
    cursor: &SessionCursor,
    skeletons: &[TreeRow],
    display_range: std::ops::Range<usize>,
    emit: impl FnMut(Vec<(usize, TreeRow)>) -> bool,
    cancelled: impl Fn() -> bool,
) {
    let requested: Vec<_> = display_range
        .filter_map(|position| {
            skeletons
                .get(position)
                .filter(|entry| !entry.hydrated)
                .cloned()
                .map(|entry| (position, entry))
        })
        .collect();
    hydrate_tree_rows(index, cursor, &requested, emit, cancelled);
}

/// Path slots hidden by a compaction checkpoint, resolved through the
/// precomputed checkpoint positions: a marker hides the path between the
/// event it keeps and itself.
fn hidden_path_positions(index: &TreeIndex, path: &[u32]) -> PositionSet {
    let mut hidden = PositionSet::with_len(path.len());
    let checkpoints = index.checkpoints();
    if checkpoints.is_empty() {
        return hidden;
    }
    let mut targets: HashSet<u32> = HashSet::with_capacity(checkpoints.len() * 2);
    for checkpoint in checkpoints {
        targets.insert(checkpoint.marker);
        if let Some(first_kept) = checkpoint.first_kept {
            targets.insert(first_kept);
        }
    }
    // Resolving every event through a map would scale with the path; only the
    // checkpoint boundaries are wanted, so one pass records those.
    let mut slots: HashMap<u32, usize> = HashMap::with_capacity(targets.len());
    for (slot, &position) in path.iter().enumerate() {
        if targets.contains(&position) {
            // First occurrence holds: a later duplicate must not win the hide
            // start, matching the lineage projection's own rule.
            slots.entry(position).or_insert(slot);
        }
    }
    for checkpoint in checkpoints {
        let Some(first_kept) = checkpoint.first_kept else {
            continue;
        };
        if !checkpoint.checkpointed_tail {
            continue;
        }
        let (Some(&start), Some(&marker)) = (slots.get(&first_kept), slots.get(&checkpoint.marker))
        else {
            continue;
        };
        if start < marker {
            for slot in start..marker {
                hidden.insert(slot);
            }
        }
    }
    hidden
}

fn top_level_tree_nodes(index: &TreeIndex) -> Vec<usize> {
    (0..index.len())
        .filter(|&position| {
            if !is_tree_node(index.kind(position)) {
                return false;
            }
            let mut parent = index.parent(position);
            let mut steps = 0usize;
            while let Some(ancestor) = parent {
                if is_tree_node(index.kind(ancestor)) {
                    return false;
                }
                if steps > index.len() {
                    return false;
                }
                steps += 1;
                parent = index.parent(ancestor);
            }
            true
        })
        .collect()
}

#[allow(clippy::too_many_lines)]
fn build_tree_rows_inner(
    index: &TreeIndex,
    leaf: Option<u32>,
    cursor: &SessionCursor,
    hydrate: bool,
    retain_prefill: bool,
    hydrate_only: Option<&HashSet<usize>>,
) -> Vec<TreeRow> {
    let native_tools = if hydrate && hydrate_only.is_none() {
        build_native_tool_map(index, cursor)
    } else {
        HashMap::new()
    };
    let active_path = index.active_path(leaf).unwrap_or_default();
    let mut active_set = PositionSet::with_len(index.len());
    for &position in &active_path {
        active_set.insert(position as usize);
    }
    let hidden = hidden_path_positions(index, &active_path);
    let trunk: Vec<usize> = active_path
        .iter()
        .enumerate()
        .filter(|(slot, _)| !hidden.contains(*slot))
        .map(|(_, &position)| position as usize)
        .filter(|&position| is_tree_node(index.kind(position)))
        .collect();

    let n = trunk.len();
    let mut out = Vec::new();
    let ctx = TreeCtx {
        index,
        cursor,
        native_tools: RefCell::new(native_tools),
        lazy_native_tools: false,
        hydrate,
        retain_prefill,
        hydrate_only,
    };
    let connectors: [Arc<str>; 2] = [
        Arc::from("\u{251c}\u{2500} "),
        Arc::from("\u{2514}\u{2500} "),
    ];
    for (pos, &position) in trunk.iter().enumerate() {
        let is_last = pos == n - 1;
        let child_indent = if is_last { "   " } else { "\u{2502}  " };
        push_tree_row(
            &ctx,
            position,
            Arc::clone(&connectors[usize::from(is_last)]),
            active_set.contains(position),
            &mut out,
        );
        let mut branches: Vec<usize> = Vec::new();
        if index.kind(position) == IndexKind::UserPrompt {
            if let Some(outcome) = find_turn_outcome(position, index) {
                if !active_set.contains(outcome) {
                    branches.push(outcome);
                }
            }
        }
        let user_branches: Vec<usize> = index
            .children(position)
            .iter()
            .copied()
            .filter(|&child| {
                index.kind(child as usize) == IndexKind::UserPrompt
                    && !active_set.contains(child as usize)
            })
            .map(|child| child as usize)
            .collect();
        branches.extend(user_branches);
        if !branches.is_empty() {
            render_branch_subtree(&ctx, &branches, child_indent, &mut out);
        }
    }
    if trunk.is_empty() && out.is_empty() {
        let roots = top_level_tree_nodes(index);
        render_branch_subtree(&ctx, &roots, "", &mut out);
    }
    out
}

fn build_native_tool_map(
    index: &TreeIndex,
    cursor: &SessionCursor,
) -> HashMap<String, Vec<(String, String)>> {
    let offsets: Vec<u64> = (0..index.len())
        .filter(|&position| index.kind(position) == IndexKind::NativeTool)
        .map(|position| index.offset(position))
        .collect();
    let mut map: HashMap<String, Vec<(String, String)>> = HashMap::new();
    let Ok(summaries) = cursor.native_tool_summaries(&offsets) else {
        return map;
    };
    for (parent, name, args) in summaries {
        map.entry(parent).or_default().push((name, one_line(&args)));
    }
    map
}

fn load_native_tools_for_turn(
    ctx: &TreeCtx<'_>,
    tool_result: usize,
    tool_use_id: &str,
) -> Vec<(String, String)> {
    let mut offsets = Vec::new();
    let mut current = tool_result;
    let mut visited = HashSet::new();
    while visited.insert(current) {
        let Some(&next) = ctx
            .index
            .children(current)
            .iter()
            .find(|&&child| ctx.index.kind(child as usize) != IndexKind::UserPrompt)
        else {
            break;
        };
        let next = next as usize;
        match ctx.index.kind(next) {
            IndexKind::NativeTool => offsets.push(ctx.index.offset(next)),
            IndexKind::TurnEnd | IndexKind::TurnFailed | IndexKind::TurnCancelled => break,
            _ => {}
        }
        current = next;
    }
    let Ok(summaries) = ctx.cursor.native_tool_summaries(&offsets) else {
        return Vec::new();
    };
    summaries
        .into_iter()
        .filter(|(parent, _, _)| parent == tool_use_id)
        .map(|(_, name, args)| (name, one_line(&args)))
        .collect()
}

fn render_branch_subtree(ctx: &TreeCtx, roots: &[usize], prefix: &str, out: &mut Vec<TreeRow>) {
    let index = ctx.index;
    let n = roots.len();
    for (pos, &root) in roots.iter().enumerate() {
        let is_last = pos == n - 1;
        let connector = if is_last {
            "\u{2514}\u{2500} "
        } else {
            "\u{251c}\u{2500} "
        };
        let child_indent = format!("{prefix}{}", if is_last { "   " } else { "\u{2502}  " });
        let chain = walk_chain(root, index);
        let chain_set: HashSet<usize> = chain.iter().copied().collect();
        let cn = chain.len();
        for (cpos, &position) in chain.iter().enumerate() {
            let cprefix = if cpos == 0 {
                format!("{prefix}{connector}")
            } else {
                let cis_last = cpos == cn - 1;
                format!(
                    "{child_indent}{}",
                    if cis_last {
                        "\u{2514}\u{2500} "
                    } else {
                        "\u{251c}\u{2500} "
                    }
                )
            };
            let sub_indent = format!(
                "{child_indent}{}",
                if cpos == cn - 1 { "   " } else { "\u{2502}  " }
            );
            push_tree_row(ctx, position, Arc::from(cprefix.as_str()), false, out);
            let mut sub_branches: Vec<usize> = Vec::new();
            if index.kind(position) == IndexKind::UserPrompt {
                if let Some(outcome) = find_turn_outcome(position, index) {
                    if !chain_set.contains(&outcome) {
                        sub_branches.push(outcome);
                    }
                }
            }
            let user_children: Vec<usize> = index
                .children(position)
                .iter()
                .copied()
                .filter(|&child| {
                    index.kind(child as usize) == IndexKind::UserPrompt
                        && !chain_set.contains(&(child as usize))
                })
                .map(|child| child as usize)
                .collect();
            sub_branches.extend(user_children);
            if !sub_branches.is_empty() {
                render_branch_subtree(ctx, &sub_branches, &sub_indent, out);
            }
        }
    }
}

#[must_use]
pub fn find_next_user_prompt(start: usize, index: &TreeIndex) -> Option<usize> {
    let children = index.children(start);
    if let Some(&prompt) = children
        .iter()
        .find(|&&child| index.kind(child as usize) == IndexKind::UserPrompt)
    {
        return Some(prompt as usize);
    }
    let compaction = children
        .iter()
        .copied()
        .find(|&child| index.kind(child as usize) == IndexKind::Compaction)?;
    index
        .children(compaction as usize)
        .iter()
        .copied()
        .find(|&child| index.kind(child as usize) == IndexKind::UserPrompt)
        .map(|child| child as usize)
}

#[must_use]
pub fn walk_chain(start: usize, index: &TreeIndex) -> Vec<usize> {
    let mut chain = vec![start];
    let mut visited = HashSet::new();
    visited.insert(start);
    let mut current = start;
    loop {
        let next = if index.kind(current) == IndexKind::UserPrompt {
            find_turn_outcome(current, index)
        } else {
            find_next_user_prompt(current, index)
        };
        match next {
            Some(next) if visited.insert(next) => {
                chain.push(next);
                current = next;
            }
            _ => break,
        }
    }
    chain
}

type TreeRowFields = (String, String, String);

/// A `/tree` row's revert target for the kinds that share one rule across the
/// skeleton and hydrated paths: a `UserPrompt`, `TurnFailed`, or
/// `TurnCancelled` reverts to its parent (an aborted outcome has no completed
/// assistant message, so reverting to the row itself would reselect the
/// broken/cancelled tail and be a no-op); anything else reverts to itself.
///
/// `TurnEnd` and `Compaction` are deliberately *not* handled here: `TurnEnd`
/// reverts to itself only conceptually and each path expresses it inline, and
/// `Compaction` resolves differently per path (skeleton has no checkpoint
/// details, hydrated consults the checkpoint positions). Those stay local to
/// the callers so the shared rule never has to know about them.
fn parent_branch_point(index: &TreeIndex, position: usize) -> String {
    match index.kind(position) {
        IndexKind::UserPrompt | IndexKind::TurnFailed | IndexKind::TurnCancelled => index
            .parent(position)
            .map(|parent| index.id(parent).to_event_id())
            .unwrap_or_default(),
        _ => index.id(position).to_event_id(),
    }
}

/// The placeholder a row shows until its label is hydrated. The picker
/// renders this instead of carrying the text on every skeleton row, where one
/// allocation per row would scale with the transcript.
#[must_use]
pub fn loading_label(kind: IndexKind) -> &'static str {
    match kind {
        IndexKind::UserPrompt => "user: loading\u{2026}",
        IndexKind::ToolResult => "tool: loading\u{2026}",
        IndexKind::TurnEnd => "agent: loading\u{2026}",
        IndexKind::TurnFailed => "agent: loading\u{2026} (failed)",
        IndexKind::TurnCancelled => "agent: loading\u{2026} (cancelled)",
        IndexKind::Compaction => "compact: loading\u{2026}",
        _ => "",
    }
}

fn skeleton_tree_row(index: &TreeIndex, position: usize) -> Option<TreeRowFields> {
    // Neither the label nor the revert target is read before hydration, and
    // confirming a row requires hydration first, so a skeleton stores no text.
    is_tree_node(index.kind(position)).then(|| (String::new(), String::new(), String::new()))
}

fn tool_result_tree_row(ctx: &TreeCtx, position: usize) -> TreeRowFields {
    let (name, tool_use_id, content, is_error) = load_tool_result(position, ctx.index, ctx.cursor);
    let marker = if is_error { "\u{2717} " } else { "" };
    let label = if name == "exec" {
        let mut tools = ctx.native_tools.borrow().get(&tool_use_id).cloned();
        if tools.is_none() && ctx.lazy_native_tools {
            let loaded = load_native_tools_for_turn(ctx, position, &tool_use_id);
            ctx.native_tools
                .borrow_mut()
                .insert(tool_use_id.clone(), loaded.clone());
            tools = Some(loaded);
        }
        if let Some(tools) = tools.filter(|tools| !tools.is_empty()) {
            let summary = tools
                .iter()
                .map(|(name, args)| format!("{name} {args}"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("exec: {marker}{}", one_line(&summary))
        } else {
            format!("exec: {marker}{}", one_line(&content))
        }
    } else {
        format!("tool: {marker}{name}: {}", one_line(&content))
    };
    (label, String::new(), ctx.index.id(position).to_event_id())
}

fn compaction_tree_row(ctx: &TreeCtx, position: usize) -> TreeRowFields {
    let (summarized, kept, _, _) = load_compaction_details(ctx.cursor, ctx.index.offset(position));
    // The index already resolved which event this checkpoint keeps; looking
    // it up here would rescan the whole tree once per visible marker.
    let checkpoint = ctx
        .index
        .checkpoints()
        .iter()
        .find(|checkpoint| checkpoint.marker as usize == position);
    let kept_parent = checkpoint
        .filter(|checkpoint| checkpoint.checkpointed_tail && checkpoint.names_first_kept)
        .and_then(|checkpoint| checkpoint.first_kept);
    let branch_point = if checkpoint
        .is_some_and(|checkpoint| checkpoint.checkpointed_tail && checkpoint.names_first_kept)
    {
        // A hovered checkpoint reverts to the parent of the event it keeps. An
        // unresolvable kept event leaves no revert target, as before.
        kept_parent
            .and_then(|kept| ctx.index.parent(kept as usize))
            .map(|parent| ctx.index.id(parent).to_event_id())
            .unwrap_or_default()
    } else {
        ctx.index
            .parent(position)
            .map(|parent| ctx.index.id(parent).to_event_id())
            .unwrap_or_default()
    };
    (
        format!("compact: Compacted {summarized} messages \u{00b7} kept {kept}"),
        String::new(),
        branch_point,
    )
}

fn hydrated_tree_row(ctx: &TreeCtx, position: usize) -> Option<TreeRowFields> {
    let index = ctx.index;
    match index.kind(position) {
        IndexKind::UserPrompt => {
            let prompt = load_prompt_text(ctx.cursor, index.offset(position));
            let prefill = if ctx.retain_prefill {
                prompt.clone()
            } else {
                String::new()
            };
            Some((
                format!("user: {}", one_line(&prompt)),
                prefill,
                parent_branch_point(index, position),
            ))
        }
        IndexKind::ToolResult => Some(tool_result_tree_row(ctx, position)),
        IndexKind::TurnEnd => {
            let preview = load_assistant_preview(position, index, ctx.cursor);
            let preview = if preview.is_empty() {
                "(turn end)".to_string()
            } else {
                preview
            };
            Some((
                format!("agent: {preview}"),
                String::new(),
                index.id(position).to_event_id(),
            ))
        }
        IndexKind::TurnFailed => Some((
            format!(
                "agent: {} (failed)",
                one_line(&load_failed_error(ctx.cursor, index.offset(position)))
            ),
            String::new(),
            parent_branch_point(index, position),
        )),
        IndexKind::TurnCancelled => {
            let preview = load_assistant_preview(position, index, ctx.cursor);
            let preview = if preview.is_empty() {
                "(cancelled)".to_string()
            } else {
                format!("{preview} (cancelled)")
            };
            Some((
                format!("agent: {preview}"),
                String::new(),
                parent_branch_point(index, position),
            ))
        }
        IndexKind::Compaction => Some(compaction_tree_row(ctx, position)),
        _ => None,
    }
}

fn push_tree_row(
    ctx: &TreeCtx,
    position: usize,
    prefix: Arc<str>,
    is_active: bool,
    out: &mut Vec<TreeRow>,
) {
    if ctx
        .hydrate_only
        .is_some_and(|sources| !sources.contains(&position))
    {
        return;
    }
    let fields = if ctx.hydrate {
        hydrated_tree_row(ctx, position)
    } else {
        skeleton_tree_row(ctx.index, position)
    };
    let Some((label, prefill, branch_point)) = fields else {
        return;
    };
    out.push(TreeRow {
        prefix,
        label: label.into_boxed_str(),
        prefill: prefill.into_boxed_str(),
        branch_point: branch_point.into_boxed_str(),
        is_active,
        source_index: position as u32,
        source_offset: ctx.index.offset(position),
        source_kind: ctx.index.kind(position),
        hydrated: ctx.hydrate,
    });
}

#[must_use]
pub fn is_tree_node(kind: IndexKind) -> bool {
    matches!(
        kind,
        IndexKind::UserPrompt
            | IndexKind::ToolResult
            | IndexKind::TurnEnd
            | IndexKind::TurnFailed
            | IndexKind::TurnCancelled
            | IndexKind::Compaction
    )
}

#[must_use]
pub fn find_turn_outcome(start: usize, index: &TreeIndex) -> Option<usize> {
    let mut current = start;
    let mut visited = HashSet::new();
    loop {
        if !visited.insert(current) {
            return None;
        }
        match index.kind(current) {
            IndexKind::TurnEnd | IndexKind::TurnFailed | IndexKind::TurnCancelled => {
                return Some(current);
            }
            _ => {}
        }
        current = index
            .children(current)
            .iter()
            .copied()
            .find(|&child| index.kind(child as usize) != IndexKind::UserPrompt)?
            as usize;
    }
}

#[must_use]
pub fn load_assistant_preview(
    turn_end: usize,
    index: &TreeIndex,
    cursor: &SessionCursor,
) -> String {
    let mut current = turn_end;
    let mut visited = HashSet::new();
    while let Some(parent) = index.parent(current) {
        if !visited.insert(current) {
            break;
        }
        match index.kind(parent) {
            IndexKind::UserPrompt => break,
            IndexKind::AssistantMessage => {
                if let Some(text) = load_assistant_text(cursor, index.offset(parent)) {
                    return one_line(&text);
                }
            }
            _ => {}
        }
        current = parent;
    }
    String::new()
}

#[must_use]
pub fn load_prompt_text(cursor: &SessionCursor, offset: u64) -> String {
    let Ok(ev) = cursor.display_event_at(offset) else {
        return String::new();
    };
    if let SessionEventKind::Message(message) = ev.kind {
        if message.role == Role::User {
            return message
                .blocks
                .iter()
                .find_map(|block| match block {
                    ContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .unwrap_or_default();
        }
    }
    String::new()
}

#[must_use]
pub fn load_tool_result(
    position: usize,
    index: &TreeIndex,
    cursor: &SessionCursor,
) -> (String, String, String, bool) {
    let Ok(ev) = cursor.display_event_at(index.offset(position)) else {
        return (String::new(), String::new(), String::new(), false);
    };
    let SessionEventKind::Message(message) = ev.kind else {
        return (String::new(), String::new(), String::new(), false);
    };
    let Some(block) = message.blocks.iter().find_map(|block| match block {
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
            ..
        } => Some((tool_use_id.clone(), content.clone(), *is_error)),
        _ => None,
    }) else {
        return (String::new(), String::new(), String::new(), false);
    };
    let (tool_use_id, content, is_error) = block;
    let name = index
        .parent(position)
        .and_then(|parent| {
            let parent_event = cursor.display_event_at(index.offset(parent)).ok()?;
            let SessionEventKind::Message(message) = parent_event.kind else {
                return None;
            };
            message.blocks.iter().find_map(|block| match block {
                ContentBlock::ToolUse { id, name, .. } if id == &tool_use_id => Some(name.clone()),
                _ => None,
            })
        })
        .unwrap_or_else(|| "?".to_string());
    (name, tool_use_id, content, is_error)
}

#[must_use]
pub fn load_assistant_text(cursor: &SessionCursor, offset: u64) -> Option<String> {
    let ev = cursor.display_event_at(offset).ok()?;
    let SessionEventKind::Message(message) = ev.kind else {
        return None;
    };
    if message.role != Role::Assistant {
        return None;
    }
    message.blocks.iter().find_map(|block| match block {
        ContentBlock::Text { text } => Some(text.clone()),
        _ => None,
    })
}

#[must_use]
pub fn load_failed_error(cursor: &SessionCursor, offset: u64) -> String {
    let Ok(ev) = cursor.display_event_at(offset) else {
        return String::new();
    };
    if let SessionEventKind::TurnFailed { error, .. } = ev.kind {
        error
    } else {
        String::new()
    }
}

fn load_compaction_details(cursor: &SessionCursor, offset: u64) -> (usize, usize, bool, String) {
    cursor.compaction_details_at(offset)
}

#[must_use]
pub fn one_line(s: &str) -> String {
    const MAX: usize = 60;
    let collapsed = s.replace('\n', " \u{23ce} ");
    if collapsed.chars().count() <= MAX {
        collapsed
    } else {
        let mut out: String = collapsed.chars().take(MAX).collect();
        out.push('\u{2026}');
        out
    }
}
