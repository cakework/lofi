use lofi_error::Result;
use lofi_types::{ContentBlock, Message, NativeToolRecord, Role, SessionEventKind};
use std::collections::{HashMap, HashSet};

use super::{
    filter_stopwords, format_recall_output, format_search_output, looks_like_regex,
    message_match_count, message_matches, message_snippet, message_word_count, render_message,
    safe_regex, store, CompactionTarget, RecallEntry, RecallOutcome, RecallRequest, RecallScope,
    SearchHit, DEFAULT_RECENT, MAX_PAGES, MAX_SEARCH_RESULTS, PAGE_SIZE,
};

#[derive(Default)]
struct FileScope {
    allowed: Option<HashSet<u32>>,
}

#[must_use]
pub fn recall_cursor(cursor: &store::SessionCursor, req: &RecallRequest) -> RecallOutcome {
    recall_cursor_inner(cursor, req).unwrap_or_else(|_| RecallOutcome {
        text: "recall: session file unreadable.".to_string(),
        status: "error".to_string(),
    })
}

fn recall_cursor_inner(
    cursor: &store::SessionCursor,
    req: &RecallRequest,
) -> Result<RecallOutcome> {
    let snap = cursor.tree_index()?;
    let index = &snap.index;
    let scope = resolve_scope(cursor, index, snap.leaf_pos, &req.scope)?;

    let mut selected: Vec<(u64, usize)> = Vec::new();
    let mut native_offsets: Vec<u64> = Vec::new();
    let mut global = 0usize;
    for pos in 0..index.len() {
        let allowed = scope
            .allowed
            .as_ref()
            .is_none_or(|set| set.contains(&(pos as u32)));
        let kind = index.kind(pos);
        if is_message(kind) {
            if allowed {
                selected.push((index.offset(pos), global));
            }
            global += 1;
        } else if allowed && kind == store::IndexKind::NativeTool {
            native_offsets.push(index.offset(pos));
        }
    }

    let query = req.query.as_deref().unwrap_or("").trim();
    if query.is_empty() {
        if req.expand.is_empty() {
            return recent(cursor, &selected, &native_offsets, req);
        }
        return expand(cursor, &selected, &native_offsets, req);
    }

    search(cursor, &selected, &native_offsets, query, req)
}

fn messages_need_natives(messages: &[Message]) -> bool {
    messages.iter().any(|m| {
        m.role == Role::Assistant
            && m.blocks
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
    })
}

fn load_native_records(
    cursor: &store::SessionCursor,
    native_offsets: &[u64],
    messages: &[Message],
) -> Result<Vec<NativeToolRecord>> {
    if native_offsets.is_empty() || !messages_need_natives(messages) {
        return Ok(Vec::new());
    }
    Ok(cursor
        .native_tool_summaries(native_offsets)?
        .into_iter()
        .map(|(parent, name, args)| NativeToolRecord {
            parent,
            call_id: 0,
            name,
            args,
            result: String::new(),
            is_error: false,
        })
        .collect())
}

fn index_native_by_parent(records: &[NativeToolRecord]) -> HashMap<String, Vec<&NativeToolRecord>> {
    let mut native_by_parent: HashMap<String, Vec<&NativeToolRecord>> = HashMap::new();
    for record in records {
        native_by_parent
            .entry(record.parent.clone())
            .or_default()
            .push(record);
    }
    native_by_parent
}

fn recent(
    cursor: &store::SessionCursor,
    selected: &[(u64, usize)],
    native_offsets: &[u64],
    req: &RecallRequest,
) -> Result<RecallOutcome> {
    let slice = &selected[selected.len().saturating_sub(DEFAULT_RECENT)..];
    let offsets: Vec<u64> = slice.iter().map(|(offset, _)| *offset).collect();
    let events = if offsets.is_empty() {
        Vec::new()
    } else {
        cursor.display_events_at(&offsets)?
    };
    let messages: Vec<(usize, Message)> = slice
        .iter()
        .zip(events)
        .filter_map(|((_, global), event)| match event.kind {
            SessionEventKind::Message(message) => Some((*global, message)),
            _ => None,
        })
        .collect();
    let only_msgs: Vec<Message> = messages.iter().map(|(_, m)| m.clone()).collect();
    let native_records = load_native_records(cursor, native_offsets, &only_msgs)?;
    let native_by_parent = index_native_by_parent(&native_records);
    let entries: Vec<RecallEntry> = messages
        .iter()
        .map(|(global, msg)| render_message(msg, *global, false, &native_by_parent))
        .collect();
    Ok(RecallOutcome {
        text: format_recall_output(
            &entries,
            None,
            matches!(req.scope, RecallScope::All).then(|| "Scope: all".to_string()),
        ),
        status: format!("{} entries", entries.len()),
    })
}

fn expand(
    cursor: &store::SessionCursor,
    selected: &[(u64, usize)],
    native_offsets: &[u64],
    req: &RecallRequest,
) -> Result<RecallOutcome> {
    let by_global: HashMap<usize, u64> = selected
        .iter()
        .map(|&(offset, global)| (global, offset))
        .collect();
    let invalid: Vec<usize> = req
        .expand
        .iter()
        .copied()
        .filter(|i| !by_global.contains_key(i))
        .collect();
    if !invalid.is_empty() {
        return Ok(RecallOutcome {
            text: format!(
                "Cannot expand indices outside selected scope: {}",
                invalid
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            status: format!("{} invalid", invalid.len()),
        });
    }
    let mut wanted_pairs: Vec<(u64, usize)> = req
        .expand
        .iter()
        .filter_map(|&global| by_global.get(&global).map(|&offset| (offset, global)))
        .collect();
    wanted_pairs.sort_unstable_by_key(|&(offset, _)| offset);
    wanted_pairs.dedup_by_key(|(offset, _)| *offset);
    let wanted_offsets: Vec<u64> = wanted_pairs.iter().map(|(offset, _)| *offset).collect();
    let events = if wanted_offsets.is_empty() {
        Vec::new()
    } else {
        cursor.display_events_at(&wanted_offsets)?
    };
    let mut messages_by_global: HashMap<usize, Message> = HashMap::with_capacity(events.len());
    let mut raw_msgs = Vec::with_capacity(events.len());
    for ((_, global), event) in wanted_pairs.iter().zip(events) {
        if let SessionEventKind::Message(message) = event.kind {
            raw_msgs.push(message.clone());
            messages_by_global.insert(*global, message);
        }
    }
    let native_records = load_native_records(cursor, native_offsets, &raw_msgs)?;
    let native_by_parent = index_native_by_parent(&native_records);
    let ordered: Vec<RecallEntry> = req
        .expand
        .iter()
        .filter_map(|global| {
            messages_by_global
                .get(global)
                .map(|msg| render_message(msg, *global, true, &native_by_parent))
        })
        .collect();
    Ok(RecallOutcome {
        text: format_recall_output(&ordered, None, None),
        status: format!("expanded {}", ordered.len()),
    })
}

struct SearchResults {
    matched: Vec<(usize, usize)>,
    snippet_re: Option<regex::Regex>,
}

struct Bm25Candidate {
    position: usize,
    dl: usize,
    tfs: Vec<(usize, usize)>,
}

fn find_matches(
    cursor: &store::SessionCursor,
    offsets: &[u64],
    raw_query: &str,
) -> Result<SearchResults> {
    if offsets.is_empty() {
        return Ok(SearchResults {
            matched: Vec::new(),
            snippet_re: None,
        });
    }
    if looks_like_regex(raw_query) {
        let Some(pattern) = safe_regex(raw_query) else {
            return Ok(SearchResults {
                matched: Vec::new(),
                snippet_re: None,
            });
        };
        let mut matched = Vec::new();
        let mut position = 0usize;
        cursor.visit_display_events(offsets, |event| {
            if let SessionEventKind::Message(message) = event.kind {
                if matched.len() < MAX_SEARCH_RESULTS
                    && message_matches(&message, |text| pattern.is_match(text))
                {
                    matched.push((position, 1));
                }
                position += 1;
            }
            Ok(())
        })?;
        return Ok(SearchResults {
            matched,
            snippet_re: Some(pattern),
        });
    }
    find_bm25_matches(cursor, offsets, raw_query)
}

fn find_bm25_matches(
    cursor: &store::SessionCursor,
    offsets: &[u64],
    raw_query: &str,
) -> Result<SearchResults> {
    let raw_terms: Vec<&str> = raw_query.split_whitespace().collect();
    let terms = filter_stopwords(&raw_terms);
    let patterns: Vec<regex::Regex> = terms
        .iter()
        .filter_map(|term| regex::Regex::new(&regex::escape(term)).ok())
        .collect();
    if patterns.is_empty() {
        return Ok(SearchResults {
            matched: Vec::new(),
            snippet_re: None,
        });
    }

    let min_match = if patterns.len() >= 3 { 2 } else { 1 };
    let mut n = 0usize;
    let mut total_words = 0usize;
    let mut df = vec![0usize; patterns.len()];
    let mut candidates: Vec<Bm25Candidate> = Vec::new();
    cursor.visit_display_events(offsets, |event| {
        if let SessionEventKind::Message(message) = event.kind {
            let dl = message_word_count(&message);
            total_words += dl;
            let mut tfs = Vec::new();
            for (term_idx, pattern) in patterns.iter().enumerate() {
                let tf = message_match_count(&message, pattern);
                if tf > 0 {
                    df[term_idx] += 1;
                    tfs.push((term_idx, tf));
                }
            }
            if tfs.len() >= min_match {
                candidates.push(Bm25Candidate {
                    position: n,
                    dl,
                    tfs,
                });
            }
            n += 1;
        }
        Ok(())
    })?;

    let avg_dl = total_words as f64 / n.max(1) as f64;
    let mut scored: Vec<(f64, usize, usize)> = candidates
        .into_iter()
        .map(|cand| {
            let match_count = cand.tfs.len();
            let dl = cand.dl as f64;
            let mut score = 0.0;
            for (term_index, tf) in cand.tfs {
                let idf = (((n - df[term_index]) as f64 + 0.5) / (df[term_index] as f64 + 0.5)
                    + 1.0)
                    .ln();
                let k = 1.2;
                let b = 0.75;
                let tfn = (tf as f64 * (k + 1.0))
                    / (tf as f64 + k * (1.0 - b + b * dl / avg_dl.max(1.0)));
                score += idf * tfn;
            }
            (score, cand.position, match_count)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    if scored.len() > 1 && patterns.len() >= 2 {
        let top = scored[0].0;
        if top > 0.0 {
            let threshold = top * 0.1;
            if let Some(cut) = scored.iter().position(|(score, _, _)| *score < threshold) {
                scored.truncate(cut);
            }
        }
    }
    scored.truncate(MAX_SEARCH_RESULTS.min(scored.len()));
    let snippet_re = regex::Regex::new(
        &patterns
            .iter()
            .map(regex::Regex::as_str)
            .collect::<Vec<_>>()
            .join("|"),
    )
    .ok();
    Ok(SearchResults {
        matched: scored
            .into_iter()
            .map(|(_, pos, count)| (pos, count))
            .collect(),
        snippet_re,
    })
}

fn search(
    cursor: &store::SessionCursor,
    selected: &[(u64, usize)],
    native_offsets: &[u64],
    query: &str,
    req: &RecallRequest,
) -> Result<RecallOutcome> {
    let offsets: Vec<u64> = selected.iter().map(|(offset, _)| *offset).collect();
    let results = find_matches(cursor, &offsets, query.trim())?;
    let total = results.matched.len();
    if total == 0 {
        return Ok(RecallOutcome {
            text: format!("No matches for \"{query}\" in selected scope."),
            status: "0 matches".to_string(),
        });
    }
    let total_pages = total.div_ceil(PAGE_SIZE).max(1);
    let page = req.page.max(1);
    if page > MAX_PAGES.min(total_pages) {
        return Ok(RecallOutcome {
            text: format!("Too many results to page through ({total} matches across {total_pages} pages). Try a more specific query or scope."),
            status: format!("{total} matches"),
        });
    }
    let start = (page - 1) * PAGE_SIZE;
    let page_slice = &results.matched[start..(start + PAGE_SIZE).min(total)];

    let mut page_offsets: Vec<u64> = page_slice.iter().map(|&(pos, _)| selected[pos].0).collect();
    page_offsets.sort_unstable();
    page_offsets.dedup();
    let events = cursor.display_events_at(&page_offsets)?;
    let mut by_offset: HashMap<u64, Message> = HashMap::with_capacity(events.len());
    let mut raw_msgs = Vec::with_capacity(events.len());
    for (&offset, event) in page_offsets.iter().zip(events) {
        if let SessionEventKind::Message(message) = event.kind {
            raw_msgs.push(message.clone());
            by_offset.insert(offset, message);
        }
    }
    let native_records = load_native_records(cursor, native_offsets, &raw_msgs)?;
    let native_by_parent = index_native_by_parent(&native_records);

    let mut hits = Vec::with_capacity(page_slice.len());
    for &(pos, match_count) in page_slice {
        let (offset, global) = selected[pos];
        let Some(message) = by_offset.get(&offset) else {
            continue;
        };
        let is_expanded = req.expand.contains(&global);
        let entry = render_message(message, global, is_expanded, &native_by_parent);
        let snippet = if is_expanded {
            Some(entry.summary.clone())
        } else {
            results
                .snippet_re
                .as_ref()
                .and_then(|re| message_snippet(message, re))
        };
        hits.push(SearchHit {
            entry,
            snippet,
            match_count,
        });
    }

    let header = if total_pages > 1 {
        format!("Page {page}/{total_pages} ({total} total matches)")
    } else {
        format!("{total} matches")
    };
    Ok(RecallOutcome {
        text: format_search_output(&hits, query, &header),
        status: format!("{total} matches"),
    })
}

fn is_message(kind: store::IndexKind) -> bool {
    matches!(
        kind,
        store::IndexKind::UserPrompt
            | store::IndexKind::AssistantMessage
            | store::IndexKind::ToolResult
            | store::IndexKind::SystemMessage
    )
}

fn resolve_scope(
    cursor: &store::SessionCursor,
    index: &store::TreeIndex,
    leaf_pos: Option<u32>,
    scope: &RecallScope,
) -> Result<FileScope> {
    if matches!(scope, RecallScope::All) {
        return Ok(FileScope::default());
    }
    let lineage = index.active_path(leaf_pos)?;
    if matches!(scope, RecallScope::Lineage) {
        return Ok(FileScope {
            allowed: Some(lineage.into_iter().collect()),
        });
    }
    let compactions: Vec<usize> = lineage
        .iter()
        .map(|&p| p as usize)
        .filter(|&p| index.kind(p) == store::IndexKind::Compaction)
        .collect();
    let RecallScope::Compaction(target) = scope else {
        return Ok(FileScope::default());
    };
    let selected = match target {
        CompactionTarget::Latest => compactions.last().copied(),
        CompactionTarget::Index(n) => compactions.get(*n).copied(),
    };
    let Some(selected) = selected else {
        return Ok(FileScope {
            allowed: Some(lineage.into_iter().collect()),
        });
    };
    let marker = cursor.display_event_at(index.offset(selected))?;
    let SessionEventKind::Compaction {
        summarized_range, ..
    } = marker.kind
    else {
        return Ok(FileScope {
            allowed: Some(HashSet::new()),
        });
    };
    let lineage_ids: HashSet<&store::IndexId> =
        lineage.iter().map(|&p| index.id(p as usize)).collect();
    let first = (0..index.len()).find(|&p| index.id(p).matches(&summarized_range[0]));
    let last = (0..index.len()).find(|&p| index.id(p).matches(&summarized_range[1]));
    let allowed = match (first, last) {
        (Some(first), Some(last)) if first <= last => (first..=last)
            .filter(|&p| lineage_ids.contains(index.id(p)))
            .map(|p| p as u32)
            .collect(),
        _ => HashSet::new(),
    };
    Ok(FileScope {
        allowed: Some(allowed),
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use std::io::Write;

    use lofi_types::{ContentBlock, Message, PromptKind, Role, SessionEvent};

    use super::*;

    fn transcript(events: &[SessionEvent]) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "{}",
            serde_json::json!({
                "type": "meta",
                "version": 1,
                "created": 0,
                "cwd": "/tmp",
                "model": {"provider":"p", "id":"m", "thinking":"off"}
            })
        )
        .unwrap();
        for event in events {
            writeln!(file, "{}", serde_json::to_string(event).unwrap()).unwrap();
        }
        file
    }

    fn message(id: &str, parent: Option<&str>, role: Role, text: &str) -> SessionEvent {
        SessionEvent {
            id: id.to_string(),
            parent_id: parent.map(str::to_string),
            kind: SessionEventKind::Message(Message {
                origin: None,
                role,
                blocks: vec![ContentBlock::Text {
                    text: text.to_string(),
                }],
                kind: PromptKind::default(),
            }),
        }
    }

    #[test]
    fn file_search_preserves_global_indices_and_scope() {
        let events = vec![
            message("a", None, Role::User, "root needle"),
            message("b", Some("a"), Role::Assistant, "active needle"),
            message("x", Some("a"), Role::Assistant, "branch needle"),
            message("c", Some("b"), Role::User, "leaf"),
        ];
        let file = transcript(&events);
        let cursor = store::SessionCursor::open(file.path().to_path_buf()).unwrap();
        let outcome = recall_cursor(
            &cursor,
            &RecallRequest {
                query: Some("needle".to_string()),
                scope: RecallScope::Lineage,
                ..Default::default()
            },
        );
        assert!(outcome.text.contains("#0 [user]"), "{}", outcome.text);
        assert!(outcome.text.contains("#1 [assistant]"), "{}", outcome.text);
        assert!(!outcome.text.contains("#2 [assistant]"), "{}", outcome.text);
    }

    #[test]
    fn file_expand_reads_full_selected_message() {
        let long = "x".repeat(500);
        let file = transcript(&[message("a", None, Role::User, &long)]);
        let cursor = store::SessionCursor::open(file.path().to_path_buf()).unwrap();
        let outcome = recall_cursor(
            &cursor,
            &RecallRequest {
                scope: RecallScope::All,
                expand: vec![0],
                ..Default::default()
            },
        );
        assert!(outcome.text.contains(&long));
        assert!(!outcome.text.contains('…'));
    }
}
