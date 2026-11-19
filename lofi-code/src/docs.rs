use std::collections::HashMap;
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DocFileEffect {
    Create,
    Modify,
    Read,
}

pub(crate) struct DocEntry {
    pub(crate) name: String,
    pub(crate) header: String,
    body: String,
    pub(crate) summary: String,
    returns: Option<String>,
    name_lo: String,
    header_lo: String,
    body_lo: String,
    pub(crate) effect: Option<DocFileEffect>,
}

struct ParsedDocs {
    entries: Vec<DocEntry>,
    by_lower: HashMap<String, usize>,
    by_short: HashMap<String, usize>,
    index: serde_json::Value,
}

pub const DOCS_MD: &str = include_str!("docs/api.md");

static PARSED_DOCS: LazyLock<ParsedDocs> = LazyLock::new(|| {
    let entries = parse_entries();
    let mut by_lower = HashMap::with_capacity(entries.len());
    let mut by_short = HashMap::with_capacity(entries.len());
    let mut arr = Vec::with_capacity(entries.len());
    for (idx, e) in entries.iter().enumerate() {
        by_lower.entry(e.name_lo.clone()).or_insert(idx);
        by_short.entry(e.name.clone()).or_insert(idx);
        if let Some((_, suffix)) = e.name.rsplit_once('.') {
            by_short.entry(suffix.to_string()).or_insert(idx);
        }
        let mut entry = serde_json::json!({
            "name": e.name,
            "summary": e.summary,
        });
        if let Some(returns) = &e.returns {
            entry["returns"] = serde_json::json!(returns);
        }
        arr.push(entry);
    }
    let index = serde_json::json!({ "ok": true, "entries": arr });
    ParsedDocs {
        entries,
        by_lower,
        by_short,
        index,
    }
});

fn classify_effect(summary: &str) -> Option<DocFileEffect> {
    let lower = summary.to_ascii_lowercase();
    if lower.contains("write") || lower.contains("create") {
        Some(DocFileEffect::Create)
    } else if lower.contains("edit") || lower.contains("replace") {
        Some(DocFileEffect::Modify)
    } else if lower.contains("read") || lower.contains("list") || lower.contains("search") {
        Some(DocFileEffect::Read)
    } else {
        None
    }
}

fn parse_entries() -> Vec<DocEntry> {
    let mut entries = Vec::new();
    let mut current_name = String::new();
    let mut current_header = String::new();
    let mut current_body = String::new();
    let mut in_entry = false;

    for line in DOCS_MD.lines() {
        if let Some(rest) = line.strip_prefix("## ") {
            if in_entry {
                entries.push(finish_entry(
                    std::mem::take(&mut current_name),
                    std::mem::take(&mut current_header),
                    std::mem::take(&mut current_body),
                ));
            }
            current_header = rest.to_string();
            current_name = extract_name(rest);
            current_body.clear();
            in_entry = true;
        } else if in_entry {
            current_body.push_str(line);
            current_body.push('\n');
        }
    }
    if in_entry {
        entries.push(finish_entry(current_name, current_header, current_body));
    }
    entries
}

fn finish_entry(name: String, header: String, body: String) -> DocEntry {
    let summary = body
        .lines()
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .unwrap_or("")
        .to_string();
    let returns = returns_line(&body);
    let name_lo = name.to_ascii_lowercase();
    let header_lo = header.to_ascii_lowercase();
    let body_lo = body.to_ascii_lowercase();
    let effect = classify_effect(&summary);
    DocEntry {
        name,
        header,
        body,
        summary,
        returns,
        name_lo,
        header_lo,
        body_lo,
        effect,
    }
}

/// The first Returns paragraph of an entry, joined to one line because the
/// shape often wraps mid-list. The capture ends at the shape's closing
/// backtick: the explanation after it, and any later Returns variant,
/// belong to the full entry read.
fn returns_line(body: &str) -> Option<String> {
    let mut paragraph: Option<String> = None;
    for line in body.lines().map(str::trim) {
        if let Some(out) = &mut paragraph {
            if line.is_empty() {
                break;
            }
            out.push(' ');
            out.push_str(line);
        } else if line.starts_with("**Returns") {
            let Some(pos) = line.find(":**") else {
                continue;
            };
            let tail = line[pos + 3..].trim();
            if !tail.is_empty() {
                paragraph = Some(tail.to_string());
            }
        }
    }
    let mut out = paragraph?;
    let mut backticks = 0;
    for (idx, ch) in out.char_indices() {
        if ch == '`' {
            backticks += 1;
            if backticks == 2 {
                out.truncate(idx + 1);
                break;
            }
        }
    }
    if out.chars().count() > 200 {
        out = out.chars().take(200).collect::<String>() + "…";
    }
    Some(out)
}

fn extract_name(header: &str) -> String {
    if let Some(paren) = header.find('(') {
        header[..paren].trim().to_string()
    } else {
        header.trim().to_string()
    }
}

pub(crate) fn find_tool_entry(tool_name: &str) -> Option<&'static DocEntry> {
    let docs = &*PARSED_DOCS;
    docs.by_short
        .get(tool_name)
        .and_then(|&idx| docs.entries.get(idx))
}

/// Exposed so other modules (e.g. the compaction hook) can use the docs
/// registry as the single source of truth for tool names and descriptions
/// instead of maintaining a parallel hard-coded table.
#[must_use]
pub fn entries() -> Vec<(String, String, String)> {
    PARSED_DOCS
        .entries
        .iter()
        .map(|e| (e.name.clone(), e.header.clone(), e.summary.clone()))
        .collect()
}

#[must_use]
pub fn docs_index() -> serde_json::Value {
    PARSED_DOCS.index.clone()
}

#[must_use]
pub fn docs_entry(name: &str) -> serde_json::Value {
    let docs = &*PARSED_DOCS;
    let needle = name.to_ascii_lowercase();
    if let Some(&idx) = docs.by_lower.get(&needle) {
        let e = &docs.entries[idx];
        let content = format!("## {}\n\n{}", e.header, e.body);
        return serde_json::json!({
            "ok": true,
            "name": e.name,
            "content": content,
        });
    }
    serde_json::json!({
        "ok": false,
        "error": format!("no doc entry named '{name}'"),
    })
}

#[must_use]
pub fn docs_search(query: &str) -> serde_json::Value {
    let entries = &PARSED_DOCS.entries;
    let terms: Vec<String> = query
        .split_whitespace()
        .map(str::to_ascii_lowercase)
        .collect();
    if terms.is_empty() {
        return serde_json::json!({ "ok": true, "results": [] });
    }

    let mut scored: Vec<(usize, &DocEntry)> = Vec::new();
    for e in entries {
        let mut score = 0usize;
        for term in &terms {
            if e.name_lo.contains(term) {
                score += 3;
            }
            if e.header_lo.contains(term) {
                score += 2;
            }
            if e.body_lo.contains(term) {
                score += 1;
            }
        }
        if score > 0 {
            scored.push((score, e));
        }
    }
    scored.sort_by_key(|&(score, _)| std::cmp::Reverse(score));

    let results: Vec<serde_json::Value> = scored
        .iter()
        .take(10)
        .map(|(score, e)| {
            // The sentence that answers the query usually sits past a
            // generic opening line, so rank lines by term hits.
            let excerpt = e
                .body
                .lines()
                .filter_map(|l| {
                    let lo = l.to_ascii_lowercase();
                    let hits = terms.iter().filter(|t| lo.contains(t.as_str())).count();
                    (hits > 0).then_some((hits, l))
                })
                .max_by_key(|(hits, _)| *hits)
                .map_or(e.summary.as_str(), |(_, l)| l)
                .trim()
                .to_string();
            serde_json::json!({
                "name": e.name,
                "score": score,
                "excerpt": excerpt,
            })
        })
        .collect();
    serde_json::json!({ "ok": true, "results": results })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn index_has_entries() {
        let idx = docs_index();
        let entries = idx["entries"].as_array().unwrap();
        assert!(!entries.is_empty(), "index should not be empty");
        let names: Vec<&str> = entries
            .iter()
            .map(|e| e["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"lofi.read"));
        assert!(names.contains(&"lofi.bash"));
    }

    #[test]
    fn entry_returns_full_text() {
        let entry = docs_entry("lofi.read");
        assert_eq!(entry["ok"], true);
        assert_eq!(entry["name"], "lofi.read");
        let content = entry["content"].as_str().unwrap();
        assert!(content.contains("path"));
        assert!(content.contains("offset"));
    }

    #[test]
    fn entry_case_insensitive() {
        let entry = docs_entry("LOFI.READ");
        assert_eq!(entry["ok"], true);
    }

    #[test]
    fn entry_not_found() {
        let entry = docs_entry("lofi.nonexistent");
        assert_eq!(entry["ok"], false);
        assert!(entry["error"].as_str().unwrap().contains("nonexistent"));
    }

    #[test]
    fn search_finds_by_name() {
        let res = docs_search("bash");
        let results = res["results"].as_array().unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0]["name"], "lofi.bash");
    }

    #[test]
    fn search_excerpt_picks_the_most_matching_line() {
        let res = docs_search("wait for background job completion");
        let results = res["results"].as_array().unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0]["name"].as_str().unwrap(), "lofi.jobSpawn");
        let excerpt = results[0]["excerpt"].as_str().unwrap();
        assert!(
            excerpt.contains("completion notification"),
            "got: {excerpt}"
        );
    }

    #[test]
    fn search_finds_by_concept() {
        let res = docs_search("truncated filtering");
        let results = res["results"].as_array().unwrap();
        assert!(!results.is_empty());
        let names: Vec<&str> = results
            .iter()
            .map(|r| r["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"Truncated results and filtering"));
    }

    #[test]
    fn search_empty_query_returns_empty() {
        let res = docs_search("");
        assert_eq!(res["results"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn search_multi_term_scores_higher() {
        let res = docs_search("read file");
        let results = res["results"].as_array().unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0]["name"], "lofi.read");
    }

    #[test]
    fn search_limited_to_ten() {
        let res = docs_search("lofi");
        let results = res["results"].as_array().unwrap();
        assert!(results.len() <= 10);
    }
}
