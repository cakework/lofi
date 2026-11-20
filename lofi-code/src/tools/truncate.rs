#[cfg(test)]
pub(crate) const DEFAULT_MAX_LINES: usize = 2000;
#[cfg(test)]
pub(crate) const DEFAULT_MAX_BYTES: usize = 50 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::struct_field_names)]
pub struct Truncated {
    pub content: String,
    pub truncated: bool,
    pub total_lines: usize,
    pub output_lines: usize,
    pub total_bytes: usize,
    pub output_bytes: usize,
}

#[cfg(test)]
#[must_use]
pub(crate) fn truncate_head(content: &str) -> Truncated {
    truncate_head_with(content, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES)
}

#[must_use]
pub(crate) fn truncate_head_with(content: &str, max_lines: usize, max_bytes: usize) -> Truncated {
    let total_bytes = content.len();
    let total_lines = content.split('\n').count();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return Truncated {
            content: content.to_string(),
            truncated: false,
            total_lines,
            output_lines: total_lines,
            total_bytes,
            output_bytes: total_bytes,
        };
    }

    let mut kept_lines = 0usize;
    let mut cut = 0usize;
    for (i, end) in content
        .match_indices('\n')
        .map(|(pos, _)| pos)
        .chain(std::iter::once(total_bytes))
        .enumerate()
    {
        if i >= max_lines || end > max_bytes {
            break;
        }
        kept_lines = i + 1;
        cut = end;
    }

    let content_out = if kept_lines == 0 {
        String::new()
    } else {
        content[..cut].to_string()
    };
    let output_bytes = content_out.len();
    Truncated {
        content: content_out,
        truncated: true,
        total_lines,
        output_lines: kept_lines,
        total_bytes,
        output_bytes,
    }
}

#[must_use]
pub fn truncate_tail_with(content: &str, max_lines: usize, max_bytes: usize) -> Truncated {
    let total_bytes = content.len();
    let total_lines = content.split('\n').count();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return Truncated {
            content: content.to_string(),
            truncated: false,
            total_lines,
            output_lines: total_lines,
            total_bytes,
            output_bytes: total_bytes,
        };
    }

    let mut kept_lines = 0usize;
    let mut cut = total_bytes;
    for (i, start) in content
        .rmatch_indices('\n')
        .map(|(nl, _)| nl + 1)
        .chain(std::iter::once(0))
        .enumerate()
    {
        if i >= max_lines || total_bytes - start > max_bytes {
            break;
        }
        kept_lines = i + 1;
        cut = start;
    }

    let content_out = if kept_lines == 0 {
        String::new()
    } else {
        content[cut..].to_string()
    };
    let output_bytes = content_out.len();
    Truncated {
        content: content_out,
        truncated: true,
        total_lines,
        output_lines: kept_lines,
        total_bytes,
        output_bytes,
    }
}

#[must_use]
pub(crate) fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_keeps_short_unchanged() {
        let t = truncate_head("a\nb\nc");
        assert!(!t.truncated);
        assert_eq!(t.content, "a\nb\nc");
        assert_eq!(t.output_lines, 3);
    }

    #[test]
    fn head_truncates_by_lines() {
        let big = (0..100)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let t = truncate_head_with(&big, 10, 1_000_000);
        assert!(t.truncated);
        assert_eq!(t.output_lines, 10);
        assert_eq!(t.total_lines, 100);
        assert!(t.content.starts_with("line0\nline1"));
    }

    #[test]
    fn head_truncates_by_bytes() {
        let big = (0..50)
            .map(|_| "x".repeat(100))
            .collect::<Vec<_>>()
            .join("\n");
        let t = truncate_head_with(&big, 1_000_000, 250);
        assert!(t.truncated);
        assert!(t.output_bytes <= 250);
        assert!(t.output_lines < 50);
    }

    #[test]
    fn tail_keeps_last_lines() {
        let big = (0..100)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let t = truncate_tail_with(&big, 10, 1_000_000);
        assert!(t.truncated);
        assert_eq!(t.output_lines, 10);
        assert!(t.content.contains("line99"));
        assert!(t.content.contains("line90"));
        assert!(!t.content.contains("line89"));
    }

    #[test]
    fn format_size_units() {
        assert_eq!(format_size(500), "500B");
        assert_eq!(format_size(1024), "1.0KB");
        assert_eq!(format_size(1024 * 1024), "1.0MB");
    }
}
