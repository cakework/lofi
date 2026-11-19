use super::analyze::analyze;
use super::emit::{emit_blank, emit_code, emit_heading, emit_paragraph, emit_quote, emit_table};
use super::height::md_block_height;
use super::model::MdBlock;
use super::plain::{plain_paragraph, render_plain_paragraph_window};
use crate::tui::theme::Theme;
use crate::tui::view::prim::RenderLine;
use ratatui::style::Style;
use ratatui::text::Span;

#[allow(clippy::too_many_arguments)]
pub(in crate::tui::view) fn render_markdown_body_window(
    text: &str,
    t: Theme,
    w: usize,
    content_w: usize,
    base_style: Style,
    lead_fn: impl Fn(usize) -> Vec<Span<'static>>,
    range: std::ops::Range<usize>,
) -> Vec<RenderLine> {
    if text.is_empty() || range.start >= range.end {
        return Vec::new();
    }
    if plain_paragraph(text) {
        return render_plain_paragraph_window(text, content_w, base_style, &lead_fn, range);
    }
    let mut out = Vec::with_capacity(range.end.saturating_sub(range.start).min(256));
    let mut row = 0usize;
    let mut block_buf = Vec::new();
    for block in analyze(text, t, base_style) {
        if row >= range.end {
            break;
        }
        let bh = md_block_height(text, &block, content_w);
        if row + bh <= range.start {
            row += bh;
            continue;
        }
        block_buf.clear();
        let block_start = row;
        emit_md_block(
            &mut block_buf,
            &mut row,
            text,
            block,
            t,
            w,
            content_w,
            base_style,
            &lead_fn,
        );
        let skip = range.start.saturating_sub(block_start);
        let take = range
            .end
            .saturating_sub(block_start.max(range.start))
            .min(block_buf.len().saturating_sub(skip));
        out.extend(block_buf.drain(..).skip(skip).take(take));
    }
    out
}

pub(in crate::tui::view) fn render_markdown_body(
    text: &str,
    t: Theme,
    w: usize,
    content_w: usize,
    base_style: Style,
    lead_fn: impl Fn(usize) -> Vec<Span<'static>>,
) -> Vec<RenderLine> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut row = 0usize;
    for block in analyze(text, t, base_style) {
        emit_md_block(
            &mut out, &mut row, text, block, t, w, content_w, base_style, &lead_fn,
        );
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn emit_md_block(
    out: &mut Vec<RenderLine>,
    row: &mut usize,
    text: &str,
    block: MdBlock,
    t: Theme,
    w: usize,
    content_w: usize,
    base_style: Style,
    lead_fn: &impl Fn(usize) -> Vec<Span<'static>>,
) {
    match block {
        MdBlock::Code { lang, lines, fence } => {
            emit_code(
                out,
                row,
                text,
                lang.as_deref(),
                &lines,
                &fence,
                t,
                w,
                content_w,
                lead_fn,
            );
        }
        MdBlock::Table {
            header,
            aligns,
            rows,
            src,
        } => {
            emit_table(
                out, row, text, &header, &aligns, &rows, &src, t, content_w, lead_fn,
            );
        }
        MdBlock::Heading {
            level,
            content_start,
            src,
            ..
        } => {
            emit_heading(
                out,
                row,
                text,
                level,
                content_start,
                &src,
                t,
                base_style,
                content_w,
                lead_fn,
            );
        }
        MdBlock::Quote { lines, .. } => {
            emit_quote(out, row, text, &lines, t, content_w, lead_fn);
        }
        MdBlock::Paragraph { spans, src } => {
            emit_paragraph(out, row, text, &spans, &src, content_w, lead_fn);
        }
        MdBlock::Blank { src } => {
            emit_blank(out, row, text, &src, lead_fn);
        }
    }
}
