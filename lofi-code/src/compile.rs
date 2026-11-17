use swc_common::sync::Lrc;
use swc_common::{FileName, FilePathMapping, Globals, Mark, SourceMap, Span, Spanned, GLOBALS};
use swc_ecma_ast::{Module, Pass, Program};
use swc_ecma_codegen::to_code_default;
use swc_ecma_parser::{Parser, StringInput, Syntax, TsSyntax};
use swc_ecma_transforms_typescript::strip;

use lofi_error::{Error, Result};

use crate::convert::SANDBOX_ERROR_KEY;

fn span_loc(cm: &SourceMap, span: Span) -> String {
    let loc = cm.lookup_char_pos(span.lo());
    let line = loc.line.saturating_sub(1).max(1);
    let col = loc.col.0 + 1;
    format!("{line}:{col}")
}

/// # Errors
/// Returns a sandbox error when TypeScript parsing or transformation fails.
pub fn compile_ts(src: &str) -> Result<String> {
    let wrapped = format!(
        "(async () => {{ try {{\n{src}\n}} catch (e) {{ return {{ {SANDBOX_ERROR_KEY}: (e && e.message) ? e.message : String(e) }}; }} }})()"
    );

    let cm: Lrc<SourceMap> = Lrc::new(SourceMap::new(FilePathMapping::empty()));
    let fm = cm.new_source_file(FileName::Custom("lofi.ts".into()).into(), wrapped);

    let code = GLOBALS.set(&Globals::new(), || -> Result<String> {
        let mut parser = Parser::new(
            Syntax::Typescript(TsSyntax::default()),
            StringInput::from(&*fm),
            None,
        );
        let module: Module = parser.parse_module().map_err(|e| {
            Error::Sandbox(format!(
                "parse error: {} at {}",
                e.kind().msg(),
                span_loc(&cm, e.span())
            ))
        })?;
        if let Some(e) = parser.take_errors().into_iter().next() {
            return Err(Error::Sandbox(format!(
                "parse error: {} at {}",
                e.kind().msg(),
                span_loc(&cm, e.span())
            )));
        }
        let mut program = Program::Module(module);
        strip(Mark::new(), Mark::new()).process(&mut program);
        let js = to_code_default(cm, None, &program);
        Ok(js)
    })?;

    Ok(code)
}
