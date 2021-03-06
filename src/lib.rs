pub mod ast;
pub use ast::Indent;

use axon_parseast_parser::parse as ap_parse;
use std::convert::TryInto;
use thiserror::Error;

/// If the axon input represents a function, return a `ast::Func`.
///
/// The `axon` argument to this function should be a string containing the
/// output of running `toAxonCode(parseAst( ... ))` in SkySpark.
///
/// # Example
/// ```rust
/// use axon_parser::parse_func;
///
/// let axon = r###"{type:"func", params:[], body:{type:"block", exprs:[{type:"literal", val:"hello world"}]}}"###;
/// let func = parse_func(axon).unwrap();
/// assert!(func.params.len() == 0)
/// ```
///
pub fn parse_func(axon: &str) -> Result<ast::Func, Error> {
    let val = &ap_parse(axon).map_err(|_| Error::Parse)?;
    let func: Result<ast::Func, ()> = val.try_into();
    let func = func.map_err(|_| Error::AstConversion)?;
    Ok(func)
}

/// Encapsulates the types of errors that can occur in this library.
#[derive(Debug, Error)]
pub enum Error {
    #[error("Could not parse the representation of Axon code")]
    Parse,
    #[error(
        "Could not parse the loosely-typed Axon Val into a strongly-typed AST"
    )]
    AstConversion,
}

/// If the axon input represents a function, parse it and return a `Vec<String>`,
/// each string being a formatted line of code.
///
/// The `axon` argument to this function should be a string containing the
/// output of running `toAxonCode(parseAst( ... ))` in SkySpark.
///
/// # Example
/// ```rust
/// use axon_parser::parse_func_to_formatted_lines;
/// use axon_parser::Indent;
///
/// let indent = Indent::new("  ".to_string(), 0); // 2-space indentation
///
/// let axon = r###"{type:"func", params:[], body:{type:"block", exprs:[{type:"literal", val:"hello world"}]}}"###;
/// let lines = parse_func_to_formatted_lines(axon, &indent).unwrap();
/// assert_eq!(lines.len(), 3);
/// assert_eq!(lines[0], "() => do");
/// assert_eq!(lines[1], "  \"hello world\"");
/// assert_eq!(lines[2], "end");
/// ```
///
pub fn parse_func_to_formatted_lines(
    axon: &str,
    indent: &Indent,
) -> Result<Vec<String>, Error> {
    let func = parse_func(axon)?;
    let lines = func.to_lines(indent);
    let strings = lines.into_iter().map(|line| format!("{}", line)).collect();
    Ok(strings)
}
