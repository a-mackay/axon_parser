pub mod ast;
pub use ast::Indent;

use axon_parseast_parser::parse as ap_parse;
use std::convert::TryInto;
use thiserror::Error;

/// If the axon input represents a function, return a `ast::Func`.
pub fn parse_func(axon: &str) -> Result<ast::Func, Error> {
    let val = &ap_parse(axon).map_err(|_| Error::Parse)?;
    let func: Result<ast::Func, ()> = val.try_into();
    let func = func.map_err(|_| Error::AstConversion)?;
    Ok(func)
}

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
pub fn parse_func_to_formatted_lines(
    axon: &str,
    indent: &Indent,
) -> Result<Vec<String>, Error> {
    let func = parse_func(axon)?;
    let lines = func.to_lines(indent);
    let strings = lines.into_iter().map(|line| format!("{}", line)).collect();
    Ok(strings)
}
