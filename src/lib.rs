pub mod ast;
pub mod fmt;

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
    #[error("Could not rewrite the AST")]
    Rewrite,
}

/// If the axon input represents a function, parse it and return a `String`
/// which contains formatted code.
///
/// The `axon` argument to this function should be a string containing the
/// output of running `<some function src>.parseAst.toAxonCode` in SkySpark.
///
/// # Example
/// ```rust
/// use axon_parser::format_func;
///
/// let desired_width = 80;
/// let axon = r###"{type:"func", params:[], body:{type:"block", exprs:[{type:"literal", val:"hello world"}]}}"###;
/// let formatted_code = format_func(axon, desired_width).unwrap();
/// let expected_code = "() => do\n    \"hello world\"\nend";
/// assert_eq!(formatted_code, expected_code);
/// ```
///
pub fn format_func(axon: &str, desired_width: usize) -> Result<String, Error> {
    let func = parse_func(axon)?;
    let context = fmt::Context::new(0, desired_width);
    let widen = true;
    let code = func.default_rewrite(context, widen);
    code.ok_or(Error::Rewrite)
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let code = include_str!("./test2.txt");
        let x = crate::format_func(code, 80).unwrap(); // todo fails at 107
        println!("{}", x);
    }
}
