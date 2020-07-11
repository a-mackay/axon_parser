use nom::{
    branch::alt,
    bytes::complete::{escaped, tag},
    character::complete::{alphanumeric0, none_of, one_of},
    combinator::{cond, opt, map},
    multi::{many0, many1},
    sequence::{delimited, separated_pair, terminated, preceded},
    IResult,
};

#[derive(Debug, Eq, PartialEq)]
pub struct TryCatchBlock {
    try_expr: Expr,
    caught_id: Option<Id>,
    catch_expr: Expr,
}

impl TryCatchBlock {
    fn new(try_expr: Expr, caught_id: Option<Id>, catch_expr: Expr) -> Self {
        Self {
            try_expr,
            caught_id,
            catch_expr,
        }
    }
}

fn parse_try_catch_expr(input: &str) -> IResult<&str, Expr> {
    let (input, try_catch) = parse_try_catch(input)?;
    Ok((input, Expr::new_try_catch(try_catch)))
}

fn parse_try_catch(input: &str) -> IResult<&str, TryCatchBlock> {
    let (input, _) = terminated(tag("try"), gap)(input)?;

    let (input, opt_try_do) = opt(terminated(tag("do"), gap))(input)?;
    let has_try_do_keyword = opt_try_do.is_some();

    let (input, try_exprs) = parse_exprs(input)?;

    // Remove any blank space, in the case there is any:
    let (input, _) = opt(gap)(input)?;

    let (input, opt_try_end) = opt(terminated(tag("end"), gap))(input)?;
    let has_try_end_keyword = opt_try_end.is_some();

    if (has_try_end_keyword && !has_try_do_keyword) {
        todo!();
    }

    // Remove any blank space, in the case there was no 'end' keyword:
    let (input, _) = opt(gap)(input)?;

    let (input, _) = terminated(tag("catch"), gap)(input)?;

    let open_paren = terminated(tag("("), gap);
    let close_paren = terminated(tag(")"), opt(gap));
    let parse_id_and_gap = terminated(parse_id, opt(gap));
    let (input, opt_id) = opt(delimited(open_paren, parse_id_and_gap, close_paren))(input)?;

    // Remove any blank space, in case there was any after the optional ')':
    let (input, _) = opt(gap)(input)?;

    let (input, opt_catch_do) = opt(terminated(tag("do"), gap))(input)?;
    let has_catch_do_keyword = opt_catch_do.is_some();

    let (input, catch_exprs) = parse_exprs(input)?;

    // Remove any blank space, in the case there is any:
    let (input, _) = opt(gap)(input)?;

    let (input, opt_catch_end) = opt(terminated(tag("end"), opt(gap)))(input)?;
    let has_catch_end_keyword = opt_catch_end.is_some();

    if (has_catch_end_keyword && !has_catch_do_keyword) {
        todo!();
    }

    let try_do_block = Expr::new_do_block(DoBlock::new(try_exprs));
    let catch_do_block = Expr::new_do_block(DoBlock::new(catch_exprs));
    let try_catch = TryCatchBlock::new(try_do_block, opt_id, catch_do_block);
    Ok((input, try_catch))
}

fn gap(input: &str) -> IResult<&str, ()> {
    let (input, _) = many1(alt((parse_multi_whitespace, parse_newline_and_whitespace)))(input)?;
    Ok((input, ()))
}

#[derive(Debug, Eq, PartialEq)]
pub struct DoBlock {
    exprs: Exprs,
}

impl DoBlock {
    fn new(exprs: Exprs) -> Self {
        Self { exprs }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum Expr {
    DoBlock(DoBlock),
    Def {
        var_name: Id,
        expr: Box<Expr>,
    },
    Return { expr: Box<Expr> },
    Throw { expr: Box<Expr> },
    TryCatch(Box<TryCatchBlock>),
    Literal(Literal),
}

impl Expr {
    fn new_do_block(do_block: DoBlock) -> Self {
        Self::DoBlock(do_block)
    }

    fn new_try_catch(try_catch: TryCatchBlock) -> Self {
        Self::TryCatch(Box::new(try_catch))
    }
}

#[derive(Debug, Eq, PartialEq)]
struct Exprs {
    exprs: Vec<Expr>,
}

impl Exprs {
    fn new(exprs: Vec<Expr>) -> Self {
        Self { exprs }
    }
}

fn parse_exprs(input: &str) -> IResult<&str, Exprs> {
    let (input, first_expr) = parse_expr(input)?;
    let separators = many1(alt((parse_newline_and_whitespace, parse_expr_separator_and_whitespace)));
    let (input, mut remaining_exprs) = many0(preceded(separators, parse_expr))(input)?;
    remaining_exprs.insert(0, first_expr);
    let exprs = Exprs::new(remaining_exprs);
    Ok((input, exprs))
}

fn parse_expr_separator_and_whitespace(input: &str) -> IResult<&str, ()> {
    let (input, _) = opt(parse_multi_whitespace)(input)?;
    let (input, _) = parse_exprs_separator(input)?;
    let (input, _) = opt(parse_multi_whitespace)(input)?;
    Ok((input, ()))
}

fn parse_exprs_separator(input: &str) -> IResult<&str, ()> {
    map(
        tag(";"),
        |_| ()
    )(input)
}

#[derive(Debug, Eq, PartialEq)]
pub struct Id {
    id_str: String
}

impl Id {
    fn new(s: &str) -> Self {
        // TODO check if s is in the correct format.
        Self {
            id_str: s.to_owned(),
        }
    }
}

fn parse_id(input: &str) -> IResult<&str, Id> {
    let lowercase_chars = "abcdefghijklmnopqrstuvwxyz";
    let (input, first_char) = one_of(lowercase_chars)(input)?;
    let (input, remaining_chars) = alphanumeric0(input)?;
    let id_str = format!("{}{}", first_char, remaining_chars);
    let id = Id::new(&id_str);
    Ok((input, id))
}

impl Expr {
    fn new_def(var_name: Id, expr: Expr) -> Self {
        Self::Def {
            var_name: var_name,
            expr: Box::new(expr),
        }
    }

    fn new_return(expr: Expr) -> Self {
        Self::Return {
            expr: Box::new(expr),
        }
    }

    fn new_throw(expr: Expr) -> Self {
        Self::Throw {
            expr: Box::new(expr),
        }
    }

    fn new_literal(lit: Literal) -> Self {
        Self::Literal(lit)
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum Literal {
    Str(String),
}

impl Literal {
    fn str(s: &str) -> Self {
        Literal::Str(s.to_owned())
    }
}

fn parse_do_block_expr(input: &str) -> IResult<&str, Expr> {
    let (input, do_block) = parse_do_block(input)?;
    Ok((input, Expr::new_do_block(do_block)))
}

fn parse_do_block(input: &str) -> IResult<&str, DoBlock> {
    let (input, _) = tag("do")(input)?;
    let gap = alt((parse_multi_whitespace, parse_newline_and_whitespace));
    let (input, _) = many1(gap)(input)?;

    let gap = alt((parse_multi_whitespace, parse_newline_and_whitespace));
    let ending = preceded(many1(gap), tag("end"));
    let (input, exprs) = terminated(parse_exprs, ending)(input)?;
    let do_block = DoBlock::new(exprs);
    Ok((input, do_block))
}

fn parse_def_expr(input: &str) -> IResult<&str, Expr> {
    let (input, id) = terminated(parse_id, tag(":"))(input)?;
    let (input, _) = many0(alt((parse_multi_whitespace, parse_newline_and_whitespace)))(input)?;
    let (input, expr) = parse_expr(input)?;
    let def_expr = Expr::new_def(id, expr);
    Ok((input, def_expr))
}

fn parse_newline_and_whitespace(input: &str) -> IResult<&str, ()> {
    let (input, _) = opt(parse_multi_whitespace)(input)?;
    let (input, _) = parse_newline(input)?;
    let (input, _) = opt(parse_multi_whitespace)(input)?;
    Ok((input, ()))
}

fn parse_newline(input: &str) -> IResult<&str, ()> {
    map(
        tag("\n"),
        |_| ()
    )(input)
}

fn parse_throw_expr(input: &str) -> IResult<&str, Expr> {
    let (input, _) = parse_throw_keyword(input)?;
    let (input, _) = parse_multi_whitespace(input)?; // TODO newlines here?
    let (input, expr) = parse_expr(input)?;
    let throw_expr = Expr::new_throw(expr);
    Ok((input, throw_expr))
}

fn parse_throw_keyword(input: &str) -> IResult<&str, ()> {
    map(
        tag("throw"),
        |_| ()
    )(input)
}

fn parse_literal_expr(input: &str) -> IResult<&str, Expr> {
    let (input, literal_contents) = parse_double_quoted_string_literal(input)?;
    let expr = Expr::new_literal(Literal::str(literal_contents));
    Ok((input, expr))
}

fn parse_expr(input: &str) -> IResult<&str, Expr> {
    alt((parse_try_catch_expr, parse_literal_expr, parse_return_expr, parse_throw_expr, parse_def_expr, parse_do_block_expr))(input)
}

fn parse_single_whitespace(input: &str) -> IResult<&str, ()> {
    let space = tag(" ");
    let tab = tag("\t");
    let (input, _) = alt((space, tab))(input)?;
    Ok((input, ()))
}

fn parse_multi_whitespace(input: &str) -> IResult<&str, ()> {
    map(many1(parse_single_whitespace), |_| ())(input)
}

fn parse_return_keyword(input: &str) -> IResult<&str, ()> {
    let (input, _) = tag("return")(input)?;
    Ok((input, ()))
}

fn parse_return_expr(input: &str) -> IResult<&str, Expr> {
    let (input, _) = parse_return_keyword(input)?;
    let (input, _) = parse_multi_whitespace(input)?; // TODO newlines here?
    let (input, expr) = parse_expr(input)?;
    let return_expr = Expr::new_return(expr);
    Ok((input, return_expr))
}

fn parse_double_quoted_string_literal(input: &str) -> IResult<&str, &str> {
    let esc = escaped(none_of("\\\""), '\\', one_of(r#"fnrtv\"$"#));
    let esc_or_empty = alt((esc, tag("")));
    let res = delimited(tag(r#"""#), esc_or_empty, tag(r#"""#))(input)?;

    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wrap_in_do_block(expr: Expr) -> Expr {
        let exprs = Exprs::new(vec![expr]);
        let do_block = DoBlock::new(exprs);
        Expr::new_do_block(do_block)
    }

    #[test]
    fn parse_try_catch_works_on_single_line() {
        let s = "try \"a\" catch \"b\"";
        let expect_try_expr = wrap_in_do_block(Expr::new_literal(Literal::str("a")));
        let expect_catch_expr = wrap_in_do_block(Expr::new_literal(Literal::str("b")));
        let expect = TryCatchBlock::new(expect_try_expr, None, expect_catch_expr);
        assert_eq!(parse_try_catch(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_try_catch_works_on_single_line_with_do_end() {
        let s = "try do \"a\" end catch do \"b\" end";
        let expect_try_expr = wrap_in_do_block(Expr::new_literal(Literal::str("a")));
        let expect_catch_expr = wrap_in_do_block(Expr::new_literal(Literal::str("b")));
        let expect = TryCatchBlock::new(expect_try_expr, None, expect_catch_expr);
        assert_eq!(parse_try_catch(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_try_catch_works_on_single_line_with_do_but_no_end() {
        let s = "try do \"a\" catch do \"b\"";
        let expect_try_expr = wrap_in_do_block(Expr::new_literal(Literal::str("a")));
        let expect_catch_expr = wrap_in_do_block(Expr::new_literal(Literal::str("b")));
        let expect = TryCatchBlock::new(expect_try_expr, None, expect_catch_expr);
        assert_eq!(parse_try_catch(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_try_catch_works_on_multi_lines() {
        let s = "try \n \"a\" \ncatch\n\"b\"";
        let expect_try_expr = wrap_in_do_block(Expr::new_literal(Literal::str("a")));
        let expect_catch_expr = wrap_in_do_block(Expr::new_literal(Literal::str("b")));
        let expect = TryCatchBlock::new(expect_try_expr, None, expect_catch_expr);
        assert_eq!(parse_try_catch(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_try_catch_works_on_multi_lines_with_do_end() {
        let s = "try do\n\"a\"end\ncatch do\n\"b\"end";
        let expect_try_expr = wrap_in_do_block(Expr::new_literal(Literal::str("a")));
        let expect_catch_expr = wrap_in_do_block(Expr::new_literal(Literal::str("b")));
        let expect = TryCatchBlock::new(expect_try_expr, None, expect_catch_expr);
        assert_eq!(parse_try_catch(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_try_catch_works_on_multi_lines_with_do_but_no_end() {
        let s = "try do\n\"a\"\ncatch do\n\"b\"";
        let expect_try_expr = wrap_in_do_block(Expr::new_literal(Literal::str("a")));
        let expect_catch_expr = wrap_in_do_block(Expr::new_literal(Literal::str("b")));
        let expect = TryCatchBlock::new(expect_try_expr, None, expect_catch_expr);
        assert_eq!(parse_try_catch(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_try_catch_works_on_a_mixed_bag() {
        let s = "try do   \n   \"a\"   \n\n\t\n   \"b\" \n end \ncatch do\n\"c\"\n\"d\"    \t";
        let a = Expr::new_literal(Literal::str("a"));
        let b = Expr::new_literal(Literal::str("b"));
        let try_exprs = Expr::new_do_block(DoBlock::new(Exprs::new(vec![a, b])));
        let c = Expr::new_literal(Literal::str("c"));
        let d = Expr::new_literal(Literal::str("d"));
        let catch_exprs = Expr::new_do_block(DoBlock::new(Exprs::new(vec![c, d])));
        let expect = TryCatchBlock::new(try_exprs, None, catch_exprs);
        assert_eq!(parse_try_catch(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_do_block_works_on_single_line() {
        let s = "do \"Hello\" end";
        let expect_expr = Expr::Literal(Literal::Str("Hello".to_owned()));
        let expect_exprs = Exprs::new(vec![expect_expr]);
        let expect = DoBlock::new(expect_exprs);
        assert_eq!(parse_do_block(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_do_block_works_on_multiple_lines() {
        let s = "do \"Hello\" \n \"World\" end";
        let expect_expr1 = Expr::Literal(Literal::Str("Hello".to_owned()));
        let expect_expr2 = Expr::Literal(Literal::Str("World".to_owned()));
        let expect_exprs = Exprs::new(vec![expect_expr1, expect_expr2]);
        let expect = DoBlock::new(expect_exprs);
        assert_eq!(parse_do_block(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_exprs_works_for_single_expr() {
        let s = "return \"test\"";
        let expect_expr = Expr::new_return(Expr::new_literal(Literal::str("test")));
        let expect = Exprs::new(vec![expect_expr]);
        assert_eq!(parse_exprs(s).unwrap(), ("", expect))
    }

    #[test]
    fn parse_exprs_works_for_semicolon_separated_exprs() {
        let s = "return \"one\"; return \"two\"";
        let expect_expr1 = Expr::new_return(Expr::new_literal(Literal::str("one")));
        let expect_expr2 = Expr::new_return(Expr::new_literal(Literal::str("two")));
        let expect = Exprs::new(vec![expect_expr1, expect_expr2]);
        assert_eq!(parse_exprs(s).unwrap(), ("", expect));

        let s = "return \"one\";return \"two\"";
        let expect_expr1 = Expr::new_return(Expr::new_literal(Literal::str("one")));
        let expect_expr2 = Expr::new_return(Expr::new_literal(Literal::str("two")));
        let expect = Exprs::new(vec![expect_expr1, expect_expr2]);
        assert_eq!(parse_exprs(s).unwrap(), ("", expect));

        let s = "return \"one\"  \t ; \t return \"two\"";
        let expect_expr1 = Expr::new_return(Expr::new_literal(Literal::str("one")));
        let expect_expr2 = Expr::new_return(Expr::new_literal(Literal::str("two")));
        let expect = Exprs::new(vec![expect_expr1, expect_expr2]);
        assert_eq!(parse_exprs(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_exprs_works_for_newline_separated_exprs() {
        let s = "return \"one\"\nreturn \"two\"";
        let expect_expr1 = Expr::new_return(Expr::new_literal(Literal::str("one")));
        let expect_expr2 = Expr::new_return(Expr::new_literal(Literal::str("two")));
        let expect = Exprs::new(vec![expect_expr1, expect_expr2]);
        assert_eq!(parse_exprs(s).unwrap(), ("", expect));

        let s = "return \"one\"  \t  \n  \t return \"two\"";
        let expect_expr1 = Expr::new_return(Expr::new_literal(Literal::str("one")));
        let expect_expr2 = Expr::new_return(Expr::new_literal(Literal::str("two")));
        let expect = Exprs::new(vec![expect_expr1, expect_expr2]);
        assert_eq!(parse_exprs(s).unwrap(), ("", expect));
    }
    #[test]
    fn parse_exprs_works_for_multi_newline_separated_exprs() {
        let s = "return \"one\"\n\n\n   return \"two\"";
        let expect_expr1 = Expr::new_return(Expr::new_literal(Literal::str("one")));
        let expect_expr2 = Expr::new_return(Expr::new_literal(Literal::str("two")));
        let expect = Exprs::new(vec![expect_expr1, expect_expr2]);
        assert_eq!(parse_exprs(s).unwrap(), ("", expect));

        let s = "return \"one\" \n \t  \n\n  \t return \"two\"";
        let expect_expr1 = Expr::new_return(Expr::new_literal(Literal::str("one")));
        let expect_expr2 = Expr::new_return(Expr::new_literal(Literal::str("two")));
        let expect = Exprs::new(vec![expect_expr1, expect_expr2]);
        assert_eq!(parse_exprs(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_id_works_for_valid_ids() {
        let s = "x";
        let expected = Id::new("x");
        assert_eq!(parse_id(s).unwrap(), ("", expected));

        let s = "varName";
        let expected = Id::new("varName");
        assert_eq!(parse_id(s).unwrap(), ("", expected));

        let s = "v0";
        let expected = Id::new("v0");
        assert_eq!(parse_id(s).unwrap(), ("", expected));
    }

    #[test]
    fn parse_id_fails_for_empty_id() {
        let s = "";
        assert!(parse_id(s).is_err());
    }

    #[test]
    fn parse_id_fails_for_invalid_ids() {
        let s = "Capital";
        assert!(parse_id(s).is_err());

        let s = "1stVar";
        assert!(parse_id(s).is_err());
    }

    #[test]
    fn parse_whitespace_works() {
        let s = " afterWhitespace";
        assert_eq!(parse_multi_whitespace(s).unwrap(), ("afterWhitespace", ()));

        let s = "\tafterWhitespace";
        assert_eq!(parse_multi_whitespace(s).unwrap(), ("afterWhitespace", ()));

        let s = " \t  \tafterWhitespace";
        assert_eq!(parse_multi_whitespace(s).unwrap(), ("afterWhitespace", ()));
    }

    #[test]
    fn parse_expr_def_works() {
        let e = "x: \"Hello\"";
        let expect_id = Id::new("x");
        let expect = Expr::new_def(expect_id, Expr::new_literal(Literal::str("Hello")));
        assert_eq!(
            parse_expr(e).unwrap(),
            ("", expect)
        );
    }

    #[test]
    fn parse_expr_def_works_with_no_separating_whitespace() {
        let e = "x:\"Hello\"";
        let expect_id = Id::new("x");
        let expect = Expr::new_def(expect_id, Expr::new_literal(Literal::str("Hello")));
        assert_eq!(
            parse_expr(e).unwrap(),
            ("", expect)
        );
    }

    #[test]
    fn parse_expr_def_works_with_many_separating_whitespace() {
        let e = "x: \n  \t   \"Hello\"";
        let expect_id = Id::new("x");
        let expect = Expr::new_def(expect_id, Expr::new_literal(Literal::str("Hello")));
        assert_eq!(
            parse_expr(e).unwrap(),
            ("", expect)
        );
    }

    #[test]
    fn parse_expr_return_works() {
        let e = "return \"Hello\"";
        let expect = Expr::new_return(Expr::new_literal(Literal::str("Hello")));
        assert_eq!(
            parse_expr(e).unwrap(),
            ("", expect)
        );
    }

    #[test]
    fn parse_expr_throw_works() {
        let e = "throw \"Some error\"";
        let expect = Expr::new_throw(Expr::new_literal(Literal::str("Some error")));
        assert_eq!(
            parse_expr(e).unwrap(),
            ("", expect)
        );
    }

    #[test]
    fn parse_double_quoted_string_literal_works_on_normal_input() {
        let s = "\"Contents of this string is an Axon Str literal\"";
        assert_eq!(
            parse_double_quoted_string_literal(s).unwrap(),
            ("", "Contents of this string is an Axon Str literal")
        );
    }

    #[test]
    fn parse_double_quoted_string_literal_works_on_empty_input() {
        let s = "\"\"";
        assert_eq!(parse_double_quoted_string_literal(s).unwrap(), ("", ""));
    }

    #[test]
    fn parse_double_quoted_string_literal_works_on_input_with_escaped_double_quotes(
    ) {
        let s = r#""Name is \"Andrew\"""#;
        assert_eq!(
            parse_double_quoted_string_literal(s).unwrap(),
            ("", r#"Name is \"Andrew\""#)
        );
    }

    #[test]
    fn parse_double_quoted_string_literal_works_on_input_with_common_escaped_chars(
    ) {
        let s = r#""First\nSecond\nTab\tDone""#;
        assert_eq!(
            parse_double_quoted_string_literal(s).unwrap(),
            ("", r#"First\nSecond\nTab\tDone"#)
        );
    }

    #[test]
    fn parse_double_quoted_string_literal_works_on_input_with_uncommon_escaped_chars(
    ) {
        let s = r#""First\fSecond\rTab\vDone""#;
        assert_eq!(
            parse_double_quoted_string_literal(s).unwrap(),
            ("", r#"First\fSecond\rTab\vDone"#)
        );
    }

    #[test]
    fn parse_double_quoted_string_literal_works_on_input_with_escaped_backslash(
    ) {
        let s = r#""Backslash: \\""#;
        assert_eq!(
            parse_double_quoted_string_literal(s).unwrap(),
            ("", r#"Backslash: \\"#)
        );
    }

    #[test]
    fn parse_double_quoted_string_literal_works_on_input_with_escaped_dollar_sign(
    ) {
        let s = r#""\$equipRef \$navName""#;
        assert_eq!(
            parse_double_quoted_string_literal(s).unwrap(),
            ("", r#"\$equipRef \$navName"#)
        );
    }
}
