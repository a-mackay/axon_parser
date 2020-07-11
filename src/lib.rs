use nom::{
    branch::alt,
    bytes::complete::{escaped, tag},
    character::complete::{alphanumeric0, none_of, one_of},
    combinator::map,
    multi::{many0, many1},
    sequence::{delimited, separated_pair, terminated},
    IResult,
};

#[derive(Debug, Eq, PartialEq)]
enum Expr {
    Def {
        var_name: Id,
        expr: Box<Expr>,
    },
    Return { expr: Box<Expr> },
    Throw { expr: Box<Expr> },
    Literal(Literal),
}

#[derive(Debug, Eq, PartialEq)]
struct Id {
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
enum Literal {
    Str(String),
}

impl Literal {
    fn str(s: &str) -> Self {
        Literal::Str(s.to_owned())
    }
}

fn parse_def_expr(input: &str) -> IResult<&str, Expr> {
    let (input, id) = terminated(parse_id, tag(":"))(input)?;
    let (input, _) = many0(alt((parse_whitespace, parse_newline_and_whitespace)))(input)?;
    let (input, expr) = parse_expr(input)?;
    let def_expr = Expr::new_def(id, expr);
    Ok((input, def_expr))
}

fn parse_newline_and_whitespace(input: &str) -> IResult<&str, ()> {
    let (input, _) = many0(parse_whitespace)(input)?;
    let (input, _) = parse_newline(input)?;
    let (input, _) = many0(parse_whitespace)(input)?;
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
    let (input, _) = parse_whitespace(input)?;
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
    alt((parse_literal_expr, parse_return_expr, parse_throw_expr, parse_def_expr))(input)
}

fn parse_whitespace(input: &str) -> IResult<&str, ()> {
    let space = tag(" ");
    let tab = tag("\t");
    let space_or_tab = alt((space, tab));
    map(many1(space_or_tab), |_| ())(input)
}

fn parse_return_keyword(input: &str) -> IResult<&str, ()> {
    let (input, _) = tag("return")(input)?;
    Ok((input, ()))
}

fn parse_return_expr(input: &str) -> IResult<&str, Expr> {
    let (input, _) = parse_return_keyword(input)?;
    let (input, _) = parse_whitespace(input)?;
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
        assert_eq!(parse_whitespace(s).unwrap(), ("afterWhitespace", ()));

        let s = "\tafterWhitespace";
        assert_eq!(parse_whitespace(s).unwrap(), ("afterWhitespace", ()));

        let s = " \t  \tafterWhitespace";
        assert_eq!(parse_whitespace(s).unwrap(), ("afterWhitespace", ()));
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
