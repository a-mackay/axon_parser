use chrono::{NaiveDate, NaiveTime};
use nom::{
    branch::alt,
    bytes::complete::{escaped, tag},
    character::complete::{digit1, none_of, one_of},
    combinator::{map, not, opt, peek},
    multi::{count, many0, many1},
    sequence::{delimited, pair, preceded, terminated, tuple},
    IResult,
};

pub fn parse_function(input: &str) -> Result<LambdaMany, ()> {
    let input = remove_line_comments(input);
    let (input, _) =
        opt(gap)(&input).expect("since this is optional, it should never fail");
    let (_, lambda_many) = parse_lambda_many(input).unwrap(); //TODO
    Ok(lambda_many)
}

pub fn remove_line_comments(i: &str) -> String {
    i.lines()
        .into_iter()
        .filter(|line| !line.trim().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_if<'a>(i: &'a str) -> IResult<&'a str, If> {
    let conditional_start = tuple((tag("if"), optgap, tag("(")));
    let (i, _) = terminated(conditional_start, optgap)(i)?;
    let conditional_end = pair(optgap, tag(")"));
    let (i, condition) = terminated(parse_expr, conditional_end)(i)?;

    // Parse the consequent Expr or DoBlock:

    let parse_else_keyword = |i: &'a str| parse_specific_keyword(i, "else");
    let parse_else = |i: &'a str| map(parse_else_keyword, |_| ())(i);
    let parse_alt_do_block =
        |i: &'a str| parse_do_block_with_no_end_keyword(i, parse_else);

    let parse_exprs = preceded(
        optgap,
        alt((
            parse_do_block,
            parse_alt_do_block,
            map(parse_expr, DoBlock::new_single_expr),
        )),
    );

    let (i, consequent) = preceded(optgap, parse_exprs)(i)?;

    // Parse the alternative Expr or DoBlock:

    let parse_else_keyword = |i: &'a str| parse_specific_keyword(i, "else");

    let parse_exprs = preceded(
        optgap,
        alt((parse_do_block, map(parse_expr, DoBlock::new_single_expr))),
    );
    let (i, opt_alternative) =
        opt(preceded(pair(optgap, parse_else_keyword), parse_exprs))(i)?;

    let opt_alternative =
        opt_alternative.map(|do_block| Expr::DoBlock(do_block));
    let if_struct =
        If::new(condition, Expr::DoBlock(consequent), opt_alternative);
    Ok((i, if_struct))
}

fn optgap(i: &str) -> IResult<&str, ()> {
    map(opt(gap), |_| ())(i)
}

#[derive(Debug, Eq, PartialEq)]
pub struct If {
    condition: Expr,
    consequent: Expr,
    alternative: Option<Expr>,
}

impl If {
    fn new(
        condition: Expr,
        consequent: Expr,
        alternative: Option<Expr>,
    ) -> Self {
        Self {
            condition,
            consequent,
            alternative,
        }
    }
}

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

    if has_try_end_keyword && !has_try_do_keyword {
        todo!();
    }

    // Remove any blank space, in the case there was no 'end' keyword:
    let (input, _) = opt(gap)(input)?;

    let (input, _) = terminated(tag("catch"), gap)(input)?;

    let open_paren = terminated(tag("("), gap);
    let close_paren = terminated(tag(")"), opt(gap));
    let parse_id_and_gap = terminated(parse_id, opt(gap));
    let (input, opt_id) =
        opt(delimited(open_paren, parse_id_and_gap, close_paren))(input)?;

    // Remove any blank space, in case there was any after the optional ')':
    let (input, _) = opt(gap)(input)?;

    let (input, opt_catch_do) = opt(terminated(tag("do"), gap))(input)?;
    let has_catch_do_keyword = opt_catch_do.is_some();

    let (input, catch_exprs) = parse_exprs(input)?;

    // Remove any blank space, in the case there is any:
    let (input, _) = opt(gap)(input)?;

    let (input, opt_catch_end) = opt(terminated(tag("end"), opt(gap)))(input)?;
    let has_catch_end_keyword = opt_catch_end.is_some();

    if has_catch_end_keyword && !has_catch_do_keyword {
        todo!();
    }

    let try_do_block = Expr::new_do_block(DoBlock::new(try_exprs));
    let catch_do_block = Expr::new_do_block(DoBlock::new(catch_exprs));
    let try_catch = TryCatchBlock::new(try_do_block, opt_id, catch_do_block);
    Ok((input, try_catch))
}

fn gap(input: &str) -> IResult<&str, ()> {
    let (input, _) = many1(alt((
        parse_multi_whitespace,
        parse_newline_and_whitespace,
    )))(input)?;
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

    fn new_single_expr(expr: Expr) -> Self {
        Self {
            exprs: Exprs::new(vec![expr]),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum ComparisonOperator {
    Equals,
    NotEquals,
    LessThan,
    LessThanOrEqual,
    GreaterThanOrEqual,
    GreaterThan,
    Compare,
}

impl ComparisonOperator {
    pub fn to_symbol(&self) -> &str {
        match self {
            Self::Equals => "==",
            Self::NotEquals => "!=",
            Self::LessThan => "<",
            Self::LessThanOrEqual => "<=",
            Self::GreaterThanOrEqual => ">=",
            Self::GreaterThan => ">",
            Self::Compare => "<=>",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "==" => Some(Self::Equals),
            "!=" => Some(Self::NotEquals),
            "<" => Some(Self::LessThan),
            "<=" => Some(Self::LessThanOrEqual),
            ">=" => Some(Self::GreaterThanOrEqual),
            ">" => Some(Self::GreaterThan),
            "<=>" => Some(Self::Compare),
            _ => None,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum UnaryExprSign {
    Not,
    Minus,
}

impl UnaryExprSign {
    pub fn to_symbol(&self) -> &str {
        match self {
            Self::Not => "not",
            Self::Minus => "-",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "-" => Some(Self::Minus),
            "not" => Some(Self::Not),
            _ => None,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct Param {
    name: Id,
    default_value: Option<Expr>,
}

impl Param {
    fn new(name: Id, default_value: Option<Expr>) -> Self {
        Self {
            name,
            default_value,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct LambdaOne {
    var_name: Id,
    expr: Expr,
}

impl LambdaOne {
    fn new(var_name: Id, expr: Expr) -> Self {
        Self { var_name, expr }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct LambdaMany {
    params: Vec<Param>,
    expr: Expr,
}

impl LambdaMany {
    fn new(params: Vec<Param>, expr: Expr) -> Self {
        Self { params, expr }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum Lambda {
    One(LambdaOne),
    Many(LambdaMany),
}

#[derive(Debug, Eq, PartialEq)]
enum TermBase {
    Var(Id),
    GroupedExpr(Box<Expr>),
    Literal(Literal),
}

#[derive(Debug, Eq, PartialEq)]
enum CallArg {
    Expr(Expr),
    DiscardedExpr,
}

#[derive(Debug, Eq, PartialEq)]
struct Call {
    call_args: Vec<CallArg>,
    lambda: Option<Lambda>,
}

impl Call {
    fn new(call_args: Vec<CallArg>, lambda: Option<Lambda>) -> Self {
        Self { call_args, lambda }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum TrailingCall {
    Call(Call),
    Lambda(Lambda),
}

#[derive(Debug, Eq, PartialEq)]
struct DotCall {
    function_name: Id,
    trailing_call: Option<TrailingCall>,
}

impl DotCall {
    fn new(function_name: Id, trailing_call: Option<TrailingCall>) -> Self {
        Self {
            function_name,
            trailing_call,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum TermChain {
    Call(Call),
    DotCall(DotCall),
    Index(Box<Expr>),
    TrapCall(Id),
}

#[derive(Debug, Eq, PartialEq)]
pub struct TermExpr {
    term_base: TermBase,
    term_chain: Vec<TermChain>,
}

impl TermExpr {
    fn new(term_base: TermBase, term_chain: Vec<TermChain>) -> Self {
        Self {
            term_base,
            term_chain,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct UnaryExpr {
    sign: Option<UnaryExprSign>,
    term_expr: TermExpr,
}

impl UnaryExpr {
    fn new(sign: Option<UnaryExprSign>, term_expr: TermExpr) -> Self {
        Self { sign, term_expr }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct OperatedUnaryExpr {
    is_multiply: bool,
    unary_expr: UnaryExpr,
}

impl OperatedUnaryExpr {
    fn new(is_multiply: bool, unary_expr: UnaryExpr) -> Self {
        Self {
            is_multiply,
            unary_expr,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct MultExpr {
    first_unary_expr: UnaryExpr,
    remaining_operated_unary_exprs: Vec<OperatedUnaryExpr>,
}

impl MultExpr {
    fn new(
        first_unary_expr: UnaryExpr,
        remaining_operated_unary_exprs: Vec<OperatedUnaryExpr>,
    ) -> Self {
        Self {
            first_unary_expr,
            remaining_operated_unary_exprs,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct SignedMultExpr {
    is_add: bool,
    mult_expr: MultExpr,
}

impl SignedMultExpr {
    fn new(is_add: bool, mult_expr: MultExpr) -> Self {
        Self { is_add, mult_expr }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct AddExpr {
    first_mult_expr: MultExpr,
    remaining_signed_mult_exprs: Vec<SignedMultExpr>,
}

impl AddExpr {
    fn new(
        first_mult_expr: MultExpr,
        remaining_signed_mult_exprs: Vec<SignedMultExpr>,
    ) -> Self {
        Self {
            first_mult_expr,
            remaining_signed_mult_exprs,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct RangeExpr {
    left_add_expr: AddExpr,
    right_add_expr: AddExpr,
}

impl RangeExpr {
    fn new(left_add_expr: AddExpr, right_add_expr: AddExpr) -> Self {
        Self {
            left_add_expr,
            right_add_expr,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct ComparedRangeExpr {
    operator: ComparisonOperator,
    range_expr: RangeExpr,
}

impl ComparedRangeExpr {
    fn new(operator: ComparisonOperator, range_expr: RangeExpr) -> Self {
        Self {
            operator,
            range_expr,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct CompareExpr {
    first_range_expr: RangeExpr,
    compared_range_exprs: Vec<ComparedRangeExpr>,
}

impl CompareExpr {
    fn new(
        first_range_expr: RangeExpr,
        compared_range_exprs: Vec<ComparedRangeExpr>,
    ) -> Self {
        Self {
            first_range_expr,
            compared_range_exprs,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct CondAndExpr {
    first_compare_expr: CompareExpr,
    remaining_compare_exprs: Vec<CompareExpr>,
}

impl CondAndExpr {
    fn new(first: CompareExpr, remaining: Vec<CompareExpr>) -> Self {
        Self {
            first_compare_expr: first,
            remaining_compare_exprs: remaining,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct CondOrExpr {
    first_cond_and_expr: CondAndExpr,
    remaining_cond_and_exprs: Vec<CondAndExpr>,
}

impl CondOrExpr {
    fn new(first: CondAndExpr, remaining: Vec<CondAndExpr>) -> Self {
        Self {
            first_cond_and_expr: first,
            remaining_cond_and_exprs: remaining,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct AssignExpr {
    cond_or: CondOrExpr,
    assign: Option<Box<AssignExpr>>,
}

impl AssignExpr {
    fn new(cond_or: CondOrExpr, assign: Option<AssignExpr>) -> Self {
        Self {
            cond_or,
            assign: assign.map(|a| Box::new(a)),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum Expr {
    Assign(AssignExpr),
    /// This differs from the Axon grammar page, I don't think
    /// expressions like `x = 5` can be expressed in the grammar?
    Assignment {
        var_name: Id,
        expr: Box<Expr>,
    },
    CondOr(CondOrExpr),
    DoBlock(DoBlock),
    Def {
        var_name: Id,
        expr: Box<Expr>,
    },
    If(Box<If>),
    Return {
        expr: Box<Expr>,
    },
    /// This differs from the Axon grammar page, I don't think
    /// expressions like `x.toStr()` can be expressed in the grammar?
    TermExpr(TermExpr),
    Throw {
        expr: Box<Expr>,
    },
    TryCatch(Box<TryCatchBlock>),
    Literal(Literal),
}

impl Expr {
    fn new_assignment(id: Id, expr: Expr) -> Self {
        Self::Assignment {
            var_name: id,
            expr: Box::new(expr),
        }
    }

    fn new_if(if_struct: If) -> Self {
        Self::If(Box::new(if_struct))
    }

    fn new_assign(assign_expr: AssignExpr) -> Self {
        Self::Assign(assign_expr)
    }

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
    let separators = many1(alt((
        parse_newline_and_whitespace,
        parse_expr_separator_and_whitespace,
    )));
    let (input, mut remaining_exprs) =
        many0(preceded(separators, parse_expr))(input)?;
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
    map(tag(";"), |_| ())(input)
}

#[derive(Debug, Eq, PartialEq)]
pub struct Id {
    id_str: String,
}

impl Id {
    fn new(s: &str) -> Self {
        // TODO check if s is in the correct format.
        Self {
            id_str: s.to_owned(),
        }
    }
}

fn parse_id(i: &str) -> IResult<&str, Id> {
    // If it's a keyword, it can't be an id, so return early if its a keyword.
    let (i, _) = peek(not(parse_keyword))(i)?;

    let lower_char = one_of("abcdefghijklmnopqrstuvwxyz");
    let chars = one_of(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_",
    );
    let other_chars = many0(chars);
    let mut id = pair(lower_char, other_chars);

    let (i, (first_char, remaining_chars)) = id(i)?;
    let remaining_str: String = remaining_chars.into_iter().collect();
    let id_str = format!("{}{}", first_char, remaining_str);
    let id = Id::new(&id_str);
    Ok((i, id))
}

fn parse_specific_keyword<'a>(
    i: &'a str,
    target_keyword: &str,
) -> IResult<&'a str, String> {
    let parse_lower_word = map(
        many1(one_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_",
        )),
        |char_vec| char_vec.into_iter().collect::<String>(),
    );
    let (i, word) = peek(parse_lower_word)(i)?;
    if word == target_keyword {
        let mut parse_lower_word = map(many1(one_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_")), |char_vec| char_vec.into_iter().collect::<String>());
        parse_lower_word(i)
    } else {
        // TODO is this the right approach to returning errors? Prob not
        Err(nom::Err::Error((i, nom::error::ErrorKind::Tag)))
    }
}

fn parse_keyword(i: &str) -> IResult<&str, String> {
    let parse_and = |i| parse_specific_keyword(i, "and");
    let parse_catch = |i| parse_specific_keyword(i, "catch");
    let parse_defcomp = |i| parse_specific_keyword(i, "defcomp");
    let parse_deflinks = |i| parse_specific_keyword(i, "deflinks");
    let parse_do = |i| parse_specific_keyword(i, "do");
    let parse_else = |i| parse_specific_keyword(i, "else");
    let parse_end = |i| parse_specific_keyword(i, "end");
    let parse_false = |i| parse_specific_keyword(i, "false");
    let parse_if = |i| parse_specific_keyword(i, "if");
    let parse_not = |i| parse_specific_keyword(i, "not");
    let parse_null = |i| parse_specific_keyword(i, "null");
    let parse_or = |i| parse_specific_keyword(i, "or");
    let parse_return = |i| parse_specific_keyword(i, "return");
    let parse_throw = |i| parse_specific_keyword(i, "throw");
    let parse_true = |i| parse_specific_keyword(i, "true");
    let parse_try = |i| parse_specific_keyword(i, "try");

    alt((
        parse_and,
        parse_catch,
        parse_defcomp,
        parse_deflinks,
        parse_do,
        parse_else,
        parse_end,
        parse_false,
        parse_if,
        parse_not,
        parse_null,
        parse_or,
        parse_return,
        parse_throw,
        parse_true,
        parse_try,
    ))(i)
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
    Bool(bool),
    Date(NaiveDate),
    Ref(String),
    Str(String),
    Time(NaiveTime),
    DateMonth(Month),
    Null,
    Number {
        integral: String,
        fractional: Option<String>,
        exponent: Option<String>,
        unit: Option<String>,
    },
}

impl Literal {
    fn date(year: i32, month: u32, day: u32) -> Self {
        Self::Date(NaiveDate::from_ymd(year, month, day))
    }

    fn bool(b: bool) -> Self {
        Self::Bool(b)
    }

    fn ref_lit(ref_str: &str) -> Self {
        Self::Ref(ref_str.to_owned())
    }

    fn str(s: &str) -> Self {
        Self::Str(s.to_owned())
    }

    fn month(month: Month) -> Self {
        Self::DateMonth(month)
    }

    fn num(
        integral: String,
        fractional: Option<String>,
        exponent: Option<String>,
        unit: Option<String>,
    ) -> Self {
        Self::Number {
            integral,
            fractional,
            exponent,
            unit,
        }
    }

    fn int(n: i32) -> Self {
        Self::num(format!("{}", n), None, None, None)
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct Month {
    year: i32,
    month: u32,
}

impl Month {
    pub fn new(year: i32, month: u32) -> Option<Self> {
        match month {
            (1..=12) => Some(Self { year, month }),
            _ => None,
        }
    }

    pub fn year(&self) -> i32 {
        self.year
    }

    /// Returns a number between 1 and 12 inclusive.
    pub fn month(&self) -> u32 {
        self.month
    }
}

fn parse_do_block_expr(input: &str) -> IResult<&str, Expr> {
    let (input, do_block) = parse_do_block(input)?;
    Ok((input, Expr::new_do_block(do_block)))
}

// TODO use this function in try-do block
fn parse_do_block_with_no_end_keyword<'a, F>(
    i: &'a str,
    parse_terminator: F,
) -> IResult<&'a str, DoBlock>
where
    F: FnMut(&'a str) -> IResult<&'a str, ()>,
{
    let parse_do_keyword = |i| parse_specific_keyword(i, "do");
    let (i, _) = pair(parse_do_keyword, optgap)(i)?;
    let ending = pair(optgap, parse_terminator);
    let (i, exprs) = terminated(parse_exprs, peek(ending))(i)?;
    let do_block = DoBlock::new(exprs);
    Ok((i, do_block))
}

fn parse_do_block(i: &str) -> IResult<&str, DoBlock> {
    let parse_do_keyword = |i| parse_specific_keyword(i, "do");
    let (i, _) = pair(parse_do_keyword, optgap)(i)?;
    let parse_end_keyword = |i| parse_specific_keyword(i, "end");
    let ending = pair(optgap, parse_end_keyword);
    let (i, exprs) = terminated(parse_exprs, ending)(i)?;
    let do_block = DoBlock::new(exprs);
    Ok((i, do_block))
}

fn parse_assignment_expr(input: &str) -> IResult<&str, Expr> {
    let parse_equals = delimited(opt(gap), tag("="), opt(gap));
    let (input, id) = terminated(parse_id, parse_equals)(input)?;
    let (input, expr) = parse_expr(input)?;
    let assignment_expr = Expr::new_assignment(id, expr);
    Ok((input, assignment_expr))
}

fn parse_def_expr(input: &str) -> IResult<&str, Expr> {
    let parse_colon = preceded(opt(gap), tag(":"));
    let (input, id) = terminated(parse_id, parse_colon)(input)?;
    let (input, _) = many0(alt((
        // TODO replace with opt(gap)?
        parse_multi_whitespace,
        parse_newline_and_whitespace,
    )))(input)?;
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
    map(tag("\n"), |_| ())(input)
}

fn parse_throw_expr(input: &str) -> IResult<&str, Expr> {
    let (input, _) = parse_throw_keyword(input)?;
    let (input, _) = parse_multi_whitespace(input)?; // TODO newlines here?
    let (input, expr) = parse_expr(input)?;
    let throw_expr = Expr::new_throw(expr);
    Ok((input, throw_expr))
}

fn parse_throw_keyword(input: &str) -> IResult<&str, ()> {
    map(tag("throw"), |_| ())(input)
}

fn parse_literal_expr(input: &str) -> IResult<&str, Expr> {
    map(parse_literal, |literal| Expr::new_literal(literal))(input)
}

fn parse_literal(input: &str) -> IResult<&str, Literal> {
    alt((
        parse_double_quoted_string_literal,
        parse_number_literal,
        parse_bool_literal,
        parse_null_literal,
        parse_date_literal,
        parse_month_literal,
        parse_ref_literal,
        map(parse_naive_time, Literal::Time),
    ))(input)
}

fn parse_ref_literal(input: &str) -> IResult<&str, Literal> {
    let (input, _) = tag("@")(input)?;
    let ref_char = one_of(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_:.~-",
    );
    let (input, ref_chars) = many1(ref_char)(input)?;
    let ref_str = ref_chars.into_iter().collect::<String>().to_owned();
    let ref_str = format!("@{}", ref_str);
    Ok((input, Literal::ref_lit(&ref_str)))
}

fn valid_digit(input: &str) -> IResult<&str, char> {
    one_of("1234567890")(input)
}

fn parse_naive_time(i: &str) -> IResult<&str, NaiveTime> {
    let parse_hours = alt((count(valid_digit, 2), count(valid_digit, 1)));
    let (i, hours_str) = terminated(parse_hours, tag(":"))(i)?;
    let (i, mins_str) = count(valid_digit, 2)(i)?;

    let (i, opt_secs_str) = opt(preceded(tag(":"), count(valid_digit, 2)))(i)?;
    let (i, opt_millisecs_str) = opt(preceded(tag("."), digit1))(i)?;

    if opt_secs_str.is_none() && opt_millisecs_str.is_some() {
        todo!();
    }

    let secs = match opt_secs_str {
        Some(secs_str) => secs_str
            .into_iter()
            .collect::<String>()
            .parse()
            .expect("Should parse as an int"),
        None => 0,
    };

    let millisecs = match opt_millisecs_str {
        Some(millisecs_str) => {
            millisecs_str.parse().expect("Should parse as an int")
        }
        None => 0,
    };

    let time = NaiveTime::from_hms_milli(
        hours_str
            .into_iter()
            .collect::<String>()
            .parse()
            .expect("Should parse as an int"),
        mins_str
            .into_iter()
            .collect::<String>()
            .parse()
            .expect("Should parse as an int"),
        secs,
        millisecs,
    );

    Ok((i, time))
}

fn parse_month_literal(input: &str) -> IResult<&str, Literal> {
    let parse_year = terminated(count(valid_digit, 4), tag("-"));
    let parse_month = count(valid_digit, 2);
    let (input, (year, month)) = pair(parse_year, parse_month)(input)?;
    let year = year
        .into_iter()
        .collect::<String>()
        .parse()
        .expect("Year should be a valid int");
    let month = month
        .into_iter()
        .collect::<String>()
        .parse()
        .expect("Month should be a valid int");
    let year_month =
        Month::new(year, month).expect("Month literal should be valid");
    Ok((input, Literal::month(year_month)))
}

fn parse_date_literal(input: &str) -> IResult<&str, Literal> {
    let parse_year = terminated(count(valid_digit, 4), tag("-"));
    let parse_month = terminated(count(valid_digit, 2), tag("-"));
    let parse_day = count(valid_digit, 2);

    let (input, (year, month, day)) =
        tuple((parse_year, parse_month, parse_day))(input)?;

    let year = year
        .into_iter()
        .collect::<String>()
        .parse()
        .expect("Year should be a valid int");
    let month = month
        .into_iter()
        .collect::<String>()
        .parse()
        .expect("Month should be a valid int");
    let day = day
        .into_iter()
        .collect::<String>()
        .parse()
        .expect("Day should be a valid int");

    Ok((input, Literal::date(year, month, day)))
}

fn parse_null_literal(input: &str) -> IResult<&str, Literal> {
    map(tag("null"), |_| Literal::Null)(input)
}

fn parse_bool_literal(input: &str) -> IResult<&str, Literal> {
    let (input, bool_str) = alt((tag("true"), tag("false")))(input)?;
    let b = match bool_str {
        "true" => true,
        "false" => false,
        _ => unreachable!(),
    };
    Ok((input, Literal::bool(b)))
}

fn parse_number_literal(i: &str) -> IResult<&str, Literal> {
    let (i, opt_integral_negation) = opt(terminated(tag("-"), opt(gap)))(i)?;
    let is_integral_negative = opt_integral_negation.is_some();

    let (i, integral) = digit1(i)?;
    let (i, opt_fractional) = opt(preceded(tag("."), digit1))(i)?;

    let exponent_num = pair(opt(tag("-")), digit1);
    let (i, exponent_info) =
        opt(preceded(alt((tag("e"), tag("E"))), exponent_num))(i)?;

    let unit_chars =
        one_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ%_/$");
    let (i, unit) = opt(many1(unit_chars))(i)?;
    let unit = unit.map(|s| s.into_iter().collect());

    let integral = if is_integral_negative {
        format!("-{}", integral)
    } else {
        integral.to_owned()
    };

    let opt_fractional = opt_fractional.map(|f| f.to_owned());
    let opt_exponent = match exponent_info {
        Some((None, exponent)) => Some(exponent.to_owned()),
        Some((Some(_), exponent)) => Some(format!("-{}", exponent)),
        None => None,
    };

    let lit = Literal::num(integral, opt_fractional, opt_exponent, unit);
    Ok((i, lit))
}

fn parse_expr(input: &str) -> IResult<&str, Expr> {
    alt((
        map(parse_condorexpr, Expr::CondOr),
        map(parse_if, Expr::new_if),
        parse_try_catch_expr,
        parse_return_expr,
        parse_throw_expr,
        parse_do_block_expr,
        parse_literal_expr, // TODO Axon grammar says literals are parsed in <assignExpr>
        parse_def_expr,
        parse_assignment_expr,
        map(parse_termexpr, Expr::TermExpr),
        // parse_assignexpr_expr,
    ))(input)
}

// fn parse_assignexpr_expr(input: &str) -> IResult<&str, Expr> {
//     map(parse_assignexpr, |assignexpr| Expr::new_assign(assignexpr))(input)
// }

// fn parse_assignexpr(input: &str) -> IResult<&str, AssignExpr> {
//     let (input, first_cond_or_expr) = parse_condorexpr(input)?;
//     let (input, _) = opt(gap)(input)?;
//     let parse_equals = terminated(tag("="), opt(gap));
//     let mut parse_next_cond_or_expr =
//         opt(preceded(parse_equals, parse_assignexpr));
//     let (input, opt_cond_or_expr) = parse_next_cond_or_expr(input)?;

//     let assign_expr = AssignExpr::new(first_cond_or_expr, opt_cond_or_expr);
//     Ok((input, assign_expr))
// }

fn parse_condorexpr(input: &str) -> IResult<&str, CondOrExpr> {
    let (input, first_cond_and_expr) = parse_condandexpr(input)?;
    let parse_and = delimited(opt(gap), tag("and"), opt(gap));
    let and_another_cond = preceded(parse_and, parse_condandexpr);
    let (input, remaining_cond_and_exprs) = many0(and_another_cond)(input)?;
    let cond_or_expr =
        CondOrExpr::new(first_cond_and_expr, remaining_cond_and_exprs);
    Ok((input, cond_or_expr))
}

fn parse_condandexpr(input: &str) -> IResult<&str, CondAndExpr> {
    let (input, first_compare_expr) = parse_compareexpr(input)?;
    let parse_or = delimited(opt(gap), tag("or"), opt(gap));
    let or_another_cond = preceded(parse_or, parse_compareexpr);
    let (input, remaining_compare_exprs) = many0(or_another_cond)(input)?;
    let cond_and_expr =
        CondAndExpr::new(first_compare_expr, remaining_compare_exprs);
    Ok((input, cond_and_expr))
}

fn parse_compareexpr(input: &str) -> IResult<&str, CompareExpr> {
    let (input, first_range_expr) = parse_rangeexpr(input)?;
    let (input, _) = opt(gap)(input)?;
    let (input, compared_range_exprs) = many0(parse_comparedrangeexprs)(input)?;
    let compare_expr = CompareExpr::new(first_range_expr, compared_range_exprs);
    Ok((input, compare_expr))

    let (input, left_add_expr) = parse_addexpr(input)?;
    let mut parse_dots = delimited(opt(gap), tag(".."), opt(gap));
    let (input, _) = parse_dots(input)?;
    let (input, right_add_expr) = parse_addexpr(input)?;
    let range_expr = RangeExpr::new(left_add_expr, right_add_expr);
    Ok((input, range_expr))
}

fn parse_comparedrangeexprs(i: &str) -> IResult<&str, ComparedRangeExpr> {
    let operators = alt((
        tag("=="),
        tag("!="),
        tag("<"),
        tag("<="),
        tag(">="),
        tag(">"),
        tag("<=>"),
    ));
    let (i, operator) = terminated(operators, opt(gap))(i)?;
    let (i, range_expr) = parse_rangeexpr(i)?;

    let comparison_op = ComparisonOperator::from_str(operator)
        .expect("Unimplemented comparison operator");
    let comp_range_expr = ComparedRangeExpr::new(comparison_op, range_expr);
    Ok((i, comp_range_expr))
}

fn parse_rangeexpr(input: &str) -> IResult<&str, RangeExpr> {
    let (input, left_add_expr) = parse_addexpr(input)?;
    let mut parse_dots = delimited(opt(gap), tag(".."), opt(gap));
    let (input, _) = parse_dots(input)?;
    let (input, right_add_expr) = parse_addexpr(input)?;
    let range_expr = RangeExpr::new(left_add_expr, right_add_expr);
    Ok((input, range_expr))
}

fn parse_addexpr(i: &str) -> IResult<&str, AddExpr> {
    let (i, first_mult_expr) = parse_multexpr(i)?;
    let (i, remaining_signed_mult_exprs) =
        many0(preceded(opt(gap), parse_signedmultexpr))(i)?;
    let add_expr = AddExpr::new(first_mult_expr, remaining_signed_mult_exprs);
    Ok((i, add_expr))
}

fn parse_signedmultexpr(i: &str) -> IResult<&str, SignedMultExpr> {
    let parse_symbol = alt((tag("+"), tag("-")));
    let mut parse_symbol_and_gap = terminated(parse_symbol, opt(gap));
    let (i, symbol) = parse_symbol_and_gap(i)?;
    let (i, mult_expr) = parse_multexpr(i)?;
    let is_add = symbol == "+";
    let signed_mult_expr = SignedMultExpr::new(is_add, mult_expr);
    Ok((i, signed_mult_expr))
}

fn parse_multexpr(i: &str) -> IResult<&str, MultExpr> {
    let (i, first_unary_expr) = parse_unaryexpr(i)?;
    let (i, remaining_operated_unary_exprs) =
        many0(preceded(opt(gap), parse_operatedunaryexprs))(i)?;
    let mult_expr =
        MultExpr::new(first_unary_expr, remaining_operated_unary_exprs);
    Ok((i, mult_expr))
}

fn parse_operatedunaryexprs(i: &str) -> IResult<&str, OperatedUnaryExpr> {
    let parse_operator = alt((tag("*"), tag("/")));
    let mut parse_operator_and_gap = terminated(parse_operator, opt(gap));
    let (i, operator) = parse_operator_and_gap(i)?;
    let (i, unary_expr) = parse_unaryexpr(i)?;
    let is_multiply = operator == "*";
    let operated_unary_expr = OperatedUnaryExpr::new(is_multiply, unary_expr);
    Ok((i, operated_unary_expr))
}

fn parse_unaryexpr(i: &str) -> IResult<&str, UnaryExpr> {
    let parse_minus = terminated(tag("-"), opt(gap));
    let parse_not = terminated(tag("not"), gap);
    let (i, opt_sign) = opt(alt((parse_minus, parse_not)))(i)?;
    let (i, term_expr) = parse_termexpr(i)?;

    let unary_expr_sign = opt_sign.map(|sign| {
        UnaryExprSign::from_str(sign)
            .expect("Unimplemented unary expression sign")
    });
    let unary_expr = UnaryExpr::new(unary_expr_sign, term_expr);
    Ok((i, unary_expr))
}

fn parse_termexpr(i: &str) -> IResult<&str, TermExpr> {
    let (i, term_base) = parse_term_base(i)?;
    let (i, term_chains) = many0(preceded(opt(gap), parse_term_chain))(i)?;
    let term_expr = TermExpr::new(term_base, term_chains);
    Ok((i, term_expr))
}

fn parse_term_base(i: &str) -> IResult<&str, TermBase> {
    if let Ok((i, expr)) = parse_grouped_expr(i) {
        let term_base = TermBase::GroupedExpr(Box::new(expr));
        return Ok((i, term_base));
    };
    if let Ok((i, literal)) = parse_literal(i) {
        let term_base = TermBase::Literal(literal);
        return Ok((i, term_base));
    };
    map(parse_id, |id| TermBase::Var(id))(i)
}

fn parse_grouped_expr(i: &str) -> IResult<&str, Expr> {
    let open_paren = terminated(tag("("), optgap);
    let (i, expr) = preceded(open_paren, parse_expr)(i)?;

    println!("{:?}", expr);

    let (i, _) = preceded(optgap, tag(")"))(i)?;

    Ok((i, expr))
}

fn parse_term_chain(i: &str) -> IResult<&str, TermChain> {
    alt((
        map(parse_call, TermChain::Call),
        map(parse_dot_call, TermChain::DotCall),
        parse_index,
        parse_trap_call,
    ))(i)
}

fn parse_dot_call(i: &str) -> IResult<&str, DotCall> {
    let (i, _) = tag(".")(i)?;
    let (i, id) = parse_id(i)?;
    let parse_trailing = alt((
        map(parse_call, TrailingCall::Call),
        map(parse_lambda, TrailingCall::Lambda),
    ));
    let (i, opt_trailing_call) = opt(preceded(opt(gap), parse_trailing))(i)?;
    let dot_call = DotCall::new(id, opt_trailing_call);
    Ok((i, dot_call))
}

fn parse_call(i: &str) -> IResult<&str, Call> {
    let open_paren = terminated(tag("("), opt(gap));
    let close_paren = preceded(opt(gap), tag(")"));
    let parse_comma = delimited(opt(gap), tag(","), opt(gap));
    let parse_remaining_call_args =
        many0(preceded(parse_comma, parse_call_arg));
    let parse_call_args =
        opt(tuple((parse_call_arg, parse_remaining_call_args)));
    let mut parse_call_args_in_paren =
        delimited(open_paren, parse_call_args, close_paren);
    let (i, opt_all_call_args) = parse_call_args_in_paren(i)?;

    let call_args: Vec<CallArg> = match opt_all_call_args {
        Some((first_call_args, mut remaining_call_args)) => {
            remaining_call_args.insert(0, first_call_args);
            remaining_call_args
        }
        None => vec![],
    };

    let (i, opt_lambda) = opt(preceded(opt(gap), parse_lambda))(i)?;
    let call = Call::new(call_args, opt_lambda);
    Ok((i, call))
}

fn parse_lambda(i: &str) -> IResult<&str, Lambda> {
    alt((
        map(parse_lambda_many, Lambda::Many),
        map(parse_lambda_one, Lambda::One),
    ))(i)
}

fn parse_lambda_many(i: &str) -> IResult<&str, LambdaMany> {
    let parse_arrow = delimited(opt(gap), tag("=>"), opt(gap));
    let (i, params) = terminated(parse_params, parse_arrow)(i)?;
    let (i, expr) = parse_expr(i)?;
    let lambda = LambdaMany::new(params, expr);
    Ok((i, lambda))
}

fn parse_params(i: &str) -> IResult<&str, Vec<Param>> {
    let open_paren = terminated(tag("("), opt(gap));
    let close_paren = preceded(opt(gap), tag(")"));

    let parse_comma = delimited(opt(gap), tag(","), opt(gap));
    let parse_subsequent_param = preceded(parse_comma, parse_param);
    let parse_subsequent_params = many0(parse_subsequent_param);
    let parse_all_params = opt(tuple((parse_param, parse_subsequent_params)));

    let (i, opt_params) =
        delimited(open_paren, parse_all_params, close_paren)(i)?;
    let params = match opt_params {
        Some((first_param, mut remaining_params)) => {
            remaining_params.insert(0, first_param);
            remaining_params
        }
        None => vec![],
    };
    Ok((i, params))
}

fn parse_param(i: &str) -> IResult<&str, Param> {
    let (i, id) = parse_id(i)?;
    let parse_colon = delimited(opt(gap), tag(":"), opt(gap));
    let parse_colon_and_expr = preceded(parse_colon, parse_expr);
    let (i, opt_expr) = opt(parse_colon_and_expr)(i)?;
    let param = Param::new(id, opt_expr);
    Ok((i, param))
}

fn parse_lambda_one(i: &str) -> IResult<&str, LambdaOne> {
    let parse_arrow = delimited(opt(gap), tag("=>"), opt(gap));
    let (i, id) = terminated(parse_id, parse_arrow)(i)?;
    let (i, expr) = parse_expr(i)?;
    let lambda = LambdaOne::new(id, expr);
    Ok((i, lambda))
}

fn parse_call_arg(i: &str) -> IResult<&str, CallArg> {
    let res: IResult<&str, &str> = tag("_")(i);
    if let Ok((i, _)) = res {
        return Ok((i, CallArg::DiscardedExpr));
    };
    map(parse_expr, CallArg::Expr)(i)
}

fn parse_trap_call(i: &str) -> IResult<&str, TermChain> {
    let parse_trap = terminated(tag("->"), opt(gap));
    map(preceded(parse_trap, parse_id), TermChain::TrapCall)(i)
}

fn parse_index(i: &str) -> IResult<&str, TermChain> {
    let open_bracket = terminated(tag("["), opt(gap));
    let close_bracket = preceded(opt(gap), tag("]"));
    let (i, expr) = delimited(open_bracket, parse_expr, close_bracket)(i)?;
    Ok((i, TermChain::Index(Box::new(expr))))
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

fn parse_double_quoted_string_literal(input: &str) -> IResult<&str, Literal> {
    let (input, string_contents) = parse_double_quoted_string_contents(input)?;
    let expr = Literal::str(string_contents);
    Ok((input, expr))
}

fn parse_double_quoted_string_contents(input: &str) -> IResult<&str, &str> {
    let esc = escaped(none_of("\\\""), '\\', one_of(r#"fnrtv\"$"#));
    let esc_or_empty = alt((esc, tag("")));
    let res = delimited(tag(r#"""#), esc_or_empty, tag(r#"""#))(input)?;

    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_expr(n: i32) -> Expr {
        Expr::new_literal(Literal::int(n))
    }

    fn bool_expr(b: bool) -> Expr {
        Expr::new_literal(Literal::Bool(b))
    }

    fn term_base_into_addexpr(base: TermBase) -> AddExpr {
        AddExpr::new(
            MultExpr::new(
                UnaryExpr::new(None, TermExpr::new(base, vec![])),
                vec![],
            ),
            vec![],
        )
    }

    #[test]
    fn parse_rangeexpr_works_for_int_range() {
        let s = "1..5 ";
        let left = term_base_into_addexpr(TermBase::Literal(Literal::int(1)));
        let right = term_base_into_addexpr(TermBase::Literal(Literal::int(5)));
        let e = RangeExpr::new(left, right);
        assert_eq!(parse_rangeexpr(s).unwrap(), (" ", e));

        let s = "1 ..\n5 ";
        let left = term_base_into_addexpr(TermBase::Literal(Literal::int(1)));
        let right = term_base_into_addexpr(TermBase::Literal(Literal::int(5)));
        let e = RangeExpr::new(left, right);
        assert_eq!(parse_rangeexpr(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_rangeexpr_works_for_var_name_range() {
        let s = "1..testVar ";
        let left = term_base_into_addexpr(TermBase::Literal(Literal::int(1)));
        let right = term_base_into_addexpr(TermBase::Var(Id::new("testVar")));
        let e = RangeExpr::new(left, right);
        assert_eq!(parse_rangeexpr(s).unwrap(), (" ", e));

        let s = "someVar .. \ntestVar ";
        let left = term_base_into_addexpr(TermBase::Var(Id::new("someVar")));
        let right = term_base_into_addexpr(TermBase::Var(Id::new("testVar")));
        let e = RangeExpr::new(left, right);
        assert_eq!(parse_rangeexpr(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_rangeexpr_works_for_expr_range() {
        let s = "(today() - 7day)..yesterday() ";
        let left = term_base_into_addexpr(TermBase::Literal(Literal::int(1)));
        let right = term_base_into_addexpr(TermBase::Var(Id::new("testVar")));
        let e = RangeExpr::new(left, right);
        assert_eq!(parse_rangeexpr(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_if_works_for_single_line_if_with_no_else() {
        let s = "if (true) 1 ";
        let e = If::new(bool_expr(true), wrap_in_do_block(int_expr(1)), None);
        assert_eq!(parse_if(s).unwrap(), (" ", e));

        let s = "if(true)1 ";
        let e = If::new(bool_expr(true), wrap_in_do_block(int_expr(1)), None);
        assert_eq!(parse_if(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_if_works_for_single_line_if_with_else() {
        let s = "if (true) 1 else 2 ";
        let e = If::new(
            bool_expr(true),
            wrap_in_do_block(int_expr(1)),
            Some(wrap_in_do_block(int_expr(2))),
        );
        assert_eq!(parse_if(s).unwrap(), (" ", e));

        let s = "if(true)1 else 2 ";
        let e = If::new(
            bool_expr(true),
            wrap_in_do_block(int_expr(1)),
            Some(wrap_in_do_block(int_expr(2))),
        );
        assert_eq!(parse_if(s).unwrap(), (" ", e));

        let s = "if(true)1 elseSomeFunction() ";
        let e = If::new(bool_expr(true), wrap_in_do_block(int_expr(1)), None);
        assert_eq!(parse_if(s).unwrap(), (" elseSomeFunction() ", e));
    }

    #[test]
    fn parse_if_works_for_single_line_if_with_end_else() {
        let s = "if (true) do 1 end else 2 ";
        let e = If::new(
            bool_expr(true),
            wrap_in_do_block(int_expr(1)),
            Some(wrap_in_do_block(int_expr(2))),
        );
        assert_eq!(parse_if(s).unwrap(), (" ", e));

        let s = "if(true)do 1 else 2 ";
        let e = If::new(
            bool_expr(true),
            wrap_in_do_block(int_expr(1)),
            Some(wrap_in_do_block(int_expr(2))),
        );
        assert_eq!(parse_if(s).unwrap(), (" ", e));

        let s = "if(true)1 endSomeFunction() ";
        let e = If::new(bool_expr(true), wrap_in_do_block(int_expr(1)), None);
        assert_eq!(parse_if(s).unwrap(), (" endSomeFunction() ", e));
    }

    #[test]
    fn parse_if_works_for_multi_line_if_with_no_else() {
        let s = "if (true)\n    1 ";
        let e = If::new(bool_expr(true), wrap_in_do_block(int_expr(1)), None);
        assert_eq!(parse_if(s).unwrap(), (" ", e));

        let s = "if(true)\n     1 ";
        let e = If::new(bool_expr(true), wrap_in_do_block(int_expr(1)), None);
        assert_eq!(parse_if(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_if_works_for_multi_line_if_with_else() {
        let s = "if (true)\n 1 \nelse\n 2 ";
        let e = If::new(
            bool_expr(true),
            wrap_in_do_block(int_expr(1)),
            Some(wrap_in_do_block(int_expr(2))),
        );
        assert_eq!(parse_if(s).unwrap(), (" ", e));

        let s = "if(true)\n1 else \n2 ";
        let e = If::new(
            bool_expr(true),
            wrap_in_do_block(int_expr(1)),
            Some(wrap_in_do_block(int_expr(2))),
        );
        assert_eq!(parse_if(s).unwrap(), (" ", e));

        let s = "if(true)1 elseSomeFunction() ";
        let e = If::new(bool_expr(true), wrap_in_do_block(int_expr(1)), None);
        assert_eq!(parse_if(s).unwrap(), (" elseSomeFunction() ", e));
    }

    #[test]
    fn parse_if_works_for_multi_line_if_with_end_else() {
        let s = "if (true) do\n 1\n end\nelse \n2 ";
        let e = If::new(
            bool_expr(true),
            wrap_in_do_block(int_expr(1)),
            Some(wrap_in_do_block(int_expr(2))),
        );
        assert_eq!(parse_if(s).unwrap(), (" ", e));

        let s = "if(true)\ndo 1\n \nelse 2 ";
        let e = If::new(
            bool_expr(true),
            wrap_in_do_block(int_expr(1)),
            Some(wrap_in_do_block(int_expr(2))),
        );
        assert_eq!(parse_if(s).unwrap(), (" ", e));

        let s = "if(true)1 endSomeFunction() ";
        let e = If::new(bool_expr(true), wrap_in_do_block(int_expr(1)), None);
        assert_eq!(parse_if(s).unwrap(), (" endSomeFunction() ", e));
    }

    #[test]
    fn parse_if_works_for_if_in_else() {
        let s = "if (true) do\n 1\n end\nelse if (false)\ndo 2 end ";
        let nested_if =
            If::new(bool_expr(false), wrap_in_do_block(int_expr(2)), None);
        let e = If::new(
            bool_expr(true),
            wrap_in_do_block(int_expr(1)),
            Some(wrap_in_do_block(Expr::If(Box::new(nested_if)))),
        );
        assert_eq!(parse_if(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_if_works_for_if_in_else2() {
        let s = "if (true) do\n 1\n \nelse if (false)\ndo 2 end else 3 ";
        let nested_if = If::new(
            bool_expr(false),
            wrap_in_do_block(int_expr(2)),
            Some(wrap_in_do_block(int_expr(3))),
        );
        let e = If::new(
            bool_expr(true),
            wrap_in_do_block(int_expr(1)),
            Some(wrap_in_do_block(Expr::If(Box::new(nested_if)))),
        );
        assert_eq!(parse_if(s).unwrap(), (" ", e));
    }

    #[test]
    fn parsing_basic_function_works() {
        let f = r#"
        () => do
            x: 5
            x = 10
            x.toStr()
            x
        end
        "#;
        parse_function(f).unwrap();
    }

    #[test]
    fn parsing_intermediate_function_works() {
        let f = r#"
        (siteRefOrPlaceholder,dates) => do
equip: read(equip and siteRefOrPlaceholder)

volumes: readAll(volume and equipRef == equip->id)
voumeHistory: volumes.hrnl(dates).hisRollup(avg,15min).hisFindAll(value => not isNaNanOrNull(value))

//scenario 1
volumeDown:voumeHistory.sfxHisDelta.hisFindAll(x=>x<0)
scenarioOneConsumption: volumeDown.foldCol("v0",sum).abs

//scenario 2
volumeUp:voumeHistory.sfxHisDelta.hisFindAll(x=>x>0)
        gas: toGas(equip)
        gasStatusPoint: gas.toPoint("status")

        gasOnPeriods: gasStatusPoint
            .hisReadNoLimit(dates)
            .hisFindPeriods(isGasOn => isGasOn)

scenarioTwoConsumption: volumeUp.hisFindInPeriods(gasOnPeriods).size*5*15 // 15 assumption

totalUsage: scenarioOneConsumption + scenarioTwoConsumption
return totalUsage

end
        "#;
        parse_function(f).unwrap();
    }

    fn simple_multexpr() -> MultExpr {
        parse_multexpr("-0").unwrap().1
    }

    fn simple_unaryexpr() -> UnaryExpr {
        parse_unaryexpr("-0").unwrap().1
    }

    #[test]
    fn parse_simple_assignment_works() {
        let s = "x = 5 ";
        let e = Expr::new_assignment(Id::new("x"), int_expr(5));
        assert_eq!(parse_assignment_expr(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_addexpr_works() {
        let s = "-0 ";
        let e = AddExpr::new(simple_multexpr(), vec![]);
        assert_eq!(parse_addexpr(s).unwrap(), (" ", e));

        let s = "-0 + -0--0 ";
        let remaining_signed_mult_exprs = vec![
            SignedMultExpr::new(true, simple_multexpr()),
            SignedMultExpr::new(false, simple_multexpr()),
        ];
        let e = AddExpr::new(simple_multexpr(), remaining_signed_mult_exprs);
        assert_eq!(parse_addexpr(s).unwrap(), (" ", e))
    }

    #[test]
    fn parse_addexpr_works_for_simple_expr() {
        let s = "today() - 7day ";

        let left_base = TermBase::Var(Id::new("today"));
        let left_chain = TermChain::Call(Call::new(vec![], None));
        let first_mult = MultExpr::new(
            UnaryExpr::new(None, TermExpr::new(left_base, vec![left_chain])),
            vec![],
        );
        let next_base = TermBase::Literal(Literal::num(
            "7".to_owned(),
            None,
            None,
            Some("day".to_owned()),
        ));
        let next_mult = MultExpr::new(
            UnaryExpr::new(None, TermExpr::new(next_base, vec![])),
            vec![],
        );

        let remaining_signed_mult_exprs =
            vec![SignedMultExpr::new(false, next_mult)];
        let e = AddExpr::new(first_mult, remaining_signed_mult_exprs);
        assert_eq!(parse_addexpr(s).unwrap(), (" ", e))
    }

    #[test]
    fn parse_multexpr_works() {
        let s = "-0 ";
        let e = MultExpr::new(simple_unaryexpr(), vec![]);
        assert_eq!(parse_multexpr(s).unwrap(), (" ", e));

        let s = "-0 * -0 ";
        let op_unary_expr = OperatedUnaryExpr::new(true, simple_unaryexpr());
        let e = MultExpr::new(simple_unaryexpr(), vec![op_unary_expr]);
        assert_eq!(parse_multexpr(s).unwrap(), (" ", e));

        let s = "-0*-0 /\n-0 ";
        let remaining_operated_unary_exprs = vec![
            OperatedUnaryExpr::new(true, simple_unaryexpr()),
            OperatedUnaryExpr::new(false, simple_unaryexpr()),
        ];
        let e =
            MultExpr::new(simple_unaryexpr(), remaining_operated_unary_exprs);
        assert_eq!(parse_multexpr(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_unaryexpr_minus_works() {
        let s = "-5 ";
        let e = UnaryExpr::new(
            Some(UnaryExprSign::Minus),
            TermExpr::new(TermBase::Literal(Literal::int(5)), vec![]),
        );
        assert_eq!(parse_unaryexpr(s).unwrap(), (" ", e));

        let s = "- \n 5 ";
        let e = UnaryExpr::new(
            Some(UnaryExprSign::Minus),
            TermExpr::new(TermBase::Literal(Literal::int(5)), vec![]),
        );
        assert_eq!(parse_unaryexpr(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_unaryexpr_not_works() {
        let s = "not someBool ";
        let e = UnaryExpr::new(
            Some(UnaryExprSign::Not),
            TermExpr::new(TermBase::Var(Id::new("someBool")), vec![]),
        );
        assert_eq!(parse_unaryexpr(s).unwrap(), (" ", e));

        let s = "not\nsomeBool ";
        let e = UnaryExpr::new(
            Some(UnaryExprSign::Not),
            TermExpr::new(TermBase::Var(Id::new("someBool")), vec![]),
        );
        assert_eq!(parse_unaryexpr(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_unaryexpr_works_with_no_sign() {
        let s = "someBool ";
        let e = UnaryExpr::new(
            None,
            TermExpr::new(TermBase::Var(Id::new("someBool")), vec![]),
        );
        assert_eq!(parse_unaryexpr(s).unwrap(), (" ", e));

        let s = "someBool ";
        let e = UnaryExpr::new(
            None,
            TermExpr::new(TermBase::Var(Id::new("someBool")), vec![]),
        );
        assert_eq!(parse_unaryexpr(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_grouped_expr_works() {
        let s = "(do 5 end) ";
        let e = Expr::new_do_block(DoBlock::new(Exprs::new(vec![int_expr(5)])));
        assert_eq!(parse_grouped_expr(s).unwrap(), (" ", e));

        let s = "(functionName()) ";
        let e = Expr::TermExpr(TermExpr::new(
            TermBase::Var(Id::new("functionName")),
            vec![TermChain::Call(Call::new(vec![], None))],
        ));
        assert_eq!(parse_grouped_expr(s).unwrap(), (" ", e))
    }

    #[test]
    fn test() {
        println!("here");
        let result = parse_expr("today(").unwrap();
        println!("{:?}", result);
    }

    #[test]
    fn parse_grouped_expr_works_for_complex_expr() {
        let s = "(today() - 7) ";
        let e = Expr::TermExpr(TermExpr::new(
            TermBase::Var(Id::new("functionName")),
            vec![TermChain::Call(Call::new(vec![], None))],
        ));
        assert_eq!(parse_grouped_expr(s).unwrap(), (" ", e))
    }

    #[test]
    fn parse_def_expr_works() {
        let s = "x: 5 ";
        let e = Expr::new_def(Id::new("x"), int_expr(5));
        assert_eq!(parse_def_expr(s).unwrap(), (" ", e));

        let s = "x:5 ";
        let e = Expr::new_def(Id::new("x"), int_expr(5));
        assert_eq!(parse_def_expr(s).unwrap(), (" ", e));

        let s = "x:\n    5 ";
        let e = Expr::new_def(Id::new("x"), int_expr(5));
        assert_eq!(parse_def_expr(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_termexpr_works() {
        let s = "1.toStr.toAxonCode()[5]->varName ";
        let e = TermExpr::new(
            TermBase::Literal(Literal::int(1)),
            vec![
                TermChain::DotCall(DotCall::new(Id::new("toStr"), None)),
                TermChain::DotCall(DotCall::new(
                    Id::new("toAxonCode"),
                    Some(TrailingCall::Call(Call::new(vec![], None))),
                )),
                TermChain::Index(Box::new(int_expr(5))),
                TermChain::TrapCall(Id::new("varName")),
            ],
        );
        assert_eq!(parse_termexpr(s).unwrap(), (" ", e))
    }

    #[test]
    fn parse_termexpr_works2() {
        let s = "1\n    .toStr\n    .toAxonCode ( ) \n[5]\n->\nvarName ";
        let e = TermExpr::new(
            TermBase::Literal(Literal::int(1)),
            vec![
                TermChain::DotCall(DotCall::new(Id::new("toStr"), None)),
                TermChain::DotCall(DotCall::new(
                    Id::new("toAxonCode"),
                    Some(TrailingCall::Call(Call::new(vec![], None))),
                )),
                TermChain::Index(Box::new(int_expr(5))),
                TermChain::TrapCall(Id::new("varName")),
            ],
        );
        assert_eq!(parse_termexpr(s).unwrap(), (" ", e))
    }

    #[test]
    fn parse_termexpr_works_with_trailing_lambda() {
        let s = "1.map varName=> 2 ";
        let lambda =
            Lambda::One(LambdaOne::new(Id::new("varName"), int_expr(2)));
        let e = TermExpr::new(
            TermBase::Literal(Literal::int(1)),
            vec![TermChain::DotCall(DotCall::new(
                Id::new("map"),
                Some(TrailingCall::Lambda(lambda)),
            ))],
        );
        assert_eq!(parse_termexpr(s).unwrap(), (" ", e))
    }

    #[test]
    fn parse_termexpr_works_with_trailing_lambda2() {
        // This should be interpretted as varName being passed as an
        // argument to map, not as part of the trailing lambda for map.
        let s = "1.map (varName)=> 2 ";
        let var_name = TermExpr::new(TermBase::Var(Id::new("varName")), vec![]);
        let call =
            Call::new(vec![CallArg::Expr(Expr::TermExpr(var_name))], None);
        let e = TermExpr::new(
            TermBase::Literal(Literal::int(1)),
            vec![TermChain::DotCall(DotCall::new(
                Id::new("map"),
                Some(TrailingCall::Call(call)),
            ))],
        );
        assert_eq!(parse_termexpr(s).unwrap(), ("=> 2 ", e))
    }

    #[test]
    fn parse_termexpr_works_with_trailing_lambda3() {
        let s = "1.map ( ) (varName)=> 2 ";
        let param = Param::new(Id::new("varName"), None);
        let lambda = Lambda::Many(LambdaMany::new(vec![param], int_expr(2)));
        let call = Call::new(vec![], Some(lambda));
        let e = TermExpr::new(
            TermBase::Literal(Literal::int(1)),
            vec![TermChain::DotCall(DotCall::new(
                Id::new("map"),
                Some(TrailingCall::Call(call)),
            ))],
        );
        assert_eq!(parse_termexpr(s).unwrap(), (" ", e))
    }

    #[test]
    fn parse_dot_call_works_with_no_trailing_call() {
        let s = ".funcName ";
        let e = DotCall::new(Id::new("funcName"), None);
        assert_eq!(parse_dot_call(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_dot_call_works_with_simple_trailing_call() {
        let s = ".funcName ( ) ";
        let call = Call::new(vec![], None);
        let e =
            DotCall::new(Id::new("funcName"), Some(TrailingCall::Call(call)));
        assert_eq!(parse_dot_call(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_dot_call_works_with_simple_lambda() {
        let s = ".funcName varName => 1 ";
        let lambda =
            Lambda::One(LambdaOne::new(Id::new("varName"), int_expr(1)));
        let e = DotCall::new(
            Id::new("funcName"),
            Some(TrailingCall::Lambda(lambda)),
        );
        assert_eq!(parse_dot_call(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_call_works_with_one_arg_and_no_lambda() {
        let s = "(1) ";
        let arg1 = CallArg::Expr(int_expr(1));
        let e = Call::new(vec![arg1], None);
        assert_eq!(parse_call(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_call_works_with_no_args_and_no_lambda() {
        let s = "() ";
        let e = Call::new(vec![], None);
        assert_eq!(parse_call(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_call_works_with_args_and_no_lambda() {
        let s = "(1, 2 , \n 3 \t ) ";
        let arg1 = CallArg::Expr(int_expr(1));
        let arg2 = CallArg::Expr(int_expr(2));
        let arg3 = CallArg::Expr(int_expr(3));
        let e = Call::new(vec![arg1, arg2, arg3], None);
        assert_eq!(parse_call(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_call_works_with_discarded_args_and_no_lambda() {
        let s = "(_, 2 , \n _ \t,_, 5 ,_) ";
        let arg1 = CallArg::DiscardedExpr;
        let arg2 = CallArg::Expr(int_expr(2));
        let arg3 = CallArg::DiscardedExpr;
        let arg4 = CallArg::DiscardedExpr;
        let arg5 = CallArg::Expr(int_expr(5));
        let arg6 = CallArg::DiscardedExpr;
        let e = Call::new(vec![arg1, arg2, arg3, arg4, arg5, arg6], None);
        assert_eq!(parse_call(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_call_works_with_discarded_args_and_simple_lambda() {
        let s = "(_, 2 , \n _ \t,_, 5 ,_) lambdaVar  => 999 ";
        let arg1 = CallArg::DiscardedExpr;
        let arg2 = CallArg::Expr(int_expr(2));
        let arg3 = CallArg::DiscardedExpr;
        let arg4 = CallArg::DiscardedExpr;
        let arg5 = CallArg::Expr(int_expr(5));
        let arg6 = CallArg::DiscardedExpr;
        let lambda =
            Lambda::One(LambdaOne::new(Id::new("lambdaVar"), int_expr(999)));
        let e =
            Call::new(vec![arg1, arg2, arg3, arg4, arg5, arg6], Some(lambda));
        assert_eq!(parse_call(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_call_works_with_discarded_args_and_simple_lambda2() {
        let s = "(_, 2 , \n _ \t,_, 5 ,_)(lambdaVar1, lambdaVar2:33)=> 999 ";
        let arg1 = CallArg::DiscardedExpr;
        let arg2 = CallArg::Expr(int_expr(2));
        let arg3 = CallArg::DiscardedExpr;
        let arg4 = CallArg::DiscardedExpr;
        let arg5 = CallArg::Expr(int_expr(5));
        let arg6 = CallArg::DiscardedExpr;

        let params = vec![
            Param::new(Id::new("lambdaVar1"), None),
            Param::new(Id::new("lambdaVar2"), Some(int_expr(33))),
        ];
        let lambda = Lambda::Many(LambdaMany::new(params, int_expr(999)));
        let e =
            Call::new(vec![arg1, arg2, arg3, arg4, arg5, arg6], Some(lambda));
        assert_eq!(parse_call(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_lambda_many_works_on_simplest_lambda() {
        let s = "() => 1 ";
        let e = LambdaMany::new(vec![], int_expr(1));
        assert_eq!(parse_lambda_many(s).unwrap(), (" ", e));

        let s = "( )=>\n 2 ";
        let e = LambdaMany::new(vec![], int_expr(2));
        assert_eq!(parse_lambda_many(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_lambda_many_works_with_one_param() {
        let s = "( arg\n) => 1 ";
        let params = vec![Param::new(Id::new("arg"), None)];
        let e = LambdaMany::new(params, int_expr(1));
        assert_eq!(parse_lambda_many(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_lambda_many_works_with_one_param_and_default_expr() {
        let s = "(arg:5) => 1 ";
        let params = vec![Param::new(Id::new("arg"), Some(int_expr(5)))];
        let e = LambdaMany::new(params, int_expr(1));
        assert_eq!(parse_lambda_many(s).unwrap(), (" ", e));

        let s = "(arg\n:\t 5 ) =>\n 1 ";
        let params = vec![Param::new(Id::new("arg"), Some(int_expr(5)))];
        let e = LambdaMany::new(params, int_expr(1));
        assert_eq!(parse_lambda_many(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_lambda_many_works_with_many_params() {
        let s = "(\n arg1,\t arg2 , arg3) => 1 ";
        let params = vec![
            Param::new(Id::new("arg1"), None),
            Param::new(Id::new("arg2"), None),
            Param::new(Id::new("arg3"), None),
        ];
        let e = LambdaMany::new(params, int_expr(1));
        assert_eq!(parse_lambda_many(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_lambda_one_works() {
        let s = "varName=>1 ";
        let e = LambdaOne::new(Id::new("varName"), int_expr(1));
        assert_eq!(parse_lambda_one(s).unwrap(), (" ", e));

        let s = "varName\n => \t1 ";
        let e = LambdaOne::new(Id::new("varName"), int_expr(1));
        assert_eq!(parse_lambda_one(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_lambda_many_works_with_many_params_with_default_exprs() {
        let s = "(\n arg1,\t arg2 : 5 , arg3:6) => 1 ";
        let params = vec![
            Param::new(Id::new("arg1"), None),
            Param::new(Id::new("arg2"), Some(int_expr(5))),
            Param::new(Id::new("arg3"), Some(int_expr(6))),
        ];
        let e = LambdaMany::new(params, int_expr(1));
        assert_eq!(parse_lambda_many(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_index_works() {
        let s = "[1] ";
        let e = TermChain::Index(Box::new(Expr::Literal(Literal::int(1))));
        assert_eq!(parse_index(s).unwrap(), (" ", e));

        let s = "[  2 \t ] ";
        let e = TermChain::Index(Box::new(Expr::Literal(Literal::int(2))));
        assert_eq!(parse_index(s).unwrap(), (" ", e));

        let s = "[\n3\n] ";
        let e = TermChain::Index(Box::new(Expr::Literal(Literal::int(3))));
        assert_eq!(parse_index(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_trap_call_works() {
        let s = "->navName ";
        let e = TermChain::TrapCall(Id::new("navName"));
        assert_eq!(parse_trap_call(s).unwrap(), (" ", e));

        let s = "-> tag";
        let e = TermChain::TrapCall(Id::new("tag"));
        assert_eq!(parse_trap_call(s).unwrap(), ("", e));

        let s = "->\nonAnotherLine";
        let e = TermChain::TrapCall(Id::new("onAnotherLine"));
        assert_eq!(parse_trap_call(s).unwrap(), ("", e));
    }

    #[test]
    fn parse_ref_literal_works_for_common_ref() {
        let s = "@p:some_Project:r:1e85e02f-0459cf96 ";
        let expected = Literal::ref_lit("@p:some_Project:r:1e85e02f-0459cf96");
        assert_eq!(parse_ref_literal(s).unwrap(), (" ", expected));
    }

    #[test]
    fn parse_ref_literal_works_for_uncommon_ref() {
        let s = "@p:some_Project:r:1e85e02f-0459cf96~_-.: ";
        let expected =
            Literal::ref_lit("@p:some_Project:r:1e85e02f-0459cf96~_-.:");
        assert_eq!(parse_ref_literal(s).unwrap(), (" ", expected));
    }

    #[test]
    fn test_parse_naive_time_works() {
        let s = "1:23 ";
        let e = NaiveTime::from_hms(1, 23, 0);
        assert_eq!(parse_naive_time(s).unwrap(), (" ", e));

        let s = "1:23:45 ";
        let e = NaiveTime::from_hms(1, 23, 45);
        assert_eq!(parse_naive_time(s).unwrap(), (" ", e));

        let s = "1:23:45.678 ";
        let e = NaiveTime::from_hms_milli(1, 23, 45, 678);
        assert_eq!(parse_naive_time(s).unwrap(), (" ", e));

        let s = "21:23:45.678 ";
        let e = NaiveTime::from_hms_milli(21, 23, 45, 678);
        assert_eq!(parse_naive_time(s).unwrap(), (" ", e));
    }

    #[test]
    fn parse_month_literal_works() {
        let s = "1999-01 ";
        let month = Month::new(1999, 1).unwrap();
        let expected = Literal::month(month);
        assert_eq!(parse_month_literal(s).unwrap(), (" ", expected));

        let s = "2020-07 ";
        let month = Month::new(2020, 7).unwrap();
        let expected = Literal::month(month);
        assert_eq!(parse_month_literal(s).unwrap(), (" ", expected));
    }

    #[test]
    fn parse_date_literal_works() {
        let s = "1999-01-01 ";
        let expected = Literal::date(1999, 1, 1);
        assert_eq!(parse_date_literal(s).unwrap(), (" ", expected));

        let s = "2020-07-11 ";
        let expected = Literal::date(2020, 7, 11);
        assert_eq!(parse_date_literal(s).unwrap(), (" ", expected));
    }

    #[test]
    fn parse_null_literal_works() {
        let s = "null ";
        let expected = Literal::Null;
        assert_eq!(parse_null_literal(s).unwrap(), (" ", expected));
    }

    #[test]
    fn parse_bool_literal_works() {
        let t = "true ";
        let expected = Literal::bool(true);
        assert_eq!(parse_bool_literal(t).unwrap(), (" ", expected));

        let f = "false ";
        let expected = Literal::bool(false);
        assert_eq!(parse_bool_literal(f).unwrap(), (" ", expected));
    }

    #[test]
    fn parse_number_literal_works_for_basic_num() {
        let s = "123 ";
        let expected = Literal::int(123);
        assert_eq!(parse_number_literal(s).unwrap(), (" ", expected));

        let s = "-123 ";
        let expected = Literal::int(-123);
        assert_eq!(parse_number_literal(s).unwrap(), (" ", expected));
    }

    #[test]
    fn parse_number_literal_works_for_basic_decimal() {
        let s = "123.456 ";
        let expected =
            Literal::num("123".to_owned(), Some("456".to_owned()), None, None);
        assert_eq!(parse_number_literal(s).unwrap(), (" ", expected));

        let s = "-123.456 ";
        let expected =
            Literal::num("-123".to_owned(), Some("456".to_owned()), None, None);
        assert_eq!(parse_number_literal(s).unwrap(), (" ", expected));

        let s = "-123.456kPa ";
        let expected = Literal::num(
            "-123".to_owned(),
            Some("456".to_owned()),
            None,
            Some("kPa".to_owned()),
        );
        assert_eq!(parse_number_literal(s).unwrap(), (" ", expected));
    }

    #[test]
    fn parse_number_literal_works_for_exponent() {
        let s = "123e-10 ";
        let expected =
            Literal::num("123".to_owned(), None, Some("-10".to_owned()), None);
        assert_eq!(parse_number_literal(s).unwrap(), (" ", expected));

        let s = "-123.456E99 ";
        let expected = Literal::num(
            "-123".to_owned(),
            Some("456".to_owned()),
            Some("99".to_owned()),
            None,
        );
        assert_eq!(parse_number_literal(s).unwrap(), (" ", expected));

        let s = "-123.456E99kW ";
        let expected = Literal::num(
            "-123".to_owned(),
            Some("456".to_owned()),
            Some("99".to_owned()),
            Some("kW".to_owned()),
        );
        assert_eq!(parse_number_literal(s).unwrap(), (" ", expected));
    }

    #[test]
    fn parse_number_literal_works_for_a_mess() {
        let s = "-  \n \t 0.0E-0exampleUnit% ";
        let expected = Literal::num(
            "-0".to_owned(),
            Some("0".to_owned()),
            Some("-0".to_owned()),
            Some("exampleUnit%".to_owned()),
        );
        assert_eq!(parse_number_literal(s).unwrap(), (" ", expected));
    }

    fn wrap_in_do_block(expr: Expr) -> Expr {
        let exprs = Exprs::new(vec![expr]);
        let do_block = DoBlock::new(exprs);
        Expr::new_do_block(do_block)
    }

    #[test]
    fn parse_try_catch_works_on_single_line() {
        let s = "try 1 catch 2";
        let expect_try_expr =
            wrap_in_do_block(Expr::new_literal(Literal::int(1)));
        let expect_catch_expr =
            wrap_in_do_block(Expr::new_literal(Literal::int(2)));
        let expect =
            TryCatchBlock::new(expect_try_expr, None, expect_catch_expr);
        assert_eq!(parse_try_catch(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_try_catch_works_on_single_line_with_do_end() {
        let s = "try do \"a\" end catch do \"b\" end";
        let expect_try_expr =
            wrap_in_do_block(Expr::new_literal(Literal::str("a")));
        let expect_catch_expr =
            wrap_in_do_block(Expr::new_literal(Literal::str("b")));
        let expect =
            TryCatchBlock::new(expect_try_expr, None, expect_catch_expr);
        assert_eq!(parse_try_catch(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_try_catch_works_on_single_line_with_do_but_no_end() {
        let s = "try do \"a\" catch do \"b\"";
        let expect_try_expr =
            wrap_in_do_block(Expr::new_literal(Literal::str("a")));
        let expect_catch_expr =
            wrap_in_do_block(Expr::new_literal(Literal::str("b")));
        let expect =
            TryCatchBlock::new(expect_try_expr, None, expect_catch_expr);
        assert_eq!(parse_try_catch(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_try_catch_works_on_multi_lines() {
        let s = "try \n \"a\" \ncatch\n\"b\"";
        let expect_try_expr =
            wrap_in_do_block(Expr::new_literal(Literal::str("a")));
        let expect_catch_expr =
            wrap_in_do_block(Expr::new_literal(Literal::str("b")));
        let expect =
            TryCatchBlock::new(expect_try_expr, None, expect_catch_expr);
        assert_eq!(parse_try_catch(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_try_catch_works_on_multi_lines_with_do_end() {
        let s = "try do\n\"a\"end\ncatch do\n\"b\"end";
        let expect_try_expr =
            wrap_in_do_block(Expr::new_literal(Literal::str("a")));
        let expect_catch_expr =
            wrap_in_do_block(Expr::new_literal(Literal::str("b")));
        let expect =
            TryCatchBlock::new(expect_try_expr, None, expect_catch_expr);
        assert_eq!(parse_try_catch(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_try_catch_works_on_multi_lines_with_do_but_no_end() {
        let s = "try do\n\"a\"\ncatch do\n\"b\"";
        let expect_try_expr =
            wrap_in_do_block(Expr::new_literal(Literal::str("a")));
        let expect_catch_expr =
            wrap_in_do_block(Expr::new_literal(Literal::str("b")));
        let expect =
            TryCatchBlock::new(expect_try_expr, None, expect_catch_expr);
        assert_eq!(parse_try_catch(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_try_catch_works_on_a_mixed_bag() {
        let s = "try do   \n   \"a\"   \n\n\t\n   \"b\" \n end \ncatch do\n\"c\"\n\"d\"    \t";
        let a = Expr::new_literal(Literal::str("a"));
        let b = Expr::new_literal(Literal::str("b"));
        let try_exprs =
            Expr::new_do_block(DoBlock::new(Exprs::new(vec![a, b])));
        let c = Expr::new_literal(Literal::str("c"));
        let d = Expr::new_literal(Literal::str("d"));
        let catch_exprs =
            Expr::new_do_block(DoBlock::new(Exprs::new(vec![c, d])));
        let expect = TryCatchBlock::new(try_exprs, None, catch_exprs);
        assert_eq!(parse_try_catch(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_specific_keyword_works() {
        assert_eq!(
            parse_specific_keyword("do{", "do").unwrap(),
            ("{", "do".to_owned())
        );
        assert_eq!(
            parse_specific_keyword("do\n", "do").unwrap(),
            ("\n", "do".to_owned())
        );
        assert!(parse_specific_keyword("doFunction", "do").is_err());
    }

    #[test]
    fn parse_keyword_works() {
        assert_eq!(parse_keyword("and(").unwrap(), ("(", "and".to_owned()));
        assert_eq!(
            parse_keyword("catch\"").unwrap(),
            ("\"", "catch".to_owned())
        );
        assert_eq!(parse_keyword("try[").unwrap(), ("[", "try".to_owned()));
        assert_eq!(parse_keyword("throw{").unwrap(), ("{", "throw".to_owned()));
        assert_eq!(parse_keyword("do@").unwrap(), ("@", "do".to_owned()));
        assert_eq!(parse_keyword("end)").unwrap(), (")", "end".to_owned()));
        assert_eq!(parse_keyword("end}").unwrap(), ("}", "end".to_owned()));
        assert_eq!(parse_keyword("end]").unwrap(), ("]", "end".to_owned()));
        assert_eq!(parse_keyword("do\n").unwrap(), ("\n", "do".to_owned()));
        assert_eq!(parse_keyword("not ").unwrap(), (" ", "not".to_owned()));
        assert_eq!(parse_keyword("catch`").unwrap(), ("`", "catch".to_owned()));
        assert_eq!(parse_keyword("catch").unwrap(), ("", "catch".to_owned()));
    }

    #[test]
    fn parse_do_block_works_on_single_line() {
        let s = "do \"Hello\" end ";
        let expect_expr = Expr::Literal(Literal::Str("Hello".to_owned()));
        let expect_exprs = Exprs::new(vec![expect_expr]);
        let expect = DoBlock::new(expect_exprs);
        assert_eq!(parse_do_block(s).unwrap(), (" ", expect));
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
        let expect_expr =
            Expr::new_return(Expr::new_literal(Literal::str("test")));
        let expect = Exprs::new(vec![expect_expr]);
        assert_eq!(parse_exprs(s).unwrap(), ("", expect))
    }

    #[test]
    fn parse_exprs_works_for_semicolon_separated_exprs() {
        let s = "return \"one\"; return \"two\"";
        let expect_expr1 =
            Expr::new_return(Expr::new_literal(Literal::str("one")));
        let expect_expr2 =
            Expr::new_return(Expr::new_literal(Literal::str("two")));
        let expect = Exprs::new(vec![expect_expr1, expect_expr2]);
        assert_eq!(parse_exprs(s).unwrap(), ("", expect));

        let s = "return \"one\";return \"two\"";
        let expect_expr1 =
            Expr::new_return(Expr::new_literal(Literal::str("one")));
        let expect_expr2 =
            Expr::new_return(Expr::new_literal(Literal::str("two")));
        let expect = Exprs::new(vec![expect_expr1, expect_expr2]);
        assert_eq!(parse_exprs(s).unwrap(), ("", expect));

        let s = "return \"one\"  \t ; \t return \"two\"";
        let expect_expr1 =
            Expr::new_return(Expr::new_literal(Literal::str("one")));
        let expect_expr2 =
            Expr::new_return(Expr::new_literal(Literal::str("two")));
        let expect = Exprs::new(vec![expect_expr1, expect_expr2]);
        assert_eq!(parse_exprs(s).unwrap(), ("", expect));
    }

    #[test]
    fn parse_exprs_works_for_newline_separated_exprs() {
        let s = "return \"one\"\nreturn \"two\"";
        let expect_expr1 =
            Expr::new_return(Expr::new_literal(Literal::str("one")));
        let expect_expr2 =
            Expr::new_return(Expr::new_literal(Literal::str("two")));
        let expect = Exprs::new(vec![expect_expr1, expect_expr2]);
        assert_eq!(parse_exprs(s).unwrap(), ("", expect));

        let s = "return \"one\"  \t  \n  \t return \"two\"";
        let expect_expr1 =
            Expr::new_return(Expr::new_literal(Literal::str("one")));
        let expect_expr2 =
            Expr::new_return(Expr::new_literal(Literal::str("two")));
        let expect = Exprs::new(vec![expect_expr1, expect_expr2]);
        assert_eq!(parse_exprs(s).unwrap(), ("", expect));
    }
    #[test]
    fn parse_exprs_works_for_multi_newline_separated_exprs() {
        let s = "return \"one\"\n\n\n   return \"two\"";
        let expect_expr1 =
            Expr::new_return(Expr::new_literal(Literal::str("one")));
        let expect_expr2 =
            Expr::new_return(Expr::new_literal(Literal::str("two")));
        let expect = Exprs::new(vec![expect_expr1, expect_expr2]);
        assert_eq!(parse_exprs(s).unwrap(), ("", expect));

        let s = "return \"one\" \n \t  \n\n  \t return \"two\"";
        let expect_expr1 =
            Expr::new_return(Expr::new_literal(Literal::str("one")));
        let expect_expr2 =
            Expr::new_return(Expr::new_literal(Literal::str("two")));
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

        let s = "with_underscore";
        let expected = Id::new("with_underscore");
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
        let expect =
            Expr::new_def(expect_id, Expr::new_literal(Literal::str("Hello")));
        assert_eq!(parse_expr(e).unwrap(), ("", expect));
    }

    #[test]
    fn parse_expr_def_works_with_no_separating_whitespace() {
        let e = "x:\"Hello\"";
        let expect_id = Id::new("x");
        let expect =
            Expr::new_def(expect_id, Expr::new_literal(Literal::str("Hello")));
        assert_eq!(parse_expr(e).unwrap(), ("", expect));
    }

    #[test]
    fn parse_expr_def_works_with_many_separating_whitespace() {
        let e = "x: \n  \t   \"Hello\"";
        let expect_id = Id::new("x");
        let expect =
            Expr::new_def(expect_id, Expr::new_literal(Literal::str("Hello")));
        assert_eq!(parse_expr(e).unwrap(), ("", expect));
    }

    #[test]
    fn parse_expr_return_works() {
        let e = "return \"Hello\"";
        let expect = Expr::new_return(Expr::new_literal(Literal::str("Hello")));
        assert_eq!(parse_expr(e).unwrap(), ("", expect));
    }

    #[test]
    fn parse_expr_throw_works() {
        let e = "throw \"Some error\"";
        let expect =
            Expr::new_throw(Expr::new_literal(Literal::str("Some error")));
        assert_eq!(parse_expr(e).unwrap(), ("", expect));
    }

    #[test]
    fn parse_double_quoted_string_literal_works_on_normal_input() {
        let s = "\"Contents of this string is an Axon Str literal\"";
        assert_eq!(
            parse_double_quoted_string_contents(s).unwrap(),
            ("", "Contents of this string is an Axon Str literal")
        );
    }

    #[test]
    fn parse_double_quoted_string_literal_works_on_empty_input() {
        let s = "\"\"";
        assert_eq!(parse_double_quoted_string_contents(s).unwrap(), ("", ""));
    }

    #[test]
    fn parse_double_quoted_string_literal_works_on_input_with_escaped_double_quotes(
    ) {
        let s = r#""Name is \"Andrew\"""#;
        assert_eq!(
            parse_double_quoted_string_contents(s).unwrap(),
            ("", r#"Name is \"Andrew\""#)
        );
    }

    #[test]
    fn parse_double_quoted_string_literal_works_on_input_with_common_escaped_chars(
    ) {
        let s = r#""First\nSecond\nTab\tDone""#;
        assert_eq!(
            parse_double_quoted_string_contents(s).unwrap(),
            ("", r#"First\nSecond\nTab\tDone"#)
        );
    }

    #[test]
    fn parse_double_quoted_string_literal_works_on_input_with_uncommon_escaped_chars(
    ) {
        let s = r#""First\fSecond\rTab\vDone""#;
        assert_eq!(
            parse_double_quoted_string_contents(s).unwrap(),
            ("", r#"First\fSecond\rTab\vDone"#)
        );
    }

    #[test]
    fn parse_double_quoted_string_literal_works_on_input_with_escaped_backslash(
    ) {
        let s = r#""Backslash: \\""#;
        assert_eq!(
            parse_double_quoted_string_contents(s).unwrap(),
            ("", r#"Backslash: \\"#)
        );
    }

    #[test]
    fn parse_double_quoted_string_literal_works_on_input_with_escaped_dollar_sign(
    ) {
        let s = r#""\$equipRef \$navName""#;
        assert_eq!(
            parse_double_quoted_string_contents(s).unwrap(),
            ("", r#"\$equipRef \$navName"#)
        );
    }
}
