use axon_parseast_parser as ap;
use axon_parseast_parser::Val;
use chrono::{NaiveDate, NaiveTime};
use raystack_core::{Number, Qname, Ref, Symbol, TagName};
use std::collections::HashMap;
use std::convert::{From, TryFrom, TryInto};
use uuid::Uuid;

/// The size of a single block of indentation, the number of spaces (' ').
const SPACES: usize = 4;

struct Context {
    /// The parent expression within which this code is nested.
    parent_expr: Option<Expr>,
    /// The number of spaces across this code should be.
    indent: usize,
    /// The maximum width allowed for this code.
    max_width: usize,
}

trait Rewrite {
    fn rewrite(&self, context: Context) -> Option<String>;
}

// TODO later:
// defcomps don't seem to work in parseAst

// 1 - (2 + 3)
// Min (1) (2 + 3)
// 5    5

// precedence (1 is highest)

// do i need parenthesis?
// 1. look at parent precedence
// 2. if parent is lower  precedence: no
//    if parent is higher precedence: yes
//    if parent is same precedence:
//          if i am on same side as parent associativity: no
//          if i am on wrong side of parent associativity: yes

//"(if (true) utilsAssert else parseRef).params(if (true) true else false)"
//left if requires parens, right if does not

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Associativity {
    Left,
    Right,
}

macro_rules! impl_try_from_val_ref_for {
    ($type_name:ty, $bin_op:expr) => {
        impl TryFrom<&Val> for $type_name {
            type Error = ();

            fn try_from(val: &Val) -> Result<Self, Self::Error> {
                let op = val_to_bin_op(val, $bin_op).map_err(|_| ())?;
                Ok(Self(op))
            }
        }
    };
}

macro_rules! impl_line_and_lines_for {
    ($type_name:ty, $bin_op:expr) => {
        impl $type_name {
            pub fn to_line(&self, indent: &Indent) -> Line {
                let self_prec = Some(self.0.precedence());
                let bin_op = &self.0;
                let zero_indent = zero_indent();
                let left_line = bin_op.lhs.to_line(&zero_indent);

                let left_prec = bin_op.lhs.precedence();
                let left_line =
                    group_line_if_necessary(left_line, left_prec, self_prec);

                let right_line = bin_op.rhs.to_line(&zero_indent);

                let right_prec = bin_op.rhs.precedence();
                let right_line =
                    group_line_if_necessary(right_line, right_prec, self_prec);

                let left_str = left_line.inner_str();
                let right_str = right_line.inner_str();
                let op_symbol = $bin_op.to_symbol();
                Line::new(
                    indent.clone(),
                    format!("{} {} {}", left_str, op_symbol, right_str),
                )
            }

            pub fn to_lines(&self, indent: &Indent) -> Lines {
                let line = self.to_line(indent);
                vec![line]
            }
        }
    };
}

fn group_line_if_necessary(
    line: Line,
    line_precedence: Option<u8>,
    compare_precedence: Option<u8>,
) -> Line {
    match (line_precedence, compare_precedence) {
        (Some(line_prec), Some(compare_prec)) => {
            group_line_with_precedence_if_necessary(
                line,
                line_prec,
                compare_prec,
            )
        }
        _ => line,
    }
}

fn group_line_with_precedence_if_necessary(
    line: Line,
    line_precedence: u8,
    compare_precedence: u8,
) -> Line {
    use std::cmp::Ordering;

    match line_precedence.cmp(&compare_precedence) {
        Ordering::Greater => line.grouped(), // Since 1 = highest priority
        _ => line,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Add(BinOp);
impl_try_from_val_ref_for!(Add, BinOpId::Add);
impl_line_and_lines_for!(Add, BinOpId::Add);

#[derive(Debug, Clone, PartialEq)]
pub struct And(BinOp);
impl_try_from_val_ref_for!(And, BinOpId::And);
impl_line_and_lines_for!(And, BinOpId::And);

#[derive(Debug, Clone, PartialEq)]
pub struct Cmp(BinOp);
impl_try_from_val_ref_for!(Cmp, BinOpId::Cmp);
impl_line_and_lines_for!(Cmp, BinOpId::Cmp);

#[derive(Debug, Clone, PartialEq)]
pub struct Div(BinOp);
impl_try_from_val_ref_for!(Div, BinOpId::Div);
impl_line_and_lines_for!(Div, BinOpId::Div);

#[derive(Debug, Clone, PartialEq)]
pub struct Eq(BinOp);
impl_try_from_val_ref_for!(Eq, BinOpId::Eq);
impl_line_and_lines_for!(Eq, BinOpId::Eq);

#[derive(Debug, Clone, PartialEq)]
pub struct Gt(BinOp);
impl_try_from_val_ref_for!(Gt, BinOpId::Gt);
impl_line_and_lines_for!(Gt, BinOpId::Gt);

#[derive(Debug, Clone, PartialEq)]
pub struct Gte(BinOp);
impl_try_from_val_ref_for!(Gte, BinOpId::Gte);
impl_line_and_lines_for!(Gte, BinOpId::Gte);

#[derive(Debug, Clone, PartialEq)]
pub struct Lt(BinOp);
impl_try_from_val_ref_for!(Lt, BinOpId::Lt);
impl_line_and_lines_for!(Lt, BinOpId::Lt);

#[derive(Debug, Clone, PartialEq)]
pub struct Lte(BinOp);
impl_try_from_val_ref_for!(Lte, BinOpId::Lte);
impl_line_and_lines_for!(Lte, BinOpId::Lte);

#[derive(Debug, Clone, PartialEq)]
pub struct Mul(BinOp);
impl_try_from_val_ref_for!(Mul, BinOpId::Mul);
impl_line_and_lines_for!(Mul, BinOpId::Mul);

#[derive(Debug, Clone, PartialEq)]
pub struct Ne(BinOp);
impl_try_from_val_ref_for!(Ne, BinOpId::Ne);
impl_line_and_lines_for!(Ne, BinOpId::Ne);

#[derive(Debug, Clone, PartialEq)]
pub struct Sub(BinOp);
impl_try_from_val_ref_for!(Sub, BinOpId::Sub);
impl_line_and_lines_for!(Sub, BinOpId::Sub);

#[derive(Debug, Clone, PartialEq)]
pub struct Or(BinOp);
impl_try_from_val_ref_for!(Or, BinOpId::Or);
impl_line_and_lines_for!(Or, BinOpId::Or);

#[derive(Debug, Clone)]
pub struct BinOp {
    id: Uuid,
    pub lhs: Expr,
    pub bin_op_id: BinOpId,
    pub rhs: Expr,
}

impl PartialEq for BinOp {
    fn eq(&self, other: &Self) -> bool {
        self.lhs == other.lhs && self.bin_op_id == other.bin_op_id && self.rhs == other.rhs
    }
}

impl BinOp {
    pub fn new(lhs: Expr, bin_op_id: BinOpId, rhs: Expr) -> Self {
        Self {
            id: Uuid::new_v4(),
            lhs,
            bin_op_id,
            rhs,
        }
    }

    /// Returns an int representing how high the operator's precendence is,
    /// where 2 is the highest precedence for a binary operation.
    pub fn precedence(&self) -> u8 {
        self.bin_op_id.precedence()
    }

    pub fn associativity(&self) -> Option<Associativity> {
        self.bin_op_id.associativity()
    }
}

fn val_to_bin_op(
    val: &Val,
    bin_op_id: BinOpId,
) -> Result<BinOp, MapForTypeError> {
    let type_str = bin_op_id.type_str();
    let hash_map = map_for_type(val, type_str)?;
    let lhs =
        get_val(hash_map, "lhs").expect("bin op {:?} should have 'lhs' tag");
    let rhs =
        get_val(hash_map, "rhs").expect("bin op {:?} should have 'rhs' tag");
    let lhs_expr = lhs.try_into().unwrap_or_else(|_| {
        panic!("bin op {:?} 'lhs' could not be parsed as an Expr", lhs)
    });
    let rhs_expr = rhs.try_into().unwrap_or_else(|_| {
        panic!("bin op {:?} 'rhs' could not be parsed as an Expr", rhs)
    });
    let bin_op = BinOp::new(lhs_expr, bin_op_id, rhs_expr);
    Ok(bin_op)
}

/// Enumerates the types of binary operations present in Axon code.
#[derive(Debug, Clone, PartialEq)]
pub enum BinOpId {
    Add,
    And,
    Cmp,
    Div,
    Eq,
    Gt,
    Gte,
    Lt,
    Lte,
    Mul,
    Ne,
    Or,
    Sub,
}

impl BinOpId {
    /// Returns the value which SkySpark uses to differentiate the types of AST
    /// dicts.
    pub fn type_str(&self) -> &str {
        match self {
            Self::Add => "add",
            Self::And => "and",
            Self::Cmp => "cmp",
            Self::Div => "div",
            Self::Eq => "eq",
            Self::Gt => "gt",
            Self::Gte => "ge",
            Self::Lt => "lt",
            Self::Lte => "le",
            Self::Mul => "mul",
            Self::Ne => "ne",
            Self::Or => "or",
            Self::Sub => "sub",
        }
    }

    /// Returns the Axon symbol for this `BinOp`.
    pub fn to_symbol(&self) -> &str {
        match self {
            Self::Add => "+",
            Self::And => "and",
            Self::Cmp => "<=>",
            Self::Div => "/",
            Self::Eq => "==",
            Self::Gt => ">",
            Self::Gte => ">=",
            Self::Lt => "<",
            Self::Lte => "<=",
            Self::Mul => "*",
            Self::Ne => "!=",
            Self::Or => "or",
            Self::Sub => "-",
        }
    }

    /// Returns an int representing how high the operator's precendence is,
    /// where 20 is the highest precedence for a binary operation.
    pub fn precedence(&self) -> u8 {
        match self {
            Self::Add => 30,
            Self::And => 60,
            Self::Cmp => 50,
            Self::Div => 20,
            Self::Eq => 40,
            Self::Gt => 50,
            Self::Gte => 50,
            Self::Lt => 50,
            Self::Lte => 50,
            Self::Mul => 20,
            Self::Ne => 40,
            Self::Or => 70,
            Self::Sub => 30,
        }
    }

    pub fn associativity(&self) -> Option<Associativity> {
        match self {
            Self::Add => Some(Associativity::Right), // Based on the parsed AST.
            Self::And => Some(Associativity::Left),
            Self::Cmp => None, // 1 <=> 1 <=> 1 does not parse in SkySpark.
            Self::Div => Some(Associativity::Left),
            Self::Eq => None, // 5 == 5 == true does not parse in SkySpark.
            Self::Gt => None,
            Self::Gte => None,
            Self::Lt => None,
            Self::Lte => None,
            Self::Mul => Some(Associativity::Left),
            Self::Ne => None,
            Self::Or => Some(Associativity::Right), // Based on the parsed AST.
            Self::Sub => Some(Associativity::Left),
        }
    }
}

/// Represents lines of Axon code.
pub type Lines = Vec<Line>;

/// Represents a line of Axon code, including the indentation at the start
/// of the line.
#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    /// The indentation of the line (excluding the Axon code).
    pub indent: Indent,
    /// The remaining part of the line (excluding the indentation).
    pub line: String,
}

impl Line {
    pub fn new(indent: Indent, line: String) -> Self {
        Self { indent, line }
    }

    /// Return a new line with parentheses added around it.
    pub fn grouped(&self) -> Line {
        self.prefix_str("(").suffix_str(")")
    }

    /// Return the contents of this line, excluding the indent.
    pub fn inner_str(&self) -> &str {
        &self.line
    }

    /// Return the indent of this line.
    pub fn indent(&self) -> &Indent {
        &self.indent
    }

    /// Return a new line with the string prefixed to the start of the
    /// contents of this line.
    pub fn prefix_str(&self, prefix: &str) -> Self {
        Self::new(self.indent.clone(), format!("{}{}", prefix, self.line))
    }

    /// Return a new line with the string suffixed to the end of the line.
    pub fn suffix_str(&self, suffix: &str) -> Self {
        Self::new(self.indent.clone(), format!("{}{}", self.line, suffix))
    }

    /// The number of characters in this line, including indentation.
    pub fn len(&self) -> usize {
        let code = format!("{}", self);
        code.chars().count()
    }
}

impl std::fmt::Display for Line {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.indent, self.line)
    }
}

#[derive(Debug, Clone)]
pub struct Neg {
    id: Uuid,
    pub operand: Expr,
}

impl PartialEq for Neg {
    fn eq(&self, other: &Self) -> bool {
        self.operand == other.operand
    }
}

impl Neg {
    pub fn new(operand: Expr) -> Neg {
        Self {
            id: Uuid::new_v4(),
            operand
        }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        let s = self.operand.to_line(indent);
        s.prefix_str("-")
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        let mut lines = self.operand.to_lines(indent);
        let first_line =
            lines.first().expect("Neg should contain at least one line");
        let new_str = format!("-{}", first_line.inner_str());
        lines[0] = Line::new(first_line.indent().clone(), new_str);
        lines
    }
}

impl TryFrom<&Val> for Neg {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "neg").map_err(|_| ())?;
        let operand = get_val(hash_map, "operand")
            .expect("neg should have 'operand' tag");
        let operand_expr = operand
            .try_into()
            .expect("neg 'operand' could not be parsed as an Expr");
        Ok(Self::new(operand_expr))
    }
}

#[derive(Debug, Clone)]
pub struct TryCatch {
    id: Uuid,
    pub try_expr: Expr,
    pub exception_name: Option<String>,
    pub catch_expr: Expr,
}

impl PartialEq for TryCatch {
    fn eq(&self, other: &Self) -> bool {
        self.try_expr == other.try_expr && self.exception_name == other.exception_name && self.catch_expr == other.catch_expr
    }
}

impl TryCatch {
    pub fn new(
        try_expr: Expr,
        exception_name: Option<String>,
        catch_expr: Expr,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            try_expr,
            exception_name,
            catch_expr,
        }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        // try do a; b end catch <(ex)> do c; d end
        let try_expr = self.try_expr.clone().blockify();
        let line = try_expr.to_line(indent).prefix_str("try ");

        let line = if let Some(exc_name) = &self.exception_name {
            line.suffix_str(&format!(" catch ({}) ", exc_name))
        } else {
            line.suffix_str(" catch ")
        };

        let catch_expr = self.catch_expr.clone().blockify();
        let zero_indent = zero_indent();
        let catch_line = catch_expr.to_line(&zero_indent);
        let catch_str = catch_line.inner_str();

        line.suffix_str(catch_str)
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        let try_expr = self.try_expr.clone().blockify();
        let mut lines = try_expr.to_lines(indent);

        assert!(lines.len() >= 3);

        let first_line = lines
            .first()
            .expect("TryCatch should contain at least one line");
        let new_first_line = first_line.prefix_str("try ");
        lines[0] = new_first_line;

        let last_try_line = lines
            .last()
            .expect("TryCatch should contain at least one line")
            .clone();
        lines.pop();

        if let Some(exc_name) = &self.exception_name {
            let new_last_try_line =
                last_try_line.suffix_str(&format!(" catch ({}) ", exc_name));
            lines.push(new_last_try_line);
        } else {
            let new_last_try_line = last_try_line.suffix_str(" catch ");
            lines.push(new_last_try_line);
        }

        // Something like 'end catch (abc) '
        let end_catch_line = lines.last().unwrap().clone();
        let end_catch_str = end_catch_line.inner_str();
        lines.pop();

        let catch_expr = self.catch_expr.clone().blockify();
        let mut catch_lines = catch_expr.to_lines(indent);
        let first_catch_line = catch_lines
            .first()
            .expect("TryCatch catch expr should contain at least one line");
        let new_first_catch_line = first_catch_line.prefix_str(end_catch_str);
        catch_lines[0] = new_first_catch_line;

        lines.append(&mut catch_lines);
        lines
    }
}

impl TryFrom<&Val> for TryCatch {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "try").map_err(|_| ())?;
        let try_val = get_val(hash_map, "tryExpr")
            .expect("try should contain 'tryExpr' tag");
        let catch_val = get_val(hash_map, "catchExpr")
            .expect("try should contain 'catchExpr' tag");
        let exc_name_val = get_val(hash_map, "errVarName");

        let try_expr = try_val
            .try_into()
            .expect("try 'tryExpr' could not be parsed as an Expr");
        let catch_expr = catch_val
            .try_into()
            .expect("try 'catchExpr' could not be parsed as an Expr");
        let exc_name = match exc_name_val {
            Some(Val::Lit(ap::Lit::Str(exc_name))) => Some(exc_name.to_owned()),
            None => None,
            _ => panic!(
                "expected try 'errVarName' to be a string literal: {:?}",
                val
            ),
        };

        Ok(Self::new(try_expr, exc_name, catch_expr))
    }
}

/// Represents a chunk of code containing multiple nested
/// if / else if / ... / else expressions.
#[derive(Debug, Clone)]
pub struct FlatIf {
    id: Uuid,
    pub cond_exprs: Vec<ConditionalExpr>,
    pub else_expr: Option<Expr>,
}

impl PartialEq for FlatIf {
    fn eq(&self, other: &Self) -> bool {
        self.cond_exprs == other.cond_exprs && self.else_expr == other.else_expr
    }
}

impl FlatIf {
    pub fn new(
        cond_exprs: Vec<ConditionalExpr>,
        else_expr: Option<Expr>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            cond_exprs,
            else_expr,
        }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        let zero_indent = zero_indent();
        let mut strings = self
            .cond_exprs
            .iter()
            .map(|ce| ce.to_line(&zero_indent).inner_str().to_owned())
            .collect::<Vec<_>>();

        if let Some(else_expr) = self.else_expr.clone() {
            let else_expr = else_expr.blockify();
            let else_line = else_expr.to_line(indent);
            let else_string = else_line.inner_str().to_owned();
            strings.push(else_string);
        }

        let string = strings.join(" else ");
        Line::new(indent.clone(), string)
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        let cond_lines: Vec<Lines> = self
            .cond_exprs
            .iter()
            .map(|ce| ce.to_lines(indent))
            .collect();

        let mut final_lines: Lines = vec![];

        for mut lines in cond_lines {
            let last_final_line = final_lines.last();
            if let Some(last_final_line) = last_final_line {
                assert!(lines.len() >= 3);

                // Something like 'if (a) do' ...
                let first_line = lines.first().unwrap();

                let first_line_str = first_line.inner_str();
                let new_first_line = last_final_line
                    .suffix_str(&format!(" else {}", first_line_str));
                lines[0] = new_first_line;

                final_lines.pop();
                final_lines.append(&mut lines);
            } else {
                final_lines.append(&mut lines);
            }
        }

        if let Some(else_expr) = self.else_expr.clone() {
            let last_final_line = final_lines.last().expect(
                "FlatIf should contain at least one line before its else expr",
            );

            let else_expr = else_expr.blockify();
            let mut else_lines = else_expr.to_lines(indent);

            // Should be  'do' ...
            let first_line = else_lines
                .first()
                .expect("FlatIf else expr should contain at least one line");

            let first_line_str = first_line.inner_str();
            let new_first_line = last_final_line
                .suffix_str(&format!(" else {}", first_line_str));
            else_lines[0] = new_first_line;

            final_lines.pop();
            final_lines.append(&mut else_lines)
        }

        final_lines
    }
}

#[derive(Debug, Clone)]
pub struct ConditionalExpr {
    id: Uuid,
    /// The conditional expression, for example x == true
    pub cond: Expr,
    /// The expression that gets executed if the condition is true
    pub expr: Expr,
}

impl PartialEq for ConditionalExpr {
    fn eq(&self, other: &Self) -> bool {
        self.cond == other.cond && self.expr == other.expr
    }
}

impl ConditionalExpr {
    pub fn new(cond: Expr, expr: Expr) -> Self {
        Self {             id: Uuid::new_v4(),cond, expr }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        // if (something) do a; b; c end
        let line = self.cond.to_line(indent).grouped();
        let line = line.prefix_str("if ").suffix_str(" ");

        let expr = self.expr.clone().blockify();
        let zero_indent = zero_indent();
        let expr_line = expr.to_line(&zero_indent);
        let expr_str = expr_line.inner_str();

        line.suffix_str(expr_str)
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        let zero_indent = zero_indent();
        let cond_line = self
            .cond
            .to_line(&zero_indent)
            .grouped()
            .prefix_str("if ")
            .suffix_str(" ");
        let cond_str = cond_line.inner_str();

        let expr = self.expr.clone().blockify();
        let mut lines = expr.to_lines(indent);
        let first_line = lines
            .first()
            .expect("ConditionalExpr expr should contain at least one line");
        let new_first_line = first_line.prefix_str(cond_str);
        lines[0] = new_first_line;
        lines
    }
}

#[derive(Debug, Clone)]
pub struct If {
    id: Uuid,
    pub cond: Expr,
    pub if_expr: Expr,
    pub else_expr: Option<Expr>,
}

impl PartialEq for If {
    fn eq(&self, other: &Self) -> bool {
        self.cond == other.cond && self.if_expr == other.if_expr && self.else_expr == other.else_expr
    }
}

impl If {
    pub fn new(cond: Expr, if_expr: Expr, else_expr: Option<Expr>) -> Self {
        Self {
            id: Uuid::new_v4(),
            cond,
            if_expr,
            else_expr,
        }
    }

    /// Return a `FlatIf` containing the same contents as this `If`.
    pub fn flatten(&self) -> FlatIf {
        let first_cond_expr =
            ConditionalExpr::new(self.cond.clone(), self.if_expr.clone());

        match &self.else_expr {
            Some(Expr::If(nested_if)) => {
                let nested_flat_if = nested_if.flatten();
                let mut cond_exprs = nested_flat_if.cond_exprs;
                let else_expr = nested_flat_if.else_expr;
                cond_exprs.insert(0, first_cond_expr);
                FlatIf::new(cond_exprs, else_expr)
            }
            Some(non_if_expr) => {
                FlatIf::new(vec![first_cond_expr], Some(non_if_expr.clone()))
            }
            None => FlatIf::new(vec![first_cond_expr], None),
        }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        let flat_if = self.flatten();
        flat_if.to_line(indent)
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        let flat_if = self.flatten();
        flat_if.to_lines(indent)
    }
}

impl TryFrom<&Val> for If {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "if").map_err(|_| ())?;
        let cond_val =
            get_val(hash_map, "cond").expect("if should contain 'cond' tag");
        let if_val = get_val(hash_map, "ifExpr")
            .expect("if should contain 'ifExpr' tag");
        let else_val = get_val(hash_map, "elseExpr");

        let cond_expr = cond_val
            .try_into()
            .expect("if 'cond' could not be parsed as an Expr");
        let if_expr = if_val.try_into().unwrap_or_else(|_| {
            panic!(
                "if 'ifExpr' could not be parsed as an Expr: \n{:#?}",
                if_val
            )
        });
        let else_expr = else_val.map(|val| {
            val.try_into()
                .expect("if 'elseExpr' could not be parsed as an Expr")
        });

        Ok(Self::new(cond_expr, if_expr, else_expr))
    }
}

#[derive(Debug, Clone)]
pub struct TrapCall {
    id: Uuid,
    pub target: Expr,
    pub key: String,
}

impl PartialEq for TrapCall {
    fn eq(&self, other: &Self) -> bool {
        self.target == other.target && self.key == other.key
    }
}

impl TrapCall {
    pub fn new(target: Expr, key: String) -> Self {
        Self {             id: Uuid::new_v4(),target, key }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        let line = self.target.to_line(indent);
        line.suffix_str(&format!("->{}", self.key))
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        let mut lines = self.target.to_lines(indent);
        let last_line = lines
            .last()
            .expect("TrapCall target should have at least one line");
        let inner_str = last_line.inner_str();
        let new_inner_str = format!("{}->{}", inner_str, self.key);
        let new_last_line =
            Line::new(last_line.indent().clone(), new_inner_str);
        let _ = lines.pop();
        lines.push(new_last_line);
        lines
    }
}

impl TryFrom<&Val> for TrapCall {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "trapCall").map_err(|_| ())?;
        let args = get_vals(hash_map, "args")
            .expect("trapCall should have 'args' tag");

        assert!(
            args.len() == 2,
            "trapCall 'args' list should have exactly two elements: {:?}",
            args
        );
        let target = &args[0];
        let key = &args[1];
        let target = target
            .try_into()
            .expect("trapCall 'target' could not be parsed as an Expr");

        let key = match key {
            Val::Dict(key_hash_map) => {
                let key_str = get_literal_str(key_hash_map, "val")
                    .expect("trapCall key hash map should contain 'val' tag");
                key_str.to_owned()
            }
            _ => panic!("expected trapCall key Val to be a Dict"),
        };

        Ok(Self::new(target, key))
    }
}

#[derive(Debug, Clone)]
pub struct DotCall {
    id: Uuid,
    pub func_name: FuncName,
    pub target: Box<Expr>,
    pub args: Vec<Expr>,
}

impl PartialEq for DotCall {
    fn eq(&self, other: &Self) -> bool {
        self.func_name == other.func_name && self.target == other.target && self.args == other.args
    }
}

impl DotCall {
    pub fn new(
        func_name: FuncName,
        target: Box<Expr>,
        args: Vec<Expr>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            func_name,
            target,
            args,
        }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        let line = self.target.to_line(indent);
        let zero_indent = zero_indent();
        let trailing_line = arg_exprs_to_line(&self.args, &zero_indent);
        let trailing_args_str = trailing_line.inner_str();
        match &self.func_name {
            FuncName::TagName(tag_name) => {
                let tag_name: &str = tag_name.as_ref();
                if tag_name == "get" {
                    line.suffix_str(&format!("[{}]", trailing_args_str))
                } else {
                    line.suffix_str(&format!(
                        ".{}({})",
                        self.func_name, trailing_args_str
                    ))
                }
            }
            _ => line.suffix_str(&format!(
                ".{}({})",
                self.func_name, trailing_args_str
            )),
        }
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        let mut lines = self.target.to_lines(indent);
        let last_target_line = lines
            .last()
            .expect("DotCall target should contain at least one line");

        let args = &self.args;

        let is_get = match &self.func_name {
            FuncName::TagName(tag_name) => {
                let tag_name: &str = tag_name.as_ref();
                tag_name == "get"
            }
            _ => false,
        };

        if args.is_empty() {
            let new_last_target_line =
                last_target_line.suffix_str(&format!(".{}()", self.func_name));
            lines.pop();
            lines.push(new_last_target_line);
        } else if is_get && args.len() == 1 {
            let only_arg = args.first().unwrap();
            let zero_indent = zero_indent();
            let only_arg_line = only_arg.to_line(&zero_indent);
            let only_arg_str = only_arg_line.inner_str();

            let new_last_target_line =
                last_target_line.suffix_str(&format!("[{}]", only_arg_str));
            lines.pop();
            lines.push(new_last_target_line);
        } else {
            let mut arg_lines = arg_exprs_to_lines(args, indent);
            let first_arg_line = arg_lines
                .first()
                .expect("DotCall args should contain at least one line");
            let first_arg_line_str = first_arg_line.inner_str();

            let new_last_target_line = last_target_line.suffix_str(&format!(
                ".{}{}",
                self.func_name, first_arg_line_str
            ));
            lines.pop();
            lines.push(new_last_target_line);

            arg_lines.remove(0);
            lines.append(&mut arg_lines);
        }

        lines
    }
}

impl TryFrom<&Val> for DotCall {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "dotCall").map_err(|_| ())?;
        let target = get_val(hash_map, "target")
            .expect("dotCall should have 'target' tag");
        match target {
            Val::Dict(target_hash_map) => {
                let func_name = get_literal_str(target_hash_map, "name")
                    .expect("dotCall 'target' should have 'name' string tag");
                let func_name = func_name.to_owned();
                let args = get_vals(hash_map, "args")
                    .expect("dotCall should have 'args' tag");

                let mut exprs = vec![];

                for arg in args {
                    let expr = arg.try_into().unwrap_or_else(|_| {
                        panic!(
                            "dotCall arg could not be parsed as an Expr: {:?}",
                            arg
                        )
                    });
                    exprs.push(expr);
                }

                let target = exprs.remove(0);

                if let Some(func_name) = TagName::new(func_name.clone()) {
                    Ok(Self::new(
                        FuncName::TagName(func_name),
                        Box::new(target),
                        exprs,
                    ))
                } else {
                    let qname = Qname::new(func_name);
                    Ok(Self::new(
                        FuncName::Qname(qname),
                        Box::new(target),
                        exprs,
                    ))
                }
            }
            _ => panic!("expected dotCall 'target' to be a Dict"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FuncName {
    TagName(TagName),
    Qname(Qname),
}

impl std::fmt::Display for FuncName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TagName(tag_name) => write!(f, "{}", tag_name),
            Self::Qname(qname) => write!(f, "{}", qname),
        }
    }
}

impl FuncName {
    pub fn to_line(&self, indent: &Indent) -> Line {
        let s = format!("{}", self);
        Line::new(indent.clone(), s)
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        vec![self.to_line(indent)]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PartialCallArgument {
    Expr(Expr),
    Placeholder,
}

impl PartialCallArgument {
    pub fn to_line(&self, indent: &Indent) -> Line {
        match self {
            Self::Expr(expr) => expr.to_line(indent),
            Self::Placeholder => Line::new(indent.clone(), "_".to_owned()),
        }
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        match self {
            Self::Expr(expr) => expr.to_lines(indent),
            Self::Placeholder => vec![self.to_line(indent)],
        }
    }

    fn is_func(&self) -> bool {
        match self {
            Self::Expr(expr) => expr.is_func(),
            Self::Placeholder => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PartialCall {
    id: Uuid,
    pub func_name: FuncName,
    pub args: Vec<PartialCallArgument>,
}

impl PartialEq for PartialCall {
    fn eq(&self, other: &Self) -> bool {
        self.func_name == other.func_name && self.args == other.args
    }
}

impl PartialCall {
    pub fn new(func_name: FuncName, args: Vec<PartialCallArgument>) -> Self {
        Self { id: Uuid::new_v4(),func_name, args }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        let args_line = partial_call_arg_exprs_to_line(&self.args, indent);
        args_line
            .prefix_str(&format!("{}(", self.func_name))
            .suffix_str(")")
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        let mut lines = partial_call_arg_exprs_to_lines(&self.args, indent);
        let first_arg_line = lines
            .first()
            .expect("Call args should contain at least one line");
        let func_name = format!("{}", &self.func_name);
        let new_first_arg_line = first_arg_line.prefix_str(&func_name);
        lines[0] = new_first_arg_line;
        lines
    }
}

impl TryFrom<&Val> for PartialCall {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "partialCall").map_err(|_| ())?;
        let target = get_val(hash_map, "target")
            .expect("partialCall should have 'target' tag");
        match target {
            Val::Dict(target_hash_map) => {
                let func_name = get_literal_str(target_hash_map, "name")
                    .expect(
                        "partialCall 'target' should have 'name' string tag",
                    );
                let func_name = func_name.to_owned();
                let args = get_vals(hash_map, "args")
                    .expect("partialCall should have 'args' tag");

                let mut exprs = vec![];

                for arg in args {
                    if matches!(arg, Val::Lit(ap::Lit::Null)) {
                        exprs.push(PartialCallArgument::Placeholder);
                    } else {
                        let expr: Expr = arg.try_into().unwrap_or_else(|_| {
                            panic!(
                                "partialCall arg could not be parsed as an Expr: {:?}",
                                arg
                            )
                        });
                        exprs.push(PartialCallArgument::Expr(expr));
                    }
                }

                if let Some(func_name) = TagName::new(func_name.clone()) {
                    Ok(Self::new(FuncName::TagName(func_name), exprs))
                } else {
                    // We assume it's a qname:
                    let qname = Qname::new(func_name);
                    Ok(Self::new(FuncName::Qname(qname), exprs))
                }
            }
            _ => panic!("expected partialCall 'target' to be a Dict"),
        }
    }
}

/// Converts expressions, which represent arguments to a function, into a `Line`.
fn partial_call_arg_exprs_to_line(
    args: &[PartialCallArgument],
    indent: &Indent,
) -> Line {
    // Should return something like
    // arg1, arg2, {arg3}
    // with no enclosing parentheses.
    let line_strs = args
        .iter()
        .map(|arg| arg.to_line(indent).inner_str().to_owned())
        .collect::<Vec<_>>();
    let line_str = line_strs.join(", ");
    Line::new(indent.clone(), line_str)
}

/// Converts expressions, which represent arguments to a function, into `Lines`.
fn partial_call_arg_exprs_to_lines(
    args: &[PartialCallArgument],
    indent: &Indent,
) -> Lines {
    // Should return something like
    // (arg1, arg2) (lambdaArg1, lambdaArg2) => do
    //     ...
    // end
    if args.is_empty() {
        vec![Line::new(indent.clone(), "()".to_owned())]
    } else {
        let last_arg = args.last().unwrap();
        if last_arg.is_func() {
            let func = last_arg.clone();
            // Format it as a trailing lambda.
            let mut lines = func.to_lines(indent);
            let first_func_line = lines
                .first()
                .expect("func should contain at least one line");

            // everything except the trailing func expr:
            let preceding_args = if args.len() == 1 {
                &[] // The func was the only argument, there are no preceding args.
            } else {
                &args[..args.len()]
            };
            let zero_indent = zero_indent();
            let preceding_line =
                partial_call_arg_exprs_to_line(preceding_args, &zero_indent)
                    .grouped();
            let preceding_str = preceding_line.inner_str();
            let new_first_func_line =
                first_func_line.prefix_str(&format!("{} ", preceding_str));
            lines[0] = new_first_func_line;
            lines
        } else {
            let line = partial_call_arg_exprs_to_line(args, indent).grouped();
            vec![line]
        }
    }
}

#[derive(Debug, Clone)]
pub struct Call {
    id: Uuid,
    pub target: CallTarget,
    pub args: Vec<Expr>,
}

impl PartialEq for Call {
    fn eq(&self, other: &Self) -> bool {
        self.target == other.target && self.args == other.args
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CallTarget {
    Expr(Box<Expr>),
    FuncName(FuncName),
}

impl CallTarget {
    pub fn to_line(&self, indent: &Indent) -> Line {
        match self {
            Self::Expr(expr) => expr.to_line(indent),
            Self::FuncName(func_name) => func_name.to_line(indent),
        }
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        match self {
            Self::Expr(expr) => expr.to_lines(indent),
            Self::FuncName(func_name) => func_name.to_lines(indent),
        }
    }
}

/// Converts expressions, which represent arguments to a function, into a `Line`.
fn arg_exprs_to_line(args: &[Expr], indent: &Indent) -> Line {
    // Should return something like
    // arg1, arg2, {arg3}
    // with no enclosing parentheses.
    let line_strs = args
        .iter()
        .map(|arg| arg.to_line(indent).inner_str().to_owned())
        .collect::<Vec<_>>();
    let line_str = line_strs.join(", ");
    Line::new(indent.clone(), line_str)
}

/// Converts expressions, which represent arguments to a function, into `Lines`.
fn arg_exprs_to_lines(args: &[Expr], indent: &Indent) -> Lines {
    // Should return something like
    // (arg1, arg2) (lambdaArg1, lambdaArg2) => do
    //     ...
    // end
    if args.is_empty() {
        vec![Line::new(indent.clone(), "()".to_owned())]
    } else {
        let last_arg = args.last().unwrap();
        if last_arg.is_func() {
            let func = last_arg.clone();
            // Format it as a trailing lambda.
            let mut lines = func.to_lines(indent);
            let first_func_line = lines
                .first()
                .expect("func should contain at least one line");

            // everything except the trailing func expr:
            let preceding_args = if args.len() == 1 {
                &[] // The func was the only argument, there are no preceding args.
            } else {
                &args[..args.len()]
            };
            let zero_indent = zero_indent();
            let preceding_line =
                arg_exprs_to_line(preceding_args, &zero_indent).grouped();
            let preceding_str = preceding_line.inner_str();
            let new_first_func_line =
                first_func_line.prefix_str(&format!("{} ", preceding_str));
            lines[0] = new_first_func_line;
            lines
        } else {
            let line = arg_exprs_to_line(args, indent).grouped();
            vec![line]
        }
    }
}

impl Call {
    pub fn new(target: CallTarget, args: Vec<Expr>) -> Self {
        Self { id: Uuid::new_v4(),target, args }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        let args_line = arg_exprs_to_line(&self.args, indent);
        let zero_indent = zero_indent();
        args_line
            .prefix_str(&format!(
                "{}(",
                self.target.to_line(&zero_indent).inner_str()
            ))
            .suffix_str(")")
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        let mut lines = arg_exprs_to_lines(&self.args, indent);
        let first_arg_line = lines
            .first()
            .expect("Call args should contain at least one line");
        let zero_indent = zero_indent();
        let first_line_prefix =
            format!("{}", &self.target.to_line(&zero_indent));
        let new_first_arg_line = first_arg_line.prefix_str(&first_line_prefix);
        lines[0] = new_first_arg_line;
        lines
    }
}

impl TryFrom<&Val> for Call {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "call").map_err(|_| ())?;
        let target =
            get_val(hash_map, "target").expect("call should have 'target' tag");
        let target: CallTarget = match target {
            Val::Dict(target_hash_map) => {
                if let Some(func_name) =
                    get_literal_str(target_hash_map, "name")
                {
                    let func_name = func_name.to_owned();

                    if let Some(func_name) = TagName::new(func_name.clone()) {
                        let func_name = FuncName::TagName(func_name);
                        CallTarget::FuncName(func_name)
                    } else {
                        // We assume it's a qname:
                        let qname = Qname::new(func_name);
                        let func_name = FuncName::Qname(qname);
                        CallTarget::FuncName(func_name)
                    }
                } else {
                    let expr = target.try_into().unwrap_or_else(|_| panic!("if call target is not a func name, it should be an expression: {:#?}", target));
                    CallTarget::Expr(Box::new(expr))
                }
            }
            _ => panic!("expected call 'target' to be a Dict"),
        };

        let args =
            get_vals(hash_map, "args").expect("call should have 'args' tag");

        let mut exprs = vec![];

        for arg in args {
            let expr = arg.try_into().unwrap_or_else(|_| {
                panic!("call arg could not be parsed as an Expr: {:?}", arg)
            });
            exprs.push(expr);
        }

        Ok(Self::new(target, exprs))
    }
}

#[derive(Debug, Clone)]
pub struct Not {
    id: Uuid,
    pub operand: Expr,
}

impl PartialEq for Not {
    fn eq(&self, other: &Self) -> bool {
        self.operand == other.operand
    }
}

impl Not {
    pub fn new(operand: Expr) -> Self {
        Self { id: Uuid::new_v4(),operand }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        let line = self.operand.to_line(indent);
        line.prefix_str("not ")
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        let mut lines = self.operand.to_lines(indent);
        let first_line = lines
            .first()
            .expect("Not operand should contain at least one line (first)");
        let new_first_line = first_line.prefix_str("not (");
        let last_line = lines
            .last()
            .expect("Not operand should contain at least one line (last)");
        let new_last_line = last_line.suffix_str(")");
        lines[0] = new_first_line;
        lines.pop();
        lines.push(new_last_line);
        lines
    }
}

impl TryFrom<&Val> for Not {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "not").map_err(|_| ())?;
        let operand = get_val(hash_map, "operand")
            .expect("not should have 'operand' tag");
        let operand_expr = operand
            .try_into()
            .expect("not 'operand' could not be parsed as an Expr");
        Ok(Self::new(operand_expr))
    }
}

#[derive(Debug, Clone)]
pub struct Range {
    id: Uuid,
    pub start: Expr,
    pub end: Expr,
}

impl PartialEq for Range {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.end == other.end
    }
}

impl Range {
    pub fn new(start: Expr, end: Expr) -> Self {
        Self {id: Uuid::new_v4(), start, end }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        let line = self.start.to_line(indent);
        let zero_indent = zero_indent();
        let right_line = self.end.to_line(&zero_indent);
        let right_str = right_line.inner_str();
        line.suffix_str(&format!("..{}", right_str)).grouped()
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        vec![self.to_line(indent)]
    }
}

impl TryFrom<&Val> for Range {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "range").map_err(|_| ())?;
        let start =
            get_val(hash_map, "start").expect("range should have 'start' tag");
        let end =
            get_val(hash_map, "end").expect("range should have 'end' tag");
        let start_expr = start
            .try_into()
            .expect("range 'start' could not be parsed as an Expr");
        let end_expr = end
            .try_into()
            .expect("range 'end' could not be parsed as an Expr");
        Ok(Self::new(start_expr, end_expr))
    }
}

#[derive(Debug, Clone)]
pub struct Func {
    id: Uuid,
    pub params: Vec<Param>,
    pub body: Expr,
}

impl PartialEq for Func {
    fn eq(&self, other: &Self) -> bool {
        self.params == other.params && self.body == other.body
    }
}

impl Func {
    pub fn new(params: Vec<Param>, body: Expr) -> Self {
        Self { id: Uuid::new_v4(),params, body }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        let func = self.clone();
        let func = func.blockify();
        let line = params_to_line(&func.params, indent);
        let zero_indent = zero_indent();
        let body_line = func.body.to_line(&zero_indent);
        let body_str = body_line.inner_str();
        line.prefix_str("(")
            .suffix_str(&format!(") => {}", body_str))
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        let func = self.clone();
        let func = func.blockify();
        let zero_indent = zero_indent();
        let params_line = params_to_line(&func.params, &zero_indent);
        let params_str = params_line.inner_str();

        let mut body_lines = func.body.to_lines(indent);
        let first_body_line = body_lines
            .first()
            .expect("Func body should have at least one line");
        let new_first_body_line =
            first_body_line.prefix_str(&format!("({}) => ", params_str));
        body_lines[0] = new_first_body_line;
        body_lines
    }

    pub fn blockify(mut self) -> Self {
        self.body = self.body.blockify();
        self
    }
}

impl TryFrom<&Val> for Func {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "func").map_err(|_| ())?;
        let param_vals = get_vals(hash_map, "params")
            .expect("func should contain 'params' tag");

        let mut params = vec![];
        for param_val in param_vals {
            let param = param_val
                .try_into()
                .expect("func param val could not be converted to a Param");
            params.push(param);
        }

        let body =
            get_val(hash_map, "body").expect("func should have a 'body' tag");
        let body_expr = body
            .try_into()
            .expect("func body val could not be converted to a Expr");

        Ok(Self::new(params, body_expr))
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    id: Uuid,
    pub exprs: Vec<Expr>,
}
impl PartialEq for Block {
fn eq(&self, other: &Self) -> bool {
    self.exprs == other.exprs
}
}

fn zero_indent() -> Indent {
    Indent::new("".to_owned(), 0)
}

fn exprs_to_line(exprs: &[Expr], indent: &Indent) -> Line {
    separated_exprs_line(exprs, indent, "; ")
}

fn comma_separated_exprs_line(exprs: &[Expr], indent: &Indent) -> Line {
    separated_exprs_line(exprs, indent, ", ")
}

fn separated_exprs_line(
    exprs: &[Expr],
    indent: &Indent,
    separator: &str,
) -> Line {
    let zero_indent = zero_indent();
    let expr_lines = exprs
        .iter()
        .map(|expr| expr.to_line(&zero_indent).inner_str().to_owned())
        .collect::<Vec<_>>();
    let line_str = expr_lines.join(separator);
    Line::new(indent.clone(), line_str)
}

impl Block {
    pub fn new(exprs: Vec<Expr>) -> Self {
        Self { id: Uuid::new_v4(),exprs }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        let line = exprs_to_line(&self.exprs, indent);
        line.prefix_str("do ").suffix_str(" end")
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        let open_brace = Line::new(indent.clone(), "do".to_owned());
        let close_brace = Line::new(indent.clone(), "end".to_owned());
        let mut lines = vec![open_brace];

        let next_indent = indent.increase();

        for expr in &self.exprs {
            let mut expr_lines = expr.to_lines(&next_indent);
            lines.append(&mut expr_lines);
        }

        lines.push(close_brace);
        lines
    }
}

impl TryFrom<&Val> for Block {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "block").map_err(|_| ())?;
        let vals = get_vals(hash_map, "exprs")
            .expect("block should contain 'exprs' tag");

        let mut exprs = vec![];
        for val in vals {
            let expr = val.try_into().unwrap_or_else(|_| panic!("val in block 'exprs' tag could not be parsed as an Expr: {:?}", val));
            exprs.push(expr);
        }

        Ok(Self::new(exprs))
    }
}

#[derive(Debug, Clone)]
pub struct Dict {
    id: Uuid,
    pub map: HashMap<TagName, DictVal>,
}

impl PartialEq for Dict {
    fn eq(&self, other: &Self) -> bool {
        self.map == other.map
    }
}

impl Dict {
    pub fn new(map: HashMap<TagName, DictVal>) -> Self {
        Self {id: Uuid::new_v4(), map }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        let zero_indent = zero_indent();
        let mut entries = vec![];
        for (tag_name, dict_val) in self.map.iter() {
            let line = dict_val.to_line(&zero_indent);
            let entry = format!("{}: {}", tag_name, line.inner_str());
            entries.push(entry);
        }
        entries.sort();
        let entries_str = entries.join(", ");
        Line::new(indent.clone(), format!("{{{}}}", entries_str))
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        if self.map.is_empty() {
            vec![Line::new(indent.clone(), "{}".to_owned())]
        } else {
            let open_brace = Line::new(indent.clone(), "{".to_owned());
            let close_brace = Line::new(indent.clone(), "}".to_owned());
            let mut lines = vec![open_brace];

            let next_indent = indent.increase();

            let sorted_entries = sorted_dict_entries(self.map.clone());

            for (tag_name, expr) in sorted_entries {
                let mut dict_val_lines = expr.to_lines(&next_indent);
                let first_dict_val_line = dict_val_lines
                    .first()
                    .expect("dict val should contain at least one line");
                let inner_str = first_dict_val_line.inner_str();
                let new_str = format!("{}: {}", tag_name, inner_str);
                let new_first_dict_val_line =
                    Line::new(first_dict_val_line.indent().clone(), new_str);
                dict_val_lines[0] = new_first_dict_val_line;

                let mut comma_dict_val_lines = dict_val_lines
                    .into_iter()
                    .map(|ln| ln.suffix_str(","))
                    .collect();
                lines.append(&mut comma_dict_val_lines);
            }

            lines.push(close_brace);
            lines
        }
    }
}

fn sorted_dict_entries(
    dict: HashMap<TagName, DictVal>,
) -> Vec<(TagName, DictVal)> {
    let mut entries: Vec<(TagName, DictVal)> = dict.into_iter().collect();
    entries.sort_by(|(t1, _), (t2, _)| {
        t1.clone().into_string().cmp(&t2.clone().into_string())
    });
    entries
}

impl TryFrom<&Val> for Dict {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "dict").map_err(|_| ())?;

        let names = get_vals(hash_map, "names")
            .expect("dict should contain 'names' tag");
        let mut tag_names = vec![];

        for name in names {
            match name {
                Val::Lit(ap::Lit::Str(s)) => {
                    let tag_name =
                        TagName::new(s.to_owned()).unwrap_or_else(|| {
                            panic!(
                                "tag name '{}' in dict was not a valid TagName",
                                s
                            )
                        });
                    tag_names.push(tag_name);
                }
                _ => panic!("expected all dict names to be string literals"),
            }
        }

        let vals =
            get_vals(hash_map, "vals").expect("dict should contain 'vals' tag");
        assert!(vals.len() == names.len());
        let mut dict_vals = vec![];

        for val in vals {
            if val_is_marker(val) {
                dict_vals.push(DictVal::Marker);
            } else if val_is_remove_marker(val) {
                dict_vals.push(DictVal::RemoveMarker);
            } else {
                let expr = val.try_into().unwrap_or_else(|_| {
                    panic!(
                        "dict val could not be converted to an Expr: {:?}",
                        val
                    )
                });
                dict_vals.push(DictVal::Expr(expr));
            }
        }

        assert!(tag_names.len() == dict_vals.len());

        let map = tag_names.into_iter().zip(dict_vals.into_iter()).collect();
        Ok(Self::new(map))
    }
}

fn val_is_marker(val: &Val) -> bool {
    // val will look like
    // Dict({TagName("val"): Lit(DictMarker), TagName("type"): Lit(Str("literal"))})
    // if this function returns true.
    match val {
        Val::Dict(hash_map) => {
            let is_literal_type = type_str(hash_map) == "literal";
            if !is_literal_type {
                return false;
            }

            let key = tn("val");
            let identifier = hash_map.get(&key);

            match identifier {
                Some(inner_val) => {
                    matches!(inner_val.as_ref(), Val::Lit(ap::Lit::DictMarker))
                }
                None => false,
            }
        }
        _ => false,
    }
}

fn val_is_remove_marker(val: &Val) -> bool {
    // val will look like
    // Dict({TagName("val"): Lit(DictRemoveMarker), TagName("type"): Lit(Str("literal"))})
    // if this function returns true.
    match val {
        Val::Dict(hash_map) => {
            let is_literal_type = type_str(hash_map) == "literal";
            if !is_literal_type {
                return false;
            }

            let key = tn("val");
            let identifier = hash_map.get(&key);
            match identifier {
                Some(inner_val) => {
                    matches!(
                        inner_val.as_ref(),
                        Val::Lit(ap::Lit::DictRemoveMarker)
                    )
                }
                None => false,
            }
        }
        _ => false,
    }
}

/// Represents the value of some tag in an Axon dict.
#[derive(Clone, Debug, PartialEq)]
pub enum DictVal {
    /// An Axon expression.
    Expr(Expr),
    /// The Axon marker value.
    Marker,
    /// The Axon removeMarker value.
    RemoveMarker,
}

impl DictVal {
    pub fn to_line(&self, indent: &Indent) -> Line {
        // TODO should this be renamed to to_axon_code?
        match self {
            Self::Expr(expr) => expr.to_line(indent),
            Self::Marker => Line::new(indent.clone(), "marker()".to_owned()),
            Self::RemoveMarker => {
                Line::new(indent.clone(), "removeMarker()".to_owned())
            }
        }
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        match self {
            Self::Expr(expr) => expr.to_lines(indent),
            Self::Marker => {
                vec![Line::new(indent.clone(), "marker()".to_owned())]
            }
            Self::RemoveMarker => {
                vec![Line::new(indent.clone(), "removeMarker()".to_owned())]
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Return {
    id: Uuid,
    pub expr: Expr,
}

impl PartialEq for Return {
    fn eq(&self, other: &Self) -> bool {
        self.expr == other.expr
    }
}

impl Return {
    pub fn new(expr: Expr) -> Self {
        Self { id: Uuid::new_v4(),expr }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        let line = self.expr.to_line(indent);
        line.prefix_str("return ")
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        let mut expr_lines = self.expr.to_lines(&indent);
        let first_expr_line = expr_lines
            .first()
            .expect("Return expr should contain at least one line");
        let new_inner_str = format!("return {}", first_expr_line.inner_str());
        let new_first_line =
            Line::new(first_expr_line.indent().clone(), new_inner_str);
        expr_lines[0] = new_first_line;
        expr_lines
    }
}

impl TryFrom<&Val> for Return {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "return").map_err(|_| ())?;
        let val = get_val(hash_map, "expr")
            .expect("return should contain 'expr' tag");
        let expr: Expr = val
            .try_into()
            .expect("return 'expr' could not be converted into an Expr");
        Ok(Self::new(expr))
    }
}

#[derive(Debug, Clone)]
pub struct Throw {
    id: Uuid,
    pub expr: Expr,
}

impl PartialEq for Throw {
    fn eq(&self, other: &Self) -> bool {
        self.expr == other.expr
    }
}

impl Throw {
    pub fn new(expr: Expr) -> Self {
        Self { id: Uuid::new_v4(),expr }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        let line = self.expr.to_line(indent);
        line.prefix_str("throw ")
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        let mut expr_lines = self.expr.to_lines(&indent);
        let first_expr_line = expr_lines
            .first()
            .expect("Throw expr should contain at least one line");
        let new_inner_str = format!("throw {}", first_expr_line.inner_str());
        let new_first_line =
            Line::new(first_expr_line.indent().clone(), new_inner_str);
        expr_lines[0] = new_first_line;
        expr_lines
    }
}

impl TryFrom<&Val> for Throw {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "throw").map_err(|_| ())?;
        let val =
            get_val(hash_map, "expr").expect("throw should contain 'expr' tag");
        let expr: Expr = val
            .try_into()
            .expect("throw 'expr' could not be converted into an Expr");
        Ok(Self::new(expr))
    }
}

#[derive(Debug, Clone)]
pub struct List {
    id: Uuid,
    pub vals: Vec<Expr>,
}

impl PartialEq for List {
    fn eq(&self, other: &Self) -> bool {
        self.vals == other.vals
    }
}

impl List {
    pub fn new(vals: Vec<Expr>) -> Self {
        Self {id: Uuid::new_v4(), vals }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        let line = comma_separated_exprs_line(&self.vals, indent);
        line.prefix_str("[").suffix_str("]")
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        if self.vals.is_empty() {
            vec![Line::new(indent.clone(), "[]".to_owned())]
        } else {
            let open_brace = Line::new(indent.clone(), "[".to_owned());
            let close_brace = Line::new(indent.clone(), "]".to_owned());
            let mut lines = vec![open_brace];

            let next_indent = indent.increase();

            for expr in &self.vals {
                let mut expr_lines = expr.to_lines(&next_indent);
                let last_expr_line = expr_lines.last().expect(
                    "List expressions should contain at least one line",
                );
                let new_last_expr_line = last_expr_line.suffix_str(",");
                expr_lines.pop();
                expr_lines.push(new_last_expr_line);

                lines.append(&mut expr_lines);
            }

            lines.push(close_brace);
            lines
        }
    }
}

impl TryFrom<&Val> for List {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "list").map_err(|_| ())?;
        let elements = get_vals(hash_map, "vals")
            .expect("'vals' tag in list should contain a vec of elements");
        let exprs: Result<Vec<Expr>, ()> =
            elements.iter().map(|elem| elem.try_into()).collect();
        let exprs = exprs.expect(
            "at least one list element could not be converted into an Expr",
        );
        Ok(Self::new(exprs))
    }
}

#[derive(Debug, Clone)]
pub struct Id {
    id: Uuid,
    name: TagName,
}

impl PartialEq for Id {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Id {
    pub fn new(name: TagName) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
        }
    }

    pub fn name(&self) -> &TagName {
        &self.name
    }
}

/// Enumerates the types of Axon expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Add(Box<Add>),
    And(Box<And>),
    Cmp(Box<Cmp>),
    Div(Box<Div>),
    Eq(Box<Eq>),
    Gt(Box<Gt>),
    Gte(Box<Gte>),
    Lt(Box<Lt>),
    Lte(Box<Lte>),
    Mul(Box<Mul>),
    Ne(Box<Ne>),
    Or(Box<Or>),
    Sub(Box<Sub>),
    Assign(Assign),
    Block(Block),
    Call(Call),
    Def(Def),
    Dict(Dict),
    DotCall(DotCall),
    Func(Box<Func>),
    Id(Id),
    If(Box<If>),
    List(List),
    Lit(Lit),
    Neg(Box<Neg>),
    Not(Box<Not>),
    PartialCall(PartialCall),
    Range(Box<Range>),
    Return(Box<Return>),
    Throw(Box<Throw>),
    TrapCall(Box<TrapCall>),
    TryCatch(Box<TryCatch>),
}

/// Represents the indentation at the start of a line of Axon code.
#[derive(Debug, Clone, PartialEq)]
pub struct Indent {
    /// The identation characters, for example 2-space indentation would
    /// be represented by "  ".
    pattern: String,
    /// The number of times the indentation characters appear, for example
    /// if the line of code was indented three times, using 2-space indentation,
    /// for example "      <code goes here>",
    /// `size` would be 3 and `pattern` would be "  ".
    size: usize,
}

impl Indent {
    pub fn new(pattern: String, size: usize) -> Self {
        Self { pattern, size }
    }

    /// Return a new `Indent` with its size equal to the current `Indent`s
    /// size plus one.
    pub fn increase(&self) -> Self {
        Self::new(self.pattern.clone(), self.size + 1)
    }
}

impl Expr {
    pub fn is_if(&self) -> bool {
        matches!(self, Self::If(_))
    }

    pub fn is_block(&self) -> bool {
        matches!(self, Self::Block(_))
    }

    pub fn is_func(&self) -> bool {
        matches!(self, Self::Func(_))
    }

    pub fn id(&self) -> Uuid {
        match self {
            Self::Add(add) => add.0.id,
            Self::And(and) => and.0.id,
            Self::Cmp(cmp) => cmp.0.id,
            Self::Div(div) => div.0.id,
            Self::Eq(eq) => eq.0.id,
            Self::Gt(gt) => gt.0.id,
            Self::Gte(gte) => gte.0.id,
            Self::Lt(lt) => lt.0.id,
            Self::Lte(lte) => lte.0.id,
            Self::Mul(mul) => mul.0.id,
            Self::Ne(ne) => ne.0.id,
            Self::Or(or) => or.0.id,
            Self::Sub(sub) => sub.0.id,
            Self::Assign(asn) => asn.id,
            Self::Block(blk) => blk.id,
            Self::Call(call) => call.id,
            Self::Def(def) => def.id,
            Self::Dict(dict) => dict.id,
            Self::DotCall(dotc) => dotc.id,
            Self::Func(func) => func.id,
            Self::Id(id) => id.id,
            Self::If(iff) => iff.id,
            Self::List(lst) => lst.id,
            Self::Lit(lit) => lit.id,
            Self::Neg(neg) => neg.id,
            Self::Not(not) => not.id,
            Self::PartialCall(pcl) => pcl.id,
            Self::Range(rng) => rng.id,
            Self::Return(ret) => ret.id,
            Self::Throw(thr) => thr.id,
            Self::TrapCall(tcl) => tcl.id,
            Self::TryCatch(tc) => tc.id,
        }
    }

    /// May return an int representing how high the expression's precendence is,
    /// where 1 is the highest precedence.
    pub fn precedence(&self) -> Option<u8> {
        match self {
            Self::Add(add) => Some(add.0.precedence()),
            Self::And(and) => Some(and.0.precedence()),
            Self::Cmp(cmp) => Some(cmp.0.precedence()),
            Self::Div(div) => Some(div.0.precedence()),
            Self::Eq(eq) => Some(eq.0.precedence()),
            Self::Gt(gt) => Some(gt.0.precedence()),
            Self::Gte(gte) => Some(gte.0.precedence()),
            Self::Lt(lt) => Some(lt.0.precedence()),
            Self::Lte(lte) => Some(lte.0.precedence()),
            Self::Mul(mul) => Some(mul.0.precedence()),
            Self::Ne(ne) => Some(ne.0.precedence()),
            Self::Or(or) => Some(or.0.precedence()),
            Self::Sub(sub) => Some(sub.0.precedence()),
            Self::Assign(_) => Some(80),
            Self::Block(_) => None,
            Self::Call(_) => Some(1),
            Self::Def(_) => Some(80),
            Self::Dict(_) => None,
            Self::DotCall(_) => Some(1),
            Self::Func(_) => None,
            Self::Id(_) => None,
            Self::If(_) => None,
            Self::List(_) => None,
            Self::Lit(_) => None,
            Self::Neg(_) => Some(10),
            Self::Not(_) => Some(10),
            Self::PartialCall(_) => Some(1),
            Self::Range(_) => Some(5),
            Self::Return(_) => None,
            Self::Throw(_) => None,
            Self::TrapCall(_) => Some(1),
            Self::TryCatch(_) => None,
        }
    }

    pub fn associativity(&self) -> Option<Associativity> {
        match self {
            Self::Add(add) => add.0.associativity(),
            Self::And(and) => and.0.associativity(),
            Self::Cmp(cmp) => cmp.0.associativity(),
            Self::Div(div) => div.0.associativity(),
            Self::Eq(eq) => eq.0.associativity(),
            Self::Gt(gt) => gt.0.associativity(),
            Self::Gte(gte) => gte.0.associativity(),
            Self::Lt(lt) => lt.0.associativity(),
            Self::Lte(lte) => lte.0.associativity(),
            Self::Mul(mul) => mul.0.associativity(),
            Self::Ne(ne) => ne.0.associativity(),
            Self::Or(or) => or.0.associativity(),
            Self::Sub(sub) => sub.0.associativity(),
            Self::Assign(_) => Some(Associativity::Right),
            Self::Block(_) => None, // Requires explicit parentheses to parse.
            Self::Call(_) => None,  // Requires explicit parentheses to parse.
            Self::Def(_) => Some(Associativity::Right),
            Self::Dict(_) => None,
            Self::DotCall(_) => None, // Sort of left associative, but a.(b.c) will not parse.
            Self::Func(_) => None,
            Self::Id(_) => None,
            Self::If(_) => None,
            Self::List(_) => None,
            Self::Lit(_) => None,
            Self::Neg(_) => None, // --1 will not parse.
            Self::Not(_) => None, // "not not true" will not parse.
            Self::PartialCall(_) => None,
            Self::Range(_) => None, // 1..2..3 will not parse.
            Self::Return(_) => Some(Associativity::Right),
            Self::Throw(_) => Some(Associativity::Right),
            Self::TrapCall(_) => Some(Associativity::Left),
            Self::TryCatch(_) => None,
        }
    }

    /// Returns true if this expression is a binary operation.
    pub fn is_bin_op(&self) -> bool {
        match self {
            Self::Add(_) => true,
            Self::And(_) => true,
            Self::Cmp(_) => true,
            Self::Div(_) => true,
            Self::Eq(_) => true,
            Self::Gt(_) => true,
            Self::Gte(_) => true,
            Self::Lt(_) => true,
            Self::Lte(_) => true,
            Self::Mul(_) => true,
            Self::Ne(_) => true,
            Self::Or(_) => true,
            Self::Sub(_) => true,
            _ => false,
        }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        match self {
            Self::Add(add) => add.to_line(indent),
            Self::And(and) => and.to_line(indent),
            Self::Cmp(cmp) => cmp.to_line(indent),
            Self::Div(div) => div.to_line(indent),
            Self::Eq(eq) => eq.to_line(indent),
            Self::Gt(gt) => gt.to_line(indent),
            Self::Gte(gte) => gte.to_line(indent),
            Self::Lt(lt) => lt.to_line(indent),
            Self::Lte(lte) => lte.to_line(indent),
            Self::Mul(mul) => mul.to_line(indent),
            Self::Ne(ne) => ne.to_line(indent),
            Self::Or(or) => or.to_line(indent),
            Self::Sub(sub) => sub.to_line(indent),
            Self::Assign(assign) => assign.to_line(indent),
            Self::Block(block) => block.to_line(indent),
            Self::Call(call) => call.to_line(indent),
            Self::Def(def) => def.to_line(indent),
            Self::Dict(dict) => dict.to_line(indent),
            Self::Func(func) => func.to_line(indent),
            Self::DotCall(dot_call) => dot_call.to_line(indent),
            Self::Id(id) => {
                Line::new(indent.clone(), id.name().clone().into_string())
            }
            Self::If(iff) => iff.to_line(indent),
            Self::List(list) => list.to_line(indent),
            Self::Lit(lit) => Line::new(indent.clone(), lit.lit().to_axon_code()),
            Self::Neg(neg) => neg.to_line(indent),
            Self::Not(not) => not.to_line(indent),
            Self::PartialCall(partial_call) => partial_call.to_line(indent),
            Self::Range(range) => range.to_line(indent),
            Self::Return(ret) => ret.to_line(indent),
            Self::Throw(throw) => throw.to_line(indent),
            Self::TrapCall(trap_call) => trap_call.to_line(indent),
            Self::TryCatch(try_catch) => try_catch.to_line(indent),
        }
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        match self {
            Self::Add(add) => add.to_lines(indent),
            Self::And(and) => and.to_lines(indent),
            Self::Cmp(cmp) => cmp.to_lines(indent),
            Self::Div(div) => div.to_lines(indent),
            Self::Eq(eq) => eq.to_lines(indent),
            Self::Gt(gt) => gt.to_lines(indent),
            Self::Gte(gte) => gte.to_lines(indent),
            Self::Lt(lt) => lt.to_lines(indent),
            Self::Lte(lte) => lte.to_lines(indent),
            Self::Mul(mul) => mul.to_lines(indent),
            Self::Ne(ne) => ne.to_lines(indent),
            Self::Or(or) => or.to_lines(indent),
            Self::Sub(sub) => sub.to_lines(indent),
            Self::Assign(assign) => assign.to_lines(indent),
            Self::Block(block) => block.to_lines(indent),
            Self::Call(call) => call.to_lines(indent),
            Self::Def(def) => def.to_lines(indent),
            Self::Dict(dict) => dict.to_lines(indent),
            Self::DotCall(dot_call) => dot_call.to_lines(indent),
            Self::Func(func) => func.to_lines(indent),
            Self::Id(id) => {
                vec![Line::new(indent.clone(), id.name().clone().into_string())]
            }
            Self::If(iff) => iff.to_lines(indent),
            Self::List(list) => list.to_lines(indent),
            Self::Lit(lit) => {
                vec![Line::new(indent.clone(), lit.lit().to_axon_code())]
            }
            Self::Neg(neg) => neg.to_lines(indent),
            Self::Not(not) => not.to_lines(indent),
            Self::PartialCall(partial_call) => partial_call.to_lines(indent),
            Self::Range(range) => range.to_lines(indent),
            Self::Return(ret) => ret.to_lines(indent),
            Self::Throw(throw) => throw.to_lines(indent),
            Self::TrapCall(trap_call) => trap_call.to_lines(indent),
            Self::TryCatch(try_catch) => try_catch.to_lines(indent),
        }
    }

    /// If the expression is not a block, wrap the expression in a block.
    pub fn blockify(self) -> Self {
        if self.is_block() {
            self
        } else {
            Expr::Block(Block::new(vec![self]))
        }
    }
}

impl std::fmt::Display for Indent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.pattern.repeat(self.size))
    }
}

impl TryFrom<&Val> for Expr {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let lit: Option<LitInner> = val.try_into().ok();
        if let Some(lit) = lit {
            return Ok(Expr::Lit(Lit::new(lit)));
        };

        if let Some(tag_name) = var_val_to_tag_name(val) {
            return Ok(Expr::Id(Id::new(tag_name)));
        };

        let assign: Option<Assign> = val.try_into().ok();
        if let Some(assign) = assign {
            return Ok(Expr::Assign(assign));
        };

        let def: Option<Def> = val.try_into().ok();
        if let Some(def) = def {
            return Ok(Expr::Def(def));
        };

        let list: Option<List> = val.try_into().ok();
        if let Some(list) = list {
            return Ok(Expr::List(list));
        };

        let dict: Option<Dict> = val.try_into().ok();
        if let Some(dict) = dict {
            return Ok(Expr::Dict(dict));
        };

        let block: Option<Block> = val.try_into().ok();
        if let Some(block) = block {
            return Ok(Expr::Block(block));
        }

        let range: Option<Range> = val.try_into().ok();
        if let Some(range) = range {
            return Ok(Expr::Range(Box::new(range)));
        }

        let call: Option<Call> = val.try_into().ok();
        if let Some(call) = call {
            return Ok(Expr::Call(call));
        }

        let dot_call: Option<DotCall> = val.try_into().ok();
        if let Some(dot_call) = dot_call {
            return Ok(Expr::DotCall(dot_call));
        }

        let trap_call: Option<TrapCall> = val.try_into().ok();
        if let Some(trap_call) = trap_call {
            return Ok(Expr::TrapCall(Box::new(trap_call)));
        }

        let add: Option<Add> = val.try_into().ok();
        if let Some(add) = add {
            return Ok(Expr::Add(Box::new(add)));
        }

        let and: Option<And> = val.try_into().ok();
        if let Some(and) = and {
            return Ok(Expr::And(Box::new(and)));
        }

        let cmp: Option<Cmp> = val.try_into().ok();
        if let Some(cmp) = cmp {
            return Ok(Expr::Cmp(Box::new(cmp)));
        }

        let div: Option<Div> = val.try_into().ok();
        if let Some(div) = div {
            return Ok(Expr::Div(Box::new(div)));
        }

        let eq: Option<Eq> = val.try_into().ok();
        if let Some(eq) = eq {
            return Ok(Expr::Eq(Box::new(eq)));
        }

        let gt: Option<Gt> = val.try_into().ok();
        if let Some(gt) = gt {
            return Ok(Expr::Gt(Box::new(gt)));
        }

        let gte: Option<Gte> = val.try_into().ok();
        if let Some(gte) = gte {
            return Ok(Expr::Gte(Box::new(gte)));
        }

        let lt: Option<Lt> = val.try_into().ok();
        if let Some(lt) = lt {
            return Ok(Expr::Lt(Box::new(lt)));
        }

        let lte: Option<Lte> = val.try_into().ok();
        if let Some(lte) = lte {
            return Ok(Expr::Lte(Box::new(lte)));
        }

        let mul: Option<Mul> = val.try_into().ok();
        if let Some(mul) = mul {
            return Ok(Expr::Mul(Box::new(mul)));
        }

        let ne: Option<Ne> = val.try_into().ok();
        if let Some(ne) = ne {
            return Ok(Expr::Ne(Box::new(ne)));
        }

        let or: Option<Or> = val.try_into().ok();
        if let Some(or) = or {
            return Ok(Expr::Or(Box::new(or)));
        }

        let sub: Option<Sub> = val.try_into().ok();
        if let Some(sub) = sub {
            return Ok(Expr::Sub(Box::new(sub)));
        }

        let try_catch: Option<TryCatch> = val.try_into().ok();
        if let Some(try_catch) = try_catch {
            return Ok(Expr::TryCatch(Box::new(try_catch)));
        }

        let iff: Option<If> = val.try_into().ok();
        if let Some(iff) = iff {
            return Ok(Expr::If(Box::new(iff)));
        }

        let func: Option<Func> = val.try_into().ok();
        if let Some(func) = func {
            return Ok(Expr::Func(Box::new(func)));
        }

        let throw: Option<Throw> = val.try_into().ok();
        if let Some(throw) = throw {
            return Ok(Expr::Throw(Box::new(throw)));
        }

        let ret: Option<Return> = val.try_into().ok();
        if let Some(ret) = ret {
            return Ok(Expr::Return(Box::new(ret)));
        }

        let neg: Option<Neg> = val.try_into().ok();
        if let Some(neg) = neg {
            return Ok(Expr::Neg(Box::new(neg)));
        }

        let not: Option<Not> = val.try_into().ok();
        if let Some(not) = not {
            return Ok(Expr::Not(Box::new(not)));
        }

        let partial_call: Option<PartialCall> = val.try_into().ok();
        if let Some(partial_call) = partial_call {
            return Ok(Expr::PartialCall(partial_call));
        }

        Err(())
    }
}

/// Converts something like '{type:"var", name:"siteId"}' to
/// a TagName like `TagName::new("siteId")`.
fn var_val_to_tag_name(val: &Val) -> Option<TagName> {
    let hash_map = map_for_type(val, "var").ok()?;
    let tag_name = get_literal_str(hash_map, "name")?;
    let tag_name = TagName::new(tag_name.to_owned()).unwrap_or_else(|| {
        panic!(
            "'name' tag in a var should be a valid TagName: {}",
            tag_name
        )
    });
    Some(tag_name)
}

#[derive(Clone, Debug)]
pub struct Assign {
    id: Uuid,
    /// lhs
    pub name: TagName,
    /// rhs
    pub expr: Box<Expr>,
}

impl PartialEq for Assign {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.expr == other.expr
    }
}

impl Assign {
    pub fn new(name: TagName, expr: Expr) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            expr: Box::new(expr),
        }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        let line = self.expr.to_line(indent);
        line.prefix_str(&format!("{} = ", self.name))
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        let mut expr_lines = self.expr.to_lines(&indent);
        let first_expr_line = expr_lines
            .first()
            .expect("Assign expr should contain at least one line");
        let new_inner_str =
            format!("{} = {}", self.name, first_expr_line.inner_str());
        let new_first_line =
            Line::new(first_expr_line.indent().clone(), new_inner_str);
        expr_lines[0] = new_first_line;
        expr_lines
    }
}

impl TryFrom<&Val> for Assign {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "assign").map_err(|_| ())?;
        let lhs =
            get_val(hash_map, "lhs").expect("assign should have a 'lhs' tag");
        let name = var_val_to_tag_name(lhs)
            .expect("assign lhs should be a var with a 'name' tag");
        let assign_val = get_val(hash_map, "rhs")
            .expect("assign should have a 'rhs' tag")
            .try_into()?;
        Ok(Self::new(name, assign_val))
    }
}

#[derive(Clone, Debug)]
pub struct Def {
    id: Uuid,
    pub name: TagName,
    pub expr: Box<Expr>,
}

impl PartialEq for Def {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.expr == other.expr
    }
}

impl Def {
    pub fn new(name: TagName, expr: Expr) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            expr: Box::new(expr),
        }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        let line = self.expr.to_line(indent);
        line.prefix_str(&format!("{}: ", self.name))
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        let mut expr_lines = self.expr.to_lines(&indent);
        let first_expr_line = expr_lines
            .first()
            .expect("Def expr should contain at least one line");
        let new_inner_str =
            format!("{}: {}", self.name, first_expr_line.inner_str());
        let new_first_line =
            Line::new(first_expr_line.indent().clone(), new_inner_str);
        expr_lines[0] = new_first_line;
        expr_lines
    }
}

impl TryFrom<&Val> for Def {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "def").map_err(|_| ())?;
        let name = tn(get_literal_str(hash_map, "name")
            .expect("def should have a string 'name' tag"));
        let def_val = get_val(hash_map, "val")
            .expect("def should have a 'val' tag")
            .try_into()?;
        Ok(Self::new(name, def_val))
    }
}

fn params_to_line(params: &[Param], indent: &Indent) -> Line {
    let lines = params
        .iter()
        .map(|param| param.to_line(indent).inner_str().to_owned())
        .collect::<Vec<_>>();
    let line_str = lines.join(", ");
    Line::new(indent.clone(), line_str)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: TagName,
    pub default: Option<Expr>,
}

impl Param {
    pub fn new(name: TagName, default: Option<Expr>) -> Self {
        Self { name, default }
    }

    pub fn to_line(&self, indent: &Indent) -> Line {
        match &self.default {
            Some(default) => {
                let zero_indent = zero_indent();
                let default_line = default.to_line(&zero_indent);
                let default_str = default_line.inner_str();
                Line::new(
                    indent.clone(),
                    format!("{}: {}", self.name, default_str),
                )
            }
            None => Line::new(indent.clone(), self.name.clone().into_string()),
        }
    }

    pub fn to_lines(&self, indent: &Indent) -> Lines {
        // TODO handle multi line params
        let line = self.to_line(indent);
        vec![line]
    }
}

fn tn(tag_name: &str) -> TagName {
    TagName::new(tag_name.to_owned()).unwrap_or_else(|| {
        panic!("expected '{}' to be a valid tag name", tag_name)
    })
}

impl TryFrom<&ap::Val> for Param {
    type Error = ();

    fn try_from(val: &ap::Val) -> Result<Self, Self::Error> {
        // NOTE: Since there is no 'type' tag to identify which `Val::Dict`s
        // are actually params, maybe this could accidentally
        // parse a non-param `Val` into a param.
        match val {
            Val::Dict(hash_map) => match hash_map.get(&tn("type")) {
                Some(_) => Err(()),
                None => {
                    let name = get_literal_str(hash_map, "name").expect(
                        "param should have 'name' tag with a string value",
                    );
                    let def_val = get_val(hash_map, "def");
                    let param = match def_val {
                        Some(def_val) => {
                            let param_def_val = def_val.try_into()?;
                            Param::new(tn(name), Some(param_def_val))
                        }
                        None => Param::new(tn(name), None),
                    };
                    Ok(param)
                }
            },
            _ => Err(()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Lit {
    id: Uuid,
    lit: LitInner,
}

impl PartialEq for Lit {
    fn eq(&self, other: &Self) -> bool {
        self.lit == other.lit
    }
}

impl Lit {
    pub fn new(lit: LitInner) -> Self {
        Self {
            id: Uuid::new_v4(),
            lit,
        }
    }

    pub fn lit(&self) -> &LitInner {
        &self.lit
    }
}

/// A Axon literal value.
#[derive(Clone, Debug, PartialEq)]
pub enum LitInner {
    Bool(bool),
    Date(NaiveDate),
    Null,
    Num(Number),
    Ref(Ref),
    Str(String),
    Symbol(Symbol),
    Time(NaiveTime),
    Uri(String),
    YearMonth(YearMonth),
}

impl LitInner {
    pub fn to_axon_code(&self) -> String {
        match self {
            Self::Bool(true) => "true".to_owned(),
            Self::Bool(false) => "false".to_owned(),
            Self::Date(d) => d.format("%Y-%m-%d").to_string(),
            Self::Null => "null".to_owned(),
            Self::Num(n) => number_to_axon_code(n),
            Self::Ref(r) => r.to_axon_code().to_owned(),
            Self::Str(s) => format!("{:?}", s).replace(r"\\$", r"\$"),
            Self::Symbol(s) => s.to_axon_code().to_owned(),
            Self::Time(t) => t.format("%H:%M:%S%.f").to_string(),
            Self::Uri(s) => format!("`{}`", s),
            Self::YearMonth(ym) => ym.to_axon_code(),
        }
    }
}

fn number_to_axon_code(n: &Number) -> String {
    let scalar = n.value();
    match n.unit() {
        Some(unit) => format!("{}{}", scalar, unit),
        None => format!("{}", scalar),
    }
}

/// Enumerates the types of errors that can occur when converting to a
/// `Lit`.
pub enum ConvertLitError {
    IsDictMarker,
    IsDictRemoveMarker,
}

impl TryFrom<&ap::Lit> for LitInner {
    type Error = ConvertLitError;

    fn try_from(lit: &ap::Lit) -> Result<Self, Self::Error> {
        match lit {
            ap::Lit::Bool(bool) => Ok(LitInner::Bool(*bool)),
            ap::Lit::Date(date) => Ok(LitInner::Date(*date)),
            ap::Lit::DictMarker => Err(ConvertLitError::IsDictMarker),
            ap::Lit::DictRemoveMarker => {
                Err(ConvertLitError::IsDictRemoveMarker)
            }
            ap::Lit::Null => Ok(LitInner::Null),
            ap::Lit::Num(number) => Ok(LitInner::Num(number.clone())),
            ap::Lit::Ref(reff) => Ok(LitInner::Ref(reff.clone())),
            ap::Lit::Str(string) => Ok(LitInner::Str(string.clone())),
            ap::Lit::Symbol(symbol) => Ok(LitInner::Symbol(symbol.clone())),
            ap::Lit::Time(time) => Ok(LitInner::Time(*time)),
            ap::Lit::Uri(uri) => Ok(LitInner::Uri(uri.clone())),
            ap::Lit::YearMonth(ym) => Ok(LitInner::YearMonth(ym.into())),
        }
    }
}

impl TryFrom<&Val> for LitInner {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "literal").map_err(|_| ())?;
        let val = get_val(hash_map, "val")
            .expect("type 'literal' should have 'val' tag");
        match val {
            Val::Lit(lit) => {
                let lit = lit.try_into().map_err(|_| ())?;
                Ok(lit)
            }
            _ => panic!("expected type 'literal' 'val' tag to be a literal"),
        }
    }
}

fn map_for_type<'a, 'b>(
    val: &'a Val,
    type_name: &'b str,
) -> Result<&'a HashMap<TagName, Box<Val>>, MapForTypeError> {
    match val {
        Val::Dict(hash_map) => {
            if type_str(hash_map) == type_name {
                Ok(hash_map)
            } else {
                Err(MapForTypeError::TypeStringMismatch)
            }
        }
        _ => Err(MapForTypeError::NotDict),
    }
}

/// Enumerates the types of errors that can occur when trying to convert
/// a loosely-typed value into a strongly-typed map.
#[derive(Debug, Clone, PartialEq)]
pub enum MapForTypeError {
    NotDict,
    TypeStringMismatch,
}

fn get_vals<'a, 'b>(
    hash_map: &'a HashMap<TagName, Box<Val>>,
    tag_name: &'b str,
) -> Option<&'a Vec<Val>> {
    let tag_name = tn(tag_name);
    let val = hash_map.get(&tag_name).map(|val| val.as_ref())?;
    match val {
        Val::List(elements) => Some(elements),
        _ => None,
    }
}

fn get_literal_str<'a, 'b>(
    hash_map: &'a HashMap<TagName, Box<Val>>,
    tag_name: &'b str,
) -> Option<&'a str> {
    let tag_name = tn(tag_name);
    let val = hash_map.get(&tag_name).map(|val| val.as_ref())?;
    match val {
        Val::Lit(ap::Lit::Str(s)) => Some(s),
        _ => None,
    }
}

fn get_val<'a, 'b>(
    hash_map: &'a HashMap<TagName, Box<Val>>,
    tag_name: &'b str,
) -> Option<&'a Val> {
    let tag_name = tn(tag_name);
    hash_map.get(&tag_name).map(|val| val.as_ref())
}

fn type_str(hash_map: &HashMap<TagName, Box<Val>>) -> &str {
    get_literal_str(hash_map, "type")
        .expect("expected the 'type' tag's value to be a literal string")
}

/// Represents a month in a specific year.
#[derive(Clone, Debug, PartialEq)]
pub struct YearMonth {
    pub year: u32,
    pub month: Month,
}

impl YearMonth {
    pub fn new(year: u32, month: Month) -> Self {
        Self { year, month }
    }

    pub fn to_axon_code(&self) -> String {
        let month: u32 = (&self.month).into();
        format!("{}-{:0>2}", self.year, month)
    }
}

impl From<&ap::YearMonth> for YearMonth {
    fn from(ym: &ap::YearMonth) -> Self {
        let year = ym.year;
        let month = ym.month.clone();
        Self {
            year,
            month: month.into(),
        }
    }
}

/// Represents a month.
#[derive(Clone, Debug, PartialEq)]
pub enum Month {
    Jan,
    Feb,
    Mar,
    Apr,
    May,
    Jun,
    Jul,
    Aug,
    Sep,
    Oct,
    Nov,
    Dec,
}

impl From<&Month> for u32 {
    fn from(m: &Month) -> u32 {
        match m {
            Month::Jan => 1,
            Month::Feb => 2,
            Month::Mar => 3,
            Month::Apr => 4,
            Month::May => 5,
            Month::Jun => 6,
            Month::Jul => 7,
            Month::Aug => 8,
            Month::Sep => 9,
            Month::Oct => 10,
            Month::Nov => 11,
            Month::Dec => 12,
        }
    }
}

impl From<ap::Month> for Month {
    fn from(month: ap::Month) -> Self {
        match month {
            ap::Month::Jan => Month::Jan,
            ap::Month::Feb => Month::Feb,
            ap::Month::Mar => Month::Mar,
            ap::Month::Apr => Month::Apr,
            ap::Month::May => Month::May,
            ap::Month::Jun => Month::Jun,
            ap::Month::Jul => Month::Jul,
            ap::Month::Aug => Month::Aug,
            ap::Month::Sep => Month::Sep,
            ap::Month::Oct => Month::Oct,
            ap::Month::Nov => Month::Nov,
            ap::Month::Dec => Month::Dec,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_parseast_parser::parse as ap_parse;
    use raystack_core::{Number, Qname};
    use std::convert::TryInto;

    fn lit_str(s: &str) -> Lit {
        Lit::new(LitInner::Str(s.to_owned()))
    }

    fn lit_num(n: f64) -> Lit {
        Lit::new(LitInner::Num(Number::new(n, None)))
    }

    fn idtn(tag_name: &str) -> Id {
        Id::new(tn(tag_name))
    }

    fn tn(tag_name: &str) -> TagName {
        TagName::new(tag_name.to_owned()).unwrap()
    }

    const HELLO_WORLD: &str = r###"{type:"func", params:[], body:{type:"block", exprs:[{type:"literal", val:"hello world"}]}}"###;

    #[test]
    fn hello_world_works() {
        let val = &ap_parse(HELLO_WORLD).unwrap();
        let params = vec![];
        let lit_expr = Expr::Lit(lit_str("hello world"));
        let block = Block::new(vec![lit_expr]);
        let body = Expr::Block(block);
        let expected = Func::new(params, body);
        let func: Func = val.try_into().unwrap();
        assert_eq!(func, expected);
    }

    #[test]
    fn val_to_lit_works() {
        let val = &ap_parse(r#"{type:"literal", val:"hello"}"#).unwrap();
        let expected = LitInner::Str("hello".to_owned());
        let lit: LitInner = val.try_into().unwrap();
        assert_eq!(lit, expected);
    }

    #[test]
    fn val_to_lit_symbol_works() {
        let val = &ap_parse(r#"{type:"literal", val:^steam-boiler}"#).unwrap();
        let expected =
            LitInner::Symbol(Symbol::new("^steam-boiler".to_owned()).unwrap());
        let lit: LitInner = val.try_into().unwrap();
        assert_eq!(lit, expected);
    }

    #[test]
    fn val_to_simple_param_works() {
        let val = &ap_parse(r#"{name:"ahu"}"#).unwrap();
        let expected = Param::new(tn("ahu"), None);
        let param: Param = val.try_into().unwrap();
        assert_eq!(param, expected);
    }

    #[test]
    fn val_to_param_with_default_val_literal_works() {
        let val =
            &ap_parse(r#"{name:"ahu", def:{type:"literal", val:1}}"#).unwrap();
        let default = Expr::Lit(lit_num(1.0));
        let expected = Param::new(tn("ahu"), Some(default));
        let param: Param = val.try_into().unwrap();
        assert_eq!(param, expected);
    }

    #[test]
    fn val_to_def_literal_works() {
        let val = &ap_parse(
            r#"{type:"def", name:"siteId", val:{type:"literal", val:1}}"#,
        )
        .unwrap();
        let expr = Expr::Lit(lit_num(1.0));
        let expected = Def::new(tn("siteId"), expr);
        let def: Def = val.try_into().unwrap();
        assert_eq!(def, expected);
    }

    #[test]
    fn val_to_assign_literal_works() {
        let val = &ap_parse(
            r#"{type:"assign", lhs:{type:"var", name:"siteId"}, rhs:{type:"literal", val:1}}"#,
        )
        .unwrap();
        let assign_val = Expr::Lit(lit_num(1.0));
        let expected = Assign::new(tn("siteId"), assign_val);
        let assign: Assign = val.try_into().unwrap();
        assert_eq!(assign, expected);
    }

    #[test]
    fn val_to_throw_str_literal_works() {
        let val = &ap_parse(
            r#"{type:"throw", expr:{type:"literal", val:"error message"}}"#,
        )
        .unwrap();
        let expr = Expr::Lit(lit_str("error message"));
        let expected = Throw::new(expr);
        let throw: Throw = val.try_into().unwrap();
        assert_eq!(throw, expected);
    }

    #[test]
    fn val_to_non_empty_list_works() {
        let val = &ap_parse(r#"{type:"list", vals:[{type:"literal", val:1}]}"#)
            .unwrap();
        let expr = Expr::Lit(lit_num(1.0));
        let expected = List::new(vec![expr]);
        let list: List = val.try_into().unwrap();
        assert_eq!(list, expected);
    }

    #[test]
    fn val_to_non_empty_dict_works() {
        let val = &ap_parse(r#"{type:"dict", names:["markerTag", "deleteThisTag", "standardTag"], vals:[{type:"literal", val}, {type:"literal", val:removeMarker()}, {type:"literal", val:1}]}"#).unwrap();

        let mut map: HashMap<TagName, DictVal> = HashMap::new();
        let tag1 = tn("markerTag");
        let val1 = DictVal::Marker;
        let tag2 = tn("deleteThisTag");
        let val2 = DictVal::RemoveMarker;
        let tag3 = tn("standardTag");
        let val3 = DictVal::Expr(Expr::Lit(lit_num(1.0)));
        map.insert(tag1, val1);
        map.insert(tag2, val2);
        map.insert(tag3, val3);

        let expected = Dict::new(map);
        let dict: Dict = val.try_into().unwrap();
        assert_eq!(dict, expected);
    }

    #[test]
    fn val_to_single_expr_block_works() {
        let val = &ap_parse(
            r#"{type:"block", exprs:[{type:"literal", val:"hello"}]}"#,
        )
        .unwrap();
        let expr = Expr::Lit(lit_str("hello"));
        let exprs = vec![expr];
        let expected = Block::new(exprs);

        let block: Block = val.try_into().unwrap();
        assert_eq!(block, expected);
    }

    #[test]
    fn val_to_simple_range_works() {
        let val = &ap_parse(r#"{type:"range", start:{type:"literal", val:1}, end:{type:"literal", val:2}}"#).unwrap();
        let start = Expr::Lit(lit_num(1.0));
        let end = Expr::Lit(lit_num(2.0));
        let expected = Range::new(start, end);
        let range: Range = val.try_into().unwrap();
        assert_eq!(range, expected);
    }

    #[test]
    fn val_to_not_works() {
        let val = &ap_parse(r#"{type:"not", operand:{type:"literal", val:1}}"#)
            .unwrap();
        let operand = Expr::Lit(lit_num(1.0));
        let expected = Not::new(operand);
        let not: Not = val.try_into().unwrap();
        assert_eq!(not, expected);
    }

    #[test]
    fn val_to_simple_call_works() {
        let val = &ap_parse(r#"{type:"call", target:{type:"var", name:"readAll"}, args:[{type:"literal", val:1}]}"#).unwrap();
        let func_name = tn("readAll");
        let arg = Expr::Lit(lit_num(1.0));
        let args = vec![arg];
        let func_name = FuncName::TagName(func_name);
        let target = CallTarget::FuncName(func_name);
        let expected = Call::new(target, args);
        let call: Call = val.try_into().unwrap();
        assert_eq!(call, expected);
    }

    #[test]
    fn val_to_qname_call_works() {
        let val = &ap_parse(r#"{type:"call", target:{type:"var", name:"core::readAll"}, args:[{type:"literal", val:1}]}"#).unwrap();
        let qname = Qname::new("core::readAll".to_owned());
        let arg = Expr::Lit(lit_num(1.0));
        let args = vec![arg];
        let func_name = FuncName::Qname(qname);
        let target = CallTarget::FuncName(func_name);
        let expected = Call::new(target, args);
        let call: Call = val.try_into().unwrap();
        assert_eq!(call, expected);
    }

    #[test]
    fn val_to_expr_call_works() {
        let val = &ap_parse(r#"{type:"call", target:{type:"trapCall", target:{type:"var", name:"trap"}, args:[{type:"var", name:"x"}, {type:"literal", val:"someFunc"}]}, args:[]}"#).unwrap();
        let var_expr = Expr::Id(idtn("x"));
        let trap_call = TrapCall::new(var_expr, "someFunc".to_owned());
        let target = Expr::TrapCall(Box::new(trap_call));
        let target = CallTarget::Expr(Box::new(target));
        let args = vec![];
        let expected = Call::new(target, args);
        let call: Call = val.try_into().unwrap();
        assert_eq!(call, expected);
    }

    #[test]
    fn val_to_simple_dot_call_works() {
        let val = &ap_parse(r#"{type:"dotCall", target:{type:"var", name:"parseNumber"}, args:[{type:"literal", val:1}]}"#).unwrap();
        let func_name = tn("parseNumber");
        let target = Expr::Lit(lit_num(1.0));
        let args = vec![];
        let expected =
            DotCall::new(FuncName::TagName(func_name), Box::new(target), args);
        let dot_call: DotCall = val.try_into().unwrap();
        assert_eq!(dot_call, expected);
    }

    #[test]
    fn val_to_qname_dot_call_works() {
        let val = &ap_parse(r#"{type:"dotCall", target:{type:"var", name:"core::parseNumber"}, args:[{type:"literal", val:1}]}"#).unwrap();
        let qname = Qname::new("core::parseNumber".to_owned());
        let target = Expr::Lit(lit_num(1.0));
        let args = vec![];
        let expected =
            DotCall::new(FuncName::Qname(qname), Box::new(target), args);
        let dot_call: DotCall = val.try_into().unwrap();
        assert_eq!(dot_call, expected);
    }

    #[test]
    fn val_to_partial_call_works() {
        let val = &ap_parse(r#"{type:"partialCall", target:{type:"var", name:"utilsAssert"}, args:[null, {type:"literal", val:1}]}"#).unwrap();
        let func_name = FuncName::TagName(tn("utilsAssert"));
        let null = PartialCallArgument::Placeholder;
        let arg2 = PartialCallArgument::Expr(Expr::Lit(lit_num(1.0)));
        let expected = PartialCall::new(func_name, vec![null, arg2]);

        let partial_call: PartialCall = val.try_into().unwrap();
        assert_eq!(partial_call, expected);
    }

    #[test]
    fn val_to_trap_call_works() {
        let val = &ap_parse(r#"{type:"trapCall", target:{type:"var", name:"trap"}, args:[{type:"dict", names:[], vals:[]}, {type:"literal", val:"missingTag"}]}"#).unwrap();
        let key = "missingTag".to_owned();
        let target = Expr::Dict(Dict::new(HashMap::new()));
        let expected = TrapCall::new(target, key);
        let trap_call: TrapCall = val.try_into().unwrap();
        assert_eq!(trap_call, expected);
    }

    #[test]
    fn val_to_and_works() {
        let val = &ap_parse(r#"{type:"and", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(idtn("a"));
        let rhs = Expr::Id(idtn("b"));
        let expected = And(BinOp::new(lhs, BinOpId::And, rhs));
        let and: And = val.try_into().unwrap();
        assert_eq!(and, expected);
    }

    #[test]
    fn val_to_or_works() {
        let val = &ap_parse(r#"{type:"or", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(idtn("a"));
        let rhs = Expr::Id(idtn("b"));
        let expected = Or(BinOp::new(lhs, BinOpId::Or, rhs));
        let or: Or = val.try_into().unwrap();
        assert_eq!(or, expected);
    }

    #[test]
    fn val_to_add_works() {
        let val = &ap_parse(r#"{type:"add", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(idtn("a"));
        let rhs = Expr::Id(idtn("b"));
        let expected = Add(BinOp::new(lhs, BinOpId::Add, rhs));
        let add: Add = val.try_into().unwrap();
        assert_eq!(add, expected);
    }

    #[test]
    fn val_to_cmp_works() {
        let val = &ap_parse(r#"{type:"cmp", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(idtn("a"));
        let rhs = Expr::Id(idtn("b"));
        let expected = Cmp(BinOp::new(lhs, BinOpId::Cmp, rhs));
        let cmp: Cmp = val.try_into().unwrap();
        assert_eq!(cmp, expected);
    }

    #[test]
    fn val_to_div_works() {
        let val = &ap_parse(r#"{type:"div", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(idtn("a"));
        let rhs = Expr::Id(idtn("b"));
        let expected = Div(BinOp::new(lhs, BinOpId::Div, rhs));
        let div: Div = val.try_into().unwrap();
        assert_eq!(div, expected);
    }

    #[test]
    fn val_to_eq_works() {
        let val = &ap_parse(r#"{type:"eq", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(idtn("a"));
        let rhs = Expr::Id(idtn("b"));
        let expected = Eq(BinOp::new(lhs, BinOpId::Eq, rhs));
        let eq: Eq = val.try_into().unwrap();
        assert_eq!(eq, expected);
    }

    #[test]
    fn val_to_gt_works() {
        let val = &ap_parse(r#"{type:"gt", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(idtn("a"));
        let rhs = Expr::Id(idtn("b"));
        let expected = Gt(BinOp::new(lhs, BinOpId::Gt, rhs));
        let gt: Gt = val.try_into().unwrap();
        assert_eq!(gt, expected);
    }

    #[test]
    fn val_to_gte_works() {
        let val = &ap_parse(r#"{type:"ge", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(idtn("a"));
        let rhs = Expr::Id(idtn("b"));
        let expected = Gte(BinOp::new(lhs, BinOpId::Gte, rhs));
        let gte: Gte = val.try_into().unwrap();
        assert_eq!(gte, expected);
    }

    #[test]
    fn val_to_lt_works() {
        let val = &ap_parse(r#"{type:"lt", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(idtn("a"));
        let rhs = Expr::Id(idtn("b"));
        let expected = Lt(BinOp::new(lhs, BinOpId::Lt, rhs));
        let lt: Lt = val.try_into().unwrap();
        assert_eq!(lt, expected);
    }

    #[test]
    fn val_to_lte_works() {
        let val = &ap_parse(r#"{type:"le", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(idtn("a"));
        let rhs = Expr::Id(idtn("b"));
        let expected = Lte(BinOp::new(lhs, BinOpId::Lte, rhs));
        let lte: Lte = val.try_into().unwrap();
        assert_eq!(lte, expected);
    }

    #[test]
    fn val_to_mul_works() {
        let val = &ap_parse(r#"{type:"mul", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(idtn("a"));
        let rhs = Expr::Id(idtn("b"));
        let expected = Mul(BinOp::new(lhs, BinOpId::Mul, rhs));
        let mul: Mul = val.try_into().unwrap();
        assert_eq!(mul, expected);
    }

    #[test]
    fn val_to_ne_works() {
        let val = &ap_parse(r#"{type:"ne", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(idtn("a"));
        let rhs = Expr::Id(idtn("b"));
        let expected = Ne(BinOp::new(lhs, BinOpId::Ne, rhs));
        let ne: Ne = val.try_into().unwrap();
        assert_eq!(ne, expected);
    }

    #[test]
    fn val_to_sub_works() {
        let val = &ap_parse(r#"{type:"sub", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(idtn("a"));
        let rhs = Expr::Id(idtn("b"));
        let expected = Sub(BinOp::new(lhs, BinOpId::Sub, rhs));
        let sub: Sub = val.try_into().unwrap();
        assert_eq!(sub, expected);
    }

    #[test]
    fn val_to_if_no_else_works() {
        let val = &ap_parse(r#"{type:"if", cond:{type:"var", name:"a"}, ifExpr:{type:"var", name:"b"}}"#).unwrap();
        let cond = Expr::Id(idtn("a"));
        let if_expr = Expr::Id(idtn("b"));
        let else_expr = None;
        let expected = If::new(cond, if_expr, else_expr);
        let iff: If = val.try_into().unwrap();
        assert_eq!(iff, expected);
    }

    #[test]
    fn val_to_if_with_else_works() {
        let val = &ap_parse(r#"{type:"if", cond:{type:"var", name:"a"}, ifExpr:{type:"var", name:"b"}, elseExpr:{type:"var", name:"c"}}"#).unwrap();
        let cond = Expr::Id(idtn("a"));
        let if_expr = Expr::Id(idtn("b"));
        let else_expr = Some(Expr::Id(idtn("c")));
        let expected = If::new(cond, if_expr, else_expr);
        let iff: If = val.try_into().unwrap();
        assert_eq!(iff, expected);
    }

    #[test]
    fn val_to_try_catch_no_exc_name_works() {
        let val = &ap_parse(r#"{type:"try", tryExpr:{type:"var", name:"a"}, catchExpr:{type:"var", name:"b"}}"#).unwrap();
        let try_expr = Expr::Id(idtn("a"));
        let catch_expr = Expr::Id(idtn("b"));
        let exc_name = None;
        let expected = TryCatch::new(try_expr, exc_name, catch_expr);
        let try_catch: TryCatch = val.try_into().unwrap();
        assert_eq!(try_catch, expected);
    }

    #[test]
    fn val_to_try_catch_with_exc_name_works() {
        let val = &ap_parse(r#"{type:"try", tryExpr:{type:"var", name:"a"}, errVarName:"ex", catchExpr:{type:"var", name:"b"}}"#).unwrap();
        let try_expr = Expr::Id(idtn("a"));
        let catch_expr = Expr::Id(idtn("b"));
        let exc_name = Some("ex".to_owned());
        let expected = TryCatch::new(try_expr, exc_name, catch_expr);
        let try_catch: TryCatch = val.try_into().unwrap();
        assert_eq!(try_catch, expected);
    }

    #[test]
    fn val_to_neg_works() {
        let val = &ap_parse(r#"{type:"neg", operand:{type:"var", name:"a"}}"#)
            .unwrap();
        let operand = Expr::Id(idtn("a"));
        let expected = Neg::new(operand);
        let neg: Neg = val.try_into().unwrap();
        assert_eq!(neg, expected);
    }

    #[test]
    fn misc_func_works() {
        let val =
            &ap_parse(include_str!("../test_input/misc_func.txt")).unwrap();
        let func: Func = val.try_into().unwrap();
        println!("{:#?}", func);
    }

    #[test]
    fn validate_works() {
        let val =
            &ap_parse(include_str!("../test_input/validate.txt")).unwrap();
        let func: Func = val.try_into().unwrap();
        println!("{:#?}", func);
    }

    #[test]
    fn partial_call_works() {
        let val =
            &ap_parse(include_str!("../test_input/partial_call.txt")).unwrap();
        let func: Func = val.try_into().unwrap();
        println!("{:#?}", func);
    }

    #[test]
    fn trap_a_func_works() {
        let val =
            &ap_parse(include_str!("../test_input/trap_a_func.txt")).unwrap();
        let func: Func = val.try_into().unwrap();
        println!("{:#?}", func);
    }
}

#[cfg(test)]
mod format_tests {
    use super::*;
    use axon_parseast_parser::parse as ap_parse;

    const INDENT: &str = "    ";

    fn idtn(s: &str) -> Id {
        Id::new(tn(s))
    }

    fn tn(s: &str) -> TagName {
        TagName::new(s.to_owned()).expect("s is not a valid tagName")
    }

    fn zero_ind() -> Indent {
        Indent::new(INDENT.to_owned(), 0)
    }

    fn lit_str_expr(s: &str) -> Expr {
        Expr::Lit(lit_str(s))
    }

    fn lit_str(s: &str) -> Lit {
        Lit::new(LitInner::Str(s.to_owned()))
    }

    fn lit_num_expr(n: f64) -> Expr {
        Expr::Lit(lit_num(n))
    }

    fn lit_num(n: f64) -> Lit {
        Lit::new(LitInner::Num(Number::new(n, None)))
    }

    fn stringify(lines: &Lines) -> Vec<String> {
        lines.iter().map(|ln| format!("{}", ln)).collect()
    }

    #[test]
    fn year_month_works() {
        let ym = YearMonth::new(2020, Month::Jan);
        assert_eq!(ym.to_axon_code(), "2020-01");
        let ym = YearMonth::new(2020, Month::Dec);
        assert_eq!(ym.to_axon_code(), "2020-12");
    }

    #[test]
    fn empty_list_works() {
        let list = List::new(vec![]);
        let lines = stringify(&list.to_lines(&zero_ind()));
        assert_eq!(lines[0], "[]");
    }

    #[test]
    fn list_works() {
        let item1 = lit_num_expr(1.0);
        let item2 = lit_num_expr(2.0);
        let item3 = lit_num_expr(3.0);
        let list = List::new(vec![item1, item2, item3]);
        let lines = stringify(&list.to_lines(&zero_ind()));
        assert_eq!(lines[0], "[");
        assert_eq!(lines[1], "    1,");
        assert_eq!(lines[2], "    2,");
        assert_eq!(lines[3], "    3,");
        assert_eq!(lines[4], "]");
    }

    #[test]
    fn empty_dict_works() {
        let dict = Dict::new(HashMap::new());
        let lines = stringify(&dict.to_lines(&zero_ind()));
        assert_eq!(lines[0], "{}");
    }

    #[test]
    fn dict_works() {
        let item1 = DictVal::Expr(lit_num_expr(1.0));
        let item2 = DictVal::Marker;
        let item3 = DictVal::RemoveMarker;
        let mut hash_map = HashMap::new();
        hash_map.insert(tn("a"), item1);
        hash_map.insert(tn("b"), item2);
        hash_map.insert(tn("c"), item3);
        let dict = Dict::new(hash_map);
        let lines = stringify(&dict.to_lines(&zero_ind()));
        assert_eq!(lines[0], "{");
        assert!(lines.contains(&"    a: 1,".to_owned()));
        assert!(lines.contains(&"    b: marker(),".to_owned()));
        assert!(lines.contains(&"    c: removeMarker(),".to_owned()));
        assert_eq!(lines[4], "}");
        assert_eq!(lines.len(), 5);
    }

    #[test]
    fn multi_line_dict_entries_get_sorted_works() {
        let item1 = DictVal::Expr(lit_num_expr(1.0));
        let item2 = DictVal::Marker;
        let item3 = DictVal::RemoveMarker;
        let mut hash_map = HashMap::new();
        hash_map.insert(tn("c"), item1);
        hash_map.insert(tn("b"), item2);
        hash_map.insert(tn("a"), item3);
        let dict = Dict::new(hash_map);
        let lines = stringify(&dict.to_lines(&zero_ind()));
        assert_eq!(lines[0], "{");
        assert!(lines.contains(&"    a: removeMarker(),".to_owned()));
        assert!(lines.contains(&"    b: marker(),".to_owned()));
        assert!(lines.contains(&"    c: 1,".to_owned()));
        assert_eq!(lines[4], "}");
        assert_eq!(lines.len(), 5);
    }

    #[test]
    fn single_line_dict_entries_get_sorted_works() {
        let item1 = DictVal::Expr(lit_num_expr(1.0));
        let item2 = DictVal::Marker;
        let item3 = DictVal::RemoveMarker;
        let mut hash_map = HashMap::new();
        hash_map.insert(tn("c"), item1);
        hash_map.insert(tn("b"), item2);
        hash_map.insert(tn("a"), item3);
        let dict = Dict::new(hash_map);
        let line = format!("{}", &dict.to_line(&zero_ind()));
        assert_eq!(line, "{a: removeMarker(), b: marker(), c: 1}");
    }

    #[test]
    fn single_line_def_works() {
        let expr = lit_num_expr(1.0);
        let def = Def::new(tn("varName"), expr);
        let lines = stringify(&def.to_lines(&zero_ind()));
        assert_eq!(lines[0], "varName: 1");
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn single_line_assign_works() {
        let expr = lit_num_expr(1.0);
        let assign = Assign::new(tn("varName"), expr);
        let lines = stringify(&assign.to_lines(&zero_ind()));
        assert_eq!(lines[0], "varName = 1");
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn multi_line_assign_works() {
        let expr = Expr::Block(Block::new(vec![lit_num_expr(1.0)]));
        let assign = Assign::new(tn("varName"), expr);
        let lines = stringify(&assign.to_lines(&zero_ind()));
        assert_eq!(lines[0], "varName = do");
        assert_eq!(lines[1], "    1");
        assert_eq!(lines[2], "end");
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn multi_line_def_works() {
        let expr = Expr::Block(Block::new(vec![lit_num_expr(1.0)]));
        let def = Def::new(tn("varName"), expr);
        let lines = stringify(&def.to_lines(&zero_ind()));
        assert_eq!(lines[0], "varName: do");
        assert_eq!(lines[1], "    1");
        assert_eq!(lines[2], "end");
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn single_line_trap_call_works() {
        let expr = lit_num_expr(1.0);
        let trap = TrapCall::new(expr, "varName".to_owned());
        let lines = stringify(&trap.to_lines(&zero_ind()));
        assert_eq!(lines[0], "1->varName");
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn multi_line_trap_call_works() {
        let item1 = DictVal::Expr(lit_num_expr(1.0));
        let mut hash_map = HashMap::new();
        hash_map.insert(tn("a"), item1);
        let dict = Dict::new(hash_map);

        let trap = TrapCall::new(Expr::Dict(dict), "varName".to_owned());
        let lines = stringify(&trap.to_lines(&zero_ind()));
        assert_eq!(lines[0], "{");
        assert_eq!(lines[1], "    a: 1,");
        assert_eq!(lines[2], "}->varName");
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn func_works() {
        let val = &ap_parse(r#"{type:"func", params:[{name:"point"}, {name:"dates"}, {name:"minuteInterval", def:{type:"literal", val:15}}], body:{type:"block", exprs:[{type:"assign", lhs:{type:"var", name:"dates"}, rhs:{type:"sub", lhs:{type:"var", name:"dates"}, rhs:{type:"literal", val:2day}}}, {type:"def", name:"startt", val:{type:"dotCall", target:{type:"var", name:"start"}, args:[{type:"dotCall", target:{type:"var", name:"toDateSpan"}, args:[{type:"var", name:"dates"}]}]}}, {type:"def", name:"endt", val:{type:"dotCall", target:{type:"var", name:"end"}, args:[{type:"dotCall", target:{type:"var", name:"toDateSpan"}, args:[{type:"var", name:"dates"}]}]}}, {type:"def", name:"intCount", val:{type:"if", cond:{type:"gt", lhs:{type:"sub", lhs:{type:"var", name:"endt"}, rhs:{type:"var", name:"startt"}}, rhs:{type:"literal", val:0}}, ifExpr:{type:"div", lhs:{type:"mul", lhs:{type:"dotCall", target:{type:"var", name:"to"}, args:[{type:"dotCall", target:{type:"var", name:"to"}, args:[{type:"sub", lhs:{type:"var", name:"endt"}, rhs:{type:"var", name:"startt"}}, {type:"literal", val:1day}]}, {type:"literal", val:1}]}, rhs:{type:"literal", val:1440}}, rhs:{type:"var", name:"minuteInterval"}}, elseExpr:{type:"div", lhs:{type:"literal", val:1440}, rhs:{type:"var", name:"minuteInterval"}}}}, {type:"assign", lhs:{type:"var", name:"intCount"}, rhs:{type:"sub", lhs:{type:"var", name:"intCount"}, rhs:{type:"literal", val:1}}}, {type:"def", name:"trut", val:{type:"list", vals:[]}}, {type:"dotCall", target:{type:"var", name:"each"}, args:[{type:"range", start:{type:"literal", val:0}, end:{type:"var", name:"intCount"}}, {type:"func", params:[{name:"x"}], body:{type:"block", exprs:[{type:"assign", lhs:{type:"var", name:"trut"}, rhs:{type:"dotCall", target:{type:"var", name:"add"}, args:[{type:"var", name:"trut"}, {type:"dict", names:["ts", "v0"], vals:[{type:"add", lhs:{type:"call", target:{type:"var", name:"dateTime"}, args:[{type:"var", name:"startt"}, {type:"literal", val:00:00:00}, {type:"trapCall", target:{type:"var", name:"trap"}, args:[{type:"var", name:"point"}, {type:"literal", val:"tz"}]}]}, rhs:{type:"dotCall", target:{type:"var", name:"to"}, args:[{type:"mul", lhs:{type:"var", name:"x"}, rhs:{type:"var", name:"minuteInterval"}}, {type:"literal", val:1min}]}}, {type:"neg", operand:{type:"literal", val:2}}]}]}}]}}]}, {type:"assign", lhs:{type:"var", name:"trut"}, rhs:{type:"dotCall", target:{type:"var", name:"addMeta"}, args:[{type:"dotCall", target:{type:"var", name:"toGrid"}, args:[{type:"var", name:"trut"}]}, {type:"dict", names:["hisStart", "hisEnd"], vals:[{type:"call", target:{type:"var", name:"dateTime"}, args:[{type:"var", name:"startt"}, {type:"literal", val:00:00:00}, {type:"trapCall", target:{type:"var", name:"trap"}, args:[{type:"var", name:"point"}, {type:"literal", val:"tz"}]}]}, {type:"call", target:{type:"var", name:"dateTime"}, args:[{type:"var", name:"endt"}, {type:"literal", val:23:59:59}, {type:"trapCall", target:{type:"var", name:"trap"}, args:[{type:"var", name:"point"}, {type:"literal", val:"tz"}]}]}]}]}}, {type:"def", name:"datae", val:{type:"dotCall", target:{type:"var", name:"hisRollup"}, args:[{type:"dotCall", target:{type:"var", name:"hisRead"}, args:[{type:"var", name:"point"}, {type:"var", name:"dates"}, {type:"dict", names:["limit"], vals:[{type:"literal", val:null}]}]}, {type:"var", name:"sum"}, {type:"literal", val:15min}]}}, {type:"if", cond:{type:"eq", lhs:{type:"var", name:"datae"}, rhs:{type:"literal", val:null}}, ifExpr:{type:"return", expr:{type:"dotCall", target:{type:"var", name:"hisFindPeriods"}, args:[{type:"var", name:"trut"}, {type:"func", params:[{name:"x"}], body:{type:"eq", lhs:{type:"var", name:"x"}, rhs:{type:"neg", operand:{type:"literal", val:2}}}}]}}}, {type:"if", cond:{type:"eq", lhs:{type:"dotCall", target:{type:"var", name:"size"}, args:[{type:"var", name:"datae"}]}, rhs:{type:"literal", val:0}}, ifExpr:{type:"return", expr:{type:"dotCall", target:{type:"var", name:"hisFindPeriods"}, args:[{type:"var", name:"trut"}, {type:"func", params:[{name:"x"}], body:{type:"eq", lhs:{type:"var", name:"x"}, rhs:{type:"neg", operand:{type:"literal", val:2}}}}]}}}, {type:"def", name:"thereisNullPeriod", val:{type:"dotCall", target:{type:"var", name:"findAll"}, args:[{type:"dotCall", target:{type:"var", name:"hisRollup"}, args:[{type:"dotCall", target:{type:"var", name:"hisFindPeriods"}, args:[{type:"dotCall", target:{type:"var", name:"map"}, args:[{type:"call", target:{type:"var", name:"hisJoin"}, args:[{type:"list", vals:[{type:"var", name:"datae"}, {type:"var", name:"trut"}]}]}, {type:"func", params:[{name:"x"}], body:{type:"dict", names:["ts", "v0"], vals:[{type:"dotCall", target:{type:"var", name:"get"}, args:[{type:"var", name:"x"}, {type:"literal", val:"ts"}]}, {type:"if", cond:{type:"eq", lhs:{type:"dotCall", target:{type:"var", name:"size"}, args:[{type:"dotCall", target:{type:"var", name:"findAll"}, args:[{type:"dotCall", target:{type:"var", name:"vals"}, args:[{type:"var", name:"x"}]}, {type:"func", params:[{name:"y"}], body:{type:"ne", lhs:{type:"var", name:"y"}, rhs:{type:"literal", val:null}}}]}]}, rhs:{type:"literal", val:2}}, ifExpr:{type:"neg", operand:{type:"literal", val:1}}, elseExpr:{type:"dotCall", target:{type:"var", name:"get"}, args:[{type:"var", name:"x"}, {type:"literal", val:"v0"}]}}]}}]}, {type:"func", params:[{name:"x"}], body:{type:"eq", lhs:{type:"var", name:"x"}, rhs:{type:"neg", operand:{type:"literal", val:1}}}}]}, {type:"var", name:"sum"}, {type:"literal", val:1h}]}, {type:"func", params:[{name:"x"}], body:{type:"ge", lhs:{type:"dotCall", target:{type:"var", name:"get"}, args:[{type:"var", name:"x"}, {type:"literal", val:"v0"}]}, rhs:{type:"literal", val:1h}}}]}}, {type:"if", cond:{type:"eq", lhs:{type:"var", name:"thereisNullPeriod"}, rhs:{type:"literal", val:null}}, ifExpr:{type:"return", expr:{type:"literal", val:null}}}, {type:"if", cond:{type:"eq", lhs:{type:"dotCall", target:{type:"var", name:"size"}, args:[{type:"var", name:"thereisNullPeriod"}]}, rhs:{type:"literal", val:0}}, ifExpr:{type:"return", expr:{type:"literal", val:null}}, elseExpr:{type:"return", expr:{type:"call", target:{type:"var", name:"hisSlidingWindows"}, args:[{type:"var", name:"dates"}, {type:"literal", val:24h}, {type:"literal", val:24h}]}}}]}}"#).unwrap();
        let func: Func = val.try_into().unwrap();
        let strings = stringify(&func.to_lines(&zero_ind()));
        for s in strings {
            println!("{}", s);
        }
    }

    #[test]
    fn escaped_chars_work() {
        let val = &ap_parse(r#"{type:"literal", val:"Hello\nWorld\t!\$ \\"}"#)
            .unwrap();
        let expr: Expr = val.try_into().unwrap();

        let lines = stringify(&expr.to_lines(&zero_ind()));
        let expected = r#""Hello\nWorld\t!\$ \\""#;
        assert_eq!(lines[0], expected);
    }

    #[test]
    fn simple_precedence_left_works() {
        // (1 + 2) / 3
        let add_bin_op =
            BinOp::new(lit_num_expr(1.0), BinOpId::Add, lit_num_expr(2.0));
        let add = Box::new(Add(add_bin_op));
        let div_bin_op =
            BinOp::new(Expr::Add(add), BinOpId::Div, lit_num_expr(3.0));
        let div = Div(div_bin_op);
        let lines = stringify(&div.to_lines(&zero_ind()));
        assert_eq!(lines[0], "(1 + 2) / 3");
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn simple_precedence_right_works() {
        // (1 + 2) / 3
        let add_bin_op =
            BinOp::new(lit_num_expr(2.0), BinOpId::Add, lit_num_expr(3.0));
        let add = Box::new(Add(add_bin_op));
        let div_bin_op =
            BinOp::new(lit_num_expr(1.0), BinOpId::Div, Expr::Add(add));
        let div = Div(div_bin_op);
        let lines = stringify(&div.to_lines(&zero_ind()));
        assert_eq!(lines[0], "1 / (2 + 3)");
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn flat_if_no_nesting_with_no_else_works() {
        let cond1 = Expr::Id(idtn("a"));
        let expr1 = lit_str_expr("a-expr");
        let ce1 = ConditionalExpr::new(cond1, expr1);
        let flat_if = FlatIf::new(vec![ce1], None);
        let lines = stringify(&flat_if.to_lines(&zero_ind()));
        assert_eq!(lines[0], "if (a) do");
        assert_eq!(lines[1], "    \"a-expr\"");
        assert_eq!(lines[2], "end");
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn flat_if_no_nesting_with_else_works() {
        let cond1 = Expr::Id(idtn("a"));
        let expr1 = lit_str_expr("a-expr");
        let ce1 = ConditionalExpr::new(cond1, expr1);

        let else1 = lit_str_expr("else-expr");

        let flat_if = FlatIf::new(vec![ce1], Some(else1));
        let lines = stringify(&flat_if.to_lines(&zero_ind()));
        assert_eq!(lines[0], "if (a) do");
        assert_eq!(lines[1], "    \"a-expr\"");
        assert_eq!(lines[2], "end else do");
        assert_eq!(lines[3], "    \"else-expr\"");
        assert_eq!(lines[4], "end");
        assert_eq!(lines.len(), 5);
    }

    #[test]
    fn flat_if_some_nesting_with_no_else_works() {
        let cond1 = Expr::Id(idtn("a"));
        let expr1 = lit_str_expr("a-expr");
        let ce1 = ConditionalExpr::new(cond1, expr1);

        let cond2 = Expr::Id(idtn("b"));
        let expr2 = lit_str_expr("b-expr");
        let ce2 = ConditionalExpr::new(cond2, expr2);

        let else1 = lit_str_expr("else-expr");

        let flat_if = FlatIf::new(vec![ce1, ce2], Some(else1));
        let lines = stringify(&flat_if.to_lines(&zero_ind()));

        assert_eq!(lines[0], "if (a) do");
        assert_eq!(lines[1], "    \"a-expr\"");
        assert_eq!(lines[2], "end else if (b) do");
        assert_eq!(lines[3], "    \"b-expr\"");
        assert_eq!(lines[4], "end else do");
        assert_eq!(lines[5], "    \"else-expr\"");
        assert_eq!(lines[6], "end");
        assert_eq!(lines.len(), 7);
    }

    #[test]
    fn simple_try_catch_no_exc_name_works() {
        let try_expr = lit_str_expr("try-expr");
        let catch_expr = lit_str_expr("catch-expr");
        let tc = TryCatch::new(try_expr, None, catch_expr);

        let lines = stringify(&tc.to_lines(&zero_ind()));

        assert_eq!(lines[0], "try do");
        assert_eq!(lines[1], "    \"try-expr\"");
        assert_eq!(lines[2], "end catch do");
        assert_eq!(lines[3], "    \"catch-expr\"");
        assert_eq!(lines[4], "end");
        assert_eq!(lines.len(), 5);
    }

    #[test]
    fn simple_try_catch_with_exc_name_works() {
        let try_expr = lit_str_expr("try-expr");
        let catch_expr = lit_str_expr("catch-expr");
        let tc = TryCatch::new(try_expr, Some("ex".to_owned()), catch_expr);

        let lines = stringify(&tc.to_lines(&zero_ind()));

        assert_eq!(lines[0], "try do");
        assert_eq!(lines[1], "    \"try-expr\"");
        assert_eq!(lines[2], "end catch (ex) do");
        assert_eq!(lines[3], "    \"catch-expr\"");
        assert_eq!(lines[4], "end");
        assert_eq!(lines.len(), 5);
    }

    #[test]
    fn partial_call_works() {
        let func_name = FuncName::TagName(tn("utilsAssert"));
        let null = PartialCallArgument::Placeholder;
        let arg2 = PartialCallArgument::Expr(Expr::Lit(lit_num(1.0)));
        let pc = PartialCall::new(func_name, vec![null, arg2]);

        let lines = stringify(&pc.to_lines(&zero_ind()));

        assert_eq!(lines[0], "utilsAssert(_, 1)");
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn get_single_line_works() {
        let axon = r#"{type:"dotCall", target:{type:"var", name:"get"}, args:[{type:"dict", names:["x"], vals:[{type:"literal", val:0}]}, {type:"literal", val:"x"}]}"#;
        let val = &axon_parseast_parser::parse(axon).unwrap();
        let dot_call: DotCall = val.try_into().unwrap();
        let line = dot_call.to_line(&zero_ind());
        let line = format!("{}", line);
        assert_eq!(line, "{x: 0}[\"x\"]");
    }

    #[test]
    fn get_multi_target_line_works() {
        let axon = r#"{type:"dotCall", target:{type:"var", name:"get"}, args:[{type:"dict", names:["x"], vals:[{type:"literal", val:0}]}, {type:"literal", val:"x"}]}"#;
        let val = &axon_parseast_parser::parse(axon).unwrap();
        let dot_call: DotCall = val.try_into().unwrap();
        let lines = stringify(&dot_call.to_lines(&zero_ind()));
        assert_eq!(lines[0], "{");
        assert_eq!(lines[1], "    x: 0,");
        assert_eq!(lines[2], "}[\"x\"]");
    }
}
