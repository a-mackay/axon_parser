use crate::ast::{
    Assign, Associativity, BinOp, Def, Expr, Id, List, Lit, Neg, Not, Range,
    Return, Throw,
};

/// The size of a single block of indentation, the number of spaces (' ').
const SPACES: usize = 4;

#[derive(Clone, Copy, Debug)]
struct Context {
    /// The number of spaces across this code should be.
    indent: usize,
    /// The maximum width allowed for this code.
    max_width: usize,
}

impl Context {
    fn new(indent: usize, max_width: usize) -> Self {
        Self { indent, max_width }
    }

    fn max_width(&self) -> usize {
        self.max_width
    }

    fn indent(&self) -> String {
        " ".repeat(self.indent)
    }

    fn str_within_max_width(&self, s: &str) -> bool {
        let max_len = s
            .lines()
            .map(|ln| ln.len())
            .reduce(std::cmp::max)
            .expect("s is a rewritten str, it shouldn't be empty so lines shouldn't be empty");
        max_len <= self.max_width
    }

    fn increase_indent(&self) -> Context {
        Context::new(self.indent + SPACES, self.max_width())
    }
}

impl Rewrite for List {
    fn rewrite(&self, context: Context) -> Option<String> {
        let exprs = &self.vals[..];
        let ind = context.indent();
        let new_code = if exprs.is_empty() {
            format!("{ind}[]", ind = ind)
        } else {
            // Try fit the entire list on one line:
            let one_line_context =
                Context::new(context.indent, context.max_width() - 2);
            let items: Option<Vec<String>> =
                exprs.iter().map(|e| e.rewrite(one_line_context)).collect();

            match items {
                Some(items) => {
                    let items = items
                        .iter()
                        .map(|item| item.trim())
                        .collect::<Vec<_>>();
                    let items_str = items.join(", ");
                    // There was enough space for each expression to be
                    // rewritten, so we see if it fits on one line:
                    let one_line =
                        format!("{ind}[{items}]", ind = ind, items = items_str);

                    if context.str_within_max_width(&one_line) {
                        one_line
                    } else {
                        self.default_rewrite(context)?
                    }
                }
                None => self.default_rewrite(context)?,
            }
        };

        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }
}

impl List {
    /// Write the list over multiple lines, each line containing one
    /// list element.
    fn default_rewrite(&self, context: Context) -> Option<String> {
        let exprs = &self.vals[..];
        let ind = context.indent();
        let items_context = context.increase_indent();
        let items: Option<Vec<String>> = exprs
            .iter()
            .map(|expr| {
                expr.rewrite(items_context)
                    .map(|string| format!("{},\n", string))
            })
            .collect();
        let items = items?;
        let items_str = items.join("");
        let new_code =
            format!("{ind}[\n{items}{ind}]", ind = ind, items = items_str);

        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }
}

impl Rewrite for Range {
    fn rewrite(&self, context: Context) -> Option<String> {
        let start_needs_parens = Self::needs_parens(&self.start);
        let end_needs_parens = Self::needs_parens(&self.end);
        let mut start = self.start.rewrite(context)?;
        let mut end = self.end.rewrite(context)?;
        if start_needs_parens {
            start = add_parens(&start);
        }
        if end_needs_parens {
            end = add_parens(&end);
        }
        let new_code = match (is_one_line(&start), is_one_line(&end)) {
            (true, true) => {
                let one_line = format!(
                    "{start}..{end}",
                    start = start,
                    end = end.trim_start()
                );
                if context.str_within_max_width(&one_line) {
                    one_line
                } else {
                    let ind = context.indent();
                    format!(
                        "{start}\n{ind}..\n{end}",
                        start = start,
                        ind = ind,
                        end = end
                    )
                }
            }
            _ => {
                let ind = context.indent();
                format!(
                    "{start}\n{ind}..\n{end}",
                    start = start,
                    ind = ind,
                    end = end
                )
            }
        };

        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }
}

impl Range {
    fn needs_parens(expr: &Expr) -> bool {
        match expr {
            Expr::Add(_) => true,
            Expr::And(_) => true,
            Expr::Cmp(_) => true,
            Expr::Div(_) => true,
            Expr::Eq(_) => true,
            Expr::Gt(_) => true,
            Expr::Gte(_) => true,
            Expr::Lt(_) => true,
            Expr::Lte(_) => true,
            Expr::Mul(_) => true,
            Expr::Ne(_) => true,
            Expr::Or(_) => true,
            Expr::Sub(_) => true,
            Expr::Assign(_) => true,
            Expr::Block(_) => true,
            Expr::Call(_) => false,
            Expr::Def(_) => true,
            Expr::Dict(_) => true,
            Expr::DotCall(_) => false,
            Expr::Func(_) => true,
            Expr::Id(_) => false,
            Expr::If(_) => true,
            Expr::List(_) => true,
            Expr::Lit(_) => false,
            Expr::Neg(_) => false,
            Expr::Not(_) => true,
            Expr::PartialCall(_) => true,
            Expr::Range(_) => true,
            Expr::Return(_) => true,
            Expr::Throw(_) => true,
            Expr::TrapCall(_) => true,
            Expr::TryCatch(_) => true,
        }
    }
}

impl Rewrite for Expr {
    fn rewrite(&self, context: Context) -> Option<String> {
        match self {
            Self::Add(x) => x.0.rewrite(context),
            Self::And(x) => x.0.rewrite(context),
            Self::Cmp(x) => x.0.rewrite(context),
            Self::Div(x) => x.0.rewrite(context),
            Self::Eq(x) => x.0.rewrite(context),
            Self::Gt(x) => x.0.rewrite(context),
            Self::Gte(x) => x.0.rewrite(context),
            Self::Lt(x) => x.0.rewrite(context),
            Self::Lte(x) => x.0.rewrite(context),
            Self::Mul(x) => x.0.rewrite(context),
            Self::Ne(x) => x.0.rewrite(context),
            Self::Or(x) => x.0.rewrite(context),
            Self::Sub(x) => x.0.rewrite(context),
            Self::Assign(x) => x.rewrite(context),
            // Self::Block(_) => true,
            // Self::Call(_) => false,
            Self::Def(x) => x.rewrite(context),
            // Self::Dict(_) => false,
            // Self::DotCall(_) => false,
            // Self::Func(_) => true,
            Self::Id(x) => x.rewrite(context),
            // Self::If(_) => true,
            Self::List(x) => x.rewrite(context),
            Self::Lit(x) => x.rewrite(context),
            Self::Neg(x) => x.rewrite(context),
            Self::Not(x) => x.rewrite(context),
            // Self::PartialCall(_) => false,
            Self::Range(x) => x.rewrite(context),
            Self::Return(x) => x.rewrite(context),
            Self::Throw(x) => x.rewrite(context),
            // Self::TrapCall(_) => true,
            // Self::TryCatch(_) => true,
            _ => todo!(),
        }
    }
}

impl Rewrite for Assign {
    fn rewrite(&self, context: Context) -> Option<String> {
        let expr = self.expr.rewrite(context)?;
        let prefix = format!("{} = ", self.name);
        let new_code = add_after_leading_indent(&prefix, &expr);
        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }
}

impl Rewrite for Def {
    fn rewrite(&self, context: Context) -> Option<String> {
        let expr = self.expr.rewrite(context)?;
        let prefix = format!("{}: ", self.name);
        let new_code = add_after_leading_indent(&prefix, &expr);
        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }
}

impl Rewrite for Return {
    fn rewrite(&self, context: Context) -> Option<String> {
        let expr = self.expr.rewrite(context)?;
        let new_code = add_after_leading_indent("return ", &expr);
        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }
}

impl Rewrite for Throw {
    fn rewrite(&self, context: Context) -> Option<String> {
        let expr = self.expr.rewrite(context)?;
        let new_code = add_after_leading_indent("throw ", &expr);
        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }
}

trait Rewrite {
    fn rewrite(&self, context: Context) -> Option<String>;
}

impl Rewrite for Neg {
    fn rewrite(&self, context: Context) -> Option<String> {
        let needs_parens = match &self.operand {
            Expr::Add(_) => true,
            Expr::And(_) => true,
            Expr::Cmp(_) => true,
            Expr::Div(_) => true,
            Expr::Eq(_) => true,
            Expr::Gt(_) => true,
            Expr::Gte(_) => true,
            Expr::Lt(_) => true,
            Expr::Lte(_) => true,
            Expr::Mul(_) => true,
            Expr::Ne(_) => true,
            Expr::Or(_) => true,
            Expr::Sub(_) => true,
            Expr::Assign(_) => true,
            Expr::Block(_) => true,
            Expr::Call(_) => false,
            Expr::Def(_) => true,
            Expr::Dict(_) => true,
            Expr::DotCall(_) => false,
            Expr::Func(_) => true,
            Expr::Id(_) => false,
            Expr::If(_) => true,
            Expr::List(_) => true,
            Expr::Lit(_) => false,
            Expr::Neg(_) => true,
            Expr::Not(_) => true,
            Expr::PartialCall(_) => true,
            Expr::Range(_) => true,
            Expr::Return(_) => true,
            Expr::Throw(_) => true,
            Expr::TrapCall(_) => false,
            Expr::TryCatch(_) => true,
        };
        let mut op = self.operand.rewrite(context)?;
        if needs_parens {
            op = add_parens(&op);
        }
        let new_code = add_after_leading_indent("-", &op);
        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }
}

impl Rewrite for Not {
    fn rewrite(&self, context: Context) -> Option<String> {
        let needs_parens = match &self.operand {
            Expr::Add(_) => true,
            Expr::And(_) => true,
            Expr::Cmp(_) => true,
            Expr::Div(_) => true,
            Expr::Eq(_) => true,
            Expr::Gt(_) => true,
            Expr::Gte(_) => true,
            Expr::Lt(_) => true,
            Expr::Lte(_) => true,
            Expr::Mul(_) => true,
            Expr::Ne(_) => true,
            Expr::Or(_) => true,
            Expr::Sub(_) => true,
            Expr::Assign(_) => true,
            Expr::Block(_) => true,
            Expr::Call(_) => false,
            Expr::Def(_) => true,
            Expr::Dict(_) => true,
            Expr::DotCall(_) => false,
            Expr::Func(_) => true,
            Expr::Id(_) => false,
            Expr::If(_) => true,
            Expr::List(_) => true,
            Expr::Lit(_) => false,
            Expr::Neg(_) => true,
            Expr::Not(_) => true,
            Expr::PartialCall(_) => true,
            Expr::Range(_) => true,
            Expr::Return(_) => true,
            Expr::Throw(_) => true,
            Expr::TrapCall(_) => false,
            Expr::TryCatch(_) => true,
        };
        let mut op = self.operand.rewrite(context)?;
        if needs_parens {
            op = add_parens(&op);
        }
        let new_code = add_after_leading_indent("not ", &op);
        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }
}

/// Add parentheses surrounding the rewritten code, properly
/// accounting for possible indentation at the start.
/// We assume the rewritten code has no trailing whitespace.
fn add_parens(code: &str) -> String {
    let ind_count = code.chars().take_while(|&c| c == ' ').count();
    let ind = " ".repeat(ind_count);
    format!("{ind}({code})", ind = ind, code = code.trim())
}

/// Add s to the start of the rewritten code, properly
/// accounting for possible indentation at the start.
fn add_after_leading_indent(s: &str, code: &str) -> String {
    let ind_count = code.chars().take_while(|&c| c == ' ').count();
    let ind = " ".repeat(ind_count);
    format!("{ind}{s}{code}", ind = ind, s = s, code = code.trim_start())
}

fn is_one_line(s: &str) -> bool {
    s.lines().count() == 1
}

impl Rewrite for BinOp {
    fn rewrite(&self, context: Context) -> Option<String> {
        let left_needs_parens = self.needs_parens(true);
        let right_needs_parens = self.needs_parens(false);

        let mut ls = self.lhs.rewrite(context)?;
        if left_needs_parens {
            ls = add_parens(&ls);
        }
        let mut rs = self.rhs.rewrite(context)?;
        if right_needs_parens {
            rs = add_parens(&rs);
        }
        let ind = context.indent();
        let op = self.bin_op_id.to_symbol();

        let new_code = match (is_one_line(&ls), is_one_line(&rs)) {
            (true, true) => {
                let ls = ls.trim();
                let rs = rs.trim();

                let line = format!(
                    "{ind}{lhs} {op} {rhs}",
                    ind = ind,
                    lhs = ls,
                    op = op,
                    rhs = rs
                );
                if context.str_within_max_width(&line) {
                    line
                } else {
                    // Single line was too wide:
                    format!(
                        "{ind}{lhs}\n{ind}{op} {rhs}",
                        ind = ind,
                        lhs = ls,
                        op = op,
                        rhs = rs
                    )
                }
            }
            _ => {
                format!(
                    "{lhs}\n{ind}{op} {rhs}",
                    ind = ind,
                    lhs = ls,
                    op = op,
                    rhs = rs.trim_start()
                )
            }
        };

        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }
}

impl BinOp {
    fn needs_parens(&self, is_left: bool) -> bool {
        let prec = self.precedence();
        let assoc = self.associativity();

        let child = if is_left { &self.lhs } else { &self.rhs };

        match child {
            Expr::Add(x) => {
                needs_parens(prec, assoc, x.0.precedence(), is_left)
            }
            Expr::And(x) => {
                needs_parens(prec, assoc, x.0.precedence(), is_left)
            }
            Expr::Cmp(x) => {
                needs_parens(prec, assoc, x.0.precedence(), is_left)
            }
            Expr::Div(x) => {
                needs_parens(prec, assoc, x.0.precedence(), is_left)
            }
            Expr::Eq(x) => needs_parens(prec, assoc, x.0.precedence(), is_left),
            Expr::Gt(x) => needs_parens(prec, assoc, x.0.precedence(), is_left),
            Expr::Gte(x) => {
                needs_parens(prec, assoc, x.0.precedence(), is_left)
            }
            Expr::Lt(x) => needs_parens(prec, assoc, x.0.precedence(), is_left),
            Expr::Lte(x) => {
                needs_parens(prec, assoc, x.0.precedence(), is_left)
            }
            Expr::Mul(x) => {
                needs_parens(prec, assoc, x.0.precedence(), is_left)
            }
            Expr::Ne(x) => needs_parens(prec, assoc, x.0.precedence(), is_left),
            Expr::Or(x) => needs_parens(prec, assoc, x.0.precedence(), is_left),
            Expr::Sub(x) => {
                needs_parens(prec, assoc, x.0.precedence(), is_left)
            }
            Expr::Assign(_) => true,
            Expr::Block(_) => true,
            Expr::Call(_) => false,
            Expr::Def(_) => true,
            Expr::Dict(_) => false,
            Expr::DotCall(_) => false,
            Expr::Func(_) => true,
            Expr::Id(_) => false,
            Expr::If(_) => true,
            Expr::List(_) => false,
            Expr::Lit(_) => false,
            Expr::Neg(_) => false,
            Expr::Not(_) => false,
            Expr::PartialCall(_) => false,
            Expr::Range(_) => true,
            Expr::Return(_) => true,
            Expr::Throw(_) => true,
            Expr::TrapCall(_) => true,
            Expr::TryCatch(_) => true,
        }
    }
}

impl Rewrite for Lit {
    fn rewrite(&self, context: Context) -> Option<String> {
        let code = self.to_axon_code();
        let new_code =
            format!("{indent}{code}", indent = context.indent(), code = code);
        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }
}

impl Rewrite for Id {
    fn rewrite(&self, context: Context) -> Option<String> {
        let code = self.to_axon_code();
        let new_code =
            format!("{indent}{code}", indent = context.indent(), code = code);
        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }
}

fn needs_parens(
    parent_precedence: usize,
    parent_assoc: Option<Associativity>,
    child_precedence: usize,
    child_is_left: bool,
) -> bool {
    if child_precedence < parent_precedence {
        false
    } else if child_precedence > parent_precedence || parent_assoc.is_none() {
        true
    } else {
        let parent_assoc = parent_assoc.unwrap();
        let both_left = parent_assoc == Associativity::Left && child_is_left;
        let both_right = parent_assoc == Associativity::Right && !child_is_left;
        let same_side = both_left || both_right;
        !same_side
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        Add, And, Assign, BinOp, BinOpId, Def, Expr, Id, List, Lit, LitInner,
        Mul, Neg, Not, Return, Sub, Throw,
    };
    use raystack_core::{Number, TagName};

    fn c() -> Context {
        let large_width = 800;
        Context::new(0, large_width)
    }

    // new context
    fn nc(indent: usize, max_width: usize) -> Context {
        Context::new(indent, max_width)
    }

    fn lit_bool(b: bool) -> Lit {
        let inner = LitInner::Bool(b);
        Lit::new(inner)
    }

    fn ex_lit_bool(b: bool) -> Expr {
        Expr::Lit(lit_bool(b))
    }

    fn lit_num(n: usize) -> Lit {
        let num = Number::new(n as f64, None);
        let inner = LitInner::Num(num);
        Lit::new(inner)
    }

    fn ex_lit_num(n: usize) -> Expr {
        Expr::Lit(lit_num(n))
    }

    fn tn(s: &str) -> TagName {
        TagName::new(s.to_owned()).unwrap()
    }

    fn ex_id(s: &str) -> Expr {
        Expr::Id(Id::new(tn(s)))
    }

    #[test]
    fn bin_op_precedence_works_1() {
        let mul = BinOp::new(ex_lit_num(2), BinOpId::Mul, ex_lit_num(3));
        let mul = Expr::Mul(Box::new(Mul(mul)));
        let add = BinOp::new(ex_lit_num(1), BinOpId::Add, mul);
        let add = Expr::Add(Box::new(Add(add)));

        let code = add.rewrite(c()).unwrap();
        assert_eq!(code, "1 + 2 * 3")
    }

    #[test]
    fn bin_op_precedence_works_2() {
        let add = BinOp::new(ex_lit_num(1), BinOpId::Add, ex_lit_num(2));
        let add = Expr::Add(Box::new(Add(add)));
        let mul = BinOp::new(add, BinOpId::Mul, ex_lit_num(3));
        let mul = Expr::Mul(Box::new(Mul(mul)));

        let code = mul.rewrite(c()).unwrap();
        assert_eq!(code, "(1 + 2) * 3")
    }

    #[test]
    fn bin_op_precedence_works_3() {
        let add = BinOp::new(ex_lit_num(1), BinOpId::Add, ex_lit_num(2));
        let add = Expr::Add(Box::new(Add(add)));
        let mul = BinOp::new(ex_lit_num(3), BinOpId::Mul, add);
        let mul = Expr::Mul(Box::new(Mul(mul)));

        let code = mul.rewrite(c()).unwrap();
        assert_eq!(code, "3 * (1 + 2)")
    }

    #[test]
    fn bin_op_equal_precedence_works_1() {
        let sub = BinOp::new(ex_lit_num(1), BinOpId::Sub, ex_lit_num(2));
        let sub = Expr::Sub(Box::new(Sub(sub)));
        let add = BinOp::new(sub, BinOpId::Add, ex_lit_num(3));
        let add = Expr::Add(Box::new(Add(add)));

        let code = add.rewrite(c()).unwrap();
        assert_eq!(code, "1 - 2 + 3")
    }

    #[test]
    fn bin_op_equal_precedence_works_2() {
        let add = BinOp::new(ex_lit_num(2), BinOpId::Add, ex_lit_num(3));
        let add = Expr::Add(Box::new(Add(add)));
        let sub = BinOp::new(ex_lit_num(1), BinOpId::Sub, add);
        let sub = Expr::Sub(Box::new(Sub(sub)));

        let code = sub.rewrite(c()).unwrap();
        assert_eq!(code, "1 - (2 + 3)")
    }

    #[test]
    fn bin_op_equal_precedence_works_3() {
        let and_r = BinOp::new(ex_id("b"), BinOpId::And, ex_id("c"));
        let and_r = Expr::And(Box::new(And(and_r)));
        let and_l = BinOp::new(ex_id("a"), BinOpId::And, and_r);
        let and_l = Expr::And(Box::new(And(and_l)));

        let code = and_l.rewrite(c()).unwrap();
        assert_eq!(code, "a and b and c")
    }

    #[test]
    fn bin_op_equal_precedence_works_4() {
        let and_l = BinOp::new(ex_id("a"), BinOpId::And, ex_id("b"));
        let and_l = Expr::And(Box::new(And(and_l)));
        let and_r = BinOp::new(and_l, BinOpId::And, ex_id("c"));
        let and_r = Expr::And(Box::new(And(and_r)));

        let code = and_r.rewrite(c()).unwrap();
        assert_eq!(code, "(a and b) and c")
    }

    #[test]
    fn bin_op_single_line_too_wide_works() {
        let add = BinOp::new(ex_lit_num(10), BinOpId::Add, ex_lit_num(2));
        let add = Expr::Add(Box::new(Add(add)));

        let code = add.rewrite(nc(4, 7)).unwrap();
        assert_eq!(code, "    10\n    + 2")
    }

    #[test]
    fn bin_op_single_line_with_precedence_too_wide_works_1() {
        let add = BinOp::new(ex_lit_num(1), BinOpId::Add, ex_lit_num(2));
        let add = Expr::Add(Box::new(Add(add)));
        let mul = BinOp::new(add, BinOpId::Mul, ex_lit_num(3));
        let mul = Expr::Mul(Box::new(Mul(mul)));

        let code = mul.rewrite(nc(4, 11)).unwrap();
        assert_eq!(code, "    (1 + 2)\n    * 3")
    }

    #[test]
    fn bin_op_single_line_with_precedence_too_wide_works_2() {
        let add = BinOp::new(ex_lit_num(1), BinOpId::Add, ex_lit_num(2));
        let add = Expr::Add(Box::new(Add(add)));
        let mul = BinOp::new(add, BinOpId::Mul, ex_lit_num(3));
        let mul = Expr::Mul(Box::new(Mul(mul)));

        // 10 max width isn't wide enough
        assert!(mul.rewrite(nc(4, 10)).is_none());
    }

    #[test]
    fn neg_single_line_works() {
        let neg = Neg::new(ex_lit_num(1));
        let code = neg.rewrite(c()).unwrap();
        assert_eq!(code, "-1");
    }

    #[test]
    fn neg_single_line_parens_works() {
        let neg = Neg::new(ex_lit_num(1));
        let neg = Neg::new(Expr::Neg(Box::new(neg)));
        let code = neg.rewrite(c()).unwrap();
        assert_eq!(code, "-(-1)");
    }

    #[test]
    fn neg_multi_line_works() {
        todo!()
    }

    #[test]
    fn neg_multi_line_parens_works() {
        todo!()
    }

    #[test]
    fn not_single_line_works() {
        let not = Not::new(ex_lit_bool(true));
        let code = not.rewrite(c()).unwrap();
        assert_eq!(code, "not true");
    }

    #[test]
    fn not_single_line_parens_works() {
        let not = Not::new(ex_lit_bool(true));
        let not = Not::new(Expr::Not(Box::new(not)));
        let code = not.rewrite(c()).unwrap();
        assert_eq!(code, "not (not true)");
    }

    #[test]
    fn not_multi_line_works() {
        todo!()
    }

    #[test]
    fn not_multi_line_parens_works() {
        todo!()
    }

    #[test]
    fn assign_single_line_works() {
        let ass = Assign::new(tn("a"), ex_lit_num(0));
        let ass = Expr::Assign(ass);
        let code = ass.rewrite(c()).unwrap();
        assert_eq!(code, "a = 0");
    }

    #[test]
    fn assign_multi_line_works() {
        todo!()
    }

    #[test]
    fn def_single_line_works() {
        let def = Def::new(tn("a"), ex_lit_num(0));
        let def = Expr::Def(def);
        let code = def.rewrite(c()).unwrap();
        assert_eq!(code, "a: 0");
    }

    #[test]
    fn def_multi_line_works() {
        todo!()
    }

    #[test]
    fn return_single_line_works() {
        let ret = Return::new(ex_lit_num(0));
        let ret = Expr::Return(Box::new(ret));
        let code = ret.rewrite(c()).unwrap();
        assert_eq!(code, "return 0");
    }

    #[test]
    fn return_multi_line_works() {
        todo!()
    }

    #[test]
    fn throw_single_line_works() {
        let thr = Throw::new(ex_lit_num(0));
        let thr = Expr::Throw(Box::new(thr));
        let code = thr.rewrite(c()).unwrap();
        assert_eq!(code, "throw 0");
    }

    #[test]
    fn throw_multi_line_works() {
        todo!()
    }

    #[test]
    fn range_single_line_works() {
        let start = ex_lit_num(100);
        let end = ex_lit_num(200);
        let range = Range::new(start, end);
        let code = range.rewrite(c()).unwrap();
        assert_eq!(code, "100..200")
    }

    #[test]
    fn range_multi_line_from_single_lines_works() {
        let start = ex_lit_num(100);
        let end = ex_lit_num(200);
        let range = Range::new(start, end);
        let code = range.rewrite(nc(4, 7)).unwrap();
        assert_eq!(code, "    100\n    ..\n    200")
    }

    #[test]
    fn range_multi_line_works() {
        todo!()
    }

    #[test]
    fn list_empty_works() {
        let list = List::new(vec![]);
        let code = list.rewrite(c()).unwrap();
        assert_eq!(code, "[]");
    }

    #[test]
    fn list_one_line_works() {
        let list = List::new(vec![ex_lit_num(1)]);
        let code = list.rewrite(c()).unwrap();
        assert_eq!(code, "[1]");

        let list = List::new(vec![ex_lit_num(1), ex_lit_num(2)]);
        let code = list.rewrite(c()).unwrap();
        assert_eq!(code, "[1, 2]");
    }

    #[test]
    fn list_multi_line_works() {
        let list =
            List::new(vec![ex_lit_num(100), ex_lit_num(200), ex_lit_num(300)]);
        let code = list.rewrite(nc(4, 14)).unwrap();
        assert_eq!(
            code,
            "    [\n        100,\n        200,\n        300,\n    ]"
        );
    }
}
