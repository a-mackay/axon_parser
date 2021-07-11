use crate::ast::{Associativity, BinOp, Expr, Lit};

/// The size of a single block of indentation, the number of spaces (' ').
const SPACES: usize = 4;
/// The maximum possible width. This value is arbitrary.
const MAX_WIDTH: usize = 800;

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
        s.len() <= self.max_width
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
            // Self::Assign(_) => true,
            // Self::Block(_) => true,
            // Self::Call(_) => false,
            // Self::Def(_) => true,
            // Self::Dict(_) => false,
            // Self::DotCall(_) => false,
            // Self::Func(_) => true,
            // Self::Id(_) => false,
            // Self::If(_) => true,
            // Self::List(_) => false,
            Self::Lit(x) => x.rewrite(context),
            // Self::Neg(_) => false,
            // Self::Not(_) => false,
            // Self::PartialCall(_) => false,
            // Self::Range(_) => true,
            // Self::Return(_) => true,
            // Self::Throw(_) => true,
            // Self::TrapCall(_) => true,
            // Self::TryCatch(_) => true,
            _ => todo!(),
        }
    }
}

trait Rewrite {
    fn rewrite(&self, context: Context) -> Option<String>;
}

impl Rewrite for BinOp {
    fn rewrite(&self, context: Context) -> Option<String> {
        // TODO: this only rewrites to a single line,
        // when it could rewrite to multiple lines.

        let left_needs_parens = self.needs_parens(true);
        let right_needs_parens = self.needs_parens(false);

        let child_context = Context::new(0, MAX_WIDTH);
        let mut ls = self
            .lhs
            .rewrite(child_context)
            .expect("BinOp LHS exceeds MAX_WIDTH");
        if left_needs_parens {
            ls = format!("({})", ls);
        }
        let mut rs = self
            .rhs
            .rewrite(child_context)
            .expect("BinOp RHS exceeds MAX_WIDTH");
        if right_needs_parens {
            rs = format!("({})", rs);
        }
        let ind = context.indent();
        let new_code = format!(
            "{ind}{lhs} {op} {rhs}",
            ind = ind,
            lhs = ls,
            op = self.bin_op_id.to_symbol(),
            rhs = rs
        );
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

        let child = if is_left {
            &self.lhs
        } else {
            &self.rhs
        };

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
        both_left || both_right
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Add, BinOp, BinOpId, Expr, Lit, LitInner, Mul};
    use raystack_core::Number;

    fn c() -> Context {
        Context::new(0, MAX_WIDTH)
    }

    fn lit_num(n: usize) -> Lit {
        let num = Number::new(n as f64, None);
        let inner = LitInner::Num(num);
        Lit::new(inner)
    }

    fn ex_lit_num(n: usize) -> Expr {
        Expr::Lit(lit_num(n))
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
}
