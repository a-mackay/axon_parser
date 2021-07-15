use crate::ast::{
    Assign, Associativity, BinOp, Block, Def, Dict, DictVal, DotCall, Expr,
    FlatIf, Func, Id, If, List, Lit, Neg, Not, Range, Return, Throw, TrapCall,
    TryCatch,
};

/// The size of a single block of indentation, the number of spaces (' ').
const SPACES: usize = 4;
/// An arbitrary big maximum width, supposed to be bigger than the width
/// of any sane line of code.
const MAX_WIDTH: usize = 1000;

#[derive(Clone, Copy, Debug)]
struct Context {
    /// The number of spaces across this code should be.
    indent: usize,
    /// The maximum width allowed for this code (max number of characters allowed per line).
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

impl Rewrite for DictVal {
    fn rewrite(&self, context: Context) -> Option<String> {
        let ind = context.indent();
        let new_code = match self {
            DictVal::Expr(expr) => expr.rewrite(context)?,
            DictVal::Marker => format!("{ind}marker()", ind = ind),
            DictVal::RemoveMarker => format!("{ind}removeMarker()", ind = ind),
        };

        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }
}

impl Rewrite for Dict {
    fn rewrite(&self, context: Context) -> Option<String> {
        let pairs = self.pairs();
        let ind = context.indent();

        let new_code = if pairs.is_empty() {
            format!("{ind}{{}}", ind = ind)
        } else {
            // Try fit the entire dict on one line:
            let one_line_context =
                Context::new(context.indent, context.max_width() - 2);
            let pairs: Option<Vec<String>> = pairs
                .iter()
                .map(|(name, value)| {
                    value.rewrite(one_line_context).map(|code| {
                        let prefix = format!("{}: ", name);
                        add_after_leading_indent(&prefix, &code)
                    })
                })
                .collect();

            match pairs {
                Some(pairs) => {
                    let pairs = pairs
                        .iter()
                        .map(|item| item.trim())
                        .collect::<Vec<_>>();
                    let pairs_str = pairs.join(", ");
                    // There was enough space for each expression to be
                    // rewritten, so we see if it fits on one line:
                    let one_line = format!(
                        "{ind}{{{pairs}}}",
                        ind = ind,
                        pairs = pairs_str
                    );

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

impl Dict {
    /// Write the dict over multiple lines, each name/value pair is indented
    /// under the dict brackets.
    fn default_rewrite(&self, context: Context) -> Option<String> {
        let ind = context.indent();
        let pairs = self.pairs();
        let pairs_context = context.increase_indent();
        let pairs: Option<Vec<String>> = pairs
            .into_iter()
            .map(|(name, value)| {
                let code = value.rewrite(pairs_context);
                let prefix = format!("{}: ", name);
                code.map(|code| {
                    let code = add_after_leading_indent(&prefix, &code);
                    format!("{},\n", code)
                })
                .and_then(|pair_str| {
                    // pair_str is something like tagName: "some value"
                    if context.str_within_max_width(&pair_str) {
                        Some(pair_str)
                    } else {
                        None
                    }
                })
            })
            .collect();
        let pairs = pairs?;
        let pairs_str = pairs.join("");

        let new_code =
            format!("{ind}{{\n{pairs}{ind}}}", ind = ind, pairs = pairs_str);

        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }

    fn pairs(&self) -> Vec<(String, DictVal)> {
        let mut pairs = self
            .map
            .iter()
            .map(|(name, value)| (name.to_string(), value.clone()))
            .collect::<Vec<_>>();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        pairs
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
    /// Write the list over multiple lines, each element is
    /// indented under the list brackets.
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
            Self::Block(x) => x.rewrite(context),
            // Self::Call(_) => false,
            Self::Def(x) => x.rewrite(context),
            Self::Dict(x) => x.rewrite(context),
            Self::DotCall(x) => x.rewrite(context),
            Self::Func(x) => x.rewrite(context),
            Self::Id(x) => x.rewrite(context),
            Self::If(x) => x.rewrite(context),
            Self::List(x) => x.rewrite(context),
            Self::Lit(x) => x.rewrite(context),
            Self::Neg(x) => x.rewrite(context),
            Self::Not(x) => x.rewrite(context),
            // Self::PartialCall(_) => false,
            Self::Range(x) => x.rewrite(context),
            Self::Return(x) => x.rewrite(context),
            Self::Throw(x) => x.rewrite(context),
            Self::TrapCall(x) => x.rewrite(context),
            Self::TryCatch(x) => x.rewrite(context),
            _ => todo!(),
        }
    }
}

impl Rewrite for DotCall {
    fn rewrite(&self, context: Context) -> Option<String> {
        // By default, allow trailing lambdas.
        let use_trailing_lambda = true;
        self.rewrite_inner(context, use_trailing_lambda)
    }
}

impl DotCall {
    fn target_needs_parens(&self) -> bool {
        match self.target.as_ref() {
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
            Expr::Dict(_) => false,
            Expr::DotCall(_) => false,
            Expr::Func(_) => true,
            Expr::Id(_) => false,
            Expr::If(_) => true,
            Expr::List(_) => false,
            Expr::Lit(_) => false,
            Expr::Neg(_) => true,
            Expr::Not(_) => true,
            Expr::PartialCall(_) => false,
            Expr::Range(_) => true,
            Expr::Return(_) => true,
            Expr::Throw(_) => true,
            Expr::TrapCall(_) => false,
            Expr::TryCatch(_) => true,
        }
    }

    fn rewrite_inner(
        &self,
        context: Context,
        use_trailing_lambda: bool,
    ) -> Option<String> {
        let one_line = self.rewrite_one_line(context, use_trailing_lambda);
        if one_line.is_some() {
            return one_line;
        } else {
            self.rewrite_multi_line(context, use_trailing_lambda)
        }
    }

    fn rewrite_multi_line(
        &self,
        context: Context,
        use_trailing_lambda: bool,
    ) -> Option<String> {
        let lambda_pos = if use_trailing_lambda {
            LambdaPos::Trailing
        } else {
            LambdaPos::NotTrailing
        };

        let target_can_have_trailing_lambda = false;
        let target =
            self.rewrite_target(context, target_can_have_trailing_lambda)?;

        let has_lambda = self.has_lambda_last_arg();
        let is_target_one_line = is_one_line(&target);

        match (is_target_one_line, has_lambda) {
            (true, true) => self.rewrite_ml_targetone_lambda(context, lambda_pos),
            (true, false) => self.rewrite_ml_targetone(context),
            (false, true) => self.rewrite_ml_targetmulti_lambda(context, lambda_pos),
            (false, false) => self.rewrite_ml_targetmulti(context),
        }
    }

    fn rewrite_ml_targetone_lambda(&self, context: Context, lambda_pos: LambdaPos) -> Option<String> {
        let target_can_have_trailing_lambda = false;
        let target =
            self.rewrite_target(context, target_can_have_trailing_lambda)?;
        if !is_one_line(&target) {
            panic!("expected target to be one line");
        }
        if !self.has_lambda_last_arg() {
            panic!("expected self to have lambda last arg");
        }
        let name = &self.func_name;

        let cat = self.call_arg_type();
        let style = |layout: CallArgLayout| {
            CallArgStyle::new(layout, lambda_pos)
        };

        // See if putting only the lambda on multiple lines
        // works:
        let call1 = cat.rewrite(context, style(CallArgLayout::OneLineArgsMultiLineLambda))?;
        let prefix1 = format!("{target}.{name}", target = target.trim(), name = name);
        let code1 = add_after_leading_indent(&prefix1, &call1);
        if context.str_within_max_width(&code1) {
            return Some(code1);
        }

        // Now see if putting all args on multiple lines works:
        let call2 = cat.rewrite(context, style(CallArgLayout::MultiLine))?;
        let prefix2 = format!("{target}.{name}", target = target.trim(), name = name);
        let code2 = add_after_leading_indent(&prefix2, &call2);
        if context.str_within_max_width(&code2) {
            return Some(code2);
        }

        // Below here, we now decide to put the call on a separate line
        // to the target:

        // Try put the call all on one line first:
        let call3 = cat.rewrite(context.increase_indent(), style(CallArgLayout::OneLine))?;
        let prefix3 = format!(".{}", name);
        let call3 = add_after_leading_indent(&prefix3, &call3);
        let code3 = format!("{target}\n{call}", target = target, call = call3);
        if context.str_within_max_width(&code3) {
            return Some(code3);
        }

        // Try put the just the lambda over multiple lines:
        let call4 = self.call_arg_type().rewrite(context.increase_indent(), style(CallArgLayout::OneLineArgsMultiLineLambda))?;
        let prefix4 = format!(".{}", name);
        let call4 = add_after_leading_indent(&prefix4, &call4);
        let code4 = format!("{target}\n{call}", target = target, call = call4);
        if context.str_within_max_width(&code4) {
            return Some(code4);
        }

        // Fallback, just write the entire call across multiple lines.
        let call5 = self.call_arg_type().rewrite(context.increase_indent(), style(CallArgLayout::MultiLine))?;
        let prefix5 = format!(".{}", self.func_name);
        let call5 = add_after_leading_indent(&prefix5, &call5);
        let code5 = format!("{target}\n{call}", target = target, call = call5);
        if context.str_within_max_width(&code5) {
            return Some(code5);
        }

        None
    }

    fn rewrite_ml_targetone(&self, context: Context) -> Option<String> {
        let target_can_have_trailing_lambda = false;
        let target =
            self.rewrite_target(context, target_can_have_trailing_lambda)?;
        if !is_one_line(&target) {
            panic!("expected target to be one line");
        }
        if self.has_lambda_last_arg() {
            panic!("expected self to not have lambda last arg");
        }
        let name = &self.func_name;

        let cat = self.call_arg_type();
        let style = |layout: CallArgLayout| {
            let irrelevant_lambda_pos = LambdaPos::NotTrailing;
            CallArgStyle::new(layout, irrelevant_lambda_pos)
        };

        // Try put the call all on one line first:
        let call2 = cat.rewrite(context.increase_indent(), style(CallArgLayout::OneLine))?;
        let prefix2 = format!(".{}", name);
        let call2 = add_after_leading_indent(&prefix2, &call2);
        let code2 = format!("{target}\n{call}", target = target, call = call2);
        if context.str_within_max_width(&code2) {
            return Some(code2);
        }

        // See if putting all args on multiple lines works:
        let call1 = cat.rewrite(context, style(CallArgLayout::MultiLine))?;
        let prefix1 = format!("{target}.{name}", target = target.trim(), name = name);
        let code1 = add_after_leading_indent(&prefix1, &call1);
        if context.str_within_max_width(&code1) {
            return Some(code1);
        }

        // Below here, we now decide to put the call on a separate line
        // to the target:

        // Fallback, just write the entire call across multiple lines.
        let call3 = self.call_arg_type().rewrite(context.increase_indent(), style(CallArgLayout::MultiLine))?;
        let prefix3 = format!(".{}", self.func_name);
        let call3 = add_after_leading_indent(&prefix3, &call3);
        let code3 = format!("{target}\n{call}", target = target, call = call3);
        if context.str_within_max_width(&code3) {
            return Some(code3);
        }

        None
    }


    fn rewrite_ml_targetmulti_lambda(&self, context: Context, lambda_pos: LambdaPos) -> Option<String> {
        let target_can_have_trailing_lambda = false;
        let target =
            self.rewrite_target(context, target_can_have_trailing_lambda)?;
        if is_one_line(&target) {
            panic!("expected target to be multi line");
        }
        if !self.has_lambda_last_arg() {
            panic!("expected self to have lambda last arg");
        }
        let name = &self.func_name;

        let cat = self.call_arg_type();
        let style = |layout: CallArgLayout| {
            CallArgStyle::new(layout, lambda_pos)
        };

        // Try put the call all on one line first:
        let call3 = cat.rewrite(context.increase_indent(), style(CallArgLayout::OneLine))?;
        let prefix3 = format!(".{}", name);
        let call3 = add_after_leading_indent(&prefix3, &call3);
        let code3 = format!("{target}\n{call}", target = target, call = call3);
        if context.str_within_max_width(&code3) {
            return Some(code3);
        }

        // Try put the just the lambda over multiple lines:
        let call4 = self.call_arg_type().rewrite(context.increase_indent(), style(CallArgLayout::OneLineArgsMultiLineLambda))?;
        let prefix4 = format!(".{}", name);
        let call4 = add_after_leading_indent(&prefix4, &call4);
        let code4 = format!("{target}\n{call}", target = target, call = call4);
        if context.str_within_max_width(&code4) {
            return Some(code4);
        }

        // Fallback, just write the entire call across multiple lines.
        let call5 = self.call_arg_type().rewrite(context.increase_indent(), style(CallArgLayout::MultiLine))?;
        let prefix5 = format!(".{}", self.func_name);
        let call5 = add_after_leading_indent(&prefix5, &call5);
        let code5 = format!("{target}\n{call}", target = target, call = call5);
        if context.str_within_max_width(&code5) {
            return Some(code5);
        }

        None
    }


    fn rewrite_ml_targetmulti(&self, context: Context) -> Option<String> {
        let target_can_have_trailing_lambda = false;
        let target =
            self.rewrite_target(context, target_can_have_trailing_lambda)?;
        if is_one_line(&target) {
            panic!("expected target to be multi line");
        }
        if self.has_lambda_last_arg() {
            panic!("expected self to not have lambda last arg");
        }
        let name = &self.func_name;

        let cat = self.call_arg_type();
        let style = |layout: CallArgLayout| {
            let irrelevant_lambda_pos = LambdaPos::NotTrailing;
            CallArgStyle::new(layout, irrelevant_lambda_pos)
        };

        // Try put the call all on one line first:
        let call2 = cat.rewrite(context.increase_indent(), style(CallArgLayout::OneLine))?;
        let prefix2 = format!(".{}", name);
        let call2 = add_after_leading_indent(&prefix2, &call2);
        let code2 = format!("{target}\n{call}", target = target, call = call2);
        if context.str_within_max_width(&code2) {
            return Some(code2);
        }

        // Fallback, just write the entire call across multiple lines.
        let call3 = self.call_arg_type().rewrite(context.increase_indent(), style(CallArgLayout::MultiLine))?;
        let prefix3 = format!(".{}", self.func_name);
        let call3 = add_after_leading_indent(&prefix3, &call3);
        let code3 = format!("{target}\n{call}", target = target, call = call3);
        if context.str_within_max_width(&code3) {
            return Some(code3);
        }

        None
    }


    /// Rewrite the target.
    fn rewrite_target(
        &self,
        context: Context,
        target_can_have_trailing_lambda: bool,
    ) -> Option<String> {
        let target_needs_parens = self.target_needs_parens();
        let mut target = if target_can_have_trailing_lambda {
            // Rewrite the target in a way that allows trailing lambdas, if
            // they are present.
            self.target.rewrite(context)? // TODO check this logic
        } else {
            match self.target.as_ref() {
                Expr::Add(x) => x.0.rewrite(context)?,
                Expr::And(x) => x.0.rewrite(context)?,
                Expr::Cmp(x) => x.0.rewrite(context)?,
                Expr::Div(x) => x.0.rewrite(context)?,
                Expr::Eq(x) => x.0.rewrite(context)?,
                Expr::Gt(x) => x.0.rewrite(context)?,
                Expr::Gte(x) => x.0.rewrite(context)?,
                Expr::Lt(x) => x.0.rewrite(context)?,
                Expr::Lte(x) => x.0.rewrite(context)?,
                Expr::Mul(x) => x.0.rewrite(context)?,
                Expr::Ne(x) => x.0.rewrite(context)?,
                Expr::Or(x) => x.0.rewrite(context)?,
                Expr::Sub(x) => x.0.rewrite(context)?,
                Expr::Assign(x) => x.rewrite(context)?,
                Expr::Block(x) => x.rewrite(context)?,
                Expr::Call(_) => todo!(
                    "rewrite the target such that there is no trailing lambda"
                ),
                Expr::Def(x) => x.rewrite(context)?,
                Expr::Dict(x) => x.rewrite(context)?,
                Expr::DotCall(x) => x.rewrite_inner(context, false)?,
                Expr::Func(x) => x.rewrite(context)?,
                Expr::Id(x) => x.rewrite(context)?,
                Expr::If(x) => x.rewrite(context)?,
                Expr::List(x) => x.rewrite(context)?,
                Expr::Lit(x) => x.rewrite(context)?,
                Expr::Neg(x) => x.rewrite(context)?,
                Expr::Not(x) => x.rewrite(context)?,
                Expr::PartialCall(_) => todo!(
                    "rewrite the target such that there is no trailing lambda"
                ),
                Expr::Range(x) => x.rewrite(context)?,
                Expr::Return(x) => x.rewrite(context)?,
                Expr::Throw(x) => x.rewrite(context)?,
                Expr::TrapCall(x) => x.rewrite(context)?,
                Expr::TryCatch(x) => x.rewrite(context)?,
            }
        };
        if target_needs_parens {
            target = add_parens(&target);
        }

        if context.str_within_max_width(&target) {
            return Some(target);
        } else {
            None
        }
    }

    /// Try rewrite this DotCall on a single line.
    fn rewrite_one_line(
        &self,
        context: Context,
        use_trailing_lambda: bool,
    ) -> Option<String> {
        let lambda_pos = if use_trailing_lambda {
            LambdaPos::Trailing
        } else {
            LambdaPos::NotTrailing
        };
        let target_can_have_trailing_lambda = false;
        let target =
            self.rewrite_target(context, target_can_have_trailing_lambda)?;

        if !is_one_line(&target) {
            return None;
        }

        let style = CallArgStyle::new(CallArgLayout::OneLine, lambda_pos);
        let call = self.call_arg_type().rewrite(context, style)?;

        let prefix = format!(
            "{target}.{name}",
            target = target.trim(),
            name = self.func_name
        );
        let new_code = add_after_leading_indent(&prefix, &call);

        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }
}

impl DotCall {
    fn call_arg_type(&self) -> CallArgType {
        let mut args = self.args.clone();
        if args.is_empty() {
            CallArgType::NoArgs(NoArgs)
        } else {
            let last_index = args.len() - 1;
            let last_arg = args.remove(last_index);
            match last_arg {
                Expr::Func(func) => {
                    if args.is_empty() {
                        CallArgType::OnlyLambda(OnlyLambda::new(*func))
                    } else {
                        let args = args.into_iter().map(Arg::Expr).collect();
                        CallArgType::ArgsAndLambda(ArgsAndLambda::new(
                            args, *func,
                        ))
                    }
                }
                _ => {
                    args.push(last_arg);
                    let all_args =
                        args.into_iter().map(|a| Arg::Expr(a)).collect();
                    CallArgType::OnlyArgs(OnlyArgs::new(all_args))
                }
            }
        }
    }
}

/// Categorize the types of arguments that are passed to a function call.
/// This only includes the arguments passed at the end (that is, not
/// including the target argument for a DotCall).
enum CallArgType {
    // e.g. func()
    NoArgs(NoArgs),
    // e.g. func(1, 2, ..., 3)
    OnlyArgs(OnlyArgs),
    // e.g. func(() => 100)
    OnlyLambda(OnlyLambda),
    // e.g. func(1, 2, ... , () => 100)
    ArgsAndLambda(ArgsAndLambda),
}

impl CallArgType {
    /// Rewrite these call arguments, including the necessary parentheses.
    fn rewrite(&self, context: Context, style: CallArgStyle) -> Option<String> {
        match self {
            Self::NoArgs(x) => x.rewrite(context),
            Self::OnlyArgs(x) => x.rewrite(context, style.layout),
            Self::OnlyLambda(x) => x.rewrite(context, style),
            Self::ArgsAndLambda(x) => x.rewrite(context, style),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Arg {
    Expr(Expr),
    Placeholder,
}

impl Rewrite for Arg {
    fn rewrite(&self, context: Context) -> Option<String> {
        match self {
            Self::Expr(expr) => expr.rewrite(context),
            Self::Placeholder => {
                let code = format!("{ind}_", ind = context.indent());
                if context.str_within_max_width(&code) {
                    Some(code)
                } else {
                    None
                }
            }
        }
    }
}

struct NoArgs;

impl NoArgs {
    fn rewrite(&self, context: Context) -> Option<String> {
        let new_code = format!("{ind}()", ind = context.indent());
        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }
}

struct OnlyArgs {
    args: Vec<Arg>,
}

impl OnlyArgs {
    fn new(args: Vec<Arg>) -> Self {
        Self { args }
    }

    fn rewrite(
        &self,
        context: Context,
        layout: CallArgLayout,
    ) -> Option<String> {
        let new_code = match layout {
            CallArgLayout::OneLine
            | CallArgLayout::OneLineArgsMultiLineLambda => {
                let arg_context = Context::new(0, MAX_WIDTH);
                let args: Vec<String> = self
                    .args
                    .iter()
                    .map(|arg| {
                        arg.rewrite(arg_context).expect(
                            "should be able to rewrite arg within MAX_WIDTH",
                        )
                    })
                    .collect();
                let args_str = args.join(", ");
                if is_one_line(&args_str) {
                    format!("{ind}({args})", ind = context.indent(), args = args_str)
                } else {
                    return None;
                }
            }
            CallArgLayout::MultiLine => {
                let arg_context = context.increase_indent();
                let args: Option<Vec<String>> = self
                    .args
                    .iter()
                    .map(|arg| arg.rewrite(arg_context))
                    .collect();
                let args = args?;
                let args_str = args.join(",\n");
                format!(
                    "{ind}(\n{args}\n{ind})",
                    ind = context.indent(),
                    args = args_str
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

struct OnlyLambda {
    lambda: Func,
}

impl OnlyLambda {
    fn new(lambda: Func) -> Self {
        Self { lambda }
    }

    fn rewrite(&self, context: Context, style: CallArgStyle) -> Option<String> {
        let style = (style.layout, style.lambda_pos);
        let func = self.lambda.clone();
        let new_code = match style {
            (CallArgLayout::OneLine, LambdaPos::NotTrailing) => {
                let code = func.rewrite(context)?;
                if is_one_line(&code) {
                    let code = add_after_leading_indent("(", &code);
                    format!("{})", code)
                } else {
                    return None;
                }
            }
            (CallArgLayout::OneLine, LambdaPos::Trailing) => {
                let code = func.rewrite(context)?;
                if is_one_line(&code) {
                    add_after_leading_indent("() ", &code)
                } else {
                    return None;
                }
            }
            (
                CallArgLayout::OneLineArgsMultiLineLambda,
                LambdaPos::NotTrailing,
            ) => {
                let func = func.blockify();
                let code = func.rewrite(context)?;
                let code = add_after_leading_indent("(", &code);
                format!("{})", code)
            }
            (
                CallArgLayout::OneLineArgsMultiLineLambda,
                LambdaPos::Trailing,
            ) => {
                let func = func.blockify();
                let code = func.rewrite(context)?;
                add_after_leading_indent("() ", &code)
            }
            (CallArgLayout::MultiLine, LambdaPos::NotTrailing) => {
                let func = func.blockify();
                let code = func.rewrite(context)?;
                let code = add_after_leading_indent("(", &code);
                format!("{})", code)
            }
            (CallArgLayout::MultiLine, LambdaPos::Trailing) => {
                let func = func.blockify();
                let code = func.rewrite(context)?;
                add_after_leading_indent("() ", &code)
            }
        };

        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }
}

struct ArgsAndLambda {
    args: Vec<Arg>,
    lambda: Func,
}

impl ArgsAndLambda {
    fn new(args: Vec<Arg>, lambda: Func) -> Self {
        Self { args, lambda }
    }

    fn strip_parens(s: &str) -> String {
        // Assumes s is like "(something)"
        let mut chars = s.chars();
        chars.next();
        chars.next_back();
        chars.as_str().to_owned()
    }

    fn rewrite(&self, context: Context, style: CallArgStyle) -> Option<String> {
        let style = (style.layout, style.lambda_pos);
        // This is safe to do because we know self.args is not empty.
        let only_args = OnlyArgs::new(self.args.clone());
        let func = self.lambda.clone();
        let new_code = match style {
            (CallArgLayout::OneLine, LambdaPos::NotTrailing) => {
                let code = func.rewrite(context)?;
                if is_one_line(&code) {
                    let args_context = Context::new(0, MAX_WIDTH);
                    let args = only_args
                        .rewrite(args_context, CallArgLayout::OneLine)
                        .expect(
                            "could not rewrite args on a single MAX_WIDTH line",
                        );
                    let prefix = format!("({}, ", Self::strip_parens(&args));
                    let code = add_after_leading_indent(&prefix, &code);
                    format!("{})", code)
                } else {
                    return None;
                }
            }
            (CallArgLayout::OneLine, LambdaPos::Trailing) => {
                let code = func.rewrite(context)?;
                if is_one_line(&code) {
                    let args_context = Context::new(0, MAX_WIDTH);
                    let args = only_args
                        .rewrite(args_context, CallArgLayout::OneLine)
                        .expect(
                            "could not rewrite args on a single MAX_WIDTH line",
                        );
                    let prefix = format!("({}) ", Self::strip_parens(&args));
                    add_after_leading_indent(&prefix, &code)
                } else {
                    return None;
                }
            }
            (
                CallArgLayout::OneLineArgsMultiLineLambda,
                LambdaPos::NotTrailing,
            ) => {
                let func = func.blockify();
                let code = func.rewrite(context)?;

                let args_context = Context::new(0, MAX_WIDTH);
                let args = only_args
                    .rewrite(args_context, CallArgLayout::OneLine)
                    .expect(
                        "could not rewrite args on a single MAX_WIDTH line",
                    );
                let prefix = format!("({}, ", Self::strip_parens(&args));
                let code = add_after_leading_indent(&prefix, &code);
                format!("{})", code)
            }
            (
                CallArgLayout::OneLineArgsMultiLineLambda,
                LambdaPos::Trailing,
            ) => {
                let func = func.blockify();
                let code = func.rewrite(context)?;

                let args_context = Context::new(0, MAX_WIDTH);
                let args = only_args
                    .rewrite(args_context, CallArgLayout::OneLine)
                    .expect(
                        "could not rewrite args on a single MAX_WIDTH line",
                    );
                let prefix = format!("({}) ", Self::strip_parens(&args));
                add_after_leading_indent(&prefix, &code)
            }
            (CallArgLayout::MultiLine, LambdaPos::NotTrailing) => {
                let args_context = context.increase_indent();

                let func = func.blockify();
                let func = func.rewrite(args_context)?;

                let args = self.args.clone();
                let arg_strs: Option<Vec<String>> =
                    args.iter().map(|arg| arg.rewrite(args_context)).collect();
                let args_str = arg_strs?.join(",\n");

                let ind = context.indent();
                format!(
                    "{ind}(\n{args},\n{func}\n{ind})",
                    ind = ind,
                    args = args_str,
                    func = func
                )
            }
            (CallArgLayout::MultiLine, LambdaPos::Trailing) => {
                let func = func.blockify();
                let func = func.rewrite(context)?;

                let args_context = context.increase_indent();
                let args = self.args.clone();
                let arg_strs: Option<Vec<String>> =
                    args.iter().map(|arg| arg.rewrite(args_context)).collect();
                let args_str = arg_strs?.join(",\n");

                let code = add_after_leading_indent(") ", &func);

                let ind = context.indent();
                format!(
                    "{ind}(\n{args}\n{code}",
                    ind = ind,
                    args = args_str,
                    code = code
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

#[derive(Clone, Copy, Debug, PartialEq)]
struct CallArgStyle {
    layout: CallArgLayout,
    lambda_pos: LambdaPos,
}

impl CallArgStyle {
    fn new(layout: CallArgLayout, lambda_pos: LambdaPos) -> Self {
        Self { layout, lambda_pos }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum LambdaPos {
    Trailing,
    NotTrailing,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum CallArgLayout {
    OneLine,
    OneLineArgsMultiLineLambda,
    MultiLine,
}

impl Rewrite for If {
    fn rewrite(&self, context: Context) -> Option<String> {
        self.flatten().rewrite(context)
    }
}

impl Rewrite for FlatIf {
    fn rewrite(&self, context: Context) -> Option<String> {
        if self.has_single_condition() {
            let one_line = self.rewrite_one_line(context);
            if one_line.is_some() {
                return one_line;
            }
        }

        self.default_rewrite(context)
    }
}

impl FlatIf {
    fn has_single_condition(&self) -> bool {
        self.cond_exprs.len() == 1
    }

    fn rewrite_one_line(&self, context: Context) -> Option<String> {
        // We know we have exactly one conditional expr:
        let cond_expr = self.cond_exprs.clone().remove(0);
        let cond = cond_expr.cond.rewrite(context)?;
        let is_cond_one = is_one_line(&cond);
        let mut if_expr = cond_expr.expr.clone().rewrite(context)?;
        let is_if_one = is_one_line(&if_expr);

        if FlatIf::one_line_expr_needs_parens(&cond_expr.expr) {
            if_expr = add_parens(&if_expr);
        }

        match &self.else_expr {
            Some(else_expr) => {
                let mut else_expr_str = else_expr.rewrite(context)?;
                let is_else_one = is_one_line(&else_expr_str);

                if FlatIf::one_line_expr_needs_parens(else_expr) {
                    else_expr_str = add_parens(&else_expr_str);
                }

                match (is_if_one, is_cond_one, is_else_one) {
                    (true, true, true) => {
                        let cond = add_after_leading_indent("if (", &cond);
                        let cond = format!("{})", cond);
                        let if_expr = if_expr.trim();
                        let else_expr_str = else_expr_str.trim();
                        let code = format!(
                            "{} {} else {}",
                            cond, if_expr, else_expr_str
                        );

                        if context.str_within_max_width(&code) {
                            Some(code)
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            None => match (is_if_one, is_cond_one) {
                (true, true) => {
                    let cond = add_after_leading_indent("if (", &cond);
                    let cond = format!("{})", cond);
                    let if_expr = if_expr.trim();
                    let code = format!("{} {}", cond, if_expr);

                    if context.str_within_max_width(&code) {
                        Some(code)
                    } else {
                        None
                    }
                }
                _ => None,
            },
        }
    }

    /// Given an expression which will be written on one line, return
    /// true if it needs to be surrounded by parentheses.
    fn one_line_expr_needs_parens(expr: &Expr) -> bool {
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
            Expr::Call(_) => true,
            Expr::Def(_) => true,
            Expr::Dict(_) => false,
            Expr::DotCall(_) => true,
            Expr::Func(_) => true,
            Expr::Id(_) => false,
            Expr::If(_) => true,
            Expr::List(_) => false,
            Expr::Lit(_) => false,
            Expr::Neg(_) => false,
            Expr::Not(_) => true,
            Expr::PartialCall(_) => true,
            Expr::Range(_) => true,
            Expr::Return(_) => true,
            Expr::Throw(_) => true,
            Expr::TrapCall(_) => false,
            Expr::TryCatch(_) => true,
        }
    }

    fn default_rewrite(&self, context: Context) -> Option<String> {
        let mut conds = vec![];
        for (index, cond_expr) in self.cond_exprs.iter().enumerate() {
            let is_first_cond = index == 0;
            let cond = cond_expr.cond.clone();
            let cond_code = FlatIf::default_rewrite_cond(
                &cond_expr.cond,
                context,
                is_first_cond,
            )?;
            let expr = cond_expr.expr.clone().blockify();
            let expr_code = expr.rewrite(context)?;
            conds.push(CondExprAndCode::new(cond, cond_code, expr, expr_code));
        }

        let mut strs = vec![];
        for cond in conds.into_iter() {
            strs.push(cond.cond_code);
            strs.push(strip_do_end_from_block(&cond.expr_code));
        }

        let ind = context.indent();

        let new_code = match &self.else_expr {
            Some(else_expr) => {
                strs.push(format!("{ind}end else do", ind = ind));
                let else_expr = else_expr.clone().blockify();
                let else_expr_code = else_expr.rewrite(context)?;
                strs.push(strip_do_end_from_block(&else_expr_code));
                strs.push(format!("{ind}end", ind = ind));
                strs.join("\n")
            }
            None => {
                strs.push(format!("{ind}end", ind = ind));
                strs.join("\n")
            }
        };

        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }

    // We assume each cond's related expression is a Block.
    fn default_rewrite_cond(
        cond: &Expr,
        context: Context,
        is_first_cond: bool,
    ) -> Option<String> {
        let multi_line_context = context.increase_indent();

        // Try put the condition on one line first:
        let code = cond.rewrite(context);
        if let Some(code) = code {
            if is_one_line(&code) {
                let code = add_parens(&code);
                let prefix = if is_first_cond { "if " } else { "end else if " };
                let code = add_after_leading_indent(prefix, &code);
                let code = format!("{} do", code);

                if context.str_within_max_width(&code) {
                    return Some(code);
                }
            }
        }

        // Try put the condition by itself on a new line:
        let code = cond.rewrite(multi_line_context)?;
        let ind = context.indent();
        let code = if is_first_cond {
            format!("{ind}if (\n{cond}\n{ind}) do", ind = ind, cond = code)
        } else {
            format!(
                "{ind}end else if (\n{cond}\n{ind}) do",
                ind = ind,
                cond = code
            )
        };

        if context.str_within_max_width(&code) {
            Some(code)
        } else {
            None
        }
    }
}

fn strip_do_end_from_block(block_code: &str) -> String {
    let mut lines = block_code.lines().collect::<Vec<_>>();
    lines.remove(0);
    let last_index = lines.len() - 1;
    lines.remove(last_index);
    lines.join("\n")
}

#[derive(Debug)]
struct CondExprAndCode {
    cond: Expr,
    cond_code: String,
    expr: Expr,
    expr_code: String,
}

impl CondExprAndCode {
    fn new(
        cond: Expr,
        cond_code: String,
        expr: Expr,
        expr_code: String,
    ) -> Self {
        Self {
            cond,
            cond_code,
            expr,
            expr_code,
        }
    }
}

impl Rewrite for TryCatch {
    fn rewrite(&self, context: Context) -> Option<String> {
        let one_line = self.rewrite_one_line(context);
        if one_line.is_some() {
            one_line
        } else {
            self.default_rewrite(context)
        }
    }
}

impl TryCatch {
    /// Try to rewrite the expression to a single line.
    fn rewrite_one_line(&self, context: Context) -> Option<String> {
        let try_expr = self.try_expr.rewrite(context)?;
        let catch_expr = self.catch_expr.rewrite(context)?;

        match (is_one_line(&try_expr), is_one_line(&catch_expr)) {
            (true, true) => match &self.exception_name {
                Some(exception_name) => {
                    let code = add_after_leading_indent("try ", &try_expr);
                    let code = format!(
                        "{} catch ({}) {}",
                        code, exception_name, catch_expr
                    );
                    if context.str_within_max_width(&code) {
                        Some(code)
                    } else {
                        None
                    }
                }
                None => {
                    let code = add_after_leading_indent("try ", &try_expr);
                    let code = format!("{} catch {}", code, catch_expr);
                    if context.str_within_max_width(&code) {
                        Some(code)
                    } else {
                        None
                    }
                }
            },
            _ => None,
        }
    }

    /// Rewrite over multiple lines.
    fn default_rewrite(&self, context: Context) -> Option<String> {
        let try_expr = self.try_expr.clone().blockify().rewrite(context)?;
        let try_expr = add_after_leading_indent("try ", &try_expr);
        let catch_expr = self.catch_expr.clone().blockify().rewrite(context)?;
        let catch_expr = catch_expr.trim_start();

        let new_code = match &self.exception_name {
            Some(exception_name) => {
                format!(
                    "{} catch ({}) {}",
                    try_expr, exception_name, catch_expr
                )
            }
            None => {
                format!("{} catch {}", try_expr, catch_expr)
            }
        };

        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }
}

impl Rewrite for Func {
    fn rewrite(&self, context: Context) -> Option<String> {
        let one_line = self.rewrite_one_line(context);
        if one_line.is_some() {
            one_line
        } else {
            self.default_rewrite(context)
        }
    }
}

impl Func {
    fn rewrite_one_line(&self, context: Context) -> Option<String> {
        let body = self.body.rewrite(context)?;
        let params = &self.params[..];
        if is_one_line(&body) {
            let mut param_strs = vec![];
            for param in params {
                let name = param.name.to_string();
                let default: Option<String> = match &param.default {
                    Some(expr) => {
                        let default_code = expr.rewrite(context);
                        match default_code {
                            Some(default_code) => Some(default_code),
                            // If we can't rewrite the default argument
                            // on a single line, we cannot write the entire
                            // function on one line, so we return early.
                            None => return None,
                        }
                    }
                    None => None,
                };

                let param_code = match default {
                    Some(default) => {
                        format!(
                            "{name}: {default}",
                            name = name,
                            default = default
                        )
                    }
                    None => name,
                };
                param_strs.push(param_code);
            }

            let params_str = param_strs.join(", ");
            let needs_parens = self.needs_parens_around_params();
            let prefix = if needs_parens {
                format!("({}) => ", params_str)
            } else {
                format!("{} => ", params_str)
            };
            let new_code = add_after_leading_indent(&prefix, &body);

            if context.str_within_max_width(&new_code) {
                Some(new_code)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn needs_parens_around_params(&self) -> bool {
        if self.params.len() == 1 {
            let only_param = self.params.first().unwrap();
            // If this parameter has a default argument, we needs parentheses:
            only_param.default.is_some()
        } else {
            // Either no params, or multiple params, so we need parentheses:
            true
        }
    }

    fn default_rewrite(&self, context: Context) -> Option<String> {
        let func = self.clone();
        let func = func.blockify();

        let param_context = Context::new(0, MAX_WIDTH);
        let mut params = vec![];
        for param in &func.params {
            let name = param.name.to_string();
            let default: Option<String> = param.default.as_ref().and_then(|default| {
                let code = default.rewrite(param_context).expect("rewriting a default param did not return a string when using MAX_WIDTH");
                // We hope the param fits on a single line of code
                // because we told it to rewrite itself within MAX_WIDTH
                // which is very wide.
                if is_one_line(&code) {
                    Some(code)
                } else {
                    panic!("a rewritten default param using MAX_WIDTH was not a single line")
                }
            });

            let param_code = match default {
                Some(default) => {
                    format!("{name}: {default}", name = name, default = default)
                }
                None => name,
            };
            params.push(param_code);
        }

        let params_str = params.join(", ");
        // Because of the call to blockify above, we know the body is a Block.
        let prefix = format!("({}) => ", params_str);
        let body = func.body.rewrite(context)?;
        let new_code = add_after_leading_indent(&prefix, &body);

        // We deliberately do not check new_code fits within the given
        // width because we know the body already does, and I want the
        // function parameters to all be on a single line regardless of
        // desired width.
        Some(new_code)
    }
}

struct ExprAndCode {
    expr: Expr,
    code: String,
}

impl ExprAndCode {
    fn new(expr: Expr, code: String) -> Self {
        Self { expr, code }
    }
}

/// Returns true if this expression should be followed by an additional newline
/// character. For example, a Block should have a blank line before the following
/// expression, for readability.
fn needs_newline(expr: &Expr, next_expr: Option<&Expr>) -> bool {
    let is_followed_by_expr = next_expr.is_some();
    match expr {
        Expr::Add(_) => false,
        Expr::And(_) => false,
        Expr::Cmp(_) => false,
        Expr::Div(_) => false,
        Expr::Eq(_) => false,
        Expr::Gt(_) => false,
        Expr::Gte(_) => false,
        Expr::Lt(_) => false,
        Expr::Lte(_) => false,
        Expr::Mul(_) => false,
        Expr::Ne(_) => false,
        Expr::Or(_) => false,
        Expr::Sub(_) => false,
        Expr::Assign(_) => false,
        Expr::Block(_) => is_followed_by_expr,
        Expr::Call(_) => false,
        Expr::Def(_) => false,
        Expr::Dict(_) => false,
        Expr::DotCall(_) => false,
        Expr::Func(_) => is_followed_by_expr,
        Expr::Id(_) => false,
        Expr::If(_) => is_followed_by_expr,
        Expr::List(_) => false,
        Expr::Lit(_) => false,
        Expr::Neg(_) => false,
        Expr::Not(_) => false,
        Expr::PartialCall(_) => false,
        Expr::Range(_) => false,
        Expr::Return(_) => false,
        Expr::Throw(_) => false,
        Expr::TrapCall(_) => false,
        Expr::TryCatch(_) => is_followed_by_expr,
    }
}

impl Rewrite for Block {
    fn rewrite(&self, context: Context) -> Option<String> {
        let exprs: &[Expr] = &self.exprs;
        let ind = context.indent();
        let expr_context = context.increase_indent();

        let mut expr_codes = vec![];
        for expr in exprs {
            let code = expr.rewrite(expr_context)?;
            let code = format!("{}\n", code);
            expr_codes.push(ExprAndCode::new(expr.clone(), code));
        }

        let mut exprs_str: Vec<String> = vec![];
        let mut iter = expr_codes.into_iter().peekable();
        while let Some(expr_code) = iter.next() {
            let expr = expr_code.expr;
            let code = expr_code.code;
            let next_expr = iter.peek().map(|ec| &ec.expr);
            exprs_str.push(code);

            if needs_newline(&expr, next_expr) {
                exprs_str.push("\n".to_owned());
            }
        }

        let exprs_str = exprs_str.join("");

        let new_code =
            format!("{ind}do\n{exprs}{ind}end", ind = ind, exprs = exprs_str);

        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
        }
    }
}

// todo consider trailing lambda func calls and how they affect
// requiring parentheses. Eg abc(() => 1)->xyz works, abc() () => 1->xyz wont.

impl Rewrite for TrapCall {
    fn rewrite(&self, context: Context) -> Option<String> {
        let needs_parens = match &self.target {
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
            Expr::Dict(_) => false,
            Expr::DotCall(_) => false,
            Expr::Func(_) => true,
            Expr::Id(_) => false,
            Expr::If(_) => true,
            Expr::List(_) => false,
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
        let mut target = self.target.rewrite(context)?;
        if needs_parens {
            target = add_parens(&target);
        }
        let key = self.key.to_string();
        let new_code = format!("{target}->{key}", target = target, key = key);
        if context.str_within_max_width(&new_code) {
            Some(new_code)
        } else {
            None
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
        Add, And, Assign, BinOp, BinOpId, Block, Def, Dict, DotCall, Expr,
        FuncName, Id, If, List, Lit, LitInner, Mul, Neg, Not, Or, Param,
        Return, Sub, Throw, TrapCall, TryCatch,
    };
    use raystack_core::{Number, TagName};
    use std::collections::HashMap;

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

    fn longish_list() -> Expr {
        Expr::List(List::new(vec![
            ex_lit_num(1000),
            ex_lit_num(2000),
            ex_lit_num(3000),
        ]))
    }

    #[test]
    fn neg_multi_line_works() {
        let add = BinOp::new(ex_lit_num(1000), BinOpId::Add, ex_lit_num(2000));
        let add = Expr::Add(Box::new(Add(add)));
        let neg = Expr::Neg(Box::new(Neg::new(add)));
        let code = neg.rewrite(nc(1, 8)).unwrap();
        assert_eq!(code, " -(1000\n + 2000)");
    }

    #[test]
    fn neg_multi_line_parens_works() {
        let neg = Neg::new(longish_list());
        let code = neg.rewrite(nc(1, 10)).unwrap();
        assert_eq!(code, " -([\n     1000,\n     2000,\n     3000,\n ])")
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
        let or = BinOp::new(ex_lit_bool(true), BinOpId::Or, ex_lit_bool(false));
        let or = Expr::Or(Box::new(Or(or)));
        let not = Expr::Not(Box::new(Not::new(or)));
        let code = not.rewrite(nc(1, 10)).unwrap();
        assert_eq!(code, " not (true\n or false)");
    }

    #[test]
    fn not_multi_line_parens_works() {
        let not = Not::new(longish_list());
        let code = not.rewrite(nc(1, 10)).unwrap();
        assert_eq!(code, " not ([\n     1000,\n     2000,\n     3000,\n ])")
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
        let ass = Assign::new(tn("a"), longish_list());
        let ass = Expr::Assign(ass);
        let code = ass.rewrite(nc(1, 10)).unwrap();
        assert_eq!(code, " a = [\n     1000,\n     2000,\n     3000,\n ]")
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
        let def = Def::new(tn("a"), longish_list());
        let def = Expr::Def(def);
        let code = def.rewrite(nc(1, 10)).unwrap();
        assert_eq!(code, " a: [\n     1000,\n     2000,\n     3000,\n ]")
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
        let ret = Return::new(longish_list());
        let ret = Expr::Return(Box::new(ret));
        let code = ret.rewrite(nc(1, 10)).unwrap();
        assert_eq!(code, " return [\n     1000,\n     2000,\n     3000,\n ]")
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
        let throw = Throw::new(longish_list());
        let throw = Expr::Throw(Box::new(throw));
        let code = throw.rewrite(nc(1, 10)).unwrap();
        assert_eq!(code, " throw [\n     1000,\n     2000,\n     3000,\n ]")
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
        let start = Expr::Block(Block::new(vec![ex_lit_num(100)]));
        let end = Expr::Block(Block::new(vec![ex_lit_num(200)]));
        let range = Range::new(start, end);
        let code = range.rewrite(nc(4, 11)).unwrap();
        let expected = "    (do
        100
    end)
    ..
    (do
        200
    end)";
        assert_eq!(code, expected)
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

    fn dict(items: Vec<(&str, DictVal)>) -> Dict {
        let mut map = HashMap::new();
        for (name, value) in items {
            map.insert(tn(name), value);
        }
        Dict::new(map)
    }

    #[test]
    fn dict_one_line_works() {
        let dict = dict(vec![
            ("xyz", DictVal::Expr(ex_lit_num(10))),
            ("abc", DictVal::Marker),
            ("def", DictVal::RemoveMarker),
        ]);
        let code = dict.rewrite(c()).unwrap();
        assert_eq!(code, "{abc: marker(), def: removeMarker(), xyz: 10}")
    }

    #[test]
    fn dict_multi_line_works() {
        let dict = dict(vec![
            ("xyz", DictVal::Expr(ex_lit_num(10))),
            ("abc", DictVal::Marker),
            ("def", DictVal::RemoveMarker),
        ]);
        let code = dict.rewrite(nc(1, 25)).unwrap();
        assert_eq!(code, " {\n     abc: marker(),\n     def: removeMarker(),\n     xyz: 10,\n }")
    }

    #[test]
    fn dict_multi_line_not_enough_space_works() {
        let dict = dict(vec![
            ("xyz", DictVal::Expr(ex_lit_num(10))),
            ("abc", DictVal::Marker),
            ("def", DictVal::RemoveMarker),
        ]);
        assert!(dict.rewrite(nc(1, 24)).is_none());
    }

    #[test]
    fn trap_call_one_line_works() {
        let trap_call = TrapCall::new(ex_id("target"), "trapped".to_owned());
        let code = trap_call.rewrite(c()).unwrap();
        assert_eq!(code, "target->trapped");
    }

    #[test]
    fn trap_call_multi_line_works() {
        let dict = Expr::Dict(dict(vec![
            ("xyz", DictVal::Expr(ex_lit_num(10))),
            ("abc", DictVal::Marker),
            ("def", DictVal::RemoveMarker),
        ]));
        let trap_call = TrapCall::new(dict, "trapped".to_owned());
        let code = trap_call.rewrite(nc(1, 25)).unwrap();
        assert_eq!(code, " {\n     abc: marker(),\n     def: removeMarker(),\n     xyz: 10,\n }->trapped");
    }

    #[test]
    fn block_one_expr_works() {
        let block = Expr::Block(Block::new(vec![ex_lit_num(0)]));
        let code = block.rewrite(c()).unwrap();
        assert_eq!(code, "do\n    0\nend")
    }

    #[test]
    fn block_multi_expr_works() {
        let block = Expr::Block(Block::new(vec![ex_lit_num(0), ex_lit_num(1)]));
        let code = block.rewrite(c()).unwrap();
        assert_eq!(code, "do\n    0\n    1\nend")
    }

    #[test]
    fn block_multi_expr_with_newline_works() {
        let nested_block =
            Expr::Block(Block::new(vec![ex_id("iWantSpacingAfterThisBlock")]));
        let block = Expr::Block(Block::new(vec![nested_block, ex_lit_num(0)]));
        let code = block.rewrite(c()).unwrap();
        let expected = "do
    do
        iWantSpacingAfterThisBlock
    end

    0
end";
        assert_eq!(code, expected);
    }

    #[test]
    fn func_with_no_params_works() {
        let func = Func::new(vec![], ex_lit_num(100));
        let code = func.rewrite(nc(0, 8)).unwrap();
        assert_eq!(code, "() => do\n    100\nend");
    }

    #[test]
    fn func_with_no_params_one_line_works() {
        let func = Func::new(vec![], ex_lit_num(100));
        let code = func.rewrite(nc(0, 9)).unwrap();
        assert_eq!(code, "() => 100");
    }

    #[test]
    fn func_with_one_param_works() {
        let params = vec![Param::new(tn("a"), Some(ex_lit_num(1)))];
        let func = Func::new(params, ex_lit_num(100));
        let code = func.rewrite(nc(0, 12)).unwrap();
        assert_eq!(code, "(a: 1) => do\n    100\nend");
    }

    #[test]
    fn func_with_one_param_one_line_works() {
        let params = vec![Param::new(tn("a"), Some(ex_lit_num(1)))];
        let func = Func::new(params, ex_lit_num(100));
        let code = func.rewrite(nc(0, 13)).unwrap();
        assert_eq!(code, "(a: 1) => 100");
    }

    #[test]
    fn func_with_one_param_no_default_works() {
        let params = vec![Param::new(tn("a"), None)];
        let func = Func::new(params, ex_lit_num(100));
        let code = func.rewrite(nc(0, 7)).unwrap();
        // If a function is rewritten over multiple lines, ensure it always
        // surrounds parameters with parentheses.
        assert_eq!(code, "(a) => do\n    100\nend");
    }

    #[test]
    fn func_with_one_param_no_default_one_line_works() {
        let params = vec![Param::new(tn("a"), None)];
        let func = Func::new(params, ex_lit_num(100));
        let code = func.rewrite(nc(0, 8)).unwrap();
        assert_eq!(code, "a => 100");
    }

    #[test]
    fn func_with_params_works() {
        let params = vec![
            Param::new(tn("a"), None),
            Param::new(tn("b"), Some(ex_lit_num(1))),
        ];
        let func = Func::new(params, ex_lit_num(100));
        let code = func.rewrite(nc(0, 15)).unwrap();
        assert_eq!(code, "(a, b: 1) => do\n    100\nend");
    }

    #[test]
    fn func_with_params_one_line_works() {
        let params = vec![
            Param::new(tn("a"), None),
            Param::new(tn("b"), Some(ex_lit_num(1))),
        ];
        let func = Func::new(params, ex_lit_num(100));
        let code = func.rewrite(nc(0, 16)).unwrap();
        assert_eq!(code, "(a, b: 1) => 100");
    }

    #[test]
    fn func_with_params_exceeding_desired_width_works() {
        let params = vec![
            Param::new(tn("first"), None),
            Param::new(tn("second"), Some(ex_lit_num(1))),
        ];
        let func = Func::new(params, ex_lit_num(0));
        let code = func.rewrite(nc(1, 6)).unwrap();
        assert_eq!(code, " (first, second: 1) => do\n     0\n end");
    }

    #[test]
    fn try_catch_one_line_works() {
        let tc = TryCatch::new(ex_lit_num(0), None, ex_lit_num(1));
        let code = tc.rewrite(c()).unwrap();
        assert_eq!(code, "try 0 catch 1");
    }

    #[test]
    fn try_catch_one_line_exception_works() {
        let tc =
            TryCatch::new(ex_lit_num(0), Some("ex".to_owned()), ex_lit_num(1));
        let code = tc.rewrite(c()).unwrap();
        assert_eq!(code, "try 0 catch (ex) 1");
    }

    #[test]
    fn try_catch_multi_line_works() {
        let tc = TryCatch::new(ex_id("name"), None, ex_id("name2"));
        let code = tc.rewrite(nc(1, 13)).unwrap();
        assert_eq!(code, " try do\n     name\n end catch do\n     name2\n end");
    }

    #[test]
    fn try_catch_multi_line_exception_works() {
        let tc =
            TryCatch::new(ex_id("name"), Some("ex".to_owned()), ex_id("name2"));
        let code = tc.rewrite(nc(1, 18)).unwrap();
        assert_eq!(
            code,
            " try do\n     name\n end catch (ex) do\n     name2\n end"
        );
    }

    #[test]
    fn if_one_line_no_else_works() {
        let iff = If::new(ex_id("someBool"), ex_lit_num(0), None);
        let code = iff.rewrite(c()).unwrap();
        assert_eq!(code, "if (someBool) 0");
    }

    #[test]
    fn if_one_line_else_works() {
        let iff =
            If::new(ex_id("someBool"), ex_lit_num(0), Some(ex_lit_num(1)));
        let code = iff.rewrite(c()).unwrap();
        assert_eq!(code, "if (someBool) 0 else 1");
    }

    #[test]
    fn if_multi_line_no_else_works() {
        let iff = If::new(ex_id("someBool"), ex_lit_num(1000000000000), None);
        let code = iff.rewrite(nc(4, 21)).unwrap();
        let expected = "    if (someBool) do
        1000000000000
    end";
        assert_eq!(code, expected);

        // Also check it fails if there's not enough room:
        assert!(iff.rewrite(nc(4, 20)).is_none());
    }

    #[test]
    fn if_multi_line_else_works() {
        let iff = If::new(
            ex_id("someBool"),
            ex_lit_num(1000000000000),
            Some(ex_lit_num(0)),
        );
        let code = iff.rewrite(nc(4, 21)).unwrap();
        let expected = "    if (someBool) do
        1000000000000
    end else do
        0
    end";
        assert_eq!(code, expected);

        // Also check it fails if there's not enough room:
        assert!(iff.rewrite(nc(4, 20)).is_none());
    }

    #[test]
    fn if_multi_line_cond_no_else_works() {
        let iff =
            If::new(ex_id("someLongBool"), ex_lit_num(1000000000000), None);
        let code = iff.rewrite(nc(4, 21)).unwrap();
        let expected = "    if (
        someLongBool
    ) do
        1000000000000
    end";
        assert_eq!(code, expected);

        // Also check it fails if there's not enough room:
        assert!(iff.rewrite(nc(4, 20)).is_none());
    }

    #[test]
    fn if_multi_line_cond_else_works() {
        let iff = If::new(
            ex_id("someLongBool"),
            ex_lit_num(1000000000000),
            Some(ex_lit_num(0)),
        );
        let code = iff.rewrite(nc(4, 21)).unwrap();
        let expected = "    if (
        someLongBool
    ) do
        1000000000000
    end else do
        0
    end";
        assert_eq!(code, expected);

        // Also check it fails if there's not enough room:
        assert!(iff.rewrite(nc(4, 20)).is_none());
    }

    #[test]
    fn if_nested_no_else_works() {
        let iff2 = Expr::If(Box::new(If::new(
            ex_id("reallyLong"),
            ex_lit_num(2),
            None,
        )));
        let iff1 = Expr::If(Box::new(If::new(
            ex_id("longer"),
            ex_lit_num(1),
            Some(iff2),
        )));
        let iff0 = If::new(ex_id("short"), ex_lit_num(0), Some(iff1));
        let code = iff0.rewrite(nc(4, 31)).unwrap();
        let expected = "    if (short) do
        0
    end else if (longer) do
        1
    end else if (reallyLong) do
        2
    end";
        assert_eq!(code, expected);
    }

    #[test]
    fn if_nested_no_else_works_restricted_width() {
        let iff2 = Expr::If(Box::new(If::new(
            ex_id("reallyLong"),
            ex_lit_num(2),
            None,
        )));
        let iff1 = Expr::If(Box::new(If::new(
            ex_id("longer"),
            ex_lit_num(1),
            Some(iff2),
        )));
        let iff0 = If::new(ex_id("short"), ex_lit_num(0), Some(iff1));
        let code = iff0.rewrite(nc(4, 30)).unwrap(); // Note the reduced width
        let expected = "    if (short) do
        0
    end else if (longer) do
        1
    end else if (
        reallyLong
    ) do
        2
    end";
        assert_eq!(code, expected);

        // Also check it fails if there's not enough room:
        assert!(iff0.rewrite(nc(4, 17)).is_none());
    }

    #[test]
    fn dot_call_one_line_no_args_works() {
        let target = Box::new(ex_id("value"));
        let name = FuncName::TagName(tn("someFunc"));
        let args = vec![];
        let dot_call = DotCall::new(name, target, args);
        let code = dot_call.rewrite(c()).unwrap();
        assert_eq!(code, "value.someFunc()")
    }

    #[test]
    fn dot_call_one_line_one_arg_works() {
        let target = Box::new(ex_id("value"));
        let name = FuncName::TagName(tn("someFunc"));
        let args = vec![ex_lit_num(1)];
        let dot_call = DotCall::new(name, target, args);
        let code = dot_call.rewrite(c()).unwrap();
        assert_eq!(code, "value.someFunc(1)")
    }

    #[test]
    fn dot_call_one_line_two_args_works() {
        let target = Box::new(ex_id("value"));
        let name = FuncName::TagName(tn("someFunc"));
        let args = vec![ex_lit_num(1), ex_lit_num(2)];
        let dot_call = DotCall::new(name, target, args);
        let code = dot_call.rewrite(c()).unwrap();
        assert_eq!(code, "value.someFunc(1, 2)")
    }

    fn lambda_zero() -> Expr {
        let body = ex_lit_num(100);
        Expr::Func(Box::new(Func::new(vec![], body)))
    }

    #[test]
    fn dot_call_one_line_no_args_trailing_lambda_zero_works() {
        let target = Box::new(ex_id("value"));
        let name = FuncName::TagName(tn("someFunc"));
        let args = vec![lambda_zero()];
        let dot_call = DotCall::new(name, target, args);

        let code = dot_call.rewrite(c()).unwrap();
        assert_eq!(code, "value.someFunc() () => 100");

        let code = dot_call.rewrite_inner(c(), false).unwrap();
        assert_eq!(code, "value.someFunc(() => 100)");
    }

    #[test]
    fn dot_call_one_line_one_arg_trailing_lambda_zero_works() {
        let target = Box::new(ex_id("value"));
        let name = FuncName::TagName(tn("someFunc"));
        let args = vec![ex_lit_num(1), lambda_zero()];
        let dot_call = DotCall::new(name, target, args);

        let code = dot_call.rewrite(c()).unwrap();
        assert_eq!(code, "value.someFunc(1) () => 100");

        let code = dot_call.rewrite_inner(c(), false).unwrap();
        assert_eq!(code, "value.someFunc(1, () => 100)");
    }

    #[test]
    fn dot_call_one_line_two_args_trailing_lambda_zero_works() {
        let target = Box::new(ex_id("value"));
        let name = FuncName::TagName(tn("someFunc"));
        let args = vec![ex_lit_num(1), ex_lit_num(2), lambda_zero()];
        let dot_call = DotCall::new(name, target, args);

        let code = dot_call.rewrite(c()).unwrap();
        assert_eq!(code, "value.someFunc(1, 2) () => 100");

        let code = dot_call.rewrite_inner(c(), false).unwrap();
        assert_eq!(code, "value.someFunc(1, 2, () => 100)");
    }

    // ========================================================================

    #[test]
    fn callargtype_no_args_works() {
        let cat = CallArgType::NoArgs(NoArgs);
        let layout = CallArgLayout::MultiLine;
        let lambda_pos = LambdaPos::Trailing;
        let style = CallArgStyle::new(layout, lambda_pos);
        let code = cat.rewrite(nc(1, 3), style).unwrap();
        assert_eq!(code, " ()");
    }

    #[test]
    fn dot_call_rewrite_1() {
        let target = Box::new(ex_id("value"));
        let name = FuncName::TagName(tn("func"));
        let args = vec![];
        let dot_call = DotCall::new(name, target, args);

        let code = dot_call.rewrite_inner(nc(1, 12), true).unwrap();
        assert_eq!(code, " value\n     .func()");
    }

    #[test]
    fn dot_call_rewrite_2() {
        let target = Box::new(ex_id("value"));
        let name = FuncName::TagName(tn("func"));
        let args = vec![];
        let dot_call = DotCall::new(name, target, args);

        assert!(dot_call.rewrite_inner(nc(1, 11), true).is_none());
    }

    #[test]
    fn dot_call_rewrite_3() {
        let target = Box::new(ex_id("value"));
        let name = FuncName::TagName(tn("func"));
        let args = vec![
            ex_lit_num(100),
            ex_lit_num(200),
        ];
        let dot_call = DotCall::new(name, target, args);

        let code = dot_call.rewrite_inner(nc(1, 20), true).unwrap();
        assert_eq!(code, " value\n     .func(100, 200)");

        let code = dot_call.rewrite_inner(nc(1, 19), true).unwrap();
        assert_eq!(code, " value.func(\n     100,\n     200\n )");
    }

    // =======================================================================

    fn lambda_long() -> Expr {
        let body = ex_lit_num(9999999);
        Expr::Func(Box::new(Func::new(vec![], body)))
    }

    // TODO test dot call multi-line trailing lambda
    // TODO test dot call multi-line
}
