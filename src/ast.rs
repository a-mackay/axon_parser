use axon_parseast_parser as ap;
use axon_parseast_parser::Val;
use chrono::{NaiveDate, NaiveTime};
use raystack_core::{Number, Qname, Ref, Symbol, TagName};
use std::collections::HashMap;
use std::convert::{From, TryFrom, TryInto};

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

// does my child need parenthesis?
// 1. if child is higher, no
// 2. if child is lower, yes
// 3. if child is equal:
//    if i have no associativity, yes
//    if child is on the same side as my associativity, no
//    if child is on the wrong side as my associativity, yes

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

#[derive(Debug, Clone, PartialEq)]
pub struct Add(pub BinOp);
impl_try_from_val_ref_for!(Add, BinOpId::Add);

#[derive(Debug, Clone, PartialEq)]
pub struct And(pub BinOp);
impl_try_from_val_ref_for!(And, BinOpId::And);

#[derive(Debug, Clone, PartialEq)]
pub struct Cmp(pub BinOp);
impl_try_from_val_ref_for!(Cmp, BinOpId::Cmp);

#[derive(Debug, Clone, PartialEq)]
pub struct Div(pub BinOp);
impl_try_from_val_ref_for!(Div, BinOpId::Div);

#[derive(Debug, Clone, PartialEq)]
pub struct Eq(pub BinOp);
impl_try_from_val_ref_for!(Eq, BinOpId::Eq);

#[derive(Debug, Clone, PartialEq)]
pub struct Gt(pub BinOp);
impl_try_from_val_ref_for!(Gt, BinOpId::Gt);

#[derive(Debug, Clone, PartialEq)]
pub struct Gte(pub BinOp);
impl_try_from_val_ref_for!(Gte, BinOpId::Gte);

#[derive(Debug, Clone, PartialEq)]
pub struct Lt(pub BinOp);
impl_try_from_val_ref_for!(Lt, BinOpId::Lt);

#[derive(Debug, Clone, PartialEq)]
pub struct Lte(pub BinOp);
impl_try_from_val_ref_for!(Lte, BinOpId::Lte);

#[derive(Debug, Clone, PartialEq)]
pub struct Mul(pub BinOp);
impl_try_from_val_ref_for!(Mul, BinOpId::Mul);

#[derive(Debug, Clone, PartialEq)]
pub struct Ne(pub BinOp);
impl_try_from_val_ref_for!(Ne, BinOpId::Ne);

#[derive(Debug, Clone, PartialEq)]
pub struct Sub(pub BinOp);
impl_try_from_val_ref_for!(Sub, BinOpId::Sub);

#[derive(Debug, Clone, PartialEq)]
pub struct Or(pub BinOp);
impl_try_from_val_ref_for!(Or, BinOpId::Or);

#[derive(Debug, Clone, PartialEq)]
pub struct BinOp {
    pub lhs: Expr,
    pub bin_op_id: BinOpId,
    pub rhs: Expr,
}

impl BinOp {
    pub fn new(lhs: Expr, bin_op_id: BinOpId, rhs: Expr) -> Self {
        Self {
            lhs,
            bin_op_id,
            rhs,
        }
    }

    /// Returns an int representing how high the operator's precendence is,
    /// where 2 is the highest precedence for a binary operation.
    pub fn precedence(&self) -> usize {
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
    pub fn precedence(&self) -> usize {
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
            Self::Add => Some(Associativity::Left),
            Self::And => Some(Associativity::Right), // Based on the parsed AST.
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

#[derive(Debug, Clone, PartialEq)]
pub struct Neg {
    pub operand: Expr,
}

impl Neg {
    pub fn new(operand: Expr) -> Neg {
        Self { operand }
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

#[derive(Debug, Clone, PartialEq)]
pub struct TryCatch {
    pub try_expr: Expr,
    pub exception_name: Option<String>,
    pub catch_expr: Expr,
}

impl TryCatch {
    pub fn new(
        try_expr: Expr,
        exception_name: Option<String>,
        catch_expr: Expr,
    ) -> Self {
        Self {
            try_expr,
            exception_name,
            catch_expr,
        }
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
#[derive(Debug, Clone, PartialEq)]
pub struct FlatIf {
    pub cond_exprs: Vec<ConditionalExpr>,
    pub else_expr: Option<Expr>,
}

impl FlatIf {
    pub fn new(
        cond_exprs: Vec<ConditionalExpr>,
        else_expr: Option<Expr>,
    ) -> Self {
        Self {
            cond_exprs,
            else_expr,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConditionalExpr {
    /// The conditional expression, for example x == true
    pub cond: Expr,
    /// The expression that gets executed if the condition is true
    pub expr: Expr,
}

impl ConditionalExpr {
    pub fn new(cond: Expr, expr: Expr) -> Self {
        Self { cond, expr }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct If {
    pub cond: Expr,
    pub if_expr: Expr,
    pub else_expr: Option<Expr>,
}

impl If {
    pub fn new(cond: Expr, if_expr: Expr, else_expr: Option<Expr>) -> Self {
        Self {
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

#[derive(Debug, Clone, PartialEq)]
pub struct TrapCall {
    pub target: Expr,
    pub key: String,
}

impl TrapCall {
    pub fn new(target: Expr, key: String) -> Self {
        Self { target, key }
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

/// 2 or more DotCalls chained together, like a.b().c()
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DotCallsChain {
    /// The first argument (at the start of the chain). This cannot be a DotCall.
    pub(crate) target: Expr,
    /// The calls in the middle of the chain. Will be at least one element.
    pub(crate) chain: Vec<ChainedDotCall>,
    /// The final call in the chain.
    pub(crate) last: ChainedDotCall,
}

/// Only used to build a DotCallsChain
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BuildDotCallsChain {
    pub(crate) target: Option<Expr>,
    pub(crate) chain: Vec<ChainedDotCall>,
}

impl BuildDotCallsChain {
    fn build(self) -> Option<DotCallsChain> {
        if self.chain.len() < 2 {
            // If there is 1 element in the chain, this is just a regular
            // DotCall on a non-DotCall target.
            None
        } else {
            // There are at least 2 elements in the chain.
            let mut new_chain = self.chain.clone();
            let last_index = new_chain.len() - 1;
            let last = new_chain.remove(last_index);
            Some(DotCallsChain {
                target: self.target?,
                chain: new_chain,
                last,
            })
        }
    }

    fn new() -> Self {
        Self {
            target: None,
            chain: vec![],
        }
    }

    fn add(&self, cdc: ChainedDotCall) -> Self {
        let mut new_chain = self.chain.clone();
        new_chain.insert(0, cdc);
        Self {
            target: self.target.clone(),
            chain: new_chain,
        }
    }

    fn set_target(&self, target: Expr) -> Self {
        Self {
            target: Some(target),
            chain: self.chain.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ChainedDotCall {
    pub(crate) func_name: FuncName,
    pub(crate) args: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DotCall {
    pub func_name: FuncName,
    /// The first argument (at the front of the dot).
    pub target: Box<Expr>,
    /// Arguments after the dot.
    pub args: Vec<Expr>,
}

impl DotCall {
    pub fn new(
        func_name: FuncName,
        target: Box<Expr>,
        args: Vec<Expr>,
    ) -> Self {
        Self {
            func_name,
            target,
            args,
        }
    }

    pub(crate) fn to_chain(&self) -> Option<DotCallsChain> {
        self.to_chain_inner(BuildDotCallsChain::new()).build()
    }

    fn to_chain_inner(&self, build: BuildDotCallsChain) -> BuildDotCallsChain {
        let cdc = self.to_chained_dot_call();
        let build = build.add(cdc);

        match self.target.as_ref() {
            Expr::DotCall(dot_call) => dot_call.to_chain_inner(build),
            _ => {
                let initial_target = self.target.as_ref().clone();
                build.set_target(initial_target)
            }
        }
    }

    fn to_chained_dot_call(&self) -> ChainedDotCall {
        ChainedDotCall {
            func_name: self.func_name.clone(),
            args: self.args.clone(),
        }
    }

    pub fn has_lambda_last_arg(&self) -> bool {
        if !self.args.is_empty() {
            let last_arg = self.args.last().unwrap();
            matches!(last_arg, Expr::Func(_))
        } else {
            false
        }
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

#[derive(Debug, Clone, PartialEq)]
pub enum PartialCallArgument {
    Expr(Expr),
    Placeholder,
}

impl PartialCallArgument {
    fn _is_func(&self) -> bool {
        // TODO
        match self {
            Self::Expr(expr) => expr.is_func(),
            Self::Placeholder => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PartialCall {
    pub target: CallTarget,
    pub args: Vec<PartialCallArgument>,
}

impl PartialCall {
    pub fn new(target: CallTarget, args: Vec<PartialCallArgument>) -> Self {
        Self { target, args }
    }

    pub fn has_lambda_last_arg(&self) -> bool {
        if !self.args.is_empty() {
            let last_arg = self.args.last().unwrap();
            matches!(last_arg, PartialCallArgument::Expr(Expr::Func(_)))
        } else {
            false
        }
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

                if let Some(func_name) =
                    get_literal_str(target_hash_map, "name")
                {
                    let func_name = func_name.to_owned();

                    if let Some(func_name) = TagName::new(func_name.clone()) {
                        let func_name = FuncName::TagName(func_name);
                        let call_target = CallTarget::FuncName(func_name);
                        Ok(Self::new(call_target, exprs))
                    } else {
                        // We assume it's a qname:
                        let qname = Qname::new(func_name);
                        let func_name = FuncName::Qname(qname);
                        let call_target = CallTarget::FuncName(func_name);
                        Ok(Self::new(call_target, exprs))
                    }
                } else {
                    let target_expr: Expr = target.try_into().unwrap_or_else(|_| {
                        panic!(
                            "partialCall target could not be parsed as an Expr: {:?}",
                            target
                        )
                    });
                    let call_target = CallTarget::Expr(Box::new(target_expr));

                    Ok(Self::new(call_target, exprs))
                }
            }
            _ => panic!("expected partialCall 'target' to be a Dict"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub target: CallTarget,
    pub args: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CallTarget {
    Expr(Box<Expr>),
    FuncName(FuncName),
}

impl Call {
    pub fn new(target: CallTarget, args: Vec<Expr>) -> Self {
        Self { target, args }
    }

    pub fn has_lambda_last_arg(&self) -> bool {
        if !self.args.is_empty() {
            let last_arg = self.args.last().unwrap();
            matches!(last_arg, Expr::Func(_))
        } else {
            false
        }
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

#[derive(Debug, Clone, PartialEq)]
pub struct Not {
    pub operand: Expr,
}

impl Not {
    pub fn new(operand: Expr) -> Self {
        Self { operand }
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

#[derive(Debug, Clone, PartialEq)]
pub struct Range {
    pub start: Expr,
    pub end: Expr,
}

impl Range {
    pub fn new(start: Expr, end: Expr) -> Self {
        Self { start, end }
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

#[derive(Debug, Clone, PartialEq)]
pub struct Func {
    pub params: Vec<Param>,
    pub body: Expr,
}

impl Func {
    pub fn new(params: Vec<Param>, body: Expr) -> Self {
        Self { params, body }
    }

    pub fn blockify(mut self) -> Self {
        self.body = self.body.blockify();
        self
    }

    pub(crate) fn block_body(&self) -> Option<Block> {
        match &self.body {
            Expr::Block(block) => Some(block.clone()),
            _ => None,
        }
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

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub exprs: Vec<Expr>,
}

impl Block {
    pub fn new(exprs: Vec<Expr>) -> Self {
        Self { exprs }
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

#[derive(Debug, Clone, PartialEq)]
pub struct Dict {
    pub map: HashMap<TagName, DictVal>,
}

impl Dict {
    pub fn new(map: HashMap<TagName, DictVal>) -> Self {
        Self { map }
    }
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

#[derive(Debug, Clone, PartialEq)]
pub struct Return {
    pub expr: Expr,
}

impl Return {
    pub fn new(expr: Expr) -> Self {
        Self { expr }
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

#[derive(Debug, Clone, PartialEq)]
pub struct Throw {
    pub expr: Expr,
}

impl Throw {
    pub fn new(expr: Expr) -> Self {
        Self { expr }
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

#[derive(Debug, Clone, PartialEq)]
pub struct List {
    pub vals: Vec<Expr>,
}

impl List {
    pub fn new(vals: Vec<Expr>) -> Self {
        Self { vals }
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

#[derive(Debug, Clone, PartialEq)]
pub struct Id {
    name: TagName,
}

impl Id {
    pub fn new(name: TagName) -> Self {
        Self { name }
    }

    pub fn name(&self) -> &TagName {
        &self.name
    }

    pub fn to_axon_code(&self) -> String {
        self.name().to_string()
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
    Lit(Lit), //
    Neg(Box<Neg>),
    Not(Box<Not>),
    PartialCall(PartialCall),
    Range(Box<Range>),
    Return(Box<Return>),
    Throw(Box<Throw>),
    TrapCall(Box<TrapCall>),
    TryCatch(Box<TryCatch>),
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

    /// May return an int representing how high the expression's precendence is,
    /// where 1 is the highest precedence.
    pub fn precedence(&self) -> Option<usize> {
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
        matches!(
            self,
            Self::Add(_)
                | Self::And(_)
                | Self::Cmp(_)
                | Self::Div(_)
                | Self::Eq(_)
                | Self::Gt(_)
                | Self::Gte(_)
                | Self::Lt(_)
                | Self::Lte(_)
                | Self::Mul(_)
                | Self::Ne(_)
                | Self::Or(_)
                | Self::Sub(_)
        )
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

#[derive(Clone, Debug, PartialEq)]
pub struct Assign {
    /// lhs
    pub name: TagName,
    /// rhs
    pub expr: Box<Expr>,
}

impl Assign {
    pub fn new(name: TagName, expr: Expr) -> Self {
        Self {
            name,
            expr: Box::new(expr),
        }
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

#[derive(Clone, Debug, PartialEq)]
pub struct Def {
    pub name: TagName,
    pub expr: Box<Expr>,
}

impl Def {
    pub fn new(name: TagName, expr: Expr) -> Self {
        Self {
            name,
            expr: Box::new(expr),
        }
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

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: TagName,
    pub default: Option<Expr>,
}

impl Param {
    pub fn new(name: TagName, default: Option<Expr>) -> Self {
        Self { name, default }
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

#[derive(Clone, Debug, PartialEq)]
pub struct Lit {
    lit: LitInner,
}

impl Lit {
    pub fn new(lit: LitInner) -> Self {
        Self { lit }
    }

    pub fn lit(&self) -> &LitInner {
        &self.lit
    }

    pub fn to_axon_code(&self) -> String {
        self.lit().to_axon_code()
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
        let target = CallTarget::FuncName(func_name);
        let null = PartialCallArgument::Placeholder;
        let arg2 = PartialCallArgument::Expr(Expr::Lit(lit_num(1.0)));
        let expected = PartialCall::new(target, vec![null, arg2]);

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

    #[test]
    fn year_month_works() {
        let ym = YearMonth::new(2020, Month::Jan);
        assert_eq!(ym.to_axon_code(), "2020-01");
        let ym = YearMonth::new(2020, Month::Dec);
        assert_eq!(ym.to_axon_code(), "2020-12");
    }
}
