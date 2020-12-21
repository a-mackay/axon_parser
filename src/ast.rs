use axon_parseast_parser as ap;
use axon_parseast_parser::Val;
use chrono::{NaiveDate, NaiveTime};
use raystack::{Number, Ref, TagName};
use std::collections::HashMap;
use std::convert::{From, TryFrom, TryInto};

// TODO later
// defcomp
// qname
// _ params like run(_, _)
// symbol literals ^symbol

macro_rules! impl_try_from_val_ref_for {
    ($type_name:ty, $bin_op:expr) => {
        impl TryFrom<&Val> for $type_name {
            type Error = ();

            fn try_from(val: &Val) -> Result<Self, Self::Error> {
                let op = val_to_bin_op(val, $bin_op).map_err(|_| ())?;
                Ok(Self(op))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Add(BinOp);
impl_try_from_val_ref_for!(Add, BinOpId::Add);

#[derive(Debug, Clone, PartialEq)]
pub struct And(BinOp);
impl_try_from_val_ref_for!(And, BinOpId::And);

#[derive(Debug, Clone, PartialEq)]
pub struct Cmp(BinOp);
impl_try_from_val_ref_for!(Cmp, BinOpId::Cmp);

#[derive(Debug, Clone, PartialEq)]
pub struct Div(BinOp);
impl_try_from_val_ref_for!(Div, BinOpId::Div);

#[derive(Debug, Clone, PartialEq)]
pub struct Eq(BinOp);
impl_try_from_val_ref_for!(Eq, BinOpId::Eq);

#[derive(Debug, Clone, PartialEq)]
pub struct Gt(BinOp);
impl_try_from_val_ref_for!(Gt, BinOpId::Gt);

#[derive(Debug, Clone, PartialEq)]
pub struct Gte(BinOp);
impl_try_from_val_ref_for!(Gte, BinOpId::Gte);

#[derive(Debug, Clone, PartialEq)]
pub struct Lt(BinOp);
impl_try_from_val_ref_for!(Lt, BinOpId::Lt);

#[derive(Debug, Clone, PartialEq)]
pub struct Lte(BinOp);
impl_try_from_val_ref_for!(Lte, BinOpId::Lte);

#[derive(Debug, Clone, PartialEq)]
pub struct Mul(BinOp);
impl_try_from_val_ref_for!(Mul, BinOpId::Mul);

#[derive(Debug, Clone, PartialEq)]
pub struct Ne(BinOp);
impl_try_from_val_ref_for!(Ne, BinOpId::Ne);

#[derive(Debug, Clone, PartialEq)]
pub struct Sub(BinOp);
impl_try_from_val_ref_for!(Sub, BinOpId::Sub);

#[derive(Debug, Clone, PartialEq)]
pub struct Or(BinOp);
impl_try_from_val_ref_for!(Or, BinOpId::Or);

#[derive(Debug, Clone, PartialEq)]
pub struct BinOp {
    lhs: Expr,
    bin_op_id: BinOpId,
    rhs: Expr,
}

impl BinOp {
    pub fn new(lhs: Expr, bin_op_id: BinOpId, rhs: Expr) -> Self {
        Self { lhs, bin_op_id, rhs }
    }
}

fn val_to_bin_op(val: &Val, bin_op_id: BinOpId) -> Result<BinOp, MapForTypeError> {
    let type_str = bin_op_id.type_str();
    let hash_map = map_for_type(val, type_str)?;
    let lhs = get_val(hash_map, "lhs").expect("bin op {:?} should have 'lhs' tag");
    let rhs = get_val(hash_map, "rhs").expect("bin op {:?} should have 'rhs' tag");
    let lhs_expr = lhs.try_into().unwrap_or_else(|_| panic!("bin op {:?} 'lhs' could not be parsed as an Expr", lhs));
    let rhs_expr = rhs.try_into().unwrap_or_else(|_| panic!("bin op {:?} 'rhs' could not be parsed as an Expr", rhs));
    let bin_op = BinOp::new(lhs_expr, bin_op_id, rhs_expr);
    Ok(bin_op)
}

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
    fn type_str(&self) -> &str {
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

    fn to_symbol(&self) -> &str {
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct Neg {
    operand: Expr,
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
        let operand = get_val(hash_map, "operand").expect("neg should have 'operand' tag");
        let operand_expr = operand.try_into().expect("neg 'operand' could not be parsed as an Expr");
        Ok(Self::new(operand_expr))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TryCatch {
    try_expr: Expr,
    exception_name: Option<String>,
    catch_expr: Expr,
}

impl TryCatch {
    pub fn new(try_expr: Expr, exception_name: Option<String>, catch_expr: Expr) -> Self {
        Self { try_expr, exception_name, catch_expr }
    }
}

impl TryFrom<&Val> for TryCatch {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "try").map_err(|_| ())?;
        let try_val = get_val(hash_map, "tryExpr").expect("try should contain 'tryExpr' tag");
        let catch_val = get_val(hash_map, "catchExpr").expect("try should contain 'catchExpr' tag");
        let exc_name_val = get_val(hash_map, "errVarName");

        let try_expr = try_val.try_into().expect("try 'tryExpr' could not be parsed as an Expr");
        let catch_expr = catch_val.try_into().expect("try 'catchExpr' could not be parsed as an Expr");
        let exc_name = match exc_name_val {
            Some(Val::Lit(ap::Lit::Str(exc_name))) => Some(exc_name.to_owned()),
            None => None,
            _ => panic!("expected try 'errVarName' to be a string literal: {:?}", val)
        };

        Ok(Self::new(try_expr, exc_name, catch_expr))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct If {
    cond: Expr,
    if_expr: Expr,
    else_expr: Option<Expr>,
}

impl If {
    pub fn new(cond: Expr, if_expr: Expr, else_expr: Option<Expr>) -> Self {
        Self { cond, if_expr, else_expr }
    }
}

impl TryFrom<&Val> for If {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "if").map_err(|_| ())?;
        let cond_val = get_val(hash_map, "cond").expect("if should contain 'cond' tag");
        let if_val = get_val(hash_map, "ifExpr").expect("if should contain 'ifExpr' tag");
        let else_val = get_val(hash_map, "elseExpr");

        let cond_expr = cond_val.try_into().expect("if 'cond' could not be parsed as an Expr");
        let if_expr = if_val.try_into().expect("if 'ifExpr' could not be parsed as an Expr");
        let else_expr = match else_val {
            Some(else_val) => Some(else_val.try_into().expect("if 'elseExpr' could not be parsed as an Expr")),
            None => None,
        };

        Ok(Self::new(cond_expr, if_expr, else_expr))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrapCall {
    target: Expr,
    key: String,
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
        let args = get_vals(hash_map, "args").expect("trapCall should have 'args' tag");

        assert!(args.len() == 2, format!("trapCall 'args' list should have exactly two elements: {:?}", args));
        let target = &args[0];
        let key = &args[1];
        let target = target.try_into().expect("trapCall 'target' could not be parsed as an Expr");

        let key = match key {
            Val::Dict(key_hash_map) => {
                let key_str = get_literal_str(key_hash_map, "val").expect("trapCall key hash map should contain 'val' tag");
                key_str.to_owned()
            },
            _ => panic!("expected trapCall key Val to be a Dict")
        };

        Ok(Self::new(target, key))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DotCall {
    func_name: String,
    args: Vec<Expr>,
}

impl DotCall {
    pub fn new(func_name: String, args: Vec<Expr>) -> Self {
        Self { func_name, args }
    }
}

impl TryFrom<&Val> for DotCall {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "dotCall").map_err(|_| ())?;
        let target = get_val(hash_map, "target").expect("dotCall should have 'target' tag");
        match target {
            Val::Dict(target_hash_map) => {
                let func_name = get_literal_str(target_hash_map, "name").expect("dotCall 'target' should have 'name' string tag");
                let func_name = func_name.to_owned();
                let args = get_vals(hash_map, "args").expect("dotCall should have 'args' tag");

                let mut exprs = vec![];

                for arg in args {
                    let expr = arg.try_into().unwrap_or_else(|_| panic!("dotCall arg could not be parsed as an Expr: {:?}", arg));
                    exprs.push(expr);
                }

                Ok(Self::new(func_name, exprs))
            },
            _ => panic!("expected dotCall 'target' to be a Dict"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    func_name: String,
    args: Vec<Expr>,
}

impl Call {
    pub fn new(func_name: String, args: Vec<Expr>) -> Self {
        Self { func_name, args }
    }
}

impl TryFrom<&Val> for Call {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "call").map_err(|_| ())?;
        let target = get_val(hash_map, "target").expect("call should have 'target' tag");
        match target {
            Val::Dict(target_hash_map) => {
                let func_name = get_literal_str(target_hash_map, "name").expect("call 'target' should have 'name' string tag");
                let func_name = func_name.to_owned();
                let args = get_vals(hash_map, "args").expect("call should have 'args' tag");

                let mut exprs = vec![];

                for arg in args {
                    let expr = arg.try_into().unwrap_or_else(|_| panic!("call arg could not be parsed as an Expr: {:?}", arg));
                    exprs.push(expr);
                }

                Ok(Self::new(func_name, exprs))
            },
            _ => panic!("expected call 'target' to be a Dict"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Not {
    operand: Expr,
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
        let operand = get_val(hash_map, "operand").expect("not should have 'operand' tag");
        let operand_expr = operand.try_into().expect("not 'operand' could not be parsed as an Expr");
        Ok(Self::new(operand_expr))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Range {
    start: Expr,
    end: Expr,
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
        let start = get_val(hash_map, "start").expect("range should have 'start' tag");
        let end = get_val(hash_map, "end").expect("range should have 'end' tag");
        let start_expr = start.try_into().expect("range 'start' could not be parsed as an Expr");
        let end_expr = end.try_into().expect("range 'end' could not be parsed as an Expr");
        Ok(Self::new(start_expr, end_expr))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Func {
    params: Vec<Param>,
    body: Expr,
}

impl Func {
    pub fn new(params: Vec<Param>, body: Expr) -> Self {
        Self { params, body }
    }
}

impl TryFrom<&Val> for Func {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let hash_map = map_for_type(val, "func").map_err(|_| ())?;
        let param_vals = get_vals(hash_map, "params").expect("func should contain 'params' tag");

        let mut params = vec![];
        for param_val in param_vals {
            let param = param_val.try_into().expect("func param val could not be converted to a Param");
            params.push(param);
        }

        let body = get_val(hash_map, "body").expect("func should have a 'body' tag");
        let body_expr = body.try_into().expect("func body val could not be converted to a Expr");

        Ok(Self::new(params, body_expr))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    exprs: Vec<Expr>,
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

#[derive(Clone, Debug, PartialEq)]
pub enum DictVal {
    Expr(Expr),
    Marker,
    RemoveMarker,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Throw {
    expr: Expr,
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
    Id(TagName),
    If(Box<If>),
    List(List),
    Lit(Lit),
    Neg(Box<Neg>),
    Range(Box<Range>),
    Throw(Box<Throw>),
    TrapCall(Box<TrapCall>),
    TryCatch(Box<TryCatch>),
}

impl TryFrom<&Val> for Expr {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let lit: Option<Lit> = val.try_into().ok();
        if let Some(lit) = lit {
            return Ok(Expr::Lit(lit));
        };

        if let Some(tag_name) = var_val_to_tag_name(val) {
            return Ok(Expr::Id(tag_name));
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
            return Ok(Expr::Call(call))
        }

        let dot_call: Option<DotCall> = val.try_into().ok();
        if let Some(dot_call) = dot_call {
            return Ok(Expr::DotCall(dot_call))
        }

        let trap_call: Option<TrapCall> = val.try_into().ok();
        if let Some(trap_call) = trap_call {
            return Ok(Expr::TrapCall(Box::new(trap_call)))
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
            return Ok(Expr::Gt(Box::new(gt)))
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

        let neg: Option<Neg> = val.try_into().ok();
        if let Some(neg) = neg {
            return Ok(Expr::Neg(Box::new(neg)));
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
        Self { name, expr: Box::new(expr) }
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
        Self { name, expr: Box::new(expr) }
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
pub enum Lit {
    Bool(bool),
    Date(NaiveDate),
    Null,
    Num(Number),
    Ref(Ref),
    Str(String),
    Time(NaiveTime),
    Uri(String),
    YearMonth(YearMonth),
}

pub enum ConvertLitError {
    IsDictMarker,
    IsDictRemoveMarker,
}

impl TryFrom<&ap::Lit> for Lit {
    type Error = ConvertLitError;

    fn try_from(lit: &ap::Lit) -> Result<Self, Self::Error> {
        match lit {
            ap::Lit::Bool(bool) => Ok(Lit::Bool(*bool)),
            ap::Lit::Date(date) => Ok(Lit::Date(*date)),
            ap::Lit::DictMarker => Err(ConvertLitError::IsDictMarker),
            ap::Lit::DictRemoveMarker => {
                Err(ConvertLitError::IsDictRemoveMarker)
            }
            ap::Lit::Null => Ok(Lit::Null),
            ap::Lit::Num(number) => Ok(Lit::Num(number.clone())),
            ap::Lit::Ref(reff) => Ok(Lit::Ref(reff.clone())),
            ap::Lit::Str(string) => Ok(Lit::Str(string.clone())),
            ap::Lit::Time(time) => Ok(Lit::Time(*time)),
            ap::Lit::Uri(uri) => Ok(Lit::Uri(uri.clone())),
            ap::Lit::YearMonth(ym) => Ok(Lit::YearMonth(ym.into())),
        }
    }
}

impl TryFrom<&Val> for Lit {
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

#[derive(Clone, Debug, PartialEq)]
pub struct YearMonth {
    year: u32,
    month: Month,
}

impl YearMonth {
    pub fn new(year: u32, month: Month) -> Self {
        Self { year, month }
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
    use raystack::Number;
    use std::convert::TryInto;

    fn lit_str(s: &str) -> Lit {
        Lit::Str(s.to_owned())
    }

    fn lit_num(n: f64) -> Lit {
        Lit::Num(Number::new(n, None))
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
        let expected = Lit::Str("hello".to_owned());
        let lit: Lit = val.try_into().unwrap();
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
        let val = &ap_parse(r#"{type:"block", exprs:[{type:"literal", val:"hello"}]}"#).unwrap();
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
        let val = &ap_parse(r#"{type:"not", operand:{type:"literal", val:1}}"#).unwrap();
        let operand = Expr::Lit(lit_num(1.0));
        let expected = Not::new(operand);
        let not: Not = val.try_into().unwrap();
        assert_eq!(not, expected);
    }

    #[test]
    fn val_to_simple_call_works() {
        let val = &ap_parse(r#"{type:"call", target:{type:"var", name:"readAll"}, args:[{type:"literal", val:1}]}"#).unwrap();
        let func_name = "readAll".to_owned();
        let arg = Expr::Lit(lit_num(1.0));
        let args = vec![arg];
        let expected = Call::new(func_name, args);
        let call: Call = val.try_into().unwrap();
        assert_eq!(call, expected);
    }

    #[test]
    fn val_to_simple_dot_call_works() {
        let val = &ap_parse(r#"{type:"dotCall", target:{type:"var", name:"parseNumber"}, args:[{type:"literal", val:1}]}"#).unwrap();
        let func_name = "parseNumber".to_owned();
        let arg = Expr::Lit(lit_num(1.0));
        let args = vec![arg];
        let expected = DotCall::new(func_name, args);
        let dot_call: DotCall = val.try_into().unwrap();
        assert_eq!(dot_call, expected);
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
        let lhs = Expr::Id(tn("a"));
        let rhs = Expr::Id(tn("b"));
        let expected = And(BinOp::new(lhs, BinOpId::And, rhs));
        let and: And = val.try_into().unwrap();
        assert_eq!(and, expected);
    }

    #[test]
    fn val_to_or_works() {
        let val = &ap_parse(r#"{type:"or", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(tn("a"));
        let rhs = Expr::Id(tn("b"));
        let expected = Or(BinOp::new(lhs, BinOpId::Or, rhs));
        let or: Or = val.try_into().unwrap();
        assert_eq!(or, expected);
    }

    #[test]
    fn val_to_add_works() {
        let val = &ap_parse(r#"{type:"add", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(tn("a"));
        let rhs = Expr::Id(tn("b"));
        let expected = Add(BinOp::new(lhs, BinOpId::Add, rhs));
        let add: Add = val.try_into().unwrap();
        assert_eq!(add, expected);
    }

    #[test]
    fn val_to_cmp_works() {
        let val = &ap_parse(r#"{type:"cmp", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(tn("a"));
        let rhs = Expr::Id(tn("b"));
        let expected = Cmp(BinOp::new(lhs, BinOpId::Cmp, rhs));
        let cmp: Cmp = val.try_into().unwrap();
        assert_eq!(cmp, expected);
    }

    #[test]
    fn val_to_div_works() {
        let val = &ap_parse(r#"{type:"div", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(tn("a"));
        let rhs = Expr::Id(tn("b"));
        let expected = Div(BinOp::new(lhs, BinOpId::Div, rhs));
        let div: Div = val.try_into().unwrap();
        assert_eq!(div, expected);
    }

    #[test]
    fn val_to_eq_works() {
        let val = &ap_parse(r#"{type:"eq", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(tn("a"));
        let rhs = Expr::Id(tn("b"));
        let expected = Eq(BinOp::new(lhs, BinOpId::Eq, rhs));
        let eq: Eq = val.try_into().unwrap();
        assert_eq!(eq, expected);
    }

    #[test]
    fn val_to_gt_works() {
        let val = &ap_parse(r#"{type:"gt", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(tn("a"));
        let rhs = Expr::Id(tn("b"));
        let expected = Gt(BinOp::new(lhs, BinOpId::Gt, rhs));
        let gt: Gt = val.try_into().unwrap();
        assert_eq!(gt, expected);
    }

    #[test]
    fn val_to_gte_works() {
        let val = &ap_parse(r#"{type:"ge", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(tn("a"));
        let rhs = Expr::Id(tn("b"));
        let expected = Gte(BinOp::new(lhs, BinOpId::Gte, rhs));
        let gte: Gte = val.try_into().unwrap();
        assert_eq!(gte, expected);
    }

    #[test]
    fn val_to_lt_works() {
        let val = &ap_parse(r#"{type:"lt", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(tn("a"));
        let rhs = Expr::Id(tn("b"));
        let expected = Lt(BinOp::new(lhs, BinOpId::Lt, rhs));
        let lt: Lt = val.try_into().unwrap();
        assert_eq!(lt, expected);
    }

    #[test]
    fn val_to_lte_works() {
        let val = &ap_parse(r#"{type:"le", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(tn("a"));
        let rhs = Expr::Id(tn("b"));
        let expected = Lte(BinOp::new(lhs, BinOpId::Lte, rhs));
        let lte: Lte = val.try_into().unwrap();
        assert_eq!(lte, expected);
    }

    #[test]
    fn val_to_mul_works() {
        let val = &ap_parse(r#"{type:"mul", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(tn("a"));
        let rhs = Expr::Id(tn("b"));
        let expected = Mul(BinOp::new(lhs, BinOpId::Mul, rhs));
        let mul: Mul = val.try_into().unwrap();
        assert_eq!(mul, expected);
    }

    #[test]
    fn val_to_ne_works() {
        let val = &ap_parse(r#"{type:"ne", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(tn("a"));
        let rhs = Expr::Id(tn("b"));
        let expected = Ne(BinOp::new(lhs, BinOpId::Ne, rhs));
        let ne: Ne = val.try_into().unwrap();
        assert_eq!(ne, expected);
    }

    #[test]
    fn val_to_sub_works() {
        let val = &ap_parse(r#"{type:"sub", lhs:{type:"var", name:"a"}, rhs:{type:"var", name:"b"}}"#).unwrap();
        let lhs = Expr::Id(tn("a"));
        let rhs = Expr::Id(tn("b"));
        let expected = Sub(BinOp::new(lhs, BinOpId::Sub, rhs));
        let sub: Sub = val.try_into().unwrap();
        assert_eq!(sub, expected);
    }

    #[test]
    fn val_to_if_no_else_works() {
        let val = &ap_parse(r#"{type:"if", cond:{type:"var", name:"a"}, ifExpr:{type:"var", name:"b"}}"#).unwrap();
        let cond = Expr::Id(tn("a"));
        let if_expr = Expr::Id(tn("b"));
        let else_expr = None;
        let expected = If::new(cond, if_expr, else_expr);
        let iff: If = val.try_into().unwrap();
        assert_eq!(iff, expected);
    }

    #[test]
    fn val_to_if_with_else_works() {
        let val = &ap_parse(r#"{type:"if", cond:{type:"var", name:"a"}, ifExpr:{type:"var", name:"b"}, elseExpr:{type:"var", name:"c"}}"#).unwrap();
        let cond = Expr::Id(tn("a"));
        let if_expr = Expr::Id(tn("b"));
        let else_expr = Some(Expr::Id(tn("c")));
        let expected = If::new(cond, if_expr, else_expr);
        let iff: If = val.try_into().unwrap();
        assert_eq!(iff, expected);
    }

    #[test]
    fn val_to_try_catch_no_exc_name_works() {
        let val = &ap_parse(r#"{type:"try", tryExpr:{type:"var", name:"a"}, catchExpr:{type:"var", name:"b"}}"#).unwrap();
        let try_expr = Expr::Id(tn("a"));
        let catch_expr = Expr::Id(tn("b"));
        let exc_name = None;
        let expected = TryCatch::new(try_expr, exc_name, catch_expr);
        let try_catch: TryCatch = val.try_into().unwrap();
        assert_eq!(try_catch, expected);
    }

    #[test]
    fn val_to_try_catch_with_exc_name_works() {
        let val = &ap_parse(r#"{type:"try", tryExpr:{type:"var", name:"a"}, errVarName:"ex", catchExpr:{type:"var", name:"b"}}"#).unwrap();
        let try_expr = Expr::Id(tn("a"));
        let catch_expr = Expr::Id(tn("b"));
        let exc_name = Some("ex".to_owned());
        let expected = TryCatch::new(try_expr, exc_name, catch_expr);
        let try_catch: TryCatch = val.try_into().unwrap();
        assert_eq!(try_catch, expected);
    }

    #[test]
    fn val_to_neg_works() {
        let val = &ap_parse(r#"{type:"neg", operand:{type:"var", name:"a"}}"#).unwrap();
        let operand = Expr::Id(tn("a"));
        let expected = Neg::new(operand);
        let neg: Neg = val.try_into().unwrap();
        assert_eq!(neg, expected);
    }

    #[test]
    fn old_chart_demo_works() {
        let val = &ap_parse(include_str!("../tests/old_chart_demo.txt")).unwrap();
        let func: Func = val.try_into().unwrap();
        println!("{:#?}", func);
    }

    #[test]
    fn old_chart_demo_kwh_works() {
        let val = &ap_parse(include_str!("../tests/old_chart_demo_kwh.txt")).unwrap();
        let func: Func = val.try_into().unwrap();
        println!("{:#?}", func);
    }
}