use axon_parseast_parser as ap;
use axon_parseast_parser::Val;
use chrono::{NaiveDate, NaiveTime};
use raystack::{Number, TagName};
use std::collections::HashMap;
use std::convert::{From, TryFrom, TryInto};

#[derive(Clone, Debug, PartialEq)]
struct Def {
    pub name: TagName,
    pub val: DefValue,
}

impl Def {
    pub fn new(name: TagName, val: DefValue) -> Self {
        Self { name, val }
    }
}

impl TryFrom<&Val> for Def {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        unimplemented!()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DefValue {
    Lit(Lit),
}

#[derive(Clone, Debug, PartialEq)]
struct Param {
    pub name: TagName,
    pub default_val: Option<ParamDefaultValue>,
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
enum ParamDefaultValue {
    Lit(Lit),
    Todo,
}

impl TryFrom<&Val> for ParamDefaultValue {
    type Error = ();

    fn try_from(val: &Val) -> Result<Self, Self::Error> {
        let lit: Option<Lit> = val.try_into().ok();
        if let Some(lit) = lit {
            return Ok(Self::Lit(lit));
        };
        Err(())
    }
}

impl Param {
    pub fn new(name: TagName, default_val: Option<ParamDefaultValue>) -> Self {
        Self { name, default_val }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Lit {
    Bool(bool),
    Date(NaiveDate),
    Null,
    Num(Number),
    Str(String),
    Time(NaiveTime),
    Uri(String),
    YearMonth(YearMonth),
}

impl From<&ap::Lit> for Lit {
    fn from(lit: &ap::Lit) -> Self {
        match lit {
            ap::Lit::Bool(bool) => Lit::Bool(*bool),
            ap::Lit::Date(date) => Lit::Date(*date),
            ap::Lit::Null => Lit::Null,
            ap::Lit::Num(number) => Lit::Num(number.clone()),
            ap::Lit::Str(string) => Lit::Str(string.clone()),
            ap::Lit::Time(time) => Lit::Time(*time),
            ap::Lit::Uri(uri) => Lit::Uri(uri.clone()),
            ap::Lit::YearMonth(ym) => Lit::YearMonth(ym.into()),
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
            Val::Lit(lit) => Ok(lit.into()),
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

enum MapForTypeError {
    NotDict,
    TypeStringMismatch,
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

    fn lit_num(n: f64) -> Lit {
        Lit::Num(Number::new(n, None))
    }

    fn tn(tag_name: &str) -> TagName {
        TagName::new(tag_name.to_owned()).unwrap()
    }

    const HELLO_WORLD: &str = r###"{type:"func", params:[], body:{type:"block", exprs:[{type:"literal", val:"hello world"}]}}"###;

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
        let def_val = ParamDefaultValue::Lit(lit_num(1.0));
        let expected = Param::new(tn("ahu"), Some(def_val));
        let param: Param = val.try_into().unwrap();
        assert_eq!(param, expected);
    }

    #[test]
    fn val_to_def_literal_works() {
        let val = &ap_parse(r#"{type:"def", name:"varName", val:{type:"literal", val:1}}"#).unwrap();
        let def_val = DefValue::Lit(lit_num(1.0));
        let expected = Def::new(tn("siteId"), def_val);
        let def: Def = val.try_into().unwrap();
        assert_eq!(def, expected);
    }
}
