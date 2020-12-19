use axon_parseast_parser as ap;
use axon_parseast_parser::Val;
use chrono::{NaiveDate, NaiveTime};
use raystack::{Number, TagName};
use std::collections::HashMap;
use std::convert::{From, TryFrom};

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
        match val {
            Val::Dict(hash_map) => {
                if type_str(hash_map) == "literal" {
                    let val = get_val(hash_map, "val")
                        .expect("type 'literal' should have 'val' tag");
                    match val {
                        Val::Lit(lit) => Ok(lit.into()),
                        _ => panic!(
                            "expected type 'literal' 'val' tag to be a literal"
                        ),
                    }
                } else {
                    Err(())
                }
            }
            _ => Err(()),
        }
    }
}

fn get_val<'a, 'b>(
    hash_map: &'a HashMap<TagName, Box<Val>>,
    tag_name: &'b str,
) -> Option<&'a Val> {
    let tag_name = TagName::new(tag_name.into()).unwrap_or_else(|| {
        panic!("expected '{}' to be a valid tag name", tag_name)
    });
    hash_map.get(&tag_name).map(|val| val.as_ref())
}

fn type_str(hash_map: &HashMap<TagName, Box<Val>>) -> &str {
    let tag_name = TagName::new("type".into())
        .expect("expected 'type' to be a valid tag name");
    let val = hash_map
        .get(&tag_name)
        .expect("expected every dict to contain the 'type' tag");
    match val.as_ref() {
        Val::Lit(ap::Lit::Str(s)) => s,
        _ => panic!("expected the 'type' tag's value to be a literal string"),
    }
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
    use std::convert::TryInto;

    const HELLO_WORLD: &str = r###"{type:"func", params:[], body:{type:"block", exprs:[{type:"literal", val:"hello world"}]}}"###;

    #[test]
    fn val_to_lit_works() {
        let val = &ap_parse(r#"{type:"literal", val:"hello"}"#).unwrap();
        let expected = Lit::Str("hello".to_owned());
        let lit: Lit = val.try_into().unwrap();
        assert_eq!(lit, expected)
    }
}
