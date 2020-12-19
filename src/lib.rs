#[macro_use]
extern crate lalrpop_util;

use chrono::{NaiveDate, NaiveTime};
use raystack::{Number, TagName};
use std::collections::HashMap;

lalrpop_mod!(pub grammar); // synthesized by LALRPOP

/// Parse the output of `toAxonCode(parseAst( ... ))` and return a `Val`.
pub fn parse(axon: &str) -> Result<Val, impl std::error::Error + '_> {
    let parser = grammar::ValParser::new();
    parser.parse(axon)
}

#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Dict(HashMap<TagName, Box<Val>>),
    List(Vec<Val>),
    Lit(Lit),
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

impl Month {
    /// Convert from a number between 1 and 12 inclusive.
    fn from_int(int: u32) -> Option<Self> {
        match int {
            1 => Some(Month::Jan),
            2 => Some(Month::Feb),
            3 => Some(Month::Mar),
            4 => Some(Month::Apr),
            5 => Some(Month::May),
            6 => Some(Month::Jun),
            7 => Some(Month::Jul),
            8 => Some(Month::Aug),
            9 => Some(Month::Sep),
            10 => Some(Month::Oct),
            11 => Some(Month::Nov),
            12 => Some(Month::Dec),
            _ => None,
        }
    }
}

#[cfg(test)]
mod test {
    use super::grammar;
    use super::{Lit, Month, Val, YearMonth};
    use chrono::{NaiveDate, NaiveTime};
    use raystack::{Number, TagName};
    use std::collections::HashMap;

    const HELLO_WORLD: &str = r###"{type:"func", params:[], body:{type:"block", exprs:[{type:"literal", val:"hello world"}]}}"###;
    const AHU_TEMP_DIFF: &str = include_str!("../test_input/ahu_temp_diff.txt");
    const AHU_FUNC: &str = include_str!("../test_input/ahu_func.txt");
    const EQUIP_FUNC: &str = include_str!("../test_input/equip_func.txt");
    const OLD_CHART_DEMO: &str =
        include_str!("../test_input/old_chart_demo.txt");

    #[test]
    fn it_works() {
        assert_eq!(1 + 1, 2);
    }

    #[test]
    fn time_parser_works() {
        let p = grammar::TimeParser::new();
        assert_eq!(
            p.parse("12:34:56").unwrap(),
            NaiveTime::from_hms(12, 34, 56)
        );
        assert_eq!(p.parse("12:34").unwrap(), NaiveTime::from_hms(12, 34, 0));
    }

    #[test]
    fn year_month_parser_works() {
        let p = grammar::YearMonthParser::new();
        assert_eq!(
            p.parse("2020-12").unwrap(),
            YearMonth::new(2020, Month::Dec)
        )
    }

    #[test]
    fn date_parser_works() {
        let p = grammar::DateParser::new();
        assert_eq!(
            p.parse("2020-12-01").unwrap(),
            NaiveDate::from_ymd(2020, 12, 1)
        )
    }

    #[test]
    fn uri_parser_works() {
        let p = grammar::UriParser::new();
        assert_eq!(
            p.parse(r"`http://www.google.com/search?q=hello&q2=world`")
                .unwrap(),
            "http://www.google.com/search?q=hello&q2=world".to_owned()
        );
    }

    #[test]
    fn str_parser_works() {
        let p = grammar::StrParser::new();
        assert_eq!(
            p.parse(r#""hello world""#).unwrap(),
            "hello world".to_owned()
        );
        assert_eq!(p.parse(r#""\n""#).unwrap(), "\n".to_owned());
        assert_eq!(p.parse(r#""\t""#).unwrap(), "\t".to_owned());
        assert_eq!(p.parse(r#""\\""#).unwrap(), r"\".to_owned());
        assert_eq!(
            p.parse(r#""hello \"world\" quoted""#).unwrap(),
            r#"hello "world" quoted"#.to_owned()
        );
    }

    #[test]
    fn tag_name_parser_works() {
        let p = grammar::TagNameParser::new();
        assert_eq!(
            p.parse("lower").unwrap(),
            TagName::new("lower".to_owned()).unwrap()
        );
        assert_eq!(
            p.parse("camelCase").unwrap(),
            TagName::new("camelCase".to_owned()).unwrap()
        );
        assert_eq!(
            p.parse("elundis_core").unwrap(),
            TagName::new("elundis_core".to_owned()).unwrap()
        );
    }

    #[test]
    fn empty_dict_works() {
        let p = grammar::ValParser::new();
        let expected = Val::Dict(HashMap::new());
        assert_eq!(p.parse("{}").unwrap(), expected);
    }

    #[test]
    fn dict1_works() {
        let p = grammar::ValParser::new();
        let name = TagName::new("tagName".to_owned()).unwrap();
        let val = Val::Lit(Lit::Str("hello world".to_owned()));
        let mut hash_map = HashMap::new();
        hash_map.insert(name, Box::new(val));
        let expected = Val::Dict(hash_map);
        assert_eq!(p.parse(r#"{tagName:"hello world"}"#).unwrap(), expected);
    }

    #[test]
    fn dict2_works() {
        let p = grammar::ValParser::new();
        let name1 = TagName::new("tagName1".to_owned()).unwrap();
        let val1 = Val::Lit(Lit::Str("hello world".to_owned()));
        let mut hash_map = HashMap::new();
        hash_map.insert(name1, Box::new(val1));

        let name2 = TagName::new("tagName2".to_owned()).unwrap();
        let val2 = Val::Lit(Lit::Str("test".to_owned()));
        hash_map.insert(name2, Box::new(val2));

        let expected = Val::Dict(hash_map);
        assert_eq!(
            p.parse(r#"{tagName1:"hello world", tagName2:"test"}"#)
                .unwrap(),
            expected
        );
    }

    #[test]
    fn empty_list_works() {
        let p = grammar::ValParser::new();
        let expected = Val::List(vec![]);
        assert_eq!(p.parse("[]").unwrap(), expected);
    }

    #[test]
    fn list1_works() {
        let p = grammar::ValParser::new();
        let val = Val::Lit(Lit::Str("hello world".to_owned()));
        let expected = Val::List(vec![val]);
        assert_eq!(p.parse(r#"["hello world"]"#).unwrap(), expected);
    }

    #[test]
    fn list2_works() {
        let p = grammar::ValParser::new();
        let val1 = Val::Lit(Lit::Str("hello world".to_owned()));
        let val2 = Val::Lit(Lit::Str("test".to_owned()));
        let expected = Val::List(vec![val1, val2]);
        assert_eq!(p.parse(r#"["hello world", "test"]"#).unwrap(), expected);
    }

    #[test]
    fn float_parser_works() {
        let p = grammar::FloatParser::new();
        assert_eq!(p.parse("123").unwrap(), 123.0);
        assert_eq!(p.parse("-123").unwrap(), -123.0);
        assert_eq!(p.parse("123.45").unwrap(), 123.45);
        assert_eq!(p.parse("-123.45").unwrap(), -123.45);
    }

    #[test]
    fn number_parser_no_units_works() {
        let p = grammar::NumParser::new();
        assert_eq!(p.parse("123").unwrap(), Number::new(123.0, None));
        assert_eq!(p.parse("-123").unwrap(), Number::new(-123.0, None));
        assert_eq!(p.parse("123.45").unwrap(), Number::new(123.45, None));
        assert_eq!(p.parse("-123.45").unwrap(), Number::new(-123.45, None));
    }

    #[test]
    fn number_parser_units_works() {
        let p = grammar::NumParser::new();
        assert_eq!(
            p.parse("123percent").unwrap(),
            Number::new(123.0, Some("percent".to_owned()))
        );
        assert_eq!(
            p.parse("-123db").unwrap(),
            Number::new(-123.0, Some("db".to_owned()))
        );
        assert_eq!(
            p.parse("123.45db").unwrap(),
            Number::new(123.45, Some("db".to_owned()))
        );
        assert_eq!(
            p.parse("-123.45%").unwrap(),
            Number::new(-123.45, Some("%".to_owned()))
        );
    }

    #[test]
    fn hello_world_works() {
        let p = grammar::ValParser::new();
        p.parse(HELLO_WORLD).unwrap();
    }

    #[test]
    fn ahu_temp_diff_works() {
        let p = grammar::ValParser::new();
        p.parse(AHU_TEMP_DIFF).unwrap();
    }

    #[test]
    fn ahu_func_works() {
        let p = grammar::ValParser::new();
        p.parse(AHU_FUNC).unwrap();
    }

    #[test]
    fn equip_func_works() {
        let p = grammar::ValParser::new();
        p.parse(EQUIP_FUNC).unwrap();
    }

    #[test]
    fn old_chart_demo_works() {
        let p = grammar::ValParser::new();
        p.parse(OLD_CHART_DEMO).unwrap();
    }
}
