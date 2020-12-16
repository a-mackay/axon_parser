#[macro_use]
extern crate lalrpop_util;

use raystack::{Number, TagName};
use std::collections::HashMap;

lalrpop_mod!(pub grammar); // synthesized by LALRPOP

fn parse_to_hash_map(axon: &str) -> HashMap<TagName, Val> {
    println!("hello");
    unimplemented!()
}

#[derive(Debug, PartialEq)]
pub enum Val {
    Dict(HashMap<TagName, Box<Val>>),
    List(Vec<Val>),
    Lit(Lit),
}

#[derive(Debug, PartialEq)]
pub enum Lit {
    Str(String),
    Num(Number),
}

#[cfg(test)]
mod test {
    use raystack::{Number, TagName};
    use std::collections::HashMap;
    use super::grammar;
    use super::{Lit, Val};

    // {type:"func", params:[], body:{type:"block", exprs:[{type:"literal", val:"hello \"world\""}]}}
    const HELLO_WORLD: &str = r###"{type:"func", params:[], body:{type:"block", exprs:[{type:"literal", val:"hello world"}]}}"###;

    #[test]
    fn it_works() {
        assert_eq!(1 + 1, 2);
    }

    #[test]
    fn str_parser_works() {
        let p = grammar::StrParser::new();
        assert_eq!(p.parse(r#""hello world""#).unwrap(), "hello world".to_owned());
        assert_eq!(p.parse(r#""\n""#).unwrap(), "\n".to_owned());
        assert_eq!(p.parse(r#""\t""#).unwrap(), "\t".to_owned());
        assert_eq!(p.parse(r#""\\""#).unwrap(), r"\".to_owned());
        assert_eq!(p.parse(r#""hello \"world\" quoted""#).unwrap(), r#"hello "world" quoted"#.to_owned());
    }

    #[test]
    fn tag_name_parser_works() {
        let p = grammar::TagNameParser::new();
        assert_eq!(p.parse("lower").unwrap(), TagName::new("lower".to_owned()).unwrap());
        assert_eq!(p.parse("camelCase").unwrap(), TagName::new("camelCase".to_owned()).unwrap());
        assert_eq!(p.parse("elundis_core").unwrap(), TagName::new("elundis_core".to_owned()).unwrap());
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
        assert_eq!(p.parse(r#"{tagName1:"hello world", tagName2:"test"}"#).unwrap(), expected);
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
        assert_eq!(p.parse("123percent").unwrap(), Number::new(123.0, Some("percent".to_owned())));
        assert_eq!(p.parse("-123db").unwrap(), Number::new(-123.0, Some("db".to_owned())));
        assert_eq!(p.parse("123.45gH₂O/kgAir").unwrap(), Number::new(123.45, Some("gH₂O/kgAir".to_owned())));
        assert_eq!(p.parse("-123.45°daysF").unwrap(), Number::new(-123.45, Some("°daysF".to_owned())));
    }

    #[test]
    fn hello_world_works() {
        let p = grammar::ValParser::new();
        p.parse(HELLO_WORLD).unwrap();
    }
}