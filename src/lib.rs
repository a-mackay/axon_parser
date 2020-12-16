#[macro_use]
extern crate lalrpop_util;

use raystack::TagName;
use std::collections::HashMap;

lalrpop_mod!(pub grammar); // synthesized by LALRPOP

fn parse_to_hash_map(axon: &str) -> HashMap<TagName, Val> {
    println!("hello");
    unimplemented!()
}

#[derive(Debug)]
enum Val {
    Dict(HashMap<TagName, Box<Val>>),
    List(Vec<Val>),
    Lit(Lit),
}

#[derive(Debug)]
enum Lit {
    Str(String),
    Num(f64),
}

#[cfg(test)]
mod test {
    use super::grammar;

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
    fn hello_world_works() {
        // assert!(grammar::TermParser::new().parse(HELLO_WORLD).is_ok());
        // assert!(grammar::TermParser::new().parse("22").is_ok());
        // println!("{:?}", grammar::TermParser::new().parse("(22)"));
        // assert!(grammar::TermParser::new().parse("((((22))))").is_ok());
        // assert!(grammar::TermParser::new().parse("((22)").is_err());
    }
}