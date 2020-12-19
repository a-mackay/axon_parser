pub mod ast;

// Dict(
//     {
//         TagName(
//             "params",
//         ): List(
//             [],
//         ),
//         TagName(
//             "body",
//         ): Dict(
//             {
//                 TagName(
//                     "type",
//                 ): Lit(
//                     Str(
//                         "block",
//                     ),
//                 ),
//                 TagName(
//                     "exprs",
//                 ): List(
//                     [
//                         Dict(
//                             {
//                                 TagName(
//                                     "val",
//                                 ): Lit(
//                                     Str(
//                                         "hello world",
//                                     ),
//                                 ),
//                                 TagName(
//                                     "type",
//                                 ): Lit(
//                                     Str(
//                                         "literal",
//                                     ),
//                                 ),
//                             },
//                         ),
//                     ],
//                 ),
//             },
//         ),
//         TagName(
//             "type",
//         ): Lit(
//             Str(
//                 "func",
//             ),
//         ),
//     },
// )

// fn convert(val: Val) -> () {}

#[cfg(test)]
mod tests {
    use axon_parseast_parser::parse;

    const HELLO_WORLD: &str = r###"{type:"func", params:[], body:{type:"block", exprs:[{type:"literal", val:"hello world"}]}}"###;

    #[test]
    fn it_works() {
        let x = parse(HELLO_WORLD).unwrap();
        println!("{:#?}", x);
        assert_eq!(2 + 2, 4);
    }
}
