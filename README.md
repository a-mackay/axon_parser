# Axon Parser

Parses the output of SkySpark's `toAxonCode(parseAst( ... ))` into an abstract syntax tree.

## Features

* Parse an AST representing a Axon func.
* Code formatting

## Limitations and Missing Features

* Parse an AST representing an Axon defcomp.
* Some parsing limitations detailed in the [axon_parseast_parser repository](https://github.com/a-mackay/axon_parseast_parser)

## Code Formatting

The code formatting is basic and opinionated, and strays from idiomatic Axon code to
keep the implementation relatively simple. The configuration is currently
limited to the type and size of indentation used.

For example, Axon like:

```
( param1  ,param2:"arg2")=>"hello world"
```

will become:
```
(param1, param2: "arg2") => do
  "hello world"
end
```

It may not produce the prettiest Axon code, plus the formatted output has not been extensively tested for correctness.

## Getting Started

See the [documentation](https://docs.rs/axon_parser/) for the `parse_func` and `parse_func_to_formatted_lines` functions.