use sep1::glr::grammar::{prod, nt, t};

fn main() {
    // Create a simple arithmetic expression grammar
    // S' -> E
    // E -> E + T | T
    // T -> T * F | F
    // F -> ( E ) | num
    
    let productions = vec![
        // S' -> E (start production)
        prod("S'", vec![nt("E")]),
        // E -> E + T
        prod("E", vec![nt("E"), t("+"), nt("T")]),
        // E -> T
        prod("E", vec![nt("T")]),
        // T -> T * F
        prod("T", vec![nt("T"), t("*"), nt("F")]),
        // T -> F
        prod("T", vec![nt("F")]),
        // F -> ( E )
        prod("F", vec![t("("), nt("E"), t(")")]),
        // F -> num
        prod("F", vec![t("num")]),
    ];
    
    // Build the parser
    let parser = sep1::glr::table::generate_glr_parser(&productions, None);
    
    // Display the parser with pretty-printed parse table
    println!("{}", parser);
}

