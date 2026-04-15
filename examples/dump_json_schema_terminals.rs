use std::io::{self, Read};

fn main() {
    let mut schema_json = String::new();
    io::stdin()
        .read_to_string(&mut schema_json)
        .expect("failed to read JSON schema from stdin");
    let terminals = glrmask::dump_json_schema_terminals_prepared(&schema_json)
        .expect("failed to dump JSON schema terminals");
    print!("{terminals}");
}