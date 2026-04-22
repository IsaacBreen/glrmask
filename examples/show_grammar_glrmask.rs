//! Dump a JSON schema grammar in the GLRM format.
//! Usage: cargo run --example show_grammar_glrmask -- '<json-schema>'

fn main() {
    let schema = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: show_grammar_glrmask '<json-schema>'");
        std::process::exit(1);
    });

    match glrmask::dump_json_schema_grammar_glrm(&schema) {
        Ok(grammar) => print!("{}", grammar),
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}
