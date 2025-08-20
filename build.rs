// build.rs
use std::env;

fn main() {
    // Check if the custom environment variable is set.
    if env::var("COMPILED_IN_RUSTROVER").is_ok() {
        // If it's set, emit a custom cfg flag.
        println!("cargo:rustc-cfg=rustrover");
    }
}