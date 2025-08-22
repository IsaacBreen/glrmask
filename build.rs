// build.rs
use std::env;

fn main() {
    if (env::var("ENABLE_PROGRESS_BAR").is_err() || cfg!(pbar)) && !cfg!(no_pbar) {
        println!("cargo:rustc-cfg=rustrover");
    }

    // if env::var("COMPILED_IN_RUSTROVER").is_ok() {
    //     println!("cargo:rustc-cfg=rustrover");
    // }
}