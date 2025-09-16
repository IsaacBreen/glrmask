// build.rs
use std::env;

fn main() {
    if (env::var("ENABLE_PROGRESS_BAR").is_ok() || cfg!(feature = "pbar")) && !cfg!(feature = "no_pbar") {
        // Enable progress bar
    } else {
        // Disable progress bar
        println!("cargo:rustc-cfg=rustrover");
    }

    // if env::var("COMPILED_IN_RUSTROVER").is_ok() {
    //     println!("cargo:rustc-cfg=rustrover");
    // }
}