// build.rs
use std::env;
use std::path::PathBuf;

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

    let colpack_root = PathBuf::from("third_party/colpack");
    let cmake_dir = colpack_root.join("build/cmake");
    let colpack_install = cmake::Config::new(&cmake_dir)
        .define("ENABLE_OPENMP", "OFF")
        .define("ENABLE_EXAMPLES", "OFF")
        .build_target("ColPack_static")
        .build();

    let colpack_lib_dir = colpack_install.join("build");
    println!("cargo:rustc-link-search=native={}", colpack_lib_dir.display());
    println!("cargo:rustc-link-lib=static=ColPack");

    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
    }

    cc::Build::new()
        .cpp(true)
        .flag_if_supported("-std=c++11")
        .file("src/colpack_wrapper.cpp")
        .include(colpack_root.join("inc"))
        .include(colpack_root.join("src/Utilities"))
        .include(colpack_root.join("src/BipartiteGraphPartialColoring"))
        .include(colpack_root.join("src/BipartiteGraphBicoloring"))
        .include(colpack_root.join("src/GeneralGraphColoring"))
        .include(colpack_root.join("src/SMPGC"))
        .include(colpack_root.join("src/Recovery"))
        .compile("colpack_wrapper");

    println!("cargo:rerun-if-changed=src/colpack_wrapper.cpp");
    println!("cargo:rerun-if-changed=third_party/colpack/build/cmake/CMakeLists.txt");
}