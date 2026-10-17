fn main() {
    // The installed shared library must not retain a build-directory install
    // name on macOS. Consumers resolve it through their usual @rpath policy.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref()==Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libglrmask_c.dylib");
    }
}
