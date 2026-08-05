use glrmask::__private::run_o21137_subgrammar_benchmark;

fn main() {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "verify".to_string());
    run_o21137_subgrammar_benchmark(&mode);
}
