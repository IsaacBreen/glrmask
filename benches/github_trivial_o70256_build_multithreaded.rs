#[path = "support/cfa_sweep.rs"]
mod cfa_sweep;

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_cfa_sweep_schema_build_multithreaded(c: &mut Criterion) {
    cfa_sweep::assert_release_benchmark("github_trivial_o70256_build_multithreaded");

    let vocab = cfa_sweep::load_llama3_vocab();
    assert_eq!(vocab.len(), 128_002, "expected the full Llama 3 vocabulary");
    eprintln!(
        "[bench][cfa_sweep_schema_build_multithreaded] cases={} vocab_tokens={} threading_env=external",
        cfa_sweep::CASES.len(),
        vocab.len()
    );
    cfa_sweep::bench_cases(c, "cfa_sweep_schema_build_multithreaded", &vocab);
}

criterion_group!(benches, bench_cfa_sweep_schema_build_multithreaded);
criterion_main!(benches);
