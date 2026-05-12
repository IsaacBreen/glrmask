#[path = "support/cfa_sweep.rs"]
mod cfa_sweep;

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_cfa_sweep_schema_build_single_threaded(c: &mut Criterion) {
    cfa_sweep::assert_release_benchmark("cfa_sweep_schema_build_single_threaded");
    cfa_sweep::force_single_threaded_compile();

    let vocab = cfa_sweep::load_llama3_vocab();
    assert_eq!(vocab.len(), 128_002, "expected the full Llama 3 vocabulary");
    let cases = cfa_sweep::selected_cases("cfa_sweep_schema_build_single_threaded");
    eprintln!(
        "[bench][cfa_sweep_schema_build_single_threaded] selected_cases={} total_cases={} vocab_tokens={} profile_once=1",
        cases.len(),
        cfa_sweep::CASES.len(),
        vocab.len()
    );
    cfa_sweep::profile_single_builds(&cases, &vocab);
    cfa_sweep::bench_cases(c, "cfa_sweep_schema_build_single_threaded", &cases, &vocab);
}

criterion_group!(benches, bench_cfa_sweep_schema_build_single_threaded);
criterion_main!(benches);
