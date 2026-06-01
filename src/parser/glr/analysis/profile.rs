fn compile_profile_enabled() -> bool {
    analysis_profile_enabled()
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn emit_normalize_profile(
    stage: &str,
    iteration: Option<usize>,
    elapsed_ms: f64,
    rules_before: usize,
    rules_after: usize,
    extra: &str,
) {
    match iteration {
        Some(iteration) => eprintln!(
            "[glrmask-profile] normalize_grammar stage={} iteration={} ms={:.3} rules_before={} rules_after={}{}",
            stage,
            iteration,
            elapsed_ms,
            rules_before,
            rules_after,
            extra,
        ),
        None => eprintln!(
            "[glrmask-profile] normalize_grammar stage={} ms={:.3} rules_before={} rules_after={}{}",
            stage,
            elapsed_ms,
            rules_before,
            rules_after,
            extra,
        ),
    }
}

fn emit_inline_null_profile(
    stage: &str,
    elapsed_ms: f64,
    rules_before: usize,
    rules_after: usize,
    extra: &str,
) {
    eprintln!(
        "[glrmask-profile] inline_null_productions stage={} ms={:.3} rules_before={} rules_after={}{}",
        stage,
        elapsed_ms,
        rules_before,
        rules_after,
        extra,
    );
}
