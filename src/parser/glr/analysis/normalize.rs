pub fn normalize_grammar(rules: &mut Vec<Rule>, start: NonterminalID) {
    use std::cell::Cell;

    let profiling = compile_profile_enabled();
    let next_nt = Cell::new(max_nt_id(rules) + 1);
    let mut fresh_nt = || {
        let id = next_nt.get();
        next_nt.set(id + 1);
        id
    };

    let mut iteration = 0;
    loop {
        let iteration_rules_before = rules.len();
        let iteration_started_at = profiling.then(Instant::now);

        let clone_started_at = profiling.then(Instant::now);
        let snap = rules.clone();
        if let Some(started_at) = clone_started_at {
            emit_normalize_profile(
                "clone_snapshot",
                Some(iteration + 1),
                elapsed_ms(started_at),
                iteration_rules_before,
                snap.len(),
                "",
            );
        }

        let inline_rules_before = rules.len();
        let inline_started_at = profiling.then(Instant::now);
        replace_rules_with_resync(rules, &next_nt, inline_null_productions);
        if let Some(started_at) = inline_started_at {
            emit_normalize_profile(
                "inline_null_productions",
                Some(iteration + 1),
                elapsed_ms(started_at),
                inline_rules_before,
                rules.len(),
                " phase=fixed_point",
            );
        }

        let rr_rules_before = rules.len();
        let rr_started_at = profiling.then(Instant::now);
        with_resynced_next_nonterminal(rules, &next_nt, |rules| {
            eliminate_right_recursion(rules, &mut fresh_nt);
        });
        if let Some(started_at) = rr_started_at {
            emit_normalize_profile(
                "eliminate_right_recursion",
                Some(iteration + 1),
                elapsed_ms(started_at),
                rr_rules_before,
                rules.len(),
                "",
            );
        }

        let hlr_rules_before = rules.len();
        let hlr_started_at = profiling.then(Instant::now);
        let mut nullable_count = 0usize;
        with_resynced_next_nonterminal(rules, &next_nt, |rules| {
            let nullable = compute_nullable(rules, max_nt_id(rules) + 1);
            nullable_count = nullable.len();
            eliminate_hidden_left_recursion(rules, &nullable, iteration + 1);
        });

        if let Some(started_at) = hlr_started_at {
            emit_normalize_profile(
                "compute_nullable_and_eliminate_hidden_left_recursion",
                Some(iteration + 1),
                elapsed_ms(started_at),
                hlr_rules_before,
                rules.len(),
                &format!(" nullable_count={nullable_count}"),
            );
        }

        let dedup_rules_before = rules.len();
        let dedup_started_at = profiling.then(Instant::now);
        dedup_rules(rules);

        if let Some(started_at) = dedup_started_at {
            emit_normalize_profile(
                "dedup_rules",
                Some(iteration + 1),
                elapsed_ms(started_at),
                dedup_rules_before,
                rules.len(),
                " phase=fixed_point",
            );
        }

        let equality_started_at = profiling.then(Instant::now);
        let converged = *rules == snap;
        if let Some(started_at) = equality_started_at {
            emit_normalize_profile(
                "equality_convergence_check",
                Some(iteration + 1),
                elapsed_ms(started_at),
                rules.len(),
                snap.len(),
                &format!(" converged={converged}"),
            );
        }

        if let Some(started_at) = iteration_started_at {
            emit_normalize_profile(
                "fixed_point_iteration_total",
                Some(iteration + 1),
                elapsed_ms(started_at),
                iteration_rules_before,
                rules.len(),
                &format!(" converged={converged}"),
            );
        }

        iteration += 1;

        if converged {
            break;
        }
    }

    let post_inline_rules_before = rules.len();
    let post_inline_started_at = profiling.then(Instant::now);
    replace_rules_with_resync(rules, &next_nt, inline_null_productions);
    if let Some(started_at) = post_inline_started_at {
        emit_normalize_profile(
            "inline_null_productions",
            None,
            elapsed_ms(started_at),
            post_inline_rules_before,
            rules.len(),
            " phase=post_loop",
        );
    }

    let unreachable_rules_before = rules.len();
    let unreachable_started_at = profiling.then(Instant::now);
    *rules = remove_unreachable_rules(rules, start);
    if let Some(started_at) = unreachable_started_at {
        emit_normalize_profile(
            "remove_unreachable_rules",
            None,
            elapsed_ms(started_at),
            unreachable_rules_before,
            rules.len(),
            "",
        );
    }

    let final_dedup_rules_before = rules.len();
    let final_dedup_started_at = profiling.then(Instant::now);
    dedup_rules(rules);
    if let Some(started_at) = final_dedup_started_at {
        emit_normalize_profile(
            "dedup_rules",
            None,
            elapsed_ms(started_at),
            final_dedup_rules_before,
            rules.len(),
            " phase=final",
        );
    }
}

fn replace_rules_with_resync(
    rules: &mut Vec<Rule>,
    next_nt: &std::cell::Cell<u32>,
    update: impl FnOnce(&[Rule], u32) -> Vec<Rule>,
) {
    *rules = update(rules, next_nt.get());
    resync_next_nonterminal(rules, next_nt);
}

fn with_resynced_next_nonterminal(
    rules: &mut Vec<Rule>,
    next_nt: &std::cell::Cell<u32>,
    update: impl FnOnce(&mut Vec<Rule>),
) {
    update(rules);
    resync_next_nonterminal(rules, next_nt);
}

fn resync_next_nonterminal(rules: &[Rule], next_nt: &std::cell::Cell<u32>) {
    next_nt.set(max_nt_id(rules) + 1);
}
