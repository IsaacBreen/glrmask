fn inline_null_productions_exhaustive(rules: &[Rule], num_nt: u32) -> Vec<Rule> {
    let profile_enabled = compile_profile_enabled();
    let total_started_at = profile_enabled.then(Instant::now);
    let rules_before = rules.len();

    let compute_nullable_started_at = profile_enabled.then(Instant::now);
    let nullable = compute_nullable(rules, num_nt);
    if let Some(started_at) = compute_nullable_started_at {
        let extra = format!(" nullable_count={}", nullable.len());
        emit_inline_null_profile(
            "exhaustive_compute_nullable",
            elapsed_ms(started_at),
            rules_before,
            rules_before,
            &extra,
        );
    }
    if nullable.is_empty() {
        let result = rules.to_vec();
        if let Some(started_at) = total_started_at {
            emit_inline_null_profile(
                "inline_null_productions_exhaustive",
                elapsed_ms(started_at),
                rules_before,
                result.len(),
                &format!(
                    " nullable_count=0 scanned_rules=0 total_nullable_positions=0 max_nullable_positions=0 output_rules={} shortcut=nullable_empty",
                    result.len(),
                ),
            );
        }
        return result;
    }

    if profile_enabled {
        let scan_started_at = Instant::now();
        let nullable_positions_by_rule: Vec<Vec<usize>> = rules
            .iter()
            .map(|rule| {
                rule.rhs
                    .iter()
                    .enumerate()
                    .filter_map(|(i, sym)| match sym {
                        Symbol::Nonterminal(nt) if nullable.contains(nt) => Some(i),
                        _ => None,
                    })
                    .collect()
            })
            .collect();
        let total_nullable_positions: usize = nullable_positions_by_rule
            .iter()
            .map(Vec::len)
            .sum();
        let max_nullable_positions = nullable_positions_by_rule
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        emit_inline_null_profile(
            "exhaustive_nullable_position_scan",
            elapsed_ms(scan_started_at),
            rules_before,
            rules_before,
            &format!(
                " nullable_count={} scanned_rules={} total_nullable_positions={} max_nullable_positions={}",
                nullable.len(),
                rules.len(),
                total_nullable_positions,
                max_nullable_positions,
            ),
        );

        let emit_started_at = Instant::now();
        let mut seen = HashSet::<Rule>::new();
        let mut out = Vec::new();

        for (rule, nullable_positions) in rules.iter().zip(nullable_positions_by_rule.iter()) {
            let k = nullable_positions.len();
            assert!(
                k <= 20,
                "production for NT {} has {} nullable positions; refusing power-set",
                rule.lhs, k,
            );

            for mask in 0u64..(1u64 << k) {
                let new_rhs: Vec<Symbol> = rule
                    .rhs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| match nullable_positions.binary_search(i) {
                        Ok(idx) => mask & (1u64 << idx) != 0,
                        Err(_) => true,
                    })
                    .map(|(_, sym)| sym.clone())
                    .collect();

                if new_rhs.is_empty() {
                    continue;
                }

                let candidate = Rule {
                    lhs: rule.lhs,
                    rhs: new_rhs,
                };
                if seen.insert(candidate.clone()) {
                    out.push(candidate);
                }
            }
        }

        emit_inline_null_profile(
            "exhaustive_powerset_emit_and_dedup",
            elapsed_ms(emit_started_at),
            rules_before,
            out.len(),
            &format!(
                " nullable_count={} scanned_rules={} total_nullable_positions={} max_nullable_positions={} output_rules={}",
                nullable.len(),
                rules.len(),
                total_nullable_positions,
                max_nullable_positions,
                out.len(),
            ),
        );

        if let Some(started_at) = total_started_at {
            emit_inline_null_profile(
                "inline_null_productions_exhaustive",
                elapsed_ms(started_at),
                rules_before,
                out.len(),
                &format!(
                    " nullable_count={} scanned_rules={} total_nullable_positions={} max_nullable_positions={} output_rules={}",
                    nullable.len(),
                    rules.len(),
                    total_nullable_positions,
                    max_nullable_positions,
                    out.len(),
                ),
            );
        }

        return out;
    }

    let mut seen = HashSet::<Rule>::new();
    let mut out = Vec::new();

    for rule in rules {
        let nullable_positions: Vec<usize> = rule
            .rhs
            .iter()
            .enumerate()
            .filter_map(|(i, sym)| match sym {
                Symbol::Nonterminal(nt) if nullable.contains(nt) => Some(i),
                _ => None,
            })
            .collect();

        let k = nullable_positions.len();
        // Safety guard: refuse power-set expansion beyond 20 nullable positions
        assert!(
            k <= 20,
            "production for NT {} has {} nullable positions; refusing power-set",
            rule.lhs, k,
        );

        for mask in 0u64..(1u64 << k) {
            let new_rhs: Vec<Symbol> = rule
                .rhs
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    match nullable_positions.binary_search(i) {
                        Ok(idx) => mask & (1u64 << idx) != 0, // bit set → keep
                        Err(_) => true,                        // non-nullable → always keep
                    }
                })
                .map(|(_, sym)| sym.clone())
                .collect();

            // Drop ε-rules
            if new_rhs.is_empty() {
                continue;
            }

            let candidate = Rule {
                lhs: rule.lhs,
                rhs: new_rhs,
            };
            if seen.insert(candidate.clone()) {
                out.push(candidate);
            }
        }
    }

    out
}

fn find_nullable_runs(
    rhs: &[Symbol],
    nullable: &BTreeSet<NonterminalID>,
    threshold: usize,
) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut run_start = None;

    for (idx, symbol) in rhs.iter().enumerate() {
        let is_nullable = matches!(symbol, Symbol::Nonterminal(nt) if nullable.contains(nt));
        if is_nullable {
            if run_start.is_none() {
                run_start = Some(idx);
            }
        } else if let Some(start) = run_start.take() {
            let len = idx - start;
            if len > threshold {
                runs.push((start, idx - 1));
            }
        }
    }

    if let Some(start) = run_start {
        let len = rhs.len() - start;
        if len > threshold {
            runs.push((start, rhs.len() - 1));
        }
    }

    runs
}
