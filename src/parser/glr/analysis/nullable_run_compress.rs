fn compute_nonempty_productive(rules: &[Rule], num_nt: u32) -> BTreeSet<NonterminalID> {
    let mut productive_any = BTreeSet::new();
    let mut nonempty_productive = BTreeSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        for rule in rules {
            if rule.lhs >= num_nt {
                continue;
            }

            let mut rhs_productive = true;
            let mut rhs_nonempty = false;
            for symbol in &rule.rhs {
                match symbol {
                    Symbol::Terminal(_) => rhs_nonempty = true,
                    Symbol::Nonterminal(nonterminal) => {
                        if !productive_any.contains(nonterminal) {
                            rhs_productive = false;
                            break;
                        }
                        rhs_nonempty |= nonempty_productive.contains(nonterminal);
                    }
                }
            }

            if rhs_productive {
                changed |= productive_any.insert(rule.lhs);
                if rhs_nonempty {
                    changed |= nonempty_productive.insert(rule.lhs);
                }
            }
        }
    }
    nonempty_productive
}

fn compress_nullable_runs_with_optional_tree(rules: &[Rule], num_nt: u32) -> Vec<Rule> {
    let profile_enabled = compile_profile_enabled();
    let total_started_at = profile_enabled.then(Instant::now);
    let rules_before = rules.len();

    let compute_nullable_started_at = profile_enabled.then(Instant::now);
    let nullable = compute_nullable(rules, num_nt);
    if let Some(started_at) = compute_nullable_started_at {
        emit_inline_null_profile(
            "compress_compute_nullable",
            elapsed_ms(started_at),
            rules_before,
            rules_before,
            &format!(" nullable_count={}", nullable.len()),
        );
    }
    if nullable.is_empty() {
        let result = rules.to_vec();
        if let Some(started_at) = total_started_at {
            emit_inline_null_profile(
                "compress_nullable_runs_with_optional_tree",
                elapsed_ms(started_at),
                rules_before,
                result.len(),
                &format!(
                    " nullable_count=0 run_count=0 max_run_len=0 output_rules={} shortcut=nullable_empty",
                    result.len(),
                ),
            );
        }
        return result;
    }

    let mut run_count = 0usize;
    let mut max_run_len = 0usize;
    let run_scan_started_at = profile_enabled.then(Instant::now);
    for rule in rules {
        for (start, end) in find_nullable_runs(&rule.rhs, &nullable, 1) {
            run_count += 1;
            max_run_len = max_run_len.max(end - start + 1);
        }
    }
    if let Some(started_at) = run_scan_started_at {
        emit_inline_null_profile(
            "compress_run_scan",
            elapsed_ms(started_at),
            rules_before,
            rules_before,
            &format!(
                " nullable_count={} run_count={} max_run_len={}",
                nullable.len(),
                run_count,
                max_run_len,
            ),
        );
    }
    if run_count == 0 {
        let result = rules.to_vec();
        if let Some(started_at) = total_started_at {
            emit_inline_null_profile(
                "compress_nullable_runs_with_optional_tree",
                elapsed_ms(started_at),
                rules_before,
                result.len(),
                &format!(
                    " nullable_count={} run_count=0 max_run_len=0 output_rules={} shortcut=no_runs",
                    nullable.len(),
                    result.len(),
                ),
            );
        }
        return result;
    }

    let compute_nonempty_started_at = profile_enabled.then(Instant::now);
    let nonempty_productive = compute_nonempty_productive(rules, num_nt);
    if let Some(started_at) = compute_nonempty_started_at {
        emit_inline_null_profile(
            "compress_compute_nonempty_productive",
            elapsed_ms(started_at),
            rules_before,
            rules_before,
            &format!(
                " nullable_count={} run_count={} max_run_len={} nonempty_productive_count={}",
                nullable.len(),
                run_count,
                max_run_len,
                nonempty_productive.len(),
            ),
        );
    }

    let by_lhs_started_at = profile_enabled.then(Instant::now);
    let mut by_lhs = BTreeMap::<NonterminalID, Vec<Vec<Symbol>>>::new();
    for rule in rules {
        by_lhs.entry(rule.lhs).or_default().push(rule.rhs.clone());
    }
    if let Some(started_at) = by_lhs_started_at {
        emit_inline_null_profile(
            "compress_build_by_lhs",
            elapsed_ms(started_at),
            rules_before,
            rules_before,
            &format!(
                " nullable_count={} run_count={} max_run_len={} lhs_count={}",
                nullable.len(),
                run_count,
                max_run_len,
                by_lhs.len(),
            ),
        );
    }

    let mut next_nt = max_nt_id(rules) + 1;
    let mut fresh_nt = || {
        let id = next_nt;
        next_nt += 1;
        id
    };

    let mut nn_cache = BTreeMap::<NonterminalID, NonterminalID>::new();
    let mut result = Vec::<Rule>::new();
    let tree_build_started_at = profile_enabled.then(Instant::now);
    for rule in rules {
        let runs = find_nullable_runs(&rule.rhs, &nullable, 1);
        if runs.is_empty() {
            result.push(rule.clone());
            continue;
        }

        let mut new_rhs = rule.rhs.clone();
        for &(start, end) in runs.iter().rev() {
            let segment: Vec<Symbol> = new_rhs.drain(start..=end).collect();
            let Some(root_nn) = build_non_nullable_tree(
                &segment,
                2,
                &mut fresh_nt,
                &mut result,
                &nullable,
                &nonempty_productive,
                &by_lhs,
                &mut nn_cache,
            ) else {
                continue;
            };
            let root_opt = fresh_nt();
            result.push(Rule {
                lhs: root_opt,
                rhs: vec![Symbol::Nonterminal(root_nn)],
            });
            result.push(Rule {
                lhs: root_opt,
                rhs: vec![],
            });
            new_rhs.insert(start, Symbol::Nonterminal(root_opt));
        }

        result.push(Rule {
            lhs: rule.lhs,
            rhs: new_rhs,
        });
    }
    if let Some(started_at) = tree_build_started_at {
        emit_inline_null_profile(
            "compress_tree_build_and_emit",
            elapsed_ms(started_at),
            rules_before,
            result.len(),
            &format!(
                " nullable_count={} run_count={} max_run_len={} output_rules={} fresh_nt_upper_bound={}",
                nullable.len(),
                run_count,
                max_run_len,
                result.len(),
                next_nt,
            ),
        );
    }

    let dedup_rules_before = result.len();
    let dedup_started_at = profile_enabled.then(Instant::now);
    dedup_rules(&mut result);
    if let Some(started_at) = dedup_started_at {
        emit_inline_null_profile(
            "compress_final_dedup_rules",
            elapsed_ms(started_at),
            dedup_rules_before,
            result.len(),
            &format!(
                " nullable_count={} run_count={} max_run_len={} output_rules={}",
                nullable.len(),
                run_count,
                max_run_len,
                result.len(),
            ),
        );
    }
    if let Some(started_at) = total_started_at {
        emit_inline_null_profile(
            "compress_nullable_runs_with_optional_tree",
            elapsed_ms(started_at),
            rules_before,
            result.len(),
            &format!(
                " nullable_count={} run_count={} max_run_len={} output_rules={}",
                nullable.len(),
                run_count,
                max_run_len,
                result.len(),
            ),
        );
    }
    result
}

fn build_non_nullable_tree(
    segment: &[Symbol],
    k: usize,
    fresh_nt: &mut impl FnMut() -> NonterminalID,
    new_rules: &mut Vec<Rule>,
    nullable: &BTreeSet<NonterminalID>,
    nonempty_productive: &BTreeSet<NonterminalID>,
    by_lhs: &BTreeMap<NonterminalID, Vec<Vec<Symbol>>>,
    nn_cache: &mut BTreeMap<NonterminalID, NonterminalID>,
) -> Option<NonterminalID> {
    let k = k.max(2);
    let n = segment.len();
    if n == 0 {
        return None;
    }

    let nn_segment: Vec<Symbol> = segment
        .iter()
        .filter_map(|symbol| match symbol {
            Symbol::Terminal(terminal) => Some(Symbol::Terminal(*terminal)),
            Symbol::Nonterminal(nonterminal) if !nullable.contains(nonterminal) => {
                Some(Symbol::Nonterminal(*nonterminal))
            }
            Symbol::Nonterminal(nonterminal) if nonempty_productive.contains(nonterminal) => {
                get_or_create_non_nullable_nt(
                    *nonterminal,
                    fresh_nt,
                    new_rules,
                    nullable,
                    nonempty_productive,
                    by_lhs,
                    nn_cache,
                )
                .map(Symbol::Nonterminal)
            }
            Symbol::Nonterminal(_) => None,
        })
        .collect();
    let n = nn_segment.len();
    if n == 0 {
        return None;
    }

    if n <= k {
        let leaf_nt = fresh_nt();
        for mask in 1u64..(1u64 << n) {
            let rhs: Vec<Symbol> = nn_segment
                .iter()
                .enumerate()
                .filter(|(idx, _)| ((mask >> idx) & 1) == 1)
                .map(|(_, symbol)| symbol.clone())
                .collect();
            new_rules.push(Rule { lhs: leaf_nt, rhs });
        }
        return Some(leaf_nt);
    }

    // Keep the default right-heavy decomposition; alternate shapes were only
    // used for internal experiments.
    let (first, rest) = nn_segment.split_at(1);
    let chunks: Vec<&[Symbol]> = if rest.is_empty() {
        vec![first]
    } else {
        vec![first, rest]
    };
    let chunk_nts: Vec<NonterminalID> = chunks
        .into_iter()
        .map(|chunk| {
            build_non_nullable_tree(
                chunk,
                k,
                fresh_nt,
                new_rules,
                nullable,
                nonempty_productive,
                by_lhs,
                nn_cache,
            )
            .expect("nonempty chunk should have a nonnullable tree")
        })
        .collect();
    let chunk_symbols: Vec<Symbol> = chunk_nts
        .into_iter()
        .map(Symbol::Nonterminal)
        .collect();
    build_non_nullable_tree(
        &chunk_symbols,
        k,
        fresh_nt,
        new_rules,
        nullable,
        nonempty_productive,
        by_lhs,
        nn_cache,
    )
}

fn get_or_create_non_nullable_nt(
    nt: NonterminalID,
    fresh_nt: &mut impl FnMut() -> NonterminalID,
    new_rules: &mut Vec<Rule>,
    nullable: &BTreeSet<NonterminalID>,
    nonempty_productive: &BTreeSet<NonterminalID>,
    by_lhs: &BTreeMap<NonterminalID, Vec<Vec<Symbol>>>,
    nn_cache: &mut BTreeMap<NonterminalID, NonterminalID>,
) -> Option<NonterminalID> {
    if !nullable.contains(&nt) {
        return Some(nt);
    }
    if !nonempty_productive.contains(&nt) {
        return None;
    }
    if let Some(&cached) = nn_cache.get(&nt) {
        return Some(cached);
    }

    let Some(alts) = by_lhs.get(&nt) else {
        return None;
    };
    let nn_nt = fresh_nt();
    nn_cache.insert(nt, nn_nt);
    let mut emitted = false;
    for rhs in alts {
        if rhs.is_empty() {
            continue;
        }
        if rhs.iter().any(|symbol| match symbol {
            Symbol::Terminal(_) => true,
            Symbol::Nonterminal(inner) => !nullable.contains(inner),
        }) {
            new_rules.push(Rule { lhs: nn_nt, rhs: rhs.clone() });
            emitted = true;
            continue;
        }

        let Some(rhs_nn) = build_non_nullable_tree(
            rhs,
            2,
            fresh_nt,
            new_rules,
            nullable,
            nonempty_productive,
            by_lhs,
            nn_cache,
        ) else {
            continue;
        };
        new_rules.push(Rule {
            lhs: nn_nt,
            rhs: vec![Symbol::Nonterminal(rhs_nn)],
        });
        emitted = true;
    }

    if !emitted {
        nn_cache.remove(&nt);
        return None;
    }
    Some(nn_nt)
}

/// Inline null productions (ε-elimination).
///
/// Preprocess long nullable runs with a balanced binary tree before doing the
/// existing exhaustive elimination, to avoid the raw power-set blowups that
/// occur when many nullable nonterminals appear consecutively.
pub(crate) fn inline_null_productions(rules: &[Rule], num_nt: u32) -> Vec<Rule> {
    let profile_enabled = compile_profile_enabled();
    let total_started_at = profile_enabled.then(Instant::now);
    let rules_before = rules.len();
    let preprocessed = compress_nullable_runs_with_optional_tree(rules, num_nt);
    let result = inline_null_productions_exhaustive(&preprocessed, max_nt_id(&preprocessed) + 1);
    if let Some(started_at) = total_started_at {
        emit_inline_null_profile(
            "inline_null_productions",
            elapsed_ms(started_at),
            rules_before,
            result.len(),
            &format!(
                " preprocessed_rules={} output_rules={}",
                preprocessed.len(),
                result.len(),
            ),
        );
    }
    result
}

