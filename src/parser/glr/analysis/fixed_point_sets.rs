fn compute_nullable(rules: &[Rule], num_nt: u32) -> BTreeSet<NonterminalID> {
    let mut nullable = BTreeSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        for rule in rules {
            if rule.lhs >= num_nt {
                continue;
            }
            let rhs_nullable = rule.rhs.is_empty()
                || rule.rhs.iter().all(|symbol| match symbol {
                    Symbol::Terminal(_) => false,
                    Symbol::Nonterminal(nonterminal) => nullable.contains(nonterminal),
                });
            if rhs_nullable && nullable.insert(rule.lhs) {
                changed = true;
            }
        }
    }
    nullable
}

fn compute_first(
    rules: &[Rule],
    num_nt: u32,
    num_terminals: u32,
    nullable: &BTreeSet<NonterminalID>,
) -> Vec<BitSet> {
    let set_len = num_terminals as usize + 1;
    let mut first = vec![BitSet::new(set_len); num_nt as usize];
    let mut changed = true;
    while changed {
        changed = false;
        for rule in rules {
            let lhs = rule.lhs as usize;
            for symbol in &rule.rhs {
                match symbol {
                    Symbol::Terminal(terminal) => {
                        let bit = terminal_bit(*terminal, num_terminals);
                        if !first[lhs].contains(bit) {
                            first[lhs].set(bit);
                            changed = true;
                        }
                        break;
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        let additions = first[*nonterminal as usize].difference(&first[lhs]);
                        if !additions.is_empty() {
                            first[lhs].union_with(&additions);
                            changed = true;
                        }
                        if !nullable.contains(nonterminal) {
                            break;
                        }
                    }
                }
            }
        }
    }
    first
}

fn compute_follow(
    rules: &[Rule],
    num_nt: u32,
    num_terminals: u32,
    start: NonterminalID,
    first: &[BitSet],
    nullable: &BTreeSet<NonterminalID>,
) -> Vec<BitSet> {
    let set_len = num_terminals as usize + 1;
    let mut follow = vec![BitSet::new(set_len); num_nt as usize];
    if let Some(start_follow) = follow.get_mut(start as usize) {
        start_follow.set(terminal_bit(EOF, num_terminals));
    }

    let mut changed = true;
    while changed {
        changed = false;
        for rule in rules {
            let mut lhs_follow = None;
            for (index, symbol) in rule.rhs.iter().enumerate() {
                let Symbol::Nonterminal(nonterminal) = symbol else {
                    continue;
                };

                let suffix = &rule.rhs[index + 1..];
                let mut additions = BitSet::new(set_len);
                let mut suffix_nullable = true;
                for suffix_symbol in suffix {
                    match suffix_symbol {
                        Symbol::Terminal(terminal) => {
                            additions.set(terminal_bit(*terminal, num_terminals));
                            suffix_nullable = false;
                            break;
                        }
                        Symbol::Nonterminal(next_nonterminal) => {
                            additions.union_with(&first[*next_nonterminal as usize]);
                            if !nullable.contains(next_nonterminal) {
                                suffix_nullable = false;
                                break;
                            }
                        }
                    }
                }
                if suffix_nullable {
                    let lhs_follow = lhs_follow
                        .get_or_insert_with(|| follow[rule.lhs as usize].clone());
                    additions.union_with(lhs_follow);
                }

                let target = &mut follow[*nonterminal as usize];
                let delta = additions.difference(target);
                if !delta.is_empty() {
                    target.union_with(&delta);
                    changed = true;
                }
            }
        }
    }

    follow
}

#[inline]
fn terminal_bit(terminal: TerminalID, num_terminals: u32) -> usize {
    if terminal == EOF {
        num_terminals as usize
    } else {
        terminal as usize
    }
}

fn filter_graph_to_reachable(
    graph: BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
    reachable: &BTreeSet<NonterminalID>,
) -> BTreeMap<NonterminalID, BTreeSet<NonterminalID>> {
    graph
        .into_iter()
        .filter(|(nonterminal, _)| reachable.contains(nonterminal))
        .map(|(nonterminal, edges)| {
            (
                nonterminal,
                edges
                    .into_iter()
                    .filter(|edge| reachable.contains(edge))
                    .collect(),
            )
        })
        .collect()
}

