use super::*;

fn reference_nullable(rules: &[Rule], count: u32) -> BTreeSet<u32> {
    let mut nullable = BTreeSet::new();
    loop {
        let before = nullable.len();
        for rule in rules {
            if rule.lhs < count && rule.rhs.iter().all(|symbol| {
                matches!(symbol, Symbol::Nonterminal(nt) if *nt < count && nullable.contains(nt))
            }) {
                nullable.insert(rule.lhs);
            }
        }
        if nullable.len() == before {
            return nullable;
        }
    }
}


#[test]
fn nullable_seed_shortcut_matches_literal_fixed_point() {
    let mut seed = 0x61d28ab3u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        seed
    };
    for _ in 0..512 {
        let mut rules = Vec::new();
        for lhs in 0..12u32 {
            for _ in 0..3 {
                let length = (next() % 4) as usize;
                let rhs = (0..length).map(|_| {
                    if next() % 4 == 0 {
                        Symbol::Terminal((next() % 3) as u32)
                    } else {
                        Symbol::Nonterminal((next() % 14) as u32)
                    }
                }).collect();
                rules.push(Rule { lhs, rhs });
            }
        }
        assert_eq!(compute_nullable(&rules, 12), reference_nullable(&rules, 12));
        rules.retain(|rule| !rule.rhs.is_empty());
        assert!(reference_nullable(&rules, 12).is_empty());
        assert!(compute_nullable(&rules, 12).is_empty());
    }
}
