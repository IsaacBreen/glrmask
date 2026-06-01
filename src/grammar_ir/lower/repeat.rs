//! Bounded and unbounded repeat lowering.
//!
//! Repeat lowering is where regular-expression cardinalities become recursive
//! grammar productions.  The exported surface to `lower::mod` is deliberately
//! small: choose a tree shape and emit/cache nonterminals for exact/range repeats.

use crate::grammar_ir::ast::GrammarExpr;
use crate::grammar_ir::flat::{NonterminalID, Rule, Symbol};
use crate::GlrMaskError;

pub(super) enum RepeatTreeShape {
    Balanced,
    Left,
    Right,
    /// Deterministic bounded-range lowering that uses a countdown chain.
    Countdown,
    /// Balanced exact decomposition (O(log N) tree depth) with balanced
    /// range alternation (O(log N) close-bracket resolution).
    LeftBalanced,
}

pub(super) fn repeat_tree_shape() -> RepeatTreeShape {
    match std::env::var("GLRMASK_REPEAT_TREE_SHAPE").ok().as_deref() {
        Some(v) => repeat_tree_shape_from_value(v),
        None => RepeatTreeShape::Balanced,
    }
}

pub(super) fn repeat_tree_shape_from_value(value: &str) -> RepeatTreeShape {
    match value {
        "left" => RepeatTreeShape::Left,
        "balanced" => RepeatTreeShape::Balanced,
        "countdown" | "deterministic" => RepeatTreeShape::Countdown,
        "leftbalanced" | "left_balanced" => RepeatTreeShape::LeftBalanced,
        _ => RepeatTreeShape::Right,
    }
}

pub(super) fn right_repeat_range_front_bucket() -> usize {
    std::env::var("GLRMASK_RIGHT_REPEAT_RANGE_FRONT_BUCKET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128)
}

pub(super) fn left_repeat_range_back_bucket() -> usize {
    std::env::var("GLRMASK_LEFT_REPEAT_RANGE_BACK_BUCKET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128)
}

pub(super) fn exact_repeat_split(count: usize, shape: RepeatTreeShape) -> (usize, usize) {
    debug_assert!(count > 1);
    match shape {
        RepeatTreeShape::Balanced | RepeatTreeShape::Countdown | RepeatTreeShape::LeftBalanced => {
            let left = count / 2;
            (left, count - left)
        }
        RepeatTreeShape::Left => (count - 1, 1),
        RepeatTreeShape::Right => (1, count - 1),
    }
}

pub(super) fn range_repeat_split(min: usize, max: usize, shape: RepeatTreeShape) -> (usize, usize) {
    debug_assert!(min < max);
    let width = max - min + 1;
    match shape {
        RepeatTreeShape::Balanced | RepeatTreeShape::Countdown | RepeatTreeShape::LeftBalanced => {
            let left_width = width / 2;
            let split = min + left_width - 1;
            (split, width - left_width)
        }
        RepeatTreeShape::Left => (max - 1, 1),
        RepeatTreeShape::Right => (min, width - 1),
    }
}

pub(super) fn highest_power_of_two_le(n: usize) -> usize {

impl super::Lowerer {
    pub(super) fn repeat_exact_nonterminal(
        &mut self,
        symbol: &Symbol,
        count: usize,
        shape: RepeatTreeShape,
    ) -> NonterminalID {
        let key = (symbol.clone(), count);
        if let Some(&nonterminal) = self.repeat_exact_cache.get(&key) {
            return nonterminal;
        }

        let (_, nonterminal) = self.fresh_nonterminal("repeat_exact");
        self.repeat_exact_cache.insert(key, nonterminal);
        match count {
            0 => self.rules.push(Rule {
                lhs: nonterminal,
                rhs: Vec::new(),
            }),
            1 => self.rules.push(Rule {
                lhs: nonterminal,
                rhs: vec![symbol.clone()],
            }),
            _ => {
                let (left, right) = exact_repeat_split(count, shape);
                let left_nonterminal =
                    self.repeat_exact_nonterminal(symbol, left, shape);
                let right_nonterminal =
                    self.repeat_exact_nonterminal(symbol, right, shape);
                self.rules.push(Rule {
                    lhs: nonterminal,
                    rhs: vec![
                        Symbol::Nonterminal(left_nonterminal),
                        Symbol::Nonterminal(right_nonterminal),
                    ],
                });
            }
        }
        nonterminal
    }

    pub(super) fn repeat_max_nonterminal(
        &mut self,
        symbol: &Symbol,
        max: usize,
    ) -> NonterminalID {
        let key = (symbol.clone(), max);
        if let Some(&nt) = self.repeat_max_cache.get(&key) {
            return nt;
        }

        if max == 0 {
            let (_, nt) = self.fresh_nonterminal("repeat_max");
            self.repeat_max_cache.insert(key, nt);
            self.rules.push(Rule {
                lhs: nt,
                rhs: Vec::new(),
            });
            return nt;
        }

        let (_, nt) = self.fresh_nonterminal("repeat_max");
        self.repeat_max_cache.insert(key, nt);

        if let Some(span) = max.checked_add(1).filter(|span| span.is_power_of_two()) {
            let high = span / 2;
            debug_assert!(high > 0);

            let low_nt = self.repeat_max_nonterminal(symbol, high - 1);
            let exact_high_nt =
                self.repeat_exact_nonterminal(symbol, high, RepeatTreeShape::LeftBalanced);

            self.rules.push(Rule {
                lhs: nt,
                rhs: vec![Symbol::Nonterminal(low_nt)],
            });

            let rhs = if high == 1 {
                vec![Symbol::Nonterminal(exact_high_nt)]
            } else {
                vec![
                    Symbol::Nonterminal(exact_high_nt),
                    Symbol::Nonterminal(low_nt),
                ]
            };
            self.rules.push(Rule { lhs: nt, rhs });
        } else {
            let high = highest_power_of_two_le(max);
            debug_assert!(high > 0);
            debug_assert!(high <= max);

            let below_nt = self.repeat_max_nonterminal(symbol, high - 1);
            let exact_high_nt =
                self.repeat_exact_nonterminal(symbol, high, RepeatTreeShape::LeftBalanced);

            self.rules.push(Rule {
                lhs: nt,
                rhs: vec![Symbol::Nonterminal(below_nt)],
            });

            if max == high {
                self.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![Symbol::Nonterminal(exact_high_nt)],
                });
            } else {
                let tail_nt = self.repeat_max_nonterminal(symbol, max - high);
                self.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![
                        Symbol::Nonterminal(exact_high_nt),
                        Symbol::Nonterminal(tail_nt),
                    ],
                });
            }
        }

        nt
    }

    /// Returns a nonterminal matching exactly 1..=max occurrences of `symbol`.
    /// Defined as: `symbol repeat_max_{max-1}` — the first element is mandatory,
    /// the rest (0..max-1) are handled by `repeat_max`.
    pub(super) fn repeat_min1_max_nonterminal(&mut self, symbol: &Symbol, max: usize) -> NonterminalID {
        debug_assert!(max >= 1);
        let key = (symbol.clone(), max);
        if let Some(&nt) = self.repeat_min1_max_cache.get(&key) {
            return nt;
        }

        let (_, nt) = self.fresh_nonterminal("repeat_min1_max");
        self.repeat_min1_max_cache.insert(key, nt);

        if max == 1 {
            self.rules.push(Rule {
                lhs: nt,
                rhs: vec![symbol.clone()],
            });
            return nt;
        }

        let tail_nt = self.repeat_max_nonterminal(symbol, max - 1);
        self.rules.push(Rule {
            lhs: nt,
            rhs: vec![symbol.clone(), Symbol::Nonterminal(tail_nt)],
        });
        nt
    }

    pub(super) fn repeat_range_nonterminal(
        &mut self,
        symbol: &Symbol,
        min: usize,
        max: usize,
        shape: RepeatTreeShape,
    ) -> NonterminalID {
        debug_assert!(min <= max);
        if min == max {
            return self.repeat_exact_nonterminal(symbol, min, shape);
        }

        let key = (symbol.clone(), min, max);
        if let Some(&nonterminal) = self.repeat_range_cache.get(&key) {
            return nonterminal;
        }

        if shape == RepeatTreeShape::Countdown {
            return self.repeat_range_nonterminal_countdown(symbol, min, max, shape);
        }

        match shape {
            RepeatTreeShape::LeftBalanced | RepeatTreeShape::Balanced => {
                return self.repeat_range_nonterminal_balanced(symbol, min, max, shape);
            }
            _ => {}
        }

        let (_, nonterminal) = self.fresh_nonterminal("repeat_range");
        self.repeat_range_cache.insert(key, nonterminal);

        match shape {
            RepeatTreeShape::Right if (max - min + 1) > right_repeat_range_front_bucket() => {
                let cutoff = (min + right_repeat_range_front_bucket() - 1).min(max);
                for count in min..=cutoff {
                    let exact_nonterminal =
                        self.repeat_exact_nonterminal(symbol, count, shape);
                    self.rules.push(Rule {
                        lhs: nonterminal,
                        rhs: vec![Symbol::Nonterminal(exact_nonterminal)],
                    });
                }
                if cutoff < max {
                    let tail_nonterminal = self.repeat_range_nonterminal(
                        symbol,
                        cutoff + 1,
                        max,
                        shape,
                    );
                    self.rules.push(Rule {
                        lhs: nonterminal,
                        rhs: vec![Symbol::Nonterminal(tail_nonterminal)],
                    });
                }
                return nonterminal;
            }
            RepeatTreeShape::Left if (max - min + 1) > left_repeat_range_back_bucket() => {
                let cutoff = max.saturating_sub(left_repeat_range_back_bucket() - 1).max(min);
                if min < cutoff {
                    let head_nonterminal = self.repeat_range_nonterminal(
                        symbol,
                        min,
                        cutoff - 1,
                        shape,
                    );
                    self.rules.push(Rule {
                        lhs: nonterminal,
                        rhs: vec![Symbol::Nonterminal(head_nonterminal)],
                    });
                }
                for count in cutoff..=max {
                    let exact_nonterminal =
                        self.repeat_exact_nonterminal(symbol, count, shape);
                    self.rules.push(Rule {
                        lhs: nonterminal,
                        rhs: vec![Symbol::Nonterminal(exact_nonterminal)],
                    });
                }
                return nonterminal;
            }
            _ => {}
        }
        let (split, _) = range_repeat_split(min, max, shape);
        let left_nonterminal = self.repeat_range_nonterminal(
            symbol,
            min,
            split,
            shape,
        );
        let right_nonterminal = self.repeat_range_nonterminal(
            symbol,
            split + 1,
            max,
            shape,
        );
        self.rules.push(Rule {
            lhs: nonterminal,
            rhs: vec![Symbol::Nonterminal(left_nonterminal)],
        });
        self.rules.push(Rule {
            lhs: nonterminal,
            rhs: vec![Symbol::Nonterminal(right_nonterminal)],
        });
        nonterminal
    }

    pub(super) fn repeat_range_nonterminal_countdown(
        &mut self,
        symbol: &Symbol,
        min: usize,
        max: usize,
        shape: RepeatTreeShape,
    ) -> NonterminalID {
        let key = (symbol.clone(), min, max);
        if let Some(&nonterminal) = self.repeat_range_cache.get(&key) {
            return nonterminal;
        }

        let (_, nonterminal) = self.fresh_nonterminal("repeat_range");
        self.repeat_range_cache.insert(key, nonterminal);

        if min == 0 {
            self.rules.push(Rule {
                lhs: nonterminal,
                rhs: Vec::new(),
            });
        }

        let tail_nonterminal = if min > 0 {
            self.repeat_range_nonterminal(symbol, min - 1, max - 1, shape)
        } else {
            self.repeat_range_nonterminal(symbol, 0, max - 1, shape)
        };
        self.rules.push(Rule {
            lhs: nonterminal,
            rhs: vec![symbol.clone(), Symbol::Nonterminal(tail_nonterminal)],
        });
        nonterminal
    }

    pub(super) fn repeat_range_nonterminal_balanced(
        &mut self,
        symbol: &Symbol,
        min: usize,
        max: usize,
        shape: RepeatTreeShape,
    ) -> NonterminalID {
        debug_assert!(min <= max);
        debug_assert!(matches!(shape, RepeatTreeShape::LeftBalanced | RepeatTreeShape::Balanced));
        if min == max {
            return self.repeat_exact_nonterminal(symbol, min, shape);
        }
        let delta = max - min;
        let max_nt = self.repeat_max_nonterminal(symbol, delta);
        if min == 0 {
            max_nt
        } else {
            let exact_nt = self.repeat_exact_nonterminal(symbol, min, shape);
            let (_, result_nt) = self.fresh_nonterminal("repeat_range");
            self.rules.push(Rule {
                lhs: result_nt,
                rhs: vec![
                    Symbol::Nonterminal(exact_nt),
                    Symbol::Nonterminal(max_nt),
                ],
            });
            result_nt
        }
    }

    pub(super) fn emit_repeat_range(
        &mut self,
        lhs: NonterminalID,
        inner: &GrammarExpr,
        min: usize,
        max: usize,
    ) -> Result<(), GlrMaskError> {
        debug_assert!(min <= max);
        let symbol = self.lower_expr_terminalish(inner)?;
        let shape = repeat_tree_shape();
        let range_nonterminal = self.repeat_range_nonterminal(
            &symbol,
            min,
            max,
            shape,
        );
        self.rules.push(Rule {
            lhs,
            rhs: vec![Symbol::Nonterminal(range_nonterminal)],
        });
        Ok(())
    }

}

