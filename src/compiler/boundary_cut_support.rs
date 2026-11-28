//! Exact factorization of the existing necessary crossing-token observer.
//! This is not an admission oracle and does not build a parser automaton.
use crate::automata::lexer::observation::{ByteRows, ConfigInterner, EpsilonImageScratch};
use glrmask_lexer::__private::automata::lexer::tokenizer::SingletonEpsilonClosures;
use glrmask_lexer::__private::automata::lexer::{Lexer, tokenizer::Tokenizer};
use glrmask_lexer::__private::ds::bitset::BitSet;
use glrmask_terminal_dwa::__private::terminal_dwa::scope::{
    BoundaryOwnership, ImmediateComponentId,
};
use glrmask_vocab::Vocab;
use rustc_hash::FxHashMap;
use super::boundary_token_support::{FollowMasks, FollowPolicy};
use std::{collections::BTreeMap, sync::Arc, time::Instant};

#[derive(Clone)]
struct Observation {
    matched: Vec<u64>,
    future: Vec<u64>,
}
impl Observation {
    fn from_states(tok: &Tokenizer, states: &[u32]) -> Self {
        let words = (tok.num_terminals() as usize).div_ceil(64);
        let mut matched = vec![0u64; words];
        let mut future = vec![0u64; words];
        for &q in states {
            for t in tok.matched_terminals_iter(q) {
                matched[t as usize / 64] |= 1u64 << (t % 64);
            }
            for (a, &b) in future
                .iter_mut()
                .zip(tok.possible_future_terminals(q).words())
            {
                *a |= b;
            }
        }
        Self { matched, future }
    }
    fn endpoint(&self, eligible: &[u64], foreign: &[u64], crossed: bool) -> bool {
        self.matched
            .iter()
            .zip(&self.future)
            .zip(eligible)
            .zip(foreign)
            .any(|(((&m, &f), &e), &outside)| {
                ((m | f) & e & if crossed { u64::MAX } else { outside }) != 0
            })
    }
}
struct SegmentCache<'a> {
    tok: &'a Tokenizer,
    flat: Option<&'a ByteRows>,
    closures: Arc<SingletonEpsilonClosures>,
    image: EpsilonImageScratch,
    configs: ConfigInterner,
    observations: Vec<Observation>,
    edges: FxHashMap<(u32, u8), u32>,
    work: usize,
    retained_raw: usize,
}
impl<'a> SegmentCache<'a> {
    fn new(tok: &'a Tokenizer, flat: Option<&'a ByteRows>) -> Self {
        let mut configs = ConfigInterner::default();
        assert_eq!(configs.intern(Vec::new()), 0);
        Self {
            tok,
            flat,
            closures: tok.all_singleton_epsilon_closures(),
            image: EpsilonImageScratch::new(tok.num_states() as usize),
            configs,
            observations: vec![Observation::from_states(tok, &[])],
            edges: FxHashMap::default(),
            work: 0,
            retained_raw: 0,
        }
    }
    fn intern(&mut self, mut states: Vec<u32>) -> Option<u32> {
        states.sort_unstable();
        states.dedup();
        if let Some(id) = self.configs.find(&states) {
            return Some(id);
        }
        if self.configs.len() >= 32768 || states.iter().any(|&q| q >= self.tok.num_states()) {
            return None;
        }
        self.retained_raw = self.retained_raw.checked_add(states.len())?;
        if self.retained_raw > 4_000_000
            || self.configs.len().checked_add(1)?
                .checked_mul((self.tok.num_terminals() as usize).div_ceil(64))?
                .checked_mul(2)? > 4_000_000
        {
            return None;
        }
        self.observations.push(Observation::from_states(self.tok, &states));
        Some(self.configs.intern(states))
    }
    fn start(&mut self, seeds: &[u32]) -> Option<u32> {
        let mut closed = Vec::new();
        for &q in seeds {
            closed.extend_from_slice(&self.tok.singleton_epsilon_closure(q));
            if closed.len() > 2_000_000 {
                return None;
            }
        }
        self.intern(closed)
    }
    fn step(&mut self, q: u32, byte: u8) -> Option<u32> {
        if q == 0 {
            return Some(0);
        }
        if let Some(&next) = self.edges.get(&(q, byte)) {
            return Some(next);
        }
        if self.edges.len() >= 1_000_000 {
            return None;
        }
        self.work = self.work.checked_add(self.configs[q as usize].len())?;
        if self.work > 16_000_000 {
            return None;
        }
        let tok = self.tok;
        let flat = self.flat;
        let raw = self.image.step(
            &self.configs[q as usize], byte, true, &self.closures, None,
            |state, byte| flat.map_or_else(|| tok.get_transition(state, byte),
                |rows| rows.trans(state as usize, byte as usize)),
        ).to_vec();
        let next = self.intern(raw)?;
        self.edges.insert((q, byte), next);
        Some(next)
    }
}
#[derive(Debug, Default)]
pub(crate) struct Profile {
    pub states: usize,
    pub edges: usize,
    pub raw_step_work: usize,
    pub cut_pairs: usize,
    pub match_events: usize,
    pub transfer_cache_entries: usize,
    pub mask_classes: usize,
    pub setup_ms: f64,
    pub solve_ms: f64,
}
#[derive(Debug)]
pub(crate) struct Result {
    pub tokens: Vec<u32>,
    pub profile: Profile,
}

struct Transfers {
    rows: FxHashMap<(u32, u32), [u32; 2]>,
}
impl Transfers {
    fn get(
        &mut self,
        observation: u32,
        matched: &[u64],
        eligible: u32,
        follows: &[u32],
        foreign: &[bool],
        pool: &mut FollowMasks,
        events: &mut usize,
    ) -> Option<[u32; 2]> {
        if eligible == 0 {
            return Some([0, 0]);
        }
        let key = (observation, eligible);
        if let Some(&row) = self.rows.get(&key) {
            return Some(row);
        }
        let mut out = [0u32; 2];
        for (i, &matched) in matched.iter().enumerate() {
            let mut bits = matched & pool.rows[eligible as usize][i];
            while bits != 0 {
                let t = i * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                *events = events.checked_add(1)?;
                if *events > 16_000_000 {
                    return None;
                }
                let lane = usize::from(foreign[t]);
                out[lane] = pool.join(out[lane], follows[t])?;
            }
        }
        if self.rows.len() < 262144 {
            self.rows.insert(key, out);
        }
        Some(out)
    }
}
fn propagate(
    transfer: [u32; 2],
    crossed: bool,
    target: &mut [u32; 2],
    pool: &mut FollowMasks,
) -> Option<()> {
    if crossed {
        let all = pool.join(transfer[0], transfer[1])?;
        target[1] = pool.join(target[1], all)?;
    } else {
        target[0] = pool.join(target[0], transfer[0])?;
        target[1] = pool.join(target[1], transfer[1])?;
    }
    Some(())
}

#[cfg(test)]
pub(crate) fn crossing_support(
    tok: &Tokenizer,
    vocab: &Vocab,
    initial: &[bool],
    ownership: &BoundaryOwnership,
    start: ImmediateComponentId,
    disallowed: &BTreeMap<u32, BitSet>,
    ignore: Option<u32>,
    transparent: Option<&BitSet>,
    adjacency: Option<&BTreeMap<u32, BitSet>>,
) -> Option<Result> {
    crossing_support_with_flat(
        tok,
        vocab,
        initial,
        ownership,
        start,
        disallowed,
        ignore,
        transparent,
        adjacency,
        None,
    )
}

/// Crate-private adapter: flat is the already-built ordinary direct table of
/// this exact immutable tokenizer, with no query-dependent remapping.
fn crossing_support_with_flat(
    tok: &Tokenizer,
    vocab: &Vocab,
    initial: &[bool],
    ownership: &BoundaryOwnership,
    start: ImmediateComponentId,
    disallowed: &BTreeMap<u32, BitSet>,
    ignore: Option<u32>,
    transparent: Option<&BitSet>,
    adjacency: Option<&BTreeMap<u32, BitSet>>,
    flat: Option<&ByteRows>,
) -> Option<Result> {
    let clock = Instant::now();
    let n = tok.num_states() as usize;
    let nt = tok.num_terminals() as usize;
    if flat.is_some_and(|f| f.num_states() != n) {
        return None;
    }
    if n == 0
        || n > 200_000
        || initial.len() != n
        || nt == 0
        || nt > 65536
        || tok.has_virtual_residual_runtime()
        || vocab
            .entries_map()
            .values()
            .try_fold(0usize, |a, b| a.checked_add(b.len()))?
            > 262_144
    {
        return None;
    }
    if (0..n as u32).any(|q| tok.state_is_virtual_runtime(q)) {
        return None;
    }
    let width = nt.div_ceil(64);
    if nt.checked_mul(width)? > 4_000_000 {
        return None;
    }
    let FollowPolicy { masks: mut pool, follows, foreign, foreign_bits } =
        FollowPolicy::new(nt, ownership, start, disallowed, ignore, transparent, adjacency, 4_000_000)?;
    let mut transfers = Transfers { rows: FxHashMap::default() };
    let mut cache = SegmentCache::new(tok, flat);
    let first = cache.start(
        &initial
            .iter()
            .enumerate()
            .filter_map(|(q, &on)| on.then_some(q as u32))
            .collect::<Vec<_>>(),
    )?;
    let reset = cache.start(&tok.deterministic_reset_states())?;
    let mut profile = Profile {
        setup_ms: clock.elapsed().as_secs_f64() * 1000.0,
        ..Profile::default()
    };
    let solve = Instant::now();
    let mut tokens = Vec::new();
    for (&id, word) in vocab.entries_map().iter() {
        if word.is_empty() {
            tokens.push(id);
            continue;
        }
        if word.len() > 4096 || word.len().checked_mul(word.len())? > 1_000_000 {
            return None;
        }
        let mut cuts = vec![[0u32; 2]; word.len()];
        let mut accepted = false;
        let mut state = first;
        for end in 1..=word.len() {
            state = cache.step(state, word[end - 1])?;
            let obs = &cache.observations[state as usize];
            if end == word.len() {
                accepted |= obs.endpoint(&pool.rows[1], &foreign_bits, false);
            } else {
                let t = transfers.get(
                    state,
                    &obs.matched,
                    1,
                    &follows,
                    &foreign,
                    &mut pool,
                    &mut profile.match_events,
                )?;
                propagate(t, false, &mut cuts[end], &mut pool)?;
            }
            if state == 0 {
                break;
            }
        }
        for begin in 1..word.len() {
            if accepted {
                break;
            }
            let eligible = cuts[begin];
            if eligible == [0, 0] {
                continue;
            }
            let mut state = reset;
            for end in begin + 1..=word.len() {
                profile.cut_pairs += 1;
                if profile.cut_pairs > 4_000_000 {
                    return None;
                }
                state = cache.step(state, word[end - 1])?;
                let obs = &cache.observations[state as usize];
                for lane in 0..2 {
                    if end == word.len() {
                        accepted |= obs.endpoint(
                            &pool.rows[eligible[lane] as usize],
                            &foreign_bits,
                            lane == 1,
                        );
                    } else {
                        let t = transfers.get(
                            state,
                            &obs.matched,
                            eligible[lane],
                            &follows,
                            &foreign,
                            &mut pool,
                            &mut profile.match_events,
                        )?;
                        propagate(t, lane == 1, &mut cuts[end], &mut pool)?;
                    }
                }
                if state == 0 {
                    break;
                }
            }
        }
        if accepted {
            tokens.push(id);
        }
    }
    tokens.sort_unstable();
    profile.solve_ms = solve.elapsed().as_secs_f64() * 1000.0;
    profile.states = cache.configs.len();
    profile.edges = cache.edges.len();
    profile.raw_step_work = cache.work;
    profile.mask_classes = pool.rows.len();
    profile.transfer_cache_entries = transfers.rows.len();
    Some(Result { tokens, profile })
}

/// The direct table is usable only with its exact immutable tokenizer.
/// None/budget exhaustion means the caller must retain the unchanged exact
/// scanner; it never means an empty or truncated boundary token language.
pub(crate) fn crossing_support_with_transitions(
    tok: &Tokenizer,
    vocab: &Vocab,
    initial: &[bool],
    ownership: &BoundaryOwnership,
    start: ImmediateComponentId,
    disallowed: &BTreeMap<u32, BitSet>,
    ignore: Option<u32>,
    transparent: Option<&BitSet>,
    adjacency: Option<&BTreeMap<u32, BitSet>>,
    transitions: Option<&super::boundary_token_support::PreparedSupportTransitions<'_>>,
) -> Option<Result> {
    let flat = match transitions {
        Some(t) => Some(t.flat_for(tok)?),
        None => None,
    };
    crossing_support_with_flat(
        tok,
        vocab,
        initial,
        ownership,
        start,
        disallowed,
        ignore,
        transparent,
        adjacency,
        flat,
    )
}

#[cfg(test)]
mod tests {
    use super::super::boundary_cut_support as cut_support;
    use super::super::boundary_token_support as reference;
    use super::*;
    use crate::ds::bitset::BitSet;
    use glrmask_lexer::__private::automata::lexer::{
        Lexer,
        ast::{Expr, bytes, choice, plus},
        compile::{build_regex_monolithic, build_regex_partitioned},
        tokenizer::Tokenizer,
    };
    use glrmask_terminal_dwa::__private::terminal_dwa::scope::{
        BoundaryOwnership, ImmediateComponentId,
    };
    use std::collections::BTreeMap;
    #[test]
    fn cut_factorization_matches_frozen_raw_frontier_observer() {
        let exprs = vec![
            choice(vec![bytes(b"a"), bytes(b"abc"), bytes(b"bba")]),
            plus(choice(vec![bytes(b"ba"), bytes(b"a")])),
            Expr::Epsilon,
            plus(bytes(b" ")),
            choice(vec![bytes(b"!"), bytes(b"bc")]),
            plus(choice(vec![bytes(b"ab"), bytes(b"c")])),
        ];
         let a = build_regex_monolithic(&exprs).into_tokenizer(6, None);
        let b = build_regex_partitioned(&exprs, &[0, 1, 2, 3, 4, 5]).into_tokenizer(6, None);
        let (c, _) = Tokenizer::disjoint_union_with_terminal_offsets(&[(&a, 0), (&b, 6)]);
        let mut seed = 8197u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 32) as usize
        };
        let mut cases = 0;
        for tok in [a, b, c] {
            let nt = tok.num_terminals() as usize;
            let owner = BoundaryOwnership::flat(&[0, (nt / 2) as u32], nt as u32).unwrap();
            for case in 0..128 {
                let mut words = vec![
                    vec![],
                    b"a!".to_vec(),
                    b"aa!".to_vec(),
                    b"ba!".to_vec(),
                    b"a !".to_vec(),
                    b"abc".to_vec(),
                ];
                for _ in 0..40 {
                    let len = 1 + next() % 7;
                    words.push((0..len).map(|_| b"abc !"[next() % 5]).collect());
                }
                let vocab = Vocab::from_entries_unchecked(
                    words
                        .iter()
                        .enumerate()
                        .map(|(i, w)| ((i * 7 + 3) as u32, w.clone()))
                        .collect(),
                );
                let initial = (0..tok.num_states())
                    .map(|_| case % 7 == 0 || next() % 3 == 0)
                    .collect::<Vec<_>>();
                let mut blocked = BTreeMap::new();
                let mut extra = BTreeMap::new();
                for t in 0..nt {
                    let mut row = BitSet::new(nt);
                    let mut x = BitSet::new(nt);
                    for u in 0..nt {
                        if next() % 3 == 0 {
                            row.set(u);
                        }
                        if next() % 5 == 0 {
                            x.set(u);
                        }
                    }
                    blocked.insert(t as u32, row);
                    extra.insert(t as u32, x);
                }
                let ignore = (case % 3 == 0).then_some(3);
                let mut transparent = BitSet::new(nt);
                if case % 5 == 0 {
                    transparent.set(1);
                }
                let trans = (case % 2 == 0).then_some(&transparent);
                let extra = (case % 4 != 0).then_some(&extra);
                let start = ImmediateComponentId((case % 2) as u32);
                let expected = reference::crossing_token_support_with_follows_and_adjacency(
                    &tok, &vocab, &initial, &owner, start, &blocked, ignore, trans, extra,
                )
                .expect("old small reference");
                let actual = cut_support::crossing_support(
                    &tok, &vocab, &initial, &owner, start, &blocked, ignore, trans, extra,
                )
                .expect("new small exact filter");
                assert_eq!(
                    actual.tokens, expected.tokens,
                    "case{case} terminals{nt} initial{initial:?} words{words:?}"
                );
                let flat =
                    ByteRows::from_tokenizer(
                        &tok,
                    );
                let cached = cut_support::crossing_support_with_flat(
                    &tok,
                    &vocab,
                    &initial,
                    &owner,
                    start,
                    &blocked,
                    ignore,
                    trans,
                    extra,
                    Some(&flat),
                )
                .unwrap();
                assert_eq!(
                    cached.tokens, expected.tokens,
                    "epsilon-closed direct-table case{case}"
                );
                cases += 1;
            }
        }
        assert_eq!(cases, 384);
        println!("CUT_EXACT_GENERATED_CASES={cases}");
    }
}

#[cfg(test)]
#[test]
fn cut_support_rejects_invalid_domain_and_wrong_tokenizer_table() {
    use super::boundary_token_support::PreparedSupportTransitions;
    use glrmask_lexer::__private::automata::lexer::{ast::bytes, compile::build_regex_monolithic};
    let a = build_regex_monolithic(&[bytes(b"abc")]).into_tokenizer(1, None);
    let b = build_regex_monolithic(&[bytes(b"abd")]).into_tokenizer(1, None);
    let flat = ByteRows::from_tokenizer(&a);
    let prepared = PreparedSupportTransitions::new(&a, &flat).unwrap();
    let owner = BoundaryOwnership::flat(&[0], 1).unwrap();
    let v = Vocab::from_entries_unchecked(vec![(2, b"bc!".to_vec())]);
    let seeds = vec![true; a.num_states() as usize];
    let follows = BTreeMap::new();
    assert!(
        crossing_support_with_transitions(
            &a,
            &v,
            &[],
            &owner,
            ImmediateComponentId(0),
            &follows,
            None,
            None,
            None,
            Some(&prepared)
        )
        .is_none()
    );
    assert!(
        crossing_support_with_transitions(
            &b,
            &v,
            &vec![true; b.num_states() as usize],
            &owner,
            ImmediateComponentId(0),
            &follows,
            None,
            None,
            None,
            Some(&prepared)
        )
        .is_none()
    );
    let oversized = Vocab::from_entries_unchecked(vec![(1, vec![b'a'; 1025])]);
    assert!(
        crossing_support_with_transitions(
            &a,
            &oversized,
            &seeds,
            &owner,
            ImmediateComponentId(0),
            &follows,
            None,
            None,
            None,
            Some(&prepared)
        )
        .is_none()
    );
}
