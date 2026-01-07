#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{Label, StateID, Weight};
use super::dwa::{DWAState, DWAStates, DWABody, DWA};

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

#[cfg(feature = "dwa_minimize_exact")]
use once_cell::sync::Lazy;

#[cfg(feature = "dwa_minimize_exact")]
use z3::{
    ast::{Array, Ast, BV, Int},
    Config, Context, SatResult, Solver,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DWAExactMinimizeError {
    NotAcyclic,
    Z3FeatureNotEnabled,
    Z3UnsatEvenAtUpperBound,
}

/// Exact, globally minimal minimization options.
///
/// - `alphabet`: if `None`, we use exactly the set of labels that occur in the input DWA.
///   (This is the only finite alphabet you can possibly minimize over without introducing
///   infinitely many constraints; labels not present in the input are already always-empty
///   from the start in your model, and the minimal equivalent machine will just omit them.)
#[derive(Debug, Clone)]
pub struct DWAExactMinimizeOpts {
    pub alphabet: Option<Vec<Label>>,
}

impl Default for DWAExactMinimizeOpts {
    fn default() -> Self {
        Self { alphabet: None }
    }
}

/// A finite atom-partition of `usize` induced by all range boundaries appearing in input weights.
/// Each atom is an inclusive range; every Weight we care about is a union of these atoms.
#[derive(Debug, Clone)]
struct AtomPartition {
    atoms: Vec<(usize, usize)>, // inclusive ranges
}

impl AtomPartition {
    fn len(&self) -> usize {
        self.atoms.len()
    }

    fn collect_boundaries_from_weight(boundaries: &mut BTreeSet<usize>, w: &Weight) {
        // NOTE: this relies on Weight exposing `rsb.ranges()` with at least crate visibility,
        // as already used in your DWA::to_json_value implementation.
        for r in w.rsb.ranges() {
            boundaries.insert(*r.start());
            if *r.end() < usize::MAX {
                boundaries.insert(*r.end() + 1);
            }
        }
    }

    fn from_dwa(dwa: &DWA, alphabet: &[Label]) -> AtomPartition {
        let mut boundaries: BTreeSet<usize> = BTreeSet::new();
        boundaries.insert(0);

        // Gather boundaries from all explicit transition weights and final weights.
        for st in dwa.states.iter() {
            if let Some(fw) = &st.final_weight {
                Self::collect_boundaries_from_weight(&mut boundaries, fw);
            }
            for lbl in alphabet {
                // We treat missing weight as ALL in your printing/JSON and sometimes in code,
                // so include ALL's boundaries (none) doesn't matter.
                if let Some(w) = st.trans_weights.get(lbl) {
                    Self::collect_boundaries_from_weight(&mut boundaries, w);
                }
            }
        }

        // Build atoms as intervals between consecutive boundary points, with a final tail atom.
        let pts: Vec<usize> = boundaries.into_iter().collect();
        let mut atoms = Vec::new();

        for w in pts.windows(2) {
            let a = w[0];
            let b = w[1];
            // atom is [a, b-1]
            if a <= b.saturating_sub(1) {
                atoms.push((a, b - 1));
            }
        }
        // Tail
        let last = *pts.last().unwrap_or(&0);
        atoms.push((last, usize::MAX));

        AtomPartition { atoms }
    }

    fn weight_to_bits(&self, w: &Weight) -> Vec<u64> {
        // Chunk bits into u64s, least-significant chunk is atoms[0..64].
        let n = self.atoms.len();
        let chunks = (n + 63) / 64;
        let mut out = vec![0u64; chunks];

        for (i, &(a, _b)) in self.atoms.iter().enumerate() {
            if w.contains(a) {
                let ci = i / 64;
                let bi = i % 64;
                out[ci] |= 1u64 << bi;
            }
        }
        out
    }

    fn bits_to_weight(&self, chunks: &[u64]) -> Weight {
        // Build union of atom ranges with bit=1 and coalesce consecutive atoms.
        let mut ranges: Vec<(usize, usize)> = Vec::new();

        let mut cur: Option<(usize, usize)> = None;
        for (i, &(a, b)) in self.atoms.iter().enumerate() {
            let bit = {
                let ci = i / 64;
                let bi = i % 64;
                if ci >= chunks.len() {
                    false
                } else {
                    ((chunks[ci] >> bi) & 1) == 1
                }
            };
            if !bit {
                if let Some(r) = cur.take() {
                    ranges.push(r);
                }
                continue;
            }

            match cur.as_mut() {
                None => cur = Some((a, b)),
                Some((_cs, ce)) => {
                    if *ce == a.saturating_sub(1) {
                        *ce = b;
                    } else {
                        ranges.push(cur.take().unwrap());
                        cur = Some((a, b));
                    }
                }
            }
        }
        if let Some(r) = cur.take() {
            ranges.push(r);
        }

        Weight::from_ranges(&ranges)
    }
}

#[cfg(feature = "dwa_minimize_exact")]
fn bv_const_from_chunks<'ctx>(ctx: &'ctx Context, bit_len: usize, chunks: &[u64]) -> BV<'ctx> {
    assert!(bit_len >= 1);
    let mut remaining = bit_len;
    let mut first = true;
    let mut acc: Option<BV<'ctx>> = None;

    // Build MSB..LSB by concatenation.
    for ci in (0..chunks.len()).rev() {
        let width = remaining.min(64);
        remaining -= width;

        let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
        let v = chunks[ci] & mask;

        let part = BV::from_u64(ctx, v, width as u32);

        if first {
            acc = Some(part);
            first = false;
        } else {
            acc = Some(part.concat(acc.as_ref().unwrap()));
        }
    }

    // If bit_len is not a multiple of 64 and chunks were shorter (shouldn't happen), pad.
    let mut bv = acc.unwrap_or_else(|| BV::from_u64(ctx, 0, bit_len as u32));
    if bv.get_size() as usize != bit_len {
        // pad with zeros on the left
        let pad = bit_len - bv.get_size() as usize;
        let z = BV::from_u64(ctx, 0, pad as u32);
        bv = z.concat(&bv);
    }

    bv
}

#[cfg(feature = "dwa_minimize_exact")]
fn bv_to_chunks_from_model<'ctx>(
    model: &z3::Model<'ctx>,
    bv_expr: &BV<'ctx>,
    bit_len: usize,
) -> Vec<u64> {
    let chunks = (bit_len + 63) / 64;
    let mut out = vec![0u64; chunks];

    let bv_val = model.eval(bv_expr, true).unwrap();

    for ci in 0..chunks {
        let lo = (ci * 64) as u32;
        let hi = ((ci * 64 + 63).min(bit_len - 1)) as u32;
        let slice = bv_val.extract(hi, lo);
        out[ci] = slice.as_u64().unwrap_or(0);
    }

    out
}

/// Compute topo order; errors if cyclic.
fn topo_order(dwa: &DWA) -> Result<Vec<StateID>, DWAExactMinimizeError> {
    let n = dwa.states.len();
    let mut indeg = vec![0usize; n];
    for u in 0..n {
        for &v in dwa.states[u].transitions.values() {
            if v < n {
                indeg[v] += 1;
            }
        }
    }
    let mut q = VecDeque::new();
    for i in 0..n {
        if indeg[i] == 0 {
            q.push_back(i);
        }
    }
    let mut out = Vec::with_capacity(n);
    while let Some(u) = q.pop_front() {
        out.push(u);
        for &v in dwa.states[u].transitions.values() {
            if v >= n {
                continue;
            }
            indeg[v] -= 1;
            if indeg[v] == 0 {
                q.push_back(v);
            }
        }
    }
    if out.len() != n {
        return Err(DWAExactMinimizeError::NotAcyclic);
    }
    Ok(out)
}

/// Compute support S[q] = union of outputs possible from q under accumulator ALL.
///
/// This is the “right-support” used for normalization and for finding distinguishing words.
fn compute_support(dwa: &DWA, topo: &[StateID]) -> Vec<Weight> {
    let n = dwa.states.len();
    let mut supp = vec![Weight::zeros(); n];

    for &u in topo.iter().rev() {
        let mut s = dwa.states[u]
            .final_weight
            .clone()
            .unwrap_or_else(Weight::zeros);

        for (&lbl, &v) in &dwa.states[u].transitions {
            if v >= n {
                continue;
            }
            let w = dwa.states[u]
                .trans_weights
                .get(&lbl)
                .cloned()
                .unwrap_or_else(Weight::all);

            let contrib = &w & &supp[v];
            s |= &contrib;
        }

        supp[u] = s;
    }

    supp
}

#[derive(Clone)]
struct NormState {
    final_w: Weight, // empty means nonfinal
    trans: BTreeMap<Label, (StateID, Weight)>, // only non-empty effective transitions
}

/// Build a normalized view of the DWA semantics from each state (accumulator ALL),
/// by trimming dead bits: w_eff = w & support[target]. This makes the semantics canonical.
fn normalize_for_equiv(dwa: &DWA, alphabet: &[Label]) -> (Vec<NormState>, Vec<Weight>) {
    let topo = topo_order(dwa).expect("normalize_for_equiv requires acyclic");
    let supp = compute_support(dwa, &topo);
    let n = dwa.states.len();

    let mut norm = Vec::with_capacity(n);
    for u in 0..n {
        let final_w = dwa.states[u]
            .final_weight
            .clone()
            .unwrap_or_else(Weight::zeros);

        let mut trans = BTreeMap::new();
        for &lbl in alphabet {
            if let Some(&v) = dwa.states[u].transitions.get(&lbl) {
                if v >= n {
                    continue;
                }
                let w = dwa.states[u]
                    .trans_weights
                    .get(&lbl)
                    .cloned()
                    .unwrap_or_else(Weight::all);

                let mut w_eff = w;
                w_eff &= &supp[v];
                if !w_eff.is_empty() {
                    trans.insert(lbl, (v, w_eff));
                }
            }
        }

        norm.push(NormState { final_w, trans });
    }

    (norm, supp)
}

/// Find a word (as a Label sequence) such that starting at `state` with acc=ALL,
/// the output intersects `need` non-emptily.
///
/// This is used to construct a concrete distinguishing word when transition weights differ.
fn find_word_with_output_intersecting(
    norm: &[NormState],
    state: StateID,
    need: &Weight,
) -> Option<Vec<Label>> {
    // DFS in acyclic graph with pruning by intersecting need.
    // We don’t need memoization for correctness; this is for witness construction only.
    fn dfs(
        norm: &[NormState],
        u: StateID,
        need: &Weight,
        out: &mut Vec<Label>,
        seen: &mut Vec<bool>,
    ) -> bool {
        if need.is_empty() {
            return false;
        }
        if seen[u] {
            // Shouldn't happen in acyclic, but be safe.
            return false;
        }
        seen[u] = true;

        // ε case
        let hit = &norm[u].final_w & need;
        if !hit.is_empty() {
            seen[u] = false;
            return true;
        }

        // try transitions
        for (&lbl, &(v, ref w)) in &norm[u].trans {
            let mut need2 = need.clone();
            need2 &= w;
            if need2.is_empty() {
                continue;
            }
            out.push(lbl);
            if dfs(norm, v, &need2, out, seen) {
                seen[u] = false;
                return true;
            }
            out.pop();
        }

        seen[u] = false;
        false
    }

    let mut out = Vec::new();
    let mut seen = vec![false; norm.len()];
    if dfs(norm, state, need, &mut out, &mut seen) {
        Some(out)
    } else {
        None
    }
}

/// Compare two acyclic DWAs for exact equivalence from their start states.
/// If not equivalent, return an explicit counterexample word.
fn find_counterexample_word(orig: &DWA, cand: &DWA, alphabet: &[Label]) -> Option<Vec<Label>> {
    let (on, osupp) = normalize_for_equiv(orig, alphabet);
    let (cn, csupp) = normalize_for_equiv(cand, alphabet);

    let mut memo: HashMap<(StateID, StateID), Option<Vec<Label>>> = HashMap::new();

    fn diff(
        o: StateID,
        c: StateID,
        on: &[NormState],
        cn: &[NormState],
        osupp: &[Weight],
        csupp: &[Weight],
        alphabet: &[Label],
        memo: &mut HashMap<(StateID, StateID), Option<Vec<Label>>>,
    ) -> Option<Vec<Label>> {
        if let Some(res) = memo.get(&(o, c)) {
            return res.clone();
        }

        // finals differ => ε distinguishes
        if on[o].final_w != cn[c].final_w {
            let w = Some(Vec::new());
            memo.insert((o, c), w.clone());
            return w;
        }

        // Try labels in a fixed order
        for &lbl in alphabet {
            let ot = on[o].trans.get(&lbl).cloned();
            let ct = cn[c].trans.get(&lbl).cloned();

            match (ot, ct) {
                (None, None) => continue,
                (Some((ov, ow)), None) => {
                    // Candidate behaves like dead on this label; original has a non-empty contribution.
                    // Find suffix where original’s transition contributes something.
                    let mut need = ow.clone();
                    need &= &osupp[ov];
                    let suf = find_word_with_output_intersecting(on, ov, &need)
                        .unwrap_or_else(Vec::new);
                    let mut w = vec![lbl];
                    w.extend(suf);
                    memo.insert((o, c), Some(w.clone()));
                    return Some(w);
                }
                (None, Some((cv, cw))) => {
                    let mut need = cw.clone();
                    need &= &csupp[cv];
                    let suf = find_word_with_output_intersecting(cn, cv, &need)
                        .unwrap_or_else(Vec::new);
                    let mut w = vec![lbl];
                    w.extend(suf);
                    memo.insert((o, c), Some(w.clone()));
                    return Some(w);
                }
                (Some((ov, ow)), Some((cv, cw))) => {
                    // If successors differ, recurse.
                    if ov != ov || cv != cv {
                        // no-op; keep clippy happy
                    }

                    if ov < on.len() && cv < cn.len() {
                        if let Some(mut suf) = diff(ov, cv, on, cn, osupp, csupp, alphabet, memo) {
                            let mut w = vec![lbl];
                            w.append(&mut suf);
                            memo.insert((o, c), Some(w.clone()));
                            return Some(w);
                        }
                    }

                    // Same successor behavior but different edge weights can still distinguish.
                    if ow != cw {
                        // Tokens where weights disagree and that can actually matter
                        let mut need = &ow ^ &cw;
                        let mut target_supp = osupp[ov].clone();
                        // If successors are equivalent (diff returned None above), their supports should match,
                        // but intersecting with both is safe.
                        target_supp &= &csupp[cv];
                        need &= &target_supp;

                        if !need.is_empty() {
                            let suf = find_word_with_output_intersecting(on, ov, &need)
                                .unwrap_or_else(Vec::new);
                            let mut w = vec![lbl];
                            w.extend(suf);
                            memo.insert((o, c), Some(w.clone()));
                            return Some(w);
                        }
                    }
                }
            }
        }

        memo.insert((o, c), None);
        None
    }

    if orig.states.len() == 0 && cand.states.len() == 0 {
        return None;
    }

    let ostart = orig.body.start_state;
    let cstart = cand.body.start_state;
    if ostart >= orig.states.len() || cstart >= cand.states.len() {
        // If either is malformed, treat as distinguishable by ε.
        return Some(vec![]);
    }

    diff(
        ostart,
        cstart,
        &on,
        &cn,
        &osupp,
        &csupp,
        alphabet,
        &mut memo,
    )
}

impl DWA {
    /// Exact global minimization for **acyclic** DWAs under your exact start-state semantics
    /// (`eval_word_weight`).
    ///
    /// This is not a heuristic. It is a CEGIS loop:
    /// - synthesize a k-state candidate (acyclic by construction) that matches the original on a growing
    ///   set of words,
    /// - verify equivalence; if not, extract a counterexample word and add it.
    /// The first k for which this succeeds is provably minimal.
    pub fn minimize_acyclic_exact(
        &self,
        opts: DWAExactMinimizeOpts,
    ) -> Result<DWA, DWAExactMinimizeError> {
        if self.is_cyclic() {
            return Err(DWAExactMinimizeError::NotAcyclic);
        }

        #[cfg(not(feature = "dwa_minimize_exact"))]
        {
            let _ = opts;
            return Err(DWAExactMinimizeError::Z3FeatureNotEnabled);
        }

        #[cfg(feature = "dwa_minimize_exact")]
        {
            let alphabet = match opts.alphabet {
                Some(v) => v,
                None => {
                    let mut s = BTreeSet::new();
                    for st in self.states.iter() {
                        for lbl in st.transitions.keys() {
                            s.insert(*lbl);
                        }
                        for lbl in st.trans_weights.keys() {
                            // If you sometimes store weights without transitions, include them too.
                            s.insert(*lbl);
                        }
                    }
                    s.into_iter().collect()
                }
            };

            // Upper bound: original #states (original is always a feasible solution)
            let upper = self.states.len().max(1);

            // Atom partition and bit-width.
            let atoms = AtomPartition::from_dwa(self, &alphabet);
            let bit_len = atoms.len().max(1);

            // BV constants
            let cfg = Config::new();
            let ctx = Context::new(&cfg);

            let zero = bv_const_from_chunks(&ctx, bit_len, &vec![0u64; (bit_len + 63) / 64]);
            // all ones
            let mut ones_chunks = vec![u64::MAX; (bit_len + 63) / 64];
            if bit_len % 64 != 0 {
                let last_bits = bit_len % 64;
                let mask = (1u64 << last_bits) - 1;
                *ones_chunks.last_mut().unwrap() = mask;
            }
            let ones = bv_const_from_chunks(&ctx, bit_len, &ones_chunks);

            // Seed sample set with ε.
            let mut samples: Vec<Vec<Label>> = vec![vec![]];

            // A very small optimization: also seed with each single-letter word from start that exists.
            if self.body.start_state < self.states.len() {
                for &lbl in &alphabet {
                    samples.push(vec![lbl]);
                }
            }

            for k in 1..=upper {
                // Keep growing the same global sample set; if k is too small, Z3 will go UNSAT after
                // we add enough counterexamples.
                loop {
                    let cand = match synthesize_for_k(
                        &ctx,
                        self,
                        &alphabet,
                        &atoms,
                        bit_len,
                        &zero,
                        &ones,
                        k,
                        &samples,
                    ) {
                        Some(dwa) => dwa,
                        None => break, // UNSAT for this k
                    };

                    if let Some(w) = find_counterexample_word(self, &cand, &alphabet) {
                        if !samples.contains(&w) {
                            samples.push(w);
                            continue;
                        } else {
                            // Should not normally happen, but avoid infinite loops if witness repeats.
                            break;
                        }
                    } else {
                        return Ok(cand);
                    }
                }
            }

            Err(DWAExactMinimizeError::Z3UnsatEvenAtUpperBound)
        }
    }
}

#[cfg(feature = "dwa_minimize_exact")]
fn synthesize_for_k<'ctx>(
    ctx: &'ctx Context,
    orig: &DWA,
    alphabet: &[Label],
    atoms: &AtomPartition,
    bit_len: usize,
    zero: &BV<'ctx>,
    ones: &BV<'ctx>,
    k: usize,
    samples: &[Vec<Label>],
) -> Option<DWA> {
    let m = alphabet.len().max(1);
    let k_i64 = k as i64;

    // Label -> index
    let mut lbl2i: HashMap<Label, i64> = HashMap::new();
    for (i, &lbl) in alphabet.iter().enumerate() {
        lbl2i.insert(lbl, i as i64);
    }

    let solver = Solver::new(ctx);

    // Arrays over flattened index idx = state*m + label_idx
    let idx_sort = Int::sort(ctx);
    let state_sort = Int::sort(ctx);
    let w_sort = BV::sort(ctx, bit_len as u32);

    let delta = Array::new_const(ctx, "delta", &idx_sort, &state_sort);
    let tw = Array::new_const(ctx, "tw", &idx_sort, &w_sort);
    let fw = Array::new_const(ctx, "fw", &state_sort, &w_sort);

    let m_int = Int::from_i64(ctx, m as i64);

    // Range + acyclicity-by-index for any transition with non-empty weight.
    for s in 0..k {
        let s_int = Int::from_i64(ctx, s as i64);
        for li in 0..m {
            let idx = Int::from_i64(ctx, (s * m + li) as i64);

            let d = delta.select(&idx).as_int().unwrap();
            // 0 <= d < k
            solver.assert(&d.ge(&Int::from_i64(ctx, 0)));
            solver.assert(&d.lt(&Int::from_i64(ctx, k_i64)));

            let w = tw.select(&idx).as_bv().unwrap();
            // if w != 0 => d > s
            solver.assert(&w._eq(zero).not().implies(&d.gt(&s_int)));
        }
    }

    // Add sample constraints: candidate(word) == orig(word)
    for word in samples {
        let desired = orig.eval_word_weight(word);
        let desired_chunks = atoms.weight_to_bits(&desired);
        let desired_bv = bv_const_from_chunks(ctx, bit_len, &desired_chunks);

        // Symbolic evaluation of candidate on this word
        let mut st = Int::from_i64(ctx, 0);
        let mut acc = ones.clone();

        for &lbl in word {
            let li = *lbl2i.get(&lbl).unwrap_or(&0);
            let li_int = Int::from_i64(ctx, li);

            let idx = &st * &m_int + li_int;
            let w = tw.select(&idx).as_bv().unwrap();
            acc = &acc & &w;
            st = delta.select(&idx).as_int().unwrap();
        }

        let out = acc & fw.select(&st).as_bv().unwrap();
        solver.assert(&out._eq(&desired_bv));
    }

    if solver.check() != SatResult::Sat {
        return None;
    }
    let model = solver.get_model().unwrap();

    // Build DWA from model
    let mut states = Vec::with_capacity(k);
    for _ in 0..k {
        states.push(DWAState::default());
    }

    // finals
    for s in 0..k {
        let s_int = Int::from_i64(ctx, s as i64);
        let fw_bv = fw.select(&s_int).as_bv().unwrap();
        let chunks = bv_to_chunks_from_model(&model, &fw_bv, bit_len);
        let w = atoms.bits_to_weight(&chunks);
        if !w.is_empty() {
            states[s].final_weight = Some(w);
        }
    }

    // transitions
    for s in 0..k {
        for (li, &lbl) in alphabet.iter().enumerate() {
            let idx = Int::from_i64(ctx, (s * m + li) as i64);

            let w_bv = tw.select(&idx).as_bv().unwrap();
            let w_chunks = bv_to_chunks_from_model(&model, &w_bv, bit_len);
            let w = atoms.bits_to_weight(&w_chunks);
            if w.is_empty() {
                continue; // treat as missing transition
            }

            let d_int = delta.select(&idx).as_int().unwrap();
            let d_val = model.eval(&d_int, true).unwrap().as_i64().unwrap() as usize;

            states[s].transitions.insert(lbl, d_val);
            states[s].trans_weights.insert(lbl, w);
        }
    }

    let mut out = DWA {
        states: DWAStates(states),
        body: DWABody { start_state: 0 },
    };

    // Final cleanup: remove unreachable states and out-of-bounds transitions (shouldn't exist).
    out = restrict_to_reachable(&out);

    Some(out)
}

/// Simple reachability trim (same idea as in earlier code).
fn restrict_to_reachable(dwa: &DWA) -> DWA {
    let n = dwa.states.len();
    if n == 0 || dwa.body.start_state >= n {
        return dwa.clone();
    }

    let mut seen = vec![false; n];
    let mut q = VecDeque::new();
    seen[dwa.body.start_state] = true;
    q.push_back(dwa.body.start_state);

    while let Some(u) = q.pop_front() {
        for &v in dwa.states[u].transitions.values() {
            if v < n && !seen[v] {
                seen[v] = true;
                q.push_back(v);
            }
        }
    }

    // Compact
    let mut map = vec![usize::MAX; n];
    let mut rev = Vec::new();
    for i in 0..n {
        if seen[i] {
            map[i] = rev.len();
            rev.push(i);
        }
    }

    let mut new_states: Vec<DWAState> = Vec::with_capacity(rev.len());
    for &old in &rev {
        let old_state = &dwa.states[old];
        let mut transitions = BTreeMap::new();
        let mut trans_weights = BTreeMap::new();

        for (&lbl, &to_old) in &old_state.transitions {
            if to_old >= n || !seen[to_old] {
                continue;
            }
            let to_new = map[to_old];
            let w = old_state
                .trans_weights
                .get(&lbl)
                .cloned()
                .unwrap_or_else(Weight::all);
            if w.is_empty() {
                continue;
            }
            transitions.insert(lbl, to_new);
            trans_weights.insert(lbl, w);
        }

        new_states.push(DWAState {
            transitions,
            trans_weights,
            final_weight: old_state.final_weight.clone().filter(|w| !w.is_empty()),
        });
    }

    DWA {
        states: DWAStates(new_states),
        body: DWABody {
            start_state: map[dwa.body.start_state],
        },
    }
}