//! Exact finite Boolean-algebra coordinates for weighted compilation.
//!
//! Two observed (TSID, model-token) points share an atom iff every source
//! weight agrees on their membership. Union, intersection and difference can
//! never split such an atom. Compile in the small atom algebra and decode
//! afterward; no terminal, parser-stack label or model-token identity is lost.
//! This is independent of grammar syntax and of boundary-path topology.

use std::collections::BTreeMap;
use rustc_hash::FxHashMap;
use range_set_blaze::RangeSetBlaze;
use crate::ds::weight::Weight;

pub(crate) struct WeightObservationQuotient {
    domain: Weight,
    atoms: Vec<Weight>,
    atom_representatives: Vec<(u32, u32)>,
    // Pin source weights: pointer keys must not be reused during conversion.
    source_encodings: FxHashMap<usize, (Weight, Weight)>,
    points: usize,
}

impl WeightObservationQuotient {
    pub fn new(domain: &Weight, sources: &[Weight]) -> Option<Self> {
        const MAX_POINTS: usize = 16_384;
        const MAX_SOURCES: usize = 2_048;
        const MAX_MEMBERSHIP_CHECKS: usize = 16_000_000;
        const MAX_ATOMS: usize = 4_096;
        if domain.is_empty() || domain.is_full() { return None; }
        let mut unique = Vec::new();
        let mut source_ids = FxHashMap::default();
        for weight in sources {
            if !source_ids.contains_key(&weight.ptr_key()) {
                source_ids.insert(weight.ptr_key(), unique.len());
                unique.push(weight.clone());
            }
        }
        if unique.len() > MAX_SOURCES { return None; }
        let mut points = Vec::<(u32, u32)>::new();
        for (lo, hi, tokens) in domain.range_entries() {
            let volume = (u128::from(hi) - u128::from(lo) + 1) * (tokens.len() as u128);
            if volume > (MAX_POINTS - points.len()) as u128 { return None; }
            for tsid in lo..=hi {
                for token in tokens.iter() { points.push((tsid, token)); }
            }
        }
        if points.len().checked_mul(unique.len())? > MAX_MEMBERSHIP_CHECKS { return None; }
        let words = unique.len().div_ceil(64);
        let mut signatures = vec![vec![0u64; words]; points.len()];
        for (source_id, weight) in unique.iter().enumerate() {
            // ALL is an identity sentinel, not an explicit range row.
            if weight.is_full() {
                for signature in &mut signatures {
                    signature[source_id / 64] |= 1u64 << (source_id % 64);
                }
                continue;
            }
            let mut ranges = weight.range_entries().peekable();
            for (point_id, &(tsid, token)) in points.iter().enumerate() {
                while ranges.peek().is_some_and(|(_, hi, _)| *hi < tsid) { ranges.next(); }
                if ranges.peek().is_some_and(|(lo, hi, tokens)|
                    *lo <= tsid && tsid <= *hi && tokens.contains(token))
                {
                    signatures[point_id][source_id / 64] |= 1u64 << (source_id % 64);
                }
            }
        }
        let mut by_signature = FxHashMap::<Vec<u64>, usize>::default();
        let mut atom_signatures = Vec::new();
        let mut groups = Vec::<Vec<(u32, u32)>>::new();
        for (point, signature) in points.iter().copied().zip(signatures) {
            let class = if let Some(&class) = by_signature.get(&signature) {
                class
            } else {
                if groups.len() == MAX_ATOMS {
                    if std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some() {
                        eprintln!("[glrmask/profile][weight_observation_atoms] selected=false reason=atom_budget points={} sources={} atoms_more_than={}", points.len(), unique.len(), MAX_ATOMS);
                    }
                    return None;
                }
                let class = groups.len();
                atom_signatures.push(signature.clone());
                by_signature.insert(signature, class);
                groups.push(Vec::new());
                class
            };
            groups[class].push(point);
        }
        let atom_representatives = groups.iter().map(|points| points[0]).collect();
        let atoms = groups.into_iter().map(|points| {
            let mut rows = BTreeMap::<u32, Vec<u32>>::new();
            for (tsid, token) in points { rows.entry(tsid).or_default().push(token); }
            Weight::from_per_tsid_token_sets(rows.into_iter()
                .map(|(tsid, tokens)| (tsid, tokens.into_iter().collect::<RangeSetBlaze<u32>>())))
        }).collect::<Vec<_>>();
        let mut source_encodings = FxHashMap::default();
        for (source_id, source) in unique.into_iter().enumerate() {
            let mut rows = BTreeMap::<u32, Vec<u32>>::new();
            for (atom, signature) in atom_signatures.iter().enumerate() {
                if signature[source_id / 64] & (1u64 << (source_id % 64)) != 0 {
                    rows.entry((atom / 64) as u32).or_default().push((atom % 64) as u32);
                }
            }
            let encoded = Weight::from_per_tsid_token_sets(rows.into_iter()
                .map(|(row, bits)| (row, bits.into_iter().collect::<RangeSetBlaze<u32>>())));
            source_encodings.insert(source.ptr_key(), (source, encoded));
        }
        Some(Self { domain: domain.clone(), atoms, atom_representatives, source_encodings, points: points.len() })
    }

    pub fn atom_count(&self) -> usize { self.atoms.len() }
    pub fn point_count(&self) -> usize { self.points }
    pub fn rows(&self) -> usize { self.atoms.len().div_ceil(64) }
    pub fn source_count(&self) -> usize { self.source_encodings.len() }

    pub fn encode(&self, weight: &Weight) -> Option<Weight> {
        self.source_encodings.get(&weight.ptr_key()).map(|(_, encoded)| encoded.clone())
    }

    /// Encode a Boolean combination of the original source predicates.
    /// Every atom has constant membership in such a combination. This entry
    /// point is for outputs of the ordinary weighted compiler, not arbitrary
    /// new predicates: callers must establish that algebraic precondition.
    pub fn encode_derived(&self, weight: &Weight) -> Weight {
        let mut rows = BTreeMap::<u32, Vec<u32>>::new();
        let mut ranges = weight.range_entries().peekable();
        for (atom, &(tsid, token)) in self.atom_representatives.iter().enumerate() {
            while ranges.peek().is_some_and(|(_, hi, _)| *hi < tsid) { ranges.next(); }
            if weight.is_full() || ranges.peek().is_some_and(|(lo, hi, tokens)|
                *lo <= tsid && tsid <= *hi && tokens.contains(token))
            {
                rows.entry((atom / 64) as u32).or_default().push((atom % 64) as u32);
            }
        }
        Weight::from_per_tsid_token_sets(rows.into_iter()
            .map(|(row, bits)| (row, bits.into_iter().collect::<RangeSetBlaze<u32>>())))
    }

    /// Decode ANY atom-algebra result. Padding/out-of-domain encoded bits have
    /// empty meaning; in particular decoding universal weight yields domain R.
    pub fn decode(&self, encoded: &Weight) -> Weight {
        if encoded.is_empty() { return Weight::empty(); }
        if encoded.is_full() { return self.domain.clone(); }
        let mut selected = vec![0u64; self.rows()];
        for (lo, hi, tokens) in encoded.range_entries() {
            if lo as usize >= self.rows() { continue; }
            let mut bits = 0u64;
            for range in tokens.ranges() {
                let start = *range.start();
                if start >= 64 { continue; }
                let end = (*range.end()).min(63);
                let lower = u64::MAX << start;
                let upper = u64::MAX >> (63 - end);
                bits |= lower & upper;
            }
            for row in lo as usize..=(hi as usize).min(self.rows() - 1) { selected[row] |= bits; }
        }
        Weight::union_all(self.atoms.iter().enumerate().filter_map(|(atom, weight)| {
            (selected[atom / 64] & (1u64 << (atom % 64)) != 0).then_some(weight)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_atoms_extend_past_the_runtime_inline_domain() {
        let domain = Weight::from_token_set_for_tsid(9, (0..2048u32).collect());
        let mut sources = vec![Weight::all(), Weight::empty()];
        for bit in 0..11 {
            sources.push(Weight::from_token_set_for_tsid(9,
                (0..2048u32).filter(|token| token & (1 << bit) != 0).collect()));
        }
        let quotient = WeightObservationQuotient::new(&domain, &sources).unwrap();
        assert_eq!(quotient.atom_count(), 2048);
        assert_eq!(quotient.rows(), 32);
        for left in &sources {
            let encoded_left = quotient.encode(left).unwrap();
            assert_eq!(quotient.decode(&encoded_left), domain.intersection(left));
            for right in &sources {
                let encoded_right = quotient.encode(right).unwrap();
                assert_eq!(quotient.decode(&encoded_left.intersection(&encoded_right)), domain.intersection(&left.intersection(right)));
                assert_eq!(quotient.decode(&encoded_left.difference(&encoded_right)), domain.intersection(left).difference(&domain.intersection(right)));
            }
        }
    }

    #[test]
    fn observation_atoms_preserve_exact_boolean_operations_and_correlation() {
        let domain = Weight::from_per_tsid_token_sets([
            (1, [0, 2, 4, 6].into_iter().collect()),
            (7, [1, 3, 5, 7].into_iter().collect()),
            (19, [2, 3, 4].into_iter().collect()),
        ]);
        let mut rng = 13u64;
        let mut sources = vec![Weight::all(), Weight::empty()];
        for _ in 0..64 {
            let rows = [0, 1, 7, 19, 25].into_iter().map(|row| {
                let tokens = (0..9).filter(|_| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    rng >> 63 != 0
                }).collect::<RangeSetBlaze<_>>();
                (row, tokens)
            }).collect::<Vec<_>>();
            sources.push(Weight::from_per_tsid_token_sets(rows));
        }
        let quotient = WeightObservationQuotient::new(&domain, &sources).unwrap();
        assert_eq!(quotient.point_count(), 11);
        assert_eq!(Weight::union_all(&quotient.atoms), domain);
        for (i, a) in sources.iter().enumerate() {
            let x = quotient.encode(a).unwrap();
            assert_eq!(quotient.decode(&x), domain.intersection(a));
            for b in sources.iter().skip(i) {
                let y = quotient.encode(b).unwrap();
                let derived = a.union(b).difference(&domain.intersection(b));
                assert_eq!(quotient.decode(&quotient.encode_derived(&derived)),
                    domain.intersection(&derived));
                assert_eq!(quotient.decode(&x.union(&y)), domain.intersection(&a.union(b)));
                assert_eq!(quotient.decode(&x.intersection(&y)), domain.intersection(&a.intersection(b)));
                // Difference from the identity sentinel is conservative in
                // Weight, not a Boolean complement. Compare in explicit R.
                assert_eq!(quotient.decode(&x.difference(&y)),
                    domain.intersection(a).difference(&domain.intersection(b)));
            }
        }
        // Rows and token projections alone would incorrectly admit (1,1).
        assert!(quotient.decode(&Weight::all()).intersection(
            &Weight::from_token_set_for_tsid(1, [1].into_iter().collect())).is_empty());
    }

    #[test]
    fn observation_atoms_decline_unbounded_and_oversized_domains() {
        assert!(WeightObservationQuotient::new(&Weight::all(), &[]).is_none());
        assert!(WeightObservationQuotient::new(&Weight::empty(), &[]).is_none());
        let huge = Weight::from_uniform(0..=u32::MAX, [0].into_iter().collect());
        assert!(WeightObservationQuotient::new(&huge, &[huge.clone()]).is_none());
    }
}
