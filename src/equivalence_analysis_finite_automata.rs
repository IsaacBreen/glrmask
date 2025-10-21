use crate::finite_automata::{ExecutionResult, GroupID, Match, Regex};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::collections::{BTreeMap, BTreeSet};
use hashbrown::{HashMap, HashSet};
use smallvec::SmallVec;

// For debugging: verify equivalence classes using a brute-force method.
const VERIFY_EQUIVALENCE_CLASSES: bool = false;

// A canonical representation of a signature. It can be hashed and compared.
// It's derived from the graph of tokenization possibilities.
// It represents the result of a single greedy tokenization step.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CanonicalSignature {
    // All possible matches and the signature of their remainders (sorted and deduped).
    matches: Vec<(GroupID, usize, SignatureId)>,
    // If the string can be fully consumed without a match, the final DFA state.
    final_state: Option<usize>,
}

// An ID representing a unique canonical signature.
type SignatureId = usize;

// Manages interning of canonical signatures to unique IDs.
struct SignatureInterner {
    signatures: Vec<CanonicalSignature>,
    buckets: HashMap<u128, Vec<SignatureId>>,
}

impl SignatureInterner {
    fn new() -> Self {
        SignatureInterner {
            signatures: Vec::new(),
            buckets: HashMap::new(),
        }
    }

    fn intern(&mut self, sig: CanonicalSignature) -> SignatureId {
        let fp = fingerprint_canonical_signature(&sig);
        if let Some(ids) = self.buckets.get_mut(&fp) {
            for &id in ids.iter() {
                if self.signatures[id] == sig {
                    return id;
                }
            }
            let id = self.signatures.len();
            self.signatures.push(sig);
            ids.push(id);
            return id;
        } else {
            let id = self.signatures.len();
            self.signatures.push(sig);
            self.buckets.insert(fp, vec![id]);
            return id;
        }
    }
}

// Canonical representation of a suffix by referencing an original string and an offset.
#[derive(Clone, Copy)]
struct CanonicalSuffixRep {
    str_idx: usize,
    offset: usize,
}

// Deduplicates suffixes across all strings by content using a lightweight 128-bit hash,
// verifying equality by comparing the actual bytes on collisions. Also caches lookups
// by (str_idx, offset) to avoid re-hashing within the same process.
struct SuffixDeduper<'a> {
    strings: &'a [Vec<u8>],
    // hash -> list of node IDs (we verify equality inside bucket)
    buckets: HashMap<u128, Vec<usize>>,
    // node_id -> canonical representative
    nodes: Vec<CanonicalSuffixRep>,
    // (str_idx, offset) -> node_id
    offset_cache: HashMap<(usize, usize), usize>,
}

impl<'a> SuffixDeduper<'a> {
    fn new(strings: &'a [Vec<u8>]) -> Self {
        SuffixDeduper {
            strings,
            buckets: HashMap::new(),
            nodes: Vec::new(),
            offset_cache: HashMap::new(),
        }
    }

    #[inline]
    fn slice_of(&self, node_id: usize) -> &[u8] {
        let rep = self.nodes[node_id];
        &self.strings[rep.str_idx][rep.offset..]
    }

    fn get_or_intern(&mut self, str_idx: usize, offset: usize) -> usize {
        if let Some(&nid) = self.offset_cache.get(&(str_idx, offset)) {
            return nid;
        }
        let bytes = &self.strings[str_idx][offset..];
        let h = hash128(bytes);
        if let Some(bucket) = self.buckets.get(&h) {
            for &nid in bucket.iter() {
                let rep = self.nodes[nid];
                let rep_bytes = &self.strings[rep.str_idx][rep.offset..];
                if rep_bytes == bytes {
                    self.offset_cache.insert((str_idx, offset), nid);
                    return nid;
                }
            }
        }
        let nid = self.nodes.len();
        self.nodes.push(CanonicalSuffixRep { str_idx, offset });
        self.buckets.entry(h).or_default().push(nid);
        self.offset_cache.insert((str_idx, offset), nid);
        nid
    }

    #[inline]
    fn remainder_of(&mut self, node_id: usize, pos: usize) -> usize {
        let rep = self.nodes[node_id];
        self.get_or_intern(rep.str_idx, rep.offset + pos)
    }
}

// 128-bit non-cryptographic hash for byte slices, computed in one pass.
#[inline]
fn hash128(bytes: &[u8]) -> u128 {
    const FNV_OFFSET_BASIS1: u64 = 1469598103934665603;
    const FNV_OFFSET_BASIS2: u64 = 1099511628211;
    const FNV_PRIME1: u64 = 1099511628211;
    const FNV_PRIME2: u64 = 14029467366897019727;

    let mut h1: u64 = FNV_OFFSET_BASIS1;
    let mut h2: u64 = FNV_OFFSET_BASIS2 ^ 0x9E3779B97F4A7C15;

    for &b in bytes {
        h1 ^= b as u64;
        h1 = h1.wrapping_mul(FNV_PRIME1);

        let rb = (b as u64).rotate_left(5);
        h2 ^= rb;
        h2 = h2.wrapping_mul(FNV_PRIME2);
    }

    ((h1 as u128) << 64) | (h2 as u128)
}

#[inline]
fn mix64(mut x: u64) -> u64 {
    // SplitMix64 mix function
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

// Computes a 128-bit fingerprint for CanonicalSignature to bucket interning candidates.
// This avoids hashing long vectors with the default hasher on every insertion/lookup.
#[inline]
fn fingerprint_canonical_signature(sig: &CanonicalSignature) -> u128 {
    let mut h1: u64 = 0x9E3779B97F4A7C15;
    let mut h2: u64 = 0xD6E8FEB86659FD93;
    for &(g, p, r) in &sig.matches {
        let k1 = mix64((g as u64).wrapping_mul(0x9E3779B185EBCA87) ^ (p as u64));
        let k2 = mix64((r as u64).wrapping_mul(0x94D049BB133111EB) ^ ((p as u64) << 1));
        h1 = h1.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(k1);
        h2 = h2.wrapping_mul(0x94D049BB133111EB).wrapping_add(k2);
    }
    let fsv = sig.final_state.map(|x| x as u64).unwrap_or(0xFFFF_FFFF_FFFF_FFFF);
    h1 ^= mix64(fsv);
    h2 ^= mix64(fsv.rotate_left(17));
    ((h1 as u128) << 64) | (h2 as u128)
}

pub struct EquivalenceAnalyzer<'a> {
    regex: &'a Regex,
    strings: &'a [Vec<u8>],
    initial_states: &'a [usize],
}

impl<'a> EquivalenceAnalyzer<'a> {
    pub fn new(regex: &'a Regex, strings: &'a [Vec<u8>], initial_states: &'a [usize]) -> Self {
        EquivalenceAnalyzer {
            regex,
            strings,
            initial_states,
        }
    }

    // Compute the canonical signature for a given canonical suffix node and DFA state,
    // using on-demand recursion and memoization. Remainders always recurse into start_state.
    fn compute_signature_for_state(
        &self,
        deduper: &mut SuffixDeduper<'a>,
        cache_start: &mut Vec<Option<SignatureId>>,
        cache_other: &mut Vec<HashMap<usize, SignatureId>>,
        interner: &mut SignatureInterner,
        node_id: usize,
        dfa_state: usize,
    ) -> SignatureId {
        // Ensure cache capacity for current node
        if cache_start.len() <= node_id {
            let missing = node_id + 1 - cache_start.len();
            cache_start.reserve(missing);
            cache_other.reserve(missing);
            for _ in 0..missing {
                cache_start.push(None);
                cache_other.push(HashMap::new());
            }
        }
        if dfa_state == self.regex.dfa.start_state {
            if let Some(sid) = cache_start[node_id] {
                return sid;
            }
        } else {
            if let Some(sid) = cache_other[node_id].get(&dfa_state) {
                return *sid;
            }
        }

        let bytes = deduper.slice_of(node_id);
        let result = self.regex.execute_from_state2(bytes, dfa_state);

        let mut matches_vec: SmallVec<[(GroupID, usize, SignatureId); 4]> = SmallVec::new();
        for m in result.matches {
            // Filter out zero-width tokens to avoid infinite recursion.
            if m.position == 0 {
                continue;
            }
            let remainder_node = deduper.remainder_of(node_id, m.position);
            // Ensure caches for the remainder node
            if cache_start.len() <= remainder_node {
                let missing = remainder_node + 1 - cache_start.len();
                cache_start.reserve(missing);
                cache_other.reserve(missing);
                for _ in 0..missing {
                    cache_start.push(None);
                    cache_other.push(HashMap::new());
                }
            }
            let remainder_sig = self.compute_signature_for_state(
                deduper,
                cache_start,
                cache_other,
                interner,
                remainder_node,
                self.regex.dfa.start_state,
            );
            matches_vec.push((m.group_id, m.position, remainder_sig));
        }
        // Canonicalize matches ordering and dedup identical entries if any.
        matches_vec.sort_unstable();
        matches_vec.dedup();

        let sig = CanonicalSignature {
            matches: matches_vec.into_vec(),
            final_state: result.end_state,
        };
        let sid = interner.intern(sig);

        if dfa_state == self.regex.dfa.start_state {
            cache_start[node_id] = Some(sid);
        } else {
            cache_other[node_id].insert(dfa_state, sid);
        }
        sid
    }

    pub fn find_equivalence_classes(&mut self) -> BTreeMap<Vec<SignatureId>, Vec<usize>> {
        // On-demand analysis: dedupe suffixes globally; compute signatures lazily.
        crate::debug!(
            2,
            "Starting LLM token equivalence analysis (on-demand) for {} strings...",
            self.strings.len()
        );
        let pb = ProgressBar::new(self.strings.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%, {eta}) (Equivalence Analysis)")
                .expect("progress-bar"),
        );
        if !PROGRESS_BAR_ENABLED {
            pb.set_draw_target(ProgressDrawTarget::hidden());
        }

        let mut signature_interner = SignatureInterner::new();
        let mut deduper = SuffixDeduper::new(self.strings);
        // Cache for signatures at start_state and at other states.
        let mut cache_start: Vec<Option<SignatureId>> = Vec::new();
        let mut cache_other: Vec<HashMap<usize, SignatureId>> = Vec::new();

        // Classify original strings based on their signature vectors for the provided initial states.
        let mut equivalence_classes: HashMap<Vec<SignatureId>, Vec<usize>> = HashMap::new();
        for (i, _s) in self.strings.iter().enumerate() {
            pb.inc(1);
            let node_id = deduper.get_or_intern(i, 0);
            // Ensure cache capacity for this node
            if cache_start.len() <= node_id {
                let missing = node_id + 1 - cache_start.len();
                cache_start.reserve(missing);
                cache_other.reserve(missing);
                for _ in 0..missing {
                    cache_start.push(None);
                    cache_other.push(HashMap::new());
                }
            }
            let mut signature_vector = Vec::with_capacity(self.initial_states.len());
            for &initial_state in self.initial_states {
                let sig_id = self.compute_signature_for_state(
                    &mut deduper,
                    &mut cache_start,
                    &mut cache_other,
                    &mut signature_interner,
                    node_id,
                    initial_state,
                );
                signature_vector.push(sig_id);
            }
            equivalence_classes.entry(signature_vector).or_default().push(i);
        }
        pb.finish();

        if VERIFY_EQUIVALENCE_CLASSES {
            verify_equivalence_classes(self.regex, self.strings, self.initial_states, &equivalence_classes);
        }

        // Convert to BTreeMap to preserve the original return type determinism.
        let mut out: BTreeMap<Vec<SignatureId>, Vec<usize>> = BTreeMap::new();
        for (k, v) in equivalence_classes {
            out.entry(k).or_default().extend(v);
        }
        out
    }
}

/// Finds equivalence classes among a set of strings based on their tokenization
/// behavior with a given Regex, starting from a set of initial DFA states.
///
/// Two strings are considered equivalent if, for every initial DFA state provided,
/// they produce the same sequence of tokens.
pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> BTreeMap<Vec<SignatureId>, Vec<usize>> {
    let mut analyzer = EquivalenceAnalyzer::new(regex, strings, initial_states);
    analyzer.find_equivalence_classes()
}

/// Brute-force verification of equivalence classes.
/// This is slow and should only be used for debugging.
fn verify_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
    computed_classes: &BTreeMap<Vec<SignatureId>, Vec<usize>>,
) {
    println!("Verifying equivalence classes (this may be slow)...");
 
    // A canonical representation of the tokenization graph for verification.
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct InternedVerificationSignature {
        matches: BTreeSet<(GroupID, usize, InternedVerificationSignatureId)>,
        final_state: Option<usize>,
    }
    type InternedVerificationSignatureId = usize;

    struct VerificationSignatureInterner {
        signatures: Vec<InternedVerificationSignature>,
        map: HashMap<InternedVerificationSignature, InternedVerificationSignatureId>,
    }

    impl VerificationSignatureInterner {
        fn new() -> Self {
            VerificationSignatureInterner {
                signatures: Vec::new(),
                map: HashMap::new(),
            }
        }

        fn intern(&mut self, sig: InternedVerificationSignature) -> InternedVerificationSignatureId {
            if let Some(&id) = self.map.get(&sig) {
                return id;
            }
            let id = self.signatures.len();
            self.signatures.push(sig.clone());
            self.map.insert(sig, id);
            id
        }
    }

    // 1. Collect all unique suffixes.
    let mut suffixes = HashSet::new();
    suffixes.insert(&[] as &[u8]);
    for s in strings {
        for i in 0..=s.len() {
            suffixes.insert(&s[i..]);
        }
    }
    let mut sorted_suffixes: Vec<&[u8]> = suffixes.into_iter().collect();
    sorted_suffixes.sort_by_key(|s| s.len());

    // 2. Identify all relevant DFA states.
    let relevant_states: BTreeSet<usize> = initial_states
        .iter()
        .cloned()
        .chain(std::iter::once(regex.dfa.start_state))
        .collect();

    // 3. Compute signatures for all suffixes, from shortest to longest.
    let mut interner = VerificationSignatureInterner::new();
    let mut memo: HashMap<&[u8], BTreeMap<usize, InternedVerificationSignatureId>> = HashMap::new();

    let pb = ProgressBar::new(sorted_suffixes.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%, {eta}) (Verification)")
            .expect("progress-bar"),
    );
    if !PROGRESS_BAR_ENABLED {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    for &suffix in &sorted_suffixes {
        pb.inc(1);
        let mut state_sigs = BTreeMap::new();
        for &dfa_state in &relevant_states {
            let result = regex.execute_from_state2(suffix, dfa_state);

            let mut matches = BTreeSet::new();
            for m in result.matches {
                if m.position == 0 {
                    continue;
                }
                let remainder = &suffix[m.position..];
                let remainder_sigs = memo
                    .get(remainder)
                    .expect("BUG: remainder signature should be pre-computed");
                let remainder_sig_id = *remainder_sigs
                    .get(&regex.dfa.start_state)
                    .expect("BUG: remainder signature for start_state should exist");
                matches.insert((m.group_id, m.position, remainder_sig_id));
            }

            let sig = InternedVerificationSignature {
                matches,
                final_state: result.end_state,
            };
            let sig_id = interner.intern(sig);
            state_sigs.insert(dfa_state, sig_id);
        }
        memo.insert(suffix, state_sigs);
    }
    pb.finish();

    // 4. Classify original strings based on their signature vectors.
    let mut brute_force_classes: BTreeMap<Vec<InternedVerificationSignatureId>, Vec<usize>> = BTreeMap::new();
    for (i, s) in strings.iter().enumerate() {
        let mut signature_vector = Vec::with_capacity(initial_states.len());
        if let Some(string_sigs) = memo.get(s.as_slice()) {
            for &initial_state in initial_states {
                let sig_id = *string_sigs.get(&initial_state).unwrap();
                signature_vector.push(sig_id);
            }
        }
        brute_force_classes.entry(signature_vector).or_default().push(i);
    }

    // 5. Compare partitions.
    let computed_partitions: BTreeSet<BTreeSet<usize>> = computed_classes
        .values()
        .map(|class| class.iter().cloned().collect())
        .collect();

    let brute_force_partitions: BTreeSet<BTreeSet<usize>> = brute_force_classes
        .values()
        .map(|class| class.iter().cloned().collect())
        .collect();

    if computed_partitions == brute_force_partitions {
        println!("Equivalence class verification successful!");
    } else {
        eprintln!("Equivalence class verification FAILED!");
        eprintln!("Computed partitions ({}): {:?}", computed_partitions.len(), computed_partitions);
        eprintln!("Brute-force partitions ({}): {:?}", brute_force_partitions.len(), brute_force_partitions);
        panic!("Equivalence class verification failed.");
    }
}

