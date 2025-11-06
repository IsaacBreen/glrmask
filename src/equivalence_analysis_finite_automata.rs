use crate::finite_automata::{ExecutionResult, GroupID, Match, Regex};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::collections::{BTreeMap, BTreeSet};
use hashbrown::{HashMap, HashSet};
use smallvec::SmallVec;

// For debugging: verify equivalence classes using a brute-force method.
const VERIFY_EQUIVALENCE_CLASSES: bool = false;

// An ID representing a unique canonical signature. In the context of the new algorithm,
// this is simply the final block ID from the partition refinement.
type SignatureId = usize;

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

    /// Finds equivalence classes using an iterative partition refinement algorithm.
    /// This is vastly more efficient than the recursive approach for large inputs.
    pub fn find_equivalence_classes(&mut self) -> BTreeMap<Vec<SignatureId>, Vec<usize>> {
        crate::debug!(
            2,
            "Starting LLM token equivalence analysis (partition refinement) for {} strings...",
            self.strings.len()
        );

        // === Phase 1: Suffix Collection ===
        let mut deduper = SuffixDeduper::new(self.strings);
        let pb_suffixes = ProgressBar::new(self.strings.len() as u64);
        pb_suffixes.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Collecting Suffixes)")
                .expect("progress-bar"),
        );
        if !PROGRESS_BAR_ENABLED {
            pb_suffixes.set_draw_target(ProgressDrawTarget::hidden());
        }
        for (i, s) in self.strings.iter().enumerate() {
            for j in 0..=s.len() {
                deduper.get_or_intern(i, j);
            }
            pb_suffixes.inc(1);
        }
        pb_suffixes.finish_and_clear();
        let num_suffixes = deduper.nodes.len();
        crate::debug!(2, "Found {} unique suffixes.", num_suffixes);

        // === Phase 2: Initial Partition ===
        let mut partitions: Vec<usize> = vec![0; num_suffixes];
        let mut num_blocks = 0;
        {
            let mut initial_partitioner: HashMap<Vec<Option<usize>>, usize> = HashMap::new();
            for i in 0..num_suffixes {
                let suffix_bytes = deduper.slice_of(i);
                let key: Vec<Option<usize>> = self
                    .initial_states
                    .iter()
                    .map(|&state| self.regex.execute_from_state2(suffix_bytes, state).end_state)
                    .collect();

                let block_id = *initial_partitioner.entry(key).or_insert_with(|| {
                    let id = num_blocks;
                    num_blocks += 1;
                    id
                });
                partitions[i] = block_id;
            }
        }
        crate::debug!(2, "Initial partition has {} blocks.", num_blocks);

        // === Phase 3: Iterative Refinement ===
        // This struct represents the complex, one-step behavior of a suffix.
        // It's too expensive to use directly as a HashMap key in a tight loop.
        #[derive(PartialEq, Eq, Hash, Clone)]
        struct OneStepBehavior {
            key_per_state: SmallVec<[(Option<usize>, SmallVec<[(GroupID, usize, usize); 4]>); 1]>,
        }

        let pb_refine = ProgressBar::new_spinner();
        pb_refine.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] Iteration {pos}: {wide_msg}")
                .expect("progress-bar"),
        );
        if !PROGRESS_BAR_ENABLED {
            pb_refine.set_draw_target(ProgressDrawTarget::hidden());
        }

        let mut iteration = 0;
        loop {
            iteration += 1;
            pb_refine.set_position(iteration);
            pb_refine.set_message(format!("{} blocks", num_blocks));

            let mut has_changed = false;
            let mut next_partitions = vec![0; num_suffixes];
            let mut next_block_id_counter = 0;

            let mut blocks: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
            for (suffix_id, &block_id) in partitions.iter().enumerate() {
                blocks[block_id].push(suffix_id);
            }

            for block in blocks {
                if block.len() <= 1 {
                    for suffix_id in block {
                        next_partitions[suffix_id] = next_block_id_counter;
                    }
                    next_block_id_counter += 1;
                    continue;
                }

                // --- FIX START ---
                // We intern the complex `OneStepBehavior` into a simple `usize` ID.
                // This makes the `splitter` HashMap extremely fast, as it now uses
                // `usize` as its key instead of a deeply nested structure.
                let mut interner: HashMap<OneStepBehavior, usize> = HashMap::new();
                let mut splitter: HashMap<usize, Vec<usize>> = HashMap::new();
                // --- FIX END ---

                for &suffix_id in &block {
                    let rep = deduper.nodes[suffix_id];
                    let suffix_bytes = &self.strings[rep.str_idx][rep.offset..];

                    let mut key_per_state = SmallVec::new();
                    for &initial_state in self.initial_states {
                        let result = self.regex.execute_from_state2(suffix_bytes, initial_state);
                        let mut matches = SmallVec::new();
                        for m in result.matches {
                            if m.position == 0 { continue; }
                            let remainder_id = deduper.remainder_of(suffix_id, m.position);
                            let remainder_block_id = partitions[remainder_id];
                            matches.push((m.group_id, m.position, remainder_block_id));
                        }
                        matches.sort_unstable();
                        matches.dedup();
                        key_per_state.push((result.end_state, matches));
                    }

                    let behavior = OneStepBehavior { key_per_state };

                    // --- FIX START ---
                    // Intern the behavior to get a cheap ID, then use that ID as the key.
                    let next_intern_id = interner.len();
                    let interned_id = *interner.entry(behavior).or_insert(next_intern_id);
                    splitter.entry(interned_id).or_default().push(suffix_id);
                    // --- FIX END ---
                }

                if splitter.len() > 1 {
                    has_changed = true;
                }

                for sub_block in splitter.values() {
                    for &suffix_id in sub_block {
                        next_partitions[suffix_id] = next_block_id_counter;
                    }
                    next_block_id_counter += 1;
                }
            }

            partitions = next_partitions;
            num_blocks = next_block_id_counter;

            if !has_changed {
                break;
            }
        }
        pb_refine.finish_with_message(format!("Converged with {} blocks.", num_blocks));

        // === Phase 4: Result Generation ===
        let mut equivalence_classes: BTreeMap<Vec<SignatureId>, Vec<usize>> = BTreeMap::new();
        for i in 0..self.strings.len() {
            let full_string_suffix_id = deduper.get_or_intern(i, 0);
            let final_block_id = partitions[full_string_suffix_id];
            equivalence_classes.entry(vec![final_block_id]).or_default().push(i);
        }

        if VERIFY_EQUIVALENCE_CLASSES {
            let computed_for_verify: HashMap<Vec<SignatureId>, Vec<usize>> =
                equivalence_classes.clone().into_iter().collect();
            verify_equivalence_classes(self.regex, self.strings, self.initial_states, &computed_for_verify);
        }

        equivalence_classes
    }
}

/// Finds equivalence classes among a set of strings based on their tokenization
/// behavior with a given Regex, starting from a set of initial DFA states.
pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> BTreeMap<Vec<SignatureId>, Vec<usize>> {
    let mut analyzer = EquivalenceAnalyzer::new(regex, strings, initial_states);
    analyzer.find_equivalence_classes()
}

/// Brute-force verification of equivalence classes.
fn verify_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
    computed_classes: &HashMap<Vec<SignatureId>, Vec<usize>>,
) {
    println!("Verifying equivalence classes (this may be slow)...");

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

    let mut suffixes = HashSet::new();
    suffixes.insert(&[] as &[u8]);
    for s in strings {
        for i in 0..=s.len() {
            suffixes.insert(&s[i..]);
        }
    }
    let mut sorted_suffixes: Vec<&[u8]> = suffixes.into_iter().collect();
    sorted_suffixes.sort_by_key(|s| s.len());

    let relevant_states: BTreeSet<usize> = initial_states
        .iter()
        .cloned()
        .chain(std::iter::once(regex.dfa.start_state))
        .collect();

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