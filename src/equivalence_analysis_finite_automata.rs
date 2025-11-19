use std::collections::BTreeMap;
use crate::finite_automata::Regex;

mod new {
    use crate::profiler::PROGRESS_BAR_ENABLED;
    use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
    use std::collections::BTreeMap;
    use hashbrown::HashMap;
    use smallvec::SmallVec;
    use crate::finite_automata::Regex;
    // -----------------------------------------------------------------------------
    // Hashing Utilities
    // -----------------------------------------------------------------------------

    #[inline(always)]
    fn mix_u128(x: u128) -> u128 {
        let mut x = x;
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
        x ^= x >> 33;
        x
    }

    /// Generates a deterministic, odd pseudo-random weight for an initial state index.
    #[inline(always)]
    fn get_init_weight(idx: usize) -> u128 {
        mix_u128((idx as u128) << 1 | 1) | 1
    }

    /// Computes a hash for a specific outcome.
    #[inline(always)]
    fn hash_outcome(
        is_match: bool,
        match_group: u32,
        match_pos: u32,
        remainder_sig: u64,
        final_state: u32,
    ) -> u128 {
        let flags = if is_match { 1 } else { 0 } | (final_state << 1);
        let packed = ((match_pos as u128) << 96)
            | ((match_group as u128) << 64)
            | ((flags as u128) << 32);
        mix_u128(packed ^ (remainder_sig as u128))
    }

    // -----------------------------------------------------------------------------
    // Trie Definition
    // -----------------------------------------------------------------------------

    #[derive(Default, Clone)]
    struct TrieNode {
        // Sparse transitions: (byte, child_index)
        transitions: SmallVec<[(u8, u32); 4]>,
        // If this node terminates a string, which original string index is it?
        terminal_string_idx: Option<u32>,
        // Range of string indices (in the linearized DFS order) covered by this subtree
        range_start: u32,
        range_end: u32,
    }

    struct Trie {
        nodes: Vec<TrieNode>,
    }

    impl Trie {
        fn new() -> Self {
            Trie {
                nodes: vec![TrieNode::default()],
            }
        }

        fn insert(&mut self, s: &[u8], original_idx: u32) {
            let mut node_idx = 0;
            for &b in s {
                let mut found = None;
                for &(byte, child) in &self.nodes[node_idx].transitions {
                    if byte == b {
                        found = Some(child as usize);
                        break;
                    }
                }
                match found {
                    Some(child) => node_idx = child,
                    None => {
                        let new_node_idx = self.nodes.len();
                        self.nodes.push(TrieNode::default());
                        self.nodes[node_idx].transitions.push((b, new_node_idx as u32));
                        node_idx = new_node_idx;
                    }
                }
            }
            self.nodes[node_idx].terminal_string_idx = Some(original_idx);
        }

        fn linearize(&mut self) -> Vec<usize> {
            let mut mapping = Vec::new();
            self.dfs_linearize(0, &mut mapping);
            mapping
        }

        fn dfs_linearize(&mut self, node_idx: usize, mapping: &mut Vec<usize>) {
            let start = mapping.len() as u32;
            if let Some(orig_idx) = self.nodes[node_idx].terminal_string_idx {
                mapping.push(orig_idx as usize);
            }
            self.nodes[node_idx].transitions.sort_unstable_by_key(|k| k.0);
            let children = self.nodes[node_idx].transitions.clone();
            for &(_, child_idx) in &children {
                self.dfs_linearize(child_idx as usize, mapping);
            }
            let end = mapping.len() as u32;
            self.nodes[node_idx].range_start = start;
            self.nodes[node_idx].range_end = end;
        }
    }

    // -----------------------------------------------------------------------------
    // Analysis Logic
    // -----------------------------------------------------------------------------

    pub fn find_equivalence_classes(
        regex: &Regex,
        strings: &[Vec<u8>],
        initial_states: &[usize],
    ) -> BTreeMap<Vec<usize>, Vec<usize>> {
        crate::debug!(
            2,
            "Starting sparse-wavefront equivalence analysis for {} strings and {} states.",
            strings.len(),
            initial_states.len()
        );

        let pb = ProgressBar::new(5);
        pb.set_style(ProgressStyle::default_bar().template("{spinner:.green} {msg}").unwrap());
        if !PROGRESS_BAR_ENABLED {
            pb.set_draw_target(ProgressDrawTarget::hidden());
        }

        // 1. Precompute Remainder Signatures (Fast Path)
        pb.set_message("Precomputing remainder signatures...");
        let remainder_hashes = precompute_remainder_hashes(regex, strings);
        pb.inc(1);

        // 2. Build Inverted Index for DFA
        // Map: byte -> List of states that have a transition on this byte
        pb.set_message("Indexing DFA...");
        let mut states_with_transition: Vec<Vec<u32>> = vec![Vec::new(); 256];
        for (state_idx, state) in regex.dfa.states.iter().enumerate() {
            for (byte, _) in state.transitions.iter() {
                states_with_transition[byte as usize].push(state_idx as u32);
            }
        }
        pb.inc(1);

        // 3. Build Trie
        pb.set_message("Building Trie...");
        let mut trie = Trie::new();
        for (i, s) in strings.iter().enumerate() {
            trie.insert(s, i as u32);
        }
        let linearized_mapping = trie.linearize();
        pb.inc(1);

        // 4. Sparse Wavefront Traversal
        pb.set_message("Wavefront Traversal...");
        let mut accumulators = vec![0u128; strings.len()];
        let mut diffs = vec![0u128; strings.len() + 1];

        // Initial Active States
        let mut root_active: HashMap<u32, u128> = HashMap::new();
        for (idx, &state) in initial_states.iter().enumerate() {
            let w = get_init_weight(idx);
            *root_active.entry(state as u32).or_default() =
                root_active.entry(state as u32).or_default().wrapping_add(w);
        }

        process_node_sparse(
            regex,
            &trie,
            0,
            root_active,
            &states_with_transition,
            &remainder_hashes,
            &linearized_mapping,
            &mut accumulators,
            &mut diffs,
            0,
        );
        pb.inc(1);

        // 5. Finalize
        pb.set_message("Grouping...");
        let mut current_diff = 0u128;
        for (lin_idx, &orig_idx) in linearized_mapping.iter().enumerate() {
            current_diff = current_diff.wrapping_add(diffs[lin_idx]);
            accumulators[orig_idx] = accumulators[orig_idx].wrapping_add(current_diff);
        }

        let mut hash_to_sig_id: HashMap<u128, usize> = HashMap::new();
        let mut next_sig_id = 0;
        let mut classes: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();

        for (str_idx, &h) in accumulators.iter().enumerate() {
            let sig_id = *hash_to_sig_id.entry(h).or_insert_with(|| {
                let id = next_sig_id;
                next_sig_id += 1;
                id
            });
            classes.entry(vec![sig_id]).or_default().push(str_idx);
        }
        pb.finish_with_message("Done");

        classes
    }

    #[allow(clippy::too_many_arguments)]
    fn process_node_sparse(
        regex: &Regex,
        trie: &Trie,
        node_idx: usize,
        active_states: HashMap<u32, u128>, // Sparse map: State -> Weight
        states_with_transition: &[Vec<u32>],
        remainder_hashes: &[Vec<u64>],
        linearized_mapping: &[usize],
        accumulators: &mut Vec<u128>,
        diffs: &mut Vec<u128>,
        depth: u32,
    ) {
        let node = &trie.nodes[node_idx];

        // Handle Terminal (String Ends Here)
        if let Some(_orig_idx) = node.terminal_string_idx {
            let lin_idx = node.range_start as usize;
            // Calculate hash for ending at these states
            // If active_states is huge (Root), this loop is slow.
            // But active_states is only huge at Root.
            // And only empty string ends at Root.
            // So this loop runs once for empty string.
            // For deeper nodes, active_states is small.
            let mut terminal_hash = 0u128;
            for (&state, &weight) in &active_states {
                // Logic: if transitions empty, Dead (0). Else state+1.
                let end_val = if regex.dfa.states[state as usize].transitions.is_empty() {
                    0
                } else {
                    state + 1
                };
                let h = hash_outcome(false, 0, 0, 0, end_val);
                terminal_hash = terminal_hash.wrapping_add(weight.wrapping_mul(h));
            }
            diffs[lin_idx] = diffs[lin_idx].wrapping_add(terminal_hash);
            diffs[lin_idx + 1] = diffs[lin_idx + 1].wrapping_sub(terminal_hash);
        }

        if node.transitions.is_empty() {
            return;
        }

        let total_weight: u128 = active_states.values().fold(0, |a, b| a.wrapping_add(*b));
        let dead_hash_const = hash_outcome(false, 0, 0, 0, 0); // Generic Dead Hash

        for &(byte, child_idx) in &node.transitions {
            // Identify survivors: Intersection of Active States and States with transition on 'byte'
            // Since Active States can be large (at Root), but StatesWithTransition is usually small (except for Start State),
            // we iterate the smaller set if possible.
            // However, HashMap lookup is O(1).
            // So iterating StatesWithTransition and looking up in Active is O(|Transitions|).
            // This is efficient.

            let candidates = &states_with_transition[byte as usize];
            let mut next_active: HashMap<u32, u128> = HashMap::new();
            let mut survivor_weight = 0u128;

            for &state in candidates {
                if let Some(&weight) = active_states.get(&state) {
                    // This state survives
                    survivor_weight = survivor_weight.wrapping_add(weight);

                    let state_data = &regex.dfa.states[state as usize];
                    // We know transition exists because it's in candidates
                    let next_state = *state_data.transitions.get(byte).unwrap() as u32;

                    // Accumulate weight for next state
                    *next_active.entry(next_state).or_default() =
                        next_active.entry(next_state).or_default().wrapping_add(weight);

                    // Check for Matches
                    let next_state_data = &regex.dfa.states[next_state as usize];
                    if !next_state_data.finalizers.is_empty() {
                        // Apply match hash to subtree
                        let child_node = &trie.nodes[child_idx as usize];
                        let r_start = child_node.range_start as usize;
                        let r_end = child_node.range_end as usize;

                        for &group_id in &next_state_data.finalizers {
                            // Iterate subtree for variable remainder hash
                            for lin_idx in r_start..r_end {
                                let orig_idx = linearized_mapping[lin_idx];
                                let rem_sig = remainder_hashes[orig_idx][(depth + 1) as usize];
                                let h = hash_outcome(true, group_id as u32, depth + 1, rem_sig, 0);
                                let contrib = weight.wrapping_mul(h);
                                accumulators[orig_idx] = accumulators[orig_idx].wrapping_add(contrib);
                            }
                        }
                    }
                }
            }

            // Apply Dead Hash to non-survivors
            let dead_weight = total_weight.wrapping_sub(survivor_weight);
            if dead_weight != 0 {
                let child_node = &trie.nodes[child_idx as usize];
                let r_start = child_node.range_start as usize;
                let r_end = child_node.range_end as usize;
                let contrib = dead_weight.wrapping_mul(dead_hash_const);
                diffs[r_start] = diffs[r_start].wrapping_add(contrib);
                diffs[r_end] = diffs[r_end].wrapping_sub(contrib);
            }

            // Recurse if there are survivors
            if !next_active.is_empty() {
                process_node_sparse(
                    regex,
                    trie,
                    child_idx as usize,
                    next_active,
                    states_with_transition,
                    remainder_hashes,
                    linearized_mapping,
                    accumulators,
                    diffs,
                    depth + 1,
                );
            }
        }
    }

    fn precompute_remainder_hashes(regex: &Regex, strings: &[Vec<u8>]) -> Vec<Vec<u64>> {
        let mut results = Vec::with_capacity(strings.len());
        for s in strings {
            let mut row = Vec::with_capacity(s.len() + 1);
            for i in 0..=s.len() {
                let suffix = &s[i..];
                let exec = regex.execute_from_state_fast(suffix, regex.dfa.start_state);
                let mut h = 0u64;
                for m in exec.matches {
                    let k = (m.group_id as u64).wrapping_mul(0x9E3779B97F4A7C15)
                        ^ ((m.position as u64).rotate_left(32));
                    h = h.wrapping_mul(0xC6A4A7935BD1E995).wrapping_add(k);
                }
                let end_val = if let Some(fs) = exec.end_state {
                    (fs as u64).wrapping_add(1)
                } else {
                    0
                };
                h ^= end_val.rotate_left(17);
                row.push(h);
            }
            results.push(row);
        }
        results
    }
}

mod old {
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
                "Starting LLM token equivalence analysis (on-demand) for {} strings (average length: {}) and {} tokenizer states.",
                self.strings.len(),
                self.strings.iter().map(|s| s.len()).sum::<usize>() / self.strings.len(),
                self.initial_states.len(),
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
        computed_classes: &HashMap<Vec<SignatureId>, Vec<usize>>,
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
}

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> BTreeMap<Vec<usize>, Vec<usize>> {
    let new = new::find_equivalence_classes(regex, strings, initial_states);
    let old = old::find_equivalence_classes(regex, strings, initial_states);
    assert_eq!(new, old);
    new
}