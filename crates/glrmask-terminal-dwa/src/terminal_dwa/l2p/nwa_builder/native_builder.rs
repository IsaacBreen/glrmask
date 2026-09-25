
//! Experimental direct finite-coefficient specialization of the unchanged
//! terminal event producer. Generic coefficients are never built for emitted
//! transitions. The seeded raw TSID selectors retain their original meanings.
use super::*;
use crate::terminal_dwa::l2p::native_pipeline::post::NativeSink as NWA;

#[derive(Clone,Copy)]
struct Weight {bits:[u64;8],end:u32,valid:bool}
impl Weight {
 fn empty()->Self {Self{bits:[0;8],end:0,valid:true}}
 fn is_empty(&self)->bool {self.valid&&self.bits.iter().all(|&x|x==0)}
 fn from_uniform(rows:std::ops::RangeInclusive<u32>,tokens:RangeSetBlaze<u32>)->Self {
  let mut result=Self{bits:[0;8],end:*rows.end(),valid:*rows.start()==0};
  for range in tokens.ranges() {
   if *range.end()>=512{result.valid=false;continue}
   for id in range{result.bits[id as usize/64]|=1u64<<(id%64);}
  }
  result
 }
 fn union(&self,other:&Self)->Self {
  if self.is_empty(){return *other}if other.is_empty(){return *self}
  let mut result=Self{bits:[0;8],end:self.end,valid:self.valid&&other.valid&&self.end==other.end};
  for k in 0..8{result.bits[k]=self.bits[k]|other.bits[k]}result
 }
}
pub struct TerminalNwaBuilder<'tok, 'pm, 'nwa> {
    tokenizer: &'tok Tokenizer,
    terminal_coloring: TerminalColoring,
    possible_future_terminals: FxHashMap<TokenizerState, Vec<TerminalID>>,
    future_terminal_color_groups: FxHashMap<TokenizerState, FutureTerminalColorGroups>,
    future_terminal_colors: FxHashMap<TokenizerState, SmallVec<[ColorId; 8]>>,
    ignore_terminal_possible: FxHashMap<TokenizerState, bool>,
    possible_matches: &'pm mut PossibleMatchesComputer<'tok>,
    nwa: &'nwa mut NWA,
    num_tsids: u32,
    leaf_state: u32,
    ignore_terminal: Option<TerminalID>,
    /// States reached from the NWA start using only epsilon transitions.
    ///
    /// Ignore is transparent only after a terminal boundary. An ignore match
    /// from one of these sources is the token's first terminal and must remain
    /// labelled so the parser DWA can account for it with the ignore template.
    initial_source_states: Vec<bool>,
    use_terminal_coloring: bool,
    terminal_path_lengths: Option<Vec<TerminalPathLength>>,
    active_terminals: Option<Vec<bool>>,
    self_loop_bytes: FxHashMap<TokenizerState, U8Set>,
    leaf_token_ids_buffer: FxHashMap<(u32,TerminalID),LeafTokenIds>,
    future_leaf_buffer: FxHashMap<(u32, TokenizerState, ColorId), BufferedLeafTransition>,
    deferred_uncolored_future_leaf_buffer: DeferredUncoloredFutureLeafBuffer,
    defer_uncolored_future_leaves: bool,
    reachable_weight_cache: FxHashMap<usize, Weight>,
    pruned_weight_cache: FxHashMap<(usize, usize, u32, TerminalID), Weight>,
    leaf_weight_cache: FxHashMap<LeafTokenIds, Weight>,
    transition_buffer: FxHashMap<(u32, i32, u32), Weight>,
    compact_buffer:Option<native_compact_buffer::CompactBuffer>,
    epsilon_buffer: FxHashMap<(u32, u32), Weight>,
    pub profile: TerminalDwaBuildProfile,
    flat_transitions: Vec<Option<Box<[u32; 256]>>>,
    nfa_scan_cache: Option<NfaTrieScanCache<'tok>>,
    shared_flat_transitions: Option<&'tok [u32]>,
    self_loop_subtree_skip_enabled: bool,
    profile_timing: bool,
    has_epsilon_transitions: bool,
    scalar_deterministic_dispatch: bool,
    scalar_cursor:bool,
    validate_scalar_cursor:bool,
    dfa_scan_strict_reference: bool,
}

#[derive(Default)]
struct BufferedLeafTransition {
    token_ids: LeafTokenIds,
    weight: Option<Weight>,
}

fn deferred_future_token_ids_for(
    buffer: &mut DeferredUncoloredFutureLeafBuffer,
    source: u32,
    tokenizer_state: TokenizerState,
) -> &mut LeafTokenIds {
    let source = source as usize;
    if source >= buffer.len() {
        buffer.resize_with(source + 1, SmallVec::new);
    }
    let groups = &mut buffer[source];
    if let Some(index) = groups
        .iter()
        .position(|(state, _)| *state == tokenizer_state)
    {
        return &mut groups[index].1;
    }
    groups.push((tokenizer_state, LeafTokenIds::new()));
    &mut groups.last_mut().expect("just pushed deferred future group").1
}

impl<'tok, 'pm, 'nwa> TerminalNwaBuilder<'tok, 'pm, 'nwa> {
    pub fn new(
        tokenizer: &'tok Tokenizer,
        terminal_coloring: TerminalColoring,
        possible_matches: &'pm mut PossibleMatchesComputer<'tok>,
        nwa: &'nwa mut NWA,
        num_tsids: u32,
        leaf_state: u32,
        ignore_terminal: Option<TerminalID>,
        initial_source_states: Vec<bool>,
        use_terminal_coloring: bool,
        terminal_path_lengths: Option<Vec<TerminalPathLength>>,
        active_terminals: Option<Vec<bool>>,
        num_tokenizer_states: usize,
        shared_flat_transitions: Option<&'tok [u32]>,
    ) -> Self {
        let has_epsilon_transitions = tokenizer.has_epsilon_transitions();
        let scalar_deterministic_dispatch = tokenizer.has_scalar_deterministic_dispatch();
        let nfa_scan_cache = has_epsilon_transitions
            .then(|| NfaTrieScanCache::new(tokenizer, active_terminals.clone()));
        Self {
            tokenizer,
            terminal_coloring,
            possible_future_terminals: FxHashMap::default(),
            future_terminal_color_groups: FxHashMap::default(),
            future_terminal_colors: FxHashMap::default(),
            ignore_terminal_possible: FxHashMap::default(),
            possible_matches,
            nwa,
            num_tsids,
            leaf_state,
            ignore_terminal,
            initial_source_states,
            use_terminal_coloring,
            terminal_path_lengths,
            active_terminals,
            self_loop_bytes: FxHashMap::default(),
            leaf_token_ids_buffer: FxHashMap::default(),
            future_leaf_buffer: FxHashMap::default(),
            deferred_uncolored_future_leaf_buffer: Vec::new(),
            defer_uncolored_future_leaves: std::env::var_os(
                "GLRMASK_DISABLE_L2P_DEFER_UNCOLORED_FUTURE_LEAVES",
            )
            .is_none(),
            reachable_weight_cache: FxHashMap::default(),
            pruned_weight_cache: FxHashMap::default(),
            leaf_weight_cache: FxHashMap::default(),
            transition_buffer: FxHashMap::default(),
            compact_buffer:std::env::var_os("GLRMASK_BOUNDARY_NATIVE_COMPACT_EVENT_BUFFER").is_some().then(native_compact_buffer::CompactBuffer::default),
            epsilon_buffer: FxHashMap::default(),
            profile: TerminalDwaBuildProfile::default(),
            flat_transitions: vec![None; num_tokenizer_states],
            nfa_scan_cache,
            shared_flat_transitions: shared_flat_transitions
                .filter(|transitions| transitions.len() == num_tokenizer_states * 256),
            self_loop_subtree_skip_enabled: std::env::var_os(
                "GLRMASK_ENABLE_L2P_SELF_LOOP_SUBTREE_SKIP",
            )
            .is_some(),
            profile_timing: std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some(),
            has_epsilon_transitions,
            scalar_deterministic_dispatch,
            scalar_cursor:std::env::var_os("GLRMASK_BOUNDARY_NATIVE_SCALAR_CURSOR").is_some(),
            validate_scalar_cursor:std::env::var_os("GLRMASK_VALIDATE_NATIVE_SCALAR_CURSOR").is_some(),
            dfa_scan_strict_reference: std::env::var_os(
                "GLRMASK_L2P_NWA_DFA_SCAN_STRICT_REFERENCE",
            )
            .is_some(),
        }
    }

    /// O(1) DFA step using lazily-built flat transition table.
    #[inline]
    fn fast_step(&mut self, state: u32, byte: u8) -> Option<u32> {
        let state_idx = state as usize;
        if self.flat_transitions[state_idx].is_none() {
            self.flat_transitions[state_idx] = Some(self.tokenizer.transition_row(state));
        }
        let next = self.flat_transitions[state_idx].as_ref().unwrap()[byte as usize];
        if next == u32::MAX { None } else { Some(next) }
    }

    fn leaf_token_ids_for(&mut self,source:u32,label:TerminalID)->&mut LeafTokenIds {
        self.leaf_token_ids_buffer.entry((source,label)).or_default()
    }

    fn buffer_leaf_token_id(&mut self, source: u32, label: TerminalID, internal_token_id: u32) {
        self.leaf_token_ids_for(source, label).push(internal_token_id);
    }

    fn terminal_is_active(&self, terminal: TerminalID) -> bool {
        self.active_terminals.as_ref().map_or(true, |active| {
            active.get(terminal as usize).copied().unwrap_or(false)
        })
    }

    fn possible_future_terminals_for_state(&mut self, tokenizer_state: TokenizerState) -> Vec<TerminalID> {
        if let Some(existing)=self.possible_future_terminals.get(&tokenizer_state){return existing.clone()}
        let active = self.active_terminals.clone();
        self.possible_future_terminals
            .entry(tokenizer_state)
            .or_insert_with(|| {
                self.tokenizer
                    .possible_future_terminals_iter(tokenizer_state)
                    .filter(|&terminal| active.as_ref().map_or(true, |mask| {
                        mask.get(terminal as usize).copied().unwrap_or(false)
                    }))
                    .collect()
            })
            .clone()
    }

    fn populate_future_terminal_color_cache(&mut self, tokenizer_state: TokenizerState) {
        if self.future_terminal_color_groups.contains_key(&tokenizer_state) {
            return;
        }

        let mut groups = BTreeMap::<ColorId, SmallVec<[TerminalID; 4]>>::new();
        let mut colors = SmallVec::<[ColorId; 8]>::new();
        let mut ignore_present = false;

        for terminal_id in self.tokenizer.possible_future_terminals_iter(tokenizer_state) {
            if !self.terminal_is_active(terminal_id) {
                continue;
            }
            if Some(terminal_id) == self.ignore_terminal {
                ignore_present = true;
                continue;
            }
            let color = self.terminal_coloring.color_for(terminal_id);
            let entry = groups.entry(color).or_default();
            if entry.is_empty() {
                colors.push(color);
            }
            entry.push(terminal_id);
        }

        self.future_terminal_color_groups
            .insert(tokenizer_state, groups.into_iter().collect());
        self.future_terminal_colors.insert(tokenizer_state, colors);
        self.ignore_terminal_possible
            .insert(tokenizer_state, ignore_present);
    }

    fn ignore_terminal_possible_for_state(&mut self, tokenizer_state: TokenizerState) -> bool {
        if self.ignore_terminal.is_none() {
            return false;
        }
        self.populate_future_terminal_color_cache(tokenizer_state);
        self.ignore_terminal_possible
            .get(&tokenizer_state)
            .copied()
            .unwrap_or(false)
    }

    fn future_terminal_colors_for_state(
        &mut self,
        tokenizer_state: TokenizerState,
    ) -> SmallVec<[ColorId; 8]> {
        self.populate_future_terminal_color_cache(tokenizer_state);
        self.future_terminal_colors
            .get(&tokenizer_state)
            .cloned()
            .unwrap_or_default()
    }

    fn future_terminal_color_groups_for_state(
        &mut self,
        tokenizer_state: TokenizerState,
    ) -> FutureTerminalColorGroups {
        self.populate_future_terminal_color_cache(tokenizer_state);
        self.future_terminal_color_groups
            .get(&tokenizer_state)
            .cloned()
            .unwrap_or_default()
    }

    fn buffer_future_leaf_token_id(
        &mut self,
        source: u32,
        internal_tsid: TokenizerState,
        color: ColorId,
        internal_token_id: u32,
    ) {
        self.profile.future_terminal_additions += 1;
        self.future_leaf_buffer
            .entry((source, internal_tsid, color))
            .or_default()
            .token_ids
            .push(internal_token_id);
    }

    fn add_future_leaf_token_from_sources(
        &mut self,
        sources: &[u32],
        tokenizer_state: TokenizerState,
        internal_token_id: u32,
    ) {
        if !self.use_terminal_coloring {
            if self.defer_uncolored_future_leaves {
                // Ignore is the one future label whose first-vs-later source
                // semantics cannot be deferred into an ordinary leaf edge.
                // Preserve that label eagerly and defer every other label as
                // one token-set contribution per (source, scanner state).
                if let Some(ignore_terminal) = self.ignore_terminal
                    && self.ignore_terminal_possible_for_state(tokenizer_state)
                {
                    self.profile.future_terminal_additions += sources.len() as u64;
                    self.add_leaf_token_from_sources(
                        sources,
                        ignore_terminal,
                        internal_token_id,
                    );
                }
                for &source in sources {
                    deferred_future_token_ids_for(
                        &mut self.deferred_uncolored_future_leaf_buffer,
                        source,
                        tokenizer_state,
                    )
                    .push(internal_token_id);
                }
                return;
            }

            let future_terminals = self.possible_future_terminals_for_state(tokenizer_state);
            self.profile.future_terminal_additions +=
                (sources.len() * future_terminals.len()) as u64;
            for terminal_id in future_terminals {
                self.add_leaf_token_from_sources(sources, terminal_id, internal_token_id);
            }
            return;
        }

        if let Some(ignore_terminal) = self.ignore_terminal {
            if self.ignore_terminal_possible_for_state(tokenizer_state) {
                self.profile.future_terminal_additions += sources.len() as u64;
                self.add_leaf_token_from_sources(sources, ignore_terminal, internal_token_id);
            }
        }

        let colors = self.future_terminal_colors_for_state(tokenizer_state);
        for color in colors {
            for &source in sources {
                self.buffer_future_leaf_token_id(source, tokenizer_state, color, internal_token_id);
            }
        }
    }

    fn add_future_weighted_match_from_sources(
        &mut self,
        sources: &[u32],
        tokenizer_state: TokenizerState,
        weight: &Weight,
    ) {
        if !self.use_terminal_coloring {
            let future_terminals = self.possible_future_terminals_for_state(tokenizer_state);
            self.profile.future_terminal_additions +=
                (sources.len() * future_terminals.len()) as u64;
            for terminal_id in future_terminals {
                self.add_match_from_sources(sources, terminal_id, self.leaf_state, weight);
            }
            return;
        }

        if let Some(ignore_terminal) = self.ignore_terminal {
            if self.ignore_terminal_possible_for_state(tokenizer_state) {
                self.profile.future_terminal_additions += sources.len() as u64;
                self.add_match_from_sources(sources, ignore_terminal, self.leaf_state, weight);
            }
        }

        let colors = self.future_terminal_colors_for_state(tokenizer_state);
        for color in colors {
            if weight.is_empty() {
                continue;
            }
            for &source in sources {
                self.profile.future_terminal_additions += 1;
                let entry = self.future_leaf_buffer
                    .entry((source, tokenizer_state, color))
                    .or_default();
                if let Some(existing) = &mut entry.weight {
                    *existing = existing.union(weight);
                } else {
                    entry.weight = Some(weight.clone());
                }
            }
        }
    }

    fn cached_reachable_weight(&mut self, token_ids: &RangeSetBlaze<usize>) -> Weight {
        let cache_key = token_ids as *const RangeSetBlaze<usize> as usize;
        if let Some(weight) = self.reachable_weight_cache.get(&cache_key) {
            return weight.clone();
        }

        let weight = self.token_set_weight_fast(token_ids);
        self.reachable_weight_cache.insert(cache_key, weight.clone());
        weight
    }

    /// Build a weight covering all tsids for the given set of internal token IDs.
    fn token_set_weight_fast(&self,tokens:&RangeSetBlaze<usize>)->Weight {
        if self.num_tsids==0{return Weight::empty()}
        let mut result=Weight{bits:[0;8],end:self.num_tsids-1,valid:true};
        for range in tokens.ranges(){
            if *range.end()>=512{result.valid=false;continue}
            for t in range{result.bits[t/64]|=1u64<<(t%64);}
        }
        result
    }

    fn cached_leaf_weight(&mut self,tokens:LeafTokenIds)->Weight {
        if self.num_tsids==0{return Weight::empty()}
        let mut result=Weight{bits:[0;8],end:self.num_tsids-1,valid:true};
        for t in tokens {if t>=512{result.valid=false}else{result.bits[t as usize/64]|=1u64<<(t%64);}}
        result
    }

    fn continuation_weight_for_match(
        &mut self,
        child_node: &VocabPrefixTreeNode,
        leaf_token_id: u32,
        terminal_id: TerminalID,
        end_state: Option<u32>,
        remaining_segment: &[u8],
    ) -> Option<Weight> {
        if end_state.is_none() && !(remaining_segment.is_empty() && child_node.has_token()) {
            return Some(self.cached_reachable_weight(child_node.reachable_token_ids()));
        }

        let remove_leaf = remaining_segment.is_empty() && child_node.has_token();
        let possible_matches = end_state.map(|end_state| {
            self.possible_matches
                .possible_matches_for_suffix_and_node(
                    remaining_segment,
                    child_node,
                    end_state,
                )
        });
        let matches_for_terminal = possible_matches
            .as_ref()
            .and_then(|matches| matches.get(&terminal_id));

        // Most continuations do not actually exclude anything. In that case the
        // exact continuation domain is simply the child's reachable-token set,
        // whose uniform weight is already cached by node identity. Avoid cloning
        // the token set and rebuilding/reinterning the same weight per match.
        if !remove_leaf && matches_for_terminal.is_none() {
            return Some(self.cached_reachable_weight(child_node.reachable_token_ids()));
        }

        // The context-keyed cache is useful only for the rare genuinely pruned
        // continuations. Checking it after the no-pruning fast path avoids a hash
        // lookup on the overwhelmingly common case.
        let cache_key = (
            child_node as *const VocabPrefixTreeNode as usize,
            remaining_segment.len(),
            end_state.unwrap_or(u32::MAX),
            terminal_id,
        );
        if let Some(weight) = self.pruned_weight_cache.get(&cache_key) {
            return Some(weight.clone());
        }

        let mut remaining = child_node.reachable_token_ids().clone();
        if remove_leaf {
            remaining.remove(leaf_token_id as usize);
        }
        if let Some(matches_for_terminal) = matches_for_terminal {
            subtract_possible_matches(&mut remaining, matches_for_terminal);
        }

        if remaining.is_empty() {
            return None;
        }

        let weight = self.token_set_weight_fast(&remaining);
        self.pruned_weight_cache.insert(cache_key, weight.clone());
        Some(weight)
    }

    fn add_leaf_token_from_sources(
        &mut self,
        sources: &[u32],
        label: TerminalID,
        internal_token_id: u32,
    ) {
        if self.ignore_terminal == Some(label) {
            let weight = if self.num_tsids == 0 {
                Weight::empty()
            } else {
                Weight::from_uniform(
                    0..=self.num_tsids - 1,
                    RangeSetBlaze::from_iter([internal_token_id..=internal_token_id]),
                )
            };
            self.add_match_from_sources(sources, label, self.leaf_state, &weight);
            return;
        }

        for &source in sources {
            self.buffer_leaf_token_id(source, label, internal_token_id);
        }
    }

    fn can_skip_self_loop_subtree(
        &mut self,
        node: &VocabPrefixTreeNode,
        tokenizer_state: TokenizerState,
    ) -> bool {
        if self.has_epsilon_transitions
            && !self.scalar_deterministic_dispatch
        {
            return false;
        }
        self.profile.trie_self_loop_checks += 1;
        if let Some(flat_transitions) = self.shared_flat_transitions {
            let base = tokenizer_state as usize * 256;
            let can_skip = U8Set::from_words(*node.subtree_bytes())
                .iter()
                .all(|byte| flat_transitions[base + byte as usize] == tokenizer_state);
            if can_skip {
                self.profile.trie_self_loop_skips += 1;
            }
            return can_skip;
        }
        if !self.self_loop_bytes.contains_key(&tokenizer_state) {
            self.profile.trie_self_loop_cache_misses += 1;
        }
        let self_loop_bytes = self
            .self_loop_bytes
            .entry(tokenizer_state)
            .or_insert_with(|| self.tokenizer.self_loop_bytes(tokenizer_state));
        let can_skip = U8Set::from_words(*node.subtree_bytes()).is_subset(self_loop_bytes);
        if can_skip {
            self.profile.trie_self_loop_skips += 1;
        }
        can_skip
    }

    fn emit_self_loop_leaf_only_subtree(
        &mut self,
        node: &VocabPrefixTreeNode,
        assoc_by_state: &NodesByTokenizerState,
    ) {
        let mut accessible = node.reachable_token_ids().clone();
        if node.has_token() {
            accessible.remove(node.token_id() as usize);
        }
        if accessible.is_empty() {
            return;
        }
        let accessible_weight = self.token_set_weight_fast(&accessible);
        for (internal_tsid, source_nodes) in assoc_by_state.iter() {
            self.add_future_weighted_match_from_sources(
                source_nodes,
                internal_tsid,
                &accessible_weight,
            );
        }
    }

    fn add_match_from_sources(
        &mut self,
        sources: &[u32],
        label: TerminalID,
        target: u32,
        weight: &Weight,
    ) {
        let lower_ignore_to_epsilon = self.ignore_terminal == Some(label);
        for &source in sources {
            if lower_ignore_to_epsilon
                && !self
                    .initial_source_states
                    .get(source as usize)
                    .copied()
                    .unwrap_or(false)
            {
                self.epsilon_buffer
                    .entry((source, target))
                    .and_modify(|existing| *existing = existing.union(weight))
                    .or_insert_with(|| weight.clone());
            } else {
                if let Some(buffer)=&mut self.compact_buffer{buffer.add((source,label as i32,target),weight);continue}
                self.transition_buffer
                    .entry((source, label as i32, target))
                    .and_modify(|existing| *existing = existing.union(weight))
                    .or_insert_with(|| weight.clone());
            }
        }
    }

    pub fn flush_transition_buffer(&mut self) {
        let flush_start = std::time::Instant::now();
        let mut leaf_transition_buckets: Vec<FxHashMap<i32, BufferedLeafTransition>> =
            (0..self.nwa.states_len()).map(|_| FxHashMap::default()).collect();

        let leaf_buf_count = self.leaf_token_ids_buffer.len();
        for ((from,label),token_ids) in std::mem::take(&mut self.leaf_token_ids_buffer) {
            if !token_ids.is_empty(){leaf_transition_buckets[from as usize].entry(label as i32).or_default().token_ids.extend(token_ids);}
        }
        let flush_leaf_ms = flush_start.elapsed().as_secs_f64() * 1000.0;

        let deferred_uncolored =
            std::mem::take(&mut self.deferred_uncolored_future_leaf_buffer);
        if deferred_uncolored.iter().any(|groups| !groups.is_empty()) {
            let mut terminal_cache = FxHashMap::<TokenizerState, Vec<TerminalID>>::default();
            for groups in &deferred_uncolored {
                for &(tokenizer_state, _) in groups {
                    terminal_cache.entry(tokenizer_state).or_insert_with(|| {
                        self.possible_future_terminals_for_state(tokenizer_state)
                    });
                }
            }

            for (source, groups) in deferred_uncolored.into_iter().enumerate() {
                for (tokenizer_state, token_ids) in groups {
                    if token_ids.is_empty() {
                        continue;
                    }
                    let token_id_count = token_ids.len() as u64;
                    let terminals = &terminal_cache[&tokenizer_state];
                    for &terminal_id in terminals {
                        if Some(terminal_id) == self.ignore_terminal {
                            continue;
                        }
                        // Preserve the historical profile counter semantics: it
                        // counts logical terminal-token additions, even though the
                        // deferred path now realizes the whole token set at once.
                        self.profile.future_terminal_additions += token_id_count;
                        let entry = leaf_transition_buckets[source]
                            .entry(terminal_id as i32)
                            .or_default();
                        // Keep deferred future leaves as raw token IDs until the
                        // existing final per-(source, terminal) bucket pass.  The
                        // older deferred implementation eagerly materialized a
                        // Weight for every (source, scanner-state) group and then
                        // repeatedly unioned those weights here, which moved work
                        // out of the trie only to make flush substantially slower.
                        // Appending IDs preserves the exact eager semantics and
                        // lets `cached_leaf_weight` run once for the final bucket.
                        entry.token_ids.extend_from_slice(&token_ids);
                    }
                }
            }
        }

        let future_buf_count = self.future_leaf_buffer.len();
        let flush_future_start = std::time::Instant::now();
        let buffer = std::mem::take(&mut self.future_leaf_buffer);

        // Pre-compute terminal lookups to avoid repeated clone+find_map (157K→~100 unique keys)
        let mut terminal_cache: FxHashMap<(TokenizerState, ColorId), SmallVec<[TerminalID; 4]>> =
            FxHashMap::default();
        for &(_, tokenizer_state, color) in buffer.keys() {
            terminal_cache
                .entry((tokenizer_state, color))
                .or_insert_with(|| {
                    let groups = self.future_terminal_color_groups_for_state(tokenizer_state);
                    groups
                        .iter()
                        .find_map(|(gc, ts)| (*gc == color).then_some(ts.clone()))
                        .unwrap_or_default()
                });
        }

        for ((source, tokenizer_state, color), buffered) in buffer {
            if buffered.token_ids.is_empty() && buffered.weight.as_ref().map_or(true, |w| w.is_empty()) {
                continue;
            }
            let terminals = &terminal_cache[&(tokenizer_state, color)];
            for &terminal_id in terminals {
                let entry = leaf_transition_buckets[source as usize]
                    .entry(terminal_id as i32)
                    .or_default();
                if !buffered.token_ids.is_empty() {
                    entry.token_ids.extend_from_slice(&buffered.token_ids);
                }
                if let Some(w) = &buffered.weight {
                    if let Some(existing) = &mut entry.weight {
                        *existing = existing.union(w);
                    } else {
                        entry.weight = Some(w.clone());
                    }
                }
            }
        }
        let flush_future_ms = flush_future_start.elapsed().as_secs_f64() * 1000.0;

        let flush_weight_start = std::time::Instant::now();
        let mut epsilon_entries: Vec<_> = std::mem::take(&mut self.epsilon_buffer).into_iter().collect();
        epsilon_entries.sort_unstable_by_key(|((from, target), _)| (*from, *target));
        for ((from, target), weight) in epsilon_entries {
            self.nwa.append_epsilon(from,target,weight.end,weight.valid,&weight.bits);
        }

        if let Some(buffer)=&mut self.compact_buffer{buffer.flush(self.nwa);}
        let mut transition_entries: Vec<_> = std::mem::take(&mut self.transition_buffer).into_iter().collect();
        transition_entries.sort_unstable_by_key(|((from, label, target), _)| (*from, *label, *target));
        for ((from, label, target), weight) in transition_entries {
            self.nwa.append_transition(from,label,target,weight.end,weight.valid,&weight.bits);
        }

        for (from, bucket) in leaf_transition_buckets.into_iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }

            let mut entries: Vec<(i32, BufferedLeafTransition)> = bucket.into_iter().collect();
            entries.sort_unstable_by_key(|(label, _)| *label);

            let mut finalized_entries = Vec::with_capacity(entries.len());
            for (label, mut entry) in entries {
                let mut weight = entry.weight.take().unwrap_or_else(Weight::empty);
                if !entry.token_ids.is_empty() {
                    let token_weight = self.cached_leaf_weight(entry.token_ids);
                    weight = if weight.is_empty() {
                        token_weight
                    } else {
                        weight.union(&token_weight)
                    };
                }
                if !weight.is_empty() {
                    finalized_entries.push((label, weight));
                }
            }

            for (label, weight) in finalized_entries {
                self.nwa.append_transition(from as u32,label,self.leaf_state,weight.end,weight.valid,&weight.bits);
            }
        }
        let flush_weight_ms = flush_weight_start.elapsed().as_secs_f64() * 1000.0;
        self.profile.flush_leaf_ms = flush_leaf_ms;
        self.profile.flush_future_ms = flush_future_ms;
        self.profile.flush_weight_ms = flush_weight_ms;
    }

    pub fn build_from_trie(
        &mut self,
        node: &VocabPrefixTreeNode,
        assoc_by_state: &NodesByTokenizerState,
    ) {
        if !self.self_loop_subtree_skip_enabled {
            for (segment_bytes, child_node) in node.iter_children() {
                let next_level_nodes =
                    self.process_child_segment(segment_bytes, child_node, assoc_by_state);
                if !next_level_nodes.is_empty() {
                    self.build_from_trie(child_node, &next_level_nodes);
                }
            }
            return;
        }

        let self_loop_started_at = self.profile_timing.then(std::time::Instant::now);
        let mut recursive_nodes = NodesByTokenizerState::new();
        let mut self_loop_only_nodes = NodesByTokenizerState::new();
        for (tokenizer_state, source_nodes) in assoc_by_state.iter() {
            self.profile.trie_self_loop_source_nodes += source_nodes.len() as u64;
            if self.can_skip_self_loop_subtree(node, tokenizer_state) {
                self.profile.trie_self_loop_skipped_source_nodes += source_nodes.len() as u64;
                self_loop_only_nodes.merge(tokenizer_state, source_nodes);
            } else {
                recursive_nodes.merge(tokenizer_state, source_nodes);
            }
        }
        if let Some(started_at) = self_loop_started_at {
            self.profile.trie_self_loop_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }

        if !self_loop_only_nodes.is_empty() {
            self.emit_self_loop_leaf_only_subtree(node, &self_loop_only_nodes);
        }

        if recursive_nodes.is_empty() {
            return;
        }

        for (segment_bytes, child_node) in node.iter_children() {
            let next_level_nodes = self.process_child_segment(
                segment_bytes,
                child_node,
                &recursive_nodes,
            );
            if !next_level_nodes.is_empty() {
                self.build_from_trie(child_node, &next_level_nodes);
            }
        }
    }

    fn process_child_segment(
        &mut self,
        segment_bytes: &[u8],
        child_node: &VocabPrefixTreeNode,
        initial_nodes: &NodesByTokenizerState,
    ) -> NodesByTokenizerState {
        // Token IDs in the trie are already internal (equivalence class) IDs.
        let leaf_token_id = child_node.token_id() as u32;
        let mut next_level_nodes = NodesByTokenizerState::new();
        let mut pending_by_offset = BTreeMap::<usize, NodesByTokenizerState>::new();
        pending_by_offset.insert(0, initial_nodes.clone());

        // Reusable DFA longest-match buffer. The pre-epsilon lexer path used
        // this scalar scanner and is dramatically faster than routing every
        // compressed trie segment through the general state-set executor.
        let mut match_map_buf = FxHashMap::<TerminalID, (usize, u32)>::default();
        let mut matches_buf: Vec<TokenizerMatch> = Vec::new();

        while let Some((offset, nodes_at_offset)) = pending_by_offset.pop_first() {
            if offset == segment_bytes.len() {
                for (tokenizer_state, nwa_states) in nodes_at_offset {
                    next_level_nodes.merge(tokenizer_state, &nwa_states);
                }
                continue;
            }

            for (tokenizer_state, source_nodes) in nodes_at_offset {
                let remaining = &segment_bytes[offset..];
                let execute_started_at = self.profile_timing.then(std::time::Instant::now);
                let end_states = if self.has_epsilon_transitions {
                    let cache=self.nfa_scan_cache.as_mut().expect("epsilon tokenizer must initialize NFA trie scan cache");
                    if self.scalar_cursor {
                        cache.execute_native_scalar_into(remaining,tokenizer_state,&mut matches_buf,
                            self.shared_flat_transitions,self.validate_scalar_cursor)
                    }else{cache.execute_into(remaining,tokenizer_state,&mut matches_buf)}
                } else {
                    match_map_buf.clear();
                    let mut scan_state = tokenizer_state;
                    let mut scan_alive = true;
                    for (index, &byte) in segment_bytes[offset..].iter().enumerate() {
                        if let Some(next) = self.fast_step(scan_state, byte) {
                            scan_state = next;
                            for terminal in self.tokenizer.matched_terminals_iter(scan_state) {
                                if !self.terminal_is_active(terminal) {
                                    continue;
                                }
                                match_map_buf.insert(terminal, (index + 1, scan_state));
                            }
                        } else {
                            scan_alive = false;
                            break;
                        }
                    }

                    matches_buf.clear();
                    matches_buf.extend(match_map_buf.iter().map(
                        |(&id, &(width, end_state))| TokenizerMatch {
                            id,
                            width,
                            end_state,
                        },
                    ));

                    let end_states = if scan_alive {
                        SmallVec::from_buf([scan_state])
                    } else {
                        SmallVec::new()
                    };

                    if self.dfa_scan_strict_reference {
                        let mut reference = self
                            .tokenizer
                            .execute_from_state(remaining, tokenizer_state);
                        reference
                            .matches
                            .retain(|matched| self.terminal_is_active(matched.id));
                        let mut actual_matches = matches_buf.clone();
                        actual_matches.sort_unstable_by_key(|matched| {
                            (matched.id, matched.width, matched.end_state)
                        });
                        reference.matches.sort_unstable_by_key(|matched| {
                            (matched.id, matched.width, matched.end_state)
                        });
                        assert_eq!(
                            end_states, reference.end_state,
                            "historical DFA trie scanner end-state mismatch"
                        );
                        assert_eq!(
                            actual_matches, reference.matches,
                            "historical DFA trie scanner longest-match mismatch"
                        );
                    }

                    end_states
                };
                if let Some(started_at) = execute_started_at {
                    self.profile.trie_execute_ms +=
                        started_at.elapsed().as_secs_f64() * 1000.0;
                    self.profile.trie_execute_calls += 1;
                    self.profile.trie_execute_input_bytes += remaining.len() as u64;
                    self.profile.trie_matches += matches_buf.len() as u64;
                }

                let end_state_started_at = self.profile_timing.then(std::time::Instant::now);
                for &end_state in &end_states {
                    if child_node.has_token() {
                        self.add_future_leaf_token_from_sources(
                            &source_nodes,
                            end_state,
                            leaf_token_id,
                        );
                    }

                    next_level_nodes.merge(end_state, &source_nodes);
                }
                if let Some(started_at) = end_state_started_at {
                    self.profile.trie_end_state_ms +=
                        started_at.elapsed().as_secs_f64() * 1000.0;
                    self.profile.trie_end_states += end_states.len() as u64;
                }

                let match_process_started_at = self.profile_timing.then(std::time::Instant::now);
                for matched in &matches_buf {
                    let next_offset = offset + matched.width;

                    if next_offset == segment_bytes.len()
                        && child_node.has_token()
                        && ((self.has_epsilon_transitions
                            && !self.scalar_deterministic_dispatch)
                            || !end_states.iter().copied().any(|s| {
                                self.possible_future_terminals_for_state(s)
                                    .contains(&matched.id)
                            }))
                    {
                        self.profile.match_transition_additions += source_nodes.len() as u64;
                        self.add_leaf_token_from_sources(
                            &source_nodes,
                            matched.id,
                            leaf_token_id,
                        );
                    }

                    // L1 terminals never appear in multi-terminal paths, so
                    // skip continuation processing (no second terminal will
                    // follow within the same token).
                    if let Some(ref lengths) = self.terminal_path_lengths {
                        if let Some(&TerminalPathLength::One) = lengths.get(matched.id as usize) {
                            continue;
                        }
                    }

                    let continuation_weight_started_at =
                        self.profile_timing.then(std::time::Instant::now);
                    let continuation_weight = self.continuation_weight_for_match(
                        child_node,
                        leaf_token_id,
                        matched.id,
                        Some(matched.end_state),
                        &segment_bytes[next_offset..],
                    );
                    if let Some(started_at) = continuation_weight_started_at {
                        self.profile.trie_continuation_weight_ms +=
                            started_at.elapsed().as_secs_f64() * 1000.0;
                    }
                    let Some(continuation_weight) = continuation_weight else {
                        continue;
                    };
                    if continuation_weight.is_empty() {
                        continue;
                    }

                    let continuation_nodes = pending_by_offset
                        .entry(next_offset)
                        .or_insert_with(NodesByTokenizerState::new);
                    let reset_states = self.tokenizer.deterministic_reset_states();
                    let destination = ensure_continuation_state(
                        continuation_nodes,
                        &reset_states,
                        self.nwa,
                    );

                    self.profile.match_transition_additions += source_nodes.len() as u64;
                    self.add_match_from_sources(
                        &source_nodes,
                        matched.id,
                        destination,
                        &continuation_weight,
                    );
                }
                if let Some(started_at) = match_process_started_at {
                    self.profile.trie_match_process_ms +=
                        started_at.elapsed().as_secs_f64() * 1000.0;
                }
            }
        }

        next_level_nodes
    }
}
fn ensure_continuation_state(
    pending: &mut NodesByTokenizerState,
    tokenizer_states: &[TokenizerState],
    nwa: &mut NWA,
) -> NwaState {
    if let Some(existing) = pending.first_any() {
        return existing;
    }

    let state = nwa.add_state();
    for &tokenizer_state in tokenizer_states {
        pending.push_one(tokenizer_state, state);
    }
    state
}


pub struct NativeBuild {
 pub sink:NWA,
 pub profile:TerminalDwaBuildProfile,
 pub build_ms:f64,
}

#[allow(clippy::too_many_arguments)]
pub fn build<'a>(tokenizer:&'a Tokenizer,coloring:&TerminalColoring,ignore:Option<TerminalID>,
 seed:&crate::automata::weighted_u32::nwa::NWA,leaf:u32,num_tsids:u32,
 tree:&VocabPrefixTreeNode,roots:&NodesByTokenizerState,flat:Option<&'a [u32]>,active:&[bool])->Option<NativeBuild>{
 let parallel=std::env::var_os("GLRMASK_DISABLE_L2P_PARALLEL_ROOT_TRIE").is_none()
   && std::env::var_os("GLRMASK_ENABLE_L2P_SELF_LOOP_SUBTREE_SKIP").is_none()
   && tree.children().len()>=2 && rayon::current_num_threads()>1
   && (std::env::var_os("GLRMASK_L2P_PARALLEL_ROOT_TRIE").is_some()||tree.reachable_token_ids().len()>=512);
 if parallel||tree.reachable_token_ids().is_empty()||tree.reachable_token_ids().iter().any(|id|id>=512)
   ||tokenizer.num_states()>=16384||active.len()>65536||num_tsids==0{return None}
 let started=std::time::Instant::now();let mut sink=NWA::from_seed(seed,num_tsids)?;
 let mut initial=vec![false;seed.states().len()];for(_,sources)in roots.iter(){for &s in sources{*initial.get_mut(s as usize)?=true;}}
 let mut pm=PossibleMatchesComputer::new(tokenizer);
 let mut builder=TerminalNwaBuilder::new(tokenizer,coloring.clone(),&mut pm,&mut sink,num_tsids,leaf,ignore,
   initial,false,None,Some(active.to_vec()),tokenizer.num_states() as usize,flat);
 let t=std::time::Instant::now();builder.build_from_trie(tree,roots);builder.profile.trie_walk_ms=t.elapsed().as_secs_f64()*1000.;
 if std::env::var_os("GLRMASK_PROFILE_NATIVE_BUILDER").is_some(){
  let slots=builder.leaf_token_ids_buffer.len();
  let live=builder.leaf_token_ids_buffer.values().filter(|x|!x.is_empty()).count();
  eprintln!("[glrmask/profile][native_builder_buffers] sparse_leaf_slots={slots} nonempty_leaf_slots={live} estimated_entry_bytes={} transitions={} epsilons={}",slots*std::mem::size_of::<LeafTokenIds>(),builder.transition_buffer.len()+builder.compact_buffer.as_ref().map_or(0,|b|b.len()),builder.epsilon_buffer.len());
 }
 if builder.compact_buffer.as_ref().is_some_and(|b|b.failed){return None}
 let t=std::time::Instant::now();builder.flush_transition_buffer();builder.profile.flush_ms=t.elapsed().as_secs_f64()*1000.;
 let profile=builder.profile;drop(builder);let build_ms=started.elapsed().as_secs_f64()*1000.;
 Some(NativeBuild{sink,profile,build_ms})
}

#[cfg(test)] #[path="native_builder_tests.rs"] mod tests;

#[path="native_scalar_cursor.rs"] mod native_scalar_cursor;

#[path="native_compact_buffer.rs"] mod native_compact_buffer;
