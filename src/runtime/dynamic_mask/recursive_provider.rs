//! Scoped adapter for the existing ordinary epsilon/virtual lexer executor.
//!
//! Composition supplies namespaces and CALL/RETURN reset routing. Vocabulary
//! traversal, exact lexer execution, guard advancement and product caching are
//! the ordinary masking implementations, not another composition mask walk.
use super::*;

#[derive(Clone, Copy)]
struct ScopedConfig {
    leaf: usize,
    local: u32,
}

struct RecursiveConfigTransitions<'scan, 'constraint> {
    routing: RecursiveFullWalkTransitions<'constraint>,
    tables: Vec<FullWalkConfigTransitions<'scan, 'constraint>>,
    states: Vec<ScopedConfig>,
    ids: FxHashMap<(usize, u32), u32>,
    roots: FxHashMap<u32, u32>,
    resets: Vec<Option<u32>>,
    rows: Vec<Option<Box<[u64; 256]>>>,
    futures: Vec<Option<BitSet>>,
    boundary: FxHashMap<(u32, u32), bool>,
    error: Option<String>,
}

impl<'scan, 'constraint> RecursiveConfigTransitions<'scan, 'constraint> {
    fn intern(&mut self, leaf: usize, local: u32) -> Result<u32, String> {
        if let Some(&id) = self.ids.get(&(leaf, local)) { return Ok(id); }
        let id = u32::try_from(self.states.len())
            .ok().filter(|&id| id < u32::MAX - 4)
            .ok_or_else(|| "recursive config namespace exhausted".to_owned())?;
        self.states.push(ScopedConfig { leaf, local });
        self.ids.insert((leaf, local), id);
        self.rows.push(None);
        self.futures.push(None);
        Ok(id)
    }

    fn fail(&mut self, error: String) {
        if self.error.is_none() { self.error = Some(error); }
    }

    fn finish(self) -> Result<(), String> {
        if let Some(error) = self.error { return Err(error); }
        for table in self.tables { table.finish()?; }
        Ok(())
    }

    fn exact_scoped_future(&mut self, state: u32) -> &BitSet {
        if self.futures[state as usize].is_none() {
            let ScopedConfig { leaf, local } = self.states[state as usize];
            let descriptor = self.routing.leaves[leaf];
            let table = &mut self.tables[leaf];
            let config = table.generic_config_for_state(local);
            let mut candidates = BitSet::new(descriptor.terminal_count as usize);
            for index in 0..table.cache.config_len(config) {
                let raw = table.cache.config_state(config, index);
                candidates.union_with(table.cache.tokenizer().possible_future_terminals(raw));
            }
            let terminal_count = self.routing.leaves.last().map_or(0, |last| {
                last.terminal_offset as usize + last.terminal_count as usize
            });
            let mut exact = BitSet::new(terminal_count);
            for terminal in candidates.iter_ones() {
                // In particular, retained virtual residual support may be an
                // over-approximation. Ask the ordinary executor for exact
                // liveness before allowing it to certify a token endpoint.
                if table.future_contains(local, terminal as u32) {
                    exact.set(descriptor.terminal_offset as usize + terminal);
                }
            }
            self.futures[state as usize] = Some(exact);
        }
        self.futures[state as usize].as_ref().expect("future initialized above")
    }
}

impl FullWalkTransitionTable for RecursiveConfigTransitions<'_, '_> {
    type Cell = FullWalkConfigCell;

    #[inline]
    fn cell(&mut self, state: u32, byte: u8) -> Self::Cell {
        const UNKNOWN: u64 = u64::MAX;
        if let Some(row) = self.rows[state as usize].as_ref() {
            let packed = row[byte as usize];
            if packed != UNKNOWN {
                return FullWalkConfigCell {
                    target: packed as u32,
                    has_finalizer: (packed >> 32) != 0,
                };
            }
        }
        let ScopedConfig { leaf, local } = self.states[state as usize];
        let cell = self.tables[leaf].cell(local, byte);
        let target = if cell.target == u32::MAX {
            u32::MAX
        } else {
            match self.intern(leaf, cell.target) {
                Ok(target) => target,
                Err(error) => { self.fail(error); u32::MAX }
            }
        };
        let row = self.rows[state as usize].get_or_insert_with(|| Box::new([UNKNOWN; 256]));
        row[byte as usize] = u64::from(target) | (u64::from(cell.has_finalizer) << 32);
        FullWalkConfigCell { target, has_finalizer: cell.has_finalizer }
    }

    #[inline(always)]
    fn cell_is_dead(cell: Self::Cell) -> bool { cell.target == u32::MAX }
    #[inline(always)]
    fn cell_has_finalizer(cell: Self::Cell) -> bool { cell.has_finalizer }
    #[inline(always)]
    fn cell_target(cell: Self::Cell) -> u32 { cell.target }

    fn root_state(&mut self, scoped_raw: u32) -> Result<u32, String> {
        if let Some(&id) = self.roots.get(&scoped_raw) { return Ok(id); }
        let (leaf, raw) = self.routing.constraint.recursive_tokenizer_leaf_state(scoped_raw)
            .ok_or_else(|| format!("recursive lexer root {scoped_raw} is not scoped"))?;
        let local = self.tables[leaf].root_state(raw)?;
        let id = self.intern(leaf, local)?;
        self.roots.insert(scoped_raw, id);
        if self.routing.leaves[leaf].reset == scoped_raw { self.resets[leaf] = Some(id); }
        Ok(id)
    }

    fn walk_initial_state(&mut self, constraint: &Constraint, _: &DynamicMaskVocab) -> Result<u32, String> {
        debug_assert!(std::ptr::eq(constraint, self.routing.constraint));
        self.root_state(self.routing.leaves[0].reset)
    }

    fn finalizer_code(&self, state: u32) -> u32 {
        let ScopedConfig { leaf, local } = self.states[state as usize];
        let code = self.tables[leaf].finalizer_code(local);
        if code >= u32::MAX - 1 { code } else { self.routing.leaves[leaf].terminal_offset + code }
    }

    fn single_finalizer_continues(&mut self, state: u32) -> bool {
        let ScopedConfig { leaf, local } = self.states[state as usize];
        self.tables[leaf].single_finalizer_continues(local)
    }

    fn matched_terminals(&self, state: u32) -> SmallVec<[TerminalID; 4]> {
        let ScopedConfig { leaf, local } = self.states[state as usize];
        let base = self.routing.leaves[leaf].terminal_offset;
        self.tables[leaf].matched_terminals(local).into_iter().map(|t| base + t).collect()
    }

    fn future_contains(&mut self, state: u32, terminal: TerminalID) -> bool {
        let ScopedConfig { leaf, local } = self.states[state as usize];
        let descriptor = self.routing.leaves[leaf];
        let Some(local_terminal) = terminal.checked_sub(descriptor.terminal_offset) else { return false; };
        local_terminal < descriptor.terminal_count && self.tables[leaf].future_contains(local, local_terminal)
    }

    fn future_intersects(&mut self, state: u32, terminals: &BitSet) -> bool {
        !self.exact_scoped_future(state).is_disjoint(terminals)
    }

    fn merge_states(&mut self, states: &[u32]) -> Option<u32> {
        let first = *states.first()?;
        states.iter().all(|&state| state == first).then_some(first)
    }

    #[inline(always)]
    fn dense_state_count(&self) -> Option<usize> { None }
    #[inline(always)]
    fn exact_raw_state(&self, _: u32) -> Option<u32> { None }
    #[inline(always)]
    fn uses_scoped_reset_routing(&self) -> bool { true }
    #[inline(always)]
    fn parser_conditioned_dead_skip_default(&self) -> bool { true }
    #[inline(always)]
    fn prefer_adaptive_output(&self) -> bool { true }
    #[inline(always)]
    fn product_transition_cache_capacity(&self) -> usize { self.routing.product_transition_cache_capacity() }

    #[inline]
    fn terminal_is_ignore(&self, constraint: &Constraint, terminal: TerminalID) -> bool {
        self.routing.terminal_is_ignore(constraint, terminal)
    }

    fn scoped_reset_branches(
        &mut self, parser_cache: &mut FullWalkParserCache, constraint: &Constraint,
        parser_node: u32, terminal: TerminalID,
    ) -> SmallVec<[(u32, u32); 4]> {
        let raw_resets = self.routing.scoped_reset_branches(parser_cache, constraint, parser_node, terminal);
        let mut result = SmallVec::new();
        for (raw_reset, parser) in raw_resets {
            match self.root_state(raw_reset) {
                Ok(lexer) => result.push((lexer, parser)),
                Err(error) => self.fail(error),
            }
        }
        result
    }

    fn token_boundary_allowed(
        &mut self, parser_cache: &mut FullWalkParserCache, constraint: &Constraint,
        _: u32, lexer_state: u32, parser_node: u32,
    ) -> bool {
        let leaf = self.states[lexer_state as usize].leaf;
        if self.resets[leaf] == Some(lexer_state) { return true; }
        if let Some(&value) = self.boundary.get(&(lexer_state, parser_node)) { return value; }
        let ignored = self.routing.leaves[leaf].constraint.ignore_terminal
            .map(|terminal| self.routing.leaves[leaf].terminal_offset + terminal);
        let future = self.exact_scoped_future(lexer_state);
        let allowed = ignored.is_some_and(|terminal| future.contains(terminal as usize)) || {
            let gss = with_empty_accumulators(&parser_cache.nodes[parser_node as usize].gss);
            constraint.compact_segmented_parser_may_advance_on_any(&gss, future).unwrap_or(false)
        };
        self.boundary.insert((lexer_state, parser_node), allowed);
        allowed
    }
}

pub(super) fn fill(state: &ConstraintState<'_>, buf: &mut [u32]) -> Result<bool, String> {
    let routing = RecursiveFullWalkTransitions::new_for_parser_routing(state.constraint)
        .ok_or_else(|| "recursive composition has no valid scoped lexer/parser layout".to_owned())?;
    let mut scans = routing.leaves.iter()
        .map(|leaf| DynamicNfaScanCache::new(leaf.constraint, None)).collect::<Vec<_>>();
    let max_token_len = state.constraint.dynamic_mask_vocab_for_runtime().max_token_byte_len();
    let tables = scans.iter_mut().map(|scan| {
        FullWalkConfigTransitions::new(scan, max_token_len, state.generation)
    }).collect();
    let reset_count = routing.leaves.len();
    let mut provider = RecursiveConfigTransitions {
        routing, tables, states: Vec::new(), ids: FxHashMap::default(),
        roots: FxHashMap::default(), resets: vec![None; reset_count], rows: Vec::new(),
        futures: Vec::new(), boundary: FxHashMap::default(), error: None,
    };
    let result = fill_recursive_mask_using(state, buf, &mut provider);
    provider.finish()?;
    result
}
