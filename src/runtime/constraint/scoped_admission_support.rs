//! Query-local exact admission of scoped terminals. Control closure belongs
//! to the parser language, not a lexer state or leaf.
use super::*;
use crate::compiler::glr::parser::provider_closed_symbol_may_advance;

const MAX_DEPTH: u32 = 256;
const MAX_TOPS: usize = 16;
const MAX_MEMO_TERMINALS: usize = 4096;

enum ClosedLanguage {
    Native(LeveledGSS<u32, ()>),
    Lr { stacks: ParserGSS, tops: SmallVec<[u32; 8]> },
}

pub(crate) struct ScopedAdmissionMemo<'a> {
    root: &'a Constraint,
    layout: &'a RecursiveParserLayout,
    closed: ClosedLanguage,
    exact: FxHashMap<u32, bool>,
}

impl ScopedAdmissionMemo<'_> {
    fn matches_candidates(
        exact: &mut FxHashMap<u32, bool>,
        base: u32,
        candidates: impl IntoIterator<Item = u32>,
        mut admits: impl FnMut(u32, u32) -> bool,
        mut lexical: impl FnMut(u32) -> bool,
    ) -> bool {
        // These are exact leaf-local terminals, not outer aliases. Retain
        // their ascending order and invoke lexical only on parser witnesses.
        for local in candidates {
            let terminal = base + local;
            let admitted = if let Some(&result) = exact.get(&terminal) {
                result
            } else {
                let result = admits(local, terminal);
                if exact.len() < MAX_MEMO_TERMINALS { exact.insert(terminal, result); }
                result
            };
            if admitted && lexical(local) { return true; }
        }
        false
    }

    pub(crate) fn entry_count(&self) -> usize { self.exact.len() }

    pub(crate) fn matches_leaf(
        &mut self,
        leaf: usize,
        candidates: impl IntoIterator<Item = u32>,
        lexical: impl FnMut(u32) -> bool,
    ) -> bool {
        let Some(&offset) = self.layout.leaf_terminal_offsets.get(leaf) else { return false; };
        let base = self.layout.outer_terminal_count + offset;
        match &self.closed {
            ClosedLanguage::Native(stacks) => {
                let provider = self.root.template_composition_provider().expect("native composition");
                Self::matches_candidates(&mut self.exact, base, candidates,
                    |_, terminal| provider.closed_admits_native(stacks, terminal), lexical)
            }
            ClosedLanguage::Lr { stacks, tops } => {
                if stacks.is_empty() { return false; }
                let tables = RecursiveSegmentedParserTables { root: self.root, layout: self.layout };
                let provider = DisjointComponentActionProvider::with_state_offsets(
                    &tables, &self.layout.links, &self.layout.leaf_state_offsets,
                ).expect("validated scoped parser layout");
                Self::matches_candidates(&mut self.exact, base, candidates,
                    |local, _| provider_closed_symbol_may_advance(&provider, stacks, tops,
                        ScopedParserSymbol::Terminal { component: leaf as u32, terminal: local }),
                    lexical)
            }
        }
    }
}

impl Constraint {
    pub(crate) fn prepare_scoped_admission_memo(
        &self, input: &LeveledGSS<u32, ()>,
    ) -> Option<ScopedAdmissionMemo<'_>> {
        if input.max_depth() > MAX_DEPTH { return None; }
        let layout = self.recursive_parser_layout_ref()?;
        let closed = if let Some(provider) = self.template_composition_provider() {
            let stacks = provider.close_native(input);
            if stacks.max_depth() > MAX_DEPTH || stacks.top_value_count() > MAX_TOPS {
                return None;
            }
            ClosedLanguage::Native(stacks)
        } else {
            let tables = RecursiveSegmentedParserTables { root: self, layout };
            let provider = DisjointComponentActionProvider::with_state_offsets(
                &tables, &layout.links, &layout.leaf_state_offsets,
            ).ok()?;
            let annotated = input.apply(|_| TerminalsDisallowed::new());
            let stacks = close_provider_control_stacks(&provider, &annotated);
            if stacks.max_depth() > MAX_DEPTH { return None; }
            let tops = stacks.peek_values();
            if tops.len() > MAX_TOPS { return None; }
            ClosedLanguage::Lr { stacks, tops }
        };
        Some(ScopedAdmissionMemo { root: self, layout, closed, exact: FxHashMap::default() })
    }
}