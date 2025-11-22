#![allow(clippy::too_many_arguments)]

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::{self, Debug, Display, Formatter},
    sync::Arc,
};
use std::collections::BTreeMap as StdMap;

use bimap::BiBTreeMap;
use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;

use crate::{
    datastructures::{
        leveled_gss::{LeveledGSS, Merge},
        vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode},
    },
    equivalence_analysis_finite_automata,
    finite_automata::Regex,
    glr::{
        analyze::compute_terminal_follow_sets,
        grammar::Terminal,
        parser::{GLRParser, GLRParserState},
    },
    interface::{CompiledGrammar, GrammarDefinition},
    json_serialization::{JSONConvertible, JSONNode},
    precompute4::full_dwa::{precompute4, Precomputed4},
    r#macro::is_debug_level_enabled,
    tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID},
    types::{TerminalID as GrammarTokenID, TerminalID},
};
use crate::constraint_precompute1_utils::Trie1Config;
use crate::datastructures::bitset::Bitset;
use crate::datastructures::gss_acc::Acc;
use crate::glr::parser::ParseStateEdgeContent;
use crate::precompute4::weighted_automata::{DWA, NWA};

// Import from new modules
pub use crate::constraint_vocab::*;
pub use crate::constraint_trie::*;
use crate::constraint_precompute::run_precompute1;

type GSSNode = LeveledGSS<ParseStateEdgeContent, Acc>;

// ---------------------------------------------------------------------------
// Terminal allowance mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerminalAllowanceCheckMode {
    None,
    ImmediateSets,
    ImmediateProbe,
    #[default]
    StepProbe,
}

impl JSONConvertible for TerminalAllowanceCheckMode {
    fn to_json(&self) -> JSONNode {
        let s = match self {
            TerminalAllowanceCheckMode::None => "none",
            TerminalAllowanceCheckMode::ImmediateSets => "immediate_sets",
            TerminalAllowanceCheckMode::ImmediateProbe => "immediate_probe",
            TerminalAllowanceCheckMode::StepProbe => "step_probe",
        };
        JSONNode::String(s.to_string())
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::String(s) => match s.as_str() {
                "none" => Ok(TerminalAllowanceCheckMode::None),
                "immediate_sets" => Ok(TerminalAllowanceCheckMode::ImmediateSets),
                "immediate_probe" => Ok(TerminalAllowanceCheckMode::ImmediateProbe),
                "step_probe" => Ok(TerminalAllowanceCheckMode::StepProbe),
                other => Err(format!("Unknown TerminalAllowanceCheckMode '{}'", other)),
            },
            other => Err(format!(
                "Expected JSON string for TerminalAllowanceCheckMode, got {:?}",
                other
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GrammarConstraintConfig {
    pub trie1: Trie1Config,
    pub run_precompute4: bool,
    pub use_dummy_terminals: bool,
    pub dummy_terminal_map: BTreeMap<String, BTreeSet<Terminal>>,
    pub dummy_terminal_penalties: BTreeMap<String, usize>,
}

impl Default for GrammarConstraintConfig {
    fn default() -> Self {
        Self {
            trie1: Trie1Config::off(),
            run_precompute4: true,
            use_dummy_terminals: false,
            dummy_terminal_map: BTreeMap::new(),
            dummy_terminal_penalties: BTreeMap::new(),
        }
    }
}

impl GrammarConstraintConfig {
    pub fn off() -> Self { Self::default() }
}

// ---------------------------------------------------------------------------
// Main structure
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GrammarConstraint {
    pub tokenizer: Regex,
    pub parser: GLRParser,

    // Precomputations
    pub precomputed1: Precomputed,
    pub precomputed4: Precomputed4,

    pub llm_vocab: Arc<LLMVocab>,
    pub(crate) token_name_map: BiBTreeMap<Terminal, usize>,

    /// Tokenizer state -> grammar terminal -> internal LLM token bitset.
    pub possible_matches: BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,

    /// Internal-token -> start_tokenizer_state -> end_tokenizer_state.
    pub state_map_by_llm:
        DedupValueMap<LLMTokenID, BTreeMap<TokenizerStateID, TokenizerStateID>>,
    /// Internal-token -> start_tokenizer_state -> terminals.
    pub terminal_map_by_llm:
        DedupValueMap<LLMTokenID, BTreeMap<TokenizerStateID, TerminalBV>>,

    pub(crate) trie1_god: Trie1GodWrapper,

    pub run_precompute4: bool,
    pub post_commit_allow_check_mode: TerminalAllowanceCheckMode,

    pub vocab: StageVocab,

    /// Maps original terminal IDs to dummy terminal IDs (if any).
    pub(crate) original_to_dummy_map: BTreeMap<TerminalID, TerminalID>,
}

impl GrammarConstraint {
    pub fn assert_eq(&self, other: &Self) {
        assert_eq!(self.tokenizer, other.tokenizer);
        assert_eq!(self.parser, other.parser);

        assert_eq!(self.precomputed1.len(), other.precomputed1.len());
        for ((sid1, arc1), (sid2, arc2)) in
            self.precomputed1.iter().zip(other.precomputed1.iter())
        {
            assert_eq!(sid1, sid2);
            assert!(PrecomputeNode1::are_graphs_equal(
                &self.trie1_god,
                *arc1,
                &other.trie1_god,
                *arc2
            ));
        }

        assert_eq!(
            self.llm_vocab.llm_token_map,
            other.llm_vocab.llm_token_map
        );
        assert_eq!(self.token_name_map, other.token_name_map);
        assert_eq!(self.possible_matches, other.possible_matches);
        assert_eq!(
            self.post_commit_allow_check_mode,
            other.post_commit_allow_check_mode
        );
        assert_eq!(self.state_map_by_llm, other.state_map_by_llm);
        assert_eq!(self.terminal_map_by_llm, other.terminal_map_by_llm);
        assert_eq!(self.vocab, other.vocab);
        assert_eq!(self.original_to_dummy_map, other.original_to_dummy_map);
    }
}

impl JSONConvertible for GrammarConstraint {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("tokenizer".to_string(), self.tokenizer.to_json());
        obj.insert("parser".to_string(), self.parser.to_json());
        obj.insert("precomputed1".to_string(), self.precomputed1.to_json());
        obj.insert("precomputed4".to_string(), self.precomputed4.to_json());
        obj.insert("token_name_map".to_string(), self.token_name_map.to_json());
        obj.insert(
            "possible_matches".to_string(),
            self.possible_matches.to_json(),
        );
        obj.insert("trie1_god".to_string(), self.trie1_god.to_json());
        obj.insert(
            "run_precompute4".to_string(),
            self.run_precompute4.to_json(),
        );
        obj.insert(
            "post_commit_allow_check_mode".to_string(),
            self.post_commit_allow_check_mode.to_json(),
        );
        obj.insert(
            "state_map_by_llm".to_string(),
            self.state_map_by_llm.to_json(),
        );
        obj.insert(
            "terminal_map_by_llm".to_string(),
            self.terminal_map_by_llm.to_json(),
        );
        obj.insert("vocab".to_string(), self.vocab.to_json());
        obj.insert("llm_vocab".to_string(), self.llm_vocab.to_json());
        obj.insert(
            "original_to_dummy_map".to_string(),
            self.original_to_dummy_map.to_json(),
        );
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let tokenizer = obj
                    .remove("tokenizer")
                    .ok_or_else(|| "Missing field tokenizer".to_string())
                    .and_then(Regex::from_json)?;
                let parser = obj
                    .remove("parser")
                    .ok_or_else(|| "Missing field parser".to_string())
                    .and_then(GLRParser::from_json)?;
                let precomputed1 = obj
                    .remove("precomputed1")
                    .ok_or_else(|| "Missing field precomputed1".to_string())
                    .and_then(Precomputed::from_json)?;
                let precomputed4 = obj
                    .remove("precomputed4")
                    .ok_or_else(|| "Missing field precomputed4".to_string())
                    .and_then(Precomputed4::from_json)?;

                let token_name_map = obj
                    .remove("token_name_map")
                    .ok_or_else(|| "Missing field token_name_map".to_string())
                    .and_then(|n| BiBTreeMap::<Terminal, usize>::from_json(n))?;

                // possible_matches: prefer new key, fall back to old *_precompute1 for compatibility
                let possible_matches = if let Some(n) = obj.remove("possible_matches") {
                    BTreeMap::<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>::from_json(n)?
                } else if let Some(n) = obj.remove("possible_matches_precompute1") {
                    BTreeMap::<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>::from_json(n)?
                } else {
                    BTreeMap::new()
                };

                let trie1_god = obj
                    .remove("trie1_god")
                    .ok_or_else(|| "Missing field trie1_god".to_string())
                    .and_then(Trie1GodWrapper::from_json)?;

                let run_precompute4 = obj
                    .remove("run_precompute4")
                    .map(bool::from_json)
                    .transpose()?
                    .unwrap_or(true);

                let post_commit_allow_check_mode =
                    match obj.remove("post_commit_allow_check_mode") {
                        Some(n) => TerminalAllowanceCheckMode::from_json(n)?,
                        None => TerminalAllowanceCheckMode::default(),
                    };

                let state_map_by_llm =
                    match obj.remove("state_map_by_llm") {
                        Some(n) => DedupValueMap::<
                            LLMTokenID,
                            BTreeMap<TokenizerStateID, TokenizerStateID>,
                        >::from_json(n)?,
                        None => DedupValueMap::new(),
                    };
                let terminal_map_by_llm =
                    match obj.remove("terminal_map_by_llm") {
                        Some(n) => DedupValueMap::<
                            LLMTokenID,
                            BTreeMap<TokenizerStateID, TerminalBV>,
                        >::from_json(n)?,
                        None => DedupValueMap::new(),
                    };

                // Handle llm_vocab deserialization with fallback
                let llm_vocab = if let Some(n) = obj.remove("llm_vocab") {
                    Arc::new(LLMVocab::from_json(n)?)
                } else {
                    // Fallback to old format
                    let max_original_llm_token_id = obj
                        .remove("max_original_llm_token_id")
                        .ok_or_else(|| "Missing field max_original_llm_token_id".to_string())
                        .and_then(usize::from_json)?;

                    let llm_token_map = obj
                        .remove("llm_token_map")
                        .ok_or_else(|| "Missing field llm_token_map".to_string())
                        .and_then(|n| BiBTreeMap::<Vec<u8>, LLMTokenID>::from_json(n))?;

                    Arc::new(LLMVocab {
                        llm_token_map,
                        max_original_llm_token_id,
                    })
                };

                // Stage vocab: new key "vocab", fall back to old names if present.
                let mut vocab_node = if let Some(n) = obj.remove("vocab") {
                    n
                } else if let Some(n) = obj.remove("precompute_vocab") {
                    n
                } else {
                    return Err(
                        "Missing stage vocab (vocab/precompute_vocab/precompute0_vocab)"
                            .to_string(),
                    );
                };

                // For backward compatibility, inject max_original_llm_token_id into vocab JSON if needed.
                if let JSONNode::Object(ref mut vocab_obj) = vocab_node {
                    if !vocab_obj.contains_key("max_original_llm_token_id") {
                        vocab_obj.insert(
                            "max_original_llm_token_id".to_string(),
                            llm_vocab.max_original_llm_token_id.to_json(),
                        );
                    }
                }
                let vocab = StageVocab::from_json(vocab_node)?;

                let original_to_dummy_map = match obj.remove("original_to_dummy_map") {
                    Some(n) => BTreeMap::<TerminalID, TerminalID>::from_json(n)?,
                    None => BTreeMap::new(),
                };

                let gc = GrammarConstraint {
                    tokenizer,
                    parser,
                    precomputed1,
                    precomputed4,
                    llm_vocab,
                    token_name_map,
                    possible_matches,
                    state_map_by_llm,
                    terminal_map_by_llm,
                    trie1_god,
                    run_precompute4,
                    post_commit_allow_check_mode,
                    vocab,
                    original_to_dummy_map,
                };
                Ok(gc)
            }
            _ => Err("Expected JSONNode::Object for GrammarConstraint".to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl GrammarConstraint {
    pub fn from_compiled_grammar(
        compiled_grammar: CompiledGrammar,
        llm_token_map: LLMTokenMap,
        max_original_llm_token_id: usize,
    ) -> Self {
        Self::from_compiled_grammar_with_config(
            compiled_grammar,
            llm_token_map,
            max_original_llm_token_id,
            &GrammarConstraintConfig::default(),
        )
    }

    pub fn from_compiled_grammar_with_config(
        compiled_grammar: CompiledGrammar,
        llm_token_map: LLMTokenMap,
        max_original_llm_token_id: usize,
        config: &GrammarConstraintConfig,
    ) -> Self {
        let token_name_map = compiled_grammar.definition.terminal_to_group_id().clone();
        Self::build_with_config(
            compiled_grammar.tokenizer,
            compiled_grammar.glr_parser,
            llm_token_map,
            token_name_map,
            max_original_llm_token_id,
            config,
        )
    }

    pub fn new(
        tokenizer: Regex,
        parser: GLRParser,
        llm_token_map: LLMTokenMap,
        token_name_map: BiBTreeMap<Terminal, usize>,
        max_original_llm_token_id: usize,
    ) -> Self {
        Self::build_with_config(
            tokenizer,
            parser,
            llm_token_map,
            token_name_map,
            max_original_llm_token_id,
            &GrammarConstraintConfig::default(),
        )
    }

    pub fn new_with_config(
        tokenizer: Regex,
        parser: GLRParser,
        llm_token_map: LLMTokenMap,
        token_name_map: BiBTreeMap<Terminal, usize>,
        max_original_llm_token_id: usize,
        config: &GrammarConstraintConfig,
    ) -> Self {
        Self::build_with_config(
            tokenizer,
            parser,
            llm_token_map,
            token_name_map,
            max_original_llm_token_id,
            config,
        )
    }

    /// Convenience entry point from a `GrammarDefinition`.
    pub fn new_from_grammar_definition(
        grammar_definition: Arc<GrammarDefinition>,
        llm_token_map: LLMTokenMap,
        max_original_llm_token_id: usize,
        config: &GrammarConstraintConfig,
    ) -> Self {
        let initial_compiled_grammar =
            CompiledGrammar::from_definition(grammar_definition.clone());

        if !config.use_dummy_terminals {
            return Self::from_compiled_grammar_with_config(
                initial_compiled_grammar,
                llm_token_map,
                max_original_llm_token_id,
                config,
            );
        }

        let (final_productions, new_dummy_terminals) =
            crate::glr::analyze::rewrite_productions_with_dummies(
                &grammar_definition.productions,
                &config.dummy_terminal_map,
            );

        let final_compiled_grammar = if !new_dummy_terminals.is_empty() {
            let mut final_grammar_def = (*grammar_definition).clone();
            final_grammar_def.productions = final_productions;
            for dummy_terminal in new_dummy_terminals {
                if let Terminal::RegexName(name) = dummy_terminal {
                    final_grammar_def.add_external_terminal(&name);
                }
            }
            CompiledGrammar::from_definition(Arc::new(final_grammar_def))
        } else {
            initial_compiled_grammar
        };

        Self::from_compiled_grammar_with_config(
            final_compiled_grammar,
            llm_token_map,
            max_original_llm_token_id,
            config,
        )
    }

    pub(crate) fn setup_llm_token_mappings(
        original_llm_token_map: &LLMTokenMap,
        tokenizer: &Regex,
    ) -> BTreeMap<usize, usize> {
        if original_llm_token_map.len() < 10 {
            return original_llm_token_map
                .iter()
                .map(|(_bytes, id)| (id.0, id.0))
                .collect();
        }

        let mut sorted_tokens: Vec<_> = original_llm_token_map.iter().collect();
        sorted_tokens.sort_by_key(|(bytes, _id)| *bytes);

        let mut llm_token_strings: Vec<Vec<u8>> = Vec::with_capacity(sorted_tokens.len());
        let mut original_ids: Vec<LLMTokenID> = Vec::with_capacity(sorted_tokens.len());

        for (bytes, id) in sorted_tokens {
            llm_token_strings.push(bytes.clone());
            original_ids.push(*id);
        }

        let initial_states: Vec<usize> = tokenizer.iter_states().map(|s| s.0).collect();

        let equivalence_classes =
            equivalence_analysis_finite_automata::find_equivalence_classes(
                tokenizer,
                &llm_token_strings,
                &initial_states,
            );

        if is_debug_level_enabled(3) {
            let num_original_tokens = llm_token_strings.len();
            let num_classes = equivalence_classes.len();
            crate::debug!(3, "Equivalence Analysis: {} original tokens -> {} classes", num_original_tokens, num_classes);
        }

        let mut original_to_internal_map = BTreeMap::new();
        let mut internal_id_counter = 0;
        for (_signature, string_indices) in equivalence_classes {
            let internal_id = internal_id_counter;
            internal_id_counter += 1;
            for string_index in string_indices {
                let original_llm_id = original_ids[string_index];
                original_to_internal_map.insert(original_llm_id.0, internal_id);
            }
        }

        original_to_internal_map
    }

    fn build_with_config(
        tokenizer: Regex,
        parser: GLRParser,
        llm_token_map: LLMTokenMap,
        token_name_map: BiBTreeMap<Terminal, usize>,
        max_original_llm_token_id: usize,
        config: &GrammarConstraintConfig,
    ) -> Self {
        // Epsilon tokens are not supported.
        let epsilon_terminal_group_ids: BTreeSet<_> = tokenizer
            .execute_from_state(&[], tokenizer.initial_state_id())
            .matches
            .iter()
            .map(|token| token.id)
            .collect();
        assert!(
            epsilon_terminal_group_ids.is_empty(),
            "Epsilon tokens are not supported."
        );

        // Global original<->internal mapping.
        let original_to_internal_map =
            Self::setup_llm_token_mappings(&llm_token_map, &tokenizer);
        let internal_max_llm_token = original_to_internal_map
            .values()
            .copied()
            .max()
            .unwrap_or(0);

        let mut internal_to_original_map: BTreeMap<usize, LLMTokenBV> = BTreeMap::new();
        for (orig, int_id) in &original_to_internal_map {
            internal_to_original_map
                .entry(*int_id)
                .or_default()
                .insert(*orig);
        }

        // Build internal LLM token map keyed by bytes.
        let mut internal_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
        for (bytes, original_id) in llm_token_map.iter() {
            if let Some(internal_id_val) = original_to_internal_map.get(&original_id.0) {
                internal_llm_token_map.insert(bytes.clone(), LLMTokenID(*internal_id_val));
            }
        }

        // Vocab tree for internal tokens.
        crate::debug!(3, "Building internal vocab prefix tree");
        let internal_tokens_for_vocab: Vec<(usize, Vec<u8>)> =
            internal_llm_token_map.iter().map(|(b, id)| (id.0, b.clone())).collect();
        let vocab_tree = VocabPrefixTree::build(&internal_tokens_for_vocab);
        crate::debug!(4, "Done building internal vocab prefix tree");

        // Unified fast pass for maps and matches.
        crate::debug!(3, "Computing maps and possible_matches (fast parallel pass)");
        let (state_map_by_llm_raw, computed_possible_matches) =
            Self::build_maps_and_matches(&tokenizer, &vocab_tree.root);
        let terminal_map_by_llm_raw =
            Self::rearrange_possible_matches(&computed_possible_matches);

        // Compute terminal follow sets, then map to IDs.
        crate::debug!(3, "Computing terminal follow sets");
        let terminal_follow_sets_named = compute_terminal_follow_sets(&parser.productions);
        let mut terminal_follow_map: BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>> =
            BTreeMap::new();
        for (terminal1, following_terminals) in terminal_follow_sets_named {
            let t1_id = parser
                .terminal_map
                .get_by_left(&terminal1)
                .unwrap()
                .clone();
            let mut following_ids = BTreeSet::new();
            for t2 in following_terminals {
                let t2_id = *parser.terminal_map.get_by_left(&t2).unwrap();
                following_ids.insert(t2_id);
            }
            if !following_ids.is_empty() {
                terminal_follow_map.insert(t1_id, following_ids);
            }
        }

        let llm_vocab = Arc::new(LLMVocab {
            llm_token_map: llm_token_map.clone(),
            max_original_llm_token_id,
        });

        let mut vocab = StageVocab {
            original_to_internal: original_to_internal_map.clone(),
            internal_to_original: internal_to_original_map.clone(),
            internal_max_llm_token,
            max_original_llm_token_id: 0,
            internal_to_original_sparse_matrix: vec![],
        };

        let mut original_to_dummy_map: BTreeMap<TerminalID, TerminalID> = BTreeMap::new();
        for (dummy_name, original_terminals) in &config.dummy_terminal_map {
            let dummy_term = Terminal::regex_name(dummy_name);
            if let Some(&dummy_id) = parser.terminal_map.get_by_left(&dummy_term) {
                for original_terminal in original_terminals {
                    if let Some(&original_id) =
                        parser.terminal_map.get_by_left(original_terminal)
                    {
                        original_to_dummy_map.insert(original_id, dummy_id);
                    }
                }
            }
        }

        // Precompute1 - generate skeleton DWA
        let skeleton_dwa = run_precompute1(
            &tokenizer,
            Some(&parser),
            Some(llm_vocab.clone()),
            &internal_llm_token_map,
            vocab.internal_max_llm_token,
        );

        let state_map_by_llm = state_map_by_llm_raw;
        let terminal_map_by_llm = terminal_map_by_llm_raw;
        let possible_matches_precompute1 = computed_possible_matches;

        // Precompute4 (DWA).
        let max_internal_llm_token_id = vocab.internal_max_llm_token;

        // Convert the precompute1 Trie to NWA and run precompute4.
        crate::debug!(3, "Running Precompute4");
        let nwa = NWA::from_dwa(&skeleton_dwa);
        let precomputed4 = precompute4(&parser, &nwa, max_internal_llm_token_id);

        let internal_to_original_sparse_matrix =
            StageVocab::build_internal_to_original_sparse_matrix(
                &vocab.internal_to_original,
                max_original_llm_token_id,
                vocab.internal_max_llm_token,
            );
        vocab.max_original_llm_token_id = max_original_llm_token_id;
        vocab.internal_to_original_sparse_matrix = internal_to_original_sparse_matrix;

        GrammarConstraint {
            tokenizer,
            parser,
            // Trie data structures are unused/empty.
            precomputed1: Precomputed::new(),
            precomputed4,
            llm_vocab,
            token_name_map,
            possible_matches: possible_matches_precompute1,
            state_map_by_llm,
            terminal_map_by_llm,
            trie1_god: Trie1GodWrapper::new(),
            run_precompute4: config.run_precompute4,
            post_commit_allow_check_mode: TerminalAllowanceCheckMode::default(),
            vocab,
            original_to_dummy_map,
        }
    }

    // -----------------------------------------------------------------------
    // Special precomputation
    // -----------------------------------------------------------------------

    pub fn dump_precomputed4(&self) {
        println!("\n--- Precomputed4 DWA ---");
        println!("{}", self.precomputed4);
    }

    // -----------------------------------------------------------------------
    // Vocab helpers
    // -----------------------------------------------------------------------

    pub fn all_internal_llm_tokens_bitset(&self) -> LLMTokenBV {
        LLMTokenBV::ones(self.vocab.internal_max_llm_token + 1)
    }

    pub fn internal_bv_to_original(&self, internal_bv: &LLMTokenBV) -> Bitset {
        self.vocab.internal_bv_to_original(internal_bv)
    }

    pub fn original_bv_to_internal(&self, original_bv: &LLMTokenBV) -> LLMTokenBV {
        self.vocab.original_bv_to_internal(original_bv)
    }

    pub fn internal_to_original(&self, internal_id: LLMTokenID) -> Option<LLMTokenID> {
        self.vocab
            .internal_to_original
            .get(&internal_id.0)
            .and_then(|bv| bv.iter_up_to(self.vocab.internal_max_llm_token).next())
            .map(|v| LLMTokenID(v))
    }

    #[inline]
    pub fn original_id_to_internal(&self, original_id: LLMTokenID) -> Option<LLMTokenID> {
        self.vocab
            .original_to_internal
            .get(&original_id.0)
            .map(|v| LLMTokenID(*v))
    }

    // -----------------------------------------------------------------------
    // Possible-matches-related helpers
    // -----------------------------------------------------------------------

    pub fn build_maps_and_matches(
        tokenizer: &Regex,
        vocab_root: &VocabPrefixTreeNode,
    ) -> (
        DedupValueMap<LLMTokenID, BTreeMap<TokenizerStateID, TokenizerStateID>>,
        BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    ) {
        // Flatten DFA for fast access (u32::MAX represents None)
        struct FastDFA {
            transitions: Vec<u32>,           // size = num_states * 256
            finalizers: Vec<Vec<TerminalID>>, // size = num_states
        }

        // Recursive DFS helper
        fn process_vocab_node_dfs(
            node: &VocabPrefixTreeNode,
            current_states: Vec<(u32, Vec<u32>)>,
            dfa: &FastDFA,
            out_state_map: &mut Vec<(LLMTokenID, BTreeMap<TokenizerStateID, TokenizerStateID>)>,
            out_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
        ) {
            for (edge_bytes, child) in node.iter_children() {
                let mut next_grouped_map: HashMap<u32, Vec<u32>> = HashMap::new();
                let reachable_bv: LLMTokenBV = child.reachable_token_ids().into();

                for (start, sources) in &current_states {
                    let mut curr = *start;
                    let mut valid = true;

                    for &b in edge_bytes.iter() {
                        let offset = (curr as usize) * 256 + (b as usize);
                        let next = dfa.transitions[offset];
                        if next == u32::MAX {
                            valid = false;
                            break;
                        }
                        curr = next;

                        // If matched, record for all sources
                        if !dfa.finalizers[curr as usize].is_empty() {
                            for &tid in &dfa.finalizers[curr as usize] {
                                for &src in sources {
                                    out_matches
                                        .entry(TokenizerStateID(src as usize))
                                        .or_default()
                                        .entry(tid)
                                        .or_default()
                                        .union_with(&reachable_bv);
                                }
                            }
                        }
                    }

                    if valid {
                        next_grouped_map.entry(curr).or_default().extend_from_slice(sources);
                    }
                }

                let next_grouped: Vec<(u32, Vec<u32>)> =
                    next_grouped_map.into_iter().collect();

                if !next_grouped.is_empty() {
                    // Record state map
                    let mut map = BTreeMap::new();
                    for (target, sources) in &next_grouped {
                        let tgt = TokenizerStateID(*target as usize);
                        for &src in sources {
                            map.insert(TokenizerStateID(src as usize), tgt);
                        }
                    }
                    out_state_map.push((LLMTokenID(child.token_id()), map));

                    // Recurse
                    process_vocab_node_dfs(
                        child,
                        next_grouped,
                        dfa,
                        out_state_map,
                        out_matches,
                    );
                }
            }
        }

        let dfa = &tokenizer.dfa;
        let num_states = dfa.states.len();
        let mut transitions = vec![u32::MAX; num_states * 256];
        let mut finalizers = Vec::with_capacity(num_states);

        for (i, state) in dfa.states.iter().enumerate() {
            for (byte, &next) in &state.transitions {
                transitions[i * 256 + (byte as usize)] = next as u32;
            }
            finalizers.push(
                state
                    .finalizers
                    .iter()
                    .map(|&gid| TerminalID(gid))
                    .collect(),
            );
        }

        let fast_dfa = FastDFA {
            transitions,
            finalizers,
        };

        // Group initial states by current state (identity mapping at root)
        let mut initial_states: Vec<(u32, Vec<u32>)> = Vec::with_capacity(num_states);
        for i in 0..num_states {
            initial_states.push((i as u32, vec![i as u32]));
        }

        // Parallel processing of root children
        let root_children: Vec<_> = vocab_root.iter_children().collect();
        let results: Vec<_> = root_children
            .par_iter()
            .map(|(edge_bytes, child_node)| {
                let mut local_state_map_entries = Vec::new();
                let mut local_possible_matches: BTreeMap<
                    TokenizerStateID,
                    BTreeMap<TerminalID, LLMTokenBV>,
                > = BTreeMap::new();

                // Manually process the first edge to bootstrap recursion
                let mut next_grouped_map: HashMap<u32, Vec<u32>> = HashMap::new();
                let reachable_bv: LLMTokenBV =
                    child_node.reachable_token_ids().into();

                for (start, sources) in &initial_states {
                    let mut curr = *start;
                    let mut valid = true;
                    for &b in edge_bytes.iter() {
                        let offset = (curr as usize) * 256 + (b as usize);
                        let next = fast_dfa.transitions[offset];
                        if next == u32::MAX {
                            valid = false;
                            break;
                        }
                        curr = next;
                        if !fast_dfa.finalizers[curr as usize].is_empty() {
                            for &tid in &fast_dfa.finalizers[curr as usize] {
                                for &src in sources {
                                    local_possible_matches
                                        .entry(TokenizerStateID(src as usize))
                                        .or_default()
                                        .entry(tid)
                                        .or_default()
                                        .union_with(&reachable_bv);
                                }
                            }
                        }
                    }
                    if valid {
                        next_grouped_map.entry(curr).or_default().extend_from_slice(sources);
                    }
                }

                let next_grouped: Vec<(u32, Vec<u32>)> =
                    next_grouped_map.into_iter().collect();
                if !next_grouped.is_empty() {
                    let mut map = BTreeMap::new();
                    for (target, sources) in &next_grouped {
                        let tgt = TokenizerStateID(*target as usize);
                        for &src in sources {
                            map.insert(TokenizerStateID(src as usize), tgt);
                        }
                    }
                    local_state_map_entries
                        .push((LLMTokenID(child_node.token_id()), map));

                    process_vocab_node_dfs(
                        child_node,
                        next_grouped,
                        &fast_dfa,
                        &mut local_state_map_entries,
                        &mut local_possible_matches,
                    );
                }

                (local_state_map_entries, local_possible_matches)
            })
            .collect();

        // Merge
        let mut state_map_out = DedupValueMap::new();
        // Root identity
        {
            let mut root_map = BTreeMap::new();
            for i in 0..num_states {
                let s = TokenizerStateID(i);
                root_map.insert(s, s);
            }
            state_map_out.insert(LLMTokenID(vocab_root.token_id()), root_map);
        }

        let mut possible_matches: BTreeMap<
            TokenizerStateID,
            BTreeMap<TerminalID, LLMTokenBV>,
        > = BTreeMap::new();
        for (entries, matches) in results {
            for (tid, map) in entries {
                state_map_out.insert(tid, map);
            }
            for (sid, term_map) in matches {
                let target = possible_matches.entry(sid).or_default();
                for (tid, bv) in term_map {
                    target.entry(tid).or_default().union_with(&bv);
                }
            }
        }

        (state_map_out, possible_matches)
    }

    pub fn rearrange_possible_matches(
        pm: &BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    ) -> DedupValueMap<LLMTokenID, BTreeMap<TokenizerStateID, TerminalBV>> {
        let mut triples: Vec<(u32, u32, u32)> = pm
            .par_iter()
            .flat_map(|(sid, tmap)| {
                let mut local_triples = Vec::new();
                for (term, bv) in tmap {
                    if !bv.is_all() {
                        for tok in bv.iter_up_to(usize::MAX) {
                            local_triples
                                .push((tok as u32, sid.0 as u32, term.0 as u32));
                        }
                    }
                }
                local_triples
            })
            .collect();

        triples.par_sort_unstable_by_key(|t| t.0);

        let mut out = DedupValueMap::new();
        if triples.is_empty() {
            return out;
        }

        let mut current_token = triples[0].0;
        let mut current_map: BTreeMap<TokenizerStateID, TerminalBV> = BTreeMap::new();

        for (tok, sid, term) in triples {
            if tok != current_token {
                out.insert(LLMTokenID(current_token as usize), current_map);
                current_map = BTreeMap::new();
                current_token = tok;
            }
            current_map
                .entry(TokenizerStateID(sid as usize))
                .or_default()
                .insert(term as usize);
        }
        out.insert(LLMTokenID(current_token as usize), current_map);

        out
    }

    // -----------------------------------------------------------------------
    // Top-level state construction
    // -----------------------------------------------------------------------

    pub fn init(&self) -> GrammarConstraintState<'_> {
        let mut state = BTreeMap::new();
        state.insert(
            self.tokenizer.initial_state_id(),
            self.parser.init_glr_parser(Some(self.llm_vocab.clone())),
        );
        GrammarConstraintState { parent: self, state }
    }

    pub fn state_with_nodes(
        &self,
        _nodes: Vec<(usize, Arc<GSSNode>)>,
    ) -> GrammarConstraintState<'_> {
        todo!()
    }

    pub fn state_from_gss_map(
        &self,
        gss_map: &BTreeMap<TokenizerStateID, GSSNode>,
    ) -> GrammarConstraintState {
        let mut state = BTreeMap::new();
        for (i, node) in gss_map.iter() {
            state.insert(
                *i,
                self.parser.init_parse_state_with_gss(node.clone()),
            );
        }
        GrammarConstraintState { parent: self, state }
    }

    pub fn print_gss_nodes(
        &self,
        _roots: &Vec<Arc<GSSNode>>,
        _labels: Option<&[String]>,
    ) {
        // Unimplemented
    }
}

// ---------------------------------------------------------------------------
// Merge implementation for leveled GSS
// ---------------------------------------------------------------------------

impl Merge for RangeSetBlaze<usize> {
    fn merge(&self, other: &Self) -> Self { self | other }
}

impl Merge for Arc<RangeSetBlaze<usize>> {
    fn merge(&self, other: &Self) -> Self {
        if Arc::ptr_eq(self, other) {
            return self.clone();
        }
        let mut merged = self.as_ref().clone();
        merged |= other.as_ref();
        if merged == **self {
            self.clone()
        } else if merged == **other {
            other.clone()
        } else {
            Arc::new(merged)
        }
    }
}

// ---------------------------------------------------------------------------
// GrammarConstraintState
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GrammarConstraintState<'a> {
    pub parent: &'a GrammarConstraint,
    pub state: BTreeMap<TokenizerStateID, GLRParserState<'a>>,
}

impl<'a> PartialEq for GrammarConstraintState<'a> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.parent, other.parent) && self.state == other.state
    }
}

impl<'a> Eq for GrammarConstraintState<'a> {}

impl<'a> Display for GrammarConstraintState<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "GrammarConstraintState ({} active tokenizer states):",
            self.state.len()
        )?;
        Ok(())
    }
}

impl<'a> GrammarConstraintState<'a> {
    pub(crate) fn transform_gss_stacks<M, F>(&mut self, mut f: F)
    where
        M: Default,
        F: FnMut(&mut Arc<GSSNode>, &mut M),
    {
        let mut memo = M::default();
        for s in self.state.values_mut() {
            f(&mut Arc::new(s.stack.clone()), &mut memo);
        }
    }

    pub(crate) fn map_gss_stacks<M, F>(&mut self, mut f: F)
    where
        M: Default,
        F: FnMut(&mut Arc<GSSNode>, &mut M) -> Arc<GSSNode>,
    {
        let mut memo = M::default();
        for s in self.state.values_mut() {
            s.stack = f(&mut Arc::new(s.stack.clone()), &mut memo)
                .as_ref()
                .clone();
        }
    }

    pub fn compute_commit_maps(
        &self,
        llm_token_bytes: &[u8],
    ) -> (
        BTreeMap<TokenizerStateID, TokenizerStateID>,
        BTreeMap<TokenizerStateID, TerminalBV>,
    ) {
        let mut state_map: BTreeMap<TokenizerStateID, TokenizerStateID> =
            BTreeMap::new();
        let mut terminals_map: BTreeMap<TokenizerStateID, TerminalBV> =
            BTreeMap::new();
        for (tokenizer_state_id, _state) in self.state.iter() {
            let exec_result = self.parent.tokenizer.execute_from_state(
                &llm_token_bytes,
                *tokenizer_state_id,
            );
            if let Some(new_state) = exec_result.end_state {
                state_map.insert(*tokenizer_state_id, TokenizerStateID(new_state));
            }
            let mut terminals = TerminalBV::zeros();
            for token in exec_result.matches {
                terminals.insert(token.id);
            }
            terminals_map.insert(*tokenizer_state_id, terminals);
        }
        (state_map, terminals_map)
    }

    pub fn get_mask(&self) -> LLMTokenBV {
        self.get_mask4().into()
    }

    pub fn print_gss_stats(&self) {
        // Unimplemented
    }

    pub fn print_gss(&self) {
        // Unimplemented
    }

    pub fn explain_stack(&self) {
        // Unimplemented
    }

    pub fn num_unique_nodes(&self) -> usize {
        0
    }

    pub fn commit(&mut self, llm_token_id: LLMTokenID) {
        self.commit_bytes(
            &self
                .parent
                .llm_vocab
                .llm_token_map
                .get_by_right(&llm_token_id)
                .unwrap()
                .clone(),
        );
    }

    pub fn is_active(&self) -> bool { !self.state.is_empty() }

    pub fn is_valid(&self) -> bool {
        if self.state.is_empty() {
            return false;
        }
        if self.state.contains_key(&self.parent.tokenizer.initial_state_id()) {
            return true;
        }
        for (tid, glr_state) in self.state.iter() {
            for gtid in
                self.parent
                    .tokenizer
                    .tokens_accessible_from_state(TokenizerStateID(tid.0))
            {
                let mut glr_state = glr_state.clone();
                glr_state.step(gtid);
                if glr_state.is_ok() {
                    return true;
                }
            }
        }
        false
    }

    pub fn state(&self) -> &BTreeMap<TokenizerStateID, GLRParserState<'a>> {
        &self.state
    }
}
