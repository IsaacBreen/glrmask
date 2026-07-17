use std::collections::BTreeMap;
use std::sync::Arc;

use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::table::GLRTable;
use crate::grammar::flat::TerminalID;
use crate::Vocab;

use crate::runtime::{
    Constraint, ConstraintState, DynamicMaskVocab, DynamicTokenProgramPartition,
    SpecialTokenTerminal,
};

const DYNAMIC_CONSTRAINT_MAGIC: [u8; 8] = *b"GLRDYN\0\0";
const DYNAMIC_CONSTRAINT_VERSION: u16 = 8;
const DYNAMIC_CONSTRAINT_HEADER_LEN: usize = DYNAMIC_CONSTRAINT_MAGIC.len() + 2 + 8;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV1 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV1 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    eos_token_id: Option<u32>,
    token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV2 {
    v1: DynamicConstraintPayloadV1,
    special_token_terminals: Vec<SpecialTokenTerminal>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV3 {
    v2: DynamicConstraintPayloadV2,
    initial_token_program_partition: Option<DynamicTokenProgramPartition>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV2 {
    v1: LegacyDynamicConstraintPayloadV1,
    special_token_terminals: Vec<SpecialTokenTerminal>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV3 {
    v2: LegacyDynamicConstraintPayloadV2,
    initial_token_program_partition: Option<DynamicTokenProgramPartition>,
}

/// A constraint compiled only for direct lexer/parser masking.
///
/// Unlike [`Constraint`], this omits terminal-DWA, possible-match, parser-DWA,
/// token-remapping, and dense-mask compilation.
#[derive(Debug)]
pub struct DynamicConstraint {
    pub(crate) inner: Constraint,
}

impl DynamicConstraint {
    pub(crate) fn from_parts(
        table: GLRTable,
        terminal_display_names: Vec<String>,
        tokenizer: Tokenizer,
        ignore_terminal: Option<TerminalID>,
        special_token_terminals: Vec<SpecialTokenTerminal>,
        vocab: &Vocab,
    ) -> Self {
        let dynamic_mask_vocab =
            crate::compiler::constraint_possible_matches::runtime_dynamic_vocab_for_vocab(vocab);
        Self::from_payload_v2_with_dynamic_vocab(
            DynamicConstraintPayloadV2 {
                v1: DynamicConstraintPayloadV1 {
                    table,
                    terminal_display_names,
                    tokenizer,
                    ignore_terminal,
                    token_bytes: Arc::clone(&vocab.entries),
                },
                special_token_terminals,
            },
            dynamic_mask_vocab,
            None,
        )
    }

    fn migrate_legacy_v1(
        payload: LegacyDynamicConstraintPayloadV1,
    ) -> crate::Result<DynamicConstraintPayloadV1> {
        if payload.eos_token_id.is_some() {
            return Err(crate::GlrMaskError::Serialization(
                "legacy dynamic constraint artifact embeds Vocab-level EOS semantics; rebuild it with grammar-level end tokens"
                    .to_owned(),
            ));
        }
        Ok(DynamicConstraintPayloadV1 {
            table: payload.table,
            terminal_display_names: payload.terminal_display_names,
            tokenizer: payload.tokenizer,
            ignore_terminal: payload.ignore_terminal,
            token_bytes: payload.token_bytes,
        })
    }

    fn from_legacy_payload_v1(payload: LegacyDynamicConstraintPayloadV1) -> crate::Result<Self> {
        Ok(Self::from_payload_v2(DynamicConstraintPayloadV2 {
            v1: Self::migrate_legacy_v1(payload)?,
            special_token_terminals: Vec::new(),
        }))
    }

    fn from_legacy_payload_v2(payload: LegacyDynamicConstraintPayloadV2) -> crate::Result<Self> {
        Ok(Self::from_payload_v2(DynamicConstraintPayloadV2 {
            v1: Self::migrate_legacy_v1(payload.v1)?,
            special_token_terminals: payload.special_token_terminals,
        }))
    }

    fn from_legacy_payload_v3(payload: LegacyDynamicConstraintPayloadV3) -> crate::Result<Self> {
        Ok(Self::from_payload_v2_with_dynamic_vocab(
            DynamicConstraintPayloadV2 {
                v1: Self::migrate_legacy_v1(payload.v2.v1)?,
                special_token_terminals: payload.v2.special_token_terminals,
            },
            DynamicMaskVocab::default(),
            payload.initial_token_program_partition,
        ))
    }

    fn from_payload_v2(payload: DynamicConstraintPayloadV2) -> Self {
        Self::from_payload_v2_with_dynamic_vocab(
            payload,
            DynamicMaskVocab::default(),
            None,
        )
    }

    fn from_payload_v2_with_dynamic_vocab(
        payload: DynamicConstraintPayloadV2,
        dynamic_mask_vocab: DynamicMaskVocab,
        initial_token_program_partition: Option<DynamicTokenProgramPartition>,
    ) -> Self {
        let DynamicConstraintPayloadV2 {
            v1: payload,
            special_token_terminals,
        } = payload;
        let max_token_id = payload
            .token_bytes
            .keys()
            .next_back()
            .copied()
            .into_iter()
            .chain(special_token_terminals.iter().map(|special| special.token_id))
            .max()
            .unwrap_or(0);
        if let Some(partition) = initial_token_program_partition {
            dynamic_mask_vocab
                .install_initial_token_program_partition(Arc::new(partition));
        }
        let mut inner = Constraint {
            parser_dwa: DWA::new(payload.tokenizer.num_states(), max_token_id),
            parser_top_accept: BTreeMap::new(),
            table: payload.table,
            terminal_display_names: payload.terminal_display_names,
            tokenizer: payload.tokenizer,
            ignore_terminal: payload.ignore_terminal,
            special_token_terminals,
            dynamic_mask_vocab,
            possible_matches: BTreeMap::new(),
            state_to_internal_tsid: Vec::new(),
            internal_tsid_to_states: Vec::new(),
            template_dfas_by_terminal: Vec::new(),
            original_token_to_internal: Vec::new(),
            internal_token_to_tokens: Vec::new(),
            token_bytes: payload.token_bytes,
            internal_token_bytes: BTreeMap::new(),
            token_bytes_dense: Vec::new(),
            internal_token_buf_masks: Vec::new(),
            word_group_buf_masks: Vec::new(),
            pair_word_group_buf_masks: Vec::new(),
            quad_word_group_buf_masks: Vec::new(),
            super_word_group_buf_masks: Vec::new(),
            mega_word_group_buf_masks: Vec::new(),
            giga_word_group_buf_masks: Vec::new(),
            word_group_sparse_masks: Vec::new(),
            word_group_prefix_buf_masks: Vec::new(),
            word_group_sparse_prefix_entries: Vec::new(),
            quad_group_sparse_masks: Vec::new(),
            quad_group_dense_masks: Vec::new(),
            byte_group_sparse_masks: Vec::new(),
            byte_group_dense_masks: Vec::new(),
            word_group_sparse_total_entries: 0,
            word_group_sparse_max_entries: 0,
            all_tokens_buf_mask: Box::new([]),
            internal_token_dense_words: 0,
            weight_token_dense_masks: Default::default(),
            weight_token_buf_masks: Default::default(),
            weight_token_sparse_buf_masks: Default::default(),
            direct_sparse_weight_token_sets: Default::default(),
            seed_terminal_dense: Default::default(),
            seed_universe_dense: Arc::from(Vec::<u64>::new().into_boxed_slice()),
            dwa_fast_transitions: Vec::new(),
            tokenizer_fast_transitions: Vec::new(),
            heavy_token_dense_masks: Vec::new(),
            internal_token_buf_flat: Box::new([]),
            internal_token_buf_offsets: Box::new([]),
            total_internal_buf_cost: 0,
            heavy_token_indices: Vec::new(),
            heavy_total_cost: 0,
            light_avg_cost_x256: 0,
            internal_token_buf_op_costs: Vec::new(),
            word_group_buf_op_costs: Vec::new(),
            final_mask_mapping: Default::default(),
        };
        inner.rebuild_dynamic_runtime_caches();
        Self { inner }
    }

    pub fn save(&self) -> Vec<u8> {
        let payload = DynamicConstraintPayloadV3 {
            v2: DynamicConstraintPayloadV2 {
                v1: DynamicConstraintPayloadV1 {
                    table: self.inner.table.clone(),
                    terminal_display_names: self.inner.terminal_display_names.clone(),
                    tokenizer: self.inner.tokenizer.clone(),
                    ignore_terminal: self.inner.ignore_terminal,
                    token_bytes: Arc::clone(&self.inner.token_bytes),
                },
                special_token_terminals: self.inner.special_token_terminals.clone(),
            },
            initial_token_program_partition: self
                .inner
                .dynamic_mask_vocab
                .initial_token_program_partition()
                .map(|partition| partition.as_ref().clone()),
        };
        let payload = bincode::serialize(&payload)
            .expect("DynamicConstraint serialization should succeed");
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
        bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes
    }

    pub fn load(bytes: &[u8]) -> crate::Result<Self> {
        if bytes.len() < DYNAMIC_CONSTRAINT_HEADER_LEN
            || !bytes.starts_with(&DYNAMIC_CONSTRAINT_MAGIC)
        {
            return Err(crate::GlrMaskError::Serialization(
                "invalid dynamic constraint artifact header".to_owned(),
            ));
        }
        let version = u16::from_le_bytes([bytes[8], bytes[9]]);
        if !matches!(version, 1 | 2 | 7 | DYNAMIC_CONSTRAINT_VERSION) {
            return Err(crate::GlrMaskError::Serialization(format!(
                "unsupported dynamic constraint artifact version {version}",
            )));
        }
        let payload_len = usize::try_from(u64::from_le_bytes(
            bytes[10..18]
                .try_into()
                .expect("dynamic constraint header has fixed width"),
        ))
        .map_err(|_| {
            crate::GlrMaskError::Serialization(
                "dynamic constraint payload length does not fit this platform".to_owned(),
            )
        })?;
        if bytes.len() != DYNAMIC_CONSTRAINT_HEADER_LEN.saturating_add(payload_len) {
            return Err(crate::GlrMaskError::Serialization(
                "invalid dynamic constraint artifact payload length".to_owned(),
            ));
        }
        match version {
            1 => {
                let payload: LegacyDynamicConstraintPayloadV1 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_legacy_payload_v1(payload)
            }
            2 => {
                let payload: LegacyDynamicConstraintPayloadV2 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_legacy_payload_v2(payload)
            }
            7 => {
                let payload: LegacyDynamicConstraintPayloadV3 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_legacy_payload_v3(payload)
            }
            DYNAMIC_CONSTRAINT_VERSION => {
                let payload: DynamicConstraintPayloadV3 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Ok(Self::from_payload_v2_with_dynamic_vocab(
                    payload.v2,
                    DynamicMaskVocab::default(),
                    payload.initial_token_program_partition,
                ))
            }
            _ => unreachable!("version was validated above"),
        }
    }

    pub fn mask_len(&self) -> usize {
        self.inner.mask_len()
    }

    pub(crate) fn max_original_token_id(&self) -> Option<u32> {
        self.inner.max_original_token_id()
    }

    pub fn start(&self) -> DynamicConstraintState<'_> {
        DynamicConstraintState {
            inner: self.inner.start_dynamic(),
        }
    }
}

pub struct DynamicConstraintState<'a> {
    inner: ConstraintState<'a>,
}

impl<'a> DynamicConstraintState<'a> {
    pub fn commit_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.inner.commit_bytes(bytes)
    }

    pub fn commit_token(&mut self, token_id: u32) -> Result<(), String> {
        self.inner.commit_token_dynamic(token_id)
    }

    pub fn commit_tokens(&mut self, token_ids: &[u32]) -> Result<(), String> {
        self.inner.commit_tokens_dynamic(token_ids)
    }

    pub fn fill_mask(&self, buf: &mut [u32]) {
        self.inner.fill_mask_dynamic(buf);
    }

    pub fn fill_mask_bounded(&self, buf: &mut [u32], timeout_ms: u64) -> Result<(), String> {
        self.inner.fill_mask_dynamic_bounded(buf, timeout_ms)
    }

    pub fn forced(&self) -> Vec<u32> {
        self.inner.forced_dynamic()
    }

    pub fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    pub fn mask(&self) -> Vec<u32> {
        let mut mask = vec![0u32; self.inner.constraint.mask_len()];
        self.fill_mask(&mut mask);
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vocab {
        Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"aa".to_vec()),
            (4, b" ".to_vec()),
        ])
    }

    #[test]
    fn dynamic_constraint_matches_constraint_masks_and_commits() {
        let vocab = vocab();
        let grammar = r#"
            start start;
            t A ::= 'a'+;
            t B ::= 'b';
            nt start ::= A B;
        "#;
        let normal = crate::Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let dynamic = DynamicConstraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut normal_state = normal.start();
        let mut dynamic_state = dynamic.start();

        assert_eq!(normal_state.mask(), dynamic_state.mask());
        normal_state.commit_token(3).unwrap();
        dynamic_state.commit_token(3).unwrap();
        assert_eq!(normal_state.mask(), dynamic_state.mask());
        normal_state.commit_token(1).unwrap();
        dynamic_state.commit_token(1).unwrap();
        assert_eq!(normal_state.is_complete(), dynamic_state.is_complete());
        assert_eq!(normal_state.mask(), dynamic_state.mask());
    }

    #[test]
    fn dynamic_constraint_save_load_round_trip() {
        let vocab = vocab();
        let constraint = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        assert!(
            constraint
                .inner
                .dynamic_mask_vocab
                .initial_token_program_partition()
                .is_some()
        );
        let loaded = DynamicConstraint::load(&constraint.save()).unwrap();
        assert!(
            loaded
                .inner
                .dynamic_mask_vocab
                .initial_token_program_partition()
                .is_some()
        );
        assert_eq!(constraint.mask_len(), loaded.mask_len());
        assert_eq!(constraint.start().mask(), loaded.start().mask());
    }

    #[test]
    fn dynamic_forced_uses_dynamic_masks() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let constraint = DynamicConstraint::from_ebnf("start ::= 'a'", &vocab).unwrap();
        assert_eq!(constraint.start().forced(), vec![0]);
    }
}
