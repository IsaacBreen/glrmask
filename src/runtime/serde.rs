use std::collections::BTreeMap;
use std::sync::Arc;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::table::GLRTable;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;
use crate::runtime::artifact::{empty_dense_words, PossibleMatchesByTerminal};
use crate::runtime::Constraint;

/// Deliberate, execution-only persisted state for a compiled constraint.
///
/// This excludes every derived lookup table, dense mask, and cache. Loading it
/// reconstructs those structures through `Constraint::rebuild_runtime_caches`,
/// so artifact compatibility is governed by this named contract rather than the
/// incidental serde layout of `Constraint`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct RuntimePayloadV1 {
    parser_dwa: DWA,
    table: GLRTable,
    terminal_display_names: Vec<String>,
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    possible_matches: PossibleMatchesByTerminal,
    state_to_internal_tsid: Vec<u32>,
    internal_tsid_to_states: Vec<Vec<u32>>,
    original_token_to_internal: Vec<u32>,
    internal_token_to_tokens: Vec<Vec<u32>>,
    eos_token_id: Option<u32>,
    token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
    internal_token_bytes: BTreeMap<u32, Vec<u8>>,
}

/// V2 adds the depth-one parser acceptance overlay while nesting the complete
/// V1 payload unchanged. This keeps the named V1 bincode contract stable.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct RuntimePayloadV2 {
    v1: RuntimePayloadV1,
    parser_top_accept: BTreeMap<i32, Weight>,
}

impl From<&Constraint> for RuntimePayloadV1 {
    fn from(constraint: &Constraint) -> Self {
        Self {
            parser_dwa: constraint.parser_dwa.clone(),
            table: constraint.table.clone(),
            terminal_display_names: constraint.terminal_display_names.clone(),
            tokenizer: constraint.tokenizer.clone(),
            ignore_terminal: constraint.ignore_terminal,
            possible_matches: constraint.possible_matches.clone(),
            state_to_internal_tsid: constraint.state_to_internal_tsid.clone(),
            internal_tsid_to_states: constraint.internal_tsid_to_states.clone(),
            original_token_to_internal: constraint.original_token_to_internal.clone(),
            internal_token_to_tokens: constraint.internal_token_to_tokens.clone(),
            eos_token_id: constraint.eos_token_id,
            token_bytes: constraint.token_bytes.clone(),
            internal_token_bytes: constraint.internal_token_bytes.clone(),
        }
    }
}

impl RuntimePayloadV1 {
    fn into_constraint(self) -> Constraint {
        self.into_constraint_with_top_accept(BTreeMap::new())
    }

    fn into_constraint_with_top_accept(
        self,
        parser_top_accept: BTreeMap<i32, Weight>,
    ) -> Constraint {
        Constraint {
            parser_dwa: self.parser_dwa,
            parser_top_accept,
            table: self.table,
            terminal_display_names: self.terminal_display_names,
            tokenizer: self.tokenizer,
            ignore_terminal: self.ignore_terminal,
            dynamic_mask_vocab: Default::default(),
            possible_matches: self.possible_matches,
            state_to_internal_tsid: self.state_to_internal_tsid,
            internal_tsid_to_states: self.internal_tsid_to_states,
            template_dfas_by_terminal: Vec::new(),
            original_token_to_internal: self.original_token_to_internal,
            internal_token_to_tokens: self.internal_token_to_tokens,
            eos_token_id: self.eos_token_id,
            token_bytes: self.token_bytes,
            internal_token_bytes: self.internal_token_bytes,
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
            byte_group_sparse_masks: Vec::new(),
            word_group_sparse_total_entries: 0,
            word_group_sparse_max_entries: 0,
            all_tokens_buf_mask: Vec::new().into_boxed_slice(),
            internal_token_dense_words: 0,
            weight_token_dense_masks: Default::default(),
            weight_token_buf_masks: Default::default(),
            weight_token_sparse_buf_masks: Default::default(),
            direct_sparse_weight_token_sets: Default::default(),
            seed_terminal_dense: Default::default(),
            seed_universe_dense: empty_dense_words(),
            dwa_fast_transitions: Vec::new(),
            tokenizer_fast_transitions: Vec::new(),
            heavy_token_dense_masks: Vec::new(),
            internal_token_buf_flat: Vec::new().into_boxed_slice(),
            internal_token_buf_offsets: Vec::new().into_boxed_slice(),
            total_internal_buf_cost: 0,
            heavy_token_indices: Vec::new(),
            heavy_total_cost: 0,
            light_avg_cost_x256: 0,
            internal_token_buf_op_costs: Vec::new(),
            word_group_buf_op_costs: Vec::new(),
            final_mask_mapping: Default::default(),
        }
    }
}

impl From<&Constraint> for RuntimePayloadV2 {
    fn from(constraint: &Constraint) -> Self {
        Self {
            v1: RuntimePayloadV1::from(constraint),
            parser_top_accept: constraint.parser_top_accept.clone(),
        }
    }
}

impl RuntimePayloadV2 {
    fn into_constraint(self) -> Constraint {
        self.v1
            .into_constraint_with_top_accept(self.parser_top_accept)
    }
}

impl Constraint {
    /// Serialize the intentional v1 execution payload used by `glrmask-runtime`.
    /// Compiler-only data and derived runtime caches are deliberately absent.
    ///
    /// V1 cannot represent the depth-one parser acceptance overlay introduced
    /// by split parser-family compilation. Use [`Self::save_runtime_payload_v2`]
    /// when that overlay is present.
    pub fn save_runtime_payload_v1(&self) -> Vec<u8> {
        assert!(
            self.parser_top_accept.is_empty(),
            "runtime payload V1 cannot represent parser_top_accept; use save_runtime_payload_v2"
        );
        bincode::serialize(&RuntimePayloadV1::from(self))
            .expect("Runtime payload serialization should succeed")
    }

    /// Load the intentional v1 execution payload and rebuild all derived caches.
    pub fn load_runtime_payload_v1(bytes: &[u8]) -> crate::Result<Self> {
        let payload: RuntimePayloadV1 = bincode::deserialize(bytes)
            .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
        let mut constraint = payload.into_constraint();
        constraint.rebuild_runtime_caches();
        Ok(constraint)
    }

    /// Serialize the intentional v2 execution payload, including the
    /// depth-one parser acceptance overlay.
    pub fn save_runtime_payload_v2(&self) -> Vec<u8> {
        bincode::serialize(&RuntimePayloadV2::from(self))
            .expect("Runtime payload V2 serialization should succeed")
    }

    /// Load an intentional v2 execution payload and rebuild derived caches.
    pub fn load_runtime_payload_v2(bytes: &[u8]) -> crate::Result<Self> {
        let payload: RuntimePayloadV2 = bincode::deserialize(bytes)
            .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
        let mut constraint = payload.into_constraint();
        constraint.rebuild_runtime_caches();
        Ok(constraint)
    }

    pub fn save(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Constraint serialization should succeed")
    }

    pub fn load(bytes: &[u8]) -> crate::Result<Self> {
        let mut constraint: Self = bincode::deserialize(bytes)
            .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
        constraint.rebuild_runtime_caches();
        Ok(constraint)
    }
}

#[cfg(test)]
mod tests {
    use range_set_blaze::RangeSetBlaze;

    use super::*;

    #[test]
    fn runtime_payload_v2_preserves_top_accept_and_v1_rejects_it() {
        let vocab = crate::Vocab::new(vec![(0, b"a".to_vec())], None);
        let mut constraint = Constraint::from_lark(
            r#"
                start: "a"
            "#,
            &vocab,
        )
        .expect("test constraint should compile");
        constraint.parser_top_accept.insert(
            7,
            Weight::from_per_tsid_token_sets(std::iter::once((
                0,
                RangeSetBlaze::from_iter(std::iter::once(0..=0)),
            ))),
        );

        assert!(
            std::panic::catch_unwind(|| constraint.save_runtime_payload_v1()).is_err(),
            "V1 must reject an overlay it cannot represent"
        );
        let bytes = constraint.save_runtime_payload_v2();
        let loaded = Constraint::load_runtime_payload_v2(&bytes).unwrap();
        assert_eq!(loaded.parser_top_accept, constraint.parser_top_accept);
    }
}
