//! Bounded, lazy reuse of lexer derivatives across mask calls.
//!
//! Only lexical state belongs here: no parser stacks, mask output, deadlines,
//! or provisional maximal-munch guards. An admitted-terminal set is part of
//! every projected configuration key. Runtime clones and newly compiled
//! constraints start with an empty slot; nothing is serialized or prebuilt.

use super::*;
use std::sync::Mutex;

#[derive(Debug, Default)]
pub(crate) struct DynamicNfaMemoSlot(Mutex<Option<Box<DynamicNfaMemo>>>);

impl Clone for DynamicNfaMemoSlot {
    fn clone(&self) -> Self {
        // Unlike vocabulary-derived data, lexer coordinates cannot be carried
        // into another runtime/grammar merely because it shares a vocabulary.
        Self::default()
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct DynamicNfaMemo {
    owner: usize,
    projection_enabled: bool,
    config_ids: FxHashMap<Vec<u32>, u32>,
    configs: Vec<Box<[u32]>>,
    admitted_sets: Vec<BitSet>,
    admitted_set_ids: FxHashMap<Vec<u64>, u32>,
    admitted_config_ids: FxHashMap<(u32, Vec<u32>), u32>,
    config_admitted: Vec<Option<u32>>,
    fresh_reset_ids: FxHashMap<u32, u32>,
    config_is_fresh_reset: Vec<bool>,
    transitions: Vec<Option<Box<[u32; 256]>>>,
    residual_configs: Vec<u32>,
    config_matched: Vec<BitSet>,
    config_futures: Vec<BitSet>,
    raw_start_config: FxHashMap<u32, u32>,
    projected_roots: FxHashMap<(u32, u32), u32>,
    memo_member_words: usize,
}

impl DynamicNfaMemo {
    fn exchange(&mut self, cache: &mut DynamicNfaScanCache<'_>) {
        macro_rules! exchange {
            ($($field:ident),+ $(,)?) => { $(std::mem::swap(&mut self.$field, &mut cache.$field);)+ };
        }
        exchange!(
            config_ids, configs, admitted_sets, admitted_set_ids,
            admitted_config_ids, config_admitted, fresh_reset_ids,
            config_is_fresh_reset, transitions, residual_configs,
            config_matched, config_futures, raw_start_config, projected_roots,
            memo_member_words,
        );
    }
}

impl DynamicNfaScanCache<'_> {
    pub(super) fn restore_memo(&mut self, vocab: &DynamicMaskVocab) {
        if self.deterministic
            || !self.use_constraint_fast_transitions
            || std::env::var_os("GLRMASK_DISABLE_NFA_DERIVATIVE_REUSE").is_some()
        {
            return;
        }
        // Never serialize concurrent decoders behind this optional cache.
        // A busy slot simply gives the caller a private exact workspace.
        let Ok(mut slot) = vocab.nfa_memo.0.try_lock() else { return };
        let mut memo = slot.take().unwrap_or_default();
        drop(slot);
        let owner = self.tokenizer as *const Tokenizer as usize;
        if memo.owner != owner || memo.projection_enabled != self.parser_projection_enabled {
            *memo = DynamicNfaMemo::default();
            memo.owner = owner;
            memo.projection_enabled = self.parser_projection_enabled;
        }
        memo.exchange(self);
        self.memo_recycle = Some(memo);
    }

    pub(super) fn recycle_memo(&mut self, vocab: &DynamicMaskVocab) {
        let Some(mut memo) = self.memo_recycle.take() else { return };
        // Retention limits, not correctness/work limits. Oversized workspaces
        // finish normally and are simply not retained for the next request.
        if self.configs.len() > 2048
            || self.config_ids.len() > 4096
            || self.admitted_config_ids.len() > 4096
            || self.admitted_sets.len() > 256
            || self.raw_start_config.len() > 4096
            || self.projected_roots.len() > 4096
            || self.memo_member_words > 65536
        {
            return;
        }
        memo.exchange(self);
        if let Ok(mut slot) = vocab.nfa_memo.0.try_lock() {
            // Another concurrent caller may have returned its workspace first.
            // Keep that one rather than dropping a large cache under the lock.
            if slot.is_none() {
                *slot = Some(memo);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloned_slot_does_not_share_lexer_coordinates() {
        let slot = DynamicNfaMemoSlot::default();
        *slot.0.lock().unwrap() = Some(Box::default());
        let cloned = slot.clone();
        assert!(cloned.0.lock().unwrap().is_none());
        assert!(slot.0.lock().unwrap().is_some());
    }

    #[test]
    fn slot_contention_is_nonblocking() {
        let slot = DynamicNfaMemoSlot::default();
        let _owner = slot.0.lock().unwrap();
        assert!(slot.0.try_lock().is_err());
    }
}
