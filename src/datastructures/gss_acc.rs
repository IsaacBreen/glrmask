use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::leveled_gss::Merge as LGMerge;
use std::collections::BTreeMap;
use std::ops::BitOrAssign;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Acc {
    pub llm_tokens_union: HybridBitset,
    pub terminals_union: BTreeMap<usize, HybridBitset>,
}

impl Acc {
    pub fn new_fresh() -> Self {
        Self {
            llm_tokens_union: HybridBitset::max_ones(),
            terminals_union: BTreeMap::new(),
        }
    }

    pub fn new_dead() -> Self {
        Self {
            llm_tokens_union: HybridBitset::zeros(),
            terminals_union: BTreeMap::new(),
        }
    }

    pub fn is_default(&self) -> bool {
        self.llm_tokens_union == HybridBitset::max_ones()
            && self.terminals_union.is_empty()
    }

    pub fn union_llm_tokens(&self) -> HybridBitset {
        self.llm_tokens_union.clone()
    }
}

impl LGMerge for Acc {
    fn merge(&self, other: &Self) -> Self {
        let mut new_terminals_union = self.terminals_union.clone();
        for (k, v) in &other.terminals_union {
            new_terminals_union.entry(*k).or_default().bitor_assign(v);
        }

        Acc {
            llm_tokens_union: &self.llm_tokens_union | &other.llm_tokens_union,
            terminals_union: new_terminals_union,
        }
    }
}
