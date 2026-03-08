#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use range_set_blaze::RangeSetBlaze;
use std::collections::BTreeMap;

use crate::runtime::Constraint;

fn encode_ranges(set: &RangeSetBlaze<u32>) -> Vec<[u32; 2]> {
    set.ranges()
        .map(|range| [*range.start(), *range.end()])
        .collect()
}

fn decode_ranges(ranges: Vec<[u32; 2]>) -> RangeSetBlaze<u32> {
    ranges
        .into_iter()
        .map(|[start, end]| start..=end)
        .collect()
}

pub(in crate::runtime) mod serde_btmap_rsb {
    use range_set_blaze::RangeSetBlaze;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::BTreeMap;

    pub(in crate::runtime) fn serialize<S: Serializer>(
        value: &BTreeMap<u32, BTreeMap<u32, RangeSetBlaze<u32>>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let encoded: BTreeMap<u32, BTreeMap<u32, Vec<[u32; 2]>>> = value
            .iter()
            .map(|(&tokenizer_state, terminal_map)| {
                let encoded_terminal_map = terminal_map
                    .iter()
                    .map(|(&terminal, token_set)| (terminal, super::encode_ranges(token_set)))
                    .collect();
                (tokenizer_state, encoded_terminal_map)
            })
            .collect();
        encoded.serialize(serializer)
    }

    pub(in crate::runtime) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeMap<u32, BTreeMap<u32, RangeSetBlaze<u32>>>, D::Error> {
        let encoded = BTreeMap::<u32, BTreeMap<u32, Vec<[u32; 2]>>>::deserialize(deserializer)?;
        Ok(encoded
            .into_iter()
            .map(|(tokenizer_state, terminal_map)| {
                let decoded_terminal_map = terminal_map
                    .into_iter()
                    .map(|(terminal, ranges)| (terminal, super::decode_ranges(ranges)))
                    .collect();
                (tokenizer_state, decoded_terminal_map)
            })
            .collect())
    }
}

impl Constraint {
    pub fn save(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Constraint serialization should succeed")
    }

    pub fn load(bytes: &[u8]) -> crate::Result<Self> {
        bincode::deserialize(bytes)
            .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))
    }
}
