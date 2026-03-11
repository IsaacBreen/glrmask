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
        let mut constraint: Self = bincode::deserialize(bytes)
            .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
        constraint.build_buf_masks();
        Ok(constraint)
    }
}

#[cfg(test)]
mod tests {
    use crate::runtime::Constraint;
    use crate::Vocab;

    #[test]
    fn test_save_load_preserves_internal_possible_matches() {
        let vocab = Vocab::new(
            vec![
                (10, b"a".to_vec()),
                (20, b"a".to_vec()),
                (30, b"b".to_vec()),
            ],
            None,
        );
        let constraint = Constraint::from_ebnf(r#"start ::= "a""#, &vocab).unwrap();
        let saved = constraint.save();
        let loaded = Constraint::load(&saved).unwrap();

        let tokenizer_state = loaded.tokenizer.initial_state();
        let internal_token = loaded.internal_token_for_original(10);

        let internal_matches: std::collections::BTreeSet<u32> = loaded
            .possible_matches_for_state_internal(tokenizer_state)
            .values()
            .flat_map(|token_ids| token_ids.iter())
            .collect();
        assert_eq!(internal_matches, std::collections::BTreeSet::from([internal_token]));

        let mask = loaded.start().mask();
        let word_10 = 10usize / 32;
        let bit_10 = 10usize % 32;
        let word_20 = 20usize / 32;
        let bit_20 = 20usize % 32;
        let word_30 = 30usize / 32;
        let bit_30 = 30usize % 32;
        assert_ne!(mask[word_10] & (1u32 << bit_10), 0);
        assert_ne!(mask[word_20] & (1u32 << bit_20), 0);
        assert_eq!(mask[word_30] & (1u32 << bit_30), 0);
    }
}
