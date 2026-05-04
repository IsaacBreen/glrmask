use range_set_blaze::RangeSetBlaze;

use crate::runtime::Constraint;

type EncodedRanges = Vec<[u32; 2]>;

fn encode_ranges(set: &RangeSetBlaze<u32>) -> EncodedRanges {
    set.ranges()
        .map(|range| [*range.start(), *range.end()])
        .collect()
}

fn decode_ranges(ranges: EncodedRanges) -> RangeSetBlaze<u32> {
    ranges
        .into_iter()
        .map(|[start, end]| start..=end)
        .collect()
}

fn encode_terminal_ranges(
    terminal_map: &std::collections::BTreeMap<u32, RangeSetBlaze<u32>>,
) -> std::collections::BTreeMap<u32, EncodedRanges> {
    terminal_map
        .iter()
        .map(|(&terminal, token_set)| (terminal, encode_ranges(token_set)))
        .collect()
}

fn decode_terminal_ranges(
    terminal_map: std::collections::BTreeMap<u32, EncodedRanges>,
) -> std::collections::BTreeMap<u32, RangeSetBlaze<u32>> {
    terminal_map
        .into_iter()
        .map(|(terminal, ranges)| (terminal, decode_ranges(ranges)))
        .collect()
}

pub(in crate::runtime) mod serde_btreemap_rangeset {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::BTreeMap;

    type EncodedPossibleMatches = BTreeMap<u32, BTreeMap<u32, super::EncodedRanges>>;

    pub(in crate::runtime) fn serialize<S: Serializer>(
        value: &BTreeMap<u32, BTreeMap<u32, Box<[u64]>>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let encoded: EncodedPossibleMatches = value
            .iter()
            .map(|(&tokenizer_state, terminal_map)| {
                let encoded_terminals: BTreeMap<u32, super::EncodedRanges> = terminal_map
                    .iter()
                    .map(|(&terminal, bitmap)| {
                        let rangeset = super::super::constraint::bitmap_to_rangeset(bitmap);
                        (terminal, super::encode_ranges(&rangeset))
                    })
                    .collect();
                (tokenizer_state, encoded_terminals)
            })
            .collect();
        encoded.serialize(serializer)
    }

    pub(in crate::runtime) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeMap<u32, BTreeMap<u32, Box<[u64]>>>, D::Error> {
        let encoded = EncodedPossibleMatches::deserialize(deserializer)?;
        Ok(encoded
            .into_iter()
            .map(|(tokenizer_state, terminal_map)| {
                let decoded: BTreeMap<u32, Box<[u64]>> = terminal_map
                    .into_iter()
                    .map(|(terminal, ranges)| {
                        let rangeset = super::decode_ranges(ranges);
                        let max_id = rangeset.last().unwrap_or(0);
                        let num_words = (max_id as usize / 64) + 1;
                        let mut words = vec![0u64; num_words];
                        for id in rangeset.iter() {
                            words[id as usize / 64] |= 1u64 << (id % 64);
                        }
                        (terminal, words.into_boxed_slice())
                    })
                    .collect();
                (tokenizer_state, decoded)
            })
            .collect())
    }
}

impl Constraint {
    pub fn save(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Constraint serialization should succeed")
    }

    pub fn load(bytes: &[u8]) -> crate::Result<Self> {
        let profile = std::env::var_os("GLRMASK_PROFILE_LOAD").is_some();
        let total_started = std::time::Instant::now();
        let deserialize_started = std::time::Instant::now();
        let mut constraint: Self = bincode::deserialize(bytes)
            .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
        if profile {
            eprintln!(
                "[glrmask/profile][load] phase=deserialize ms={:.3} bytes={}",
                deserialize_started.elapsed().as_secs_f64() * 1000.0,
                bytes.len(),
            );
        }
        let rebuild_started = std::time::Instant::now();
        constraint.rebuild_runtime_caches();
        if profile {
            eprintln!(
                "[glrmask/profile][load] phase=rebuild_runtime_caches ms={:.3} total_ms={:.3}",
                rebuild_started.elapsed().as_secs_f64() * 1000.0,
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok(constraint)
    }
}
