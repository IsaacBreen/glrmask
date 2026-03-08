#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Vocab {
    pub entries: Vec<(u32, Vec<u8>)>,
    pub eos_token_id: Option<u32>,
}

impl Vocab {
    const EOS_BYTES: &[u8] = b"<|endoftext|>";

    pub fn new(entries: Vec<(u32, Vec<u8>)>, eos_token_id: Option<u32>) -> Self {
        let mut entries = entries;
        entries.sort_by_key(|(token_id, _)| *token_id);

        let eos_token_id = eos_token_id.or_else(|| {
            entries
                .iter()
                .find(|(_, bytes)| bytes.as_slice() == Self::EOS_BYTES)
                .map(|(token_id, _)| *token_id)
        });

        Self {
            entries,
            eos_token_id,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn max_token_id(&self) -> u32 {
        self.entries
            .iter()
            .map(|(token_id, _)| *token_id)
            .max()
            .unwrap_or(0)
    }
}
