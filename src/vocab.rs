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
        unimplemented!()
    }

    pub fn len(&self) -> usize {
        unimplemented!()
    }

    pub fn is_empty(&self) -> bool {
        unimplemented!()
    }

    pub fn max_token_id(&self) -> u32 {
        unimplemented!()
    }
}
