pub(crate) mod stages {
    pub(crate) mod equiv_types {
        #[derive(Debug, Clone)]
        pub struct ManyToOneIdMap {
            pub original_to_internal: Vec<u32>,
            pub internal_to_originals: Vec<Vec<u32>>,
            pub representative_original_ids: Vec<u32>,
        }

        impl ManyToOneIdMap {
            pub fn from_original_to_internal_allowing_unmapped(
                original_to_internal: Vec<u32>,
                num_internal: u32,
            ) -> Self {
                let mut internal_to_originals = vec![Vec::new(); num_internal as usize];
                let mut representative_original_ids = vec![u32::MAX; num_internal as usize];
                for (original, &internal) in original_to_internal.iter().enumerate() {
                    if internal == u32::MAX || (internal as usize) >= internal_to_originals.len() {
                        continue;
                    }
                    let originals = &mut internal_to_originals[internal as usize];
                    if originals.is_empty() {
                        representative_original_ids[internal as usize] = original as u32;
                    }
                    originals.push(original as u32);
                }
                Self { original_to_internal, internal_to_originals, representative_original_ids }
            }
        }
    }
}

#[path = "../../src/compiler/glr/mod.rs"]
pub(crate) mod glr;
