




#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::runtime::state::ConstraintState;

impl<'a> ConstraintState<'a> {
    
    
    
    
    
    
    
    
    
    pub fn force(&self) -> Vec<u32> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Constraint;
    use crate::Vocab;

    fn single_allowed_token(mask: &[u32]) -> Option<u32> {
        let mut found = None;
        for (word_index, &word) in mask.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as u32;
                let token = word_index as u32 * 32 + bit;
                if found.replace(token).is_some() {
                    return None;
                }
                bits &= bits - 1;
            }
        }
        found
    }

    fn mask_is_empty(mask: &[u32]) -> bool {
        mask.iter().all(|word| *word == 0)
    }

    fn make_vocab(entries: &[&str]) -> Vocab {
        let entries: Vec<(u32, Vec<u8>)> = entries
            .iter()
            .enumerate()
            .map(|(i, s)| (i as u32, s.as_bytes().to_vec()))
            .collect();
        Vocab::new(entries, None)
    }

    #[test]
    fn test_forced_token_detection() {
        let vocab = make_vocab(&["a", "b"]);
        let c = Constraint::from_ebnf(r#"start ::= "a""#, &vocab).unwrap();
        let s = c.start();
        let mask = s.mask_view().mask();

        
        assert_eq!(single_allowed_token(&mask), Some(0));
        assert!(!mask_is_empty(&mask));
    }

    #[test]
    fn test_forced_single() {
        let mut mask = vec![0u32; 1];
        mask[0] |= 1u32 << 5;
        assert_eq!(single_allowed_token(&mask), Some(5));
    }

    #[test]
    fn test_forced_multi() {
        let mut mask = vec![0u32; 1];
        mask[0] |= 1u32 << 5;
        mask[0] |= 1u32 << 7;
        assert_eq!(single_allowed_token(&mask), None);
    }

    #[test]
    fn test_forced_empty() {
        let mask = vec![0u32; 1];
        assert_eq!(single_allowed_token(&mask), None);
        assert!(mask_is_empty(&mask));
    }
}
