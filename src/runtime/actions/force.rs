




#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::runtime::state::ConstraintState;

// SEP1_MAP: this file corresponds to sep1 forcing logic in
// `grammars2024/src/constraint.rs::force()`, but glrmask targets forced token
// sequences instead of sep1's byte-prefix-first forcing flow.
impl<'a> ConstraintState<'a> {
    // SEP1_MAP: `force()` is a rewrite of sep1 `GrammarConstraintState::force()`.
    // sep1 computes a tokenization-safe forced byte prefix first; glrmask's
    // intended surface is token-level forcing directly.
    
    
    
    
    
    
    
    
    
    pub fn force(&self) -> Vec<u32> {
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

        let mut forced = Vec::new();
        let mut cursor = self.clone();

        loop {
            let mask = cursor.mask();
            let Some(token) = single_allowed_token(&mask) else {
                break;
            };
            forced.push(token);
            cursor.commit_token(token);
            if cursor.state.is_empty() || cursor.is_complete() {
                break;
            }
        }

        forced
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
        let mask = s.mask();

        
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
