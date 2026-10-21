//! Regression for source-local DEFAULT_LABEL semantics in parser-family union.
//! Adding a vocabulary token cannot invalidate another token's byte language.
use glrmask::{BuildOptions, Constraint, Grammar, MaskStrategy, Vocab};
use glrmask::__private::ConstraintStateExt as _;
const GRAMMAR: &str = include_str!("data/parser_family_fallback/reduced_sql.lark");
const PREFIX: &[u8] = b"SELECT u.name";

#[test]
fn cross_lexeme_vocabulary_token_does_not_hide_valid_comma() {
    for optimization in [MaskStrategy::TokenTrie, MaskStrategy::TokenClassTrie, MaskStrategy::ParserDwa] {
        for additional in [None, Some(b" =".as_slice()), Some(b" >".as_slice()), Some(b" ,".as_slice())] {
            let mut entries = vec![(0, b",".to_vec())];
            if let Some(bytes) = additional { entries.push((1, bytes.to_vec())); }
            let vocab = Vocab::new(entries);
            let constraint = Grammar::from_lark(GRAMMAR)
                .compile_with_build_options(&vocab, BuildOptions::default().optimization(optimization)).unwrap();
            let loaded = Constraint::load(constraint.save()).unwrap();
            for artifact in [&constraint, &loaded] {
            let mut state = artifact.start();
            state.commit_bytes(PREFIX).unwrap();
            let mask = state.mask();
            let reference = state.fill_mask_dynamic_vec();
            assert_ne!(reference[0] & 1, 0, "independent native-parser reference must admit comma");
            assert_ne!(mask[0] & 1, 0, "adding {:?} must not remove comma in {:?}", additional, optimization);
            assert_eq!(mask, reference, "mask/reference parity for {:?} in {:?}", additional, optimization);
            state.commit_token(0).unwrap();
            }
        }
    }
}
