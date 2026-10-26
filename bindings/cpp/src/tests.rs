use super::*;
use std::mem::{size_of, offset_of};

#[test]
fn c_abi_masks_and_commits_match_direct_rust_every_mode() {
    let entries = vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (31, b"ab".to_vec()), (32, b"a".to_vec()), (65, b"z".to_vec())];
    let rust_vocab = core::vocab(entries.clone(), vec![]).unwrap();
    let ffi_entries: Vec<_> = entries.iter().map(|(id, b)| glrmask_vocab_entry { id: *id, bytes: b.as_ptr(), bytes_len: b.len() }).collect();
    let source = "start: \"a\" \"b\"";
    unsafe {
        let (mut vocab, mut grammar) = (ptr::null_mut(), ptr::null_mut());
        assert_eq!(glrmask_vocab_new(ffi_entries.as_ptr(), ffi_entries.len(), ptr::null(), 0, &mut vocab), 0);
        assert_eq!(glrmask_grammar_from_source(3, source.as_ptr().cast(), source.len(), &mut grammar), 0);
        for strategy in 0..3 {
            let rust_g = g::Grammar::from_lark(source).unwrap();
            let rust_m = g::Module::new(&rust_g);
            let rust_options = core::compile_options(
                core::mask_strategy(strategy).unwrap(),
                g::BoundaryMaskStrategy::Auto,
                g::TokenTrie::new(),
            );
            let rust = rust_m.compile_with(&rust_vocab, rust_options).unwrap();
            let mut reference = rust.start();
            let mut module = ptr::null_mut();
            assert_eq!(glrmask_module_new(grammar, &mut module), 0);
            let options = glrmask_compile_options {
                struct_size: size_of::<glrmask_compile_options>(),
                mask_strategy: strategy,
                boundary_mask_strategy: 0,
                token_trie: glrmask_token_trie_settings {
                    parser_backend: 0,
                    merge_equivalent_tokens: 0,
                },
            };
            let (mut constraint, mut state) = (ptr::null_mut(), ptr::null_mut());
            assert_eq!(glrmask_module_compile(module, vocab, &options, &mut constraint), 0);
            assert_eq!(glrmask_constraint_start(constraint, &mut state), 0);
            glrmask_constraint_free(constraint);
            glrmask_module_free(module);
            for token in [Some(0), Some(1), None] {
                let mut words = vec![u32::MAX; rust.mask_len()];
                assert_eq!(glrmask_state_fill_mask(state, words.as_mut_ptr(), words.len()), 0);
                assert_eq!(words, reference.mask(), "strategy {strategy}");
                let mut accepting = 99;
                assert_eq!(glrmask_state_is_accepting(state, &mut accepting), 0);
                assert_eq!(accepting != 0, reference.is_accepting());
                if let Some(token) = token { assert_eq!(glrmask_state_commit_token(state, token), 0); assert!(reference.commit_token(token)); }
            }
            glrmask_state_free(state);
        }
        glrmask_grammar_free(grammar); glrmask_vocab_free(vocab);
    }
}

#[test]
fn invalid_foreign_lengths_and_null_outputs_fail_before_access() {
    unsafe {
        let entry = glrmask_vocab_entry { id: 0, bytes: ptr::null(), bytes_len: 0 };
        let mut vocab = ptr::dangling_mut();
        assert_eq!(glrmask_vocab_new(&entry, usize::MAX, ptr::null(), 0, &mut vocab), 1);
        assert!(vocab.is_null());
        assert_eq!(glrmask_vocab_new(&entry, 1, ptr::null(), 0, ptr::null_mut()), 1);
        assert_eq!(glrmask_constraint_load(ptr::null(), 1, ptr::null(), ptr::null_mut()), 1);
    }
}

#[test]
fn header_layout_contract_on_supported_64_bit_platforms() {
    assert_eq!(size_of::<*const u8>(), 8);
    assert_eq!(size_of::<glrmask_vocab_entry>(), 24);
    assert_eq!(size_of::<glrmask_compile_options>(), 24);
    assert_eq!(size_of::<glrmask_extern>(), 24);
    assert_eq!(offset_of!(glrmask_extern, kind), 16);
}

#[test]
fn module_ffi_snapshots_expressions_storage_and_invalid_binds() {
    unsafe {
        let entries = [(0, b"x".as_slice()), (1, b"a".as_slice()), (2, b"y".as_slice()), (3, b"xay".as_slice())];
        let ffi_entries: Vec<_> = entries.iter().map(|(id, b)| glrmask_vocab_entry { id: *id, bytes: b.as_ptr(), bytes_len: b.len() }).collect();
        let mut vocab = ptr::null_mut(); assert_eq!(glrmask_vocab_new(ffi_entries.as_ptr(), ffi_entries.len(), ptr::null(), 0, &mut vocab), 0);
        let host = "glrm 1; start start; extern grammar child; nt start = \"x\" child \"y\";";
        let leaf = "start: \"a\"";
        let (mut grammar, mut child) = (ptr::null_mut(), ptr::null_mut());
        assert_eq!(glrmask_grammar_from_source(1, host.as_ptr().cast(), host.len(), &mut grammar), 0);
        assert_eq!(glrmask_grammar_from_source(3, leaf.as_ptr().cast(), leaf.len(), &mut child), 0);
        let (mut parent, mut clone, mut prepared) = (ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
        assert_eq!(glrmask_module_new(grammar, &mut parent), 0);
        assert_eq!(glrmask_module_precompile(parent, vocab, ptr::null()), 0);
        assert_eq!(glrmask_module_clone(parent, &mut clone), 0);

        assert_eq!(glrmask_module_new(child, &mut prepared), 0);
        let parse_options = glrmask_compile_options {
            struct_size: size_of::<glrmask_compile_options>(),
            mask_strategy: 1,
            boundary_mask_strategy: 0,
            token_trie: glrmask_token_trie_settings {
                parser_backend: 0,
                merge_equivalent_tokens: 0,
            },
        };
        assert_eq!(glrmask_module_precompile(prepared, vocab, &parse_options), 0);

        // Unknown extern
        assert_ne!(glrmask_module_bind_grammar(parent, c"missing".as_ptr(), 7, child), 0);

        // Check unbound externs
        let mut slots = ptr::null_mut(); assert_eq!(glrmask_module_unbound_externs(parent, &mut slots), 0);
        let mut count = 0; assert_eq!(glrmask_externs_len(slots, &mut count), 0); assert_eq!(count, 1);
        let mut ext = glrmask_extern { name: ptr::null(), name_len: 0, kind: 0 };
        assert_eq!(glrmask_externs_get(slots, 0, &mut ext), 0);
        assert_eq!(std::slice::from_raw_parts(ext.name.cast::<u8>(), ext.name_len), b"child");
        glrmask_externs_free(slots);

        // Bind prepared child
        assert_eq!(glrmask_module_bind_module(parent, c"child".as_ptr(), 5, prepared), 0);
        // Rebinding error
        assert_ne!(glrmask_module_bind_module(parent, c"child".as_ptr(), 5, prepared), 0);

        // Save & Load
        let mut artifact = ptr::null_mut(); assert_eq!(glrmask_module_save(parent, &mut artifact), 0);
        let (mut data, mut len) = (ptr::null(), 0); assert_eq!(glrmask_bytes_data(artifact, &mut data, &mut len), 0);
        let mut loaded = ptr::null_mut(); assert_eq!(glrmask_module_load(data, len, &mut loaded), 0);
        glrmask_bytes_free(artifact); glrmask_module_free(parent); glrmask_module_free(prepared);
        glrmask_grammar_free(grammar); glrmask_grammar_free(child);

        let mut constraint = ptr::null_mut(); assert_eq!(glrmask_module_compile(loaded, vocab, ptr::null(), &mut constraint), 0);
        let mut state = ptr::null_mut(); assert_eq!(glrmask_constraint_start(constraint, &mut state), 0);
        glrmask_constraint_free(constraint); glrmask_module_free(loaded);
        let mut words = [0u32; 1]; assert_eq!(glrmask_state_fill_mask(state, words.as_mut_ptr(), 1), 0); assert_ne!(words[0] & (1 << 3), 0);
        assert_eq!(glrmask_state_commit_token(state, 3), 0); let mut accepting = 0; assert_eq!(glrmask_state_is_accepting(state, &mut accepting), 0); assert_eq!(accepting, 1); glrmask_state_free(state);

        // Expr literal test
        let mut lit_expr = ptr::null_mut();
        assert_eq!(glrmask_expr_literal(b"x".as_ptr(), 1, &mut lit_expr), 0);
        glrmask_expr_free(lit_expr);

        glrmask_module_free(clone);
        glrmask_vocab_free(vocab);
    }
}

#[test]
fn older_shorter_compile_options_default_the_new_switches_to_zero() {
    // The original ABI struct had only `struct_size`, `mask_strategy` and
    // `boundary_mask_strategy`. A caller built against it must keep working:
    // the new fields count as zero because `struct_size` does not cover them.
    #[repr(C)]
    struct OldOptions {
        struct_size: usize,
        mask_strategy: u32,
        boundary_mask_strategy: u32,
    }
    let old = OldOptions {
        struct_size: size_of::<OldOptions>(),
        mask_strategy: 2,
        boundary_mask_strategy: 1,
    };
    let options = unsafe { root_options((&old as *const OldOptions).cast()) }.unwrap();
    let reference = core::compile_options(
        g::MaskStrategy::TokenTrie(g::TokenTrie::new()),
        g::BoundaryMaskStrategy::ParserDwa,
        g::TokenTrie::new(),
    );
    assert_eq!(options, reference);

    // A struct shorter than the original ABI is still rejected.
    let too_short = OldOptions {
        struct_size: size_of::<usize>(),
        mask_strategy: 0,
        boundary_mask_strategy: 0,
    };
    assert!(unsafe { root_options((&too_short as *const OldOptions).cast()) }.is_err());
}
