use super::*;
use std::mem::{size_of,offset_of};

#[test]
fn c_abi_masks_and_commits_match_direct_rust_every_mode() {
    let entries=vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(31,b"ab".to_vec()),(32,b"a".to_vec()),(65,b"z".to_vec())];
    let rust_vocab=core::vocab(entries.clone(),vec![]).unwrap();
    let ffi_entries:Vec<_>=entries.iter().map(|(id,b)|glrmask_vocab_entry{id:*id,bytes:b.as_ptr(),bytes_len:b.len()}).collect();
    let source="start ::= \"a\" \"b\"";
    unsafe {
        let (mut vocab,mut grammar)=(ptr::null_mut(),ptr::null_mut());
        assert_eq!(glrmask_vocab_new(ffi_entries.as_ptr(),ffi_entries.len(),ptr::null(),0,&mut vocab),0);
        assert_eq!(glrmask_grammar_from_source(0,source.as_ptr().cast(),source.len(),ptr::null(),&mut grammar),0);
        for optimization in 0..4 {
            let rust=g::Grammar::from_ebnf(source).compile_with(&rust_vocab,core::build_options(optimization,vec![],None).unwrap()).unwrap();
            let mut reference=rust.start();
            let mut options=glrmask_default_build_options();options.optimization=optimization;
            let (mut constraint,mut state)=(ptr::null_mut(),ptr::null_mut());
            assert_eq!(glrmask_grammar_compile(grammar,vocab,&options,&mut constraint),0);
            assert_eq!(glrmask_constraint_start(constraint,&mut state),0);
            glrmask_constraint_free(constraint); // Independent owning state.
            for token in [Some(0),Some(1),None] {
                let mut words=vec![u32::MAX;rust.mask_len()];
                assert_eq!(glrmask_state_fill_mask(state,words.as_mut_ptr(),words.len()),0);
                assert_eq!(words,reference.mask(),"mode {optimization}");
                let (mut accepting,mut rejected)=(99,99);
                assert_eq!(glrmask_state_is_accepting(state,&mut accepting),0);
                assert_eq!(glrmask_state_is_rejected(state,&mut rejected),0);
                assert_eq!(accepting!=0,reference.is_accepting());assert_eq!(rejected!=0,reference.is_rejected());
                if let Some(token)=token {assert_eq!(glrmask_state_commit_token(state,token),0);reference.commit_token(token).unwrap();}
            }
            glrmask_state_free(state);
        }
        glrmask_grammar_free(grammar);glrmask_vocab_free(vocab);
    }
}

#[test]
fn invalid_foreign_lengths_and_null_outputs_fail_before_access() {
    unsafe {
        let entry=glrmask_vocab_entry{id:0,bytes:ptr::null(),bytes_len:0};
        let mut vocab=ptr::dangling_mut();
        assert_eq!(glrmask_vocab_new(&entry,usize::MAX,ptr::null(),0,&mut vocab),1);
        assert!(vocab.is_null());
        assert_eq!(glrmask_vocab_new(&entry,1,ptr::null(),0,ptr::null_mut()),1);
        assert_eq!(glrmask_constraint_load(ptr::null(),1,ptr::null(),ptr::null_mut()),1);
    }
}

#[test]
fn header_layout_contract_on_supported_64_bit_platforms() {
    assert_eq!(size_of::<*const u8>(),8);
    assert_eq!(size_of::<glrmask_vocab_entry>(),24);
    assert_eq!(size_of::<glrmask_build_options>(),32);
    assert_eq!(size_of::<glrmask_lark_options>(),32);
    assert_eq!(size_of::<glrmask_requirement>(),56);
    assert_eq!(size_of::<glrmask_marker_event>(),16);
    assert_eq!(offset_of!(glrmask_marker_event,byte_offset),8);
    assert_eq!(offset_of!(glrmask_requirement,start_byte),24);
}
