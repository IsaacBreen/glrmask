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

#[test]
fn module_ffi_snapshots_expressions_storage_and_invalid_binds() {
    unsafe {
        let rust_vocab=core::vocab(vec![(0,b"x".to_vec()),(1,b"a".to_vec()),(2,b"y".to_vec()),(3,b"xay".to_vec())],vec![]).unwrap();
        let entries=[(0,b"x".as_slice()),(1,b"a".as_slice()),(2,b"y".as_slice()),(3,b"xay".as_slice())];
        let ffi_entries:Vec<_>=entries.iter().map(|(id,b)|glrmask_vocab_entry{id:*id,bytes:b.as_ptr(),bytes_len:b.len()}).collect();
        let mut vocab=ptr::null_mut();assert_eq!(glrmask_vocab_new(ffi_entries.as_ptr(),ffi_entries.len(),ptr::null(),0,&mut vocab),0);
        let host="glrm 1; start start; extern grammar child; nt start = \"x\" child \"y\";";
        let leaf="start ::= \"a\"";
        let (mut grammar,mut child)=(ptr::null_mut(),ptr::null_mut());
        assert_eq!(glrmask_grammar_from_source(1,host.as_ptr().cast(),host.len(),ptr::null(),&mut grammar),0);
        assert_eq!(glrmask_grammar_from_source(0,leaf.as_ptr().cast(),leaf.len(),ptr::null(),&mut child),0);
        let (mut parent,mut clone,mut prepared)=(ptr::null_mut(),ptr::null_mut(),ptr::null_mut());
        assert_eq!(glrmask_module_new(grammar,ptr::null(),&mut parent),0);
        assert_eq!(glrmask_module_precompile(parent,vocab),0);
        assert_eq!(glrmask_module_clone(parent,&mut clone),0);
        let mut options=glrmask_default_module_options();options.optimization=2;
        assert_eq!(glrmask_module_new(child,&options,&mut prepared),0);
        assert_eq!(glrmask_module_precompile(prepared,vocab),0);
        options.optimization=3;
        let mut failed=ptr::dangling_mut();
        assert_ne!(glrmask_module_with_options(prepared,&options,&mut failed),0);assert!(failed.is_null());
        assert_ne!(glrmask_module_bind_grammar(parent,c"missing".as_ptr(),7,child),0);
        let mut slots=ptr::null_mut();assert_eq!(glrmask_module_unbound(parent,&mut slots),0);
        let mut count=0;assert_eq!(glrmask_requirements_len(slots,&mut count),0);assert_eq!(count,1);glrmask_requirements_free(slots);
        assert_eq!(glrmask_module_bind_module(parent,c"child".as_ptr(),5,prepared),0);
        assert_ne!(glrmask_module_bind_module(parent,c"child".as_ptr(),5,prepared),0);
        let mut artifact=ptr::null_mut();assert_eq!(glrmask_module_save(parent,&mut artifact),0);
        let (mut data,mut len)=(ptr::null(),0);assert_eq!(glrmask_bytes_data(artifact,&mut data,&mut len),0);
        let mut loaded=ptr::null_mut();assert_eq!(glrmask_module_load(data,len,vocab,&mut loaded),0);
        glrmask_bytes_free(artifact);glrmask_module_free(parent);glrmask_module_free(prepared);
        glrmask_grammar_free(grammar);glrmask_grammar_free(child);
        let mut constraint=ptr::null_mut();assert_eq!(glrmask_module_compile(loaded,vocab,ptr::null(),&mut constraint),0);
        let mut state=ptr::null_mut();assert_eq!(glrmask_constraint_start(constraint,&mut state),0);
        glrmask_constraint_free(constraint);glrmask_module_free(loaded);
        let mut words=[0u32;1];assert_eq!(glrmask_state_fill_mask(state,words.as_mut_ptr(),1),0);assert_ne!(words[0]&(1<<3),0);
        assert_eq!(glrmask_state_commit_token(state,3),0);let mut accepting=0;assert_eq!(glrmask_state_is_accepting(state,&mut accepting),0);assert_eq!(accepting,1);glrmask_state_free(state);
        // Self attachment snapshots the receiver before its mutable borrow.
        assert_eq!(glrmask_module_bind_module(clone,c"child".as_ptr(),5,clone),0);
        let mut slots=ptr::null_mut();assert_eq!(glrmask_module_unbound(clone,&mut slots),0);
        let mut r=glrmask_requirement{name:ptr::null(),name_len:0,kind:0,has_source_span:0,start_byte:0,end_byte:0,line:0,column:0};
        assert_eq!(glrmask_requirements_get(slots,0,&mut r),0);assert_eq!(text(r.name,r.name_len).unwrap(),"child::child");
        let mut depth=0;assert_eq!(glrmask_requirements_path_len(slots,0,&mut depth),0);assert_eq!(depth,2);
        glrmask_requirements_free(slots);glrmask_module_free(clone);
        // An expression owns its operands even after foreign handles are freed.
        let a=glrmask_grammar(g::Grammar::from_ebnf_owned("start ::= \"a\""));
        let values=[glrmask_expr_operand{kind:0,handle:(&a as *const glrmask_grammar).cast()}];
        let mut expr=ptr::null_mut();assert_eq!(glrmask_expr_new(0,values.as_ptr(),1,&mut expr),0);
        let mut m=ptr::null_mut();assert_eq!(glrmask_module_from_expr(expr,ptr::null(),&mut m),0);glrmask_expr_free(expr);
        assert!(read(m).unwrap().0.compile(&rust_vocab).unwrap().start().mask()[0]&(1<<1)!=0);
        glrmask_module_free(m);glrmask_vocab_free(vocab);
        let mut bad=glrmask_default_module_options();bad.struct_size=0;
        let mut out=ptr::dangling_mut();assert_ne!(glrmask_module_new(&a,&bad,&mut out),0);assert!(out.is_null());
    }
}
