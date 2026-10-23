#![recursion_limit = "256"]

//! C ABI for GLRMask. See include/glrmask.h for the ownership contract.
#![allow(non_camel_case_types)]
// The entire C entry surface is unsafe: validating arbitrary foreign memory is
// impossible. Null/alignment/length checks supplement, not replace, its contract.
#![allow(clippy::missing_safety_doc)]
use glrmask_binding_core::{self as core, glrmask as g, Error, ErrorKind, Result};
use std::{cell::RefCell, ffi::{CString, c_char}, mem::{align_of, size_of}, ptr, slice, sync::Arc};

pub struct glrmask_vocab(g::Vocab);
pub struct glrmask_exact_tokens(g::ExactTokens);
pub struct glrmask_grammar(g::Grammar);
pub struct glrmask_constraint(Arc<g::Constraint>);
pub struct glrmask_state(core::State);
pub struct glrmask_bytes(Vec<u8>);
pub struct glrmask_u32s(Vec<u32>);
pub struct glrmask_events(Vec<glrmask_marker_event>);

pub struct glrmask_extern_item {
    pub name: String,
    pub kind: g::ExternKind,
}
pub struct glrmask_externs(Vec<glrmask_extern_item>);

#[repr(C)] pub struct glrmask_vocab_entry { pub id: u32, pub bytes: *const u8, pub bytes_len: usize }
#[repr(C)] #[derive(Clone, Copy)] pub struct glrmask_token_range { pub first: u32, pub last: u32 }
#[repr(C)] pub struct glrmask_extern {
    pub name: *const c_char,
    pub name_len: usize,
    pub kind: u32,
}
#[repr(C)] pub struct glrmask_marker_event { pub id: u32, pub byte_offset: u64 }

thread_local! { static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) }; }
fn ffi(f: impl FnOnce() -> Result<()>) -> i32 {
    match core::guarded(f) {
        Ok(()) => { LAST_ERROR.with(|s| { s.borrow_mut().take(); }); 0 }
        Err(e) => {
            LAST_ERROR.with(|s| *s.borrow_mut() = Some(CString::new(e.message.replace('\0', "\\0")).unwrap_or_default()));
            e.kind as i32
        }
    }
}
fn valid<T>(p: *const T) -> Result<()> {
    if p.is_null() { return Err(Error::invalid("null pointer")); }
    if p.addr() % align_of::<T>() != 0 { return Err(Error::invalid("misaligned pointer")); }
    Ok(())
}
unsafe fn read<'a, T>(p: *const T) -> Result<&'a T> { valid(p)?; Ok(unsafe { &*p }) }
unsafe fn write<'a, T>(p: *mut T) -> Result<&'a mut T> { valid(p)?; Ok(unsafe { &mut *p }) }
unsafe fn input<'a, T>(p: *const T, n: usize) -> Result<&'a [T]> {
    if n == 0 { return Ok(&[]); }
    valid(p)?;
    if n > isize::MAX as usize / size_of::<T>() { return Err(Error::invalid("array length overflows address space")); }
    Ok(unsafe { slice::from_raw_parts(p, n) })
}
unsafe fn output<'a, T>(p: *mut T, n: usize) -> Result<&'a mut [T]> {
    if n == 0 { return Ok(&mut []); }
    valid(p)?;
    if n > isize::MAX as usize / size_of::<T>() { return Err(Error::invalid("array length overflows address space")); }
    Ok(unsafe { slice::from_raw_parts_mut(p, n) })
}
unsafe fn text<'a>(p: *const c_char, n: usize) -> Result<&'a str> {
    std::str::from_utf8(unsafe { input(p.cast(), n)? }).map_err(|_| Error::invalid("text must be valid UTF-8"))
}
unsafe fn produce<T>(out: *mut *mut T, f: impl FnOnce() -> Result<T>) -> Result<()> {
    let out = unsafe { write(out)? }; *out = ptr::null_mut();
    *out = Box::into_raw(Box::new(f()?)); Ok(())
}

#[unsafe(no_mangle)] pub extern "C" fn glrmask_abi_version() -> u32 { 1 }
#[unsafe(no_mangle)] pub extern "C" fn glrmask_version() -> *const c_char { concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast() }
#[unsafe(no_mangle)] pub extern "C" fn glrmask_error_message() -> *const c_char { LAST_ERROR.with(|s| s.borrow().as_ref().map_or(c"".as_ptr(), |s| s.as_ptr())) }

macro_rules! free {($name:ident,$ty:ty)=>{
    #[unsafe(no_mangle)] pub unsafe extern "C" fn $name(p:*mut $ty) {
        let _=ffi(||{if !p.is_null(){valid(p)?;unsafe{drop(Box::from_raw(p));}} Ok(())});
    }
};}
macro_rules! clone_handle {($name:ident,$ty:ident)=>{
    #[unsafe(no_mangle)] pub unsafe extern "C" fn $name(p:*const $ty,out:*mut *mut $ty)->i32 {
        ffi(||unsafe{produce(out,||Ok($ty(read(p)?.0.clone())))})
    }
};}

mod module_api;
pub use module_api::*;

free!(glrmask_vocab_free, glrmask_vocab);
free!(glrmask_exact_tokens_free, glrmask_exact_tokens);
free!(glrmask_grammar_free, glrmask_grammar);
free!(glrmask_constraint_free, glrmask_constraint);
free!(glrmask_state_free, glrmask_state);
free!(glrmask_bytes_free, glrmask_bytes);
free!(glrmask_u32s_free, glrmask_u32s);
free!(glrmask_externs_free, glrmask_externs);
free!(glrmask_events_free, glrmask_events);

clone_handle!(glrmask_vocab_clone, glrmask_vocab);
clone_handle!(glrmask_exact_tokens_clone, glrmask_exact_tokens);
clone_handle!(glrmask_grammar_clone, glrmask_grammar);
clone_handle!(glrmask_constraint_clone, glrmask_constraint);

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_vocab_new(entries: *const glrmask_vocab_entry, n: usize, exact: *const u32, exact_n: usize, out: *mut *mut glrmask_vocab) -> i32 {
    ffi(|| unsafe { produce(out, || {
        let entries = input(entries, n)?.iter().map(|e| Ok((e.id, input(e.bytes, e.bytes_len)?.to_vec()))).collect::<Result<Vec<_>>>()?;
        Ok(glrmask_vocab(core::vocab(entries, input(exact, exact_n)?.to_vec())?))
    }) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_vocab_len(p: *const glrmask_vocab, out: *mut usize) -> i32 {
    ffi(|| unsafe { *write(out)? = read(p)?.0.len(); Ok(()) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_vocab_tokens(p: *const glrmask_vocab, ids: *const u32, n: usize, out: *mut *mut glrmask_exact_tokens) -> i32 {
    ffi(|| unsafe { produce(out, || Ok(glrmask_exact_tokens(read(p)?.0.tokens(input(ids, n)?.iter().copied()).map_err(Error::engine)?))) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_vocab_token_ranges(p: *const glrmask_vocab, r: *const glrmask_token_range, n: usize, out: *mut *mut glrmask_exact_tokens) -> i32 {
    ffi(|| unsafe { produce(out, || Ok(glrmask_exact_tokens(read(p)?.0.token_ranges(input(r, n)?.iter().map(|r| r.first..=r.last)).map_err(Error::engine)?))) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_exact_tokens_len(p: *const glrmask_exact_tokens, out: *mut usize) -> i32 {
    ffi(|| unsafe { *write(out)? = read(p)?.0.len(); Ok(()) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_exact_tokens_ids(p: *const glrmask_exact_tokens, out: *mut *mut glrmask_u32s) -> i32 {
    ffi(|| unsafe { produce(out, || Ok(glrmask_u32s(read(p)?.0.ids().to_vec()))) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_exact_tokens_ranges(p: *const glrmask_exact_tokens, out: *mut glrmask_token_range, capacity: usize, n: *mut usize) -> i32 {
    ffi(|| unsafe {
        let tokens = &read(p)?.0; let needed = tokens.ranges().len(); *write(n)? = needed;
        if out.is_null() && capacity == 0 { return Ok(()); }
        if capacity < needed { return Err(Error::new(ErrorKind::BufferTooSmall, "range output is too small")); }
        for (dest, r) in output(out, capacity)?.iter_mut().zip(tokens.ranges()) { *dest = glrmask_token_range { first: *r.start(), last: *r.end() }; }
        Ok(())
    })
}

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_grammar_from_source(format: u32, source: *const c_char, n: usize, out: *mut *mut glrmask_grammar) -> i32 {
    ffi(|| unsafe { produce(out, || {
        let source = text(source, n)?.to_owned();
        let grammar = match format {
            0 => g::Grammar::from_ebnf(source),
            1 => g::Grammar::from_glrm(source),
            2 => g::Grammar::from_json_schema(source),
            3 => g::Grammar::from_lark(source),
            _ => return Err(Error::invalid("unknown source format")),
        }; Ok(glrmask_grammar(grammar))
    }) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_grammar_externs(p: *const glrmask_grammar, out: *mut *mut glrmask_externs) -> i32 {
    ffi(|| unsafe {
        let view = read(p)?.0.externs();
        produce(out, || Ok(glrmask_externs(view.iter().map(|e| glrmask_extern_item {
            name: e.name().to_string(),
            kind: e.kind(),
        }).collect())))
    })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_grammar_compile(p: *const glrmask_grammar, v: *const glrmask_vocab, o: *const glrmask_compile_options, out: *mut *mut glrmask_constraint) -> i32 {
    ffi(|| unsafe { produce(out, || {
        let grammar = &read(p)?.0;
        let vocab = &read(v)?.0;
        let options = root_options(o)?;
        Ok(glrmask_constraint(Arc::new(grammar.compile_with(vocab, options)?)))
    }) })
}

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_constraint_start(p: *const glrmask_constraint, out: *mut *mut glrmask_state) -> i32 {
    ffi(|| unsafe { produce(out, || Ok(glrmask_state(core::State::start(read(p)?.0.clone())?))) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_constraint_mask_len(p: *const glrmask_constraint, out: *mut usize) -> i32 {
    ffi(|| unsafe { *write(out)? = read(p)?.0.mask_len(); Ok(()) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_constraint_marker_ids(p: *const glrmask_constraint, out: *mut *mut glrmask_u32s) -> i32 {
    ffi(|| unsafe { produce(out, || Ok(glrmask_u32s(read(p)?.0.marker_ids().to_vec()))) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_constraint_save(p: *const glrmask_constraint, external: u32, out: *mut *mut glrmask_bytes) -> i32 {
    ffi(|| unsafe { produce(out, || {
        let c = &read(p)?.0;
        Ok(glrmask_bytes(match external { 0 => c.save(), 1 => c.save_without_vocab()?, _ => return Err(Error::invalid("external_vocab must be 0 or 1")) }))
    }) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_constraint_load(bytes: *const u8, n: usize, v: *const glrmask_vocab, out: *mut *mut glrmask_constraint) -> i32 {
    ffi(|| unsafe { produce(out, || {
        let bytes = input(bytes, n)?.to_vec();
        Ok(glrmask_constraint(Arc::new(if v.is_null() { g::Constraint::load(bytes)? } else { g::Constraint::load_with_vocab(bytes, &read(v)?.0)? })))
    }) })
}

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_state_clone(p: *const glrmask_state, out: *mut *mut glrmask_state) -> i32 {
    ffi(|| unsafe { produce(out, || Ok(glrmask_state(read(p)?.0.try_clone()?))) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_state_mask_len(p: *const glrmask_state, out: *mut usize) -> i32 {
    ffi(|| unsafe { *write(out)? = read(p)?.0.mask_len(); Ok(()) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_state_fill_mask(p: *const glrmask_state, out: *mut u32, n: usize) -> i32 {
    ffi(|| unsafe { read(p)?.0.fill_mask(output(out, n)?) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_state_commit_token(p: *mut glrmask_state, id: u32) -> i32 {
    ffi(|| unsafe { write(p)?.0.commit_token(id) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_state_commit_bytes(p: *mut glrmask_state, bytes: *const u8, n: usize) -> i32 {
    ffi(|| unsafe { write(p)?.0.commit_bytes(input(bytes, n)?) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_state_is_accepting(p: *const glrmask_state, out: *mut u32) -> i32 {
    ffi(|| unsafe { *write(out)? = u32::from(read(p)?.0.is_accepting()?); Ok(()) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_state_is_rejected(p: *const glrmask_state, out: *mut u32) -> i32 {
    ffi(|| unsafe { *write(out)? = u32::from(read(p)?.0.is_rejected()?); Ok(()) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_state_forced_tokens(p: *const glrmask_state, out: *mut *mut glrmask_u32s) -> i32 {
    ffi(|| unsafe { produce(out, || Ok(glrmask_u32s(read(p)?.0.forced_tokens()?))) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_state_marker_events(p: *const glrmask_state, out: *mut *mut glrmask_events) -> i32 {
    ffi(|| unsafe { produce(out, || Ok(glrmask_events(read(p)?.0.marker_events()?.into_iter().map(|e| glrmask_marker_event { id: e.id, byte_offset: e.byte_offset }).collect()))) })
}

macro_rules! data {($fn:ident,$handle:ident,$ty:ty)=>{
    #[unsafe(no_mangle)] pub unsafe extern "C" fn $fn(p:*const $handle,out:*mut *const $ty,n:*mut usize)->i32 {
        ffi(||unsafe{let out=write(out)?;let n=write(n)?;let data=&read(p)?.0;*out=data.as_ptr();*n=data.len();Ok(())})
    }
};}
data!(glrmask_bytes_data, glrmask_bytes, u8);
data!(glrmask_u32s_data, glrmask_u32s, u32);
data!(glrmask_events_data, glrmask_events, glrmask_marker_event);

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_externs_len(p: *const glrmask_externs, out: *mut usize) -> i32 {
    ffi(|| unsafe { *write(out)? = read(p)?.0.len(); Ok(()) })
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_externs_get(p: *const glrmask_externs, index: usize, out: *mut glrmask_extern) -> i32 {
    ffi(|| unsafe {
        let item = read(p)?.0.get(index).ok_or_else(|| Error::invalid("extern index out of range"))?;
        *write(out)? = glrmask_extern {
            name: item.name.as_ptr().cast(),
            name_len: item.name.len(),
            kind: match item.kind {
                g::ExternKind::Token => 0,
                g::ExternKind::Grammar => 1,
                _ => 1,
            },
        };
        Ok(())
    })
}

#[cfg(test)]
mod tests;
