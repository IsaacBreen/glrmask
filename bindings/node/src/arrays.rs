//! Synchronous views over checked, non-shared JavaScript typed arrays.
//!
//! No reference escapes these callbacks. We make no JavaScript calls while a
//! Rust slice exists: non-shared backing cannot be detached/resized concurrently.
//! SharedArrayBuffer is rejected rather than creating a data-racing Rust slice.
use napi::{Env, JsValue, Unknown, sys};
use glrmask_binding_core::{Error, Result};
use std::{ffi::c_void, mem::{align_of,size_of}, ptr, slice};

fn status(code:sys::napi_status) -> Result<()> {
    if code == sys::Status::napi_ok { Ok(()) }
    else { Err(Error::invalid("unable to inspect typed array")) }
}
fn info(env:&Env,value:&Unknown<'_>,expected:i32,width:usize,alignment:usize) -> Result<(*mut c_void,usize)> {
    let mut is_typed=false;
    status(unsafe {sys::napi_is_typedarray(env.raw(),value.raw(),&mut is_typed)})?;
    if !is_typed {return Err(Error::invalid("expected a typed array"));}
    let (mut kind,mut len,mut offset)=(0,0,0);
    let (mut data,mut backing)=(ptr::null_mut(),ptr::null_mut());
    status(unsafe {sys::napi_get_typedarray_info(env.raw(),value.raw(),&mut kind,&mut len,&mut data,&mut backing,&mut offset)})?;
    if kind!=expected {return Err(Error::invalid("incorrect typed array element type"));}
    let mut ordinary=false;
    status(unsafe {sys::napi_is_arraybuffer(env.raw(),backing,&mut ordinary)})?;
    if !ordinary {return Err(Error::invalid("SharedArrayBuffer-backed arrays are not supported"));}
    let mut detached=false;
    status(unsafe {sys::napi_is_detached_arraybuffer(env.raw(),backing,&mut detached)})?;
    if detached {return Err(Error::invalid("detached ArrayBuffer"));}
    if len>isize::MAX as usize/width {return Err(Error::invalid("typed array is too large"));}
    if len!=0 && (data.is_null() || data.addr()%alignment!=0) {return Err(Error::invalid("invalid typed array memory"));}
    Ok((data,len))
}
/// Copies input before any asynchronous work can be scheduled.
pub fn bytes(env:&Env,value:&Unknown<'_>) -> Result<Vec<u8>> {
    let (p,n)=info(env,value,sys::TypedarrayType::uint8_array,1,1)?;
    if n==0 {return Ok(Vec::new());}
    // SAFETY: inspected live non-shared backing, no JS callback until copied.
    Ok(unsafe {slice::from_raw_parts(p.cast::<u8>(),n)}.to_vec())
}
pub fn with_bytes<T>(env:&Env,value:&Unknown<'_>,f:impl FnOnce(&[u8])->Result<T>) -> Result<T> {
    let (p,n)=info(env,value,sys::TypedarrayType::uint8_array,1,1)?;
    if n==0 {f(&[])} else {f(unsafe {slice::from_raw_parts(p.cast::<u8>(),n)})}
}
pub fn with_words<T>(env:&Env,value:&Unknown<'_>,f:impl FnOnce(&mut [u32])->Result<T>) -> Result<T> {
    let (p,n)=info(env,value,sys::TypedarrayType::uint32_array,size_of::<u32>(),align_of::<u32>())?;
    if n==0 {f(&mut [])} else {f(unsafe {slice::from_raw_parts_mut(p.cast::<u32>(),n)})}
}
