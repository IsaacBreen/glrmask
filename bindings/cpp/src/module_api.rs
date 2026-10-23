//! Canonical owned assembly ABI. Every foreign input is snapshotted before mutation.
use super::*;
use std::ffi::c_void;

pub struct glrmask_module(pub(super) g::Module);
pub struct glrmask_expr(pub(super) g::Expr);
#[repr(C)] pub struct glrmask_compile_options {
    pub struct_size: usize,
    pub end_tokens: *const u32,
    pub end_tokens_len: usize,
    pub mask_strategy: u32,
    pub boundary_mask_strategy: u32,
}
/// kind: 0 Grammar, 1 Module, 2 Expr, 3 Token, 4 ExactTokens. Inputs are borrowed for this call only.
#[repr(C)] pub struct glrmask_expr_operand { pub kind: u32, pub handle: *const c_void }

pub(super) unsafe fn root_options(p: *const glrmask_compile_options) -> Result<g::CompileOptions> {
    if p.is_null() { return Ok(g::CompileOptions::default()); }
    let o = unsafe { read(p)? };
    if o.struct_size != size_of::<glrmask_compile_options>() { return Err(Error::invalid("compile-options struct_size does not match ABI")); }
    Ok(core::compile_options(
        unsafe { input(o.end_tokens, o.end_tokens_len)? }.to_vec(),
        core::mask_strategy(o.mask_strategy)?,
        core::boundary_mask_strategy(o.boundary_mask_strategy)?,
    ))
}

#[unsafe(no_mangle)] pub extern "C" fn glrmask_default_compile_options() -> glrmask_compile_options {
    glrmask_compile_options {
        struct_size: size_of::<glrmask_compile_options>(),
        end_tokens: ptr::null(),
        end_tokens_len: 0,
        mask_strategy: 0,
        boundary_mask_strategy: 0,
    }
}

free!(glrmask_module_free, glrmask_module);
free!(glrmask_expr_free, glrmask_expr);
clone_handle!(glrmask_module_clone, glrmask_module);
clone_handle!(glrmask_expr_clone, glrmask_expr);

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_new(p: *const glrmask_grammar, out: *mut *mut glrmask_module) -> i32 {
    ffi(|| unsafe { produce(out, || Ok(glrmask_module(g::Module::new(&read(p)?.0)))) })
}

macro_rules! bind_handle {($name:ident,$child:ty,$wrap:expr)=>{
    #[unsafe(no_mangle)] pub unsafe extern "C" fn $name(p:*mut glrmask_module,name:*const c_char,n:usize,child:*const $child)->i32 {
        ffi(||unsafe{let child=read(child)?.0.clone();let name=text(name,n)?;write(p)?.0.bind(name,$wrap(child))?;Ok(())})
    }
};}
bind_handle!(glrmask_module_bind_grammar, glrmask_grammar, |g| g::Binding::from(&g));
bind_handle!(glrmask_module_bind_module, glrmask_module, |m| g::Binding::from(m));
bind_handle!(glrmask_module_bind_expr, glrmask_expr, |e| g::Binding::from(e));
bind_handle!(glrmask_module_bind_tokens, glrmask_exact_tokens, |t| g::Binding::from(t));

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_bind_token(p: *mut glrmask_module, name: *const c_char, n: usize, id: u32) -> i32 {
    ffi(|| unsafe { let name = text(name, n)?; write(p)?.0.bind(name, g::Token::new(id))?; Ok(()) })
}

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_precompile(p: *mut glrmask_module, v: *const glrmask_vocab, o: *const glrmask_compile_options) -> i32 {
    ffi(|| unsafe {
        let vocab = read(v)?.0.clone();
        let options = root_options(o)?;
        write(p)?.0.precompile_with(&vocab, &options)?;
        Ok(())
    })
}

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_compile(p: *const glrmask_module, v: *const glrmask_vocab, o: *const glrmask_compile_options, out: *mut *mut glrmask_constraint) -> i32 {
    ffi(|| unsafe {
        let m = &read(p)?.0;
        let vocab = &read(v)?.0;
        let opt = root_options(o)?;
        let compiled = m.compile_with(vocab, opt)?;
        produce(out, || Ok(glrmask_constraint(Arc::new(compiled))))
    })
}

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_externs(p: *const glrmask_module, out: *mut *mut glrmask_externs) -> i32 {
    ffi(|| unsafe {
        let view = read(p)?.0.externs();
        produce(out, || Ok(glrmask_externs(view.iter().map(|e| glrmask_extern_item {
            name: e.name().to_string(),
            kind: e.kind(),
            span: e.span(),
        }).collect())))
    })
}

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_unbound_externs(p: *const glrmask_module, out: *mut *mut glrmask_externs) -> i32 {
    ffi(|| unsafe {
        let view = read(p)?.0.externs().unbound();
        produce(out, || Ok(glrmask_externs(view.iter().map(|e| glrmask_extern_item {
            name: e.name().to_string(),
            kind: e.kind(),
            span: e.span(),
        }).collect())))
    })
}

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_save(p: *const glrmask_module, out: *mut *mut glrmask_bytes) -> i32 {
    ffi(|| unsafe { produce(out, || Ok(glrmask_bytes(read(p)?.0.save()))) })
}

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_load(bytes: *const u8, n: usize, v: *const glrmask_vocab, out: *mut *mut glrmask_module) -> i32 {
    ffi(|| unsafe { produce(out, || Ok(glrmask_module(g::Module::load(input(bytes, n)?, if v.is_null() { None } else { Some(&read(v)?.0) })?))) })
}

unsafe fn parse_expr_operand(item: &glrmask_expr_operand) -> Result<g::Expr> {
    unsafe {
        match item.kind {
            0 => Ok(g::Expr::from(read(item.handle.cast::<glrmask_grammar>())?.0.clone())),
            1 => Ok(g::Expr::from(read(item.handle.cast::<glrmask_module>())?.0.clone())),
            2 => Ok(read(item.handle.cast::<glrmask_expr>())?.0.clone()),
            3 => Ok(g::Expr::from(g::Token::new(item.handle as usize as u32))),
            4 => Ok(g::Expr::from(read(item.handle.cast::<glrmask_exact_tokens>())?.0.clone())),
            _ => Err(Error::invalid("unknown expression operand; expected Grammar=0, Module=1, Expr=2, Token=3, ExactTokens=4")),
        }
    }
}

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_expr_choice(items: *const glrmask_expr_operand, n: usize, out: *mut *mut glrmask_expr) -> i32 {
    ffi(|| unsafe { produce(out, || {
        let parts = input(items, n)?.iter().map(|item| parse_expr_operand(item)).collect::<Result<Vec<_>>>()?;
        Ok(glrmask_expr(g::Expr::choice(parts)))
    }) })
}

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_expr_sequence(items: *const glrmask_expr_operand, n: usize, out: *mut *mut glrmask_expr) -> i32 {
    ffi(|| unsafe { produce(out, || {
        let parts = input(items, n)?.iter().map(|item| parse_expr_operand(item)).collect::<Result<Vec<_>>>()?;
        Ok(glrmask_expr(g::Expr::sequence(parts)))
    }) })
}

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_expr_literal(bytes: *const u8, n: usize, out: *mut *mut glrmask_expr) -> i32 {
    ffi(|| unsafe { produce(out, || Ok(glrmask_expr(g::Expr::literal(input(bytes, n)?)))) })
}
