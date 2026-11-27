//! Canonical owned assembly ABI. Every foreign input is snapshotted before mutation.
use super::*;
use std::ffi::c_void;

pub struct glrmask_module(pub(super) g::Module);
pub struct glrmask_expr(pub(super) g::Expr);
#[repr(C)] pub struct glrmask_compile_options {
    pub struct_size: usize,
    pub mask_strategy: u32,
    pub reserved: [u32; 3],
}
/// kind: 0 Grammar, 1 Module, 2 Expr, 4 TokenSet. Inputs are borrowed for this call only.
#[repr(C)] pub struct glrmask_expr_operand { pub kind: u32, pub handle: *const c_void }

pub(super) unsafe fn root_options(p: *const glrmask_compile_options) -> Result<g::CompileOptions> {
    if p.is_null() { return Ok(g::CompileOptions::default()); }
    // The mask preference is followed by reserved fields. Shorter structs
    // retain zero defaults for missing fields.
    let minimum = size_of::<usize>() + 2 * size_of::<u32>();
    valid(p)?;
    let struct_size = unsafe { p.cast::<usize>().read() };
    if struct_size < minimum || struct_size > isize::MAX as usize {
        return Err(Error::invalid("compile-options struct_size does not match ABI"));
    }
    let mut o: glrmask_compile_options = unsafe { std::mem::zeroed() };
    // Only whole fields are read, and unknown trailing fields are ignored.
    let known_size = struct_size.min(size_of::<glrmask_compile_options>());
    let complete_size = minimum + (known_size - minimum) / size_of::<u32>() * size_of::<u32>();
    unsafe { std::ptr::copy_nonoverlapping(p.cast::<u8>(), (&mut o as *mut glrmask_compile_options).cast::<u8>(), complete_size) };
    o.struct_size = struct_size;
    if o.reserved != [0, 0, 0] {
        return Err(Error::invalid("reserved compile-options fields must be zero"));
    }
    Ok(core::compile_options(
        core::mask_strategy(o.mask_strategy)?,
    ))
}

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_compile_options_init(opts: *mut glrmask_compile_options, struct_size: usize) -> i32 {
    ffi(|| unsafe {
        valid(opts)?;
        let minimum = size_of::<usize>() + 2 * size_of::<u32>();
        if struct_size < minimum || struct_size > isize::MAX as usize {
            return Err(Error::invalid("compile-options struct_size does not match ABI"));
        }
        // The caller owns exactly struct_size writable bytes, possibly fewer
        // than the current struct. Never form a reference to the larger type.
        ptr::write_bytes(opts.cast::<u8>(), 0, struct_size);
        opts.cast::<usize>().write(struct_size);
        Ok(())
    })
}

free!(glrmask_module_free, glrmask_module);
free!(glrmask_expr_free, glrmask_expr);
clone_handle!(glrmask_module_clone, glrmask_module);
clone_handle!(glrmask_expr_clone, glrmask_expr);

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_new(p: *const glrmask_grammar, out: *mut *mut glrmask_module) -> i32 {
    ffi(|| unsafe { produce(out, || Ok(glrmask_module(g::Module::new(&read(p)?.0)))) })
}

macro_rules! bind_handle {($name:ident,$child:ty)=>{
    #[unsafe(no_mangle)] pub unsafe extern "C" fn $name(p:*mut glrmask_module,name:*const c_char,n:usize,child:*const $child)->i32 {
        ffi(||unsafe{let child=read(child)?.0.clone();let name=text(name,n)?;write(p)?.0.bind(name,child)?;Ok(())})
    }
};}
bind_handle!(glrmask_module_bind_grammar, glrmask_grammar);
bind_handle!(glrmask_module_bind_module, glrmask_module);
bind_handle!(glrmask_module_bind_expr, glrmask_expr);
bind_handle!(glrmask_module_bind_tokens, glrmask_token_set);


#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_precompile(p: *mut glrmask_module, v: *const glrmask_vocab, o: *const glrmask_compile_options) -> i32 {
    ffi(|| unsafe {
        let vocab = read(v)?.0.clone();
        let options = root_options(o)?;
        write(p)?.0.precompile_with(&vocab, options)?;
        Ok(())
    })
}

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_compile(p: *const glrmask_module, v: *const glrmask_vocab, o: *const glrmask_compile_options, out: *mut *mut glrmask_constraint) -> i32 {
    ffi(|| unsafe { produce(out, || {
        let m = &read(p)?.0;
        let vocab = &read(v)?.0;
        let opt = root_options(o)?;
        Ok(glrmask_constraint(Arc::new(m.compile_with(vocab, opt)?)))
    }) })
}

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_externs(p: *const glrmask_module, out: *mut *mut glrmask_externs) -> i32 {
    ffi(|| unsafe { produce(out, || {
        let view = read(p)?.0.externs();
        Ok(glrmask_externs(view.iter().map(|e| glrmask_extern_item {
            name: e.name().to_string(),
            kind: e.kind(),
        }).collect()))
    }) })
}

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_unbound_externs(p: *const glrmask_module, out: *mut *mut glrmask_externs) -> i32 {
    ffi(|| unsafe { produce(out, || {
        let view = read(p)?.0.externs().unbound();
        Ok(glrmask_externs(view.iter().map(|e| glrmask_extern_item {
            name: e.name().to_string(),
            kind: e.kind(),
        }).collect()))
    }) })
}

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_save(p: *const glrmask_module, out: *mut *mut glrmask_bytes) -> i32 {
    ffi(|| unsafe { produce(out, || Ok(glrmask_bytes(read(p)?.0.save()))) })
}

#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_load(bytes: *const u8, n: usize, out: *mut *mut glrmask_module) -> i32 {
    ffi(|| unsafe { produce(out, || Ok(glrmask_module(g::Module::load(input(bytes, n)?)?))) })
}

unsafe fn parse_expr_operand(item: &glrmask_expr_operand) -> Result<g::Expr> {
    unsafe {
        match item.kind {
            0 => Ok(g::Expr::from(read(item.handle.cast::<glrmask_grammar>())?.0.clone())),
            1 => Ok(g::Expr::from(read(item.handle.cast::<glrmask_module>())?.0.clone())),
            2 => Ok(read(item.handle.cast::<glrmask_expr>())?.0.clone()),
            4 => Ok(g::Expr::from(read(item.handle.cast::<glrmask_token_set>())?.0.clone())),
            _ => Err(Error::invalid("unknown operand kind; expected a GLRMASK_OPERAND_* value")),
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
