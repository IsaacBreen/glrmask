//! Canonical owned assembly ABI. Every foreign input is snapshotted before mutation.
use super::*;
use g::IntoExpr as _;
use std::ffi::c_void;

pub struct glrmask_module(pub(super) g::Module);
pub struct glrmask_expr(g::Expr);
#[repr(C)] pub struct glrmask_module_options {pub struct_size:usize,pub optimization:u32,pub boundary_trigger:u32}
#[repr(C)] pub struct glrmask_compile_options {pub struct_size:usize,pub end_tokens:*const u32,pub end_tokens_len:usize}
/// kind: 0 Grammar, 1 Module, 2 Expr. Inputs are borrowed for this call only.
#[repr(C)] pub struct glrmask_expr_operand {pub kind:u32,pub handle:*const c_void}

unsafe fn local_options(p:*const glrmask_module_options)->Result<g::ModuleOptions> {
    if p.is_null(){return Ok(g::ModuleOptions::default());}
    let o=unsafe{read(p)?};
    if o.struct_size!=size_of::<glrmask_module_options>(){return Err(Error::invalid("module-options struct_size does not match ABI"));}
    core::module_options(o.optimization,o.boundary_trigger)
}
unsafe fn root_options(p:*const glrmask_compile_options)->Result<g::CompileOptions> {
    if p.is_null(){return Ok(g::CompileOptions::default());}
    let o=unsafe{read(p)?};
    if o.struct_size!=size_of::<glrmask_compile_options>(){return Err(Error::invalid("compile-options struct_size does not match ABI"));}
    Ok(g::CompileOptions::default().end_tokens(unsafe{input(o.end_tokens,o.end_tokens_len)?}.iter().copied()))
}
#[unsafe(no_mangle)] pub extern "C" fn glrmask_default_module_options()->glrmask_module_options {
    glrmask_module_options{struct_size:size_of::<glrmask_module_options>(),optimization:0,boundary_trigger:0}
}
#[unsafe(no_mangle)] pub extern "C" fn glrmask_default_compile_options()->glrmask_compile_options {
    glrmask_compile_options{struct_size:size_of::<glrmask_compile_options>(),end_tokens:ptr::null(),end_tokens_len:0}
}
free!(glrmask_module_free,glrmask_module);free!(glrmask_expr_free,glrmask_expr);
clone_handle!(glrmask_module_clone,glrmask_module);clone_handle!(glrmask_expr_clone,glrmask_expr);
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_new(p:*const glrmask_grammar,o:*const glrmask_module_options,out:*mut *mut glrmask_module)->i32 {
    ffi(||unsafe{produce(out,||Ok(glrmask_module(read(p)?.0.module().with_options(local_options(o)?)?)))})
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_from_expr(p:*const glrmask_expr,o:*const glrmask_module_options,out:*mut *mut glrmask_module)->i32 {
    ffi(||unsafe{produce(out,||Ok(glrmask_module(g::Module::from_expr(&read(p)?.0)?.with_options(local_options(o)?)?)))})
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_grammar_from_expr(p:*const glrmask_expr,out:*mut *mut glrmask_grammar)->i32 {
    ffi(||unsafe{produce(out,||Ok(glrmask_grammar(g::Grammar::from_expr(&read(p)?.0)?)))})
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_expr_new(kind:u32,items:*const glrmask_expr_operand,n:usize,out:*mut *mut glrmask_expr)->i32 {
    ffi(||unsafe{produce(out,||{
        let parts=input(items,n)?.iter().map(|item|Ok(match item.kind {
            0=>read(item.handle.cast::<glrmask_grammar>())?.0.clone().into_expr(),
            1=>read(item.handle.cast::<glrmask_module>())?.0.clone().into_expr(),
            2=>read(item.handle.cast::<glrmask_expr>())?.0.clone(),
            _=>return Err(Error::invalid("unknown expression operand; expected Grammar, Module or Expr")),
        })).collect::<Result<Vec<_>>>()?;
        Ok(glrmask_expr(match kind {0=>g::Expr::choice(parts),1=>g::Expr::sequence(parts),_=>return Err(Error::invalid("unknown expression operator; expected choice=0 or sequence=1"))}))
    })})
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_get_options(p:*const glrmask_module,out:*mut glrmask_module_options)->i32 {
    ffi(||unsafe{let (optimization,boundary_trigger)=core::module_option_values(read(p)?.0.options())?;*write(out)?=glrmask_module_options{struct_size:size_of::<glrmask_module_options>(),optimization,boundary_trigger};Ok(())})
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_with_options(p:*const glrmask_module,o:*const glrmask_module_options,out:*mut *mut glrmask_module)->i32 {
    ffi(||unsafe{produce(out,||Ok(glrmask_module(read(p)?.0.with_options(local_options(o)?)?)))})
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_set_options(p:*mut glrmask_module,o:*const glrmask_module_options)->i32 {
    ffi(||unsafe{let o=local_options(o)?;write(p)?.0.set_options(o)?;Ok(())})
}
macro_rules! bind_handle {($name:ident,$child:ty,$value:expr)=>{
    #[unsafe(no_mangle)] pub unsafe extern "C" fn $name(p:*mut glrmask_module,name:*const c_char,n:usize,child:*const $child)->i32 {
        ffi(||unsafe{let child=read(child)?.0.clone();let name=text(name,n)?.to_owned();write(p)?.0.bind(name,($value)(child))?;Ok(())})
    }
};}
bind_handle!(glrmask_module_bind_grammar,glrmask_grammar,|v|v);
bind_handle!(glrmask_module_bind_module,glrmask_module,|v|v);
bind_handle!(glrmask_module_bind_expr,glrmask_expr,|v|v);
bind_handle!(glrmask_module_bind_tokens,glrmask_exact_tokens,|v|v);
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_bind_token(p:*mut glrmask_module,name:*const c_char,n:usize,id:u32)->i32 {
    ffi(||unsafe{let name=text(name,n)?.to_owned();write(p)?.0.bind(name,g::Token::new(id))?;Ok(())})
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_grammar_bind_expr(p:*mut glrmask_grammar,name:*const c_char,n:usize,expr:*const glrmask_expr)->i32 {
    ffi(||unsafe{let child=g::Grammar::from_expr(&read(expr)?.0)?;let name=text(name,n)?.to_owned();write(p)?.0.bind(name,child)?;Ok(())})
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_precompile(p:*mut glrmask_module,v:*const glrmask_vocab)->i32 {
    ffi(||unsafe{let vocab=read(v)?.0.clone();write(p)?.0.precompile(&vocab)?;Ok(())})
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_compile(p:*const glrmask_module,v:*const glrmask_vocab,o:*const glrmask_compile_options,out:*mut *mut glrmask_constraint)->i32 {
    ffi(||unsafe{produce(out,||Ok(glrmask_constraint(Arc::new(read(p)?.0.compile_with(&read(v)?.0,root_options(o)?)?))))})
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_unbound(p:*const glrmask_module,out:*mut *mut glrmask_requirements)->i32 {ffi(||unsafe{produce(out,||Ok(requirements(read(p)?.0.unbound()?)))})}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_externs(p:*const glrmask_module,out:*mut *mut glrmask_requirements)->i32 {ffi(||unsafe{produce(out,||Ok(requirements(read(p)?.0.externs()?)))})}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_grammar_unbound(p:*const glrmask_grammar,out:*mut *mut glrmask_requirements)->i32 {ffi(||unsafe{produce(out,||Ok(requirements(read(p)?.0.unbound()?)))})}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_grammar_externs(p:*const glrmask_grammar,out:*mut *mut glrmask_requirements)->i32 {ffi(||unsafe{produce(out,||Ok(requirements(read(p)?.0.externs()?)))})}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_save(p:*const glrmask_module,out:*mut *mut glrmask_bytes)->i32 {ffi(||unsafe{produce(out,||Ok(glrmask_bytes(read(p)?.0.save())))})}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_module_load(bytes:*const u8,n:usize,v:*const glrmask_vocab,out:*mut *mut glrmask_module)->i32 {
    ffi(||unsafe{produce(out,||Ok(glrmask_module(if v.is_null(){g::Module::load(input(bytes,n)?)?}else{g::Module::load_with_vocab(input(bytes,n)?,&read(v)?.0)?})))})
}
/// Return exact path segments instead of reconstructing them by splitting names.
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_requirements_path(p:*const glrmask_requirements,index:usize,segment:usize,out:*mut *const c_char,n:*mut usize)->i32 {
    ffi(||unsafe{let (_,r)=read(p)?.0.get(index).ok_or_else(||Error::invalid("extern index out of range"))?;
        let path=r.path.get(segment).ok_or_else(||Error::invalid("extern path segment out of range"))?;
        *write(out)?=path.as_ptr().cast();*write(n)?=path.len();Ok(())})
}
#[unsafe(no_mangle)] pub unsafe extern "C" fn glrmask_requirements_path_len(p:*const glrmask_requirements,index:usize,n:*mut usize)->i32 {
    ffi(||unsafe{*write(n)?=read(p)?.0.get(index).ok_or_else(||Error::invalid("extern index out of range"))?.1.path.len();Ok(())})
}
