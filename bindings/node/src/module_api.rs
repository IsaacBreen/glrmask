//! Canonical owned Module/Expr surface. No JS callback runs under a native lock.
use super::*;
use g::IntoExpr as _;

#[napi(object)]
pub struct NativeModuleOptions {pub optimization:u32,pub boundary_trigger:u32}
#[napi]
pub struct NativeExpr {pub(super) inner:Handle<g::Expr>}
#[napi]
impl NativeExpr {
    #[napi(factory)]
    pub fn from_grammar(value:&NativeGrammar)->napi::Result<Self> {
        call(||Ok(Self{inner:Handle::new(live(&value.inner)?.clone().into_expr())}))
    }
    #[napi(factory)]
    pub fn from_module(value:&NativeModule)->napi::Result<Self> {
        call(||Ok(Self{inner:Handle::new(value.snapshot()?.into_expr())}))
    }
    #[napi(factory)]
    pub fn choice(values:Vec<ClassInstance<'_,NativeExpr>>)->napi::Result<Self> {
        call(||{let parts=values.iter().map(|v|live(&v.inner).map(|v|v.clone())).collect::<core::Result<Vec<_>>>()?;
            Ok(Self{inner:Handle::new(g::Expr::choice(parts))})})
    }
    #[napi(factory)]
    pub fn sequence(values:Vec<ClassInstance<'_,NativeExpr>>)->napi::Result<Self> {
        call(||{let parts=values.iter().map(|v|live(&v.inner).map(|v|v.clone())).collect::<core::Result<Vec<_>>>()?;
            Ok(Self{inner:Handle::new(g::Expr::sequence(parts))})})
    }
    #[napi]
    pub fn clone(&self)->napi::Result<Self> {call(||Ok(Self{inner:Handle::new(live(&self.inner)?.clone())}))}
    #[napi]
    pub fn dispose(&self)->napi::Result<()> {call(||self.inner.close())}
}

struct ModuleSlot {module:g::Module,revision:u64}
#[napi]
pub struct NativeModule {inner:Arc<Handle<ModuleSlot>>}
impl NativeModule {
    pub(super) fn new(module:g::Module)->Self {Self{inner:Arc::new(Handle::new(ModuleSlot{module,revision:0}))}}
    fn snapshot(&self)->core::Result<g::Module> {Ok(live(&self.inner)?.module.clone())}
    fn bind_value(&self,name:String,value:g::ModuleValue)->core::Result<()> {
        let mut target=live(&self.inner)?;target.module.bind(name,value)?;target.revision+=1;Ok(())
    }
}
#[napi]
impl NativeModule {
    #[napi(factory)]
    pub fn from_grammar(value:&NativeGrammar,mode:f64,boundary:f64)->napi::Result<Self> {
        call(||Ok(Self::new(live(&value.inner)?.module().with_options(core::module_options(id(mode)?,id(boundary)?)?)?)))
    }
    #[napi(factory)]
    pub fn from_expr(value:&NativeExpr,mode:f64,boundary:f64)->napi::Result<Self> {
        call(||Ok(Self::new(g::Module::from_expr(&*live(&value.inner)?)?.with_options(core::module_options(id(mode)?,id(boundary)?)?)?)))
    }
    #[napi]
    pub fn clone(&self)->napi::Result<Self> {call(||Ok(Self::new(self.snapshot()?)))}
    #[napi]
    pub fn dispose(&self)->napi::Result<()> {call(||self.inner.close())}
    #[napi(getter)]
    pub fn options(&self)->napi::Result<NativeModuleOptions> {
        call(||{let (optimization,boundary_trigger)=core::module_option_values(self.snapshot()?.options())?;Ok(NativeModuleOptions{optimization,boundary_trigger})})
    }
    #[napi]
    pub fn with_options(&self,mode:f64,boundary:f64)->napi::Result<Self> {
        call(||Ok(Self::new(self.snapshot()?.with_options(core::module_options(id(mode)?,id(boundary)?)?)?)))
    }
    #[napi]
    pub fn set_options(&self,mode:f64,boundary:f64)->napi::Result<()> {
        call(||{let options=core::module_options(id(mode)?,id(boundary)?)?;let mut target=live(&self.inner)?;
            target.module.set_options(options)?;target.revision+=1;Ok(())})
    }
    #[napi]
    pub fn bind_grammar(&self,name:String,value:&NativeGrammar)->napi::Result<()> {
        call(||{let value=live(&value.inner)?.module();self.bind_value(name,g::ModuleValue::Module(value))})
    }
    #[napi]
    pub fn bind_module(&self,name:String,value:&NativeModule)->napi::Result<()> {
        call(||{let value=value.snapshot()?;self.bind_value(name,g::ModuleValue::Module(value))})
    }
    #[napi]
    pub fn bind_expr(&self,name:String,value:&NativeExpr)->napi::Result<()> {
        call(||{let value=g::Module::from_expr(&*live(&value.inner)?)?;self.bind_value(name,g::ModuleValue::Module(value))})
    }
    #[napi]
    pub fn bind_token(&self,name:String,value:f64)->napi::Result<()> {call(||self.bind_value(name,g::ModuleValue::Token(g::Token::new(id(value)?))))}
    #[napi]
    pub fn bind_tokens(&self,name:String,value:&NativeExactTokens)->napi::Result<()> {
        call(||{let value=live(&value.inner)?.clone();self.bind_value(name,g::ModuleValue::ExactTokens(value))})
    }
    #[napi]
    pub fn unbound(&self)->napi::Result<Vec<NativeRequirement>> {call(||requirements(self.snapshot()?.unbound()?))}
    #[napi]
    pub fn externs(&self)->napi::Result<Vec<NativeRequirement>> {call(||requirements(self.snapshot()?.externs()?))}
    #[napi]
    pub fn precompile_sync(&self,vocab:&NativeVocab)->napi::Result<()> {
        call(||{let vocab=live(&vocab.inner)?.clone();let mut target=live(&self.inner)?;
            target.module.precompile(&vocab)?;target.revision+=1;Ok(())})
    }
    #[napi]
    pub fn precompile(&self,vocab:&NativeVocab)->napi::Result<AsyncTask<ModulePrecompileTask>> {
        call(||{let target=live(&self.inner)?;
            Ok(AsyncTask::new(ModulePrecompileTask{module:target.module.clone(),revision:target.revision,
                owner:self.inner.clone(),vocab:live(&vocab.inner)?.clone()}))})
    }
    #[napi]
    pub fn compile_sync(&self,vocab:&NativeVocab,end:Vec<f64>)->napi::Result<NativeConstraint> {
        call(||Ok(NativeConstraint{inner:Handle::new(Arc::new(self.snapshot()?.compile_with(&*live(&vocab.inner)?,g::CompileOptions::default().end_tokens(ids(end)?))?))}))
    }
    #[napi]
    pub fn compile(&self,vocab:&NativeVocab,end:Vec<f64>)->napi::Result<AsyncTask<ModuleCompileTask>> {
        call(||Ok(AsyncTask::new(ModuleCompileTask{module:self.snapshot()?,vocab:live(&vocab.inner)?.clone(),options:g::CompileOptions::default().end_tokens(ids(end)?)})))
    }
    #[napi]
    pub fn save_sync(&self)->napi::Result<Buffer> {call(||Ok(self.snapshot()?.save().into()))}
    #[napi]
    pub fn save(&self)->napi::Result<AsyncTask<SaveTask>> {
        call(||Ok(AsyncTask::new(SaveTask{object:SaveObject::Module(self.snapshot()?)})))
    }
}

pub struct ModuleCompileTask {module:g::Module,vocab:g::Vocab,options:g::CompileOptions}
#[napi]
impl Task for ModuleCompileTask {
    type Output=Arc<g::Constraint>;type JsValue=NativeConstraint;
    fn compute(&mut self)->napi::Result<Self::Output> {call(||Ok(Arc::new(self.module.compile_with(&self.vocab,self.options.clone())?)))}
    fn resolve(&mut self,_:Env,value:Self::Output)->napi::Result<Self::JsValue> {Ok(NativeConstraint{inner:Handle::new(value)})}
}
pub struct ModulePrecompileTask {module:g::Module,vocab:g::Vocab,owner:Arc<Handle<ModuleSlot>>,revision:u64}
#[napi]
impl Task for ModulePrecompileTask {
    type Output=g::Module;type JsValue=();
    fn compute(&mut self)->napi::Result<Self::Output> {call(||{self.module.precompile(&self.vocab)?;Ok(self.module.clone())})}
    fn resolve(&mut self,_:Env,value:Self::Output)->napi::Result<()> {
        call(||{let mut target=live(&self.owner)?;
            if target.revision!=self.revision {return Err(core::Error::invalid("Module changed during precompile; retry with its current snapshot"));}
            target.module=value;target.revision+=1;Ok(())})
    }
}
pub struct LoadModuleTask {bytes:Vec<u8>,vocab:Option<g::Vocab>}
#[napi]
impl Task for LoadModuleTask {
    type Output=g::Module;type JsValue=NativeModule;
    fn compute(&mut self)->napi::Result<Self::Output> {call(||match &self.vocab {
        Some(v)=>Ok(g::Module::load_with_vocab(&self.bytes,v)?),None=>Ok(g::Module::load(&self.bytes)?),
    })}
    fn resolve(&mut self,_:Env,value:Self::Output)->napi::Result<NativeModule> {Ok(NativeModule::new(value))}
}
#[napi]
pub fn load_module(env:Env,bytes:Unknown<'_>,vocab:Option<&NativeVocab>)->napi::Result<AsyncTask<LoadModuleTask>> {
    call(||Ok(AsyncTask::new(LoadModuleTask{bytes:arrays::bytes(&env,&bytes)?,vocab:vocab.map(|v|live(&v.inner).map(|v|v.clone())).transpose()?})))
}
#[napi]
pub fn load_module_sync(env:Env,bytes:Unknown<'_>,vocab:Option<&NativeVocab>)->napi::Result<NativeModule> {
    let mut task=call(||Ok(LoadModuleTask{bytes:arrays::bytes(&env,&bytes)?,vocab:vocab.map(|v|live(&v.inner).map(|v|v.clone())).transpose()?}))?;
    let value=task.compute()?;task.resolve(env,value)
}
