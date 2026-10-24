//! Canonical owned Module/Expr surface. No JS callback runs under a native lock.
use super::*;

#[napi]
pub struct NativeExpr { pub(super) inner: Handle<g::Expr> }

#[napi]
impl NativeExpr {
    #[napi(factory)]
    pub fn from_grammar(value: &NativeGrammar) -> napi::Result<Self> {
        call(|| Ok(Self { inner: Handle::new(g::Expr::from(live(&value.inner)?.clone())) }))
    }
    #[napi(factory)]
    pub fn from_module(value: &NativeModule) -> napi::Result<Self> {
        call(|| Ok(Self { inner: Handle::new(g::Expr::from(value.snapshot()?)) }))
    }
    #[napi(factory)]
    pub fn from_token(value: f64) -> napi::Result<Self> {
        call(|| Ok(Self { inner: Handle::new(g::Expr::from(id(value)?)) }))
    }
    #[napi(factory)]
    pub fn from_token_set(value: &NativeTokenSet) -> napi::Result<Self> {
        call(|| Ok(Self { inner: Handle::new(g::Expr::from(live(&value.inner)?.clone())) }))
    }
    #[napi(factory)]
    pub fn literal(env: Env, bytes: Unknown<'_>) -> napi::Result<Self> {
        call(|| Ok(Self { inner: Handle::new(g::Expr::literal(arrays::bytes(&env, &bytes)?)) }))
    }
    #[napi(factory)]
    pub fn choice(values: Vec<ClassInstance<'_, NativeExpr>>) -> napi::Result<Self> {
        call(|| {
            let parts = values.iter().map(|v| live(&v.inner).map(|v| v.clone())).collect::<core::Result<Vec<_>>>()?;
            Ok(Self { inner: Handle::new(g::Expr::choice(parts)) })
        })
    }
    #[napi(factory)]
    pub fn sequence(values: Vec<ClassInstance<'_, NativeExpr>>) -> napi::Result<Self> {
        call(|| {
            let parts = values.iter().map(|v| live(&v.inner).map(|v| v.clone())).collect::<core::Result<Vec<_>>>()?;
            Ok(Self { inner: Handle::new(g::Expr::sequence(parts)) })
        })
    }
    #[napi]
    pub fn clone(&self) -> napi::Result<Self> { call(|| Ok(Self { inner: Handle::new(live(&self.inner)?.clone()) })) }
    #[napi]
    pub fn dispose(&self) -> napi::Result<()> { call(|| self.inner.close()) }
}

struct ModuleSlot { module: g::Module, revision: u64 }
#[napi]
pub struct NativeModule { inner: Arc<Handle<ModuleSlot>> }
impl NativeModule {
    pub(super) fn new(module: g::Module) -> Self { Self { inner: Arc::new(Handle::new(ModuleSlot { module, revision: 0 })) } }
    pub(super) fn snapshot(&self) -> core::Result<g::Module> { Ok(live(&self.inner)?.module.clone()) }
    fn bind_value(&self, name: &str, value: impl Into<g::Binding>) -> core::Result<()> {
        let mut target = live(&self.inner)?;
        target.module.bind(name, value)?;
        target.revision += 1;
        Ok(())
    }
}

#[napi]
impl NativeModule {
    #[napi(factory)]
    pub fn from_grammar(value: &NativeGrammar) -> napi::Result<Self> {
        call(|| Ok(Self::new(g::Module::new(&*live(&value.inner)?))))
    }
    #[napi]
    pub fn clone(&self) -> napi::Result<Self> { call(|| Ok(Self::new(self.snapshot()?))) }
    #[napi]
    pub fn dispose(&self) -> napi::Result<()> { call(|| self.inner.close()) }

    #[napi]
    pub fn bind_grammar(&self, name: String, value: &NativeGrammar) -> napi::Result<()> {
        call(|| {
            let child = live(&value.inner)?.clone();
            self.bind_value(&name, g::Binding::from(&child))
        })
    }
    #[napi]
    pub fn bind_module(&self, name: String, value: &NativeModule) -> napi::Result<()> {
        call(|| {
            let child = value.snapshot()?;
            self.bind_value(&name, g::Binding::from(child))
        })
    }
    #[napi]
    pub fn bind_expr(&self, name: String, value: &NativeExpr) -> napi::Result<()> {
        call(|| {
            let child = live(&value.inner)?.clone();
            self.bind_value(&name, g::Binding::from(child))
        })
    }
    #[napi]
    pub fn bind_token(&self, name: String, value: f64) -> napi::Result<()> {
        call(|| self.bind_value(&name, g::TokenSet::from(id(value)?)))
    }
    #[napi]
    pub fn bind_tokens(&self, name: String, value: &NativeTokenSet) -> napi::Result<()> {
        call(|| {
            let child = live(&value.inner)?.clone();
            self.bind_value(&name, g::Binding::from(child))
        })
    }

    #[napi]
    pub fn externs(&self) -> napi::Result<Vec<NativeExtern>> {
        call(|| Ok(externs_to_native(self.snapshot()?.externs().iter())))
    }
    #[napi]
    pub fn unbound_externs(&self) -> napi::Result<Vec<NativeExtern>> {
        call(|| Ok(externs_to_native(self.snapshot()?.externs().unbound().iter())))
    }

    fn options(mode: f64, boundary: f64, template_dfas: bool, merge_equivalent_tokens: bool) -> core::Result<g::CompileOptions> {
        Ok(core::compile_options(
            core::mask_strategy(id(mode)?)?,
            core::boundary_mask_strategy(id(boundary)?)?,
            template_dfas,
            merge_equivalent_tokens,
        ))
    }

    #[napi]
    pub fn precompile_sync(&self, vocab: &NativeVocab, mode: f64, boundary: f64, template_dfas: bool, merge_equivalent_tokens: bool) -> napi::Result<()> {
        call(|| {
            let vocab = live(&vocab.inner)?.clone();
            let options = Self::options(mode, boundary, template_dfas, merge_equivalent_tokens)?;
            let mut target = live(&self.inner)?;
            target.module.precompile_with(&vocab, options)?;
            target.revision += 1;
            Ok(())
        })
    }
    #[napi]
    pub fn precompile(&self, vocab: &NativeVocab, mode: f64, boundary: f64, template_dfas: bool, merge_equivalent_tokens: bool) -> napi::Result<AsyncTask<ModulePrecompileTask>> {
        call(|| {
            let target = live(&self.inner)?;
            Ok(AsyncTask::new(ModulePrecompileTask {
                module: target.module.clone(),
                revision: target.revision,
                owner: self.inner.clone(),
                vocab: live(&vocab.inner)?.clone(),
                options: Self::options(mode, boundary, template_dfas, merge_equivalent_tokens)?,
            }))
        })
    }

    #[napi]
    pub fn compile_sync(&self, vocab: &NativeVocab, mode: f64, boundary: f64, template_dfas: bool, merge_equivalent_tokens: bool) -> napi::Result<NativeConstraint> {
        call(|| Ok(NativeConstraint {
            inner: Handle::new(Arc::new(self.snapshot()?.compile_with(&*live(&vocab.inner)?, Self::options(mode, boundary, template_dfas, merge_equivalent_tokens)?)?)),
        }))
    }
    #[napi]
    pub fn compile(&self, vocab: &NativeVocab, mode: f64, boundary: f64, template_dfas: bool, merge_equivalent_tokens: bool) -> napi::Result<AsyncTask<ModuleCompileTask>> {
        call(|| Ok(AsyncTask::new(ModuleCompileTask {
            module: self.snapshot()?,
            vocab: live(&vocab.inner)?.clone(),
            options: Self::options(mode, boundary, template_dfas, merge_equivalent_tokens)?,
        })))
    }

    #[napi]
    pub fn save_sync(&self) -> napi::Result<Buffer> { call(|| Ok(self.snapshot()?.save().into())) }
    #[napi]
    pub fn save(&self) -> napi::Result<AsyncTask<SaveTask>> {
        call(|| Ok(AsyncTask::new(SaveTask { object: SaveObject::Module(self.snapshot()?) })))
    }
}

pub struct ModuleCompileTask { module: g::Module, vocab: g::Vocab, options: g::CompileOptions }
#[napi]
impl Task for ModuleCompileTask {
    type Output = Arc<g::Constraint>;
    type JsValue = NativeConstraint;
    fn compute(&mut self) -> napi::Result<Self::Output> { call(|| Ok(Arc::new(self.module.compile_with(&self.vocab, self.options.clone())?))) }
    fn resolve(&mut self, _: Env, value: Self::Output) -> napi::Result<Self::JsValue> { Ok(NativeConstraint { inner: Handle::new(value) }) }
}

pub struct ModulePrecompileTask { module: g::Module, vocab: g::Vocab, owner: Arc<Handle<ModuleSlot>>, revision: u64, options: g::CompileOptions }
#[napi]
impl Task for ModulePrecompileTask {
    type Output = g::Module;
    type JsValue = ();
    fn compute(&mut self) -> napi::Result<Self::Output> { call(|| { self.module.precompile_with(&self.vocab, self.options.clone())?; Ok(self.module.clone()) }) }
    fn resolve(&mut self, _: Env, value: Self::Output) -> napi::Result<()> {
        call(|| {
            let mut target = live(&self.owner)?;
            if target.revision != self.revision { return Err(core::Error::invalid("Module changed during precompile; retry with its current snapshot")); }
            target.module = value;
            target.revision += 1;
            Ok(())
        })
    }
}

pub struct LoadModuleTask { bytes: Vec<u8> }
#[napi]
impl Task for LoadModuleTask {
    type Output = g::Module;
    type JsValue = NativeModule;
    fn compute(&mut self) -> napi::Result<Self::Output> {
        call(|| Ok(g::Module::load(self.bytes.as_slice())?))
    }
    fn resolve(&mut self, _: Env, value: Self::Output) -> napi::Result<NativeModule> { Ok(NativeModule::new(value)) }
}

#[napi]
pub fn load_module(env: Env, bytes: Unknown<'_>) -> napi::Result<AsyncTask<LoadModuleTask>> {
    call(|| Ok(AsyncTask::new(LoadModuleTask { bytes: arrays::bytes(&env, &bytes)? })))
}
#[napi]
pub fn load_module_sync(env: Env, bytes: Unknown<'_>) -> napi::Result<NativeModule> {
    let mut task = call(|| Ok(LoadModuleTask { bytes: arrays::bytes(&env, &bytes)? }))?;
    let value = task.compute()?; task.resolve(env, value)
}
