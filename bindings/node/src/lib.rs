#![recursion_limit = "256"]

//! Native Node-API bindings. Public JS classes live in index.cjs.
mod arrays;
use module_api::NativeExpr;
use glrmask_binding_core::{self as core, glrmask as g};
use napi::{bindgen_prelude::*, Env, Task, Unknown};
use napi_derive::napi;
use std::{ops::{Deref, DerefMut}, sync::{Arc, Mutex, MutexGuard}};

fn error(e: core::Error) -> napi::Error {
    let code = match e.kind {
        core::ErrorKind::InvalidInput => "GLRMASK_INVALID_INPUT",
        core::ErrorKind::Compilation => "GLRMASK_COMPILATION",
        core::ErrorKind::BufferTooSmall => "GLRMASK_BUFFER_TOO_SMALL",
        core::ErrorKind::Closed => "GLRMASK_CLOSED",
        core::ErrorKind::Syntax => "GLRMASK_SYNTAX",
        core::ErrorKind::ResourceLimit => "GLRMASK_RESOURCE_LIMIT",
        core::ErrorKind::StrategyUnsupported => "GLRMASK_STRATEGY_UNSUPPORTED",
        core::ErrorKind::Serialization => "GLRMASK_SERIALIZATION",
        core::ErrorKind::VocabMismatch => "GLRMASK_VOCAB_MISMATCH",
        core::ErrorKind::PrecompileRace => "GLRMASK_PRECOMPILE_RACE",
        core::ErrorKind::Internal => "GLRMASK_INTERNAL",
    };
    napi::Error::new(napi::Status::GenericFailure, format!("{code}: {}", e.message))
}
fn call<T>(f: impl FnOnce() -> core::Result<T>) -> napi::Result<T> { core::guarded(f).map_err(error) }

struct Handle<T>(Mutex<Option<T>>);
struct Live<'a, T>(MutexGuard<'a, Option<T>>);
impl<T> Handle<T> {
    fn new(value: T) -> Self { Self(Mutex::new(Some(value))) }
    fn close(&self) -> core::Result<()> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).take(); Ok(())
    }
}
impl<T> Deref for Live<'_, T> { type Target = T; fn deref(&self) -> &T { self.0.as_ref().expect("checked live handle") } }
impl<T> DerefMut for Live<'_, T> { fn deref_mut(&mut self) -> &mut T { self.0.as_mut().expect("checked live handle") } }
fn live<T>(value: &Handle<T>) -> core::Result<Live<'_, T>> {
    let guard = value.0.lock().map_err(|_| core::Error::new(core::ErrorKind::Internal, "object is unusable after an internal panic; dispose it"))?;
    if guard.is_none() { return Err(core::Error::closed()); }
    Ok(Live(guard))
}
fn uint(value: f64) -> core::Result<u32> {
    if !value.is_finite() || value.fract() != 0.0 || !(0.0..=u32::MAX as f64).contains(&value) {
        return Err(core::Error::invalid("expected an integer from 0 to 4294967295"));
    }
    Ok(value as u32)
}
fn id(value: f64) -> core::Result<u32> {
    if !value.is_finite() || value.fract() != 0.0 || !(0.0..=g::__private::MAX_TOKEN_ID as f64).contains(&value) {
        return Err(core::Error::invalid(format!("token IDs must be integers from 0 to {}", g::__private::MAX_TOKEN_ID)));
    }
    Ok(value as u32)
}
fn ids(values: Vec<f64>) -> core::Result<Vec<u32>> { values.into_iter().map(id).collect() }

#[napi(object)]
pub struct NativeRange { pub start: f64, pub end: f64 }

#[napi(object)]
pub struct NativeExtern {
    pub name: String,
    pub kind: String,
}

pub fn externs_to_native<'a>(values: impl IntoIterator<Item = &'a g::Extern>) -> Vec<NativeExtern> {
    let mut items: Vec<_> = values.into_iter().map(|e| {
        let kind = match e.kind() {
            g::ExternKind::Token => "token",
            g::ExternKind::Grammar => "grammar",
            _ => unreachable!("every extern kind has a binding name"),
        };
        NativeExtern {
            name: e.name().to_string(),
            kind: kind.into(),
        }
    }).collect();
    items.sort_by(|a,b| a.name.cmp(&b.name));
    items
}

#[napi]
pub struct NativeVocab { inner: Handle<g::Vocab> }
#[napi]
impl NativeVocab {
    #[napi(factory)]
    pub fn from_entries(env: Env, token_ids: Vec<f64>, token_bytes: Vec<Unknown<'_>>) -> napi::Result<Self> {
        call(|| {
            if token_ids.len() != token_bytes.len() { return Err(core::Error::invalid("vocabulary IDs and bytes have different lengths")); }
            let token_ids = ids(token_ids)?;
            let bytes = token_bytes.iter().map(|b| arrays::bytes(&env, b)).collect::<core::Result<Vec<_>>>()?;
            Ok(Self { inner: Handle::new(core::vocab(token_ids.into_iter().zip(bytes).collect())?) })
        })
    }
    #[napi]
    pub fn clone(&self) -> napi::Result<Self> { call(|| Ok(Self { inner: Handle::new(live(&self.inner)?.clone()) })) }
    #[napi]
    pub fn dispose(&self) -> napi::Result<()> { call(|| self.inner.close()) }
}

#[napi]
pub struct NativeTokenSet { inner: Handle<g::TokenSet> }
#[napi]
impl NativeTokenSet {
    #[napi(factory)]
    pub fn from_ids(values: Vec<f64>) -> napi::Result<Self> {
        call(|| Ok(Self { inner: Handle::new(g::TokenSet::new(ids(values)?)) }))
    }
    #[napi(factory)]
    pub fn from_ranges(ranges: Vec<NativeRange>) -> napi::Result<Self> {
        call(|| {
            let mut out = Vec::new();
            for r in ranges {
                if !r.start.is_finite() || r.start.fract() != 0.0 || r.start < 0.0 || r.start > g::__private::MAX_TOKEN_ID as f64 + 1.0 || !r.end.is_finite() || r.end.fract() != 0.0 || r.end < 0.0 || r.end > g::__private::MAX_TOKEN_ID as f64 + 1.0 {
                    return Err(core::Error::invalid("ranges must have integer endpoints in the supported range, with the end excluded"));
                }
                if r.end < r.start { return Err(core::Error::invalid("range start exceeds end")); }
                if r.end == r.start { continue; }
                out.push(r.start as u32..r.end as u32);
            }
            Ok(Self { inner: Handle::new(g::TokenSet::from_ranges(out)) })
        })
    }
    #[napi]
    pub fn clone(&self) -> napi::Result<Self> { call(|| Ok(Self { inner: Handle::new(live(&self.inner)?.clone()) })) }
    #[napi]
    pub fn dispose(&self) -> napi::Result<()> { call(|| self.inner.close()) }
    #[napi]
    pub fn equals(&self, other: &NativeTokenSet) -> napi::Result<bool> {
        call(|| Ok(*live(&self.inner)? == *live(&other.inner)?))
    }
    #[napi]
    pub fn to_string(&self) -> napi::Result<String> { call(|| Ok(format!("{:?}", *live(&self.inner)?))) }
}

fn grammar_compile_options(mode: f64) -> core::Result<g::CompileOptions> {
    Ok(core::compile_options(
        core::mask_strategy(uint(mode)?)?,
    ))
}

#[napi]
pub struct NativeGrammar { inner: Handle<g::Grammar> }
#[napi]
impl NativeGrammar {
    #[napi(factory)]
    pub fn from_source(format: String, source: String) -> napi::Result<Self> {
        call(|| {
            let grammar = match format.as_str() {
                "glrm" => g::Grammar::from_glrm(&source)?,
                "json-schema" => g::Grammar::from_json_schema(&source)?,
                "lark" => g::Grammar::from_lark(&source)?,
                _ => return Err(core::Error::invalid("unknown grammar source format")),
            }; Ok(Self { inner: Handle::new(grammar) })
        })
    }
    #[napi]
    pub fn clone(&self) -> napi::Result<Self> { call(|| Ok(Self { inner: Handle::new(live(&self.inner)?.clone()) })) }
    #[napi]
    pub fn dispose(&self) -> napi::Result<()> { call(|| self.inner.close()) }
    #[napi]
    pub fn externs(&self) -> napi::Result<Vec<NativeExtern>> {
        call(|| Ok(externs_to_native(live(&self.inner)?.externs().iter())))
    }
    #[napi]
    pub fn unbound_externs(&self) -> napi::Result<Vec<NativeExtern>> {
        call(|| Ok(externs_to_native(live(&self.inner)?.externs().unbound().iter())))
    }
    #[napi(factory)]
    pub fn from_expr(value: &NativeExpr) -> napi::Result<Self> {
        call(|| Ok(Self { inner: Handle::new(g::Grammar::from_expr(&*live(&value.inner)?)?) }))
    }
    #[napi]
    pub fn bind_grammar(&self, name: String, value: &NativeGrammar) -> napi::Result<()> {
        call(|| { live(&self.inner)?.bind(&name, live(&value.inner)?.clone())?; Ok(()) })
    }
    #[napi]
    pub fn bind_expr(&self, name: String, value: &NativeExpr) -> napi::Result<()> {
        call(|| { live(&self.inner)?.bind(&name, live(&value.inner)?.clone())?; Ok(()) })
    }
    #[napi]
    pub fn bind_tokens(&self, name: String, value: &NativeTokenSet) -> napi::Result<()> {
        call(|| { live(&self.inner)?.bind(&name, g::Expr::from(live(&value.inner)?.clone()))?; Ok(()) })
    }
    #[napi]
    pub fn compile_sync(&self, vocab: &NativeVocab, mode: f64) -> napi::Result<NativeConstraint> {
        call(|| {
            let m = g::Module::new(&*live(&self.inner)?);
            let options = grammar_compile_options(mode)?;
            Ok(NativeConstraint { inner: Handle::new(Arc::new(m.compile_with(&*live(&vocab.inner)?, options)?)) })
        })
    }
    #[napi]
    pub fn compile(&self, vocab: &NativeVocab, mode: f64) -> napi::Result<AsyncTask<CompileTask>> {
        call(|| {
            let m = g::Module::new(&*live(&self.inner)?);
            Ok(AsyncTask::new(CompileTask { module: m, vocab: live(&vocab.inner)?.clone(), options: grammar_compile_options(mode)? }))
        })
    }
}

#[napi]
pub struct NativeConstraint { inner: Handle<Arc<g::Constraint>> }
#[napi]
impl NativeConstraint {
    #[napi]
    pub fn clone(&self) -> napi::Result<Self> { call(|| Ok(Self { inner: Handle::new(live(&self.inner)?.clone()) })) }
    #[napi]
    pub fn dispose(&self) -> napi::Result<()> { call(|| self.inner.close()) }
    #[napi]
    pub fn start(&self) -> napi::Result<NativeState> { call(|| Ok(NativeState { inner: Handle::new(core::State::start(live(&self.inner)?.clone())?) })) }
    #[napi(getter)]
    pub fn mask_bit_len(&self) -> napi::Result<f64> { call(|| Ok(live(&self.inner)?.mask_bit_len() as f64)) }
    #[napi]
    pub fn save_sync(&self) -> napi::Result<Buffer> {
        call(|| Ok(live(&self.inner)?.save().into()))
    }
    #[napi]
    pub fn save(&self) -> napi::Result<AsyncTask<SaveTask>> {
        call(|| Ok(AsyncTask::new(SaveTask { object: SaveObject::Constraint(live(&self.inner)?.clone()) })))
    }
}

#[napi]
pub struct NativeState { inner: Handle<core::State> }
#[napi]
impl NativeState {
    #[napi]
    pub fn clone(&self) -> napi::Result<Self> { call(|| Ok(Self { inner: Handle::new(live(&self.inner)?.try_clone()?) })) }
    #[napi]
    pub fn dispose(&self) -> napi::Result<()> { call(|| self.inner.close()) }
    #[napi(getter)]
    pub fn mask_bit_len(&self) -> napi::Result<f64> { call(|| Ok(live(&self.inner)?.mask_bit_len() as f64)) }
    #[napi]
    pub fn mask(&self) -> napi::Result<Uint32Array> { call(|| Ok(live(&self.inner)?.mask()?.into())) }
    #[napi]
    pub fn fill_mask(&self, env: Env, output: Unknown<'_>) -> napi::Result<()> {
        call(|| arrays::with_words(&env, &output, |out| live(&self.inner)?.fill_mask(out)))
    }
    #[napi]
    pub fn commit_token(&self, value: f64) -> napi::Result<bool> {
        call(|| {
            let mut state = live(&self.inner)?;
            // Invalid input must not alter the state.
            if !value.is_finite()
                || value.fract() != 0.0
                || !(0.0..=u32::MAX as f64).contains(&value)
            {
                return Err(core::Error::invalid("token ID is outside the u32 range"));
            }
            state.commit_token(value as u32)
        })
    }
    #[napi]
    pub fn commit_bytes(&self, env: Env, bytes: Unknown<'_>) -> napi::Result<bool> {
        call(|| arrays::with_bytes(&env, &bytes, |bytes| live(&self.inner)?.commit_bytes(bytes)))
    }
    #[napi]
    pub fn is_accepting(&self) -> napi::Result<bool> { call(|| live(&self.inner)?.is_accepting()) }
    #[napi]
    pub fn is_terminated(&self) -> napi::Result<bool> { call(|| live(&self.inner)?.is_terminated()) }
    #[napi]
    pub fn forced_bytes(&self) -> napi::Result<Uint8Array> {
        call(|| Ok(live(&self.inner)?.forced_bytes()?.into()))
    }
}

pub struct CompileTask { module: g::Module, vocab: g::Vocab, options: g::CompileOptions }
#[napi]
impl Task for CompileTask {
    type Output = Arc<g::Constraint>;
    type JsValue = NativeConstraint;
    fn compute(&mut self) -> napi::Result<Self::Output> { call(|| Ok(Arc::new(self.module.compile_with(&self.vocab, self.options.clone())?))) }
    fn resolve(&mut self, _: Env, value: Self::Output) -> napi::Result<Self::JsValue> { Ok(NativeConstraint { inner: Handle::new(value) }) }
}

enum SaveObject { Module(g::Module), Constraint(Arc<g::Constraint>) }
pub struct SaveTask { object: SaveObject }
#[napi]
impl Task for SaveTask {
    type Output = Vec<u8>;
    type JsValue = Buffer;
    fn compute(&mut self) -> napi::Result<Self::Output> {
        call(|| Ok(match &self.object {
            SaveObject::Constraint(c) => c.save(),
            SaveObject::Module(m) => m.save(),
        }))
    }
    fn resolve(&mut self, _: Env, value: Self::Output) -> napi::Result<Self::JsValue> { Ok(value.into()) }
}

pub struct LoadConstraintTask { bytes: Vec<u8>, vocab: g::Vocab }
#[napi]
impl Task for LoadConstraintTask {
    type Output = Arc<g::Constraint>;
    type JsValue = NativeConstraint;
    fn compute(&mut self) -> napi::Result<Self::Output> {
        call(|| {
            let bytes = std::mem::take(&mut self.bytes);
            Ok(Arc::new(g::Constraint::load(&bytes, &self.vocab)?))
        })
    }
    fn resolve(&mut self, _: Env, value: Self::Output) -> napi::Result<Self::JsValue> { Ok(NativeConstraint { inner: Handle::new(value) }) }
}

#[napi]
pub fn load_constraint(env: Env, bytes: Unknown<'_>, vocab: &NativeVocab) -> napi::Result<AsyncTask<LoadConstraintTask>> {
    call(|| Ok(AsyncTask::new(LoadConstraintTask { bytes: arrays::bytes(&env, &bytes)?, vocab: live(&vocab.inner)?.clone() })))
}
#[napi]
pub fn load_constraint_sync(env: Env, bytes: Unknown<'_>, vocab: &NativeVocab) -> napi::Result<NativeConstraint> {
    let mut task = call(|| Ok(LoadConstraintTask { bytes: arrays::bytes(&env, &bytes)?, vocab: live(&vocab.inner)?.clone() }))?;
    let value = task.compute()?; task.resolve(env, value)
}

#[napi]
pub fn version() -> String { env!("CARGO_PKG_VERSION").to_owned() }

mod module_api;
