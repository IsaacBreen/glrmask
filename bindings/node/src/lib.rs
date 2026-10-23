#![recursion_limit = "256"]

//! Native Node-API bindings. Public JS classes live in index.cjs.
mod arrays;
use module_api::{NativeExpr, NativeModule};
use glrmask_binding_core::{self as core, glrmask as g};
use napi::{bindgen_prelude::*, Env, Task, Unknown};
use napi_derive::napi;
use std::{ops::{Deref, DerefMut}, sync::{Arc, Mutex, MutexGuard}};

fn error(e: core::Error) -> napi::Error {
    let code = match e.kind {
        core::ErrorKind::InvalidArgument => "GLRMASK_INVALID_ARGUMENT",
        core::ErrorKind::Engine => "GLRMASK_ENGINE_ERROR",
        core::ErrorKind::Panic => "GLRMASK_PANIC",
        core::ErrorKind::BufferTooSmall => "GLRMASK_BUFFER_TOO_SMALL",
        core::ErrorKind::Closed => "GLRMASK_CLOSED",
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
    let guard = value.0.lock().map_err(|_| core::Error::new(core::ErrorKind::Panic, "object is unusable after an internal panic; dispose it"))?;
    if guard.is_none() { return Err(core::Error::closed()); }
    Ok(Live(guard))
}
fn id(value: f64) -> core::Result<u32> {
    if !value.is_finite() || value.fract() != 0.0 || !(0.0..=u32::MAX as f64).contains(&value) {
        return Err(core::Error::invalid("token IDs must be integers in 0..4294967295"));
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

pub fn externs_to_native<'a>(values: impl IntoIterator<Item = g::Extern>) -> Vec<NativeExtern> {
    values.into_iter().map(|e| {
        let kind = match e.kind() {
            g::ExternKind::Token => "token",
            g::ExternKind::Grammar => "grammar",
            _ => "grammar",
        };
        NativeExtern {
            name: e.name().to_string(),
            kind: kind.into(),
        }
    }).collect()
}

#[napi(object)]
pub struct NativeMarkerEvent { pub id: u32, pub byte_offset: BigInt }

#[napi]
pub struct NativeVocab { inner: Handle<g::Vocab> }
#[napi]
impl NativeVocab {
    #[napi(factory)]
    pub fn from_entries(env: Env, token_ids: Vec<f64>, token_bytes: Vec<Unknown<'_>>, special: Vec<f64>) -> napi::Result<Self> {
        call(|| {
            if token_ids.len() != token_bytes.len() { return Err(core::Error::invalid("vocabulary IDs and bytes have different lengths")); }
            let token_ids = ids(token_ids)?;
            let bytes = token_bytes.iter().map(|b| arrays::bytes(&env, b)).collect::<core::Result<Vec<_>>>()?;
            Ok(Self { inner: Handle::new(core::vocab(token_ids.into_iter().zip(bytes).collect(), ids(special)?)?) })
        })
    }
    #[napi]
    pub fn clone(&self) -> napi::Result<Self> { call(|| Ok(Self { inner: Handle::new(live(&self.inner)?.clone()) })) }
    #[napi]
    pub fn dispose(&self) -> napi::Result<()> { call(|| self.inner.close()) }
    #[napi(getter)]
    pub fn size(&self) -> napi::Result<f64> { call(|| Ok(live(&self.inner)?.iter().len() as f64)) }
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
                let first = id(r.start)?;
                if !r.end.is_finite() || r.end.fract() != 0.0 || r.end > 4294967296.0 {
                    return Err(core::Error::invalid("ranges must have integer half-open endpoints in 0..4294967296"));
                }
                if r.end <= r.start { continue; }
                out.push(first..=((r.end as u64 - 1) as u32));
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

fn grammar_compile_options(mode: f64, boundary: f64) -> core::Result<g::CompileOptions> {
    Ok(core::compile_options(
        core::mask_strategy(id(mode)?)?,
        core::boundary_mask_strategy(id(boundary)?)?,
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
                "ebnf" => g::Grammar::from_ebnf(source),
                "glrm" => g::Grammar::from_glrm(source),
                "json-schema" => g::Grammar::from_json_schema(source),
                "lark" => g::Grammar::from_lark(source),
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
    #[napi(factory)]
    pub fn from_expr(value: &NativeExpr) -> napi::Result<Self> {
        call(|| Ok(Self { inner: Handle::new(g::Grammar::from_expr(live(&value.inner)?.clone())?) }))
    }
    #[napi]
    pub fn bind_grammar(&self, name: String, value: &NativeGrammar) -> napi::Result<()> {
        call(|| { live(&self.inner)?.bind(name, live(&value.inner)?.clone())?; Ok(()) })
    }
    #[napi]
    pub fn bind_expr(&self, name: String, value: &NativeExpr) -> napi::Result<()> {
        call(|| { live(&self.inner)?.bind(name, live(&value.inner)?.clone())?; Ok(()) })
    }
    #[napi]
    pub fn bind_token(&self, name: String, value: f64) -> napi::Result<()> {
        call(|| { live(&self.inner)?.bind(name, g::TokenSet::from(id(value)?))?; Ok(()) })
    }
    #[napi]
    pub fn bind_tokens(&self, name: String, value: &NativeTokenSet) -> napi::Result<()> {
        call(|| { live(&self.inner)?.bind(name, live(&value.inner)?.clone())?; Ok(()) })
    }
    #[napi]
    pub fn compile_sync(&self, vocab: &NativeVocab, mode: f64, boundary: f64) -> napi::Result<NativeConstraint> {
        call(|| {
            let m = g::Module::new(&*live(&self.inner)?);
            let options = grammar_compile_options(mode, boundary)?;
            Ok(NativeConstraint { inner: Handle::new(Arc::new(m.compile_with(&*live(&vocab.inner)?, options)?)) })
        })
    }
    #[napi]
    pub fn compile(&self, vocab: &NativeVocab, mode: f64, boundary: f64) -> napi::Result<AsyncTask<CompileTask>> {
        call(|| {
            let m = g::Module::new(&*live(&self.inner)?);
            Ok(AsyncTask::new(CompileTask { module: m, vocab: live(&vocab.inner)?.clone(), options: grammar_compile_options(mode, boundary)? }))
        })
    }
    #[napi]
    pub fn precompile_sync(&self, vocab: &NativeVocab, mode: f64, boundary: f64) -> napi::Result<NativeModule> {
        call(|| {
            let mut m = g::Module::new(&*live(&self.inner)?);
            m.precompile_with(&*live(&vocab.inner)?, grammar_compile_options(mode, boundary)?)?;
            Ok(NativeModule::new(m))
        })
    }
    #[napi]
    pub fn precompile(&self, vocab: &NativeVocab, mode: f64, boundary: f64) -> napi::Result<AsyncTask<GrammarPrecompileTask>> {
        call(|| {
            let m = g::Module::new(&*live(&self.inner)?);
            Ok(AsyncTask::new(GrammarPrecompileTask { module: m, vocab: live(&vocab.inner)?.clone(), options: grammar_compile_options(mode, boundary)? }))
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
    pub fn mask_len(&self) -> napi::Result<f64> { call(|| Ok(live(&self.inner)?.mask_len() as f64)) }
    #[napi]
    pub fn marker_ids(&self) -> napi::Result<Uint32Array> { call(|| Ok(live(&self.inner)?.marker_ids().to_vec().into())) }
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
    pub fn mask_len(&self) -> napi::Result<f64> { call(|| Ok(live(&self.inner)?.mask_len() as f64)) }
    #[napi]
    pub fn mask(&self) -> napi::Result<Uint32Array> { call(|| Ok(live(&self.inner)?.mask()?.into())) }
    #[napi]
    pub fn fill_mask(&self, env: Env, output: Unknown<'_>) -> napi::Result<()> {
        call(|| arrays::with_words(&env, &output, |out| live(&self.inner)?.fill_mask(out)))
    }
    #[napi]
    pub fn commit_token(&self, value: f64) -> napi::Result<bool> { call(|| live(&self.inner)?.commit_token(id(value)?)) }
    #[napi]
    pub fn commit_tokens(&self, ids: Uint32Array) -> napi::Result<bool> {
        call(|| live(&self.inner)?.commit_tokens(ids.as_ref()))
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
    pub fn forced_tokens(&self) -> napi::Result<Uint32Array> { call(|| Ok(live(&self.inner)?.forced_tokens()?.into())) }
    #[napi]
    pub fn marker_events(&self) -> napi::Result<Vec<NativeMarkerEvent>> {
        call(|| Ok(live(&self.inner)?.marker_events()?.into_iter().map(|e| NativeMarkerEvent {
            id: e.id, byte_offset: BigInt { sign_bit: false, words: vec![e.byte_offset] },
        }).collect()))
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

pub struct GrammarPrecompileTask { module: g::Module, vocab: g::Vocab, options: g::CompileOptions }
#[napi]
impl Task for GrammarPrecompileTask {
    type Output = g::Module;
    type JsValue = NativeModule;
    fn compute(&mut self) -> napi::Result<Self::Output> { call(|| { self.module.precompile_with(&self.vocab, self.options.clone())?; Ok(self.module.clone()) }) }
    fn resolve(&mut self, _: Env, value: Self::Output) -> napi::Result<Self::JsValue> { Ok(NativeModule::new(value)) }
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
            Ok(Arc::new(g::Constraint::load(bytes, &self.vocab)?))
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
