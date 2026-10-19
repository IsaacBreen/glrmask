//! Native Node-API bindings. Public JS classes live in index.cjs.
mod arrays;
use module_api::{NativeExpr,NativeModule};
use glrmask_binding_core::{self as core, glrmask as g};
use napi::{bindgen_prelude::*, Env, Task, Unknown};
use napi_derive::napi;
use std::{collections::BTreeMap, ops::{Deref,DerefMut}, sync::{Arc,Mutex,MutexGuard}};

fn error(e:core::Error)->napi::Error {
    let code=match e.kind {
        core::ErrorKind::InvalidArgument=>"GLRMASK_INVALID_ARGUMENT",
        core::ErrorKind::Engine=>"GLRMASK_ENGINE_ERROR",
        core::ErrorKind::Panic=>"GLRMASK_PANIC",
        core::ErrorKind::BufferTooSmall=>"GLRMASK_BUFFER_TOO_SMALL",
        core::ErrorKind::Closed=>"GLRMASK_CLOSED",
    };
    napi::Error::new(napi::Status::GenericFailure,format!("{code}: {}",e.message))
}
fn call<T>(f:impl FnOnce()->core::Result<T>)->napi::Result<T> {core::guarded(f).map_err(error)}
// Only shared references cross the generated NAPI class interface. Interior
// mutexes avoid creating aliased &mut references even with direct addon access,
// self-binding, argument getters or reentrant JavaScript calls.
struct Handle<T>(Mutex<Option<T>>);
struct Live<'a,T>(MutexGuard<'a,Option<T>>);
impl<T> Handle<T> {
    fn new(value:T)->Self {Self(Mutex::new(Some(value)))}
    fn close(&self)->core::Result<()> {
        // Disposal is always available, even after an internal panic.
        self.0.lock().unwrap_or_else(|e|e.into_inner()).take(); Ok(())
    }
}
impl<T> Deref for Live<'_,T> {type Target=T;fn deref(&self)->&T {self.0.as_ref().expect("checked live handle")}}
impl<T> DerefMut for Live<'_,T> {fn deref_mut(&mut self)->&mut T {self.0.as_mut().expect("checked live handle")}}
fn live<T>(value:&Handle<T>)->core::Result<Live<'_,T>> {
    let guard=value.0.lock().map_err(|_|core::Error::new(core::ErrorKind::Panic,"object is unusable after an internal panic; dispose it"))?;
    if guard.is_none() {return Err(core::Error::closed());}
    Ok(Live(guard))
}
fn id(value:f64)->core::Result<u32> {
    if !value.is_finite() || value.fract()!=0.0 || !(0.0..=u32::MAX as f64).contains(&value) {
        return Err(core::Error::invalid("token IDs must be integers in 0..4294967295"));
    }
    Ok(value as u32)
}
fn ids(values:Vec<f64>)->core::Result<Vec<u32>> {values.into_iter().map(id).collect()}
fn options(mode:f64,end:Vec<f64>,boundary:Option<f64>)->core::Result<g::BuildOptions> {
    core::build_options(id(mode)?,ids(end)?,boundary.map(id).transpose()?)
}

#[napi(object)]
pub struct NativeRange {pub start:f64,pub end:f64}
#[napi(object)]
pub struct NativeRequirement {
    pub name:String,pub path:Vec<String>,pub kind:String,
    pub source_span:Option<Vec<f64>>,
}
fn requirements(values:Vec<g::UnboundExtern>)->core::Result<Vec<NativeRequirement>> {
    values.into_iter().map(|r|{
        let kind=match r.kind {g::ExternalKind::Token=>"token",g::ExternalKind::Grammar=>"grammar",g::ExternalKind::LegacyGeneric=>"legacy"};
        Ok(NativeRequirement{name:r.name(),path:r.path,kind:kind.into(),source_span:r.source_span.map(|s|vec![s.start as f64,s.end as f64,s.line as f64,s.column as f64])})
    }).collect()
}
#[napi(object)]
pub struct NativeMarkerEvent {pub id:u32,pub byte_offset:BigInt}

#[napi]
pub struct NativeVocab {inner:Handle<g::Vocab>}
#[napi]
impl NativeVocab {
    #[napi(factory)]
    pub fn from_entries(env:Env,token_ids:Vec<f64>,token_bytes:Vec<Unknown<'_>>,exact_only:Vec<f64>)->napi::Result<Self> {
        call(||{
            if token_ids.len()!=token_bytes.len(){return Err(core::Error::invalid("vocabulary IDs and bytes have different lengths"));}
            let token_ids=ids(token_ids)?;
            let bytes=token_bytes.iter().map(|b|arrays::bytes(&env,b)).collect::<core::Result<Vec<_>>>()?;
            Ok(Self{inner:Handle::new(core::vocab(token_ids.into_iter().zip(bytes).collect(),ids(exact_only)?)?)})
        })
    }
    #[napi]
    pub fn clone(&self)->napi::Result<Self> {call(||Ok(Self{inner:Handle::new(live(&self.inner)?.clone())}))}
    #[napi]
    pub fn dispose(&self)->napi::Result<()> {call(||self.inner.close())}
    #[napi(getter)]
    pub fn size(&self)->napi::Result<f64> {call(||Ok(live(&self.inner)?.len() as f64))}
    #[napi]
    pub fn tokens(&self,values:Vec<f64>)->napi::Result<NativeExactTokens> {
        call(||Ok(NativeExactTokens{inner:Handle::new(live(&self.inner)?.tokens(ids(values)?).map_err(core::Error::engine)?)}))
    }
    #[napi]
    pub fn token_ranges(&self,ranges:Vec<NativeRange>)->napi::Result<NativeExactTokens> {
        call(||{
            let ranges=ranges.into_iter().map(|r|{
                let first=id(r.start)?;
                if !r.end.is_finite() || r.end.fract()!=0.0 || r.end<=r.start || r.end>4294967296.0 {
                    return Err(core::Error::invalid("ranges must be nonempty with integer half-open endpoints in 0..4294967296"));
                }
                Ok(first..=((r.end as u64-1) as u32))
            }).collect::<core::Result<Vec<_>>>()?;
            Ok(NativeExactTokens{inner:Handle::new(live(&self.inner)?.token_ranges(ranges).map_err(core::Error::engine)?)})
        })
    }
}
#[napi]
pub struct NativeExactTokens {inner:Handle<g::ExactTokens>}
#[napi]
impl NativeExactTokens {
    #[napi]
    pub fn clone(&self)->napi::Result<Self> {call(||Ok(Self{inner:Handle::new(live(&self.inner)?.clone())}))}
    #[napi]
    pub fn dispose(&self)->napi::Result<()> {call(||self.inner.close())}
    #[napi(getter)]
    pub fn size(&self)->napi::Result<f64> {call(||Ok(live(&self.inner)?.len() as f64))}
    #[napi]
    pub fn ids(&self)->napi::Result<Uint32Array> {call(||Ok(live(&self.inner)?.ids().to_vec().into()))}
    #[napi]
    pub fn ranges(&self)->napi::Result<Vec<NativeRange>> {
        call(||Ok(live(&self.inner)?.ranges().map(|r|NativeRange{start:*r.start() as f64,end:*r.end() as f64+1.0}).collect()))
    }
}

#[napi]
pub struct NativeGrammar {inner:Handle<g::Grammar<'static>>}
#[napi]
impl NativeGrammar {
    #[napi(factory)]
    pub fn from_expr(value:&NativeExpr)->napi::Result<Self> {
        call(||Ok(Self{inner:Handle::new(g::Grammar::from_expr(&*live(&value.inner)?)?)}))
    }
    #[napi]
    pub fn module(&self)->napi::Result<NativeModule> {call(||Ok(NativeModule::new(live(&self.inner)?.module())))}
    #[napi]
    pub fn bind_expr(&self,name:String,value:&NativeExpr)->napi::Result<()> {
        call(||{let child=g::Grammar::from_expr(&*live(&value.inner)?)?;live(&self.inner)?.bind(name,child)?;Ok(())})
    }
    #[napi]
    pub fn unbound(&self)->napi::Result<Vec<NativeRequirement>> {call(||requirements(live(&self.inner)?.unbound()?))}
    #[napi]
    pub fn externs(&self)->napi::Result<Vec<NativeRequirement>> {call(||requirements(live(&self.inner)?.externs()?))}

    #[napi(factory)]
    pub fn from_source(format:String,source:String,profile:Option<String>,token_names:Option<BTreeMap<String,Vec<f64>>>)->napi::Result<Self> {
        call(||{
            let grammar=match format.as_str() {
                "ebnf"=>g::Grammar::from_ebnf_owned(source),
                "glrm"=>g::Grammar::from_glrm_owned(source),
                "json-schema"=>g::Grammar::from_json_schema_owned(source),
                "lark"=>{
                    let mut o=match profile.as_deref().unwrap_or("recognition") {
                        "recognition"=>g::LarkOptions::default(),"llm"=>g::LarkOptions::llm(),
                        _=>return Err(core::Error::invalid("Lark profile must be recognition or llm")),
                    };
                    for (name,values) in token_names.unwrap_or_default() {o=o.token(name,ids(values)?);}
                    g::Grammar::from_lark_owned(source,o)
                }
                _=>return Err(core::Error::invalid("unknown grammar source format")),
            }; Ok(Self{inner:Handle::new(grammar)})
        })
    }
    #[napi(factory)]
    pub fn exact(tokens:&NativeExactTokens)->napi::Result<Self> {call(||Ok(Self{inner:Handle::new(g::Grammar::exact(live(&tokens.inner)?.clone()))}))}
    #[napi]
    pub fn clone(&self)->napi::Result<Self> {call(||Ok(Self{inner:Handle::new(live(&self.inner)?.clone())}))}
    #[napi]
    pub fn dispose(&self)->napi::Result<()> {call(||self.inner.close())}
    #[napi]
    pub fn bind_grammar(&self,name:String,value:&NativeGrammar)->napi::Result<()> {
        call(||{let value=live(&value.inner)?.clone();live(&self.inner)?.bind(name,value)?;Ok(())})
    }
    #[napi]
    pub fn bind_token(&self,name:String,value:f64)->napi::Result<()> {
        call(||{live(&self.inner)?.bind(name,g::Token::new(id(value)?))?;Ok(())})
    }
    #[napi]
    pub fn bind_tokens(&self,name:String,value:&NativeExactTokens)->napi::Result<()> {
        call(||{live(&self.inner)?.bind(name,live(&value.inner)?.clone())?;Ok(())})
    }
    #[napi]
    pub fn requirements(&self)->napi::Result<Vec<NativeRequirement>> {call(||requirements(live(&self.inner)?.requirements()?))}
    #[napi]
    pub fn compile_sync(&self,vocab:&NativeVocab,mode:f64,end:Vec<f64>,boundary:Option<f64>)->napi::Result<NativeConstraint> {
        call(||Ok(NativeConstraint{inner:Handle::new(Arc::new(live(&self.inner)?.compile_with(&*live(&vocab.inner)?,options(mode,end,boundary)?)?))}))
    }
    #[napi]
    pub fn compile(&self,vocab:&NativeVocab,mode:f64,end:Vec<f64>,boundary:Option<f64>)->napi::Result<AsyncTask<CompileTask>> {
        call(||Ok(AsyncTask::new(CompileTask{grammar:live(&self.inner)?.clone(),vocab:live(&vocab.inner)?.clone(),options:options(mode,end,boundary)?})))
    }
    #[napi]
    pub fn prepare_sync(&self,vocab:&NativeVocab,boundary:Option<f64>)->napi::Result<NativePreparedGrammar> {
        call(||Ok(NativePreparedGrammar{inner:Handle::new(live(&self.inner)?.prepare_with(&*live(&vocab.inner)?,options(0.0,vec![],boundary)?)?)}))
    }
    #[napi]
    pub fn prepare(&self,vocab:&NativeVocab,boundary:Option<f64>)->napi::Result<AsyncTask<PrepareTask>> {
        call(||Ok(AsyncTask::new(PrepareTask{grammar:live(&self.inner)?.clone(),vocab:live(&vocab.inner)?.clone(),options:options(0.0,vec![],boundary)?})))
    }
}

#[napi]
pub struct NativePreparedGrammar {inner:Handle<g::PreparedGrammar>}
#[napi]
impl NativePreparedGrammar {
    #[napi]
    pub fn clone(&self)->napi::Result<Self> {call(||Ok(Self{inner:Handle::new(live(&self.inner)?.clone())}))}
    #[napi]
    pub fn dispose(&self)->napi::Result<()> {call(||self.inner.close())}
    #[napi]
    pub fn bind_prepared(&self,name:String,value:&NativePreparedGrammar)->napi::Result<()> {
        call(||{let value=live(&value.inner)?.clone();live(&self.inner)?.bind(name,value)?;Ok(())})
    }
    #[napi]
    pub fn bind_token(&self,name:String,value:f64)->napi::Result<()> {
        call(||{live(&self.inner)?.bind(name,g::Token::new(id(value)?))?;Ok(())})
    }
    #[napi]
    pub fn bind_tokens(&self,name:String,value:&NativeExactTokens)->napi::Result<()> {
        call(||{let value=live(&value.inner)?.clone();live(&self.inner)?.bind(name,value)?;Ok(())})
    }
    #[napi]
    pub fn requirements(&self)->napi::Result<Vec<NativeRequirement>> {call(||requirements(live(&self.inner)?.requirements()?))}
    #[napi]
    pub fn link_sync(&self,mode:f64,end:Vec<f64>,boundary:Option<f64>)->napi::Result<NativeConstraint> {
        call(||Ok(NativeConstraint{inner:Handle::new(Arc::new(live(&self.inner)?.link_with(options(mode,end,boundary)?)?))}))
    }
    #[napi]
    pub fn link(&self,mode:f64,end:Vec<f64>,boundary:Option<f64>)->napi::Result<AsyncTask<LinkTask>> {
        call(||Ok(AsyncTask::new(LinkTask{prepared:live(&self.inner)?.clone(),options:options(mode,end,boundary)?})))
    }
    #[napi]
    pub fn save_sync(&self)->napi::Result<Buffer> {call(||Ok(live(&self.inner)?.save().into()))}
    #[napi]
    pub fn save(&self)->napi::Result<AsyncTask<SaveTask>> {
        call(||Ok(AsyncTask::new(SaveTask{object:SaveObject::Prepared(live(&self.inner)?.clone())})))
    }
}

#[napi]
pub struct NativeConstraint {inner:Handle<Arc<g::Constraint>>}
#[napi]
impl NativeConstraint {
    #[napi]
    pub fn clone(&self)->napi::Result<Self> {call(||Ok(Self{inner:Handle::new(live(&self.inner)?.clone())}))}
    #[napi]
    pub fn dispose(&self)->napi::Result<()> {call(||self.inner.close())}
    #[napi]
    pub fn start(&self)->napi::Result<NativeState> {call(||Ok(NativeState{inner:Handle::new(core::State::start(live(&self.inner)?.clone())?)}))}
    #[napi(getter)]
    pub fn mask_len(&self)->napi::Result<f64> {call(||Ok(live(&self.inner)?.mask_len() as f64))}
    #[napi]
    pub fn marker_ids(&self)->napi::Result<Uint32Array> {call(||Ok(live(&self.inner)?.marker_ids().to_vec().into()))}
    #[napi]
    pub fn save_sync(&self,external:bool)->napi::Result<Buffer> {
        call(||{let c=live(&self.inner)?;Ok(if external {c.save_without_vocab()?} else {c.save()}.into())})
    }
    #[napi]
    pub fn save(&self,external:bool)->napi::Result<AsyncTask<SaveTask>> {
        call(||Ok(AsyncTask::new(SaveTask{object:SaveObject::Constraint(live(&self.inner)?.clone(),external)})))
    }
}

#[napi]
pub struct NativeState {inner:Handle<core::State>}
#[napi]
impl NativeState {
    #[napi]
    pub fn clone(&self)->napi::Result<Self> {call(||Ok(Self{inner:Handle::new(live(&self.inner)?.try_clone()?)}))}
    #[napi]
    pub fn dispose(&self)->napi::Result<()> {call(||self.inner.close())}
    #[napi(getter)]
    pub fn mask_len(&self)->napi::Result<f64> {call(||Ok(live(&self.inner)?.mask_len() as f64))}
    #[napi]
    pub fn mask(&self)->napi::Result<Uint32Array> {call(||Ok(live(&self.inner)?.mask()?.into()))}
    #[napi]
    pub fn fill_mask(&self,env:Env,output:Unknown<'_>)->napi::Result<()> {
        call(||arrays::with_words(&env,&output,|out|live(&self.inner)?.fill_mask(out)))
    }
    #[napi]
    pub fn commit_token(&self,value:f64)->napi::Result<()> {call(||live(&self.inner)?.commit_token(id(value)?))}
    #[napi]
    pub fn commit_bytes(&self,env:Env,bytes:Unknown<'_>)->napi::Result<()> {
        call(||arrays::with_bytes(&env,&bytes,|bytes|live(&self.inner)?.commit_bytes(bytes)))
    }
    #[napi]
    pub fn is_accepting(&self)->napi::Result<bool> {call(||live(&self.inner)?.is_accepting())}
    #[napi]
    pub fn is_rejected(&self)->napi::Result<bool> {call(||live(&self.inner)?.is_rejected())}
    #[napi]
    pub fn forced_tokens(&self)->napi::Result<Uint32Array> {call(||Ok(live(&self.inner)?.forced_tokens()?.into()))}
    #[napi]
    pub fn marker_events(&self)->napi::Result<Vec<NativeMarkerEvent>> {
        call(||Ok(live(&self.inner)?.marker_events()?.into_iter().map(|e|NativeMarkerEvent{
            id:e.id,byte_offset:BigInt{sign_bit:false,words:vec![e.byte_offset]}
        }).collect()))
    }
}

// Async tasks own every Rust input. No JS memory, handle borrow or callback is
// retained by a worker; mutations/disposal after scheduling cannot change them.
pub struct CompileTask {grammar:g::Grammar<'static>,vocab:g::Vocab,options:g::BuildOptions}
#[napi]
impl Task for CompileTask {
    type Output=Arc<g::Constraint>;
    type JsValue=NativeConstraint;
    fn compute(&mut self)->napi::Result<Self::Output> {call(||Ok(Arc::new(self.grammar.compile_with(&self.vocab,self.options.clone())?)))}
    fn resolve(&mut self,_:Env,value:Self::Output)->napi::Result<Self::JsValue> {Ok(NativeConstraint{inner:Handle::new(value)})}
}
pub struct PrepareTask {grammar:g::Grammar<'static>,vocab:g::Vocab,options:g::BuildOptions}
#[napi]
impl Task for PrepareTask {
    type Output=g::PreparedGrammar;
    type JsValue=NativePreparedGrammar;
    fn compute(&mut self)->napi::Result<Self::Output> {call(||Ok(self.grammar.prepare_with(&self.vocab,self.options.clone())?))}
    fn resolve(&mut self,_:Env,value:Self::Output)->napi::Result<Self::JsValue> {Ok(NativePreparedGrammar{inner:Handle::new(value)})}
}
pub struct LinkTask {prepared:g::PreparedGrammar,options:g::BuildOptions}
#[napi]
impl Task for LinkTask {
    type Output=Arc<g::Constraint>;
    type JsValue=NativeConstraint;
    fn compute(&mut self)->napi::Result<Self::Output> {call(||Ok(Arc::new(self.prepared.link_with(self.options.clone())?)))}
    fn resolve(&mut self,_:Env,value:Self::Output)->napi::Result<Self::JsValue> {Ok(NativeConstraint{inner:Handle::new(value)})}
}
enum SaveObject {Module(g::Module),Constraint(Arc<g::Constraint>,bool),Prepared(g::PreparedGrammar)}
pub struct SaveTask {object:SaveObject}
#[napi]
impl Task for SaveTask {
    type Output=Vec<u8>;
    type JsValue=Buffer;
    fn compute(&mut self)->napi::Result<Self::Output> {
        call(||Ok(match &self.object {
            SaveObject::Constraint(c,false)=>c.save(),SaveObject::Constraint(c,true)=>c.save_without_vocab()?,
            SaveObject::Prepared(p)=>p.save(),SaveObject::Module(m)=>m.save(),
        }))
    }
    fn resolve(&mut self,_:Env,value:Self::Output)->napi::Result<Self::JsValue> {Ok(value.into())}
}
pub struct LoadConstraintTask {bytes:Vec<u8>,vocab:Option<g::Vocab>}
#[napi]
impl Task for LoadConstraintTask {
    type Output=Arc<g::Constraint>;
    type JsValue=NativeConstraint;
    fn compute(&mut self)->napi::Result<Self::Output> {
        call(||{let bytes=std::mem::take(&mut self.bytes);Ok(Arc::new(match &self.vocab {
            Some(v)=>g::Constraint::load_with_vocab(bytes,v)?,None=>g::Constraint::load(bytes)?,
        }))})
    }
    fn resolve(&mut self,_:Env,value:Self::Output)->napi::Result<Self::JsValue> {Ok(NativeConstraint{inner:Handle::new(value)})}
}
pub struct LoadPreparedTask {bytes:Vec<u8>,vocab:Option<g::Vocab>}
#[napi]
impl Task for LoadPreparedTask {
    type Output=g::PreparedGrammar;
    type JsValue=NativePreparedGrammar;
    fn compute(&mut self)->napi::Result<Self::Output> {
        call(||{let bytes=std::mem::take(&mut self.bytes);Ok(match &self.vocab {
            Some(v)=>g::PreparedGrammar::load_with_vocab(bytes,v)?,None=>g::PreparedGrammar::load(bytes)?,
        })})
    }
    fn resolve(&mut self,_:Env,value:Self::Output)->napi::Result<Self::JsValue> {Ok(NativePreparedGrammar{inner:Handle::new(value)})}
}
#[napi]
pub fn load_constraint(env:Env,bytes:Unknown<'_>,vocab:Option<&NativeVocab>)->napi::Result<AsyncTask<LoadConstraintTask>> {
    call(||Ok(AsyncTask::new(LoadConstraintTask{bytes:arrays::bytes(&env,&bytes)?,vocab:vocab.map(|v|live(&v.inner).map(|v|v.clone())).transpose()?})))
}
#[napi]
pub fn load_constraint_sync(env:Env,bytes:Unknown<'_>,vocab:Option<&NativeVocab>)->napi::Result<NativeConstraint> {
    let mut task=call(||Ok(LoadConstraintTask{bytes:arrays::bytes(&env,&bytes)?,vocab:vocab.map(|v|live(&v.inner).map(|v|v.clone())).transpose()?}))?;
    let value=task.compute()?; task.resolve(env,value)
}
#[napi]
pub fn load_prepared(env:Env,bytes:Unknown<'_>,vocab:Option<&NativeVocab>)->napi::Result<AsyncTask<LoadPreparedTask>> {
    call(||Ok(AsyncTask::new(LoadPreparedTask{bytes:arrays::bytes(&env,&bytes)?,vocab:vocab.map(|v|live(&v.inner).map(|v|v.clone())).transpose()?})))
}
#[napi]
pub fn load_prepared_sync(env:Env,bytes:Unknown<'_>,vocab:Option<&NativeVocab>)->napi::Result<NativePreparedGrammar> {
    let mut task=call(||Ok(LoadPreparedTask{bytes:arrays::bytes(&env,&bytes)?,vocab:vocab.map(|v|live(&v.inner).map(|v|v.clone())).transpose()?}))?;
    let value=task.compute()?; task.resolve(env,value)
}
#[napi]
pub fn version()->String {env!("CARGO_PKG_VERSION").to_owned()}

mod module_api;
