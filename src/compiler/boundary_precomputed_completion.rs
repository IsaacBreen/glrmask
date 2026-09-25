//! Persisted, source-certified FIRST-segment observations. Reset/continuation
//! states and all parser labels remain unchanged. No result here admits tokens.
#[allow(dead_code)]
mod prefix_observer;
#[allow(dead_code)]
mod completion_index;
mod prefix_span_support;
mod prefix_uniform;
use self::completion_index::{CompletionContext, UntrustedCompletionIndex};
use self::prefix_observer::{Limits, PrefixObserver, Profile};
use crate::automata::lexer::tokenizer::{Lexer,Tokenizer};
use crate::{Constraint,Vocab};
use glrmask_terminal_dwa::__private::terminal_dwa::scope::BoundaryAnalysisScope;
use super::boundary_first_completion::FirstRefinement;
use std::{sync::Arc,time::Instant};

const ENVELOPE: &[u8;4]=b"CMS6";
const MAX_PART:usize=128*1024*1024;

pub(crate) struct PreparedCompletion {
    context: CompletionContext,
    wire: Arc<[u8]>,
}
impl std::fmt::Debug for PreparedCompletion {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        f.debug_struct("PreparedCompletion").field("states",&self.context.raw_states())
            .field("classes",&self.context.class_count()).field("bytes",&self.wire.len()).finish()
    }
}
impl PreparedCompletion {
    pub(crate) fn matches(&self,source:&Tokenizer)->bool{std::ptr::eq(self.context.source(),source)}
    pub(crate) fn load(source:Arc<Tokenizer>,wire:Arc<[u8]>)->Result<Self,String>{
        let context=CompletionContext::load(source,&wire).map_err(|e|format!("completion index certificate: {e:?}"))?;
        Ok(Self{context,wire})
    }
}

/// The envelope is intentionally nonrecursive, and length checks precede any
/// allocation. Old composition metadata bytes are unchanged when absent.
pub(crate) fn split_envelope(input:&[u8])->Result<(&[u8],Option<&[u8]>),String>{
    if !input.starts_with(ENVELOPE){return Ok((input,None));}
    if input.len()<12{return Err("truncated prepared composition envelope".into());}
    let a=u32::from_le_bytes(input[4..8].try_into().unwrap())as usize;
    let b=u32::from_le_bytes(input[8..12].try_into().unwrap())as usize;
    let end=12usize.checked_add(a).and_then(|x|x.checked_add(b)).ok_or("prepared envelope overflow")?;
    if a>MAX_PART||b>MAX_PART||a==0||b==0||end!=input.len(){return Err("invalid prepared composition envelope lengths".into());}
    let inner=&input[12..12+a];
    if inner.starts_with(ENVELOPE){return Err("nested prepared composition envelopes are forbidden".into());}
    Ok((inner,Some(&input[12+a..])))
}
pub(crate) fn wrap_envelope(inner:Vec<u8>,wire:&[u8])->Result<Vec<u8>,String>{
    let (inner,_) = split_envelope(&inner)?;
    if inner.is_empty()||inner.len()>MAX_PART||wire.is_empty()||wire.len()>MAX_PART{return Err("prepared composition envelope exceeds bounds".into());}
    let mut out=Vec::with_capacity(12+inner.len()+wire.len());out.extend_from_slice(ENVELOPE);
    out.extend_from_slice(&(inner.len()as u32).to_le_bytes());out.extend_from_slice(&(wire.len()as u32).to_le_bytes());
    out.extend_from_slice(inner);out.extend_from_slice(wire);Ok(out)
}
pub(crate) fn saved_wire(constraint:&Constraint)->Option<&[u8]>{
    let p=constraint.boundary_completion_index.as_ref()?;
    p.matches(constraint.composition_tokenizer()).then_some(p.wire.as_ref())
}

/// Explicit independent component preparation. No caller, slot, other
/// component, link seed set, or link-refined vocabulary is an input.
pub(crate) fn prepare_component(constraint:&mut Constraint,vocab:&Vocab)->Result<bool,String>{
    let started=Instant::now();
    if saved_wire(constraint).is_some(){return Ok(true);}
    if !std::ptr::eq(constraint.composition_tokenizer(),constraint.tokenizer.as_ref())
        ||constraint.tokenizer.has_virtual_residual_runtime(){return Ok(false);}
    let (Some(ids),_)=super::boundary_candidates::boundary_candidate_ids(constraint,vocab)else{return Ok(false)};
    if ids.is_empty(){return Ok(false);}
    let mut words=ids.into_iter().map(|id|vocab.entries_map().get(&id).cloned().ok_or("summary token is absent from bound vocabulary".to_owned())).collect::<Result<Vec<_>,_>>()?;
    words.sort_unstable();words.dedup();
    let source=Arc::clone(&constraint.tokenizer);let mut profile=Profile::default();
    let Some(prefix)=PrefixObserver::prepare(source.as_ref(),&words,Limits::default(),&mut profile)else{return Ok(false)};
    let Some(certified)=prefix.certify(source.as_ref())else{return Err("fresh prefix certificate failed".into())};
    let index=match UntrustedCompletionIndex::from_certified_prefix(&certified){Ok(p)=>p,Err(completion_index::Error::ResourceLimit)=>return Ok(false),Err(e)=>return Err(format!("fresh completion index: {e:?}"))};
    let wire:Arc<[u8]>=match index.to_bytes(){Ok(b)=>Arc::from(b),Err(completion_index::Error::ResourceLimit)=>return Ok(false),Err(e)=>return Err(format!("completion index serialization: {e:?}"))};
    let prepared=PreparedCompletion::load(Arc::clone(&source),wire)?;
    let elapsed=started.elapsed().as_secs_f64()*1000.;
    if std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some(){eprintln!("[glrmask/profile][component_completion_prepare] states={} classes={} vocab={} bytes={} total_ms={elapsed:.3} profile={profile:?}",source.num_states(),prepared.context.class_count(),words.len(),prepared.wire.len());}
    constraint.boundary_completion_index=Some(Arc::new(prepared));constraint.serialized_artifact_cache=None;Ok(true)
}

/// Construct this only alongside the link's actual disjoint-union call. The
/// offset is the returned injection for this same immutable source allocation.
/// Expr restoration does not alter that union's byte or label coordinates.
pub(crate) struct PreparedSourceSpan { prepared:Arc<PreparedCompletion>, offset:u32, terminal_offset:u32 }
impl PreparedSourceSpan {
    pub(crate) fn for_component(constraint:&Constraint,offset:u32,terminal_offset:u32)->Option<Self>{
        let prepared=constraint.boundary_completion_index.as_ref()?;
        if !prepared.matches(constraint.composition_tokenizer()){return None;}
        Some(Self{prepared:Arc::clone(prepared),offset,terminal_offset})
    }
    /// Same conservative prefix-seed mask as the raw scanner, restricted to
    /// the caller's selected initial set. Only the existing certified component
    /// index is read; unknown coverage/ownership/topology declines to raw scan.
    pub(crate) fn prefix_seed_support(
        &self, merged:&Tokenizer, vocab:&Vocab, initial:&[bool],
        ownership:&glrmask_terminal_dwa::__private::terminal_dwa::scope::BoundaryOwnership,
        owner:glrmask_terminal_dwa::__private::terminal_dwa::scope::ImmediateComponentId,
    )->Option<Vec<bool>> {
        prefix_uniform::in_verified_uniform_span(
            &self.prepared.context,self.offset,self.terminal_offset,
            merged,vocab,initial,ownership,owner,
        ).map(|(mask,_)|mask)
    }
    pub(crate) fn refine(&self,merged:&Tokenizer,vocab:&Vocab,scope:&BoundaryAnalysisScope)->Option<FirstRefinement>{
        let n=merged.num_states()as usize;let local_n=self.prepared.context.raw_states();let offset=self.offset as usize;
        if !scope.require_crossing()||vocab.is_empty()||vocab.entries_map().values().any(Vec::is_empty)
            ||scope.initial_states().keep_raw().len()!=n||offset.checked_add(local_n)?>n{return None;}
        let mut local=vec![false;local_n];let mut reps=vec![false;n];let mut mapping=(0..n as u32).collect::<Vec<_>>();
        let mut before=0;let mut after=0;
        for(q,&keep)in scope.initial_states().keep_raw().iter().enumerate(){if !keep{continue;}before+=1;
            if q<offset||q>=offset+local_n{return None;}
            let pure=merged.singleton_epsilon_closure(q as u32).iter().all(|&r|merged.matched_terminals_iter(r).chain(merged.possible_future_terminals_iter(r)).all(|t|scope.ownership().owner_of_terminal(t)==Some(scope.start_component())));
            if pure{local[q-offset]=true;}else{reps[q]=true;after+=1;}
        }
        let words=vocab.entries_map().values().cloned().collect::<Vec<_>>();
        let groups=self.prepared.context.query(&local,&words).ok()?;
        for(rep,members)in groups{let r=rep.checked_add(self.offset)?;reps[r as usize]=true;after+=1;
            for q in members{mapping[q as usize+offset]=r;}}
        Some(FirstRefinement{original_to_rep:mapping,representatives:reps,before,after,steps:0,prefixes:0,signature_cells:0})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prepared_envelope_is_bounded_nonrecursive_and_exact(){
        let base=b"CMS5original".to_vec();let wire=b"GCCI0001opaque";
        let encoded=wrap_envelope(base.clone(),wire).unwrap();let(a,b)=split_envelope(&encoded).unwrap();assert_eq!(a,base);assert_eq!(b,Some(wire.as_slice()));
        for n in 4..encoded.len(){assert!(split_envelope(&encoded[..n]).is_err());}
        let mut bad=encoded.clone();bad.push(0);assert!(split_envelope(&bad).is_err());
        let mut huge=encoded.clone();huge[4..8].copy_from_slice(&u32::MAX.to_le_bytes());assert!(split_envelope(&huge).is_err());
        let mut nested=encoded; nested[12..16].copy_from_slice(ENVELOPE);assert!(split_envelope(&nested).is_err());
        assert_eq!(split_envelope(&base).unwrap(),(base.as_slice(),None));
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    use glrmask_terminal_dwa::__private::terminal_dwa::{scope::{BoundaryOwnership,ImmediateComponentId,InitialStateDomain},l1};
    use crate::automata::lexer::ast::{bytes,Expr};
    use crate::automata::lexer::compile::build_regex_monolithic;

    #[test]
    fn independent_component_index_roundtrips_and_matches_raw_first_query() {
        let vocab=Vocab::new(vec![(1,b"abc".to_vec()),(2,b"xyc".to_vec()),(3,b"c!".to_vec()),(4,b"bc!".to_vec()),(5,b"abc!".to_vec()),(6,b"xyc!".to_vec()),(7,b"!".to_vec()),(8,b"a".to_vec()),(9,b"b".to_vec()),(10,b"c".to_vec())]);
        let mut source=Constraint::from_glrm_grammar(r#"start x; nt x ::= "abc" | "xyc";"#,&vocab).unwrap();
        assert!(prepare_component(&mut source,&vocab).unwrap(),"fixture must select component-only preparation");
        let wire=source.save();
        let loaded=Constraint::load(&wire).unwrap();
        let prepared=loaded.boundary_completion_index.as_ref().expect("saved index must be certified at load");
        assert!(prepared.matches(loaded.tokenizer.as_ref()));
        assert_eq!(loaded.save(),wire);
        let (Some(ids),_)=super::super::boundary_candidates::boundary_candidate_ids(&loaded,&vocab)else{panic!("known prepared candidates")};
        assert!(!ids.is_empty());
        let selected=Vocab::new(ids.into_iter().filter(|id|!vocab.entries_map()[id].is_empty()).map(|id|(id,vocab.entries_map()[&id].clone())).collect());
        let other=build_regex_monolithic(&[bytes(b"!")]).into_tokenizer(1,None);
        let nt=loaded.tokenizer.num_terminals();
        let (merged,offsets)=Tokenizer::disjoint_union_with_terminal_offsets(&[(loaded.tokenizer.as_ref(),0),(&other,nt)]);
        let mut seeds=vec![false;merged.num_states()as usize];
        for q in 0..loaded.tokenizer.num_states(){seeds[(q+offsets[0])as usize]=true;}
        let scope=BoundaryAnalysisScope::new(InitialStateDomain::from_mask(seeds.len(),seeds).unwrap(),merged.deterministic_reset_states().into_vec(),Arc::new(BoundaryOwnership::flat(&[0,nt],nt+1).unwrap()),ImmediateComponentId(0),true,None).unwrap();
        let span=PreparedSourceSpan::for_component(&loaded,offsets[0],0).unwrap();
        let actual=span.refine(&merged,&selected,&scope).unwrap();
        let flat=l1::build_flat_transition_table(&merged);
        let expected=super::super::boundary_first_completion::prepare(&merged,&selected,&scope,&flat).unwrap();
        assert_eq!(actual.original_to_rep,expected.original_to_rep);assert_eq!(actual.representatives,expected.representatives);
        let outside=Vocab::new(vec![(42,b"not-in-component-coverage".to_vec())]);assert!(span.refine(&merged,&outside,&scope).is_none());
        let mut changed=loaded.clone();changed.tokenizer=Arc::new(build_regex_monolithic(&[Expr::Epsilon,bytes(b"z")]).into_tokenizer(2,None));
        assert!(saved_wire(&changed).is_none());assert!(PreparedSourceSpan::for_component(&changed,offsets[0],0).is_none());
        assert!(saved_wire(&loaded).is_some());
    }

    #[test]
    fn bounded_index_rejects_wrong_source_and_preserves_clone_identity() {
        let a=Arc::new(build_regex_monolithic(&[bytes(b"abc")]).into_tokenizer(1,None));
        let b=Arc::new(build_regex_monolithic(&[bytes(b"abd")]).into_tokenizer(1,None));
        let words=vec![b"abc!".to_vec(),b"bc!".to_vec(),b"c!".to_vec()];
        let prefix=PrefixObserver::prepare(a.as_ref(),&words,Limits::default(),&mut Profile::default()).unwrap().certify(a.as_ref()).unwrap();
        let index=UntrustedCompletionIndex::from_certified_prefix(&prefix).unwrap();let wire:Arc<[u8]>=Arc::from(index.to_bytes().unwrap());
        let good=PreparedCompletion::load(Arc::clone(&a),Arc::clone(&wire)).unwrap();assert!(good.matches(a.as_ref()));
        assert!(PreparedCompletion::load(b,wire).is_err());
    }
}
