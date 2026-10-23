//! Exact, short-circuit original-ID queries, separate from normal mask execution.
//!
//! Positive witnesses are final admissions, not provisional lexer candidates.
//! We may read an existing exact mask, but never construct a vocabulary-wide
//! mask. Prepared vocabulary ordering is reused only if it already exists;
//! otherwise exact individual recognition is the allocation-bounded fallback.
use super::*;
use crate::runtime::state::{MaskCacheData,frontier_is_accepting};
use crate::compiler::constraint_possible_matches::PreparedOrderedTokens;
use smallvec::SmallVec;
use rustc_hash::FxHashSet;
use std::sync::Mutex;

const REJECTION:&str="commit rejected: no valid parser states remain";
const MAX_FORCED:usize=10_000;

#[cfg(test)]
thread_local! { static BYTE_PROOFS: std::cell::Cell<usize> = const {std::cell::Cell::new(0)}; }

#[derive(Debug,PartialEq,Eq)]
pub(crate) enum AtMostOne { Zero, One(u32), MoreThanOne }
#[derive(Debug)]
enum BoundedIds { Exact(SmallVec<[u32;4]>), OverLimit }
#[derive(Clone,Copy)]
struct View<'a> {
    constraint:&'a Constraint,
    state:&'a ParserStateMap,
    terminated:bool,
    cache:Option<&'a Mutex<Option<MaskCacheData>>>,
    generation:u64,
}
impl<'a> View<'a> {
    fn of(state:&'a ConstraintState<'_>)->Self {
        Self {constraint:state.constraint,state:&state.state,terminated:state.terminated,
            cache:Some(&state.mask_cache),generation:state.generation}
    }
    fn stopping(self)->bool {frontier_is_accepting(self.constraint,self.state,self.terminated)}
}

struct Witnesses {
    ids:SmallVec<[u32;4]>,
    large:Option<FxHashSet<u32>>,
    max:usize,
}
impl Witnesses {
    fn new(max:usize)->Self {Self {ids:SmallVec::new(),large:None,max}}
    fn contains(&self,id:u32)->bool {self.large.as_ref().map_or_else(||self.ids.contains(&id),|s|s.contains(&id))}
    /// False proves over-limit. Duplicate original IDs never count twice.
    fn admit(&mut self,id:u32)->crate::Result<bool> {
        if self.contains(id){return Ok(true);}
        // No allocation at all is needed to prove K+1 after collecting K.
        if self.ids.len()==self.max{return Ok(false);}
        self.ids.try_reserve(1).map_err(|_|crate::Error::State("bounded query witness allocation failed".into()))?;
        if self.ids.len()==8 && self.large.is_none() {
            let mut set=FxHashSet::default();
            set.try_reserve(16).map_err(|_|crate::Error::State("bounded query witness allocation failed".into()))?;
            set.extend(self.ids.iter().copied());self.large=Some(set);
        }
        if let Some(set)=&mut self.large {
            set.try_reserve(1).map_err(|_|crate::Error::State("bounded query witness allocation failed".into()))?;
            set.insert(id);
        }
        self.ids.push(id);Ok(true)
    }
}
fn state_error(message:String)->crate::Error {crate::Error::State(format!("bounded token proof: {message}"))}
fn placeholder(c:&Constraint,id:u32)->bool {
    c.special_token_terminals.iter().any(|s|s.token_id==id && c.is_late_grammar_placeholder_terminal(s.terminal_id))
}
fn ordinary_id(view:View<'_>,id:u32)->bool {
    view.constraint.end_tokens.binary_search(&id).is_err() && !placeholder(view.constraint,id)
}
fn empty_bytes_are_events(c:&Constraint)->bool {
    c.has_template_parser() || c.uses_dynamic_runtime() || c.uses_compact_segmented_parser_runtime()
}

/// Same byte recognizer and endpoint normalization as ordinary ID commit, but
/// distinguish genuine rejection from a proof/resource failure.
fn bytes_successor(c:&Constraint,source:&ParserStateMap,buffers:&mut CommitBuffers,bytes:&[u8],endpoint:bool)
    ->crate::Result<Option<ParserStateMap>> {
    #[cfg(test)]
    BYTE_PROOFS.with(|count|count.set(count.get()+1));
    if bytes.is_empty() && endpoint && empty_bytes_are_events(c){return Ok(None);}
    buffers.reset_all();let mut next=source.clone();
    match commit_bytes_impl(c,&mut next,bytes,buffers) {
        Ok(())=>{},Err(message) if message==REJECTION=>return Ok(None),Err(message)=>return Err(state_error(message)),
    }
    if endpoint {
        maybe_normalize_lookahead_invariant_reductions(c,&mut next);
        coalesce_uniform_runtime_source_states(c,&mut next);
    }
    Ok((!next.is_empty()).then_some(next))
}
fn token_successor(c:&Constraint,source:&ParserStateMap,buffers:&mut CommitBuffers,id:u32)
    ->crate::Result<Option<ParserStateMap>> {
    let bytes=token_bytes_for_id(c,id);let special=c.has_special_token_id(id);
    if bytes.is_none() && !special{return Ok(None);}
    buffers.reset_all();let mut next=source.clone();expand_runtime_product_states(c,&mut next);
    let event=special.then(||advance_special_token_paths(c,&next,id)).flatten();
    if let Some(bytes)=bytes.filter(|b|!b.is_empty() || !empty_bytes_are_events(c)) {
        match commit_bytes_impl(c,&mut next,bytes,buffers) {
            Ok(())=>{},Err(message) if message==REJECTION=>{next.clear();buffers.reset_all();},
            Err(message)=>return Err(state_error(message)),
        }
    } else {next.clear();}
    merge_special_token_paths(c,&mut next,event).map_err(state_error)?;
    maybe_normalize_lookahead_invariant_reductions(c,&mut next);
    coalesce_uniform_runtime_source_states(c,&mut next);
    Ok((!next.is_empty()).then_some(next))
}

fn probe_existing_mask(view:View<'_>,witnesses:&mut Witnesses)->crate::Result<Option<bool>> {
    let Some(cache)=view.cache else{return Ok(None)};
    let cache=cache.lock().map_err(|_|state_error("mask cache poisoned".into()))?;
    let Some(data)=cache.as_ref().filter(|d|d.generation==view.generation) else{return Ok(None)};
    // Exact-event contributions and root controls are handled independently:
    // the final empty-spelling domain filter can add exact events to body data.
    for (wi,&word) in data.mask.iter().enumerate() {
        let mut bits=word;
        while bits!=0 {
            let id=(wi*32+bits.trailing_zeros() as usize) as u32;bits&=bits-1;
            if !ordinary_id(view,id){continue;}
            if empty_bytes_are_events(view.constraint)
                && view.constraint.token_bytes_for_id(id).is_some_and(<[u8]>::is_empty){continue;}
            if !witnesses.admit(id)?{return Ok(Some(false));}
        }
    }
    Ok(Some(true))
}

/// A bounded-cost negative certificate for small physical lexer graphs.
///
/// Every ordinary byte, including one reached after epsilon/finalization,
/// ignore reset or scoped CALL/RETURN, must occur on a byte transition of some
/// participating lexer. Taking the union over ALL states and components is
/// therefore conservative; it does not claim that any retained byte is live.
/// Symbolic/virtual or compressed residuals, and large graphs, decline this
/// optimization. No metadata or cache is added to ordinary mask execution.
fn physical_alphabet_bound(root:&Constraint)->[bool;256] {
    const MAX_COMPONENTS:usize=64;
    const MAX_STATES:usize=256;
    const MAX_EDGES:usize=4096;
    let mut pending:SmallVec<[&Constraint;4]>=smallvec::smallvec![root];
    let mut components=0usize;let mut states=0usize;let mut edges=0usize;
    let mut distinct=0usize;let mut result=[false;256];
    while let Some(c)=pending.pop() {
        components+=1;
        if components>MAX_COMPONENTS{return [true;256];}
        let lexer=&c.tokenizer;
        if lexer.has_any_virtual_runtime() || lexer.has_compressed_transition_segments() {
            return [true;256];
        }
        states=states.saturating_add(lexer.num_states() as usize);
        if states>MAX_STATES{return [true;256];}
        for state in 0..lexer.num_states() {
            for (byte,_) in lexer.transitions_from(state) {
                edges+=1;
                if edges>MAX_EDGES{return [true;256];}
                if !result[byte as usize] {
                    result[byte as usize]=true;distinct+=1;
                    if distinct==256{return result;}
                }
            }
        }
        if let Some(overlay)=&c.static_dynamic_overlay {
            if components+pending.len()+overlay.segmented_parser_components.len()>MAX_COMPONENTS {
                return [true;256];
            }
            pending.extend(overlay.segmented_parser_components.iter().map(|p|p.constraint.as_ref()));
        }
    }
    result
}

/// Use the tighter live physical frontier when it cannot finalize/reset. At a
/// boundary, the bounded whole-alphabet certificate still excludes impossible
/// byte groups without incorrectly treating a component-local state as global.
fn first_byte_bound(view:View<'_>)->[bool;256] {
    let c=view.constraint;
    if c.uses_compact_segmented_parser_runtime(){return physical_alphabet_bound(c);}
    let lexer=&c.tokenizer;let mut result=[false;256];
    for &state in view.state.keys() {
        if state>=lexer.num_states() || lexer.state_has_epsilon_transitions(state)
            || !lexer.matched_terminal_bitset(state).is_empty(){return physical_alphabet_bound(c);}
        for (byte,_) in lexer.transitions_from(state){result[byte as usize]=true;}
    }
    result
}

/// Skip whole sorted root ranges using the physical first-byte upper bound.
/// In a singleton literal state this jumps directly to its one possible byte,
/// rather than cloning/replaying a parser state for every root byte in vocab.
fn ordered_skip_impossible_first_bytes(ordered:&PreparedOrderedTokens,mut lo:usize,hi:usize,bound:&[bool;256])->usize {
    while lo<hi {
        let Some(&first)=ordered.bytes(lo).first() else{return lo};
        if bound[first as usize]{return lo;}
        let Some(next)=((first as usize+1)..256).find(|&b|bound[b]) else{return hi};
        let mut left=lo+1;let mut right=hi;
        while left<right {
            let mid=left+(right-left)/2;
            if ordered.bytes(mid).first().is_none_or(|&b|(b as usize)<next){left=mid+1;}else{right=mid;}
        }
        lo=left;
    }
    hi
}

fn probe_ordered(view:View<'_>,ordered:&PreparedOrderedTokens,buffers:&mut CommitBuffers,w:&mut Witnesses)->crate::Result<bool> {
    if ordered.len()==0{return Ok(true);}
    // Iterative radix traversal: at most one remainder frame per prefix.
    // Deep vocabulary prefixes cannot overflow the native thread stack.
    let initial_bound=first_byte_bound(view);
    let mut pending=vec![(0usize,ordered.len(),0usize,view.state.clone())];
    while let Some((mut lo,hi,consumed,source))=pending.pop() {
        if consumed==0 {
            lo=ordered_skip_impossible_first_bytes(ordered,lo,hi,&initial_bound);
            if lo==hi{continue;}
        }
        let first=ordered.bytes(lo);let last=ordered.bytes(hi-1);
        let mut common=consumed;
        while common<first.len().min(last.len()) && first[common]==last[common]{common+=1;}
        let frontier=if common==consumed {source} else {
            let Some(next)=bytes_successor(view.constraint,&source,buffers,&first[consumed..common],false)? else {continue};
            next
        };
        let mut start=lo;
        if first.len()==common {
            let aliases=ordered.originals(lo);
            let useful=aliases.iter().any(|&id|ordinary_id(view,id)&&!w.contains(id));
            // Validate the endpoint through the ordinary full-token byte
            // recognizer once. All byte-spelling aliases share that proof;
            // their extra exact-event interpretations were already considered.
            if useful && bytes_successor(view.constraint,view.state,buffers,first,true)?.is_some() {
                for &id in aliases {
                    if ordinary_id(view,id) && !w.admit(id)? {return Ok(false);}
                }
            }
            start+=1;
        }
        if start<hi {
            let end=ordered.child_end(start,hi,common);
            if end<hi{pending.push((end,hi,common,frontier.clone()));}
            pending.push((start,end,common,frontier));
        }
    }
    Ok(true)
}
/// Walk an already-materialized ordinary dynamic vocabulary directly. Its
/// canonical endpoints merge byte-identical aliases only. Grammar-quotient
/// representatives are deliberately excluded: their byte spellings need not
/// be interchangeable in an arbitrary full recognition coordinate.
fn probe_dynamic_trie(view:View<'_>,vocab:&crate::runtime::DynamicMaskVocab,buffers:&mut CommitBuffers,w:&mut Witnesses)->crate::Result<bool> {
    let trie=&vocab.trie;
    if trie.node_count()==0{return Ok(true);}
    let initial_bound=first_byte_bound(view);
    // (node, next child, endpoint pending, source after this prefix). Only one
    // continuation per live path is retained; rejected siblings allocate none.
    let mut pending=vec![(0u32,0usize,true,view.state.clone())];
    while let Some((node,mut child_index,endpoint_pending,source))=pending.pop() {
        if endpoint_pending {
            if let Some(canonical)=trie.node(node).token_id {
                if let Some(aliases)=vocab.token_ids(canonical) {
                    let useful=aliases.iter().copied().find(|&id|ordinary_id(view,id)&&!w.contains(id));
                    if let Some(bytes)=useful.and_then(|id|view.constraint.token_bytes_for_id(id)) {
                        // Prefix liveness alone is not a final token witness.
                        if bytes_successor(view.constraint,view.state,buffers,bytes,true)?.is_some() {
                            for &id in aliases {if ordinary_id(view,id)&&!w.admit(id)?{return Ok(false);}}
                        }
                    }
                }
            }
        }
        let children=trie.children(node);
        while let Some(edge)=children.get(child_index) {
            child_index+=1;
            let bytes=trie.edge_bytes(edge);
            if trie.node(node).prefix_byte_len==0
                && bytes.first().is_some_and(|&byte|!initial_bound[byte as usize]){continue;}
            let next=if bytes.is_empty(){Some(source.clone())}
                else{bytes_successor(view.constraint,&source,buffers,bytes,false)?};
            if let Some(next)=next {
                if child_index<children.len(){pending.push((node,child_index,false,source));}
                pending.push((edge.child,0,true,next));break;
            }
        }
    }
    Ok(true)
}

fn eligible_byte_id(view:View<'_>,id:u32)->bool {
    ordinary_id(view,id) && view.constraint.token_bytes_for_id(id)
        .is_some_and(|b|!b.is_empty()||!empty_bytes_are_events(view.constraint))
}

/// Visit original-ID members of the final static class without expanding a
/// loaded vocabulary map or producing a complete mask. The already-stored
/// sparse output fragments are just another representation of the same class.
fn visit_static_class(c:&Constraint,class:usize,visit:&mut impl FnMut(u32)->crate::Result<bool>)->crate::Result<bool> {
    if let Some(ids)=c.internal_token_to_tokens.get(class) {
        for &id in ids {if !visit(id)?{return Ok(false);}}
        return Ok(true);
    }
    if let Some(ids)=c.deferred_internal_token_to_tokens.get().and_then(|v|v.get(class)) {
        for &id in ids {if !visit(id)?{return Ok(false);}}
        return Ok(true);
    }
    let mut word=|index:u16,mut bits:u32|->crate::Result<bool>{
        while bits!=0 {
            let id=u32::from(index)*32+bits.trailing_zeros();bits&=bits-1;
            if !visit(id)?{return Ok(false);}
        }
        Ok(true)
    };
    if let (Some(&lo),Some(&hi))=(c.internal_token_buf_offsets.get(class),c.internal_token_buf_offsets.get(class+1)) {
        if let Some(entries)=c.internal_token_buf_flat.get(lo as usize..hi as usize) {
            for entry in entries {if !word(entry.word_idx,entry.mask)?{return Ok(false);}}
            return Ok(true);
        }
        if let Some(backed)=&c.backed_internal_token_buf_flat {
            if let Some(entries)=backed.slice(lo as usize,hi as usize) {
                for entry in entries {if !word(entry.word_idx,entry.mask)?{return Ok(false);}}
                return Ok(true);
            }
            // Unaligned artifact backing cannot yield a borrowed native slice.
            // Read one existing sparse entry at a time, so cutoff still stops.
            for index in lo as usize..hi as usize {
                let mut entry=None;backed.for_each_range(index,index+1,|w,b|entry=Some((w,b)));
                let (w,b)=entry.ok_or_else(||state_error("missing class member entry".into()))?;
                if !word(w,b)?{return Ok(false);}
            }
            return Ok(true);
        }
    }
    if let Some(entries)=c.internal_token_buf_masks.get(class) {
        for &(index,bits) in entries {if !word(index,bits)?{return Ok(false);}}
        return Ok(true);
    }
    Err(state_error("final class has no original-ID members".into()))
}

/// Only a quotient belonging to THIS complete standalone grammar is used.
/// These final classes are precisely what normal masking expands to original
/// IDs. Exact-event/control/empty-byte exceptions remain individually filtered.
/// A tooling partition or an embedded child's local quotient is not sufficient.
fn probe_grammar_classes(view:View<'_>,buffers:&mut CommitBuffers,w:&mut Witnesses)->crate::Result<Option<bool>> {
    let c=view.constraint;
    if c.static_dynamic_overlay.is_some()
        || c.template_parser.as_ref().is_some_and(|p|p.composition.is_some()){return Ok(None);}
    let dynamic=&c.dynamic_mask_vocab;
    let quotient=dynamic.is_initialized()&&dynamic.is_grammar_quotiented();
    let count=if quotient {dynamic.canonical_token_count()}
        else if !c.uses_dynamic_runtime() && c.has_original_token_map() {
            if !c.internal_token_to_tokens.is_empty(){c.internal_token_to_tokens.len()}
            else if c.internal_token_buf_offsets.len()>1 {c.internal_token_buf_offsets.len()-1}
            else {c.internal_token_buf_masks.len()}
        } else {return Ok(None)};
    // A cost-based structural gate, independent of source grammar families.
    // Trie traversal is preferable to scanning hundreds of classes for K=1.
    if count==0 || count>64 || count.saturating_mul(4)>=c.token_bytes_count(){return Ok(None);}
    let first=first_byte_bound(view);
    for class in 0..count {
        let mut representative=None;
        if quotient {
            if let Some(ids)=dynamic.token_ids(class as u32) {
                representative=ids.iter().copied().find(|&id|eligible_byte_id(view,id)&&!w.contains(id));
            }
        } else {
            visit_static_class(c,class,&mut |id|{
                if eligible_byte_id(view,id)&&!w.contains(id){representative=Some(id);Ok(false)}else{Ok(true)}
            })?;
        }
        let Some(id)=representative else{continue};
        let bytes=c.token_bytes_for_id(id).expect("eligible representative has bytes");
        // The same conservative physical bound used by the trie can reject a
        // representative without cloning/advancing the parser. It says nothing
        // positive: an allowed first byte still needs the full endpoint proof.
        if bytes.first().is_some_and(|&byte|!first[byte as usize]){continue;}
        // Class equality is about final grammar-visible byte admission, not
        // lexer prefix liveness and not equality of original model token IDs.
        if bytes_successor(c,view.state,buffers,bytes,true)?.is_none(){continue;}
        if quotient {
            for &id in dynamic.token_ids(class as u32).unwrap_or_default() {
                if eligible_byte_id(view,id)&&!w.admit(id)?{return Ok(Some(false));}
            }
        } else if !visit_static_class(c,class,&mut |id|{
            if eligible_byte_id(view,id){w.admit(id)}else{Ok(true)}
        })? {return Ok(Some(false));}
    }
    Ok(Some(true))
}

fn probe_owner(view:View<'_>,w:&mut Witnesses,scratch:Option<&mut CommitBuffers>)->crate::Result<bool> {
    if view.terminated || view.state.is_empty(){return Ok(true);}
    let c=view.constraint;
    if !c.end_tokens.is_empty() && view.stopping() {
        for &id in c.end_tokens.iter(){if !w.admit(id)?{return Ok(false);}}
    }
    if !c.special_token_terminals.is_empty() {
        let mut expanded=view.state.clone();expand_runtime_product_states(c,&mut expanded);
        for special in &c.special_token_terminals {
            if c.is_late_grammar_placeholder_terminal(special.terminal_id){continue;}
            let count=u64::from(special.token_id_end)-u64::from(special.token_id)+1;
            if count==c.end_tokens.iter().filter(|&&id|special.contains(id)).count() as u64 {continue;}
            if advance_special_terminal_paths(c,&expanded,special.terminal_id).is_some_and(|gss|!gss.is_empty()) {
                for id in special.token_id..=special.token_id_end {
                    if ordinary_id(view,id) && !w.admit(id)? {return Ok(false);}
                }
            }
        }
    }
    if let Some(done)=probe_existing_mask(view,w)? {return Ok(done);}
    let mut owned_buffers;
    let buffers=match scratch {Some(buffers)=>buffers,None=>{
        // A probe needs the same exact commit semantics, not the eager
        // per-sequence allocation policy (128 branch stacks and up to 256
        // spare GSS nodes). The existing lean constructor retains all limits;
        // ordinary commit paths initialize queues/stacks lazily when required.
        owned_buffers=CommitBuffers::for_mask_only_shadow();&mut owned_buffers
    }};
    if let Some(done)=probe_grammar_classes(view,buffers,w)?{return Ok(done);}
    let dynamic=if c.dynamic_mask_vocab.is_initialized() && !c.dynamic_mask_vocab.is_grammar_quotiented() {
        Some(&c.dynamic_mask_vocab)
    } else {
        c.lazy_dynamic_mask_vocab.get().filter(|v|v.is_initialized()&&!v.is_grammar_quotiented())
    };
    if let Some(vocab)=dynamic{return probe_dynamic_trie(view,vocab,buffers,w);}
    if let Some(ordered)=c.late_bind_vocab.get().and_then(crate::compiler::constraint_possible_matches::prepared_ordered_tokens) {
        return probe_ordered(view,&ordered,buffers,w);
    }
    // No setup/index work is charged to a cold query. Enumerate real original
    // IDs, not holes up to max_id, and validate through authoritative commit.
    let first=first_byte_bound(view);
    let mut first_proofs=[None;256];
    let mut previous:Option<(&[u8],bool)>=None;
    for (id,bytes) in c.token_bytes_iter() {
        if !ordinary_id(view,id)||w.contains(id){continue;}
        if let Some(&b)=bytes.first() {
            if !first[b as usize]{continue;}
            let viable=match first_proofs[b as usize] {
                Some(value)=>value,
                None=>{
                    let value=bytes_successor(c,view.state,buffers,&[b],false)?.is_some();
                    first_proofs[b as usize]=Some(value);value
                },
            };
            if !viable{continue;}
        }
        let admitted=if let Some((old,result))=previous.filter(|(old,_)|*old==bytes) {let _=old;result}
            else {bytes_successor(c,view.state,buffers,bytes,true)?.is_some()};
        previous=Some((bytes,admitted));
        if admitted && !w.admit(id)? {return Ok(false);}
    }
    Ok(true)
}
fn probe_views<'a>(views:impl IntoIterator<Item=View<'a>>,max:usize)->crate::Result<BoundedIds> {
    let mut witnesses=Witnesses::new(max);
    for view in views {if !probe_owner(view,&mut witnesses,None)?{return Ok(BoundedIds::OverLimit);}}
    Ok(BoundedIds::Exact(witnesses.ids))
}
fn one<'a>(views:impl IntoIterator<Item=View<'a>>)->crate::Result<AtMostOne> {
    Ok(match probe_views(views,1)? {
        BoundedIds::OverLimit=>AtMostOne::MoreThanOne,
        BoundedIds::Exact(ids)=>ids.first().copied().map_or(AtMostOne::Zero,AtMostOne::One),
    })
}
fn resource_guard<T>(f:impl FnOnce()->crate::Result<T>)->crate::Result<T> {
    glrmask_invariant::__private::catch_compilation_resource_limit(f).map_err(state_error)?
}

impl ConstraintState<'_> {
    /// Private exact cardinality query. No full mask is built for an over-limit
    /// result; sparse original IDs are still counted individually.
    pub(crate) fn mask_if_at_most(&self,max_allowed:usize)->crate::Result<Option<Vec<u32>>> {
        resource_guard(|| {
            let BoundedIds::Exact(ids)=probe_views([View::of(self)],max_allowed)? else{return Ok(None)};
            let mut mask=Vec::new();
            mask.try_reserve_exact(self.constraint.mask_len()).map_err(|_|state_error("mask allocation failed".into()))?;
            mask.resize(self.constraint.mask_len(),0);
            for id in ids {mask[id as usize/32]|=1<<(id%32);}
            Ok(Some(mask))
        })
    }
    pub(crate) fn probe_one_token(&self)->crate::Result<AtMostOne> {resource_guard(||one([View::of(self)]))}
}

/// Read-only locally inevitable IDs, preserving all correlated successors.
/// Probe before allocating the mutable private cursor; most states are choices.
pub(crate) fn forced_tokens(states:&[ConstraintState<'_>])->crate::Result<Vec<u32>> {
    resource_guard(|| {
        if states.is_empty() || states.iter().any(ConstraintState::is_accepting){return Ok(Vec::new());}
        let AtMostOne::One(mut id)=one(states.iter().map(View::of))? else{return Ok(Vec::new())};
        struct Cursor<'a>{constraint:&'a Constraint,state:ParserStateMap,buffers:CommitBuffers}
        let mut cursors=states.iter().filter(|s|!s.is_rejected()&&!s.terminated)
            .map(|s|Cursor {constraint:s.constraint,state:s.state.clone(),buffers:CommitBuffers::for_mask_only_shadow()})
            .collect::<Vec<_>>();
        let mut output=Vec::new();
        loop {
            let mut live=false;
            for cursor in &mut cursors {
                if cursor.state.is_empty(){continue;}
                match token_successor(cursor.constraint,&cursor.state,&mut cursor.buffers,id)? {
                    Some(next)=>{cursor.state=next;live=true;},None=>cursor.state.clear(),
                }
            }
            if !live{return Err(state_error("singleton witness had no complete token successor".into()));}
            output.push(id);
            if output.len()==MAX_FORCED || cursors.iter().any(|s|frontier_is_accepting(s.constraint,&s.state,false)){break;}
            let mut witnesses=Witnesses::new(1);
            let mut over_limit=false;
            for cursor in &mut cursors {
                let view=View {constraint:cursor.constraint,state:&cursor.state,terminated:false,cache:None,generation:0};
                if !probe_owner(view,&mut witnesses,Some(&mut cursor.buffers))?{over_limit=true;break;}
            }
            if over_limit {break;}
            match witnesses.ids.first().copied(){Some(next)=>id=next,None=>break}
        }
        Ok(output)
    })
}


pub(crate) fn mask_for_states_at_most(states:&[ConstraintState<'_>],max_allowed:usize)->crate::Result<Option<Vec<u32>>> {
    resource_guard(|| {
        let BoundedIds::Exact(ids)=probe_views(states.iter().map(View::of),max_allowed)? else{return Ok(None)};
        let width=states.iter().map(|s|s.constraint.mask_len()).max().unwrap_or(0);
        let mut mask=Vec::new();mask.try_reserve_exact(width).map_err(|_|state_error("mask allocation failed".into()))?;
        mask.resize(width,0);for id in ids {mask[id as usize/32]|=1<<(id%32);}Ok(Some(mask))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BuildOptions,Grammar,MaskStrategy,TokenSet,Vocab};

    const MODES:[MaskStrategy;4]=[MaskStrategy::Auto,MaskStrategy::TokenTrie,MaskStrategy::TokenClassTrie,MaskStrategy::ParserDwa];
    fn ids(mask:&[u32])->Vec<u32> {
        mask.iter().enumerate().flat_map(|(wi,&word)| (0..32).filter_map(move |bit|
            (word&(1<<bit)!=0).then_some((wi*32+bit) as u32))).collect()
    }
    fn inspect(state:&ConstraintState<'_>,context:&str) {
        // Cold queries must be evaluated BEFORE asking the full mask oracle.
        let initial=state.state.clone();let accepting=state.is_accepting();
        let limits=[0,1,2,3,8,usize::MAX];
        let cold=limits.map(|k|state.mask_if_at_most(k).unwrap_or_else(|e|panic!("{context}: K={k}: {e}")));
        let forced=forced_tokens(std::slice::from_ref(state))
            .unwrap_or_else(|e|panic!("{context}: strict force: {e}"));
        let full=state.mask();let allowed=ids(&full);
        for ((k,result),warm) in limits.into_iter().zip(cold).zip(limits.map(|k|state.mask_if_at_most(k).unwrap())) {
            let expected=(allowed.len()<=k).then(||full.clone());
            assert_eq!(result,expected,"cold {context}, K={k}, ids={allowed:?}");
            assert_eq!(warm,expected,"warm {context}, K={k}, ids={allowed:?}");
        }
        assert_eq!(forced,state.forced_token_ids_reference(),"strict forcing {context}");
        let mut cursor=state.clone();
        for token in forced {
            assert!(!cursor.is_accepting(),"forced past stop: {context}");
            assert_eq!(ids(&cursor.mask()),[token],"not singleton: {context}");
            cursor.commit_token(token).unwrap();
        }
        assert_eq!(state.state,initial,"query mutated parser state: {context}");
        assert_eq!(state.is_accepting(),accepting);
    }
    fn explore(c:&Constraint,label:&str) {
        let mut queue=vec![(c.start(),Vec::<u32>::new())];let mut n=0;
        while let Some((s,path))=queue.pop() {
            inspect(&s,&format!("{label}: {path:?}"));n+=1;
            if path.len()==4 || n>120{continue;}
            for id in ids(&s.mask()).into_iter().take(8) {
                let mut next=s.clone();next.commit_token(id).unwrap();let mut p=path.clone();p.push(id);queue.push((next,p));
            }
        }
    }
    #[test]
    fn finite_byte_languages_cold_warm_and_loaded_match_full_mask_and_forcing() {
        let v=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(3,b"ab".to_vec()),(4,b"aa".to_vec()),
            (9,b"ab".to_vec()),(13,b"ba".to_vec()),(69,b"aba".to_vec()),(128,Vec::new())]);
        for body in ["\"ab\"","\"a\" | \"ab\" | \"aba\"","eps | \"ab\"","/a+/ \"b\"","/a|ab/ \"a\"?","/a*/ \"b\"","(/a+/ & /aa*/ ) \"b\"","(/a+/ - /aaa+/) \"b\""] {
            let source=format!("start root nt root = {body}");
            for mode in MODES {
                let c=Grammar::from_glrm(&source).compile_with_build_options(&v,BuildOptions::default().optimization(mode)).unwrap();
                explore(&c,&format!("{mode:?} {body}"));
                let loaded=Constraint::load(c.save()).unwrap();explore(&loaded,&format!("loaded {mode:?} {body}"));
            }
        }
    }
    #[test]
    fn exact_events_aliases_root_controls_and_empty_spellings() {
        let v=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"a".to_vec()),
            (3,b"ab".to_vec()),(17,Vec::new())]).with_special_tokens([29,30,31,4096]);
        for body in ["M \"a\"", "M | \"ab\"", "M \"b\" | \"ab\"", "M? \"ab\"", "M M?"] {
            let source=format!("extern token M start root nt root={body}");
            for tokens in [vec![0],vec![17],vec![29],vec![29,30,31],vec![0,17,29,4096]] {
                for mode in MODES {
                    let mut g=Grammar::from_glrm(&source);g.bind("M",TokenSet::new(tokens.clone())).unwrap();
                    let c=g.compile_with_build_options(&v,BuildOptions::default().optimization(mode).end_tokens([4096])).unwrap();
                    explore(&c,&format!("{mode:?} {body} {tokens:?}"));
                }
            }
        }
    }
    #[test]
    fn composed_and_partial_utf8_frontiers_match_final_admission() {
        let v=Vocab::new(vec![(0,b"P".to_vec()),(1,b"a".to_vec()),(2,b" ".to_vec()),
            (3,b" X".to_vec()),(4,b"X".to_vec()),(5,b"Pa X".to_vec()),(6,vec![0xc3]),(7,vec![0xa9]),
            (8,vec![0xc3,0xa9]),(10,b"x".to_vec())]).with_special_tokens([9]);
        let mut child=Grammar::from_glrm("extern token M start item ignore SPACE t SPACE=/ +/ nt item=\"a\" M?");
        child.bind("M",TokenSet::new([9])).unwrap();
        let mut p=Grammar::from_glrm("extern grammar C start root nt root=\"P\" C \" X\"").prepare(&v).unwrap();
        p.bind("C",child.prepare(&v).unwrap()).unwrap();
        for mode in MODES {
            explore(&p.link_with(BuildOptions::default().optimization(mode)).unwrap(),&format!("composed {mode:?}"));
            let c=Grammar::from_glrm("start root nt root=\"\\u{e9}x\"").compile_with_build_options(&v,BuildOptions::default().optimization(mode)).unwrap();
            explore(&c,&format!("unicode {mode:?}"));
        }
    }
    #[test]
    fn union_counts_distinct_original_ids_not_owner_occurrences() {
        let v=Vocab::new(vec![(17,b"a".to_vec()),(29,b"b".to_vec()),(33,b"c".to_vec())]);
        for mode in MODES {
            let a=Grammar::from_glrm("start r nt r=\"ab\"").compile_with_build_options(&v,BuildOptions::default().optimization(mode)).unwrap();
            let b=Grammar::from_glrm("start r nt r=\"ac\"").compile_with_build_options(&v,BuildOptions::default().optimization(mode)).unwrap();
            let states=[a.start(),b.start()];
            assert_eq!(one(states.iter().map(View::of)).unwrap(),AtMostOne::One(17));
            assert_eq!(forced_tokens(&states).unwrap(),[17]);
            let exact=mask_for_states_at_most(&states,1).unwrap().unwrap();assert_eq!(ids(&exact),[17]);
            assert!(mask_for_states_at_most(&states,0).unwrap().is_none());
        }
    }
    #[test]
    fn ambiguous_query_stops_after_two_final_witnesses_without_filling_cache() {
        let v=Vocab::new((0..4096u32).map(|i|(i,format!("a{i:04}").into_bytes())).collect());
        let c=Grammar::from_glrm("start r nt r=/a[0-9]{4}/").compile_with_build_options(&v,BuildOptions::default().optimization(MaskStrategy::TokenTrie)).unwrap();
        let s=c.start();let before=s.mask_cache.lock().unwrap().is_some();
        BYTE_PROOFS.with(|n|n.set(0));assert_eq!(s.probe_one_token().unwrap(),AtMostOne::MoreThanOne);
        let examined=BYTE_PROOFS.with(|n|n.get());assert!(examined<40,"performed {examined} byte proofs for a two-witness query");
        assert_eq!(s.mask_cache.lock().unwrap().is_some(),before,"query populated the full mask cache");
    }

    #[test]
    fn narrow_alphabets_prune_scoped_and_finalizing_frontiers_without_full_masks() {
        let mut entries=(0..256u32).map(|id|(id,vec![id as u8])).collect::<Vec<_>>();
        for id in 256..8192 {entries.push((id,format!("word{id:07}").into_bytes()));}
        let v=Vocab::new(entries);
        for (source,expected) in [
            ("grammar CHILD { start inner nt inner=\"!?\" } start root nt root=CHILD \"!\"",vec![33,63,33]),
            ("start root nt root=\"!!!!!!!!!!!!!!!!!!!!!!!!\"",vec![33;24]),
        ] {
            for mode in MODES {
                let c=Grammar::from_glrm(source).compile_with_build_options(&v,BuildOptions::default().optimization(mode)).unwrap();
                for loaded in [false,true] {
                    let c=if loaded{Constraint::load(c.save()).unwrap()}else{c.clone()};
                    let state=c.start();
                    let cached_before=state.mask_cache.lock().unwrap().as_ref()
                        .map(|m|(m.generation,m.mask.clone()));
                    BYTE_PROOFS.with(|n|n.set(0));
                    assert_eq!(forced_tokens(std::slice::from_ref(&state)).unwrap(),expected,"{mode:?}, loaded={loaded}");
                    let work=BYTE_PROOFS.with(|n|n.get());
                    assert!(work<expected.len()*12,"{mode:?}, loaded={loaded}: {work} byte proofs for {} IDs",expected.len());
                    assert_eq!(state.mask_cache.lock().unwrap().as_ref()
                        .map(|m|(m.generation,m.mask.clone())),cached_before,
                        "query changed the pre-existing mask cache: {mode:?}, loaded={loaded}");
                    assert_eq!(state.forced_token_ids_reference(),expected);
                }
            }
        }
    }

    #[test]
    fn physical_alphabet_union_keeps_parent_suffix_child_ignore_and_nullable_returns() {
        let v=Vocab::new((0..256u32).map(|id|(id,vec![id as u8])).collect());
        for source in [
            "grammar C { start r ignore WS t WS=/ +/ nt r=\"a\"? } start r nt r=C \"Z\"",
            "grammar C { start r ignore WS t WS=/\\t+/ nt r=\"a\" | \"ab\" } start r ignore WS t WS=/ +/ nt r=\"P\" C \"Z\"",
            "grammar C { start r nt r=\"(\" r \")\" | eps } start r nt r=C \"Z\"",
        ] {
            for mode in MODES {
                let c=Grammar::from_glrm(source).compile_with_build_options(&v,BuildOptions::default().optimization(mode)).unwrap();
                explore(&c,&format!("alphabet boundary {mode:?}, {source}"));
                let loaded=Constraint::load(c.save()).unwrap();
                explore(&loaded,&format!("loaded alphabet boundary {mode:?}, {source}"));
            }
        }
    }
}


#[cfg(test)]
mod differential_tests {
    use super::*;
    use crate::{BuildOptions, Grammar, MaskStrategy, TokenSet, Vocab};
    const MODES:[MaskStrategy;4]=[MaskStrategy::Auto,MaskStrategy::TokenTrie,MaskStrategy::TokenClassTrie,MaskStrategy::ParserDwa];
    fn allowed(mask:&[u32])->Vec<u32> {
        mask.iter().enumerate().flat_map(|(word,&bits)| (0..32).filter_map(move |bit|
            (bits&(1u32<<bit)!=0).then_some((word*32+bit) as u32))).collect()
    }
    fn check(state:&ConstraintState<'_>,tag:&str) {
        let expected=state.clone().mask();let ids=allowed(&expected);
        let cached_before=state.mask_cache.lock().unwrap().as_ref().map(|c|(c.generation,c.mask.clone()));
        for k in [0,1,2,3,ids.len().saturating_sub(1),ids.len(),ids.len()+1,usize::MAX] {
            let actual=state.mask_if_at_most(k).unwrap_or_else(|e|panic!("{tag}, K={k}: {e}"));
            if ids.len()<=k {assert_eq!(actual,Some(expected.clone()),"{tag}, K={k}, ids={ids:?}");}
            else {assert!(actual.is_none(),"{tag}, K={k}, actual={actual:?}, ids={ids:?}");}
        }
        let forced=forced_tokens(std::slice::from_ref(state))
            .unwrap_or_else(|e|panic!("{tag} strict force: {e}"));
        assert_eq!(forced,state.clone().forced_token_ids_reference(),"{tag} strict forcing");
        assert_eq!(state.mask_cache.lock().unwrap().as_ref().map(|c|(c.generation,c.mask.clone())),cached_before,
            "{tag}: bounded query must not create or overwrite normal mask cache");
        let mut replay=state.clone();
        for id in forced {
            assert!(!replay.is_accepting(),"{tag}: forced past root acceptance");
            assert_eq!(allowed(&replay.mask()),[id],"{tag}: forced ID was not singleton");
            replay.commit_token(id).unwrap();
        }
        // Also test reading an existing exact/body cache and applying the same
        // root-control / exact-event / empty-spelling final domain rules.
        let warm=state.clone();let _=warm.mask();
        assert_eq!(warm.mask_if_at_most(ids.len()).unwrap(),Some(expected.clone()),"{tag} warm");
        if !ids.is_empty(){assert!(warm.mask_if_at_most(ids.len()-1).unwrap().is_none(),"{tag} warm cutoff");}
        assert_eq!(state.mask(),expected,"{tag}: state changed");
    }
    fn walk(c:&Constraint,tag:&str,depth:usize) {
        let mut states=vec![(c.start(),Vec::<u32>::new())];
        while let Some((s,trace))=states.pop() {
            check(&s,&format!("{tag} trace={trace:?}"));
            if trace.len()==depth{continue;}
            for id in allowed(&s.mask()).into_iter().take(12) {
                let mut child=s.clone();child.commit_token(id).unwrap();
                let mut t=trace.clone();t.push(id);states.push((child,t));
            }
        }
    }
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow (~15s); run with --features slow-tests"
    )]
    fn bounded_masks_and_strict_forcing_match_normal_masks_across_all_modes() {
        let v=Vocab::new(vec![
            (0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"ab".to_vec()),(3,b"abc".to_vec()),
            (4,b"a".to_vec()),(5,b" ".to_vec()),(6,b"c".to_vec()),(7,Vec::new()),
            (8,b"x".to_vec()),(9,b"(".to_vec()),(10,b")".to_vec()),(11,b"()".to_vec()),
            (12,b"a ".to_vec()),(13,b"ab ".to_vec()),(14,b" a".to_vec()),
            (32,vec![0xc3]),(33,vec![0xa9]),(34,"\u{e9}".as_bytes().to_vec()),(67,b"a".to_vec()),
        ]).with_special_tokens([99,130]);
        for source in [
            "start root nt root=\"ab\" \"c\"?",
            "start root nt root=\"a\" | \"ab\" | \"abc\"",
            "start root t WORD=/a|ab/ nt root=WORD \"c\"?",
            "start root t A=/[ab]+/ t B=/ab/ nt root=A \"c\" | B",
            "start root ignore WS t WS=/ +/ nt root=\"ab\"",
            "start root t X=(/a|ab/ - (/a|ab/ b/./s+)) nt root=X \"b\"?",
            "start root nt root=\"(\" root \" )\" | \"()\"",
            "start root nt root=\"\u{e9}\" \"a\"?",
            "start root nt root=eps",
        ] {
            for mode in MODES {
                for eos in [Vec::new(),vec![130],vec![4]] {
                    let c=Grammar::from_glrm(source).compile_with_build_options(&v,BuildOptions::default().optimization(mode).end_tokens(eos.clone())).unwrap();
                    let tag=format!("{source:?}, {mode:?}, end={eos:?}");
                    walk(&c,&tag,2);
                    let loaded=Constraint::load(&c.save()).unwrap();
                    check(&loaded.start(),&format!("{tag}, loaded"));
                }
            }
        }
    }
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow (~11s); run with --features slow-tests"
    )]
    fn bounded_exact_events_aliases_empty_spelling_and_root_controls() {
        let v=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"c".to_vec()),
            (4,b"a".to_vec()),(5,b"ab".to_vec()),(7,Vec::new())]).with_special_tokens([99,130]);
        for body in ["MARK", "MARK \"a\"", "\"ab\" | MARK \"c\"", "\"a\" MARK?", "MARK? \"b\""] {
            let source=format!("extern token MARK start root nt root={body}");
            for set in [vec![],vec![0],vec![4],vec![7],vec![99],vec![99,130],vec![0,4,7,99]] {
                for mode in MODES {
                    let mut g=Grammar::from_glrm(&source);g.bind("MARK",TokenSet::new(set.iter().copied())).unwrap();
                    let c=g.compile_with_build_options(&v,BuildOptions::default().optimization(mode).end_tokens([130])).unwrap();
                    walk(&c,&format!("{body}, set={set:?}, {mode:?}"),2);
                }
            }
        }
    }
    #[test]
    fn bounded_composed_calls_ignore_and_independent_event_lanes() {
        let v=Vocab::new(vec![(0,b"P".to_vec()),(1,b"a".to_vec()),(2,b" ".to_vec()),
            (3,b" X".to_vec()),(4,b"X".to_vec()),(5,b"Pa X".to_vec()),(6,b"\t".to_vec()),
            (7,b"a X".to_vec()),(8,b"a".to_vec())]).with_special_tokens([9]);
        let child=Grammar::from_glrm("extern token MARK start item ignore SPACE t SPACE=/ +/ nt item=\"a\" MARK?").prepare(&v).unwrap();
        let mut parent=Grammar::from_glrm("extern grammar CHILD start root nt root=\"P\" CHILD \" X\"").prepare(&v).unwrap();
        parent.bind("CHILD",&child).unwrap();parent.bind("CHILD::MARK",TokenSet::from(9)).unwrap();
        for mode in MODES {
            let c=parent.link_with(BuildOptions::default().optimization(mode)).unwrap();
            walk(&c,&format!("composed {mode:?}"),2);
            for path in [&[5][..],&[0,1,3][..],&[0,1,9,3][..]] {
                let mut state=c.start();for &id in path {check(&state,&format!("{mode:?}, path={path:?}, next={id}"));state.commit_token(id).unwrap();}
                check(&state,"composed accepting");
            }
        }
    }
    #[test]
    fn bounded_dynamic_owner_union_counts_distinct_original_ids() {
        let v=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"c".to_vec()),(9,b"a".to_vec())]);
        let a=Grammar::from_glrm("start root nt root=\"ab\"").compile(&v).unwrap();
        let b=Grammar::from_glrm("start root nt root=\"ac\"").compile(&v).unwrap();
        let sa=a.start();let sb=b.start();
        assert_eq!(mask_for_states_at_most(&[sa.clone(),sb.clone()],2).unwrap(),Some(sa.mask()));
        assert!(mask_for_states_at_most(&[sa.clone(),sb.clone()],1).unwrap().is_none());
        assert!(forced_tokens(&[sa.clone(),sb.clone()]).unwrap().is_empty());
        let mut sa=sa;let mut sb=sb;sa.commit_token(0).unwrap();sb.commit_token(0).unwrap();
        let mask=mask_for_states_at_most(&[sa.clone(),sb.clone()],2).unwrap().unwrap();assert_eq!(allowed(&mask),[1,2]);
        assert!(mask_for_states_at_most(&[sa,sb],1).unwrap().is_none());
    }

    #[test]
    fn json_schema_lark_and_regular_difference_match_full_mask_at_real_token_boundaries() {
        use std::collections::BTreeSet;
        let schema=r#"{"type":"object","properties":{"a":{"type":"array","items":{"enum":["alpha","β"]},"maxItems":2},"score":{"type":"integer","minimum":-3,"maximum":15},"enabled":{"type":"boolean"}},"required":["a","score","enabled"],"additionalProperties":false}"#;
        let json=r#"{"a": ["alpha", "β"], "score": -2, "enabled": true}"#;
        let lark="start: sum\n?sum: sum \"+\" atom | atom\n?atom: /[0-9]+/ | \"(\" sum \")\"\n%ignore /[ \\t]+/";
        let arithmetic="(17 + 28) + 3";
        let regular="start root t WORD=/[a-z]+/ - /bad/ nt root=\"[\" WORD (\",\" WORD)* \"]\"";
        let list="[alpha,bob]";
        let mut pieces=BTreeSet::new();
        for text in [json,arithmetic,list] {
            let bytes=text.as_bytes();
            for start in 0..bytes.len() {
                for end in start+1..=(start+5).min(bytes.len()) {pieces.insert(bytes[start..end].to_vec());}
            }
        }
        // Include complete invalid shorter words and valid longer extensions:
        // provisional lexer positives must survive subtraction only when final.
        for piece in [&b"bad"[..],&b"bad]"[..],&b"[bad]"[..],&b"badly"[..],&b"[badly]"[..]] {
            pieces.insert(piece.to_vec());
        }
        let mut entries=(0..256u32).map(|id|(id,vec![id as u8])).collect::<Vec<_>>();
        for piece in pieces {entries.push((entries.len() as u32,piece));}
        entries.push((entries.len() as u32,Vec::new()));
        let v=Vocab::new(entries.clone()).with_special_tokens([2000]);
        for (name,grammar,text) in [
            ("json-schema",Grammar::from_json_schema(schema),json),
            ("lark-arithmetic",Grammar::from_lark(lark),arithmetic),
            ("regular-difference",Grammar::from_glrm(regular),list),
        ] {
            for mode in MODES {
                let c=grammar.compile_with_build_options(&v,BuildOptions::default().optimization(mode).end_tokens([2000])).unwrap();
                for loaded in [false,true] {
                    let c=if loaded{Constraint::load(c.save()).unwrap()}else{c.clone()};
                    let mut state=c.start();let mut remaining=text.as_bytes();let mut step=0;
                    while !remaining.is_empty() {
                        let context=format!("frontend {name}, {mode:?}, loaded={loaded}, step={step}");
                        check(&state.clone(),&context);
                        let (&id,bytes)=entries.iter().filter(|(_,b)|!b.is_empty()&&b.len()<=3&&remaining.starts_with(b))
                            .map(|(id,b)|(id,b)).max_by_key(|(id,b)|(b.len(),std::cmp::Reverse(**id))).unwrap();
                        assert!(allowed(&state.mask()).contains(&id),"{context}, next={id}");
                        state.commit_token(id).unwrap();remaining=&remaining[bytes.len()..];step+=1;
                    }
                    assert!(state.is_accepting());check(&state.clone(),"frontend root acceptance");
                    state.commit_token(2000).unwrap();check(&state,"frontend terminated");
                }
            }
        }
    }

}
