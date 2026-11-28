
//! Experimental direct finite-coefficient specialization of the unchanged
//! terminal event producer. Generic coefficients are never built for emitted
//! transitions. The seeded raw TSID selectors retain their original meanings.
use super::*;
use crate::terminal_dwa::l2p::native_pipeline::post::NativeSink as NWA;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Weight {
    pub(super) bits: [u64; 8],
    pub(super) end: u32,
    pub(super) valid: bool,
}
impl Weight {
 pub(super) fn empty()->Self {Self{bits:[0;8],end:0,valid:true}}
 fn is_empty(&self)->bool {self.valid&&self.bits.iter().all(|&x|x==0)}
 fn from_uniform(rows:std::ops::RangeInclusive<u32>,tokens:RangeSetBlaze<u32>)->Self {
  let mut result=Self{bits:[0;8],end:*rows.end(),valid:*rows.start()==0};
  for range in tokens.ranges() {
   if *range.end()>=512{result.valid=false;continue}
   for id in range{result.bits[id as usize/64]|=1u64<<(id%64);}
  }
  result
 }
 pub(super) fn union(&self,other:&Self)->Self {
  if self.is_empty(){return *other} else if other.is_empty(){return *self}
  let mut result=Self{bits:[0;8],end:self.end,valid:self.valid&&other.valid&&self.end==other.end};
  for k in 0..8{result.bits[k]=self.bits[k]|other.bits[k]}result
 }
}
impl EventCoefficient for Weight {
    fn empty() -> Self { Weight::empty() }
    fn is_empty(&self) -> bool { Weight::is_empty(self) }
    fn union(&self, other: &Self) -> Self { Weight::union(self, other) }

    fn from_uniform(rows: std::ops::RangeInclusive<u32>, tokens: RangeSetBlaze<u32>) -> Self {
        Weight::from_uniform(rows, tokens)
    }

    fn from_reachable(num_tsids: u32, tokens: &RangeSetBlaze<usize>) -> Self {
        if num_tsids == 0 { return Self::empty(); }
        let mut result = Self { bits: [0; 8], end: num_tsids - 1, valid: true };
        for range in tokens.ranges() {
            if *range.end() >= 512 {
                result.valid = false;
                continue;
            }
            for token in range {
                result.bits[token / 64] |= 1u64 << (token % 64);
            }
        }
        result
    }

    fn from_leaf_tokens(num_tsids: u32, tokens: &[u32]) -> Self {
        if num_tsids == 0 { return Self::empty(); }
        let mut result = Self { bits: [0; 8], end: num_tsids - 1, valid: true };
        for &token in tokens {
            if token >= 512 {
                result.valid = false;
            } else {
                result.bits[token as usize / 64] |= 1u64 << (token % 64);
            }
        }
        result
    }

    fn without_token(&self, _num_tsids: u32, token: u32) -> Option<Self> {
        if token >= 512 { return None; }
        let mut result = *self;
        result.bits[token as usize / 64] &= !(1u64 << (token % 64));
        Some(result)
    }
}

impl EventSink for NWA {
    type Coefficient = Weight;
    type LeafIds = FxHashMap<(u32, TerminalID), LeafTokenIds>;
    const NATIVE: bool = true;

    fn states_len(&self) -> usize { NWA::states_len(self) }
    fn add_state(&mut self) -> u32 { NWA::add_state(self) }

    fn emit_epsilon(&mut self, from: u32, to: u32, weight: Weight) {
        self.append_epsilon(from, to, weight.end, weight.valid, &weight.bits);
    }

    fn emit_transition(&mut self, from: u32, label: i32, to: u32, weight: Weight) {
        self.append_transition(from, label, to, weight.end, weight.valid, &weight.bits);
    }
}

type TerminalNwaBuilder<'tok, 'pm, 'nwa> = super::TerminalNwaBuilder<'tok, 'pm, 'nwa, NWA>;

/// Borrowed lexical graph with exactly the original induced view's logical
/// state IDs. Numeric event/frontier keys stay logical; only immutable lexical
/// reads are translated to the shared source graph.
#[derive(Clone,Copy)]
pub struct BorrowedObservation<'a> {
    pub(super) original_to_logical:&'a [u32],
    pub(super) logical_to_original:&'a [u32],
    pub(super) has_epsilon:bool,
    pub(super) scalar_dispatch:bool,
}
impl<'a> BorrowedObservation<'a> {
    pub fn new(source:&Tokenizer,original_to_logical:&'a [u32],logical_to_original:&'a [u32])->Option<Self>{
        if source.has_virtual_residual_runtime()||original_to_logical.len()!=source.num_states()as usize
            || logical_to_original.is_empty()||logical_to_original.len()>=16384
            ||logical_to_original.windows(2).any(|w|w[0]>=w[1]){return None;}
        for (q,&raw)in logical_to_original.iter().enumerate(){
            if *original_to_logical.get(raw as usize)?!=q as u32{return None;}
            // FIRST/reset prefixes were epsilon-closed by the view proof.
            // Reject a map omitting part of any retained epsilon closure.
            for r in source.singleton_epsilon_closure(raw){
                if *original_to_logical.get(r as usize)?==u32::MAX{return None;}
            }
        }
        for (raw,&q)in original_to_logical.iter().enumerate(){
            if q!=u32::MAX&&logical_to_original.get(q as usize).copied()!=Some(raw as u32){return None;}
        }
        let has_epsilon=logical_to_original.iter().any(|&q|source.state_has_epsilon_transitions(q));
        let start=source.initial_state_id();
        let any_start=source.transitions_from(start).next().is_some();
        let kept_start=source.transitions_from(start).any(|(_,target)|original_to_logical.get(target as usize).is_some_and(|&q|q!=u32::MAX));
        // Removing the last direct reset edge can change the special dispatcher
        // interpretation. Leave that shape to the ordinary materialized path.
        if any_start!=kept_start{return None;}
        let scalar_dispatch=if source.has_scalar_deterministic_dispatch(){true}
        else if let Some(roots)=source.deterministic_dispatch_roots(){
            let mut todo=roots.to_vec();let mut seen=vec![false;logical_to_original.len()];let mut scalar=true;let mut work=0usize;
            while let Some(raw)=todo.pop(){
                let q=*original_to_logical.get(raw as usize)?;if q==u32::MAX{return None;}
                if std::mem::replace(&mut seen[q as usize],true){continue;}
                if source.state_has_epsilon_transitions(raw){scalar=false;break;}
                for (_,target) in source.transitions_from(raw){work+=1;if work>2_000_000{return None;}else if original_to_logical[target as usize]!=u32::MAX{todo.push(target);}}
            }
            scalar
        }else{false};
        Some(Self{original_to_logical,logical_to_original,has_epsilon,scalar_dispatch})
    }
    pub(super) fn original(&self,q:u32)->u32{self.logical_to_original[q as usize]}
    pub(in crate::terminal_dwa) fn logical(&self,raw:u32)->Option<u32>{self.original_to_logical.get(raw as usize).copied().filter(|&q|q!=u32::MAX)}
    pub fn len(&self)->usize{self.logical_to_original.len()}
}

pub struct NativeBuild {
 pub sink:NWA,
 pub profile:TerminalDwaBuildProfile,
 pub build_ms:f64,
}

#[allow(clippy::too_many_arguments)]
pub fn build<'a>(tokenizer:&'a Tokenizer,coloring:&TerminalColoring,ignore:Option<TerminalID>,
 seed:&crate::automata::weighted_u32::nwa::NWA,leaf:u32,num_tsids:u32,
 tree:VocabPrefixTreeNode<'_>,roots:&NodesByTokenizerState,flat:Option<&'a [u32]>,active:&[bool])->Option<NativeBuild>{
 build_impl(tokenizer,coloring,ignore,NativeSeedInput::Generic(seed),leaf,num_tsids,tree,roots,flat,active,None)
}
#[allow(clippy::too_many_arguments)]
pub fn build_borrowed<'a>(tokenizer:&'a Tokenizer,coloring:&TerminalColoring,ignore:Option<TerminalID>,
 seed:&crate::automata::weighted_u32::nwa::NWA,leaf:u32,num_tsids:u32,
 tree:VocabPrefixTreeNode<'_>,roots:&NodesByTokenizerState,flat:Option<&'a [u32]>,active:&[bool],map:BorrowedObservation<'a>)->Option<NativeBuild>{
 build_impl(tokenizer,coloring,ignore,NativeSeedInput::Generic(seed),leaf,num_tsids,tree,roots,flat,active,Some(map))
}
enum NativeSeedInput<'a>{Generic(&'a crate::automata::weighted_u32::nwa::NWA),Direct{max_token:u32,classes:&'a [u32]}}
#[allow(clippy::too_many_arguments)]
pub fn build_borrowed_direct_seed<'a>(tokenizer:&'a Tokenizer,coloring:&TerminalColoring,ignore:Option<TerminalID>,
 classes:&[u32],max_token:u32,num_tsids:u32,tree:VocabPrefixTreeNode<'_>,roots:&NodesByTokenizerState,
 flat:Option<&'a [u32]>,active:&[bool],map:BorrowedObservation<'a>)->Option<NativeBuild>{
 build_impl(tokenizer,coloring,ignore,NativeSeedInput::Direct{max_token,classes},0,num_tsids,tree,roots,flat,active,Some(map))
}
#[allow(clippy::too_many_arguments)]
fn build_impl<'a>(tokenizer:&'a Tokenizer,coloring:&TerminalColoring,ignore:Option<TerminalID>,
 seed:NativeSeedInput<'_>,leaf:u32,num_tsids:u32,
 tree:VocabPrefixTreeNode<'_>,roots:&NodesByTokenizerState,flat:Option<&'a [u32]>,active:&[bool],map:Option<BorrowedObservation<'a>>)->Option<NativeBuild>{
 let parallel=std::env::var_os("GLRMASK_DISABLE_L2P_PARALLEL_ROOT_TRIE").is_none()
   && std::env::var_os("GLRMASK_ENABLE_L2P_SELF_LOOP_SUBTREE_SKIP").is_none()
   && tree.children().len()>=2 && rayon::current_num_threads()>1
   && (std::env::var_os("GLRMASK_L2P_PARALLEL_ROOT_TRIE").is_some()||tree.reachable_token_ids().len()>=512);
 if parallel||tree.reachable_token_ids().is_empty()||tree.reachable_token_ids().iter().any(|id|id>=512)
   ||map.map_or(tokenizer.num_states()as usize,|m|m.len())>=16384||active.len()>65536||num_tsids==0{return None}
 let started=std::time::Instant::now();let mut sink=match seed{NativeSeedInput::Generic(seed)=>NWA::from_seed(seed,num_tsids)?,
  NativeSeedInput::Direct{max_token,classes}=>NWA::from_ordered_singleton_seeds(num_tsids,max_token,classes)?};
 let mut initial=vec![false;sink.states_len()];for(_,sources)in roots.iter(){for &s in sources{*initial.get_mut(s as usize)?=true;}}
 let mut pm=PossibleMatchesComputer::new(tokenizer);
 let mut builder=TerminalNwaBuilder::new(tokenizer,coloring.clone(),&mut pm,&mut sink,num_tsids,leaf,ignore,
   initial,false,Some(active.to_vec()),map.map_or(tokenizer.num_states()as usize,|m|m.len()),flat);
 if let Some(map)=map {
  // Mapped DFA scalar paths retain the original cached execution route; all
  // nontrivial configurations are read from the original immutable lexer.
  builder.borrowed_map=Some(map);
  builder.has_epsilon_transitions=map.has_epsilon;
  builder.scalar_deterministic_dispatch=map.scalar_dispatch;
  builder.shared_flat_transitions=flat.filter(|rows|rows.len()==tokenizer.num_states()as usize*256);
  builder.cached_reset_roots=builder.reset_roots().into_vec();
  if crate::terminal_dwa::env_policy::enabled("GLRMASK_BOUNDARY_BORROWED_COMPACT"){
   builder.borrowed_cursor=native_borrowed_cursor::BorrowedScalarCache::new(tokenizer,map,U8Set::from_words(*tree.subtree_bytes()),flat);
  }
  builder.lean_frontiers=true;
 }
 let t=std::time::Instant::now();builder.build_from_trie(tree,roots);builder.profile.trie_walk_ms=t.elapsed().as_secs_f64()*1000.;
 if std::env::var_os("GLRMASK_PROFILE_NATIVE_BUILDER").is_some(){
  let slots=builder.leaf_token_ids_buffer.len();
  let live=builder.leaf_token_ids_buffer.values().filter(|x|!x.is_empty()).count();
  eprintln!("[glrmask/profile][native_builder_buffers] sparse_leaf_slots={slots} nonempty_leaf_slots={live} estimated_entry_bytes={} transitions={} epsilons={}",slots*std::mem::size_of::<LeafTokenIds>(),builder.transition_buffer.len()+builder.compact_buffer.as_ref().map_or(0,|b|b.len()),builder.epsilon_buffer.len());
 }
 if builder.mapping_failed||builder.borrowed_cursor.as_ref().is_some_and(|c|c.failed.get())||builder.compact_buffer.as_ref().is_some_and(|b|b.failed){return None}
 let t=std::time::Instant::now();builder.flush_transition_buffer();if builder.leaf_flush_failed{return None}builder.profile.flush_ms=t.elapsed().as_secs_f64()*1000.;
 if std::env::var_os("GLRMASK_PROFILE_NATIVE_BUILDER").is_some(){
  eprintln!("[glrmask/profile][native_event_details] future_absence_hits={} walk_ms={:.3} flush_ms={:.3} scan_ms={:.3} match_filter_ms={:.3} end_state_ms={:.3} match_process_ms={:.3} continuation_ms={:.3} scan_calls={} scan_bytes={} matches={}",
   builder.future_absence_hits,builder.profile.trie_walk_ms,builder.profile.flush_ms,
   builder.profile.trie_execute_ms,builder.profile.trie_match_filter_ms,
   builder.profile.trie_end_state_ms,builder.profile.trie_match_process_ms,
   builder.profile.trie_continuation_weight_ms,builder.profile.trie_execute_calls,
   builder.profile.trie_execute_input_bytes,builder.profile.trie_matches);
 }
 let profile=builder.profile;drop(builder);let build_ms=started.elapsed().as_secs_f64()*1000.;
 Some(NativeBuild{sink,profile,build_ms})
}

#[cfg(test)] #[path="native_builder_tests.rs"] mod tests;

