//! Reserved recycling slots must survive intermediate cache reclamation.
use super::*;

fn unique(id:u32)->ParserGSS { let mut s=ParserGSS::from_single_stack(vec![id],TerminalsDisallowed::new());assert!(s.reserve_single_segment_capacity(64));s }
fn cache(count:usize)->FlatFrontierScratch { FlatFrontierScratch::with_preallocated_gss(count) }
#[test]
fn recycling_active_entries_keeps_its_preflight_reserved_slots() {
 let mut f=cache(256);f.retired_gss.push(unique(1000));
 let mut state=ParserStateMap{entries:smallvec::smallvec![(1,unique(2000))]};
 assert!(f.can_recycle_old_state(&state,1));let _selected=f.gss_pool.pop().unwrap();
 f.recycle_old_entries(std::mem::take(&mut state.entries));
 assert_eq!(f.gss_pool.len(),256);assert_eq!(f.retired_gss.len(),1);
 let _spare=f.gss_pool.pop().unwrap();f.reclaim_retired_gss();
 assert_eq!(f.gss_pool.len(),256);assert!(f.retired_gss.is_empty());
}
#[test]
fn shared_replacement_checks_capacity_after_reclamation() {
 let mut f=cache(255);f.retired_gss.push(unique(1000));
 let mut state=ParserStateMap{entries:smallvec::smallvec![(1,unique(2000)),(2,unique(2001))]};
 assert!(!f.replace_state_with_shared_uniform_stack_keys(&mut state,&[5],&[42],&TerminalsDisallowed::new()));
 assert_eq!(state.entries.len(),2);assert_eq!(f.gss_pool.len(),256);
}
#[test]
fn independent_replacement_does_not_lose_reusable_objects() {
 let mut f=cache(255);f.retired_gss.push(unique(1000));
 let mut state=ParserStateMap{entries:smallvec::smallvec![(1,unique(2000)),(2,unique(2001))]};
 assert!(f.replace_state_with_uniform_stack_keys(&mut state,&[5],&[42],&TerminalsDisallowed::new()));
 assert_eq!(state.entries.len(),1);assert_eq!(state.entries[0].0,5);
 assert_eq!(f.gss_pool.len(),256);assert_eq!(f.retired_gss.len(),1);
}
