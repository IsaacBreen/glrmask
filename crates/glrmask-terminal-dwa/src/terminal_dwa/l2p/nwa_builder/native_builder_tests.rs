use super::*;
use crate::ds::vocab_prefix_tree::VocabPrefixTree;
use crate::automata::lexer::ast::{bytes, choice, plus};
use crate::automata::lexer::compile::{build_regex, build_regex_partitioned};
use crate::automata::weighted_u32::nwa::NWA as GenericNwa;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::ds::weight::Weight as GenericWeight;

fn id_map(states:u32,tokens:usize)->InternalIdMap {
    let states=(0..states).collect::<Vec<_>>();
    let tokens=(0..tokens as u32).collect::<Vec<_>>();
    InternalIdMap {
        tokenizer_states:ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(states.clone(),states),
        vocab_tokens:ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(tokens.clone(),tokens),
        deferred_vocab_singleton_original_ids:None,
    }
}

fn seeded(tokenizer:&Tokenizer,map:&InternalIdMap,keep:&[bool])->(GenericNwa,u32,NodesByTokenizerState) {
    let mut seed=GenericNwa::new(map.num_tsids(),map.max_internal_token_id());
    let leaf=seed.add_state();seed.set_final_weight(leaf,GenericWeight::all());
    let start=seed.add_state();seed.start_states_mut().push(start);
    let roots=seed_root_nodes_filtered(tokenizer,&mut seed,start,map,keep);
    (seed,leaf,roots)
}

#[test]
fn native_emission_matches_original_nwa_on_generated_lexers() {
    let expressions=vec![
        choice(vec![bytes(b"a"),bytes(b"ab"),bytes(b"abc")]),
        plus(choice(vec![bytes(b"b"),bytes(b"ca")])),
        bytes(b"!"),plus(bytes(b" ")),
        choice(vec![bytes(b"aba!"),bytes(b"bc"),bytes(b"c")]),
    ];
    let mut random=710193u64;
    let mut next=||{random=random.wrapping_mul(6364136223846793005).wrapping_add(1);(random>>32) as usize};
    let mut comparisons=0;
    for partitioned in [false,true] {
        let tokenizer=if partitioned {build_regex_partitioned(&expressions,&[0,1,2,3,4])}
            else {build_regex(&expressions)}.into_tokenizer(5,Some(Arc::from(expressions.clone())));
        let flat=crate::terminal_dwa::l1::build_flat_transition_table(&tokenizer);
        for case in 0..48 {
            let mut words=vec![b"a".to_vec(),b"abc!".to_vec(),b"ab !a".to_vec(),b"bc".to_vec()];
            for _ in 0..24 {let len=1+next()%12;words.push((0..len).map(|_|b"abc! "[next()%5]).collect());}
            words.sort();words.dedup();
            let entries=words.into_iter().enumerate().collect::<Vec<_>>();
            let tree=VocabPrefixTree::build(&entries);
            let map=id_map(tokenizer.num_states(),entries.len());
            let mut keep=(0..tokenizer.num_states()).map(|_|next()%3!=0).collect::<Vec<_>>();
            keep[tokenizer.initial_state_id() as usize]=true;
            let active=(0..5).map(|t|case%3!=1||t!=4).collect::<Vec<_>>();
            let ignore=(case%2==0).then_some(3);
            let coloring=TerminalColoring::identity(5);
            for supplied in [None,Some(flat.as_slice()),Some(&flat[..flat.len()-1])] {
                let (seed,leaf,roots)=seeded(&tokenizer,&map,&keep);
                let original=(seed.start_states().to_vec(),seed.states().to_vec());
                let mut ordinary=seed.clone();let mut pm=PossibleMatchesComputer::new(&tokenizer);
                super::super::build_nwa_via_trie_walk(&tokenizer,&coloring,false,ignore,&mut ordinary,
                    leaf,map.num_tsids(),&tree.root,&roots,&mut pm,supplied,&active);
                let mut native=build(&tokenizer,&coloring,ignore,&seed,leaf,map.num_tsids(),
                    &tree.root,&roots,supplied,&active).expect("bounded serial-native eligibility");
                let native=native.sink.export_raw().expect("valid finite event coefficients");
                assert_eq!(native.start_states(),ordinary.start_states());
                assert_eq!(native.states(),ordinary.states(),"partitioned={partitioned} case={case} supplied={}",supplied.map_or(0,|v|v.len()));
                assert_eq!(original,(seed.start_states().to_vec(),seed.states().to_vec()));
                comparisons+=1;
            }
        }
    }
    assert_eq!(comparisons,288);
}

#[test]
fn native_uniform_coefficients_preserve_sparse_token_boundaries_and_invalidity() {
    for end in [0,1,128,u32::MAX] {
        let a=Weight::from_uniform(0..=end,[0,63,64,255,256,511].into_iter().collect());
        let b=Weight::from_uniform(0..=end,[1,64,127,510].into_iter().collect());
        let c=a.union(&b);assert!(c.valid);
        let actual=(0..512u32).filter(|t|c.bits[*t as usize/64]&(1u64<<(*t%64))!=0).collect::<Vec<_>>();
        assert_eq!(actual,vec![0,1,63,64,127,255,256,510,511]);
        assert_eq!(a.union(&Weight::empty()).bits,a.bits);
        assert!(!a.union(&Weight{bits:[1;8],end:end.wrapping_add(1),valid:true}).valid);
    }
    let bad=Weight::from_uniform(0..=3,[512].into_iter().collect());
    assert!(!bad.valid);assert!(!bad.is_empty());
    assert!(!Weight::empty().union(&bad).valid);
    assert!(!Weight::from_uniform(1..=3,[1].into_iter().collect()).valid);
}

#[test]
fn native_emission_declines_unsupported_seed_and_token_domains() {
    let expressions=vec![plus(bytes(b"a")),bytes(b"!")];
    let tokenizer=build_regex(&expressions).into_tokenizer(2,Some(Arc::from(expressions)));
    let map=id_map(tokenizer.num_states(),513);
    let keep=vec![true;tokenizer.num_states() as usize];let (seed,leaf,roots)=seeded(&tokenizer,&map,&keep);
    let tree=VocabPrefixTree::build(&[(512usize,b"a!".to_vec())]);
    assert!(build(&tokenizer,&TerminalColoring::identity(2),None,&seed,leaf,map.num_tsids(),&tree.root,&roots,None,&[true,true]).is_none());
    assert!(NWA::from_seed(&seed,map.num_tsids()).is_none(),"seed itself contains token512");
    let map=id_map(tokenizer.num_states(),32);
    let (seed,leaf,_)=seeded(&tokenizer,&map,&keep);
    let mut bad=seed.clone();bad.add_epsilon(1,leaf,GenericWeight::all());
    assert!(NWA::from_seed(&bad,map.num_tsids()).is_none());
    let mut bad=seed.clone();let extra=bad.add_state();bad.set_final_weight(extra,GenericWeight::all());
    assert!(NWA::from_seed(&bad,map.num_tsids()).is_none());
    let mut valid=NWA::from_seed(&seed,map.num_tsids()).unwrap();
    valid.append_transition(0,0,1,map.num_tsids()-1,false,&[0;8]);
    assert!(valid.export_raw().is_none(),"invalid must not become a silent empty edge");
    let mut valid=NWA::from_seed(&seed,map.num_tsids()).unwrap();
    valid.append_transition(0,0,1,map.num_tsids(),true,&[1;8]);
    assert!(valid.export_raw().is_none(),"incompatible uniform domains must decline");
}

#[test]
fn strict_future_absence_preserves_only_possible_zero_byte_node_match() {
    let exprs=vec![choice(vec![crate::automata::lexer::ast::Expr::Epsilon,bytes(b"a"),bytes(b"ab")]),
        plus(bytes(b"b")),bytes(&[0,255]),bytes(b"!"),choice(vec![bytes(b"a!"),bytes(b"b!")])];
    let words=[b"a".to_vec(),b"ab".to_vec(),b"aba!".to_vec(),b"b".to_vec(),b"bb!".to_vec(),
        b"abbb".to_vec(),vec![0,255],vec![0,255,b'!'],vec![255,0],b"!".to_vec()];
    let tree=VocabPrefixTree::build(&words.into_iter().enumerate().collect::<Vec<_>>());
    let mut nodes=vec![&tree.root];let mut cursor=0;
    while cursor<nodes.len(){let children=nodes[cursor].children();nodes.extend(children);cursor+=1;}
    let mut checks=0;
    for partitioned in [false,true]{
        let tok=if partitioned{build_regex_partitioned(&exprs,&[0,1,2,3,4])}else{build_regex(&exprs)}
            .into_tokenizer(5,Some(Arc::from(exprs.clone())));
        let mut pm=PossibleMatchesComputer::new(&tok);
        for q in 0..tok.num_states(){
            if tok.state_has_epsilon_transitions(q){continue}
            for t in 0..5u32{
                if tok.possible_future_terminals(q).get(t as usize){continue}
                for node in &nodes{for suffix in [&b""[..],&b"a"[..],&b"bb"[..],&b"ab!"[..],&[0,255][..]]{
                    let ordinary=pm.possible_matches_for_suffix_and_node(suffix,node,q);
                    if let Some(found)=ordinary.get(&t){
                        assert!(found.iter().all(|id|suffix.is_empty()&&node.has_token()&&id==node.token_id()as u32),
                            "future absence lost a positive extension q={q} t={t} suffix={suffix:?}");
                    }
                    checks+=1;
                }}
            }
        }
    }
    assert!(checks>500);
}

#[test]
fn native_factored_leaf_flush_exhaustion_declines_without_mutating_the_seed() {
    // Real builder overflow, not just a flag assertion: create too many
    // distinct leaf keys, then ensure no partial native graph escapes.
    let exprs=vec![bytes(b"a"),bytes(b"!")];
    let tokenizer=build_regex(&exprs).into_tokenizer(2,Some(Arc::from(exprs)));
    let map=id_map(tokenizer.num_states(),2);
    let keep=vec![true;tokenizer.num_states()as usize];
    let(seed,leaf,_)=seeded(&tokenizer,&map,&keep);
    let snapshot=(seed.start_states().to_vec(),seed.states().to_vec());
    let mut nwa=NWA::from_seed(&seed,map.num_tsids()).unwrap();
    let mut pm=PossibleMatchesComputer::new(&tokenizer);
    let mut builder=TerminalNwaBuilder::new(&tokenizer,TerminalColoring::identity(2),&mut pm,
        &mut nwa,map.num_tsids(),leaf,None,vec![true;seed.states().len()],false,None,Some(vec![true,true]),tokenizer.num_states()as usize,None);
    builder.batch_leaf_flush=true;
    for q in 0..100001u32 {builder.leaf_token_ids_buffer.insert((q,0),smallvec::smallvec![0]);}
    builder.flush_transition_buffer();
    assert!(builder.leaf_flush_failed);
    drop(builder);
    let result=nwa.export_raw().unwrap();
    assert_eq!(result.start_states(),snapshot.0);
    assert_eq!(result.states(),snapshot.1);
    assert_eq!(seed.start_states(),snapshot.0);
    assert_eq!(seed.states(),snapshot.1);
}
