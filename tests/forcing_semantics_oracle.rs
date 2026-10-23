//! Independent finite byte/event trace model. No GLRMask parser, tokenizer,
//! regular-expression engine or grammar normalization decides the expected set.
use std::collections::{BTreeSet, VecDeque};
use glrmask::{BuildOptions, Constraint, ConstraintState, Grammar, MaskStrategy, TokenSet, Vocab};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Cell { Byte(u8), Event(u32) }
use Cell::{Byte as B, Event as E};
type Residuals = BTreeSet<Vec<Cell>>;

#[derive(Clone)]
struct Entry { id: u32, bytes: Option<Vec<u8>> }

fn advance(words: &Residuals, entry: &Entry) -> Residuals {
    let mut next = Residuals::new();
    for word in words {
        if word.first() == Some(&E(entry.id)) {
            next.insert(word[1..].to_vec());
        }
        if let Some(bytes) = entry.bytes.as_ref().filter(|b| !b.is_empty()) {
            if word.len() >= bytes.len() && word.iter().zip(bytes).all(|(c,b)| *c == B(*b)) {
                next.insert(word[bytes.len()..].to_vec());
            }
        }
    }
    next
}

fn accepting(words: &Residuals) -> bool { words.contains(&Vec::new()) }

fn reference_force(mut words: Residuals, entries: &[Entry], canonical: bool) -> Vec<u32> {
    let mut output = Vec::new();
    while !words.is_empty() && !accepting(&words) {
        assert!(output.len() < 32, "a finite tape cannot make unbounded progress");
        let admitted = entries.iter().filter_map(|entry| {
            let next = advance(&words, entry);
            (!next.is_empty()).then_some((entry, next))
        }).collect::<Vec<_>>();
        let event_frontier = words.iter().any(|word| matches!(word.first(), Some(E(_))));
        let choice = if !canonical || event_frontier {
            if admitted.len() != 1 { break; }
            Some(0)
        } else {
            let prefix = words.iter().map(|word| word.iter().take_while(|c| matches!(c, B(_)))
                .map(|c| match c { B(b)=>*b, E(_)=>unreachable!() }).collect::<Vec<_>>())
                .reduce(|mut a,b| {
                    let count = a.iter().zip(&b).take_while(|(a,b)| a==b).count();
                    a.truncate(count); a
                }).unwrap_or_default();
            if prefix.is_empty() { break; }
            let mut best = None;
            let mut blocked = false;
            for (i,(entry,_)) in admitted.iter().enumerate() {
                let Some(bytes) = entry.bytes.as_ref().filter(|b| !b.is_empty()) else { continue; };
                if bytes.len() > prefix.len() && bytes.starts_with(&prefix) { blocked=true;break; }
                if prefix.starts_with(bytes) && best.is_none_or(|old: usize| {
                    let old = admitted[old].0;
                    bytes.len()>old.bytes.as_ref().unwrap().len()
                        || (bytes.len()==old.bytes.as_ref().unwrap().len() && entry.id<old.id)
                }) { best=Some(i); }
            }
            if blocked { break; }
            best
        };
        let Some(i) = choice else { break; };
        output.push(admitted[i].0.id);
        words = admitted[i].1.clone();
    }
    output
}

fn bit(mask: &[u32], id: u32) -> bool {
    mask.get(id as usize / 32).is_some_and(|word| word & (1 << (id % 32)) != 0)
}

fn grammar_source(words: &Residuals) -> (String, BTreeSet<u32>) {
    let mut events = BTreeSet::new();
    let alternatives = words.iter().map(|word| {
        if word.is_empty() { return "eps".to_owned(); }
        word.iter().map(|cell| match cell {
            B(byte) => format!("b\"\\x{byte:02X}\""),
            E(id) => { events.insert(*id); format!("EVENT_{id}") },
        }).collect::<Vec<_>>().join(" ")
    }).collect::<Vec<_>>().join(" | ");
    let declarations = events.iter().map(|id| format!("extern token EVENT_{id}\n")).collect::<String>();
    (format!("start root\n{declarations}nt root = {alternatives}\n"), events)
}

fn check_states(c: &Constraint, language: &Residuals, entries: &[Entry], label: &str) -> usize {
    let mut pending = VecDeque::<(ConstraintState<'_>, Residuals, Vec<u32>)>::new();
    pending.push_back((c.start(), language.clone(), Vec::new()));
    let mut checked = 0;
    while let Some((state, words, trace)) = pending.pop_front() {
        checked += 1;
        assert!(checked < 20_000, "bounded finite-tape test exceeded its intended workload");
        assert_eq!(state.is_accepting(), accepting(&words), "{label}; trace={trace:?}");
        // A rejected state has no allowed tokens. An accepting state may also
        // have an empty mask when no further token is possible.
        let rejected = !state.is_accepting() && state.mask().iter().all(|word| *word == 0);
        assert_eq!(rejected, words.is_empty(), "{label}; trace={trace:?}");
        let original_mask = state.mask();
        for canonical in [true,false] {
            let expected = reference_force(words.clone(), entries, canonical);
            let got = if canonical {
                let public=state.forced_tokens().expect("canonical forcing query failed");
                assert_eq!(public,state.forced_canonical_reference(),"public/reference divergence: {label}; trace={trace:?}");
                public
            } else { state.forced_token_ids_reference() };
            // Both query contracts permit a shorter answer when a conservative
            // byte/proof boundary intervenes, but never a different encoding.
            assert!(expected.starts_with(&got),
                "{label}; trace={trace:?}; canonical={canonical}; got={got:?}, expected-prefix-of={expected:?}");
            let mut replay = words.clone();
            for id in got {
                assert!(!accepting(&replay), "forced past an allowed stop");
                let entry = entries.iter().find(|e| e.id==id).expect("force emitted an unknown ID");
                replay = advance(&replay,entry);
                assert!(!replay.is_empty(), "force emitted an independently invalid ID");
            }
        }
        assert_eq!(state.mask(), original_mask, "queries mutated source state");
        for entry in entries {
            let expected = advance(&words,entry);
            assert_eq!(bit(&original_mask,entry.id), !expected.is_empty(),
                "{label}; trace={trace:?}, next={}",entry.id);
            if !expected.is_empty() {
                let mut next=state.clone(); next.commit_token(entry.id).unwrap();
                let mut history=trace.clone();history.push(entry.id);
                pending.push_back((next,expected,history));
            }
        }
    }
    checked
}

#[test]
fn exact_event_and_byte_trace_oracle_covers_all_original_id_paths() {
    let entries=vec![
        Entry{id:0,bytes:Some(b"a".to_vec())}, Entry{id:1,bytes:Some(b"b".to_vec())},
        Entry{id:2,bytes:Some(b"ab".to_vec())}, Entry{id:3,bytes:Some(b"a".to_vec())},
        Entry{id:4,bytes:Some(Vec::new())}, Entry{id:5,bytes:Some(b"ba".to_vec())},
        Entry{id:6,bytes:Some(b"aba".to_vec())}, Entry{id:7,bytes:Some(vec![0xC3])},
        Entry{id:8,bytes:Some(vec![0xA9])}, Entry{id:9,bytes:None},
        Entry{id:10,bytes:Some("é".as_bytes().to_vec())},
    ];
    let vocab=Vocab::new(entries.iter().filter_map(|e|
        e.bytes.as_ref().map(|b|(e.id,b.clone()))).collect()).with_special_tokens([9]);
    let bases = [vec![B(b'a'),B(b'b')],vec![E(3),B(b'b')],vec![B(b'a'),E(4)],
        vec![E(4),B(b'a'),B(b'b')],vec![E(9),B(b'a')]];
    let mut cases=(1usize..(1<<bases.len())).map(|bits|
        bases.iter().enumerate().filter(|(i,_)|bits&(1<<i)!=0)
            .map(|(_,word)|word.clone()).collect::<Residuals>()).collect::<Vec<_>>();
    cases.extend([
        [vec![E(3),E(9)],vec![B(b'a'),B(b'b'),E(9)]].into_iter().collect(),
        [vec![],vec![E(4),B(b'a')]].into_iter().collect(),
        [vec![B(0xC3),B(0xA9),E(9)],vec![E(3),B(0xC3),B(0xA9)]].into_iter().collect(),
        [vec![B(b'a'),E(3),B(b'b')],vec![B(b'a'),B(b'b')]].into_iter().collect(),
    ]);
    let mut count=0;
    for (case,words) in cases.iter().enumerate() {
        let (source,events)=grammar_source(words);
        let mut grammar=Grammar::from_glrm(&source);
        for id in events { grammar=grammar.bind_snapshot(format!("EVENT_{id}"),TokenSet::new([id])).unwrap(); }
        for mode in [MaskStrategy::Auto,MaskStrategy::TokenTrie,MaskStrategy::TokenClassTrie,MaskStrategy::ParserDwa] {
            let compiled=grammar.compile_with_build_options(&vocab,BuildOptions::default().optimization(mode)).unwrap();
            let label=format!("case={case},mode={mode:?}");
            count+=check_states(&compiled,words,&entries,&label);
            if case%7==0 {
                let loaded=Constraint::load_without_vocab(&compiled.save()).unwrap();
                count+=check_states(&loaded,words,&entries,&format!("loaded:{label}"));
            }
        }
    }
    eprintln!("independent mixed byte/event oracle: {} languages, 4 modes, {count} states",cases.len());
}


#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow (~11s); run with --features slow-tests"
)]
fn prepared_child_byte_event_calls_match_finite_trace_oracle_mid_token() {
    let entries=vec![
        Entry{id:0,bytes:Some(b"(".to_vec())}, Entry{id:1,bytes:Some(b")".to_vec())},
        Entry{id:2,bytes:Some(b"a".to_vec())}, Entry{id:3,bytes:Some(b"ab".to_vec())},
        Entry{id:4,bytes:Some(b"(a".to_vec())}, Entry{id:5,bytes:Some(b"b)".to_vec())},
        Entry{id:6,bytes:Some(b"(ab)".to_vec())}, Entry{id:7,bytes:Some(Vec::new())},
        Entry{id:8,bytes:Some(b"a".to_vec())}, Entry{id:9,bytes:Some(b"b".to_vec())},
        Entry{id:10,bytes:Some(b"()".to_vec())}, Entry{id:11,bytes:None},
    ];
    let vocab=Vocab::new(entries.iter().filter_map(|e|
        e.bytes.as_ref().map(|bytes|(e.id,bytes.clone()))).collect()).with_special_tokens([11]);
    let parent = Grammar::from_glrm("extern grammar CHILD start root nt root=\"(\" CHILD \")\"");
    let parent_prepared=parent.prepare(&vocab).unwrap();
    let leaf_cases: Vec<Residuals>=vec![
        [vec![B(b'a'),B(b'b')]].into_iter().collect(),
        [vec![],vec![B(b'a')],vec![B(b'a'),B(b'b')]].into_iter().collect(),
        [vec![B(b'a'),B(b'b')],vec![E(8),B(b'b')]].into_iter().collect(),
        [vec![E(7)],vec![E(11)],vec![B(b'a'),B(b'b')]].into_iter().collect(),
        [vec![B(b'a'),E(8),B(b'b')],vec![B(b'a'),B(b'b')]].into_iter().collect(),
    ];
    let mut count=0;
    for (case,leaf_language) in leaf_cases.into_iter().enumerate() {
        let (source,events)=grammar_source(&leaf_language);
        let leaf=Grammar::from_glrm(&source);
        let leaf_prepared=leaf.prepare(&vocab).unwrap();
        let assembly=parent_prepared.bind_snapshot("CHILD",&leaf_prepared).unwrap();
        let mut assembly=glrmask::PreparedGrammar::load_with_vocab(assembly.save(),&vocab).unwrap();
        let mut source_assembly=parent.bind_snapshot("CHILD",&leaf).unwrap();
        for id in events {
            let path=format!("CHILD::EVENT_{id}");
            assembly=assembly.bind_snapshot(&path,TokenSet::new([id])).unwrap();
            source_assembly=source_assembly.bind_snapshot(&path,TokenSet::new([id])).unwrap();
        }
        let expected=leaf_language.iter().map(|word| {
            let mut wrapped=vec![B(b'(')];wrapped.extend_from_slice(word);wrapped.push(B(b')'));wrapped
        }).collect::<Residuals>();
        for mode in [MaskStrategy::Auto,MaskStrategy::TokenTrie,MaskStrategy::TokenClassTrie,MaskStrategy::ParserDwa] {
            let opts=BuildOptions::default().optimization(mode);
            let prepared=assembly.link_with(opts.clone()).unwrap();
            let source=source_assembly.compile_with_build_options(&vocab,opts).unwrap();
            count+=check_states(&prepared,&expected,&entries,&format!("prepared:{case}:{mode:?}"));
            count+=check_states(&source,&expected,&entries,&format!("source:{case}:{mode:?}"));
            let loaded=Constraint::load_without_vocab(&prepared.save()).unwrap();
            count+=check_states(&loaded,&expected,&entries,&format!("loaded:{case}:{mode:?}"));
        }
    }
    eprintln!("prepared mixed-event oracle: 5 child languages, 4 modes, source/prepared/loaded, {count} states");
}

#[test]
fn unused_high_original_ids_do_not_change_event_weight_or_mask_domains() {
    for highest in [31,32,63,64,257] {
        let vocab=Vocab::new(vec![
            (0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"ab".to_vec()),(3,b"a".to_vec()),
            (highest,b"unused".to_vec())]).with_special_tokens([highest-1]);
        let g=Grammar::from_glrm("extern token MARK start root nt root=\"a\" \"b\" | MARK \"b\"")
            .bind_snapshot("MARK",TokenSet::new([3])).unwrap();
        for mode in [MaskStrategy::Auto,MaskStrategy::TokenTrie,MaskStrategy::TokenClassTrie,MaskStrategy::ParserDwa] {
            let c=g.compile_with_build_options(&vocab,BuildOptions::default().optimization(mode)).unwrap();
            for c in [c.clone(),Constraint::load_without_vocab(&c.save()).unwrap()] {
                let mut s=c.start();
                assert!(s.mask().len() >= (highest as usize + 1).div_ceil(32));
                assert!(!bit(&s.mask(),highest)); assert!(!bit(&s.mask(),highest-1));
                assert!(s.forced_canonical_reference().is_empty());
                for id in [0,2,3] {assert!(bit(&s.mask(),id));}
                s.commit_token(3).unwrap();
                assert_eq!(s.forced_token_ids_reference(),[1]);s.commit_token(1).unwrap();assert!(s.is_accepting());
            }
        }
    }
}
