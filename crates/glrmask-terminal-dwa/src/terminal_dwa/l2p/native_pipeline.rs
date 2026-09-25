//! Bounded, exact finite-token arithmetic for boundary terminal construction.
//!
//! This is NOT a lexer-state, terminal, or grammar quotient. Raw tokenizer-state
//! coordinates stay distinct; finite token sets use eight machine words and the
//! universal weight has a separate representation. Unsupported inputs return
//! `None` so the caller retains the ordinary compiler.
//!
//! The determinizer preserves the existing discovery order, including grouped
//! source weights and the singleton scheduling policy. The minimizer preserves
//! height ordering, structural hash classes, and greedy compatibility. Those
//! details matter because the existing downstream compiler observes more than
//! the weighted language of an arbitrarily reshaped terminal DWA.
//!
//! All caches are invocation-local and retain their pointer-keyed operands.
//! Allocation/work bounds decline without publishing partial output.

use glrmask_weight::__private::{SharedTokenSet, Weight};
use glrmask_weighted_automata::weighted_u32::{
    dwa::{DWA, DWAState},
    nwa::NWA,
};
use rustc_hash::FxHashMap;
use smallvec::{SmallVec,smallvec};
type Runs=SmallVec<[Run;2]>;
type Subset=SmallVec<[(u32,u32);1]>;
type RegionSignature=SmallVec<[(i32,u32);4]>;
use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
    time::Instant,
};
const WORDS: usize = 8;
const MAX_MASKS: usize = 100_000;
const MAX_STATES: usize = 100_000;
const MAX_RUNS: usize = 2_000_000;
type Bits = [u64; WORDS];
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Run {
    lo: u32,
    hi: u32,
    b: u32,
}
#[derive(Default, Debug)]
pub struct Profile {
    pub import_ms: f64,
    pub min_ms: f64,
    pub compute_ms: f64,
    pub export_ms: f64,
    pub states: usize,
    pub weights: usize,
    pub token_sets: usize,
    pub runs: usize,
}
#[derive(Default)]
struct Pool {
    failed: bool,
    total_runs: usize,
    bulk:bool,
    union_rows:Vec<Bits>,
    bits: Vec<Bits>,
    bi: FxHashMap<Bits, u32>,
    values: Vec<Runs>,
    wi: FxHashMap<Runs, u32>,
    single:FxHashMap<Run,u32>,
    token_imports: FxHashMap<usize, (SharedTokenSet, u32)>,
    hashes: FxHashMap<u32, u64>,
    token_and: FxHashMap<(u32,u32),u32>,
    token_or: FxHashMap<(u32,u32),u32>,
    and: FxHashMap<(u32, u32), u32>,
    or: FxHashMap<(u32, u32), u32>,
    source: FxHashMap<usize, (Weight, u32)>,
    exports: FxHashMap<u32, Weight>,
    texports: FxHashMap<u32, SharedTokenSet>,
}
impl Pool {
    fn new() -> Self {
        let mut p = Self::default();
        p.bulk=std::env::var_os("GLRMASK_BOUNDARY_NATIVE_BULK_ALGEBRA").is_some();
        p.intern_bits([0; WORDS]);
        p.values.push(Runs::new());
        p.values.push(Runs::new());
        p.wi.insert(Runs::new(), 0);
        p.exports.insert(0, Weight::empty());
        p.exports.insert(1, Weight::all());
        p
    }
    fn intern_bits(&mut self, b: Bits) -> u32 {
        if let Some(&v) = self.bi.get(&b) {
            return v;
        }
        if self.bits.len() >= MAX_MASKS {
            self.failed = true;
            return 0;
        }
        let v = self.bits.len() as u32;
        self.bits.push(b);
        self.bi.insert(b, v);
        v
    }
    fn intern(&mut self, r: Runs) -> u32 {
        if self.bulk&&r.len()==1{
            let run=r[0];if let Some(&v)=self.single.get(&run){return v}
            if self.values.len()>=MAX_MASKS||self.total_runs>=MAX_RUNS{self.failed=true;return 0}
            self.total_runs+=1;let id=self.values.len()as u32;self.values.push(r);self.single.insert(run,id);return id;
        }
        if let Some(&v) = self.wi.get(&r) {
            return v;
        }
        if self.values.len() >= MAX_MASKS || r.len() > MAX_RUNS - self.total_runs {
            self.failed = true;
            return 0;
        }
        self.total_runs += r.len();
        let v = self.values.len() as u32;
        self.values.push(r.clone());
        self.wi.insert(r, v);
        v
    }
    fn emit(out: &mut Runs, lo: u32, hi: u32, b: u32) {
        if b == 0 || lo > hi {
            return;
        }
        if let Some(last) = out.last_mut() {
            if last.hi.checked_add(1) == Some(lo) && last.b == b {
                last.hi = hi;
                return;
            }
        }
        out.push(Run { lo, hi, b })
    }
    fn import(&mut self, w: &Weight) -> Option<u32> {
        if w.is_empty() {
            return Some(0);
        }
        if w.is_full() {
            return Some(1);
        }
        if let Some((_, id)) = self.source.get(&w.ptr_key()) {
            return Some(*id);
        }
        let mut out = Runs::new();
        for (lo, hi, ts) in w.range_entries() {
            let key = Arc::as_ptr(ts) as usize;
            let bid = if let Some((_, id)) = self.token_imports.get(&key) {
                *id
            } else {
                let mut b = [0; WORDS];
                for r in ts.ranges() {
                    if *r.end() as usize >= WORDS * 64 {
                        return None;
                    }
                    for t in r {
                        b[t as usize / 64] |= 1 << (t % 64)
                    }
                }
                let bid = self.intern_bits(b);
                self.token_imports.insert(key, (ts.clone(), bid));
                bid
            };
            Self::emit(&mut out, lo, hi, bid)
        }
        let id = self.intern(out);
        self.source.insert(w.ptr_key(), (w.clone(), id));
        self.exports.entry(id).or_insert_with(|| w.clone());
        Some(id)
    }
    fn token_meet(&mut self,a:u32,b:u32)->u32 {
        if a==0||b==0{return 0}if a==b{return a}
        let key=if a<b{(a,b)}else{(b,a)};
        if let Some(&id)=self.token_and.get(&key){return id}
        let mut bits=[0;WORDS];for k in 0..WORDS{bits[k]=self.bits[a as usize][k]&self.bits[b as usize][k]}
        let id=self.intern_bits(bits);if self.token_and.len()<100_000{self.token_and.insert(key,id);}id
    }
    fn token_join(&mut self,a:u32,b:u32)->u32 {
        if a==0{return b}if a==b||b==0{return a}
        let key=if a<b{(a,b)}else{(b,a)};
        if let Some(&id)=self.token_or.get(&key){return id}
        let mut bits=[0;WORDS];for k in 0..WORDS{bits[k]=self.bits[a as usize][k]|self.bits[b as usize][k]}
        let id=self.intern_bits(bits);if self.token_or.len()<100_000{self.token_or.insert(key,id);}id
    }
    fn meet(&mut self, a: u32, b: u32) -> u32 {
        if a == 0 || b == 0 {
            return 0;
        }
        if a == b || b == 1 {
            return a;
        }
        if a == 1 {
            return b;
        }
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&v) = self.and.get(&key) {
            return v;
        }
        let (mut i, mut j) = (0, 0);
        let mut out = Runs::new();
        while i < self.values[a as usize].len() && j < self.values[b as usize].len() {
            let x = self.values[a as usize][i];
            let y = self.values[b as usize][j];
            let lo = x.lo.max(y.lo);
            let hi = x.hi.min(y.hi);
            if lo <= hi {
                let bid = if x.b == y.b {
                    x.b
                } else {
                    self.token_meet(x.b,y.b)
                };
                Self::emit(&mut out, lo, hi, bid)
            }
            if x.hi <= y.hi {
                i += 1
            }
            if y.hi <= x.hi {
                j += 1
            }
        }
        let id = self.intern(out);
        if self.and.len() < 1_000_000 {
            self.and.insert(key, id);
        }
        id
    }
    fn join(&mut self, a: u32, b: u32) -> u32 {
        if a == 1 || b == 1 {
            return 1;
        }
        if a == b || b == 0 {
            return a;
        }
        if a == 0 {
            return b;
        }
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&v) = self.or.get(&key) {
            return v;
        }
        if self.values[a as usize].len()==1&&self.values[b as usize].len()==1 {
            let x=self.values[a as usize][0];let y=self.values[b as usize][0];
            if x.lo==y.lo&&x.hi==y.hi {
                let bits=self.token_join(x.b,y.b);let id=self.intern(smallvec![Run{lo:x.lo,hi:x.hi,b:bits}]);
                if self.or.len()<100_000{self.or.insert(key,id);}return id;
            }
        }
        if self.bulk {
            let(mut i,mut j)=(0usize,0usize);let mut x=self.values[a as usize].first().copied();let mut y=self.values[b as usize].first().copied();let mut out=Runs::new();
            while x.is_some()||y.is_some(){
                match(x,y){
                    (Some(l),None)=>{Self::emit(&mut out,l.lo,l.hi,l.b);i+=1;x=self.values[a as usize].get(i).copied();},
                    (None,Some(r))=>{Self::emit(&mut out,r.lo,r.hi,r.b);j+=1;y=self.values[b as usize].get(j).copied();},
                    (Some(l),Some(r))if l.hi<r.lo=>{Self::emit(&mut out,l.lo,l.hi,l.b);i+=1;x=self.values[a as usize].get(i).copied();},
                    (Some(l),Some(r))if r.hi<l.lo=>{Self::emit(&mut out,r.lo,r.hi,r.b);j+=1;y=self.values[b as usize].get(j).copied();},
                    (Some(mut l),Some(mut r))=>{
                        if l.lo<r.lo{Self::emit(&mut out,l.lo,r.lo-1,l.b);l.lo=r.lo;}
                        if r.lo<l.lo{Self::emit(&mut out,r.lo,l.lo-1,r.b);r.lo=l.lo;}
                        let end=l.hi.min(r.hi);let bits=self.token_join(l.b,r.b);Self::emit(&mut out,l.lo,end,bits);
                        if l.hi==end{i+=1;x=self.values[a as usize].get(i).copied();}else{l.lo=end+1;x=Some(l);}
                        if r.hi==end{j+=1;y=self.values[b as usize].get(j).copied();}else{r.lo=end+1;y=Some(r);}
                    },_=>break,
                }
            }
            let id=self.intern(out);if self.or.len()<100_000{self.or.insert(key,id);}return id;
        }
        let mut endpoints =
            Vec::with_capacity((self.values[a as usize].len() + self.values[b as usize].len()) * 2);
        for x in self.values[a as usize]
            .iter()
            .chain(&self.values[b as usize])
        {
            endpoints.push(u64::from(x.lo));
            endpoints.push(u64::from(x.hi) + 1)
        }
        endpoints.sort_unstable();
        endpoints.dedup();
        let (mut i, mut j) = (0, 0);
        let mut out = Runs::new();
        for band in endpoints.windows(2) {
            let lo = band[0];
            let hi = band[1] - 1;
            while i < self.values[a as usize].len() && u64::from(self.values[a as usize][i].hi) < lo
            {
                i += 1
            }
            while j < self.values[b as usize].len() && u64::from(self.values[b as usize][j].hi) < lo
            {
                j += 1
            }
            let x = self.values[a as usize]
                .get(i)
                .filter(|r| u64::from(r.lo) <= lo)
                .map_or(0, |r| r.b);
            let y = self.values[b as usize]
                .get(j)
                .filter(|r| u64::from(r.lo) <= lo)
                .map_or(0, |r| r.b);
            let bid = if x == y || y == 0 {
                x
            } else if x == 0 {
                y
            } else {
                self.token_join(x,y)
            };
            Self::emit(&mut out, lo as u32, hi as u32, bid);
        }
        let id = self.intern(out);
        if self.or.len() < 100_000 {
            self.or.insert(key, id);
        }
        id
    }
    fn join_terms(&mut self, terms:impl IntoIterator<Item=u32>)->u32 {
        if !self.bulk{return self.join_all_reference(terms.into_iter().collect())}
        let mut ids:SmallVec<[u32;8]>=SmallVec::new();
        for id in terms {if id==1{return 1}if id!=0{ids.push(id)}}
        if ids.is_empty(){return 0}if ids.len()==1{return ids[0]}
        ids.sort_unstable();ids.dedup();
        if ids.len()==1{return ids[0]}if ids.len()==2{return self.join(ids[0],ids[1])}
        let mut low=u32::MAX;let mut high=0u32;let mut work=0u64;
        for &id in &ids{for r in &self.values[id as usize]{low=low.min(r.lo);high=high.max(r.hi);work+=u64::from(r.hi)-u64::from(r.lo)+1;}}
        let span=u64::from(high).saturating_sub(u64::from(low))+1;
        if span<=8192&&work<=1_000_000&&ids.len()>=8 {
            let mut rows=std::mem::take(&mut self.union_rows);rows.resize(span as usize,[0;8]);rows[..span as usize].fill([0;8]);
            for &id in &ids {for r in &self.values[id as usize]{
                let bits=self.bits[r.b as usize];
                for row in &mut rows[(r.lo-low)as usize..=(r.hi-low)as usize]{for k in 0..8{row[k]|=bits[k];}}
            }}
            let mut out=Runs::new();for (offset,&bits) in rows[..span as usize].iter().enumerate(){
                let bid=self.intern_bits(bits);let raw=low+offset as u32;Self::emit(&mut out,raw,raw,bid);
            }
            self.union_rows=rows;
            return self.intern(out);
        }
        self.join_all_reference(ids.into_vec())
    }
    fn join_all(&mut self, ids: Vec<u32>)->u32 {self.join_terms(ids)}
    fn join_all_reference(&mut self, mut ids: Vec<u32>) -> u32 {
        ids.retain(|&x| x != 0);
        ids.sort_unstable();
        ids.dedup();
        if ids.binary_search(&1).is_ok() {
            return 1;
        }
        if ids.len() > 8 {
            let cells = ids
                .iter()
                .flat_map(|&id| &self.values[id as usize])
                .map(|r| u64::from(r.hi) - u64::from(r.lo) + 1)
                .sum::<u64>();
            if cells <= 16384 {
                let mut entries = Vec::with_capacity(cells as usize);
                for &id in &ids {
                    for r in &self.values[id as usize] {
                        for row in r.lo..=r.hi {
                            entries.push((row, r.b));
                        }
                    }
                }
                entries.sort_unstable_by_key(|x| x.0);
                let mut out = Runs::new();
                let mut i = 0;
                while i < entries.len() {
                    let row = entries[i].0;
                    let mut bits = [0; WORDS];
                    let mut j = i;
                    while j < entries.len() && entries[j].0 == row {
                        let b = &self.bits[entries[j].1 as usize];
                        for k in 0..WORDS {
                            bits[k] |= b[k]
                        }
                        j += 1
                    }
                    let bid = self.intern_bits(bits);
                    Self::emit(&mut out, row, row, bid);
                    i = j
                }
                return self.intern(out);
            }
        }
        while ids.len() > 1 {
            let mut next = Vec::with_capacity(ids.len().div_ceil(2));
            for pair in ids.chunks(2) {
                next.push(if pair.len() == 1 {
                    pair[0]
                } else {
                    self.join(pair[0], pair[1])
                })
            }
            ids = next
        }
        ids.first().copied().unwrap_or(0)
    }
    fn token_export(&mut self, id: u32) -> SharedTokenSet {
        let b = &self.bits[id as usize];
        self.texports
            .entry(id)
            .or_insert_with(|| {
                Arc::new(
                    (0..WORDS * 64)
                        .filter(|&t| b[t / 64] & (1 << (t % 64)) != 0)
                        .map(|t| t as u32)
                        .collect(),
                )
            })
            .clone()
    }
    fn structural_hash(&mut self, id: u32) -> u64 {
        use std::hash::{Hash, Hasher};
        if let Some(&hash) = self.hashes.get(&id) {
            return hash;
        }
        let mut h = rustc_hash::FxHasher::default();
        (id == 1).hash(&mut h);
        if id != 1 {
            for i in 0..self.values[id as usize].len() {
                let r = self.values[id as usize][i];
                (r.lo..=r.hi).hash(&mut h);
                self.token_export(r.b).as_ref().hash(&mut h);
            }
        }
        let hash = h.finish();
        self.hashes.insert(id, hash);
        hash
    }
    fn export(&mut self, id: u32) -> Weight {
        if let Some(w) = self.exports.get(&id) {
            return w.clone();
        }
        let mut runs = Vec::new();
        for &r in &self.values[id as usize] {
            let b = &self.bits[r.b as usize];
            let ts = self.texports.entry(r.b).or_insert_with(|| {
                Arc::new(
                    (0..WORDS * 64)
                        .filter(|&t| b[t / 64] & (1 << (t % 64)) != 0)
                        .map(|t| t as u32)
                        .collect(),
                )
            });
            runs.push((r.lo, r.hi, ts.clone()));
        }
        let w = Weight::from_tsid_runs_shared(runs);
        self.exports.insert(id, w.clone());
        w
    }
}
struct SourceGroup {
    weight: u32,
    edges: Vec<(i32, u32, bool)>,
}
struct State {
    final_w: u32,
    groups: Vec<SourceGroup>,
    eps: Vec<(u32, u32)>,
}
fn closure(states: &[State], p: &mut Pool, seeds: Vec<(u32, u32)>) -> Subset {
    let mut seen = FxHashMap::<u32, u32>::default();
    let mut todo = VecDeque::new();
    for (s, w) in seeds {
        if w != 0 {
            let old = seen.get(&s).copied().unwrap_or(0);
            let nw = p.join(old, w);
            if old != nw {
                seen.insert(s, nw);
                todo.push_back(s);
            }
        }
    }
    while let Some(s) = todo.pop_front() {
        if p.failed {
            break;
        }
        let w = seen[&s];
        for &(t, ew) in &states[s as usize].eps {
            let add = p.meet(w, ew);
            if add == 0 {
                continue;
            }
            let old = seen.get(&t).copied().unwrap_or(0);
            let nw = p.join(old, add);
            if old != nw {
                seen.insert(t, nw);
                todo.push_back(t);
            }
        }
    }
    let mut pairs = seen.into_iter().collect::<Subset>();
    pairs.sort_unstable();
    pairs
}
#[cfg(test)]
fn determinize_sparse(nwa: &NWA) -> Option<(DWA, Profile)> {
    run(nwa, false)
}
pub(super) fn compile_sparse(nwa: &NWA) -> Option<(DWA, Profile)> {
    run(nwa, true)
}
fn policy_supported()->bool { !std::env::vars_os().any(|(key, _)| {
        key.to_str().is_some_and(|key| {
            key.starts_with("GLRMASK_WEIGHTED_MINIMIZE_")
                || key == "GLRMASK_DETERMINIZE_NORMALIZE_SINGLETONS"
                || key == "GLRMASK_DETERMINIZE_NORMALIZE_SINGLETON_MIN_STATES"
                || key == "GLRMASK_DISABLE_MONOTONE_POINTWISE"
                || key == "GLRMASK_POINTWISE_TSID_RANGES"
                || key == "GLRMASK_WEIGHT_UNION_COALESCE_TOKEN_RANGES"
        })
    }) }
fn run(nwa: &NWA, minimize: bool) -> Option<(DWA, Profile)> {
    if !policy_supported(){return None;}
    let n = nwa.states().len();
    if n == 0
        || n >= 16384
        || nwa.start_states().is_empty()
        || nwa.start_states().iter().any(|&s| s as usize >= n)
    {
        return None;
    }
    let mut edge_count = 0usize;
    for s in nwa.states() {
        for (l, bs) in &s.transitions {
            if *l < 0 {
                return None;
            }
            for &(d, _) in bs {
                if d as usize >= n {
                    return None;
                }
                edge_count += 1;
            }
        }
        for &(d, _) in &s.epsilons {
            if d as usize >= n {
                return None;
            }
            edge_count += 1;
        }
    }
    if edge_count > 100_000 || !nwa.is_acyclic() {
        return None;
    }
    // The original generic path normalizes residuals differently when a labelled
    // destination has epsilon arcs. Decline instead of silently changing it.
    for s in nwa.states() {
        for bs in s.transitions.values() {
            for &(t, _) in bs {
                if !nwa.states().get(t as usize)?.epsilons.is_empty() {
                    return None;
                }
            }
        }
    }
    let t = Instant::now();
    let mut p = Pool::new();
    let mut states = Vec::new();
    for s in nwa.states() {
        let fw = if let Some(w) = &s.final_weight {
            p.import(w)?
        } else {
            0
        };
        let mut group_keys = FxHashMap::<usize, usize>::default();
        let mut groups = Vec::<SourceGroup>::new();
        for (&l, bs) in &s.transitions {
            for (dst, w) in bs {
                let wid = p.import(w)?;
                let key = w.ptr_key();
                let gi = if let Some(&id) = group_keys.get(&key) {
                    id
                } else {
                    let id = groups.len();
                    groups.push(SourceGroup {
                        weight: wid,
                        edges: Vec::new(),
                    });
                    group_keys.insert(key, id);
                    id
                };
                groups[gi].edges.push((l, *dst, bs.len() == 1));
            }
        }
        let mut eps = Vec::new();
        for (t, w) in &s.epsilons {
            eps.push((*t, p.import(w)?));
        }
        states.push(State {
            final_w: fw,
            groups,
            eps,
        });
    }
    if p.failed {
        #[cfg(test)]
        eprintln!(
            "SPARSE_BUDGET weights={} bits={} runs={}",
            p.values.len(),
            p.bits.len(),
            p.total_runs
        );
        return None;
    }
    let import_ms=t.elapsed().as_secs_f64()*1000.;
    run_imported(states,nwa.start_states(),p,minimize,import_ms)
}

fn run_imported(states:Vec<State>,starts:&[u32],mut p:Pool,minimize:bool,import_ms:f64)->Option<(DWA,Profile)>{
    let mut prof = Profile {
        import_ms,
        ..Default::default()
    };
    let t = Instant::now();
    let leaf_fused=minimize&&std::env::var_os("GLRMASK_BOUNDARY_NATIVE_EARLY_LEAVES").is_some();
    let mut leaf_id=None::<u32>;let mut leaf_weights=Vec::new();
    let start = closure(
        &states,
        &mut p,
        starts.iter().map(|&s| (s, 1)).collect(),
    );
    let mut subsets = vec![start.clone()];
    let mut ids = FxHashMap::default();
    ids.insert(start, 0u32);
    let mut out = Vec::<(u32,Edges)>::new();
    let mut cursor = 0;
    let mut out_edges = 0usize;
    let mut subset_cells = subsets[0].len();
    let parallel_waves = rayon::current_num_threads() > 1
        && std::env::var_os("GLRMASK_DISABLE_DETERMINIZE_PARALLEL_DIRECT_WAVES").is_none()
        && (states.len() >= 128
            || std::env::var_os("GLRMASK_DETERMINIZE_PARALLEL_DIRECT_WAVES").is_some());
    let mut bylabel = FxHashMap::<i32, Vec<(u32, u32)>>::default();
    while cursor < subsets.len() {
        if subsets.len() > MAX_STATES || p.failed {
            return None;
        }
        if leaf_id==Some(cursor as u32){out.push((0,Edges::new()));cursor+=1;continue}
        let pairs = subsets[cursor].clone();
        let fast = pairs.len() == 1 && !parallel_waves;
        let mut fws = Vec::new();
        let mut edges = Edges::new();
        for (src, w) in pairs {
            let state = &states[src as usize];
            let fw = p.meet(w, state.final_w);
            if fw != 0 {
                fws.push(fw)
            }
            for group in &state.groups {
                let nw = p.meet(w, group.weight);
                if nw == 0 {
                    continue;
                }
                for &(label, dst, direct) in &group.edges {
                    let leaf_target=leaf_fused&&states[dst as usize].groups.iter().all(|g|g.edges.is_empty())&&states[dst as usize].eps.is_empty();
                    let nw=if leaf_target{p.meet(nw,states[dst as usize].final_w)}else{nw};
                    if nw==0{continue}
                    if fast && direct {
                        let key:Subset = smallvec![(dst, nw)];
                        let is_leaf=leaf_target;
                        let dest=if is_leaf{
                            leaf_weights.push(nw);
                            *leaf_id.get_or_insert_with(||{let id=subsets.len()as u32;subsets.push(Subset::new());id})
                        }else if let Some(&v) = ids.get(&key) {
                            v
                        } else {
                            let v = subsets.len() as u32;
                            subset_cells += 1;
                            subsets.push(key.clone());
                            ids.insert(key, v);
                            v
                        };
                        edges.push((label, dest, nw));
                        out_edges += 1;
                    } else {
                        bylabel.entry(label).or_default().push((dst, nw));
                    }
                }
            }
        }
        let fw = p.join_all(fws);
        for (label, mut bs) in bylabel.drain() {
            bs.sort_unstable_by_key(|x| x.0);
            let mut key = Subset::new();
            let mut i = 0;
            while i < bs.len() {
                let mut j = i + 1;
                while j < bs.len() && bs[j].0 == bs[i].0 {
                    j += 1
                }
                let nw = p.join_terms(bs[i..j].iter().map(|x| x.1));
                if nw != 0 {
                    key.push((bs[i].0, nw));
                }
                i = j;
            }
            let ew = p.join_terms(key.iter().map(|x| x.1));
            if ew == 0 {
                continue;
            }
            let is_leaf=leaf_fused&&key.iter().all(|&(q,_)|states[q as usize].groups.iter().all(|g|g.edges.is_empty())&&states[q as usize].eps.is_empty());
            let dest=if is_leaf{
                leaf_weights.push(ew);
                *leaf_id.get_or_insert_with(||{let id=subsets.len()as u32;subsets.push(Subset::new());id})
            }else if let Some(&v) = ids.get(&key) {
                v
            } else {
                let v = subsets.len() as u32;
                subset_cells += key.len();
                subsets.push(key.clone());
                ids.insert(key, v);
                v
            };
            edges.push((label, dest, ew));
            out_edges += 1;
        }
        if out_edges > 1_000_000 || subset_cells > 2_000_000 || p.failed {
            return None;
        }
        edges.sort_unstable_by_key(|x| x.0);
        out.push((fw, edges));
        cursor += 1;
    }

    if p.failed {
        #[cfg(test)]
        eprintln!(
            "SPARSE_BUDGET weights={} bits={} runs={}",
            p.values.len(),
            p.bits.len(),
            p.total_runs
        );
        return None;
    }
    if let Some(id)=leaf_id{out[id as usize].0=p.join_terms(leaf_weights);}
    prof.compute_ms = t.elapsed().as_secs_f64() * 1000.;
    let t = Instant::now();
    let (out, start) = if minimize {
        minimize_sparse_graph(out, &mut p,leaf_id)?
    } else {
        (out, 0)
    };
    prof.min_ms = t.elapsed().as_secs_f64() * 1000.;
    if p.failed {
        #[cfg(test)]
        eprintln!(
            "SPARSE_BUDGET weights={} bits={} runs={}",
            p.values.len(),
            p.bits.len(),
            p.total_runs
        );
        return None;
    }
    let t = Instant::now();
    let out = out
        .into_iter()
        .map(|(fw, edges)| DWAState {
            final_weight: if fw == 0 { None } else { Some(p.export(fw)) },
            transitions: edges
                .into_iter()
                .map(|(l, d, w)| (l, (d, p.export(w))))
                .collect(),
        })
        .collect::<Vec<_>>();
    prof.export_ms = t.elapsed().as_secs_f64() * 1000.;
    prof.states = out.len();
    prof.weights = p.values.len();
    prof.token_sets = p.bits.len();
    prof.runs = p.values.iter().map(|x|x.len()).sum();
    Some((DWA::from_parts(out, start), prof))
}

struct HashWeight(u64);
impl std::hash::Hash for HashWeight {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        h.write_u64(self.0)
    }
}
type Edges=SmallVec<[(i32,u32,u32);2]>;
type Class=SmallVec<[usize;1]>;
type Graph=Vec<(u32,Edges)>;
#[derive(Default)]
struct Regions {
    values: Vec<(u32, RegionSignature)>,
    ids: FxHashMap<RegionSignature, u32>,
    merges: FxHashMap<(u32, u32), Option<u32>>,
}
impl Regions {
    fn intern(&mut self, profile: RegionSignature, p: &mut Pool) -> u32 {
        if let Some(&id) = self.ids.get(&profile) {
            return id;
        }
        let mut bits = [0; WORDS];
        for &(_, bid) in &profile {
            for k in 0..WORDS {
                bits[k] |= p.bits[bid as usize][k];
            }
        }
        let domain = p.intern_bits(bits);
        if self.values.len() >= 65536 {
            p.failed = true;
            return 0;
        }
        let id = self.values.len() as u32;
        self.values.push((domain, profile.clone()));
        self.ids.insert(profile, id);
        id
    }
    fn merge(&mut self, a: u32, b: u32, p: &mut Pool) -> Option<u32> {
        if a == b {
            return Some(a);
        }
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&r) = self.merges.get(&key) {
            return r;
        }
        let ad = self.values[a as usize].0;
        let bd = self.values[b as usize].0;
        let mut overlap = [0; WORDS];
        for k in 0..WORDS {
            overlap[k] = p.bits[ad as usize][k] & p.bits[bd as usize][k]
        }
        let (mut i, mut j) = (0, 0);
        let mut out = RegionSignature::new();
        let mut valid = true;
        while i < self.values[a as usize].1.len() || j < self.values[b as usize].1.len() {
            let x = self.values[a as usize].1.get(i).copied();
            let y = self.values[b as usize].1.get(j).copied();
            let (l, ab, bb) = match (x, y) {
                (Some(x), Some(y)) if x.0 == y.0 => {
                    i += 1;
                    j += 1;
                    (x.0, x.1, y.1)
                }
                (Some(x), Some(y)) if x.0 < y.0 => {
                    i += 1;
                    (x.0, x.1, 0)
                }
                (Some(_), Some(y)) => {
                    j += 1;
                    (y.0, 0, y.1)
                }
                (Some(x), None) => {
                    i += 1;
                    (x.0, x.1, 0)
                }
                (None, Some(y)) => {
                    j += 1;
                    (y.0, 0, y.1)
                }
                _ => break,
            };
            if ab != bb
                && (0..WORDS)
                    .any(|k| (p.bits[ab as usize][k] ^ p.bits[bb as usize][k]) & overlap[k] != 0)
            {
                valid = false;
                break;
            }
            let bits = if ab == bb || bb == 0 {
                ab
            } else if ab == 0 {
                bb
            } else {
                p.token_join(ab,bb)
            };
            out.push((l, bits));
        }
        let result = if valid {
            Some(self.intern(out, p))
        } else {
            None
        };
        if self.merges.len() < 1_000_000 {
            self.merges.insert(key, result);
        }
        result
    }
}

enum RowMap {
    Dense(Vec<u32>),
    Sparse(FxHashMap<u64, u32>),
}
impl RowMap {
    fn new(slots: Option<usize>) -> Self {
        slots
            .map(|n| Self::Dense(vec![u32::MAX; n]))
            .unwrap_or_else(|| Self::Sparse(FxHashMap::default()))
    }
    fn from_entries(slots: Option<usize>, entries: impl IntoIterator<Item = (u64, u32)>) -> Self {
        let mut m = Self::new(slots);
        m.extend(entries);
        m
    }
    fn get(&self, row: &u64) -> Option<&u32> {
        match self {
            Self::Dense(v) => v.get(*row as usize).filter(|&&v| v != u32::MAX),
            Self::Sparse(m) => m.get(row),
        }
    }
    fn extend(&mut self, entries: impl IntoIterator<Item = (u64, u32)>) {
        match self {
            Self::Dense(v) => {
                for (row, value) in entries {
                    v[row as usize] = value;
                }
            }
            Self::Sparse(m) => m.extend(entries),
        }
    }
    fn into_iter(self) -> std::vec::IntoIter<(u64, u32)> {
        match self {
            Self::Dense(v) => v
                .into_iter()
                .enumerate()
                .filter_map(|(i, r)| (r != u32::MAX).then_some((i as u64, r)))
                .collect::<Vec<_>>()
                .into_iter(),
            Self::Sparse(m) => m.into_iter().collect::<Vec<_>>().into_iter(),
        }
    }
}

fn minimize_sparse_graph(mut graph: Graph, p: &mut Pool,leaf:Option<u32>) -> Option<(Graph, u32)> {
    use rustc_hash::FxHasher;
    use std::hash::{Hash, Hasher};
    let reuse_scratch=std::env::var_os("GLRMASK_BOUNDARY_NATIVE_MIN_SCRATCH").is_some();
    let profiling = std::env::var_os("GLRMASK_PROFILE_BOUNDARY_NATIVE_TERMINAL").is_some();
    let total = Instant::now();
    let mut phase = Instant::now();
    let n = graph.len();
    let mut indeg = vec![0usize; n];
    for (_, es) in &graph {
        for &(_, d, _) in es {
            *indeg.get_mut(d as usize)? += 1
        }
    }
    let mut queue = (0..n).filter(|&s| indeg[s] == 0).collect::<VecDeque<_>>();
    let mut topo = Vec::new();
    while let Some(s) = queue.pop_front() {
        topo.push(s);
        for &(_, d, _) in &graph[s].1 {
            indeg[d as usize] -= 1;
            if indeg[d as usize] == 0 {
                queue.push_back(d as usize)
            }
        }
    }
    if topo.len() != n {
        return None;
    }
    let mut needed=vec![0u32;n];
    for &s in topo.iter().rev() {
        if reuse_scratch {
        let mut ws: SmallVec<[u32; 8]> = smallvec![graph[s].0];
        let mut kept = 0;
        for index in 0..graph[s].1.len() {
            let (l,d,w) = graph[s].1[index];
            let nw=if leaf==Some(d){w}else{p.meet(w,needed[d as usize])};
            if nw != 0 {
                ws.push(nw);
                graph[s].1[kept] = (l,d,nw);
                kept += 1;
            }
        }
        graph[s].1.truncate(kept);
        needed[s] = p.join_terms(ws);
        } else {
        let mut ws = vec![graph[s].0];
        let mut es = Edges::new();
        for &(l, d, w) in &graph[s].1 {
            let nw=if leaf==Some(d){w}else{p.meet(w,needed[d as usize])};
            if nw != 0 {
                ws.push(nw);
                es.push((l, d, nw));
            }
        }
        graph[s].1 = es;
        needed[s] = p.join_all(ws);
        }
        if needed[s] == 1 {
            return None;
        }
    }
    if profiling {
        eprintln!("MIN push_ms={:.3}", phase.elapsed().as_secs_f64() * 1000.);
    }
    phase = Instant::now();
    if needed[0] == 0 {
        return Some((vec![(0, Edges::new())], 0));
    }
    let mut heights = vec![0usize; n];
    for &s in topo.iter().rev() {
        heights[s] = graph[s]
            .1
            .iter()
            .map(|&(_, d, _)| heights[d as usize] + 1)
            .max()
            .unwrap_or(0)
    }
    let mut live = vec![false; n];
    let mut todo = vec![0usize];
    while let Some(s) = todo.pop() {
        if std::mem::replace(&mut live[s], true) {
            continue;
        }
        todo.extend(graph[s].1.iter().map(|&(_, d, _)| d as usize));
    }
    let mut levels = vec![Vec::new(); *heights.iter().max()? + 1];
    for s in 0..n {
        if live[s] && needed[s] != 0 {
            levels[heights[s]].push(s)
        }
    }
    let mut mapped = vec![u32::MAX; n];
    let mut output = Vec::new();
    let mut point_budget = 0;
    let slots = needed
        .iter()
        .filter(|&&w| w > 1)
        .flat_map(|&w| &p.values[w as usize])
        .map(|r| r.hi)
        .max()
        .filter(|&hi| hi < 8192)
        .map(|hi| hi as usize + 1);
    let target_slots=graph.iter().flat_map(|(_,edges)|edges.iter().map(|&(l,_,_)|l)).max().filter(|&l|l>=0&&l<4096).map(|l|l as usize+1);
    for (h, candidates) in levels.iter().enumerate() {
        if candidates.is_empty() {
            continue;
        }
        let height_begin = Instant::now();
        let mut classes = Vec::<Class>::new();
        let coverage = |s: usize| {
            p.values[needed[s] as usize]
                .iter()
                .map(|r| u64::from(r.hi) - u64::from(r.lo) + 1)
                .sum::<u64>()
        };
        let small = candidates.len() <= 16
            || (candidates.len() <= 64
                && candidates.iter().map(|&s| coverage(s)).sum::<u64>()
                    <= candidates.len() as u64 * 8);
        if h == 0 {
            classes.push(candidates.iter().copied().collect())
        } else if small {
            classes = candidates.iter().map(|&s| smallvec![s]).collect()
        } else {
            let mut hashes = FxHashMap::<u64, Class>::default();
            for &s in candidates {
                let mut hasher = FxHasher::default();
                let fw = if graph[s].0 == 0 {
                    None
                } else {
                    Some(HashWeight(p.structural_hash(graph[s].0)))
                };
                fw.hash(&mut hasher);
                for &(l, d, w) in &graph[s].1 {
                    l.hash(&mut hasher);
                    mapped[d as usize].hash(&mut hasher);
                    HashWeight(p.structural_hash(w)).hash(&mut hasher);
                }
                graph[s].1.len().hash(&mut hasher);
                hashes.entry(hasher.finish()).or_default().push(s);
            }
            for bucket in hashes.values() {
                let mut subs = Vec::<Class>::new();
                'state: for &s in bucket {
                    for sub in &mut subs {
                        let r = sub[0];
                        if graph[s].0 == graph[r].0
                            && graph[s].1.len() == graph[r].1.len()
                            && graph[s].1.iter().zip(&graph[r].1).all(
                                |(&(la, da, wa), &(lb, db, wb))| {
                                    la == lb
                                        && mapped[da as usize] == mapped[db as usize]
                                        && wa == wb
                                },
                            )
                        {
                            sub.push(s);
                            continue 'state;
                        }
                    }
                    subs.push(smallvec![s]);
                }
                classes.extend(subs);
            }
        }
        let class_ms = height_begin.elapsed().as_secs_f64() * 1000.;
        let group_begin = Instant::now();
        struct Group {
            members: Class,
            targets: RowMap,
            behavior: RowMap,
        }
        let mut regions = Regions::default();
        let mut groups = Vec::<Group>::new();
        let mut observations = Vec::<(i32,u32)>::new();
        let mut profile = Vec::<(u64,u32)>::new();
        let mut targets = Vec::<(i32,u32)>::new();
        let mut merged = Vec::<(u64,u32)>::new();
        for members in classes {
            if p.failed {
                #[cfg(test)]
                eprintln!(
                    "SPARSE_BUDGET weights={} bits={} runs={}",
                    p.values.len(),
                    p.bits.len(),
                    p.total_runs
                );
                return None;
            }
            if h == 0 {
                groups.push(Group {
                    members,
                    targets: RowMap::new(target_slots),
                    behavior: RowMap::new(None),
                });
                continue;
            }
            let s = members[0];
            // The reference policy drops scratch storage after each class;
            // the selected policy keeps only its capacity, never its contents.
            if !reuse_scratch {
                observations=Vec::new(); profile=Vec::new();
                targets=Vec::new(); merged=Vec::new();
            }
            observations.clear();
            observations.extend(std::iter::once((i32::MIN, graph[s].0))
                .chain(graph[s].1.iter().map(|&(l, _, w)| (l, w)))
                .filter(|&(_, w)| w != 0));
            if observations.iter().any(|&(_, w)| w == 1) {
                return None;
            }
            let first = observations.first()?.1 as usize;
            let aligned = observations.iter().all(|&(_, w)| {
                p.values[w as usize].len() == p.values[first].len()
                    && p.values[w as usize]
                        .iter()
                        .zip(&p.values[first])
                        .all(|(a, b)| a.lo == b.lo && a.hi == b.hi)
            });
            profile.clear();
            if aligned {
                for i in 0..p.values[first].len() {
                    let r = p.values[first][i];
                    let count = (u64::from(r.hi) - u64::from(r.lo) + 1) * observations.len() as u64;
                    point_budget += count;
                    if point_budget > 2_000_000 {
                        return None;
                    }
                    let sig = observations
                        .iter()
                        .map(|&(l, w)| (l, p.values[w as usize][i].b))
                        .collect();
                    let reg = regions.intern(sig, p);
                    profile.extend((r.lo..=r.hi).map(|row| (u64::from(row), reg)));
                }
            } else {
                let mut byrow = FxHashMap::<u64, RegionSignature>::default();
                for &(label, w) in &observations {
                    for r in &p.values[w as usize] {
                        let len = u64::from(r.hi) - u64::from(r.lo) + 1;
                        point_budget += len;
                        if point_budget > 2_000_000 {
                            return None;
                        }
                        for row in r.lo..=r.hi {
                            byrow.entry(u64::from(row)).or_default().push((label, r.b));
                        }
                    }
                }
                profile.extend(byrow.into_iter()
                    .map(|(row, sig)| (row, regions.intern(sig, p))));
            }
            targets.clear();
            targets.extend(graph[s].1.iter().map(|&(l,d,_)|(l,mapped[d as usize])));
            let mut selected = None;
            merged.clear();
            'groups: for (i, g) in groups.iter().enumerate() {
                if !targets
                    .iter()
                    .all(|(l, d)| g.targets.get(&(*l as u64)).is_none_or(|old| old == d))
                {
                    continue;
                }
                merged.clear();
                for &(row, reg) in &profile {
                    let next = if let Some(&old) = g.behavior.get(&row) {
                        let Some(next) = regions.merge(old, reg, p) else {
                            continue 'groups;
                        };
                        next
                    } else {
                        reg
                    };
                    merged.push((row, next));
                }
                selected = Some(i);
                break;
            }
            if let Some(i) = selected {
                let g = &mut groups[i];
                g.members.extend(members);
                g.targets.extend(targets.iter().copied().map(|(l,d)|(l as u64,d)));
                g.behavior.extend(merged.iter().copied());
            } else {
                if slots.is_some_and(|n| n.saturating_mul(groups.len() + 1) > 2_000_000) || target_slots.is_some_and(|n| n.saturating_mul(groups.len()+1)>2_000_000) {
                    return None;
                }
                groups.push(Group {
                    members,
                    targets: RowMap::from_entries(target_slots,targets.iter().copied().map(|(l,d)|(l as u64,d))),
                    behavior: RowMap::from_entries(slots, profile.iter().copied()),
                });
            }
        }
        let group_ms = group_begin.elapsed().as_secs_f64() * 1000.;
        let num_groups = groups.len();
        let rebuild = Instant::now();
        for g in groups {
            let id = output.len() as u32;
            for &s in &g.members {
                mapped[s] = id
            }
            if h == 0 {
                let fw = p.join_terms(g.members.iter().map(|&s| graph[s].0));
                output.push((fw, Edges::new()));
                continue;
            }
            let mut rows = g.behavior.into_iter().collect::<Vec<_>>();
            rows.sort_unstable_by_key(|x| x.0);
            let mut fs = Runs::new();
            let mut weights = BTreeMap::<i32, Runs>::new();
            for (row, reg) in rows {
                for &(l, bits) in &regions.values[reg as usize].1 {
                    if l == i32::MIN {
                        Pool::emit(&mut fs, row as u32, row as u32, bits)
                    } else {
                        Pool::emit(weights.entry(l).or_default(), row as u32, row as u32, bits)
                    }
                }
            }
            let fw = p.intern(fs);
            let es = weights
                .into_iter()
                .map(|(l, rs)| (l, *g.targets.get(&(l as u64)).unwrap(), p.intern(rs)))
                .collect();
            output.push((fw, es));
        }
        if profiling {
            eprintln!(
                "MIN height={} candidates={} groups={} class_ms={:.3} group_ms={:.3} rebuild_ms={:.3} total_ms={:.3}",
                h,
                candidates.len(),
                num_groups,
                class_ms,
                group_ms,
                rebuild.elapsed().as_secs_f64() * 1000.,
                height_begin.elapsed().as_secs_f64() * 1000.
            );
        }
    }
    if profiling {
        eprintln!(
            "MIN afterpush_ms={:.3} total_ms={:.3}",
            phase.elapsed().as_secs_f64() * 1000.,
            total.elapsed().as_secs_f64() * 1000.
        );
    }
    Some((output, mapped[0]))
}

/// Structural comparison, deliberately stronger than weighted-language equality.
/// State numbering may differ, but final weights, labelled transitions and every
/// shared target must correspond bijectively, with no unmatched dead states.
pub(super) fn graph_isomorphism(a: &DWA, b: &DWA) -> Result<usize, String> {
    if a.num_states() != b.num_states() {
        return Err(format!(
            "state counts {} != {}",
            a.num_states(),
            b.num_states()
        ));
    }
    let mut forward = FxHashMap::default();
    let mut backward = FxHashMap::default();
    let mut pending = VecDeque::from([(a.start_state(), b.start_state())]);
    while let Some((x, y)) = pending.pop_front() {
        if let Some(&old) = forward.get(&x) {
            if old != y {
                return Err("nonbijective source map".into());
            }
            continue;
        }
        if backward.insert(y, x).is_some() {
            return Err("nonbijective target map".into());
        }
        forward.insert(x, y);
        let left = &a.states()[x as usize];
        let right = &b.states()[y as usize];
        if left.final_weight != right.final_weight {
            return Err(format!("final weights differ at {x}/{y}"));
        }
        if left.transitions.len() != right.transitions.len() {
            return Err(format!("label sets differ at {x}/{y}"));
        }
        for (&label, (to, weight)) in &left.transitions {
            let Some((other_to, other_weight)) = right.transitions.get(&label) else {
                return Err(format!("missing label {label}"));
            };
            if weight != other_weight {
                return Err(format!("edge weight differs at {x}/{y}, label {label}"));
            }
            pending.push_back((*to, *other_to));
        }
    }
    if forward.len() != a.num_states() as usize {
        return Err("unmatched unreachable states".into());
    }
    Ok(forward.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glrmask_weighted_automata::weighted_u32::{
        determinize::determinize, equivalence::find_difference, minimize::minimize_owned,
    };
    fn next(seed: &mut u64) -> usize {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*seed >> 32) as usize
    }
    fn weight(seed: &mut u64) -> Weight {
        match next(seed) % 12 {
            0 => return Weight::empty(),
            1 => return Weight::all(),
            _ => {}
        }
        let rows = [0u32, 1, 2, 7, 8, 128, 4096, u32::MAX - 1, u32::MAX];
        let toks = [0u32, 1, 7, 63, 64, 65, 127, 128, 255, 256, 510, 511];
        Weight::from_per_tsid_token_sets(rows.into_iter().filter_map(|r| {
            let t = toks
                .into_iter()
                .filter(|_| next(seed) % 3 == 0)
                .collect::<range_set_blaze::RangeSetBlaze<_>>();
            (!t.is_empty()).then_some((r, t))
        }))
    }
    #[test]
    fn exact_algebra_and_hashes_correlated_sparse_extremes() {
        let mut seed = 190257;
        let mut p = Pool::new();
        for case in 0..512 {
            let a = weight(&mut seed);
            let b = weight(&mut seed);
            let ai = p.import(&a).unwrap();
            let bi = p.import(&b).unwrap();
            assert_eq!(p.export(ai), a);
            assert_eq!(p.export(bi), b);
            assert_eq!(p.structural_hash(ai), a.structural_hash_cached());
            assert_eq!(p.structural_hash(bi), b.structural_hash_cached());
            let meet = p.meet(ai, bi);
            let union = p.join(ai, bi);
            assert_eq!(p.export(meet), a.intersection(&b), "intersection{case}");
            assert_eq!(p.export(union), a.union(&b), "union{case}");
            let words = (0..20).map(|_| weight(&mut seed)).collect::<Vec<_>>();
            let ids = words.iter().map(|w| p.import(w).unwrap()).collect();
            let all = p.join_all(ids);
            assert_eq!(p.export(all), Weight::union_all(&words));
            assert!(!p.failed);
        }
    }
    #[test]
    fn bulk_union_preserves_dense_overlaps_and_extreme_coordinates(){
        for base in [0u32,u32::MAX-1024]{for count in [3usize,8,17,128]{
            let mut pool=Pool::new();pool.bulk=true;
            let inputs=(0..count).map(|i|Weight::from_uniform(
                base+(i%64)as u32..=base+(i%64)as u32+64,
                [0u32,63,64,127,255,256,511].into_iter().filter(|t|(*t as usize+i)%3!=0).collect()
            )).collect::<Vec<_>>();
            let ids=inputs.iter().map(|w|pool.import(w).unwrap()).collect::<Vec<_>>();
            let merged=pool.join_terms(ids.clone());assert_eq!(pool.export(merged),Weight::union_all(&inputs));
            let reversed=pool.join_terms(ids.into_iter().rev());assert_eq!(merged,reversed);
            assert!(!pool.failed);assert!(pool.union_rows.len()<=8192);
        }}
        let mut pool=Pool::new();pool.bulk=true;
        let left=Weight::from_uniform(0..=u32::MAX,[0u32,64].into_iter().collect());
        let right=Weight::from_uniform(u32::MAX..=u32::MAX,[1u32,511].into_iter().collect());
        let a=pool.import(&left).unwrap();let b=pool.import(&right).unwrap();let c=pool.join(a,b);
        assert_eq!(pool.export(c),left.union(&right));
    }
    #[test]
    fn region_overlay_matches_exact_bitwise_partial_functions() {
        let mut seed = 2519;
        let mut p = Pool::new();
        let mut r = Regions::default();
        for case in 0..512 {
            let mut side = || {
                (0..5)
                    .filter_map(|label| {
                        let mut bits = [0; WORDS];
                        bits[0] = (next(&mut seed) % 65536) as u64;
                        bits[4] = (next(&mut seed) % 65536) as u64;
                        let id = p.intern_bits(bits);
                        (id != 0).then_some((label, id))
                    })
                    .collect::<Vec<_>>()
            };
            let a = side();
            let b = side();
            let ai = r.intern(a.clone().into(), &mut p);
            let bi = r.intern(b.clone().into(), &mut p);
            let result = r.merge(ai, bi, &mut p);
            let da = p.bits[r.values[ai as usize].0 as usize];
            let db = p.bits[r.values[bi as usize].0 as usize];
            let mut compatible = true;
            for tok in 0..WORDS * 64 {
                if da[tok / 64] & db[tok / 64] & (1 << (tok % 64)) == 0 {
                    continue;
                }
                let obs = |v: &Vec<(i32, u32)>| {
                    v.iter()
                        .filter(|(_, b)| p.bits[*b as usize][tok / 64] & (1 << (tok % 64)) != 0)
                        .map(|(l, _)| *l)
                        .collect::<Vec<_>>()
                };
                if obs(&a) != obs(&b) {
                    compatible = false;
                }
            }
            assert_eq!(result.is_some(), compatible, "case{case}");
            if let Some(c) = result {
                let expected = da.iter().zip(db).map(|(x, y)| x | y).collect::<Vec<_>>();
                assert_eq!(p.bits[r.values[c as usize].0 as usize].as_slice(), expected);
            }
        }
    }
    #[test]
    fn generated_nwas_preserve_determinized_graph_and_weighted_language() {
        let mut seed = 129851;
        let mut accepted = 0;
        let mut iso_min = 0;
        let mut shape_diff = 0;
        let mut first_shape = None;
        for case in 0..256 {
            let n = 4 + next(&mut seed) % 9;
            let mut nwa = NWA::new(0, 0);
            for _ in 0..n {
                nwa.add_state();
            }
            nwa.set_start_states(vec![0]);
            for dst in 1..n {
                if next(&mut seed) % 3 == 0 {
                    let w = weight(&mut seed);
                    nwa.add_epsilon(0, dst as u32, w);
                }
            }
            for s in 0..n {
                if next(&mut seed) % 2 == 0 {
                    let w = weight(&mut seed);
                    nwa.set_final_weight(s as u32, w)
                }
                for d in s + 1..n {
                    if next(&mut seed) % 3 != 0 {
                        continue;
                    }
                    for label in 0..3 {
                        if next(&mut seed) % 2 == 0 {
                            nwa.add_transition(s as u32, label, d as u32, weight(&mut seed));
                        }
                    }
                }
            }
            let original = (nwa.start_states().to_vec(), nwa.states().to_vec());
            let reference = determinize(&nwa).unwrap();
            let (native, _) = determinize_sparse(&nwa).unwrap();
            assert!(
                graph_isomorphism(&native, &reference).is_ok(),
                "determinantgraph case{case}: {:?}",
                graph_isomorphism(&native, &reference)
            );
            if let Some((fused, _)) = compile_sparse(&nwa) {
                let baseline = minimize_owned(reference);
                assert_eq!(
                    find_difference(&fused, &baseline).unwrap(),
                    None,
                    "language case{case}"
                );
                accepted += 1;
                if graph_isomorphism(&fused, &baseline).is_ok() {
                    iso_min += 1
                } else {
                    shape_diff += 1;
                    if first_shape.is_none() {
                        first_shape = Some(case)
                    }
                }
            }
            assert_eq!(
                original,
                (nwa.start_states().to_vec(), nwa.states().to_vec())
            );
        }
        eprintln!(
            "GENERATED256 eligiblemin={accepted} isomorphicmin={iso_min} shapedifferent={shape_diff} first={first_shape:?}"
        );
        assert!(accepted >= 100);
    }
    #[test]
    fn wide_layered_nwas_match_original_constructor() {
        let mut seed = 672955;
        let mut eligible = 0;
        let mut iso_count = 0;
        let mut maxdet = 0;
        let mut first = None;
        for case in 0..32 {
            let width = 8 + case % 17;
            let layers = 6;
            let n = 1 + width * layers;
            let mut nwa = NWA::new(0, 0);
            for _ in 0..n {
                nwa.add_state();
            }
            nwa.set_start_states(vec![0]);
            for k in 0..width {
                let rows = [0u32, 1, 2, 3, 4, 5, 6, 7];
                let ts = Weight::from_per_tsid_token_sets(rows.into_iter().filter_map(|r| {
                    if next(&mut seed) % 3 == 0 {
                        return None;
                    }
                    Some((r, (0..8u32).filter(|_| next(&mut seed) % 3 != 0).collect()))
                }));
                nwa.add_epsilon(0, 1 + k as u32, ts);
            }
            for level in 0..layers {
                for k in 0..width {
                    let st = 1 + level * width + k;
                    let finalw = Weight::from_uniform(
                        0..=7,
                        (0..8u32).filter(|_| next(&mut seed) % 3 != 0).collect(),
                    );
                    if level == layers - 1 || next(&mut seed) % 3 == 0 {
                        nwa.set_final_weight(st as u32, finalw);
                    }
                    if level + 1 == layers {
                        continue;
                    }
                    for label in 0..4 {
                        for _ in 0..2 {
                            let dst = 1 + (level + 1) * width + next(&mut seed) % width;
                            let ew = Weight::from_uniform(
                                0..=7,
                                (0..8u32).filter(|_| next(&mut seed) % 3 != 0).collect(),
                            );
                            nwa.add_transition(st as u32, label, dst as u32, ew);
                        }
                    }
                }
            }
            let old = determinize(&nwa).unwrap();
            maxdet = maxdet.max(old.num_states());
            let (det, _) = determinize_sparse(&nwa).expect("widelayered eligibility");
            assert!(
                graph_isomorphism(&det, &old).is_ok(),
                "wide det{case}: {:?}",
                graph_isomorphism(&det, &old)
            );
            if let Some((fused, _)) = compile_sparse(&nwa) {
                eligible += 1;
                let baseline = minimize_owned(old);
                assert_eq!(
                    find_difference(&fused, &baseline).unwrap(),
                    None,
                    "wide language{case}"
                );
                if graph_isomorphism(&fused, &baseline).is_ok() {
                    iso_count += 1
                } else if first.is_none() {
                    first = Some(case);
                }
            }
        }
        eprintln!(
            "WIDE32 eligible={eligible} graph_isomorphic={iso_count} maxdet={maxdet} first_shape_difference={first:?}"
        );
        assert_eq!(eligible, 32);
        assert_eq!(iso_count, eligible);
    }
    #[test]
    fn failure_budgets_do_not_publish_partial_output() {
        let mut p = Pool::new();
        p.total_runs = MAX_RUNS;
        let id = p.intern(smallvec![Run { lo: 0, hi: 0, b: 0 }]);
        assert_eq!(id, 0);
        assert!(p.failed);
        let mut n = NWA::new(0, 0);
        n.add_state();
        n.set_start_states(vec![0]);
        n.add_transition(0, 7, 100, Weight::all());
        assert!(compile_sparse(&n).is_none());
        let mut n = NWA::new(0, 0);
        n.add_state();
        n.set_start_states(vec![99]);
        assert!(compile_sparse(&n).is_none());
    }
    #[test]
    fn unsupported_inputs_decline_without_mutation() {
        let mut n = NWA::new(0, 0);
        n.add_state();
        n.add_state();
        n.add_state();
        n.set_start_states(vec![0]);
        n.set_final_weight(
            2,
            Weight::from_token_set_for_tsid(0, [1].into_iter().collect()),
        );
        n.add_transition(0, 7, 1, Weight::all());
        n.add_epsilon(1, 2, Weight::all());
        assert!(compile_sparse(&n).is_none());
        let mut cycle = NWA::new(0, 0);
        cycle.add_state();
        cycle.set_start_states(vec![0]);
        cycle.add_transition(0, 0, 0, Weight::all());
        assert!(compile_sparse(&cycle).is_none());
        let mut high = NWA::new(0, 0);
        high.add_state();
        high.add_state();
        high.set_start_states(vec![0]);
        high.add_transition(
            0,
            0,
            1,
            Weight::from_token_set_for_tsid(0, [512].into_iter().collect()),
        );
        assert!(compile_sparse(&high).is_none());
        let mut empty = NWA::new(0, 0);
        empty.add_state();
        assert!(compile_sparse(&empty).is_none());
    }
}

#[path="native_pipeline_post.rs"] pub(super) mod post;
