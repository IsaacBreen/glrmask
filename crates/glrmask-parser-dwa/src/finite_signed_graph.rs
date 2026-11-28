//! Exact read-only adjacency over an eager or factored signed template program.
//!
//! Logical IDs, row/key order, final presence and zero markers match the eager
//! builder. The virtual graph shares existing template TOPOLOGY only; it does
//! not merge parser states or summarize stack semantics.
use super::*;
use super::finite_template_program::FiniteTemplateProgram;
#[cfg(test)]
use super::finite_template_program::FiniteTemplateInstance;
use std::ops::Range;

pub(super) trait SignedRow: Copy {
    fn final_weight(&self) -> FastBoundaryWeightId;
    fn epsilon_count(&self) -> usize;
    fn epsilon(&self,index:usize) -> (u32,FastBoundaryWeightId);
    fn transition_count(&self) -> usize;
    fn label(&self,index:usize) -> i32;
    fn branch_count(&self,index:usize) -> usize;
    fn branch(&self,index:usize,branch:usize) -> (u32,FastBoundaryWeightId);
    fn find_label(&self,label:i32) -> Option<usize> {
        let(mut low,mut high)=(0,self.transition_count());
        while low<high {
            let mid=low+(high-low)/2;
            match self.label(mid).cmp(&label) {
                std::cmp::Ordering::Less=>low=mid+1,
                std::cmp::Ordering::Greater=>high=mid,
                std::cmp::Ordering::Equal=>return Some(mid),
            }
        }
        None
    }
}

pub(super) trait SignedGraph {
    type Row<'a>:SignedRow where Self:'a;
    fn len(&self) -> usize;
    fn row(&self,index:usize) -> Self::Row<'_>;
    fn class_read_labels(&self) -> &[i32] { &[] }
    fn class_read_matches(&self, _label:i32, _symbol:i32) -> bool { false }
}

impl SignedRow for &FastBoundaryNwaState {
    #[inline] fn final_weight(&self)->u32 {self.final_weight}
    #[inline] fn epsilon_count(&self)->usize {self.epsilons.len()}
    #[inline] fn epsilon(&self,index:usize)->(u32,u32) {self.epsilons[index]}
    #[inline] fn transition_count(&self)->usize {self.transitions.len()}
    #[inline] fn label(&self,index:usize)->i32 {self.transitions[index].0}
    #[inline] fn branch_count(&self,index:usize)->usize {self.transitions[index].1.len()}
    #[inline] fn branch(&self,index:usize,branch:usize)->(u32,u32) {self.transitions[index].1[branch]}
    #[inline] fn find_label(&self,label:i32)->Option<usize> {
        self.transitions.binary_search_by_key(&label,|(label,_)|*label).ok()
    }
}

impl SignedGraph for [FastBoundaryNwaState] {
    type Row<'a>=&'a FastBoundaryNwaState;
    #[inline] fn len(&self)->usize {<[FastBoundaryNwaState]>::len(self)}
    #[inline] fn row(&self,index:usize)->Self::Row<'_> {&self[index]}
}

/// The reference terminal/default fixed point, over a read-only positive view.
/// Negative keys are ignored because the eager resolver removes them first.
/// Zero DEFAULT branches still create dependencies: they cannot be discarded
/// or ordered using the nonzero-edge DAG certificate.
pub(super) fn terminal_default_states<G: SignedGraph + ?Sized>(
    graph: &G,
    pool: &FastBoundaryWeightInterner,
) -> Vec<bool> {
    let mut terminal = vec![false; graph.len()];
    crate::terminal_defaults::close(&mut terminal, |state, dependency| {
        let row = graph.row(state);
        let final_weight = row.final_weight();
        if final_weight == 0 || row.epsilon_count() != 0
            || (0..row.transition_count()).any(|edge| {
                let label = row.label(edge);
                !is_negative_label(label) && label != DEFAULT_LABEL && row.branch_count(edge) != 0
            })
        {
            return false;
        }
        if let Some(edge) = row.find_label(DEFAULT_LABEL) {
            for branch in 0..row.branch_count(edge) {
                let (target, weight) = row.branch(edge, branch);
                if !pool.is_subset(weight, final_weight) {
                    return false;
                }
                dependency(target);
            }
        }
        true
    });
    terminal
}

/// Final acceptance is the weighted union of the initial acceptance and
/// every epsilon, negative, or DEFAULT successor's final acceptance.
/// Traverse the checked order backwards, leaving structural guards intact.
pub(super) fn final_weights<G: SignedGraph + ?Sized>(
    graph: &G,
    pool: &mut FastBoundaryWeightInterner,
    order: &[u32],
) -> Option<Vec<u32>> {
    let n = graph.len();
    if order.len() != n {
        return None;
    }
    let mut finals = (0..n).map(|q| graph.row(q).final_weight()).collect::<Vec<_>>();
    for &q in order.iter().rev() {
        let row = graph.row(q as usize);
        let mut value = finals[q as usize];
        let mut add = |target: u32, weight: u32| {
            if let Some(&target_final) = finals.get(target as usize) {
                let contribution = pool.intersection(weight, target_final);
                value = pool.union(value, contribution);
            }
        };
        for index in 0..row.epsilon_count() {
            let (target, weight) = row.epsilon(index);
            add(target, weight);
        }
        for edge in 0..row.transition_count() {
            let label = row.label(edge);
            if label == DEFAULT_LABEL || is_negative_label(label) {
                for branch in 0..row.branch_count(edge) {
                    let (target, weight) = row.branch(edge, branch);
                    add(target, weight);
                }
            }
        }
        finals[q as usize] = value;
    }
    pool.allow_work(0, n, 0).then_some(finals)
}

struct FinalizedGraph<'a,G:SignedGraph+?Sized>{graph:&'a G,finals:&'a[u32]}
#[derive(Clone,Copy)]
struct FinalizedRow<R:SignedRow>{row:R,final_weight:u32}
impl<R:SignedRow> SignedRow for FinalizedRow<R>{
    #[inline] fn final_weight(&self)->u32{self.final_weight}
    #[inline] fn epsilon_count(&self)->usize{self.row.epsilon_count()}
    #[inline] fn epsilon(&self,index:usize)->(u32,u32){self.row.epsilon(index)}
    #[inline] fn transition_count(&self)->usize{self.row.transition_count()}
    #[inline] fn label(&self,index:usize)->i32{self.row.label(index)}
    #[inline] fn branch_count(&self,index:usize)->usize{self.row.branch_count(index)}
    #[inline] fn branch(&self,index:usize,branch:usize)->(u32,u32){self.row.branch(index,branch)}
    #[inline] fn find_label(&self,label:i32)->Option<usize>{self.row.find_label(label)}
}
impl<G:SignedGraph+?Sized> SignedGraph for FinalizedGraph<'_,G>{
    type Row<'a>=FinalizedRow<G::Row<'a>> where Self:'a;
    fn len(&self)->usize{self.graph.len()}
    fn row(&self,index:usize)->Self::Row<'_>{
        FinalizedRow{row:self.graph.row(index),final_weight:self.finals[index]}
    }
}

#[inline]
fn keep_positive_branch(label:i32,target:u32,weight:u32,final_weight:u32,
    terminal:&[bool],pool:&FastBoundaryWeightInterner,
)->bool {
    label!=DEFAULT_LABEL || final_weight==0 || target as usize>=terminal.len()
        || !terminal[target as usize] || !pool.is_subset(weight,final_weight)
}

/// Matches eager Kahn ordering, including transition-before-epsilon traversal.
pub(super) fn topological_order<G:SignedGraph+?Sized>(graph:&G)->Option<Vec<u32>> {
    let n=graph.len();
    let mut indegree=vec![0usize;n];
    for q in 0..n {
        let row=graph.row(q);
        for edge in 0..row.transition_count() {
            for branch in 0..row.branch_count(edge) {
                let(target,weight)=row.branch(edge,branch);
                if target as usize>=n{return None;}
                if weight!=0 {indegree[target as usize]=indegree[target as usize].checked_add(1)?;}
            }
        }
        for edge in 0..row.epsilon_count() {
            let(target,weight)=row.epsilon(edge);
            if target as usize>=n{return None;}
            if weight!=0 {indegree[target as usize]=indegree[target as usize].checked_add(1)?;}
        }
    }
    let mut queue=(0..n).filter(|&q|indegree[q]==0).map(|q|q as u32).collect::<VecDeque<_>>();
    let mut order=Vec::with_capacity(n);
    while let Some(q)=queue.pop_front() {
        order.push(q);
        let row=graph.row(q as usize);
        for edge in 0..row.transition_count() {
            for branch in 0..row.branch_count(edge) {
                let(target,weight)=row.branch(edge,branch);
                if weight==0 {continue;}
                indegree[target as usize]-=1;
                if indegree[target as usize]==0 {queue.push_back(target);}
            }
        }
        for edge in 0..row.epsilon_count() {
            let(target,weight)=row.epsilon(edge);
            if weight==0 {continue;}
            indegree[target as usize]-=1;
            if indegree[target as usize]==0 {queue.push_back(target);}
        }
    }
    (order.len()==n).then_some(order)
}

#[derive(Clone, Debug)]
struct TemplateRow {
    epsilons: Range<usize>,
    transitions: Range<usize>,
    present_final: bool,
}

#[derive(Clone, Debug)]
pub(super) struct TemplateTransition {
    label: i32,
    targets: Range<usize>,
}

/// A reusable signed action-word topology. Coefficients belong to instances,
/// not to its edges; an accepting state denotes a continuation port.
#[derive(Clone, Debug)]
pub struct TemplateTopology {
    rows: Vec<TemplateRow>,
    epsilons: Vec<u32>,
    transitions: Vec<TemplateTransition>,
    targets: Vec<u32>,
    starts: Vec<u32>,
    logical_edges: usize,
}

pub(super) struct TemplateRowRef<'a> {
    pub epsilons: &'a [u32],
    pub transitions: &'a [TemplateTransition],
    targets: &'a [u32],
    pub present_final: bool,
}

impl TemplateRowRef<'_> {
    pub fn edges(&self) -> impl Iterator<Item = (i32, &[u32])> + '_ {
        self.transitions.iter().map(|edge| (edge.label, &self.targets[edge.targets.clone()]))
    }
}

impl TemplateTopology {
    fn new(starts: Vec<u32>, rows: usize) -> Self {
        Self { rows: Vec::with_capacity(rows), epsilons: Vec::new(), transitions: Vec::new(),
            targets: Vec::new(), starts, logical_edges: 0 }
    }

    fn push_row<I, J>(&mut self, final_state: bool, epsilons: impl IntoIterator<Item = u32>, edges: I)
    where I: IntoIterator<Item = (i32, J)>, J: IntoIterator<Item = u32> {
        let eps = self.epsilons.len();
        self.epsilons.extend(epsilons);
        let transitions = self.transitions.len();
        let branches = self.targets.len();
        for (label, targets) in edges {
            let start = self.targets.len();
            self.targets.extend(targets);
            self.transitions.push(TemplateTransition { label, targets: start..self.targets.len() });
        }
        self.logical_edges += self.epsilons.len() - eps + self.targets.len() - branches
            + usize::from(final_state);
        self.rows.push(TemplateRow { epsilons: eps..self.epsilons.len(),
            transitions: transitions..self.transitions.len(), present_final: final_state });
    }

    pub fn from_dfa(source: &UnweightedDfa) -> Self {
        let mut graph = Self::new(vec![source.start_state], source.states.len());
        for row in &source.states {
            graph.push_row(row.is_accepting, std::iter::empty(), row.transitions.iter()
                .map(|(&label, &target)| (label, std::iter::once(target))));
        }
        graph
    }

    pub fn from_nfa(source: &crate::automata::unweighted_u32::nfa::NFA) -> Self {
        let mut graph = Self::new(source.start_states.clone(), source.states.len());
        for row in &source.states {
            graph.push_row(row.is_accepting, row.epsilons.iter().copied(), row.transitions.iter()
                .map(|(&label, targets)| (label, targets.iter().copied())));
        }
        graph
    }

    /// Compatibility with expanded reference fixtures: all edge weights and
    /// final values are intentionally ignored, but empty label keys remain.
    pub fn from_weighted_skeleton(source: &NWA) -> Self {
        let mut graph = Self::new(source.start_states().to_vec(), source.states().len());
        for row in source.states() {
            graph.push_row(row.final_weight.is_some(), row.epsilons.iter().map(|&(q, _)| q),
                row.transitions.iter().map(|(&label, edges)| (label, edges.iter().map(|&(q, _)| q))));
        }
        graph
    }

    #[inline]
    pub fn state_count(&self) -> usize { self.rows.len() }
    pub fn edge_count(&self) -> usize { self.epsilons.len() + self.targets.len() }
    pub fn start_states(&self) -> &[u32] { &self.starts }
    pub(super) fn logical_edges(&self) -> usize { self.logical_edges }

    pub(super) fn row(&self, index: usize) -> TemplateRowRef<'_> {
        let row = &self.rows[index];
        TemplateRowRef { epsilons: &self.epsilons[row.epsilons.clone()],
            transitions: &self.transitions[row.transitions.clone()], targets: &self.targets,
            present_final: row.present_final }
    }

    fn valid_coordinates(&self) -> bool {
        let n = self.state_count();
        self.starts.iter().chain(&self.epsilons).chain(&self.targets).all(|&q| (q as usize) < n)
    }

    pub(super) fn valid_alphabet(&self, alphabet: u32) -> bool {
        self.valid_coordinates() && self.transitions.iter().all(|edge| {
            let label = edge.label;
            label == DEFAULT_LABEL || (0..alphabet as i32).contains(&label)
                || (is_negative_label(label) && (0..alphabet as i32).contains(&negative_to_positive_label(label)))
        })
    }

    pub fn is_acyclic(&self) -> bool {
        if !self.valid_coordinates() { return false; }
        let mut indegree = vec![0usize; self.state_count()];
        for &target in self.epsilons.iter().chain(&self.targets) {
            indegree[target as usize] += 1;
        }
        let mut queue = indegree.iter().enumerate().filter(|(_, degree)| **degree == 0)
            .map(|(q, _)| q).collect::<VecDeque<_>>();
        let mut visited = 0;
        while let Some(q) = queue.pop_front() {
            visited += 1;
            let row = self.row(q);
            for &target in row.epsilons.iter().chain(row.edges().flat_map(|(_, targets)| targets.iter())) {
                indegree[target as usize] -= 1;
                if indegree[target as usize] == 0 { queue.push_back(target as usize); }
            }
        }
        visited == self.state_count()
    }

    /// An injective change of label coordinate, not an alphabet quotient.
    pub fn remap_labels(&mut self, mut map: impl FnMut(i32) -> Option<i32>) -> Option<()> {
        for row in &self.rows {
            let edges = &mut self.transitions[row.transitions.clone()];
            for edge in edges.iter_mut() { edge.label = map(edge.label)?; }
            edges.sort_unstable_by_key(|edge| edge.label);
            if edges.windows(2).any(|pair| pair[0].label == pair[1].label) { return None; }
        }
        Some(())
    }

    pub(super) fn append_mapped(&self, arena: &mut NWA, weight: &Weight, continuation: Option<u32>,
        mut map: impl FnMut(i32) -> i32) -> Result<NwaBody, String> {
        let offset = arena.states().len();
        if offset.checked_add(self.state_count()).is_none_or(|n| n > u32::MAX as usize)
            || continuation.is_some_and(|q| q as usize >= offset)
            || !self.valid_coordinates() {
            return Err("invalid or overflowing signed template coordinate".into());
        }
        let offset = offset as u32;
        let states = arena.states_mut();
        states.reserve(self.state_count());
        for q in 0..self.state_count() {
            let row = self.row(q);
            let mut epsilons = row.epsilons.iter().map(|&q| (offset + q, weight.clone())).collect::<Vec<_>>();
            let final_weight = if row.present_final {
                if let Some(target) = continuation {
                    epsilons.push((target, weight.clone()));
                    None
                } else { Some(weight.clone()) }
            } else { None };
            let transitions = row.edges().map(|(label, targets)| (map(label),
                targets.iter().map(|&q| (offset + q, weight.clone())).collect())).collect();
            states.push(NWAState { epsilons, transitions, final_weight });
        }
        Ok(NwaBody { start_states: self.starts.iter().map(|&q| offset + q).collect() })
    }

    /// Expand only at the range-weight fallback/reference boundary.
    pub fn append_weighted(&self, arena: &mut NWA, weight: &Weight, continuation: u32)
        -> Result<NwaBody, String> {
        self.append_mapped(arena, weight, Some(continuation), |label| label)
    }

    pub fn to_weighted_skeleton(&self) -> NWA {
        let mut graph = NWA::new(0, 0);
        let body = self.append_mapped(&mut graph, &Weight::all(), None, |label| label)
            .expect("valid template topology");
        graph.set_start_states(body.start_states);
        graph
    }
}

struct Instance {template:usize,base:u32,coefficient:u32,continuation:u32}

#[derive(Debug,Default)]
pub(super) struct VirtualGraphProfile {
    pub logical_states:usize,
    pub logical_edges:usize,
    pub stored_template_states:usize,
    pub stored_template_edges:usize,
    pub instances:usize,
    pub port_edges:usize,
}

#[derive(Debug,Default)]
pub(super) struct VirtualResolveProfile {
    pub cancellation_ms:f64,
    pub finality_ms:f64,
    pub materialize_ms:f64,
    pub prune_ms:f64,
    pub reachability_ms:f64,
    pub read_context_ms:f64,
    pub context_input_states:usize,
    pub context_allocated_states:usize,
    pub materialized_states:usize,
    pub materialized_edges:usize,
}

pub(super) struct PositiveMaterialization {
    pub states:Vec<FastBoundaryNwaState>,
    pub starts:Vec<u32>,
    pub topology:Vec<u32>,
    pub profile:VirtualResolveProfile,
}

pub(super) struct VirtualSignedGraph<'a> {
    alphabet:u32,
    templates:&'a [&'a TemplateTopology],instances:Vec<Instance>,ports:Vec<FastBoundaryNwaState>,
    node_instance:Vec<u32>,
    pop_classes:Option<crate::pop_classes::PopLabelClasses>,
    class_reads:Vec<i32>,
    pub profile:VirtualGraphProfile,
}

#[derive(Clone,Copy)]
enum RowKind<'a> {
    Port(&'a FastBoundaryNwaState),
    Template {
        epsilons:&'a[u32],transitions:&'a[TemplateTransition],targets:&'a[u32],
        base:u32,coefficient:u32,continuation:Option<u32>,
    },
}

#[derive(Clone,Copy)]
pub(super) struct VirtualRow<'a> {kind:RowKind<'a>}

impl SignedRow for VirtualRow<'_> {
    #[inline] fn final_weight(&self)->u32 {
        match self.kind {RowKind::Port(row)=>row.final_weight,RowKind::Template{..}=>0}
    }
    #[inline] fn epsilon_count(&self)->usize {
        match self.kind {RowKind::Port(row)=>row.epsilons.len(),
            RowKind::Template{epsilons,continuation,..}=>epsilons.len()+usize::from(continuation.is_some())}
    }
    #[inline] fn epsilon(&self,index:usize)->(u32,u32) {
        match self.kind {RowKind::Port(row)=>row.epsilons[index],
            RowKind::Template{epsilons,base,coefficient,continuation,..}=> {
                let target=if index<epsilons.len(){base+epsilons[index]}else{continuation.expect("present-final epsilon")};
                (target,coefficient)
            }}
    }
    #[inline] fn transition_count(&self)->usize {
        match self.kind {RowKind::Port(row)=>row.transitions.len(),RowKind::Template{transitions,..}=>transitions.len()}
    }
    #[inline] fn label(&self,index:usize)->i32 {
        match self.kind {RowKind::Port(row)=>row.transitions[index].0,RowKind::Template{transitions,..}=>transitions[index].label}
    }
    #[inline] fn branch_count(&self,index:usize)->usize {
        match self.kind {RowKind::Port(row)=>row.transitions[index].1.len(),
            RowKind::Template{transitions,coefficient,..}=>{
                let n=transitions[index].targets.len();
                if coefficient==0 {usize::from(n!=0)}else{n}
            }}
    }
    #[inline] fn branch(&self,index:usize,branch:usize)->(u32,u32) {
        match self.kind {RowKind::Port(row)=>row.transitions[index].1[branch],
            RowKind::Template{transitions,targets,base,coefficient,..}=>{
                if coefficient==0 {(0,0)}else{(base+targets[transitions[index].targets.start+branch],coefficient)}
            }}
    }
}

impl SignedGraph for VirtualSignedGraph<'_> {
    type Row<'a>=VirtualRow<'a> where Self:'a;
    #[inline] fn len(&self)->usize {self.node_instance.len()}
    fn class_read_labels(&self)->&[i32] { &self.class_reads }
    fn class_read_matches(&self,label:i32,symbol:i32)->bool {
        self.pop_classes.as_ref().is_some_and(|classes| {
            let index = label - classes.symbol_count() as i32;
            symbol >= 0 && index >= 0 && (index as usize) < classes.len()
                && classes.matches(DEFAULT_LABEL-1-index,symbol as u32)
        })
    }
    #[inline] fn row(&self,q:usize)->VirtualRow<'_> {
        if q<self.ports.len(){return VirtualRow{kind:RowKind::Port(&self.ports[q])};}
        let instance=&self.instances[self.node_instance[q]as usize];
        let template=&self.templates[instance.template];
        let row=&template.rows[q-instance.base as usize];
        VirtualRow{kind:RowKind::Template{
            epsilons:&template.epsilons[row.epsilons.clone()],transitions:&template.transitions[row.transitions.clone()],
            targets:&template.targets,base:instance.base,coefficient:instance.coefficient,
            continuation:row.present_final.then_some(instance.continuation),
        }}
    }
}

impl<'a> VirtualSignedGraph<'a> {
    pub fn build_with_pop_classes(program:&FiniteTemplateProgram<'a>,alphabet:u32,
        pool:&mut FastBoundaryWeightInterner,limits:FiniteCompileLimits,
        classes:&crate::pop_classes::PopLabelClasses,
    )->Option<(Self,Vec<u32>)> {
        if classes.symbol_count().checked_add(classes.len() as u32)? != alphabet {return None;}
        // PUSH is always a concrete stack symbol, never an opaque class.
        if program.templates.iter().flat_map(|graph| &graph.transitions).any(|edge|
            is_negative_label(edge.label)
                && negative_to_positive_label(edge.label) as u32 >= classes.symbol_count()) {return None;}
        let(mut graph,order)=Self::build(program,alphabet,pool,limits)?;
        graph.class_reads=(classes.symbol_count()..alphabet).map(|label|label as i32).collect();
        graph.pop_classes=Some(classes.clone());
        Some((graph,order))
    }

    pub fn build(program:&FiniteTemplateProgram<'a>,alphabet:u32,
        pool:&mut FastBoundaryWeightInterner,limits:FiniteCompileLimits,
    )->Option<(Self,Vec<u32>)> {
        let ports=program.port_finals.len();
        if ports==0 || ports>limits.states || alphabet==0 || alphabet as usize>limits.states
            || alphabet>=DEFAULT_LABEL as u32 || program.starts.iter().any(|&q|q as usize>=ports)
            || program.coefficients.len()>limits.weights {return None;}
        let templates = program.templates;
        // Validate even unused topology. Borrow immutable rows instead of
        // recoding each template for every lexical shard.
        let mut stored_rows = 0usize;
        let mut stored_entries = 0usize;
        for source in templates {
            let n = source.state_count();
            if n == 0 || n > limits.states || !source.valid_alphabet(alphabet) { return None; }
            stored_rows = stored_rows.checked_add(n)?;
            stored_entries = stored_entries.checked_add(source.logical_edges)?
                .checked_add(source.transitions.len())?;
            if stored_rows > limits.states || stored_entries > limits.edges { return None; }
        }
        let mut total_states=ports;let mut total_edges=0usize;
        for instance in program.instances {
            let template=templates.get(instance.template)?;
            if instance.coefficient>=program.coefficients.len() || instance.continuation as usize>=ports
                || instance.entries.start>instance.entries.end || instance.entries.end as usize>ports {return None;}
            total_states=total_states.checked_add(template.rows.len())?;
            total_edges=total_edges.checked_add(template.logical_edges)?.checked_add(
                (instance.entries.end-instance.entries.start)as usize*template.starts.len())?;
            if total_states>limits.states || total_edges>limits.edges || total_states>u32::MAX as usize{return None;}
        }
        let mut source_ids=FxHashMap::default();
        let coefficients=program.coefficients.iter().map(|w|pool.source_weight_id(w,&mut source_ids,None))
            .collect::<Option<Vec<_>>>()?;
        let mut port_rows=Vec::with_capacity(ports);
        for final_weight in program.port_finals {
            let final_weight=match final_weight{Some(id)=>*coefficients.get(*id)?,None=>0};
            port_rows.push(FastBoundaryNwaState{epsilons:Vec::new(),transitions:Vec::new(),final_weight});
        }
        let mut instances=Vec::with_capacity(program.instances.len());
        let mut node_instance=vec![u32::MAX;ports];
        node_instance.reserve(total_states-ports);
        for instance in program.instances {
            let template=&templates[instance.template];
            let base=node_instance.len()as u32;
            node_instance.resize(node_instance.len()+template.rows.len(),instances.len()as u32);
            instances.push(Instance{template:instance.template,base,coefficient:coefficients[instance.coefficient],
                continuation:instance.continuation});
            for port in instance.entries.clone() {
                port_rows[port as usize].epsilons.extend(template.starts.iter().map(|&q|(base+q,pool.all_id())));
            }
        }
        if !pool.allow_work(0,total_states,total_edges){return None;}
        let profile=VirtualGraphProfile{logical_states:total_states,logical_edges:total_edges,
            stored_template_states:templates.iter().map(|t|t.rows.len()).sum(),
            stored_template_edges:templates.iter().map(|t|t.logical_edges).sum(),
            instances:instances.len(),port_edges:port_rows.iter().map(|r|r.epsilons.len()).sum()};
        let graph=Self{alphabet,templates,instances,ports:port_rows,node_instance,profile,
            pop_classes:None,class_reads:Vec::new()};
        let order=topological_order(&graph)?;
        Some((graph,order))
    }

    /// Read the original signed graph and add cancellation epsilons as an
    /// overlay. Original negative edges remain visible through finality, then
    /// only the positive graph is materialized for the unchanged next stages.
    #[cfg(test)]
    pub fn resolve_positive(&self,pool:&mut FastBoundaryWeightInterner,initial_order:&[u32],
        reuse_topology:bool,filter:bool,
    )->Option<(Vec<FastBoundaryNwaState>,VirtualResolveProfile)> {
        let result=self.resolve_positive_mode(pool,initial_order,reuse_topology,filter,None,None)?;
        Some((result.states,result.profile))
    }

    pub fn resolve_positive_reachable(&self,pool:&mut FastBoundaryWeightInterner,initial_order:&[u32],
        reuse_topology:bool,filter:bool,starts:&[u32],
    )->Option<PositiveMaterialization> {
        self.resolve_positive_mode(pool,initial_order,reuse_topology,filter,Some(starts),None)
    }

    pub fn resolve_positive_with_context(&self,pool:&mut FastBoundaryWeightInterner,
        initial_order:&[u32],reuse_topology:bool,filter:bool,starts:&[u32],
        context:&FiniteParserReadSupport,
    )->Option<PositiveMaterialization> {
        if context.alphabet()!=self.alphabet as usize{return None;}
        self.resolve_positive_mode(pool,initial_order,reuse_topology,filter,Some(starts),Some(context))
    }

    fn resolve_positive_mode(&self,pool:&mut FastBoundaryWeightInterner,initial_order:&[u32],
        reuse_topology:bool,filter:bool,reachable_starts:Option<&[u32]>,
        read_context:Option<&FiniteParserReadSupport>,
    )->Option<PositiveMaterialization> {
        let phase=Instant::now();
        let(derived,stats)=super::finite_cancellation::compute_on_graph(self,pool,initial_order,filter)?;
        let mut profile=VirtualResolveProfile{cancellation_ms:elapsed_ms(phase),..Default::default()};
        if compile_profile_enabled(){eprintln!("[glrmask/profile][virtual_cancellation_summary] {stats:?}");}
        let phase=Instant::now();
        let extra=derived.into_iter().map(|row|row.into_entries().into_iter()
            .filter(|&(_,weight)|weight!=0).collect::<Vec<_>>()).collect::<Vec<_>>();
        let overlay=EpsilonOverlay{base:self,extra:&extra};
        if reuse_topology {
            let certificate=CheckedNativeTopology::from_order(initial_order.to_vec())?;
            if certificate.order.len()!=self.len(){return None;}
            for(q,edges)in extra.iter().enumerate(){
                if edges.iter().any(|&(target,_)|!certificate.permits(q,target)){return None;}
            }
        }
        let owned_order;
        let order=if reuse_topology {initial_order}else{
            owned_order=topological_order(&overlay)?;&owned_order
        };
        let n=self.len();
        let finals = final_weights(&overlay, pool, order)?;
        profile.finality_ms=elapsed_ms(phase);
        if let Some(starts) = reachable_starts {
            let finalized = FinalizedGraph { graph: &overlay, finals: &finals };
            // Positive pruning removes edges only; the signed/derived
            // topology remains valid for either reachability abstraction.
            return materialize_positive(&finalized, pool, order, starts, profile, read_context);
        }
        let phase=Instant::now();
        let mut states=Vec::with_capacity(n);
        for(q,final_weight)in finals.into_iter().enumerate(){
            let row=overlay.row(q);
            let epsilons=(0..row.epsilon_count()).map(|i|row.epsilon(i)).collect::<Vec<_>>();
            let mut transitions=Vec::new();
            for edge in 0..row.transition_count(){
                let label=row.label(edge);
                if is_negative_label(label){continue;}
                let branches=(0..row.branch_count(edge)).map(|branch|row.branch(edge,branch))
                    .collect::<SmallVec<[(u32,FastBoundaryWeightId);1]>>();
                transitions.push((label,branches));
            }
            profile.materialized_edges+=epsilons.len()+transitions.iter().map(|(_,b)|b.len()).sum::<usize>();
            states.push(FastBoundaryNwaState{epsilons,transitions,final_weight});
        }
        profile.materialize_ms=elapsed_ms(phase);
        let phase=Instant::now();
        // This helper is the old resolver's exact worklist, not a guessed
        // topological recurrence over possibly backward zero DEFAULT guards.
        fast_boundary_prune_terminal_defaults(&mut states,pool);
        profile.prune_ms=elapsed_ms(phase);
        profile.materialized_states=states.len();
        if !pool.allow_work(0,n,profile.materialized_edges){return None;}
        Some(PositiveMaterialization{states,starts:Vec::new(),topology:initial_order.to_vec(),profile})
    }
}

/// The optional parser read context refines ordinary structural
/// reachability. Both cases propagate only live positive edges and use
/// exactly the same guard-preserving publication.
enum PositiveSupport<'a> {
    Reachable(Vec<bool>),
    Context(super::finite_read_support::ReadContextTransfer<'a>),
}

impl<'a> PositiveSupport<'a> {
    fn new(nodes: usize, starts: &[u32], domain: Option<&'a FiniteParserReadSupport>) -> Option<Self> {
        if let Some(domain) = domain {
            return Some(Self::Context(
                super::finite_read_support::ReadContextTransfer::new(domain, nodes, starts)?,
            ));
        }
        let mut live = vec![false; nodes];
        for &start in starts {
            *live.get_mut(start as usize)? = true;
        }
        Some(Self::Reachable(live))
    }

    fn live(&self, state: usize) -> Option<bool> {
        match self {
            Self::Reachable(live) => live.get(state).copied(),
            Self::Context(transfer) => transfer.state_nonempty(state),
        }
    }

    fn load(&mut self, state: usize) -> Option<bool> {
        match self {
            Self::Reachable(live) => live.get(state).copied(),
            Self::Context(transfer) => transfer.load_source(state),
        }
    }

    fn allows(&self, label: i32) -> Option<bool> {
        match self {
            Self::Reachable(_) => Some(true),
            Self::Context(transfer) => transfer.read_allowed(label),
        }
    }

    fn epsilon(&mut self, target: u32) -> Option<()> {
        match self {
            Self::Reachable(live) => {
                *live.get_mut(target as usize)? = true;
                Some(())
            }
            Self::Context(transfer) => transfer.transfer_epsilon(target as usize),
        }
    }

    fn read(&mut self, label: i32, target: u32) -> Option<()> {
        match self {
            Self::Reachable(live) => {
                *live.get_mut(target as usize)? = true;
                Some(())
            }
            Self::Context(transfer) => transfer.transfer_read(label, target as usize),
        }
    }
}

/// Finalize the private positive view without allocating expanded rows for
/// unreachable logical nodes. DEFAULT classification is over the entire
/// finalized graph, before any context-sensitive filtering.
fn materialize_positive<G: SignedGraph + ?Sized>(
    graph: &G,
    pool: &mut FastBoundaryWeightInterner,
    order: &[u32],
    starts: &[u32],
    mut profile: VirtualResolveProfile,
    domain: Option<&FiniteParserReadSupport>,
) -> Option<PositiveMaterialization> {
    let n = graph.len();
    let phase = Instant::now();
    let terminal = terminal_default_states(graph, pool);
    profile.prune_ms = elapsed_ms(phase);

    let phase = Instant::now();
    let mut support = PositiveSupport::new(n, starts, domain)?;
    for &state in order {
        if !support.load(state as usize)? {
            continue;
        }
        let row = graph.row(state as usize);
        for edge in 0..row.epsilon_count() {
            let (target, weight) = row.epsilon(edge);
            if weight != 0 {
                support.epsilon(target)?;
            }
        }
        for edge in 0..row.transition_count() {
            let label = row.label(edge);
            if is_negative_label(label) || !support.allows(label)? {
                continue;
            }
            for branch in 0..row.branch_count(edge) {
                let (target, weight) = row.branch(edge, branch);
                if weight != 0
                    && keep_positive_branch(label, target, weight, row.final_weight(), &terminal, pool)
                {
                    support.read(label, target)?;
                }
            }
        }
    }
    if let PositiveSupport::Context(transfer) = &support {
        profile.context_input_states = n;
        profile.context_allocated_states = transfer.allocated_nodes();
        profile.read_context_ms = elapsed_ms(phase);
    }

    let mapping_started = Instant::now();
    let mut mapping = vec![u32::MAX; n];
    let mut count = 0u32;
    for (state, slot) in mapping.iter_mut().enumerate() {
        if support.live(state)? {
            *slot = count;
            count += 1;
        }
    }
    let mapped_target = |target: u32| {
        mapping.get(target as usize).copied().filter(|&target| target != u32::MAX)
    };
    let starts = starts.iter().map(|&state| mapped_target(state)).collect::<Option<Vec<_>>>()?;
    let topology = order.iter().filter_map(|&state| mapped_target(state)).collect();
    profile.reachability_ms = elapsed_ms(if domain.is_some() { mapping_started } else { phase });

    let phase = Instant::now();
    let mut states = Vec::with_capacity(count as usize);
    for (state, &mapped) in mapping.iter().enumerate() {
        if mapped == u32::MAX {
            continue;
        }
        if !support.load(state)? {
            return None;
        }
        let row = graph.row(state);
        let mut epsilons = Vec::new();
        for edge in 0..row.epsilon_count() {
            let (target, weight) = row.epsilon(edge);
            if weight != 0 {
                epsilons.push((mapped_target(target)?, weight));
            }
        }
        let mut transitions = Vec::new();
        for edge in 0..row.transition_count() {
            let label = row.label(edge);
            if is_negative_label(label) {
                continue;
            }
            let allowed = support.allows(label)?;
            let mut had_branches = false;
            let mut branches = SmallVec::<[(u32, FastBoundaryWeightId); 1]>::new();
            for branch in 0..row.branch_count(edge) {
                let (target, weight) = row.branch(edge, branch);
                if !keep_positive_branch(label, target, weight, row.final_weight(), &terminal, pool) {
                    continue;
                }
                had_branches = true;
                if allowed && weight != 0 {
                    branches.push((mapped_target(target)?, weight));
                }
            }
            // Pruned-away keys are absent; explicit blocked or zero-only
            // guards must remain, without making their old target live.
            if had_branches {
                if branches.is_empty() {
                    branches.push((0, 0));
                }
                transitions.push((label, branches));
            }
        }
        profile.materialized_edges += epsilons.len()
            + transitions.iter().map(|(_, branches)| branches.len()).sum::<usize>();
        states.push(FastBoundaryNwaState { epsilons, transitions, final_weight: row.final_weight() });
    }
    profile.materialized_states = states.len();
    profile.materialize_ms = elapsed_ms(phase);
    if !pool.allow_work(0, n, profile.materialized_edges) {
        return None;
    }
    Some(PositiveMaterialization { states, starts, topology, profile })
}

/// Original epsilon sequence followed by derived cancellation edges.
pub(super) struct EpsilonOverlay<'a,G:SignedGraph+?Sized> {
    pub base:&'a G,pub extra:&'a[Vec<(u32,FastBoundaryWeightId)>],
}
#[derive(Clone,Copy)]
pub(super) struct OverlayRow<'a,R:SignedRow> {row:R,extra:&'a[(u32,FastBoundaryWeightId)]}
impl<R:SignedRow> SignedRow for OverlayRow<'_,R> {
    #[inline] fn final_weight(&self)->u32 {self.row.final_weight()}
    #[inline] fn epsilon_count(&self)->usize {self.row.epsilon_count()+self.extra.len()}
    #[inline] fn epsilon(&self,index:usize)->(u32,u32) {
        if index<self.row.epsilon_count(){self.row.epsilon(index)}else{self.extra[index-self.row.epsilon_count()]}
    }
    #[inline] fn transition_count(&self)->usize {self.row.transition_count()}
    #[inline] fn label(&self,index:usize)->i32 {self.row.label(index)}
    #[inline] fn branch_count(&self,index:usize)->usize {self.row.branch_count(index)}
    #[inline] fn branch(&self,index:usize,branch:usize)->(u32,u32) {self.row.branch(index,branch)}
    #[inline] fn find_label(&self,label:i32)->Option<usize> {self.row.find_label(label)}
}
impl<G:SignedGraph+?Sized> SignedGraph for EpsilonOverlay<'_,G> {
    type Row<'a>=OverlayRow<'a,G::Row<'a>> where Self:'a;
    fn len(&self)->usize {self.base.len()}
    fn row(&self,index:usize)->Self::Row<'_> {OverlayRow{row:self.base.row(index),extra:&self.extra[index]}}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_class_cancellation_matches_the_independent_scoped_resolver() {
        use crate::compiler::glr::labels::encode_negative_label;
        use crate::pop_classes::PopLabelClasses;
        use crate::automata::weighted::nwa::{NWA,NWAState};
        let mut classes=PopLabelClasses::new(8).unwrap();
        let a=classes.intern_scoped_complement(2..5,[3]).unwrap().unwrap();
        let b=classes.intern_scoped_complement(4..7,[5]).unwrap().unwrap();
        for top in 0..8 {
            let mut source=NWA::from_parts(vec![NWAState::default();8],vec![0]);
            source.add_transition(0,encode_negative_label(top),1,Weight::all());
            source.add_transition(1,a,2,Weight::all());
            source.add_transition(1,b,3,Weight::all());
            source.add_transition(1,4,4,Weight::all()); // Overlap is additive.
            source.add_transition(1,3,5,Weight::all()); // Explicit dead shadow.
            source.add_transition(2,a,6,Weight::all());
            source.add_transition(3,b,6,Weight::all());
            source.add_transition(4,6,6,Weight::all());
            source.add_transition(6,encode_negative_label(7),7,Weight::all());
            source.set_final_weight(7,Weight::all());
            let mut expected=source.clone();
            crate::resolve_negatives::resolve_negative_codes_in_nwa_with_pop_classes(&mut expected,&classes).unwrap();
            let expected=classes.compile_positive(expected,10000).unwrap();
            let mut dense=source;
            for row in dense.states_mut() {
                row.transitions=std::mem::take(&mut row.transitions).into_iter().map(|(label,edges)| {
                    let label=if label==a {8} else if label==b {9} else {label};
                    (label,edges)
                }).collect();
            }
            let topology=TemplateTopology::from_weighted_skeleton(&dense);let refs=[&topology];let coefficients=[Weight::all()];let finals=[None,Some(0)];
            let instances=[FiniteTemplateInstance{template:0,coefficient:0,continuation:1,entries:0..1}];
            let program=FiniteTemplateProgram{templates:&refs,coefficients:&coefficients,port_finals:&finals,
                starts:&[0],instances:&instances};
            let mut pool=FastBoundaryWeightInterner::new(1,64).unwrap();
            let(graph,order)=VirtualSignedGraph::build_with_pop_classes(&program,10,&mut pool,Default::default(),&classes).unwrap();
            for filter in [false,true] {
                let actual=graph.resolve_positive_reachable(&mut pool,&order,false,filter,&[0]).unwrap();
                let states=actual.states.iter().map(|row| NWAState {
                    final_weight:(row.final_weight!=0).then(||pool.to_weight(row.final_weight)),
                    epsilons:row.epsilons.iter().map(|&(q,w)|(q,pool.to_weight(w))).collect(),
                    transitions:row.transitions.iter().map(|(label,edges)| {
                        let label=if *label==8 {a} else if *label==9 {b} else {*label};
                        (label,edges.iter().map(|&(q,w)|(q,pool.to_weight(w))).collect())
                    }).collect(),
                }).collect();
                let actual=classes.compile_positive(NWA::from_parts(states,actual.starts),10000).unwrap();
                let comparison=crate::parser_equivalence::compare_parser_mask_prefix_languages(
                    &expected,&actual,8,10000).unwrap();
                assert!(comparison.difference.is_none(),"PUSH {top}, filter {filter}: {:?}",comparison.difference);
            }
        }
    }

    fn assert_same_rows(left:&[FastBoundaryNwaState],right:&[FastBoundaryNwaState],case:usize){
        assert_eq!(left.len(),right.len(),"case={case}");
        for(q,(a,b))in left.iter().zip(right).enumerate(){
            assert_eq!(a.final_weight,b.final_weight,"final case={case} q={q}");
            assert_eq!(a.epsilons,b.epsilons,"eps case={case} q={q}");
            assert_eq!(a.transitions,b.transitions,"keys/branches case={case} q={q}");
        }
    }

    #[test]
    fn factored_program_matches_eager_rows_coefficients_and_kahn_order() {
        let mut seed=79331u64;
        let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32)as usize};
        for case in 0..512 {
            let mut templates=Vec::new();
            for _ in 0..3 {
                let n=3+next()%6;let mut t=NWA::new(0,0);
                for _ in 0..n{t.add_state();}
                t.set_start_states(if next()%2==0{vec![0,1]}else{vec![0]});
                for q in 0..n {
                    if next()%3==0 || q+1==n {t.set_final_weight(q as u32,Weight::empty());}
                    for target in q+1..n {
                        if next()%4==0{t.add_epsilon(q as u32,target as u32,Weight::empty());}
                        else {
                            let label=[0,1,DEFAULT_LABEL,crate::compiler::glr::labels::encode_negative_label(1)][next()%4];
                            t.add_transition(q as u32,label,target as u32,Weight::all());
                        }
                    }
                    if next()%3==0{t.states_mut()[q].transitions.entry(2).or_default();}
                }
                templates.push(t);
            }
            let topologies=templates.iter().map(TemplateTopology::from_weighted_skeleton).collect::<Vec<_>>();
            let refs=topologies.iter().collect::<Vec<_>>();
            let coefficients=[Weight::all(),Weight::from_token_set_for_tsid(0,[1,3,7].into_iter().collect()),Weight::empty()];
            let finals=[None,None,Some(1)];
            let instances=(0..3).map(|i|FiniteTemplateInstance{template:i,coefficient:(case+i)%3,
                continuation:if i==0{1}else{2},entries:if i==0{0..1}else{0..2}}).collect::<Vec<_>>();
            let program=FiniteTemplateProgram{templates:&refs,coefficients:&coefficients,
                port_finals:&finals,starts:&[0],instances:&instances};
            let mut left=FastBoundaryWeightInterner::new(1,64).unwrap();
            let mut right=FastBoundaryWeightInterner::new(1,64).unwrap();
            let(eager,edges,topo)=super::super::finite_template_program::build(&program,3,&mut left,Default::default()).unwrap();
            let(virtual_graph,vt)=VirtualSignedGraph::build(&program,3,&mut right,Default::default()).unwrap();
            assert_eq!(topo,vt,"topo case={case}");assert_eq!(edges,virtual_graph.profile.logical_edges);
            assert_eq!(left.values,right.values);assert_eq!(eager.len(),virtual_graph.len());
            for(q,a)in eager.iter().enumerate(){
                let b=virtual_graph.row(q);assert_eq!(a.final_weight,b.final_weight());
                assert_eq!(a.epsilons,(0..b.epsilon_count()).map(|i|b.epsilon(i)).collect::<Vec<_>>());
                assert_eq!(a.transitions.len(),b.transition_count());
                for (i,(label,branches))in a.transitions.iter().enumerate(){
                    assert_eq!(*label,b.label(i));assert_eq!(b.find_label(*label),Some(i));
                    assert_eq!(branches.as_slice(),(0..b.branch_count(i)).map(|j|b.branch(i,j)).collect::<Vec<_>>());
                }
            }
            let(a,sa)=super::super::finite_cancellation::compute_with_topology(&eager,&mut left,None).unwrap();
            let(b,sb)=super::super::finite_cancellation::compute_on_graph(&virtual_graph,&mut right,&vt,true).unwrap();
            assert_eq!(sa.queries,sb.queries,"memo queries case={case}");
            assert_eq!(sa.result_pairs,sb.result_pairs);
            assert_eq!(left.values,right.values,"exact interner sequence case={case}");
            for (q,(a,b))in a.into_iter().zip(b).enumerate(){
                let mut a=a.into_entries();let mut b=b.into_entries();a.sort_unstable();b.sort_unstable();
                assert_eq!(a,b,"derived epsilon case={case} state={q}");
            }
            // Rebuild pools from the same coefficient import to compare all
            // finality/interner operations, not just the cancellation relation.
            let mut left=FastBoundaryWeightInterner::new(1,64).unwrap();
            let mut right=FastBoundaryWeightInterner::new(1,64).unwrap();
            let(mut eager,_,_)=super::super::finite_template_program::build(&program,3,&mut left,Default::default()).unwrap();
            let(graph,order)=VirtualSignedGraph::build(&program,3,&mut right,Default::default()).unwrap();
            fast_boundary_resolve_negative_codes_with_topology(&mut eager,&mut left,None).unwrap();
            let(positive,_)=graph.resolve_positive(&mut right,&order,false,true).unwrap();
            assert_eq!(left.values,right.values,"post-finality interner sequence case={case}");
            assert_eq!(eager.len(),positive.len());
            for(q,(a,b))in eager.iter().zip(&positive).enumerate(){
                assert_eq!(a.epsilons,b.epsilons,"positive eps case={case} q={q}");
                assert_eq!(a.transitions,b.transitions,"positive row case={case} q={q}");
                assert_eq!(a.final_weight,b.final_weight,"positive final case={case} q={q}");
            }
            let mut compact_pool=FastBoundaryWeightInterner::new(1,64).unwrap();
            let(graph,order)=VirtualSignedGraph::build(&program,3,&mut compact_pool,Default::default()).unwrap();
            let compact=graph.resolve_positive_reachable(&mut compact_pool,&order,false,true,&[0]).unwrap();
            assert_eq!(left.values,compact_pool.values,"compact interner sequence case={case}");
            let(expected,starts)=super::super::finite_template_program::trim(eager.clone(),&[0]).unwrap();
            assert_eq!(starts,compact.starts);
            assert_same_rows(&expected,&compact.states,case);
            let certificate=CheckedNativeTopology::from_order(compact.topology).unwrap();
            assert!(certificate.certifies(&compact.states));
            // D1 moves only reachability before the existing context pass.
            // Compare complete output after both operation orders, including
            // DEFAULT forgetting and a domain that blocks some ordinary reads.
            let domain=FiniteParserReadSupport::new_checked(3,0,
                &[vec![(0,1),(1,1),(2,2)],vec![(0,1)],vec![(2,2)]],&[true,true,true],true).unwrap();
            let mut expected=eager;
            let mut actual=compact.states;
            super::super::finite_read_support::restrict(&mut expected,&[0],&domain).unwrap();
            super::super::finite_read_support::restrict_with_topology(&mut actual,&compact.starts,&domain,Some(&certificate)).unwrap();
            let(expected,expected_starts)=super::super::finite_template_program::trim(expected,&[0]).unwrap();
            let(actual,actual_starts)=super::super::finite_template_program::trim(actual,&compact.starts).unwrap();
            assert_eq!(expected_starts,actual_starts);
            assert_same_rows(&expected,&actual,case);
            for reuse in [false,true] {
                let mut reference_pool=FastBoundaryWeightInterner::new(1,64).unwrap();
                let(mut reference,_,reference_order)=super::super::finite_template_program::build(
                    &program,3,&mut reference_pool,Default::default()).unwrap();
                let reference_topology=reuse.then(||CheckedNativeTopology::from_order(reference_order).unwrap());
                fast_boundary_resolve_negative_codes_with_topology(
                    &mut reference,&mut reference_pool,reference_topology.as_ref()).unwrap();
                super::super::finite_read_support::restrict_with_topology(
                    &mut reference,&[0],&domain,reference_topology.as_ref()).unwrap();
                let(reference,reference_starts)=super::super::finite_template_program::trim(reference,&[0]).unwrap();
                let mut sparse_pool=FastBoundaryWeightInterner::new(1,64).unwrap();
                let(graph,order)=VirtualSignedGraph::build(&program,3,&mut sparse_pool,Default::default()).unwrap();
                let sparse=graph.resolve_positive_with_context(
                    &mut sparse_pool,&order,reuse,true,&[0],&domain).unwrap();
                assert_eq!(sparse.starts,reference_starts,"sparse starts case={case}");
                assert_same_rows(&reference,&sparse.states,case);
                assert_eq!(reference_pool.values,sparse_pool.values,"sparse interner case={case} reuse={reuse}");
                assert_eq!(sparse.profile.context_allocated_states,sparse.states.len());
                let certificate=CheckedNativeTopology::from_order(sparse.topology).unwrap();
                assert!(certificate.certifies(&sparse.states));
            }
        }
    }

    #[test]
    fn terminal_classification_matches_literal_least_fixed_point_with_zero_cycles(){
        let mut pool=FastBoundaryWeightInterner::new(1,8).unwrap();
        let weights=(0..16u64).map(|w|pool.intern(smallvec::smallvec![w])).collect::<Vec<_>>();
        let mut seed=89435u64;
        let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32)as usize};
        for case in 0..512 {
            let n=3+next()%19;
            let mut states=Vec::new();
            for _ in 0..n {
                let mut row=FastBoundaryNwaState{final_weight:weights[next()%16],epsilons:Vec::new(),transitions:Vec::new()};
                if next()%7==0{row.epsilons.push((next()as u32%n as u32,0));}
                for label in [0,1,DEFAULT_LABEL] {
                    if next()%2==0 {
                        let branches=(0..next()%3).map(|_|(next()as u32%n as u32,weights[next()%16]))
                            .collect::<SmallVec<[(u32,u32);1]>>();
                        row.transitions.push((label,branches));
                    }
                }
                states.push(row);
            }
            let mut expected=vec![false;n];
            loop {
                let mut changed=false;
                for(q,row)in states.iter().enumerate(){
                    if expected[q] || row.final_weight==0 || !row.epsilons.is_empty()
                        || row.transitions.iter().any(|(l,b)|*l!=DEFAULT_LABEL && !b.is_empty()){continue;}
                    let valid=row.transitions.iter().filter(|(l,_)|*l==DEFAULT_LABEL)
                        .flat_map(|(_,b)|b).all(|&(t,w)|expected[t as usize] && pool.is_subset(w,row.final_weight));
                    if valid{expected[q]=true;changed=true;}
                }
                if !changed{break;}
            }
            assert_eq!(terminal_default_states(states.as_slice(),&pool),expected,"case={case}");
        }
    }

    #[test]
    fn virtual_graph_rejects_cycles_bad_ports_and_budgets() {
        let mut t=NWA::new(0,0);t.add_state();t.set_start_states(vec![0]);t.set_final_weight(0,Weight::all());
        let topology=TemplateTopology::from_weighted_skeleton(&t);let coefficients=[Weight::all()];let refs=[&topology];let finals=[None];
        let instances=[FiniteTemplateInstance{template:0,coefficient:0,continuation:0,entries:0..1}];
        let p=FiniteTemplateProgram{templates:&refs,coefficients:&coefficients,port_finals:&finals,starts:&[0],instances:&instances};
        let mut pool=FastBoundaryWeightInterner::new(1,64).unwrap();
        assert!(VirtualSignedGraph::build(&p,4,&mut pool,Default::default()).is_none());
        assert!(VirtualSignedGraph::build(&p,4,&mut pool,FiniteCompileLimits{states:1,..Default::default()}).is_none());
        let mut bad_instances=instances.clone();
        bad_instances[0].continuation=8;
        let p=FiniteTemplateProgram{instances:&bad_instances,..p};
        assert!(VirtualSignedGraph::build(&p,4,&mut pool,Default::default()).is_none());
    }

    #[test]
    fn multiple_starts_zero_instances_and_reused_topology_match_eager() {
        let mut template=NWA::new(0,0);
        for _ in 0..4{template.add_state();}
        template.set_start_states(vec![0,1]);
        template.set_final_weight(3,Weight::empty());
        template.add_transition(0,crate::compiler::glr::labels::encode_negative_label(1),1,Weight::all());
        template.add_transition(1,1,2,Weight::all());
        template.add_transition(2,0,3,Weight::all());
        template.states_mut()[1].transitions.entry(2).or_default();
        let template=TemplateTopology::from_weighted_skeleton(&template);let refs=[&template];
        let coefficients=[Weight::empty(),Weight::from_token_set_for_tsid(0,[2,5].into_iter().collect())];
        let finals=[None,None,Some(1)];
        let instances=(0..3).map(|i|FiniteTemplateInstance{template:0,coefficient:i%2,
            continuation:2,entries:0..2}).collect::<Vec<_>>();
        let starts=[0,1];
        let program=FiniteTemplateProgram{templates:&refs,coefficients:&coefficients,
            port_finals:&finals,starts:&starts,instances:&instances};
        for reuse in [false,true] {
            let mut left=FastBoundaryWeightInterner::new(1,64).unwrap();
            let mut right=FastBoundaryWeightInterner::new(1,64).unwrap();
            let(mut expected,_,order)=super::super::finite_template_program::build(&program,3,&mut left,Default::default()).unwrap();
            let certificate=CheckedNativeTopology::from_order(order).unwrap();
            fast_boundary_resolve_negative_codes_with_topology(&mut expected,&mut left,reuse.then_some(&certificate)).unwrap();
            let(expected,expected_starts)=super::super::finite_template_program::trim(expected,&starts).unwrap();
            let(graph,order)=VirtualSignedGraph::build(&program,3,&mut right,Default::default()).unwrap();
            let actual=graph.resolve_positive_reachable(&mut right,&order,reuse,true,&starts).unwrap();
            assert_eq!(left.values,right.values);
            assert_eq!(expected_starts,actual.starts);
            assert_same_rows(&expected,&actual.states,usize::from(reuse));
        }
        // Unused malformed or excessive topology must never bypass validation.
        let unused=[&template,&template];
        let no_instances=FiniteTemplateProgram{templates:&unused,instances:&[],..program};
        let mut pool=FastBoundaryWeightInterner::new(1,64).unwrap();
        assert!(VirtualSignedGraph::build(&no_instances,3,&mut pool,
            FiniteCompileLimits{states:6,..Default::default()}).is_none());
    }
}
