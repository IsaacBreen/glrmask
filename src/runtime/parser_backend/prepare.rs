//! Synchronous native parser preparation.
//!
//! A Program is issued only after complete structural/alphabet validation.
//! Immutable source identity, domain, and fast view travel together until
//! terminal fanout. No executable LR data is retained.

use crate::ErrorExt as _;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Instant;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use glrmask_parser_dwa::__private::templates::native::{
    CompiledGroup, NativeCompileError, NativeTableIndex, ProgramCompiler,
};
use glrmask_parser_dwa::__private::templates::admissibility::{
    TemplateDomain, TopAdmission,
};

use crate::automata::unweighted_u32::dfa::DFA;
use crate::compiler::glr::table::GLRTable;
use crate::ds::bitset::BitSet;
use crate::grammar::flat::{DirectRegularAutomaton, TerminalID};
use crate::runtime::artifact::{
    FastCommitTemplateDfas,
};

#[cfg(test)]
use crate::runtime::artifact::FastTemplateDfasByTerminal;
use super::{CommitTemplateDfas, PreparedTemplateParser, TemplateParser};

#[derive(Debug)]
pub(crate) struct PreparedRelation {
    pub(crate) source: Arc<CommitTemplateDfas>,
    pub(crate) domain: Arc<TemplateDomain>,
    pub(crate) fast: Option<Arc<FastCommitTemplateDfas>>,
}

impl PreparedRelation {
    pub(crate) fn prepare(
        source: Arc<CommitTemplateDfas>,
        symbols: u32,
        runtime: bool,
    ) -> Result<Self, String> {
        let validated =
            crate::runtime::commit::template_prepare::TemplatePreparation::new(&source)?;
        validated.validate_alphabet(symbols)?;
        let domain = Arc::new(TemplateDomain::from_validated(&validated)?);
        let fast = runtime.then(|| Arc::new(
            FastCommitTemplateDfas::from_shared_preparation(
                Arc::clone(&source), &validated,
            )
        ));
        Ok(Self { source, domain, fast })
    }
}

struct DomainGroup {
    domain: Arc<TemplateDomain>,
    terminals: Vec<usize>,
}

struct Inventory {
    programs: Vec<Arc<PreparedRelation>>,
    groups: Vec<DomainGroup>,
}

fn parallel() -> bool {
    !crate::compiler::macro_parallelism_disabled() && rayon::current_num_threads() > 1
}

fn joined<A: Send, B: Send>(
    left: impl FnOnce() -> A + Send,
    right: impl FnOnce() -> B + Send,
) -> (A, B) {
    if parallel() {
        rayon::join(left, right)
    } else {
        (left(), right())
    }
}

fn inventory(
    mut groups: Vec<CompiledGroup<PreparedRelation>>,
    terminals: usize,
) -> crate::Result<Inventory> {
    let invalid = || crate::Error::Compilation(
        "terminal template inventory must cover the complete terminal domain".into()
    );
    if groups.iter().any(|group| group.terminals.is_empty()) {
        return Err(invalid());
    }
    groups.sort_unstable_by_key(|group| group.terminals[0]);
    let mut programs = vec![None; terminals];
    let mut domain_groups = Vec::with_capacity(groups.len());

    for group in groups {
        if group.terminals.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid());
        }
        let mut members = Vec::with_capacity(group.terminals.len());
        for terminal in group.terminals {
            let terminal = terminal as usize;
            let Some(slot) = programs.get_mut(terminal) else {
                return Err(invalid());
            };
            if slot.is_some() {
                return Err(invalid());
            }
            *slot = Some(Arc::clone(&group.program));
            members.push(terminal);
        }
        domain_groups.push(DomainGroup {
            domain: Arc::clone(&group.program.domain),
            terminals: members,
        });
    }
    if programs.iter().any(Option::is_none) {
        return Err(invalid());
    }
    let programs = programs.into_iter().map(Option::unwrap).collect();
    Ok(Inventory {
        programs,
        groups: domain_groups,
    })
}

fn shared_inventory(
    templates: &[Option<Arc<CommitTemplateDfas>>],
    symbols: u32,
    runtime: bool,
    use_parallel: bool,
) -> crate::Result<Inventory> {
    let mut identities = FxHashMap::<usize, usize>::default();
    let mut groups = Vec::<Vec<usize>>::new();
    for (terminal, source) in templates.iter().enumerate() {
        if let Some(source) = source {
            let identity = Arc::as_ptr(source) as usize;
            if let Some(&group) = identities.get(&identity) {
                groups[group].push(terminal);
            } else {
                identities.insert(identity, groups.len());
                groups.push(vec![terminal]);
            }
        } else {
            // Preserve error ordering relative to malformed earlier programs.
            groups.push(vec![terminal]);
        }
    }

    let build = |members: &Vec<usize>| -> crate::Result<CompiledGroup<PreparedRelation>> {
        let terminal = members[0];
        let source = templates[terminal].as_ref().ok_or_else(|| {
            crate::Error::Compilation(format!("missing template for terminal {terminal}"))
        })?;
        let program = PreparedRelation::prepare(Arc::clone(source), symbols, runtime)
            .map_err(crate::Error::Compilation)?;
        Ok(CompiledGroup {
            terminals: members.iter().map(|&terminal| terminal as u32).collect(),
            program: Arc::new(program),
        })
    };
    let results = if use_parallel && parallel() {
        groups.par_iter().map(build).collect::<Vec<_>>()
    } else {
        groups.iter().map(build).collect::<Vec<_>>()
    };
    let groups = results.into_iter().collect::<crate::Result<Vec<_>>>()?;
    inventory(groups, templates.len())
}

#[cfg(test)]
pub(super) fn prepare_terminal_inventory(
    templates: &[Option<Arc<CommitTemplateDfas>>],
    symbols: u32,
    runtime: bool,
    use_parallel: bool,
) -> crate::Result<(Vec<Arc<TemplateDomain>>, FastTemplateDfasByTerminal)> {
    let ready = shared_inventory(templates, symbols, runtime, use_parallel)?;
    Ok((ready.programs.iter().map(|p| Arc::clone(&p.domain)).collect(),
        if runtime { ready.programs.iter().map(|p| p.fast.clone()).collect() } else { Vec::new() }))
}

fn word_masks(terminals: &[usize]) -> Vec<(usize, u64)> {
    let mut masks = Vec::<(usize, u64)>::new();
    for &terminal in terminals {
        let word = terminal / 64;
        let bit = 1u64 << (terminal % 64);
        if let Some((last_word, mask)) = masks.last_mut()
            && *last_word == word
        {
            *mask |= bit;
        } else {
            masks.push((word, bit));
        }
    }
    masks
}

fn apply_certificate(
    possible: &mut BitSet,
    unconditional: &mut BitSet,
    masks: &[(usize, u64)],
    certificate: TopAdmission,
) {
    for &(word, mask) in masks {
        match certificate {
            TopAdmission::Never => {
                possible.words_mut()[word] &= !mask;
                unconditional.words_mut()[word] &= !mask;
            }
            TopAdmission::Always => {
                possible.words_mut()[word] |= mask;
                unconditional.words_mut()[word] |= mask;
            }
            TopAdmission::DependsOnSuffix => {
                possible.words_mut()[word] |= mask;
                unconditional.words_mut()[word] &= !mask;
            }
        }
    }
}

fn grouped_top_rows(
    symbols: u32,
    terminals: usize,
    groups: &[DomainGroup],
    completion: &Arc<TemplateDomain>,
) -> (Vec<BitSet>, Vec<BitSet>) {
    let mut default_possible = BitSet::new(terminals + 1);
    let mut default_unconditional = BitSet::new(terminals + 1);
    let masks = groups.iter().map(|group| word_masks(&group.terminals))
        .collect::<Vec<_>>();

    for (group, masks) in groups.iter().zip(&masks) {
        apply_certificate(
            &mut default_possible, &mut default_unconditional, masks,
            group.domain.top_admission_partition().0,
        );
    }
    let completion_mask = [(terminals / 64, 1u64 << (terminals % 64))];
    apply_certificate(
        &mut default_possible, &mut default_unconditional, &completion_mask,
        completion.top_admission_partition().0,
    );

    let mut possible = Vec::with_capacity(symbols as usize);
    let mut unconditional = Vec::with_capacity(symbols as usize);
    for _ in 0..symbols {
        possible.push(default_possible.clone());
        unconditional.push(default_unconditional.clone());
    }
    for (group, masks) in groups.iter().zip(&masks) {
        for (top, certificate) in group.domain.top_admission_partition().1 {
            let Some(row) = possible.get_mut(top as usize) else { continue; };
            apply_certificate(row, &mut unconditional[top as usize], masks, certificate);
        }
    }
    for (top, certificate) in completion.top_admission_partition().1 {
        let Some(row) = possible.get_mut(top as usize) else { continue; };
        apply_certificate(
            row, &mut unconditional[top as usize], &completion_mask, certificate,
        );
    }
    (possible, unconditional)
}

pub(super) fn top_certificate_rows(
    symbols: u32,
    domains: &[Arc<TemplateDomain>],
    completion: &Arc<TemplateDomain>,
) -> (Vec<BitSet>, Vec<BitSet>) {
    let mut identities = FxHashMap::<usize, usize>::default();
    let mut groups = Vec::<DomainGroup>::new();
    for (terminal, domain) in domains.iter().enumerate() {
        let identity = Arc::as_ptr(domain) as usize;
        if let Some(&group) = identities.get(&identity) {
            groups[group].terminals.push(terminal);
        } else {
            identities.insert(identity, groups.len());
            groups.push(DomainGroup {
                domain: Arc::clone(domain),
                terminals: vec![terminal],
            });
        }
    }
    grouped_top_rows(symbols, domains.len(), &groups, completion)
}

fn assemble(
    symbols: u32,
    terminals: u32,
    skips: BTreeSet<TerminalID>,
    inventory: Inventory,
    completion: PreparedRelation,
) -> TemplateParser {
    let (possible, unconditional) = grouped_top_rows(
        symbols, terminals as usize, &inventory.groups, &completion.domain,
    );
    TemplateParser {
        state_count: symbols,
        terminal_count: terminals,
        skip_terminals: skips,
        admission: super::TemplateAdmission::Local(super::LocalAdmission {
            programs: inventory.programs,
            completion: Arc::new(completion),
            possible,
            unconditional,
        }),
        embedding: None,
        link_grammar: None,
        profile: crate::env::var_os("GLRMASK_PROFILE_TEMPLATE_BACKEND").is_some(),
        advances: AtomicU64::new(0),
        admissions: AtomicU64::new(0),
        completions: AtomicU64::new(0),
    }
}

pub(super) fn compile_parser(
    symbols: u32,
    terminals: u32,
    skips: BTreeSet<TerminalID>,
    templates: &[Option<Arc<CommitTemplateDfas>>],
    completion: CommitTemplateDfas,
    runtime: bool,
) -> crate::Result<TemplateParser> {
    if templates.len() != terminals as usize {
        return Err(crate::Error::Compilation(
            "template parser must provide every terminal relation, including explicit empty relations"
                .into()
        ));
    }
    let states = templates.iter().filter_map(Option::as_deref).map(|template| {
        template.pop.states.len()
            .saturating_add(template.read.states.len())
            .saturating_add(template.push.states.len())
    }).fold(0usize, usize::saturating_add);
    let work = || {
        let inventory = shared_inventory(templates, symbols, runtime, states >= 4_096)?;
        let completion = PreparedRelation::prepare(Arc::new(completion), symbols, false)
            .map_err(crate::Error::Compilation)?;
        Ok(assemble(symbols, terminals, skips, inventory, completion))
    };
    if states >= 4_096 && !crate::compiler::macro_parallelism_disabled() {
        crate::compiler::pipeline::run_with_compile_thread_pool(work)
    } else {
        work()
    }
}

struct Metadata {
    embedding: Option<Arc<super::embedding::TemplateEmbedding>>,
    grammar: Option<Arc<super::link_grammar::LinkGrammar>>,
}

fn metadata(
    index: &NativeTableIndex<'_>,
    compiler: &ProgramCompiler,
    ignore: Option<u32>,
) -> crate::Result<Metadata> {
    let table = index.table();
    let mut grammar = super::link_grammar::LinkGrammar::from_compiler_table(table, ignore)
        .map_err(crate::Error::Compilation)?;
    let Ok(return_pop) = index.canonical_return_pop() else {
        return Ok(Metadata { embedding: None, grammar });
    };

    let (embedding, effects) = joined(
        || super::embedding::TemplateEmbedding::from_index(
            index, compiler, table.embedded_start_nullable(), return_pop,
        ),
        || {
            grammar.as_ref().map(|_| {
                crate::compiler::boundary_stack_support::CompilerEffects::from_index(index)
            })
        },
    );
    let embedding = embedding.ok().map(Arc::new);
    if let Some(grammar) = grammar.as_mut() {
        let effects = if embedding.is_some() {
            effects.and_then(Result::ok)
        } else {
            None
        };
        Arc::make_mut(grammar).set_stack_effects(effects);
    }
    Ok(Metadata { embedding, grammar })
}

pub(super) fn from_compiler_parts(
    table: &GLRTable,
    direct_regular: Option<&DirectRegularAutomaton>,
    retained_templates: &[Option<DFA>],
    ignore: Option<u32>,
    dynamic: bool,
    preserve_coordinate: bool,
) -> crate::Result<PreparedTemplateParser> {
    let profile = crate::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
    let started = profile.then(Instant::now);
    let sparse_regular =
        direct_regular.is_some() && table.num_rules == 0 && table.action.is_empty();

    if sparse_regular {
        let (templates, completion, symbols) =
            super::sparse_regular_templates(direct_regular.unwrap(), table.num_terminals)?;
        if preserve_coordinate && symbols != table.num_states {
            return Err(crate::Error::Compilation(
                "template conversion would change a composed parser's stack coordinate".into()
            ));
        }
        let mut parser = compile_parser(
            symbols, table.num_terminals, table.skip_terminals.clone(),
            &templates, completion, true,
        )?;
        parser.embedding = Some(Arc::new(
            super::embedding::TemplateEmbedding::from_sparse_regular(
                parser.completion_template(), parser.state_count,
                table.embedded_start_nullable(), 0..table.num_terminals,
            ).map_err(crate::Error::Compilation)?
        ));
        parser.link_grammar =
            super::link_grammar::LinkGrammar::from_compiler_table(table, ignore)
                .map_err(crate::Error::Compilation)?;
        if let Some(grammar) = parser.link_grammar.as_mut() {
            let effects =
                crate::compiler::boundary_stack_support::CompilerEffects::from_regular_programs(
                    &templates, parser.completion_template(), symbols,
                ).ok();
            Arc::make_mut(grammar).set_stack_effects(effects);
        }
        return Ok(PreparedTemplateParser {
            source_state_count: table.num_states,
            source_terminal_count: table.num_terminals,
            parser,
        });
    }

    let retained = if !dynamic
        && retained_templates.len() == table.num_terminals as usize
    {
        retained_templates
    } else {
        &[]
    };
    let selected = (0..table.num_terminals as usize).map(|terminal| {
        retained.get(terminal).is_none_or(Option::is_none)
    }).collect::<Vec<_>>();

    let index_started = profile.then(Instant::now);
    let index = NativeTableIndex::new(table, &selected)
        .map_err(crate::Error::Compilation)?;
    let index_ms = index_started.map_or(0.0, |time| time.elapsed().as_secs_f64() * 1000.0);
    let compiler = ProgramCompiler::new();

    let work = || {
        let (ordinary, (completion, metadata)) = joined(
            || {
                let (selected_result, retained_results) = joined(
                    || index.compile_selected(|characterization| {
                        let source = Arc::new(compiler.compile(characterization)?);
                        PreparedRelation::prepare(source, table.num_states, true)
                    }),
                    || {
                        let inputs = retained.iter().enumerate()
                            .filter_map(|(terminal, raw)| {
                                raw.as_ref().map(|raw| (terminal as u32, raw))
                            })
                            .collect::<Vec<_>>();
                        let build = |&(terminal, raw): &(u32, &DFA)| {
                            compiler.compile_raw(raw)
                                .and_then(|source| {
                                    PreparedRelation::prepare(Arc::new(source), table.num_states, true)
                                })
                                .map(|program| CompiledGroup {
                                    terminals: vec![terminal],
                                    program: Arc::new(program),
                                })
                                .map_err(|message| NativeCompileError { terminal, message })
                        };
                        if parallel() {
                            inputs.par_iter().map(build).collect::<Vec<_>>()
                        } else {
                            inputs.iter().map(build).collect::<Vec<_>>()
                        }
                    },
                );

                let mut failure = selected_result.as_ref().err().cloned();
                for result in &retained_results {
                    if let Err(error) = result
                        && failure.as_ref().is_none_or(|old| error.terminal < old.terminal)
                    {
                        failure = Some(error.clone());
                    }
                }
                if let Some(error) = failure {
                    return Err(crate::Error::classify_template_compile_message(
                        error.to_string(),
                    ));
                }
                let selected_result =
                    selected_result.unwrap_or_else(|_| unreachable!("checked failures"));
                let signature_groups = selected_result.action_signature_groups;
                let exact_groups = selected_result.exact_characterization_groups;
                let mut groups = selected_result.groups;
                for result in retained_results {
                    groups.push(result.unwrap_or_else(|_| unreachable!("checked failures")));
                }
                let ready = inventory(groups, table.num_terminals as usize)?;
                Ok((ready, signature_groups, exact_groups))
            },
            || joined(
                || {
                    let characterization = index.completion()?;
                    let source = compiler.compile(&characterization)?;
                    PreparedRelation::prepare(Arc::new(source), table.num_states, false)
                },
                || metadata(&index, &compiler, ignore),
            ),
        );

        // Explicit precedence: terminal errors, completion errors, metadata
        // errors. Within terminal errors choose the smallest terminal ID.
        let (ordinary, signature_groups, exact_groups) = ordinary?;
        let completion = completion.map_err(crate::Error::classify_template_compile_message)?;
        let metadata = metadata?;
        let mut parser = assemble(
            table.num_states, table.num_terminals, table.skip_terminals.clone(),
            ordinary, completion,
        );
        parser.embedding = metadata.embedding;
        parser.link_grammar = metadata.grammar;
        Ok::<_, crate::Error>((parser, signature_groups, exact_groups))
    };

    let use_pool = !crate::compiler::macro_parallelism_disabled()
        && (table.num_terminals > 16 || table.num_states > 256);
    let (parser, signature_groups, exact_groups) = if use_pool {
        crate::compiler::pipeline::run_with_compile_thread_pool(work)
    } else {
        work()
    }?;

    if let Some(started) = started {
        eprintln!(
            "[glrmask/profile][native_template_pipeline] terminals={} logical_actions={} unique_actions={} action_groups={} exact_missing_groups={} retained={} index_ms={index_ms:.3} total_ms={:.3}",
            table.num_terminals,
            index.logical_action_count(),
            index.unique_action_count(),
            signature_groups,
            exact_groups,
            retained.iter().filter(|raw| raw.is_some()).count(),
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(PreparedTemplateParser {
        source_state_count: table.num_states,
        source_terminal_count: table.num_terminals,
        parser,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::analysis::EOF;
    use crate::compiler::glr::table::{Action, testing::build_test_table};

    #[test]
    fn compiler_entry_shares_program_domain_and_fast_view_by_exact_group() {
        let table = build_test_table(
            2, 4,
            &[
                &[
                    (0, Action::Shift(1, false)),
                    (1, Action::Shift(1, false)),
                    (2, Action::Shift(1, false)),
                ],
                &[(2, Action::Accept), (EOF, Action::Accept)],
            ],
            &[&[], &[]],
        );
        let built = from_compiler_parts(&table, None, &[], None, true, false).unwrap();
        for terminal in [1usize, 2] {
            assert!(Arc::ptr_eq(
                built.parser.terminal_source(0).unwrap(),
                built.parser.terminal_source(terminal).unwrap(),
            ));
            assert!(Arc::ptr_eq(
                built.parser.local_domain(0).unwrap(), built.parser.local_domain(terminal).unwrap(),
            ));
            assert!(Arc::ptr_eq(
                built.parser.local_fast(0).unwrap(),
                built.parser.local_fast(terminal).unwrap(),
            ));
        }
        assert!(!Arc::ptr_eq(
            built.parser.terminal_source(0).unwrap(),
            built.parser.terminal_source(3).unwrap(),
        ));
        for terminal in 0..4 {
            let source = built.parser.terminal_source(terminal).unwrap();
            assert!(built.parser.local_fast(terminal).unwrap().is_for_source(source));
        }
    }

    #[test]
    fn actual_retained_raw_static_input_bypasses_missing_characterization() {
        use glrmask_parser_dwa::__private::templates::{
            characterize::try_characterize_selected_terminals_for_terminal_count,
            compile_dfa::Templates,
        };
        let table = build_test_table(
            2, 2,
            &[&[(0, Action::Shift(1, false)), (1, Action::Skip)],
              &[(EOF, Action::Accept)]],
            &[&[], &[]],
        );
        let chars = try_characterize_selected_terminals_for_terminal_count(
            &table, 2, &[true, true],
        ).unwrap();
        let mut raw = Templates::try_dfas_from_characterizations(&chars).unwrap();
        let retained = vec![raw.remove(&0), None];
        let rebuilt = from_compiler_parts(&table, None, &[], None, false, true).unwrap();
        let reused = from_compiler_parts(&table, None, &retained, None, false, true).unwrap();
        for terminal in 0..2 {
            assert_eq!(
                rebuilt.parser.local_domain(terminal).unwrap().to_bytes().unwrap(),
                reused.parser.local_domain(terminal).unwrap().to_bytes().unwrap(),
            );
        }
        assert_eq!(rebuilt.parser.local_admission().unwrap().possible, reused.parser.local_admission().unwrap().possible);
        assert_eq!(rebuilt.parser.local_admission().unwrap().unconditional, reused.parser.local_admission().unwrap().unconditional);
    }

    #[test]
    fn shared_preparation_validates_unreachable_nodes_before_alias_fanout() {
        let mut source = CommitTemplateDfas {
            pop: DFA::new(), read: DFA::new(), push: DFA::new(),
            ..CommitTemplateDfas::default()
        };
        let unreachable = source.pop.add_state();
        source.pop.add_transition(unreachable, 0, unreachable);
        let source = Arc::new(source);
        let inputs = vec![Some(Arc::clone(&source)); 32];
        for use_parallel in [false, true] {
            assert!(prepare_terminal_inventory(&inputs, 2, true, use_parallel).is_err());
        }
    }
}
