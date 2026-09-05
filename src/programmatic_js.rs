//! First-class programmatic JavaScript tool calling.
//!
//! [`ProgrammaticJsCompiler`] separates the reusable JavaScript work from the
//! per-tool-set work:
//!
//! 1. compile the full JavaScript parent and a conservative dynamic-value
//!    expression grammar once per vocabulary;
//! 2. compile each tool's JSON Schema with the shared dynamic-value grammar at
//!    nested property/array value positions;
//! 3. compose those compiled schemas into a tool dispatcher, then link that
//!    dispatcher into `tools.<name>(...)` call sites in the full JS parent.
//!
//! Literal values still take the ordinary schema branch. Dynamic expressions
//! are deliberately excluded at the schema root so a computed expression
//! cannot replace the entire arguments object and bypass its shape.

use std::collections::BTreeSet;
use std::sync::Arc;

use rayon::prelude::*;

use crate::compiler::constraint_compose::{
    CompiledSubgrammarInput, SegmentedBoundaryBackend, compose_constraints,
    compose_constraints_owned_parent_segmented_shared,
};
use crate::{Constraint, GlrMaskError, Vocab};

const JAVASCRIPT_GLRM: &str = include_str!("programmatic_js/javascript.glrm");
const PARENT_PLACEHOLDER_NAME: &str = "PROGRAMMATIC_TOOL_SUFFIX";
// The grammar/compiler stack can be substantially deeper than an ordinary
// Rayon worker stack. Keep macro-parallel reusable compilation on dedicated
// large-stack threads; each compile may still use Rayon internally.
const REUSABLE_COMPONENT_COMPILE_STACK_BYTES: usize = 64 * 1024 * 1024;

const DYNAMIC_VALUE_RULES: &str = r#"
// Opaque runtime values only. These forms retrieve/produce a runtime value
// without embedding a statically visible literal as the value itself. In
// particular there are deliberately no arithmetic/logical/conditional tails:
// schema-aware constructions such as `cond ? "open" : "closed"` are lowered
// by the JSON-Schema layer, where each result arm can be checked.
nt dynamic_value_expression ::=
    dynamic_reference_expression
  | 'await' dynamic_reference_expression
  ;

nt dynamic_reference_expression ::=
    IDENTIFIER dynamic_reference_suffix*
  | 'this' dynamic_reference_suffix*
  | 'new' '.' 'target' dynamic_reference_suffix*
  | 'import' '(' assignment_expression ')' dynamic_reference_suffix*
  | 'import' '.' 'meta' dynamic_reference_suffix*
  ;

nt dynamic_reference_suffix ::=
    '[' expression ']'
  | '?.' '[' expression ']'
  | '.' IDENTIFIER
  | '?.' IDENTIFIER
  | '.' private_identifier
  | '?.' private_identifier
  | arguments
  | '?.' arguments
  | TEMPLATE_LITERAL
  ;
"#;



/// Reusable compiler for programmatic JavaScript tool calling.
#[derive(Debug)]
pub struct ProgrammaticJsCompiler {
    parent: Arc<Constraint>,
    wrapper_parent: Arc<Constraint>,
    dynamic_value: Constraint,
    condition: Constraint,
    parent_placeholder_terminal: u32,
}

impl ProgrammaticJsCompiler {
    /// Compile all reusable programmatic-JavaScript components for `vocab`.
    pub fn new(vocab: &Vocab) -> crate::Result<Self> {
        let (parent, wrapper_parent, dynamic_value, condition) = std::thread::scope(|scope| {
            let spawn = |name: &str,
                         compile: fn(&Vocab) -> crate::Result<Constraint>|
             -> crate::Result<_> {
                std::thread::Builder::new()
                    .name(name.to_owned())
                    .stack_size(REUSABLE_COMPONENT_COMPILE_STACK_BYTES)
                    .spawn_scoped(scope, move || compile(vocab))
                    .map_err(|err| {
                        GlrMaskError::Compilation(format!(
                            "failed to spawn programmatic JavaScript compiler thread {name}: {err}"
                        ))
                    })
            };

            let parent = spawn("glrmask-js-parent", Self::compile_parent)?;
            let wrapper_parent = spawn("glrmask-js-wrapper-parent", Self::compile_wrapper_parent)?;
            let dynamic_value = spawn("glrmask-js-dynamic-value", Self::compile_dynamic_value)?;
            let condition = spawn("glrmask-js-condition", Self::compile_condition)?;

            let join = |thread: std::thread::ScopedJoinHandle<'_, crate::Result<Constraint>>| {
                match thread.join() {
                    Ok(result) => result,
                    Err(panic) => std::panic::resume_unwind(panic),
                }
            };
            Ok::<_, GlrMaskError>((
                join(parent)?,
                join(wrapper_parent)?,
                join(dynamic_value)?,
                join(condition)?,
            ))
        })?;
        Self::from_components_with_wrapper(parent, wrapper_parent, dynamic_value, condition)
    }

    /// Compile the reusable full-JavaScript parent containing the reserved
    /// `tools` dispatcher boundary. This is independent of any concrete tool
    /// schemas and may be built once per vocabulary.
    pub fn compile_parent(vocab: &Vocab) -> crate::Result<Constraint> {
        let placeholder_token_id =
            crate::import::external_placeholder_token_id_avoiding(vocab, std::iter::empty())?;
        let mut constraint =
            Constraint::from_glrm_grammar(&programmatic_parent_source(placeholder_token_id)?, vocab)?;
        constraint
            .build_boundary_trigger(crate::BoundaryTriggerDetail::Tokens)
            .map_err(GlrMaskError::Compilation)?;
        Ok(constraint)
    }

    /// Compile the reusable opaque-runtime-value expression subgrammar.
    pub fn compile_dynamic_value(vocab: &Vocab) -> crate::Result<Constraint> {
        let mut constraint = Constraint::from_glrm_grammar(&dynamic_value_source()?, vocab)?;
        constraint
            .build_boundary_trigger(crate::BoundaryTriggerDetail::Tokens)
            .map_err(GlrMaskError::Compilation)?;
        Ok(constraint)
    }

    /// Compile the reusable unrestricted JavaScript condition subgrammar used
    /// only as the test of schema-aware conditional expressions.
    pub fn compile_condition(vocab: &Vocab) -> crate::Result<Constraint> {
        let mut constraint = Constraint::from_glrm_grammar(&condition_source()?, vocab)?;
        constraint
            .build_boundary_trigger(crate::BoundaryTriggerDetail::Tokens)
            .map_err(GlrMaskError::Compilation)?;
        Ok(constraint)
    }

    fn compile_wrapper_parent(vocab: &Vocab) -> crate::Result<Constraint> {
        let wrapper_source = prepared_tool_wrapper_source();
        let mut constraint =
            Constraint::from_glrm_grammar_with_subgrammars(&wrapper_source, &[], vocab)?;
        constraint
            .build_boundary_trigger(crate::BoundaryTriggerDetail::Tokens)
            .map_err(GlrMaskError::Compilation)?;
        constraint
            .materialize_composition_metadata_for_compilation()
            .map_err(GlrMaskError::Compilation)?;
        Ok(constraint)
    }

    /// Assemble a reusable compiler from independently compiled shared parts.
    /// This exists so build systems and benchmarks can time/cache each shared
    /// component separately without changing programmatic-tool semantics.
    pub fn from_components(
        parent: Constraint,
        dynamic_value: Constraint,
        condition: Constraint,
    ) -> crate::Result<Self> {
        let vocab = crate::public_api::constraint_vocab(&parent);
        let wrapper_parent = Self::compile_wrapper_parent(&vocab)?;
        Self::from_components_with_wrapper(parent, wrapper_parent, dynamic_value, condition)
    }

    fn from_components_with_wrapper(
        mut parent: Constraint,
        mut wrapper_parent: Constraint,
        mut dynamic_value: Constraint,
        mut condition: Constraint,
    ) -> crate::Result<Self> {
        // These four constraints are the reusable intact leaves subsequently
        // embedded in every programmatic-schema composition. Give each leaf its
        // own conservative trigger before any Arc sharing/composition occurs;
        // doing this here also upgrades older cached components loaded by callers
        // without requiring Arc::make_mut or composition-specific trigger logic.
        for component in [
            &mut parent,
            &mut wrapper_parent,
            &mut dynamic_value,
            &mut condition,
        ] {
            if component.boundary_trigger.is_none() {
                component
                    .build_boundary_trigger(crate::BoundaryTriggerDetail::Tokens)
                    .map_err(GlrMaskError::Compilation)?;
            }
        }
        parent
            .materialize_composition_metadata_for_compilation()
            .map_err(GlrMaskError::Compilation)?;
        wrapper_parent
            .materialize_composition_metadata_for_compilation()
            .map_err(GlrMaskError::Compilation)?;
        let parent_placeholder_terminal = parent
            .terminal_display_names
            .iter()
            .position(|name| name == PARENT_PLACEHOLDER_NAME)
            .map(|index| index as u32)
            .ok_or_else(|| {
                GlrMaskError::Compilation(
                    "programmatic JavaScript parent has no dispatcher linker terminal".into(),
                )
            })?;
        Ok(Self {
            parent: Arc::new(parent),
            wrapper_parent: Arc::new(wrapper_parent),
            dynamic_value,
            condition,
            parent_placeholder_terminal,
        })
    }

    /// The reusable dynamic-value child used by every schema compiled through
    /// this compiler.
    pub fn dynamic_value_constraint(&self) -> &Constraint {
        &self.dynamic_value
    }

    /// Compile one tool arguments schema. The schema root stays static; nested
    /// object-property and array-item values may be dynamic JS expressions.
    pub fn compile_schema(&self, schema: &str, vocab: &Vocab) -> crate::Result<Constraint> {
        Constraint::from_json_schema_with_programmatic_values(
            schema,
            &self.dynamic_value,
            &self.condition,
            vocab,
        )
    }

    /// Compile a named tool dispatcher from already-compiled tool schemas.
    /// This is separate from the outer JavaScript link so callers can time or
    /// configure the two composition stages independently.
    pub fn compile_dispatcher(
        &self,
        tools: &[(&str, &Constraint)],
        vocab: &Vocab,
    ) -> crate::Result<Constraint> {
        validate_tool_names(tools.iter().map(|(name, _)| *name))?;
        if tools.is_empty() {
            return Err(GlrMaskError::Compilation(
                "programmatic JavaScript requires at least one tool".into(),
            ));
        }

        let dispatcher_source = dispatcher_source(tools.iter().map(|(name, _)| *name));
        let bindings = tools
            .iter()
            .enumerate()
            .map(|(index, (_, schema))| (format!("args_{index}"), *schema))
            .collect::<Vec<_>>();
        let borrowed = bindings
            .iter()
            .map(|(name, schema)| (name.as_str(), *schema))
            .collect::<Vec<_>>();
        Constraint::from_glrm_grammar_with_subgrammars(&dispatcher_source, &borrowed, vocab)
    }

    /// Compose an already-compiled static tool set behind the exact dynamic
    /// boundary walker while retaining every schema as a static component.
    /// This consuming form is used by build systems that already own/cache the
    /// component artifacts and do not want to clone them merely to publish the
    /// recursive runtime tree.
    #[doc(hidden)]
    pub fn compile_dispatcher_dynamic_boundary_owned(
        &self,
        tools: Vec<(String, Constraint)>,
        vocab: &Vocab,
    ) -> crate::Result<Constraint> {
        validate_tool_names(tools.iter().map(|(name, _)| name.as_str()))?;
        if tools.is_empty() {
            return Err(GlrMaskError::Compilation(
                "programmatic JavaScript requires at least one tool".into(),
            ));
        }

        let dispatcher_source = dispatcher_source(tools.iter().map(|(name, _)| name.as_str()));
        let parent = Constraint::from_glrm_grammar_with_subgrammars(&dispatcher_source, &[], vocab)?;

        let mut shared_children = Vec::<Arc<Constraint>>::with_capacity(tools.len());
        let mut binding_names = Vec::<String>::with_capacity(tools.len());
        for (index, (tool_name, mut child)) in tools.into_iter().enumerate() {
            child
                .materialize_composition_link_metadata_for_compilation()
                .map_err(GlrMaskError::Compilation)?;
            binding_names.push(format!("args_{index}"));
            let _ = tool_name;
            shared_children.push(Arc::new(child));
        }

        let mut terminals = Vec::<Vec<u32>>::with_capacity(binding_names.len());
        for binding_name in &binding_names {
            let matching = parent
                .late_grammar_slots
                .iter()
                .filter(|slot| slot.name == *binding_name)
                .map(|slot| slot.terminal_id)
                .collect::<Vec<_>>();
            if matching.is_empty() {
                return Err(GlrMaskError::Compilation(format!(
                    "programmatic dispatcher lost external grammar {binding_name:?}"
                )));
            }
            terminals.push(matching);
        }

        let inputs = shared_children
            .iter()
            .zip(&terminals)
            .map(|(child, terminals)| CompiledSubgrammarInput {
                placeholder_terminal: terminals[0],
                additional_placeholder_terminals: &terminals[1..],
                constraint: child.as_ref(),
            })
            .collect::<Vec<_>>();

        let mut composition = compose_constraints_owned_parent_segmented_shared(
            parent,
            &inputs,
            &shared_children,
            vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .map_err(GlrMaskError::Compilation)?;
        composition.constraint.late_grammar_slots.clear();
        Ok(composition.constraint)
    }

    /// Link a compiled tool dispatcher into the reusable full-JavaScript parent.
    pub fn compose_dispatcher(
        &self,
        dispatcher: &Constraint,
        vocab: &Vocab,
    ) -> crate::Result<Constraint> {
        compose_constraints(
            &self.parent,
            &[CompiledSubgrammarInput {
                placeholder_terminal: self.parent_placeholder_terminal,
                additional_placeholder_terminals: &[],
                constraint: dispatcher,
            }],
            vocab,
        )
        .map(|composition| composition.constraint)
        .map_err(GlrMaskError::Compilation)
    }

    /// Consume this reusable compiler and a compiled dispatcher, linking the
    /// dispatcher into the static JavaScript parent with a dynamic boundary.
    /// The parent and dispatcher remain static component runtimes; only the
    /// cross-component B evaluator is dynamic.
    #[doc(hidden)]
    pub fn compose_dispatcher_dynamic_boundary_owned(
        self,
        mut dispatcher: Constraint,
        vocab: &Vocab,
    ) -> crate::Result<Constraint> {
        dispatcher
            .materialize_composition_link_metadata_for_compilation()
            .map_err(GlrMaskError::Compilation)?;
        let shared = vec![Arc::new(dispatcher)];
        let inputs = [CompiledSubgrammarInput {
            placeholder_terminal: self.parent_placeholder_terminal,
            additional_placeholder_terminals: &[],
            constraint: shared[0].as_ref(),
        }];
        let parent = Arc::try_unwrap(self.parent).unwrap_or_else(|parent| (*parent).clone());
        compose_constraints_owned_parent_segmented_shared(
            parent,
            &inputs,
            &shared,
            vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .map(|composition| composition.constraint)
        .map_err(GlrMaskError::Compilation)
    }

    /// Compose already-compiled tool schemas into a named dispatcher and link
    /// it into the reusable full-JavaScript parent.
    pub fn compose_tools(
        &self,
        tools: &[(&str, &Constraint)],
        vocab: &Vocab,
    ) -> crate::Result<Constraint> {
        let dispatcher = self.compile_dispatcher(tools, vocab)?;
        self.compose_dispatcher(&dispatcher, vocab)
    }

    /// Compose an owned static tool set through the two-phase static boundary
    /// pipeline. The dispatcher semantic core may feed the outer JavaScript
    /// linker while its boundary publication is still finishing.
    #[doc(hidden)]
    pub fn compose_tools_static_prepared_owned(
        &self,
        tools: Vec<(String, Constraint)>,
        vocab: &Vocab,
    ) -> crate::Result<Constraint> {
        let total_started = std::time::Instant::now();
        let profile = std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
        validate_tool_names(tools.iter().map(|(name, _)| name.as_str()))?;
        if tools.is_empty() {
            return Err(GlrMaskError::Compilation(
                "programmatic JavaScript requires at least one tool".into(),
            ));
        }

        // Keep each expensive schema boundary local to a wrapper. The dispatcher owns
        // `.tool_name`, while each wrapper owns `(` + schema + `)`. This preserves
        // JavaScript trivia before the dot while keeping schema boundary work local.
        let tool_names = tools.iter().map(|(name, _)| name.clone()).collect::<Vec<_>>();
        let dispatcher_source = prepared_dispatcher_source(&tool_names);
        let wrapper_parent = Arc::clone(&self.wrapper_parent);
        // These are macro-level coordinators. Each prepared binding parks while
        // its composition worker publishes the semantic core, and that worker
        // uses Rayon for micro-parallel compiler phases. Running the parked
        // coordinators themselves on Rayon workers starves the inner pool (and
        // can deadlock when the outer fanout reaches the pool size), so keep the
        // two scheduling layers separate. The dispatcher shell is independent of
        // every schema wrapper, so compile it in the same scoped macro phase rather
        // than serializing ~40 ms of tool-specific grammar work ahead of wrappers.
        let parallel_prepare_started = std::time::Instant::now();
        let (dispatcher_parent, dispatcher_parent_ms, wrappers, wrappers_ms) =
            std::thread::scope(|scope| {
                let dispatcher_handle = std::thread::Builder::new()
                    .name("glrmask-js-dispatcher-parent".to_owned())
                    .stack_size(REUSABLE_COMPONENT_COMPILE_STACK_BYTES)
                    .spawn_scoped(scope, move || {
                        let started = std::time::Instant::now();
                        let mut dispatcher_parent =
                            Constraint::from_glrm_grammar_with_subgrammars(
                                &dispatcher_source,
                                &[],
                                vocab,
                            )?;
                        dispatcher_parent
                            .build_boundary_trigger(crate::BoundaryTriggerDetail::Tokens)
                            .map_err(GlrMaskError::Compilation)?;
                        Ok::<_, GlrMaskError>((
                            dispatcher_parent,
                            started.elapsed().as_secs_f64() * 1000.0,
                        ))
                    })
                    .map_err(|error| {
                        GlrMaskError::Compilation(format!(
                            "failed to spawn programmatic dispatcher compiler: {error}"
                        ))
                    })?;

                let wrappers_started = std::time::Instant::now();
            let mut handles = Vec::with_capacity(tools.len());
            for (index, (_name, constraint)) in tools.into_iter().enumerate() {
                let wrapper_parent = Arc::clone(&wrapper_parent);
                let handle = std::thread::Builder::new()
                    .name(format!("glrmask-js-wrapper-{index}"))
                    .spawn_scoped(scope, move || {
                        let wrapper =
                            crate::public_api::bind_static_shared_parent_shared_prepare_before_boundary(
                                wrapper_parent,
                                vec![("args".to_owned(), Arc::new(constraint))],
                                SegmentedBoundaryBackend::StaticParserDwa,
                            )?;
                        Ok::<_, GlrMaskError>((format!("call_{index}"), wrapper))
                    })
                    .map_err(|error| {
                        GlrMaskError::Compilation(format!(
                            "failed to spawn programmatic tool wrapper compiler: {error}"
                        ))
                    })?;
                handles.push(handle);
            }
                let wrappers = handles
                .into_iter()
                .map(|handle| match handle.join() {
                    Ok(result) => result,
                    Err(panic) => std::panic::resume_unwind(panic),
                })
                    .collect::<crate::Result<Vec<_>>>()?;
                let wrappers_ms = wrappers_started.elapsed().as_secs_f64() * 1000.0;
                let (dispatcher_parent, dispatcher_parent_ms) = match dispatcher_handle.join() {
                    Ok(result) => result?,
                    Err(panic) => std::panic::resume_unwind(panic),
                };
                Ok::<_, GlrMaskError>((
                    dispatcher_parent,
                    dispatcher_parent_ms,
                    wrappers,
                    wrappers_ms,
                ))
            })?;
        let parallel_prepare_ms = parallel_prepare_started.elapsed().as_secs_f64() * 1000.0;
        let dispatcher_started = std::time::Instant::now();
        let dispatcher = crate::public_api::bind_static_parent_prepared_children(
            dispatcher_parent,
            wrappers,
            SegmentedBoundaryBackend::StaticParserDwa,
        )?;
        let dispatcher_ms = dispatcher_started.elapsed().as_secs_f64() * 1000.0;
        let root_started = std::time::Instant::now();
        let root = crate::public_api::compose_static_shared_parent_prepared_child_at_terminal(
            Arc::clone(&self.parent),
            self.parent_placeholder_terminal,
            dispatcher,
            SegmentedBoundaryBackend::StaticParserDwa,
        )?;
        let root_ms = root_started.elapsed().as_secs_f64() * 1000.0;
        let finish_started = std::time::Instant::now();
        let result = root.finish();
        let finish_ms = finish_started.elapsed().as_secs_f64() * 1000.0;
        if profile {
            eprintln!(
                "[glrmask/profile][programmatic_prepared_static] parallel_prepare_wall_ms={parallel_prepare_ms:.3} wrappers_ms={wrappers_ms:.3} dispatcher_parent_ms={dispatcher_parent_ms:.3} dispatcher_publish_ms={dispatcher_ms:.3} root_publish_ms={root_ms:.3} finish_ms={finish_ms:.3} total_ms={:.3}",
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    }
    /// Convenience path that compiles every schema and then composes the full
    /// tool-calling constraint. Use [`Self::compile_schema`] and
    /// [`Self::compose_tools`] separately when build-phase timings matter.
    pub fn compile_tools(
        &self,
        tools: &[(&str, &str)],
        vocab: &Vocab,
    ) -> crate::Result<Constraint> {
        validate_tool_names(tools.iter().map(|(name, _)| *name))?;
        let compiled = tools
            .par_iter()
            .map(|(name, schema)| {
                self.compile_schema(schema, vocab)
                    .map(|constraint| ((*name).to_owned(), constraint))
            })
            .collect::<crate::Result<Vec<_>>>()?;
        self.compose_tools_static_prepared_owned(compiled, vocab)
    }
}

/// Canonical full JavaScript grammar bundled with GLRMask's programmatic tool
/// calling support. It omits CFA's textual EOF sentinel; model end-token IDs
/// remain a separate runtime concern.
pub(crate) fn javascript_glrm() -> &'static str {
    JAVASCRIPT_GLRM
}

fn reserve_tools_identifier(grammar: &mut crate::grammar::ast::NamedGrammar) -> crate::Result<()> {
    use crate::grammar::ast::GrammarExpr;

    let identifier = grammar
        .rules
        .iter_mut()
        .find(|rule| rule.name == "IDENTIFIER")
        .ok_or_else(|| {
            GlrMaskError::Compilation("bundled JavaScript grammar has no IDENTIFIER terminal".into())
        })?;
    identifier.expr = GrammarExpr::Exclude {
        expr: Box::new(identifier.expr.clone()),
        exclude: Box::new(GrammarExpr::Literal(b"tools".to_vec())),
    };
    Ok(())
}

fn pruned_javascript_source(start_rule: &str, extra_rules: &str) -> crate::Result<String> {
    let mut source = JAVASCRIPT_GLRM.to_owned();
    let start = "start program;";
    if !source.starts_with(start) {
        return Err(GlrMaskError::Compilation(
            "bundled JavaScript grammar start rule changed unexpectedly".into(),
        ));
    }
    source.replace_range(..start.len(), &format!("start {start_rule};"));
    source.push_str(extra_rules);

    let mut grammar = crate::grammar::glrm::from_glrm(&source)?;
    reserve_tools_identifier(&mut grammar)?;
    // The generic reachability pass follows grammar references from `start`,
    // while `ignore IGNORE` is metadata. Temporarily root IGNORE through a
    // synthetic rule so its lexical dependency closure survives pruning.
    let original_start = grammar.start.clone();
    let synthetic = "__ptc_prune_root".to_string();
    let mut roots = vec![crate::grammar::ast::GrammarExpr::Ref(original_start.clone())];
    if let Some(ignore) = grammar.ignore.clone() {
        roots.push(crate::grammar::ast::GrammarExpr::Ref(ignore));
    }
    grammar.rules.push(crate::grammar::ast::NamedRule {
        name: synthetic.clone(),
        expr: crate::grammar::ast::GrammarExpr::Choice(roots),
        is_terminal: false,
        is_internal: true,
    });
    grammar.start = synthetic.clone();
    crate::grammar::right_linear::retain_reachable_rules(&mut grammar);
    grammar.rules.retain(|rule| rule.name != synthetic);
    grammar.start = original_start;
    Ok(crate::grammar::glrm::to_glrm(&grammar))
}

fn dynamic_value_source() -> crate::Result<String> {
    pruned_javascript_source("dynamic_value_expression", DYNAMIC_VALUE_RULES)
}

fn condition_source() -> crate::Result<String> {
    pruned_javascript_source("coalesce_expression", "")
}

fn programmatic_parent_source(placeholder_token_id: u32) -> crate::Result<String> {
    use crate::grammar::ast::{GrammarExpr, NamedRule};

    let mut grammar = crate::grammar::glrm::from_glrm(JAVASCRIPT_GLRM)?;
    // `tools` is a reserved namespace in programmatic-tool mode. Without this
    // subtraction, `tools.foo(...)` can take the ordinary IDENTIFIER/call path
    // and bypass the schema dispatcher entirely.
    reserve_tools_identifier(&mut grammar)?;

    let member = grammar
        .rules
        .iter_mut()
        .find(|rule| rule.name == "member_expression_with_suffixes")
        .ok_or_else(|| GlrMaskError::Compilation(
            "bundled JavaScript grammar has no member_expression_with_suffixes rule".into(),
        ))?;
    let tool_expr = GrammarExpr::Sequence(vec![
        GrammarExpr::Literal(b"tools".to_vec()),
        GrammarExpr::Ref(PARENT_PLACEHOLDER_NAME.to_string()),
    ]);
    member.expr = match std::mem::replace(&mut member.expr, GrammarExpr::Epsilon) {
        GrammarExpr::Choice(mut alternatives) => {
            alternatives.insert(0, tool_expr);
            GrammarExpr::Choice(alternatives)
        }
        other => GrammarExpr::Choice(vec![tool_expr, other]),
    };
    grammar.rules.push(NamedRule {
        name: PARENT_PLACEHOLDER_NAME.to_string(),
        expr: GrammarExpr::SpecialToken(placeholder_token_id),
        is_terminal: true,
        is_internal: false,
    });
    Ok(crate::grammar::glrm::to_glrm(&grammar))
}

fn prepared_tool_wrapper_source() -> String {
    "extern grammar args;\nstart wrapped;\nnt wrapped ::= '(' args ')';\n".to_owned()
}

fn prepared_dispatcher_source(names: &[String]) -> String {
    let mut source = String::new();
    for index in 0..names.len() {
        source.push_str(&format!("extern grammar call_{index};\n"));
    }
    source.push_str("start suffix;\nnt suffix ::=\n");
    for (index, name) in names.iter().enumerate() {
        let prefix = if index == 0 { "    " } else { "  | " };
        let head = serde_json::to_string(&format!(".{name}")).expect("tool name is UTF-8");
        source.push_str(&format!("{prefix}{head} call_{index}\n"));
    }
    source.push_str("  ;\n");
    source
}
fn dispatcher_source<'a>(names: impl IntoIterator<Item = &'a str>) -> String {
    let names = names.into_iter().collect::<Vec<_>>();
    let mut source = String::new();
    for index in 0..names.len() {
        source.push_str(&format!("extern grammar args_{index};\n"));
    }
    source.push_str("start suffix;\nnt suffix ::=\n");
    for (index, name) in names.iter().enumerate() {
        let prefix = if index == 0 { "    " } else { "  | " };
        let head = serde_json::to_string(&format!(".{name}(")).expect("tool name is UTF-8");
        source.push_str(&format!("{prefix}{head} args_{index} ')'\n"));
    }
    source.push_str("  ;\n");
    source
}

fn validate_tool_names<'a>(names: impl IntoIterator<Item = &'a str>) -> crate::Result<()> {
    let mut seen = BTreeSet::new();
    for name in names {
        if !is_ascii_identifier_name(name) {
            return Err(GlrMaskError::Compilation(format!(
                "programmatic tool name {name:?} is not an ASCII JavaScript identifier name"
            )));
        }
        if !seen.insert(name) {
            return Err(GlrMaskError::Compilation(format!(
                "programmatic tool name {name:?} was supplied more than once"
            )));
        }
    }
    Ok(())
}

fn is_ascii_identifier_name(value: &str) -> bool {
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || matches!(first, b'_' | b'$'))
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vocab {
        let pieces = [
            "tools", ".lookup(", ")", "{", "}", "customer_id", "\"customer_id\"",
            ": ", "customer", ".id", "\"abc\"", "123", "[", "]", ", ", "x", "+", "1",
        ];
        Vocab::new(
            pieces
                .iter()
                .enumerate()
                .map(|(id, text)| (id as u32, text.as_bytes().to_vec()))
                .collect(),
        )
    }

    fn accepts_bytes(constraint: &Constraint, bytes: &[u8]) -> bool {
        let mut state = constraint.start();
        state.commit_bytes(bytes).is_ok() && state.is_accepting()
    }

    #[test]
    fn dynamic_expression_grammar_excludes_bare_literals() {
        let vocab = vocab();
        let compiler = ProgrammaticJsCompiler::new(&vocab).unwrap();
        let dynamic = compiler.dynamic_value_constraint();
        assert!(accepts_bytes(dynamic, b"customer"));
        assert!(accepts_bytes(dynamic, b"customer.id"));
        assert!(!accepts_bytes(dynamic, b"x + 1"));
        assert!(!accepts_bytes(dynamic, b"123"));
        assert!(!accepts_bytes(dynamic, br#""abc""#));
        assert!(!accepts_bytes(dynamic, br#"{"wrong": 123}"#));
        assert!(!accepts_bytes(dynamic, br#"[1, 2]"#));
        assert!(!accepts_bytes(dynamic, b"tools.lookup()"));
    }

    #[test]
    fn programmatic_tool_schema_accepts_dynamic_property_value_and_unquoted_key() {
        let vocab = vocab();
        let compiler = ProgrammaticJsCompiler::new(&vocab).unwrap();
        let schema = r#"{
          "type":"object",
          "properties":{"customer_id":{"type":"string"}},
          "required":["customer_id"],
          "additionalProperties":false
        }"#;
        let compiled_schema = compiler.compile_schema(schema, &vocab).unwrap();
        let constraint = compiler
            .compose_tools(&[("lookup", &compiled_schema)], &vocab)
            .unwrap();
        assert!(accepts_bytes(
            &constraint,
            b"tools.lookup({customer_id: customer.id});"
        ));
        assert!(accepts_bytes(
            &constraint,
            b"tools.lookup({customer_id:customer.id});"
        ));
        assert!(accepts_bytes(
            &constraint,
            b"tools.lookup({customer_id:\ncustomer.id});"
        ));
        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({"customer_id": "abc"});"#
        ));
        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({"customer_id":customer.id});"#
        ));
        assert!(!accepts_bytes(&constraint, b"tools.lookup(customer);"));
        assert!(!accepts_bytes(
            &constraint,
            b"tools.lookup({wrong: customer.id});"
        ));
    }

    #[test]
    fn programmatic_enum_allows_opaque_and_schema_checked_conditional() {
        let vocab = vocab();
        let compiler = ProgrammaticJsCompiler::new(&vocab).unwrap();
        let schema = r#"{
          "type":"object",
          "properties":{"status":{"enum":["open","closed"]}},
          "required":["status"],
          "additionalProperties":false
        }"#;
        let constraint = compiler
            .compile_tools(&[("lookup", schema)], &vocab)
            .unwrap();

        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({status: customer.id});"#,
        ));
        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({status: x ? "open" : "closed"});"#,
        ));
        assert!(!accepts_bytes(
            &constraint,
            br#"tools.lookup({status: x ? "open" : "bogus"});"#,
        ));
        assert!(!accepts_bytes(
            &constraint,
            br#"tools.lookup({status: "open" + x});"#,
        ));
        assert!(!accepts_bytes(
            &constraint,
            br#"tools.lookup({status: "bogus"});"#,
        ));
        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({status: customer.ready && other.flag ? "open" : "closed"});"#,
        ));
        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({status: x ? (customer.ready ? "open" : "closed") : "open"});"#,
        ));
        assert!(!accepts_bytes(
            &constraint,
            br#"tools.lookup({status: x ? (customer.ready ? "open" : "bogus") : "closed"});"#,
        ));
        assert!(!accepts_bytes(
            &constraint,
            br#"tools.unknown({status: "open"});"#,
        ));
        assert!(!accepts_bytes(
            &constraint,
            br#"tools.lookup({status: tools.unknown({}) ? "open" : "closed"});"#,
        ));
    }

    #[test]
    fn programmatic_compile_tools_reuses_shared_parent() {
        let vocab = vocab();
        let compiler = ProgrammaticJsCompiler::new(&vocab).unwrap();
        assert!(matches!(
            compiler.wrapper_parent.boundary_trigger,
            crate::runtime::BoundaryTrigger::Tokens(_)
        ));
        assert!(compiler.wrapper_parent.deferred_composition_metadata_blob.is_none());
        let schema = r#"{
          "type":"object",
          "properties":{"customer_id":{"type":"string"}},
          "required":["customer_id"],
          "additionalProperties":false
        }"#;

        let compiled_schema = compiler.compile_schema(schema, &vocab).unwrap();
        if let Some(layout) = compiled_schema.recursive_parser_layout().unwrap() {
            for leaf_index in 0..layout.leaves.len() {
                let leaf = compiled_schema.recursive_leaf_constraint(leaf_index).unwrap();
                assert!(
                    matches!(leaf.boundary_trigger, crate::runtime::BoundaryTrigger::Tokens(_)),
                    "fresh programmatic schema leaf {leaf_index} has no Tokens trigger",
                );
            }
        }

        let first = compiler.compile_tools(&[("lookup", schema)], &vocab).unwrap();
        let second = compiler
            .compile_tools(&[("lookup", schema), ("lookup2", schema)], &vocab)
            .unwrap();
        if let Some(layout) = first.recursive_parser_layout().unwrap() {
            let mut shared_wrapper_leaf_count = 0usize;
            for leaf_index in 0..layout.leaves.len() {
                let leaf = first.recursive_leaf_constraint(leaf_index).unwrap();
                if std::ptr::eq(leaf, compiler.wrapper_parent.as_ref()) {
                    shared_wrapper_leaf_count += 1;
                }
                assert!(
                    matches!(leaf.boundary_trigger, crate::runtime::BoundaryTrigger::Tokens(_)),
                    "finished programmatic constraint leaf {leaf_index} has no Tokens trigger",
                );
            }
            assert_eq!(
                shared_wrapper_leaf_count, 1,
                "one-tool composition must retain the compiler's exact shared wrapper parent"
            );
        }
        if let Some(layout) = second.recursive_parser_layout().unwrap() {
            let shared_wrapper_leaf_count = (0..layout.leaves.len())
                .filter(|&leaf_index| {
                    let leaf = second.recursive_leaf_constraint(leaf_index).unwrap();
                    std::ptr::eq(leaf, compiler.wrapper_parent.as_ref())
                })
                .count();
            assert_eq!(
                shared_wrapper_leaf_count, 2,
                "two-tool composition must reuse the exact shared wrapper parent twice"
            );
        }
        let valid = b"tools.lookup({customer_id: customer.id});";
        let valid_second_tool = b"tools.lookup2({customer_id: customer.id});";
        let valid_with_trivia = b"tools .lookup({customer_id: customer.id});";
        assert!(accepts_bytes(&first, valid));
        assert!(accepts_bytes(&second, valid));
        assert!(accepts_bytes(&second, valid_second_tool));
        assert!(accepts_bytes(&first, valid_with_trivia));
        assert!(accepts_bytes(&second, valid_with_trivia));
        let loaded = Constraint::load(first.save()).unwrap();
        assert!(accepts_bytes(&loaded, valid));
        assert!(accepts_bytes(&loaded, valid_with_trivia));
    }
    #[test]
    fn programmatic_nested_object_and_array_values_stay_schema_aware() {
        let vocab = vocab();
        let compiler = ProgrammaticJsCompiler::new(&vocab).unwrap();
        let schema = r#"{
          "type":"object",
          "properties":{
            "meta":{
              "type":"object",
              "properties":{"status":{"enum":["open","closed"]}},
              "required":["status"],
              "additionalProperties":false
            },
            "ids":{"type":"array","items":{"type":"string"}}
          },
          "required":["meta","ids"],
          "additionalProperties":false
        }"#;
        let constraint = compiler
            .compile_tools(&[("lookup", schema)], &vocab)
            .unwrap();

        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({meta: {status: customer.status}, ids: [customer.id, other.id]});"#,
        ));
        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({meta: {status: x ? "open" : "closed"}, ids: [customer.id]});"#,
        ));
        assert!(!accepts_bytes(
            &constraint,
            br#"tools.lookup({meta: {status: "bogus"}, ids: [customer.id]});"#,
        ));
        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({meta: customer, ids: [customer.id]});"#,
        ));
        assert!(!accepts_bytes(
            &constraint,
            br#"tools.lookup({meta: {status: customer.status, extra: customer.id}, ids: [customer.id]});"#,
        ));
    }
}
