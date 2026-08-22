//! First-class programmatic JavaScript tool calling.
//!
//! [`ProgrammaticJsCompiler`] separates the reusable JavaScript work from the
//! per-tool-set work:
//!
//! 1. compile the full JavaScript parent and one ordinary JavaScript
//!    `assignment_expression` child once per vocabulary;
//! 2. compile each tool's JSON Schema structurally: object-shaped nested values
//!    recurse, while non-object value positions become the shared JS expression
//!    slot;
//! 3. compose those compiled schemas into a tool dispatcher, then link that
//!    dispatcher into `tools.<name>(...)` call sites in the full JS parent.
//!
//! The schema root is never replaced by the expression child, so the top-level
//! tool-arguments object and its required/allowed keys remain constrained.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::compiler::constraint_compose::CompiledSubgrammarInput;
use crate::{Constraint, GlrMaskError, Vocab};

const JAVASCRIPT_GLRM: &str = include_str!("programmatic_js/javascript.glrm");
const PARENT_PLACEHOLDER_NAME: &str = "PROGRAMMATIC_TOOL_SUFFIX";


pub(crate) const JAVASCRIPT_IGNORE_RULES_GLRM: &str = r#"
ignore IGNORE;
t IGNORE ::= ( ' ' | '\t' | '\n' | '\r' )+ | _COMMENT ;
t _COMMENT ::= _SINGLE_LINE_COMMENT | _MULTI_LINE_COMMENT ;
t _SINGLE_LINE_COMMENT ::= '//' ( [^\n\r]/utf8 )* ( '\r\n' | '\r' | '\n' ) ;
t _MULTI_LINE_COMMENT ::= '/*' ( [^*]/utf8 | '*' [^/]/utf8 )* '*/' ;
"#;







/// Intermediate compiled tool-arguments schema. Hidden JavaScript-value linker
/// terminals remain unresolved until sibling tool schemas have been combined,
/// so one shared expression child serves every non-object value position.
#[derive(Debug)]
pub struct ProgrammaticJsSchema {
    constraint: Constraint,
    protected_terminals: Vec<u32>,
}

/// Tool dispatcher with schema call sites assembled but the shared JavaScript
/// value child still unresolved. Keeping this intermediate typed
/// prevents an unfinished constraint from being mistaken for a runnable one.
#[derive(Debug)]
pub struct ProgrammaticJsDispatcher {
    constraint: Constraint,
}

/// Reusable compiler for programmatic JavaScript tool calling.
#[derive(Debug)]
pub struct ProgrammaticJsCompiler {
    parent: Arc<Constraint>,
    value_expression: Arc<Constraint>,
    parent_placeholder_terminal: u32,
}

impl ProgrammaticJsCompiler {
    /// Compile all reusable programmatic-JavaScript components for `vocab`.
    pub fn new(vocab: &Vocab) -> crate::Result<Self> {
        let parent = Arc::new(Self::compile_parent(vocab)?);
        let value_expression = Arc::new(Self::compile_value_expression(vocab)?);
        Self::from_shared_components(parent, value_expression)
    }

    /// Compile the reusable full-JavaScript parent containing the reserved
    /// `tools` dispatcher boundary. This is independent of any concrete tool
    /// schemas and may be built once per vocabulary.
    pub fn compile_parent(vocab: &Vocab) -> crate::Result<Constraint> {
        crate::import::compile_glrm_with_protected_shift_terminals(
            &programmatic_parent_source()?,
            &[PARENT_PLACEHOLDER_NAME],
            vocab,
        )
    }

    /// Compile the reusable ordinary JavaScript value-expression subgrammar.
    /// Non-object schema value positions accept this language without attempting
    /// static JSON-Schema type/enum validation.
    pub fn compile_value_expression(vocab: &Vocab) -> crate::Result<Constraint> {
        Constraint::from_glrm_grammar(&value_expression_source()?, vocab)
    }

    /// Backward-compatible name for the shared value-expression component.
    pub fn compile_dynamic_value(vocab: &Vocab) -> crate::Result<Constraint> {
        Self::compile_value_expression(vocab)
    }

    /// Assemble a reusable compiler from independently compiled shared parts.
    /// This exists so build systems and benchmarks can time/cache each shared
    /// component separately without changing programmatic-tool semantics.
    pub fn from_components(
        parent: Constraint,
        value_expression: Constraint,
    ) -> crate::Result<Self> {
        Self::from_shared_components(Arc::new(parent), Arc::new(value_expression))
    }

    /// Assemble from shared component artifacts without deep cloning. This is
    /// used by language bindings and serving systems that already cache the
    /// reusable constraints behind `Arc`.
    pub fn from_shared_components(
        parent: Arc<Constraint>,
        value_expression: Arc<Constraint>,
    ) -> crate::Result<Self> {
        let parent_placeholder_terminal = parent
            .terminal_display_names()
            .iter()
            .position(|name| name == PARENT_PLACEHOLDER_NAME)
            .map(|index| index as u32)
            .ok_or_else(|| {
                GlrMaskError::Compilation(
                    "programmatic JavaScript parent has no dispatcher linker terminal".into(),
                )
            })?;
        Ok(Self {
            parent,
            value_expression,
            parent_placeholder_terminal,
        })
    }

    /// The reusable JavaScript expression child used by every non-object nested
    /// schema value compiled through this compiler.
    pub fn value_expression_constraint(&self) -> &Constraint {
        self.value_expression.as_ref()
    }

    /// Backward-compatible accessor name.
    pub fn dynamic_value_constraint(&self) -> &Constraint {
        self.value_expression_constraint()
    }

    /// Compile one tool arguments schema. The schema root stays structural;
    /// nested object-valued schemas recurse, while non-object values become
    /// arbitrary JavaScript assignment expressions.
    pub fn compile_schema(
        &self,
        schema: &str,
        vocab: &Vocab,
    ) -> crate::Result<ProgrammaticJsSchema> {
        let constraint = Constraint::from_json_schema_with_programmatic_placeholders(schema, vocab)?;
        let protected_terminals = ["__GLRMASK_PTC_VALUE"]
            .into_iter()
            .map(|marker| {
                constraint
                    .terminal_display_names()
                    .iter()
                    .position(|name| name == marker)
                    .map(|index| index as u32)
                    .ok_or_else(|| {
                        GlrMaskError::Compilation(format!(
                            "programmatic schema lost protected linker terminal {marker:?}"
                        ))
                    })
            })
            .collect::<crate::Result<Vec<_>>>()?;
        Ok(ProgrammaticJsSchema {
            constraint,
            protected_terminals,
        })
    }

    /// Compile a named tool dispatcher from already-compiled tool schemas.
    /// This is separate from the outer JavaScript link so callers can time or
    /// configure the two composition stages independently.
    pub fn compile_dispatcher(
        &self,
        tools: &[(&str, &ProgrammaticJsSchema)],
        vocab: &Vocab,
    ) -> crate::Result<ProgrammaticJsDispatcher> {
        validate_tool_names(tools.iter().map(|(name, _)| *name))?;
        if tools.is_empty() {
            return Err(GlrMaskError::Compilation(
                "programmatic JavaScript requires at least one tool".into(),
            ));
        }

        let (dispatcher_source, argument_markers) =
            dispatcher_source(tools.iter().map(|(name, _)| *name));
        let protected_names = argument_markers.iter().map(String::as_str).collect::<Vec<_>>();
        let dispatcher_parent = crate::import::compile_glrm_with_protected_shift_terminals(
            &dispatcher_source,
            &protected_names,
            vocab,
        )?;
        let placeholder_terminals = argument_markers
            .iter()
            .map(|marker| {
                dispatcher_parent
                    .terminal_display_names()
                    .iter()
                    .position(|name| name == marker)
                    .map(|index| index as u32)
                    .ok_or_else(|| {
                        GlrMaskError::Compilation(format!(
                            "programmatic dispatcher lost argument linker terminal {marker:?}"
                        ))
                    })
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let composition_inputs = tools
            .iter()
            .zip(placeholder_terminals.iter())
            .map(|((_, schema), &placeholder_terminal)| CompiledSubgrammarInput {
                placeholder_terminal,
                additional_placeholder_terminals: &[],
                constraint: &schema.constraint,
                protected_terminals: &schema.protected_terminals,
            })
            .collect::<Vec<_>>();
        let profile = std::env::var_os("GLRMASK_PROFILE_PROGRAMMATIC_JS").is_some();
        let dispatcher_started = std::time::Instant::now();
        let dispatcher = crate::compiler::constraint_compose::compose_constraints_owned_parent(
            dispatcher_parent,
            &composition_inputs,
            vocab,
        )
        .map(|composition| composition.constraint)
        .map_err(GlrMaskError::Compilation)?;
        if profile {
            eprintln!(
                "[glrmask/ptc-profile] schema_dispatcher_ms={:.3}",
                dispatcher_started.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(ProgrammaticJsDispatcher {
            constraint: dispatcher,
        })
    }

    /// Link a compiled tool dispatcher into the reusable full-JavaScript parent.
    pub fn compose_dispatcher(
        &self,
        dispatcher: &ProgrammaticJsDispatcher,
        vocab: &Vocab,
    ) -> crate::Result<Constraint> {
        self.compose_dispatcher_constraint(dispatcher.constraint.clone(), vocab)
    }

    /// Consuming variant that avoids cloning the intermediate dispatcher.
    pub fn compose_dispatcher_owned(
        &self,
        dispatcher: ProgrammaticJsDispatcher,
        vocab: &Vocab,
    ) -> crate::Result<Constraint> {
        self.compose_dispatcher_constraint(dispatcher.constraint, vocab)
    }

    fn bind_named_markers(
        &self,
        parent: Constraint,
        marker: &str,
        child: &Arc<Constraint>,
        vocab: &Vocab,
    ) -> crate::Result<Constraint> {
        let marker_suffix = format!("::{marker}");
        let terminals = parent
            .terminal_display_names()
            .iter()
            .enumerate()
            .filter(|(_, name)| name.as_str() == marker || name.ends_with(&marker_suffix))
            .map(|(index, _)| index as u32)
            .collect::<Vec<_>>();
        if terminals.is_empty() {
            return Err(GlrMaskError::Compilation(format!(
                "programmatic constraint lost marker terminal {marker:?}"
            )));
        }
        let (&placeholder_terminal, additional_placeholder_terminals) = terminals
            .split_first()
            .expect("non-empty marker terminal list was checked above");
        let input = CompiledSubgrammarInput {
            placeholder_terminal,
            additional_placeholder_terminals,
            constraint: child.as_ref(),
            protected_terminals: &[],
        };
        let composition = if std::env::var_os("GLRMASK_PTC_DYNAMIC_COMPOSITION").is_some() {
            // Experimental exact low-build-latency path: the generic linker can
            // publish the composed table/tokenizer directly as a Dynamic runtime
            // constraint, avoiding static boundary-DWA construction entirely.
            crate::compiler::constraint_compose::compose_constraints(
                &parent,
                std::slice::from_ref(&input),
                vocab,
            )
        } else {
            crate::compiler::constraint_compose::compose_constraints_owned_parent_shared(
                parent,
                std::slice::from_ref(&input),
                std::slice::from_ref(child),
                vocab,
            )
        };
        composition
            .map(|composition| composition.constraint)
            .map_err(GlrMaskError::Compilation)
    }

    fn compose_dispatcher_constraint(
        &self,
        mut dispatcher: Constraint,
        vocab: &Vocab,
    ) -> crate::Result<Constraint> {
        let profile = std::env::var_os("GLRMASK_PROFILE_PROGRAMMATIC_JS").is_some();
        let started = std::time::Instant::now();
        dispatcher = self.bind_named_markers(
            dispatcher,
            "__GLRMASK_PTC_VALUE",
            &self.value_expression,
            vocab,
        )?;
        if profile {
            eprintln!(
                "[glrmask/ptc-profile] bind_all_values ms={:.3}",
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        let started = std::time::Instant::now();
        let input = CompiledSubgrammarInput {
            placeholder_terminal: self.parent_placeholder_terminal,
            additional_placeholder_terminals: &[],
            constraint: &dispatcher,
            protected_terminals: &[],
        };
        let composition = if std::env::var_os("GLRMASK_PTC_DYNAMIC_COMPOSITION").is_some() {
            crate::compiler::constraint_compose::compose_constraints(
                self.parent.as_ref(),
                std::slice::from_ref(&input),
                vocab,
            )
        } else {
            crate::compiler::constraint_compose::compose_constraints_owned_parent(
                self.parent.as_ref().clone(),
                std::slice::from_ref(&input),
                vocab,
            )
        };
        let result = composition
            .map(|composition| composition.constraint)
            .map_err(GlrMaskError::Compilation);
        if profile {
            eprintln!(
                "[glrmask/ptc-profile] outer_js_link ms={:.3}",
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        result
    }

    /// Compose already-compiled tool schemas into a named dispatcher and link
    /// it into the reusable full-JavaScript parent.
    pub fn compose_tools(
        &self,
        tools: &[(&str, &ProgrammaticJsSchema)],
        vocab: &Vocab,
    ) -> crate::Result<Constraint> {
        let dispatcher = self.compile_dispatcher(tools, vocab)?;
        self.compose_dispatcher(&dispatcher, vocab)
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
            .iter()
            .map(|(name, schema)| {
                self.compile_schema(schema, vocab)
                    .map(|constraint| (*name, constraint))
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let borrowed = compiled
            .iter()
            .map(|(name, constraint)| (*name, constraint))
            .collect::<Vec<_>>();
        self.compose_tools(&borrowed, vocab)
    }
}

/// Canonical full JavaScript grammar bundled with GLRMask's programmatic tool
/// calling support. It omits CFA's textual EOF sentinel; model end-token IDs
/// remain a separate runtime concern.
pub fn javascript_glrm() -> &'static str {
    JAVASCRIPT_GLRM
}

fn reserve_tools_identifier_source(source: &mut String) -> crate::Result<()> {
    // IDENTIFIER is already encoded as a trie that excludes JavaScript
    // keywords. Reserve one additional exact identifier using the same shape
    // instead of a generic language subtraction: `IDENTIFIER - "tools"`
    // creates a very large weighted terminal DWA on real model vocabularies.
    const OLD: &str = "  | 't' [ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgijklmnopqstuvwxz0123456789$_] _IDENTIFIER_PART*";
    const NEW: &str = "  | 't' [ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgijklmnpqstuvwxz0123456789$_] _IDENTIFIER_PART*\n  | 'to'\n  | 'to' [ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnpqrstuvwxyz0123456789$_] _IDENTIFIER_PART*\n  | 'too'\n  | 'too' [ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz0123456789$_] _IDENTIFIER_PART*\n  | 'tool'\n  | 'tool' [ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrtuvwxyz0123456789$_] _IDENTIFIER_PART*\n  | 'tools' [ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789$_] _IDENTIFIER_PART*";
    let count = source.matches(OLD).count();
    if count != 1 {
        return Err(GlrMaskError::Compilation(format!(
            "bundled JavaScript IDENTIFIER trie changed unexpectedly (expected one tools-reservation insertion point, found {count})"
        )));
    }
    *source = source.replacen(OLD, NEW, 1);
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
    reserve_tools_identifier_source(&mut source)?;

    let mut grammar = crate::grammar::glrm::from_glrm(&source)?;
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

fn value_expression_source() -> crate::Result<String> {
    pruned_javascript_source(
        "programmatic_value_expression",
        "nt programmatic_value_expression ::= ':' assignment_expression;\n",
    )
}

fn programmatic_parent_source() -> crate::Result<String> {
    let mut source = JAVASCRIPT_GLRM.to_owned();
    reserve_tools_identifier_source(&mut source)?;

    // Keep the canonical grammar text intact apart from the one PTC branch.
    // This intentionally mirrors the previously benchmarked CFA parent path
    // and avoids an AST dump/reparse of the full JavaScript grammar.
    const NEEDLE: &str = "nt member_expression_with_suffixes ::=\n    primary_expression";
    let replacement = format!(
        "nt member_expression_with_suffixes ::=\n    'tools' {PARENT_PLACEHOLDER_NAME}\n  | primary_expression"
    );
    if source.matches(NEEDLE).count() != 1 {
        return Err(GlrMaskError::Compilation(
            "bundled JavaScript member-expression grammar changed unexpectedly".into(),
        ));
    }
    source = source.replacen(NEEDLE, &replacement, 1);
    source.push_str(&format!(
        "\n// hidden programmatic-tools linker terminal\nt {PARENT_PLACEHOLDER_NAME} ::= '__GLRMASK_PTC_DISPATCH_7F3A9C__';\n"
    ));
    Ok(source)
}

fn dispatcher_source<'a>(names: impl IntoIterator<Item = &'a str>) -> (String, Vec<String>) {
    let names = names.into_iter().collect::<Vec<_>>();
    let argument_markers = (0..names.len())
        .map(|index| format!("PROGRAMMATIC_ARGS_{index}"))
        .collect::<Vec<_>>();
    let mut source = String::new();
    source.push_str("start suffix;\n");
    source.push_str(JAVASCRIPT_IGNORE_RULES_GLRM);
    for (index, marker) in argument_markers.iter().enumerate() {
        let bytes = serde_json::to_string(&format!(
            "__GLRMASK_PTC_ARGS_{index}_7F3A9C__"
        ))
        .expect("argument marker is UTF-8");
        source.push_str(&format!("t {marker} ::= {bytes};\n"));
    }
    source.push_str("nt suffix ::=\n");
    for (index, name) in names.iter().enumerate() {
        let prefix = if index == 0 { "    " } else { "  | " };
        let head = serde_json::to_string(&format!(".{name}(")).expect("tool name is UTF-8");
        source.push_str(&format!(
            "{prefix}{head} {} ')'\n",
            argument_markers[index]
        ));
    }
    source.push_str("  ;\n");
    (source, argument_markers)
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
        state.commit_bytes(bytes).is_ok() && state.is_finished()
    }

    #[test]
    fn value_expression_grammar_accepts_arbitrary_javascript_values() {
        let vocab = vocab();
        let compiler = ProgrammaticJsCompiler::new(&vocab).unwrap();
        let value = compiler.value_expression_constraint();
        assert!(accepts_bytes(value, b":customer"));
        assert!(accepts_bytes(value, b":customer.id"));
        assert!(accepts_bytes(value, b":customer /* comment */ . id"));
        assert!(accepts_bytes(value, b":x + 1"));
        assert!(accepts_bytes(value, b":123"));
        assert!(accepts_bytes(value, br#":"abc""#));
        assert!(accepts_bytes(value, br#":({wrong: 123})"#));
        assert!(accepts_bytes(value, br#":[1, 2]"#));
        // `tools` remains reserved so nested tool calls cannot bypass the
        // schema dispatcher. Bind a tool result to a variable first instead.
        assert!(!accepts_bytes(value, b":tools.lookup()"));
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
    fn programmatic_reuses_one_dynamic_child_across_multiple_tool_schemas() {
        let vocab = vocab();
        let compiler = ProgrammaticJsCompiler::new(&vocab).unwrap();
        let schema = r#"{
          "type":"object",
          "properties":{"customer_id":{"type":"string"}},
          "required":["customer_id"],
          "additionalProperties":false
        }"#;
        let left = compiler.compile_schema(schema, &vocab).unwrap();
        let right = compiler.compile_schema(schema, &vocab).unwrap();
        let dispatcher = compiler
            .compile_dispatcher(&[("lookup", &left), ("other", &right)], &vocab)
            .unwrap();
        let value_markers = dispatcher
            .constraint
            .terminal_display_names()
            .iter()
            .filter(|name| name.ends_with("::__GLRMASK_PTC_VALUE"))
            .count();
        assert_eq!(value_markers, 2);
        let constraint = compiler.compose_dispatcher(&dispatcher, &vocab).unwrap();
        assert!(accepts_bytes(
            &constraint,
            b"tools.lookup({customer_id: customer.id});"
        ));
        assert!(accepts_bytes(
            &constraint,
            b"tools.other({customer_id: customer.id});"
        ));
        assert!(!accepts_bytes(&constraint, b"tools.other(customer);"));
    }

    #[test]
    fn programmatic_scalar_values_are_unconstrained_javascript() {
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
        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({status: x ? "open" : "bogus"});"#,
        ));
        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({status: "open" + x});"#,
        ));
        assert!(accepts_bytes(
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
        assert!(accepts_bytes(
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
    fn programmatic_nested_objects_stay_structural_but_leaf_values_are_unconstrained() {
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
        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({meta: {status: "bogus"}, ids: [customer.id]});"#,
        ));
        // `meta` is itself schema-declared as an object, so it must retain the
        // nested object structure rather than being replaced wholesale by a JS
        // expression. Its leaf `status` value is unrestricted.
        assert!(!accepts_bytes(
            &constraint,
            br#"tools.lookup({meta: customer, ids: [customer.id]});"#,
        ));
        // Arrays are ordinary value expressions under the current PTC policy.
        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({meta: {status: whatever}, ids: customer.ids});"#,
        ));
        assert!(!accepts_bytes(
            &constraint,
            br#"tools.lookup({meta: {status: customer.status, extra: customer.id}, ids: [customer.id]});"#,
        ));
    }
}
