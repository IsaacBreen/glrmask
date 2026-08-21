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

use crate::compiler::constraint_compose::{CompiledSubgrammarInput, compose_constraints};
use crate::{Constraint, GlrMaskError, Vocab};

const JAVASCRIPT_GLRM: &str = include_str!("programmatic_js/javascript.glrm");
const PARENT_PLACEHOLDER_NAME: &str = "PROGRAMMATIC_TOOL_SUFFIX";

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
    parent: Constraint,
    dynamic_value: Constraint,
    condition: Constraint,
    parent_placeholder_terminal: u32,
}

impl ProgrammaticJsCompiler {
    /// Compile all reusable programmatic-JavaScript components for `vocab`.
    pub fn new(vocab: &Vocab) -> crate::Result<Self> {
        let parent = Self::compile_parent(vocab)?;
        let dynamic_value = Self::compile_dynamic_value(vocab)?;
        let condition = Self::compile_condition(vocab)?;
        Self::from_components(parent, dynamic_value, condition)
    }

    /// Compile the reusable full-JavaScript parent containing the reserved
    /// `tools` dispatcher boundary. This is independent of any concrete tool
    /// schemas and may be built once per vocabulary.
    pub fn compile_parent(vocab: &Vocab) -> crate::Result<Constraint> {
        let placeholder_token_id =
            crate::import::external_placeholder_token_id_avoiding(vocab, std::iter::empty())?;
        Constraint::from_glrm_grammar(&programmatic_parent_source(placeholder_token_id)?, vocab)
    }

    /// Compile the reusable opaque-runtime-value expression subgrammar.
    pub fn compile_dynamic_value(vocab: &Vocab) -> crate::Result<Constraint> {
        Constraint::from_glrm_grammar(&dynamic_value_source()?, vocab)
    }

    /// Compile the reusable unrestricted JavaScript condition subgrammar used
    /// only as the test of schema-aware conditional expressions.
    pub fn compile_condition(vocab: &Vocab) -> crate::Result<Constraint> {
        Constraint::from_glrm_grammar(&condition_source()?, vocab)
    }

    /// Assemble a reusable compiler from independently compiled shared parts.
    /// This exists so build systems and benchmarks can time/cache each shared
    /// component separately without changing programmatic-tool semantics.
    pub fn from_components(
        parent: Constraint,
        dynamic_value: Constraint,
        condition: Constraint,
    ) -> crate::Result<Self> {
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
            parent,
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

    /// Compose already-compiled tool schemas into a named dispatcher and link
    /// it into the reusable full-JavaScript parent.
    pub fn compose_tools(
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
        let dispatcher = Constraint::from_glrm_grammar_with_subgrammars(
            &dispatcher_source,
            &borrowed,
            vocab,
        )?;
        compose_constraints(
            &self.parent,
            &[CompiledSubgrammarInput {
                placeholder_terminal: self.parent_placeholder_terminal,
                constraint: &dispatcher,
            }],
            vocab,
        )
        .map(|composition| composition.constraint)
        .map_err(GlrMaskError::Compilation)
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

fn dispatcher_source<'a>(names: impl IntoIterator<Item = &'a str>) -> String {
    let names = names.into_iter().collect::<Vec<_>>();
    let mut source = String::new();
    for index in 0..names.len() {
        source.push_str(&format!("extern g args_{index};\n"));
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
        state.commit_bytes(bytes).is_ok() && state.is_finished()
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
