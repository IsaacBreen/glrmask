from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> str:
    return (ROOT / path).read_text()


def save(path: str, text: str) -> None:
    (ROOT / path).write_text(text)


def replace(path: str, old: str, new: str) -> None:
    text = load(path)
    if old in text:
        save(path, text.replace(old, new, 1))
    elif new not in text:
        raise RuntimeError(f"{path}: replacement target missing")


# Add generic exact-token augmentation at the named-grammar boundary.
replace(
    "src/import/mod.rs",
    "pub mod numeric_range;\n\nuse crate::compiler::compile::{",
    "pub mod numeric_range;\n\nuse std::collections::BTreeSet;\n\nuse crate::compiler::compile::{",
)
replace(
    "src/import/mod.rs",
    '''pub(crate) fn sequence_or_single(mut items: Vec<ast::GrammarExpr>) -> ast::GrammarExpr {
    match items.len() {
        0 => ast::GrammarExpr::Sequence(Vec::new()),
        1 => items.pop().unwrap(),
        _ => ast::GrammarExpr::Sequence(items),
    }
}
''',
    '''pub(crate) fn sequence_or_single(mut items: Vec<ast::GrammarExpr>) -> ast::GrammarExpr {
    match items.len() {
        0 => ast::GrammarExpr::Sequence(Vec::new()),
        1 => items.pop().unwrap(),
        _ => ast::GrammarExpr::Sequence(items),
    }
}

fn append_end_token_choice(grammar: &mut ast::NamedGrammar, end_token_ids: &[u32]) {
    let end_token_ids = end_token_ids.iter().copied().collect::<BTreeSet<_>>();
    if end_token_ids.is_empty() {
        return;
    }

    let original_start = grammar.start.clone();
    let base = "__glrmask_start_with_end_token";
    let mut generated_start = base.to_owned();
    let mut suffix = 2usize;
    while grammar.rules.iter().any(|rule| rule.name == generated_start) {
        generated_start = format!("{base}_{suffix}");
        suffix += 1;
    }

    let end = choice_or_single(
        end_token_ids
            .into_iter()
            .map(ast::GrammarExpr::SpecialToken)
            .collect(),
    );
    grammar.rules.push(ast::NamedRule {
        name: generated_start.clone(),
        expr: sequence_or_single(vec![ast::GrammarExpr::Ref(original_start), end]),
        is_terminal: false,
        is_internal: false,
    });
    grammar.start = generated_start;
}
''',
)
replace(
    "src/import/mod.rs",
    '''fn lower_factored_named_grammar(
    source: &str,
    parse_named: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
) -> crate::Result<GrammarDef> {''',
    '''fn lower_factored_named_grammar(
    source: &str,
    parse_named: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<GrammarDef> {''',
)
replace(
    "src/import/mod.rs",
    '''        emit_import_phase_end("transform_named_grammar", transform_started_at);
    }

    let ast_lower_started_at''',
    '''        emit_import_phase_end("transform_named_grammar", transform_started_at);
    }
    append_end_token_choice(&mut factored, end_token_ids);

    let ast_lower_started_at''',
)
replace(
    "src/import/mod.rs",
    '''    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
) -> crate::Result<Constraint> {''',
    '''    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<Constraint> {''',
)
replace(
    "src/import/mod.rs",
    "lower_factored_named_grammar(source, parse, transform)?",
    "lower_factored_named_grammar(source, parse, transform, end_token_ids)?",
)
replace(
    "src/import/mod.rs",
    "lower_factored_named_grammar(source, parse, transform)?",
    "lower_factored_named_grammar(source, parse, transform, end_token_ids)?",
)
replace(
    "src/import/mod.rs",
    '''    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
) -> crate::Result<DynamicConstraint> {''',
    '''    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<DynamicConstraint> {''',
)
replace(
    "src/import/mod.rs",
    "lower_factored_named_grammar(source, parse, transform)?",
    "lower_factored_named_grammar(source, parse, transform, end_token_ids)?",
)
replace(
    "src/import/mod.rs",
    '''        Some(json_schema::prepare_named_grammar),
    )?;
    std::hint::black_box(&grammar);''',
    '''        Some(json_schema::prepare_named_grammar),
        &[],
    )?;
    std::hint::black_box(&grammar);''',
)

# Ordinary grammar constructors have no implicit end token.
for old in [
    '''            ebnf::parse_ebnf_to_named,
            None,
        )''',
    '''            lark::parse_lark_to_named,
            None,
        )''',
    '''            crate::grammar::glrm::from_glrm,
            None,
        )''',
]:
    new = old.replace("            None,\n", "            None,\n            &[],\n")
    text = load("src/import/mod.rs")
    count = text.count(old)
    if count:
        save("src/import/mod.rs", text.replace(old, new))

replace(
    "src/import/mod.rs",
    '''    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        crate::compiler::stages::id_map_and_terminal_dwa::l2p::with_ti_pool(|| {
            compile_from_source(
                schema,
                vocab,
                "json_schema",
                GlrTableConstruction::LegacyRowBisim,
                parse_json_schema_to_named,
                Some(json_schema::prepare_named_grammar),
            )
        })
    }
''',
    '''    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_json_schema_with_end_tokens(schema, vocab, &[])
    }

    pub fn from_json_schema_with_end_tokens(
        schema: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        crate::compiler::stages::id_map_and_terminal_dwa::l2p::with_ti_pool(|| {
            compile_from_source(
                schema,
                vocab,
                "json_schema",
                GlrTableConstruction::LegacyRowBisim,
                parse_json_schema_to_named,
                Some(json_schema::prepare_named_grammar),
                end_token_ids,
            )
        })
    }
''',
)
replace(
    "src/import/mod.rs",
    '''    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_dynamic_from_source(
            schema,
            vocab,
            GlrTableConstruction::Lalr,
            parse_json_schema_to_named,
            Some(json_schema::prepare_named_grammar),
        )
    }
''',
    '''    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_json_schema_with_end_tokens(schema, vocab, &[])
    }

    pub fn from_json_schema_with_end_tokens(
        schema: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        compile_dynamic_from_source(
            schema,
            vocab,
            GlrTableConstruction::Lalr,
            parse_json_schema_to_named,
            Some(json_schema::prepare_named_grammar),
            end_token_ids,
        )
    }
''',
)

# Python exposes the schema-level role, not EOS metadata on Vocab.
replace(
    "python/src/lib.rs",
    '''    #[staticmethod]
    fn from_json_schema(schema: &str, vocab: &PyVocab) -> PyResult<Self> {
        Self::from_constraint_result(
            glrmask::Constraint::from_json_schema(schema, &vocab.inner),
            vocab,
        )
    }
''',
    '''    #[staticmethod]
    #[pyo3(signature = (schema, vocab, end_token_ids=None))]
    fn from_json_schema(
        schema: &str,
        vocab: &PyVocab,
        end_token_ids: Option<Vec<u32>>,
    ) -> PyResult<Self> {
        Self::from_constraint_result(
            glrmask::Constraint::from_json_schema_with_end_tokens(
                schema,
                &vocab.inner,
                end_token_ids.as_deref().unwrap_or(&[]),
            ),
            vocab,
        )
    }
''',
)
replace(
    "python/src/lib.rs",
    '''    #[staticmethod]
    fn from_json_schema(schema: &str, vocab: &PyVocab) -> PyResult<Self> {
        Self::from_constraint_result(
            glrmask::DynamicConstraint::from_json_schema(schema, &vocab.inner),
            vocab,
        )
    }
''',
    '''    #[staticmethod]
    #[pyo3(signature = (schema, vocab, end_token_ids=None))]
    fn from_json_schema(
        schema: &str,
        vocab: &PyVocab,
        end_token_ids: Option<Vec<u32>>,
    ) -> PyResult<Self> {
        Self::from_constraint_result(
            glrmask::DynamicConstraint::from_json_schema_with_end_tokens(
                schema,
                &vocab.inner,
                end_token_ids.as_deref().unwrap_or(&[]),
            ),
            vocab,
        )
    }
''',
)

# Retire the old metadata-EOS integration test; exact-token tests remain.
text = load("tests/integration.rs")
text = re.sub(
    r"\n#\[test\]\nfn plain_eos_mask_and_commit_agree\(\) \{.*?\n\}\n",
    "\n",
    text,
    flags=re.S,
)
text = text.replace(
    "fn named_special_terminal_and_explicit_special_eos_are_parser_controlled()",
    "fn named_special_terminal_and_explicit_end_token_are_parser_controlled()",
)
save("tests/integration.rs", text)

# Focused tests for grammar augmentation and caller-sized masks.
path = "src/import/mod.rs"
text = load(path)
marker = """    #[test]
    fn glrm_import_uses_core_merged_table_by_default() {"""
addition = '''    fn token_allowed(mask: &[u32], token_id: u32) -> bool {
        mask.get(token_id as usize / 32)
            .is_some_and(|word| word & (1u32 << (token_id % 32)) != 0)
    }

    #[test]
    fn json_schema_end_tokens_are_exact_parser_terminals() {
        let vocab = vocab(&["\\\"", "a", "\\\"a\\\""]);
        let constraint = Constraint::from_json_schema_with_end_tokens(
            r#"{"const":"a"}"#,
            &vocab,
            &[101, 100, 101],
        )
        .unwrap();
        assert_eq!(constraint.mask_len(), 4);

        let mut state = constraint.start();
        assert!(!token_allowed(&state.mask(), 100));
        assert!(!token_allowed(&state.mask(), 101));
        state.commit_token(2).unwrap();
        assert!(!state.is_complete());
        let mask = state.mask();
        assert!(token_allowed(&mask, 100));
        assert!(token_allowed(&mask, 101));
        assert_eq!(state.forced(), Vec::<u32>::new());
        state.commit_token(100).unwrap();
        assert!(state.is_complete());
    }

    #[test]
    fn caller_sized_masks_zero_unknown_trailing_tokens() {
        let vocab = vocab(&["\\\"a\\\""]);
        let constraint = Constraint::from_json_schema_with_end_tokens(
            r#"{"const":"a"}"#,
            &vocab,
            &[100],
        )
        .unwrap();
        let state = constraint.start();
        let mut oversized = vec![u32::MAX; constraint.mask_len() + 3];
        state.fill_mask(&mut oversized);
        assert!(oversized[constraint.mask_len()..].iter().all(|&word| word == 0));

        let dynamic = DynamicConstraint::from_json_schema_with_end_tokens(
            r#"{"const":"a"}"#,
            &vocab,
            &[100],
        )
        .unwrap();
        let mut dynamic_mask = vec![u32::MAX; dynamic.mask_len() + 3];
        dynamic.start().fill_mask(&mut dynamic_mask);
        assert!(dynamic_mask[dynamic.mask_len()..].iter().all(|&word| word == 0));
    }

'''
if addition not in text:
    if marker not in text:
        raise RuntimeError("test insertion marker missing")
    text = text.replace(marker, addition + marker, 1)
save(path, text)
