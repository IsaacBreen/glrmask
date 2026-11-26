use glrmask::{CompileOptions, Grammar, MaskStrategy, Vocab};

const JAVASCRIPT: &str = include_str!("../src/programmatic_js/javascript.glrm");

fn bytes_vocab() -> Vocab {
    Vocab::new((0..=255u32).map(|b| (b, vec![b as u8])).collect::<Vec<_>>()).unwrap()
}

#[test]
fn public_regex_slash_inside_classes() {
    let vocab = bytes_vocab();
    let cases = [
        (r"/[/]/", b"/".as_slice(), b"a".as_slice()),
        (r"/[^/]/", b"a".as_slice(), b"/".as_slice()),
        (r"b/[/]/", b"/".as_slice(), b"a".as_slice()),
        (r"/[[/][a]]/", b"/".as_slice(), b"b".as_slice()),
        (r"/[]/]/", b"]".as_slice(), b"a".as_slice()),
        (r"/[\[/]/", b"[".as_slice(), b"a".as_slice()),
        (r"/[a/]/i", b"A".as_slice(), b"b".as_slice()),
        (r"/\//", b"/".as_slice(), b"a".as_slice()),
        (r"/[\/]/", b"/".as_slice(), b"a".as_slice()),
        (r"/./s", b"\n".as_slice(), b"".as_slice()),
        ("/a # [comment\n b/x", b"ab".as_slice(), b"a".as_slice()),
        ("/(?x:a # [comment\n b)/", b"ab".as_slice(), b"a".as_slice()),
        ("/(?x:[a # ] comment\n /])/", b"/".as_slice(), b"b".as_slice()),
        ("/[a # comment\n /]/x", b"/".as_slice(), b"b".as_slice()),
        ("/[a # ] comment\n \\/]/x", b"/".as_slice(), b"b".as_slice()),
    ];
    for (pattern, yes, no) in cases {
        let source = format!("start root; t RX = {pattern}; nt root = RX;");
        let grammar = Grammar::from_glrm(&source).unwrap_or_else(|e| panic!("{pattern}: {e}"));
        let c = grammar.compile(&vocab).unwrap_or_else(|e| panic!("{pattern}: {e}"));
        let mut state = c.start();
        assert!(state.commit_bytes(yes) && state.is_accepting(), "{pattern}: {yes:?}");
        let mut state = c.start();
        assert!(!state.commit_bytes(no) || !state.is_accepting(), "{pattern}: {no:?}");
    }
    assert!(Grammar::from_glrm("start root; nt root = /[a/;").is_err());
    assert!(Grammar::from_glrm("start root; pragma unknown { /[/}]/ } nt root = 'a'").is_ok());
}

#[test]
fn bundled_javascript_accepts_all_curated_examples() {
    let grammar = Grammar::from_glrm(JAVASCRIPT).expect("bundled JavaScript must import publicly");
    let vocab = bytes_vocab();
    let constraint = grammar.compile_with(
        &vocab, CompileOptions::default().mask_strategy(MaskStrategy::FAST_MASKS)
    ).expect("static JavaScript compilation");
    let examples = [
        ("01_lexical_literals.js", include_bytes!("data/javascript/01_lexical_literals.js").as_slice()),
        ("02_identifier_keyword_boundaries.js", include_bytes!("data/javascript/02_identifier_keyword_boundaries.js").as_slice()),
        ("03_declarations_and_destructuring.js", include_bytes!("data/javascript/03_declarations_and_destructuring.js").as_slice()),
        ("04_arrays_objects_and_properties.js", include_bytes!("data/javascript/04_arrays_objects_and_properties.js").as_slice()),
        ("05_functions_parameters_and_yield.js", include_bytes!("data/javascript/05_functions_parameters_and_yield.js").as_slice()),
        ("06_arrows_and_assignments.js", include_bytes!("data/javascript/06_arrows_and_assignments.js").as_slice()),
        ("07_operators_and_precedence.js", include_bytes!("data/javascript/07_operators_and_precedence.js").as_slice()),
        ("08_members_calls_optional_and_new.js", include_bytes!("data/javascript/08_members_calls_optional_and_new.js").as_slice()),
        ("09_control_flow_and_loops.js", include_bytes!("data/javascript/09_control_flow_and_loops.js").as_slice()),
        ("10_switch_try_labels_and_jumps.js", include_bytes!("data/javascript/10_switch_try_labels_and_jumps.js").as_slice()),
        ("11_classes_fields_private_and_inheritance.js", include_bytes!("data/javascript/11_classes_fields_private_and_inheritance.js").as_slice()),
        ("12_modules_imports_exports.js", include_bytes!("data/javascript/12_modules_imports_exports.js").as_slice()),
        ("13_templates_regex_and_tagged_templates.js", include_bytes!("data/javascript/13_templates_regex_and_tagged_templates.js").as_slice()),
        ("14_unary_calls_and_spread.js", include_bytes!("data/javascript/14_unary_calls_and_spread.js").as_slice()),
        ("15_comma_conditional_and_nullish.js", include_bytes!("data/javascript/15_comma_conditional_and_nullish.js").as_slice()),
        ("16_dangling_else_and_nested_prefixes.js", include_bytes!("data/javascript/16_dangling_else_and_nested_prefixes.js").as_slice()),
        ("17_function_expressions_and_async.js", include_bytes!("data/javascript/17_function_expressions_and_async.js").as_slice()),
        ("18_wide_repetition_lists.js", include_bytes!("data/javascript/18_wide_repetition_lists.js").as_slice()),
        ("19_deep_nesting.js", include_bytes!("data/javascript/19_deep_nesting.js").as_slice()),
        ("20_long_lexemes_and_chains.js", include_bytes!("data/javascript/20_long_lexemes_and_chains.js").as_slice()),
        ("21_empty_and_minimal_statements.js", include_bytes!("data/javascript/21_empty_and_minimal_statements.js").as_slice()),
        ("22_comprehensive_integration.js", include_bytes!("data/javascript/22_comprehensive_integration.js").as_slice()),
        ("23_template_substitution_and_tagged.js", include_bytes!("data/javascript/23_template_substitution_and_tagged.js").as_slice()),
        ("24_comments_and_formatted_whitespace.js", include_bytes!("data/javascript/24_comments_and_formatted_whitespace.js").as_slice()),
        ("25_automatic_semicolon_insertion.js", include_bytes!("data/javascript/25_automatic_semicolon_insertion.js").as_slice()),
        ("26_class_expression.js", include_bytes!("data/javascript/26_class_expression.js").as_slice()),
        ("27_for_in_of_and_relational_keywords.js", include_bytes!("data/javascript/27_for_in_of_and_relational_keywords.js").as_slice()),
        ("28_class_fields_private_and_static.js", include_bytes!("data/javascript/28_class_fields_private_and_static.js").as_slice()),
        ("29_export_default.js", include_bytes!("data/javascript/29_export_default.js").as_slice()),
        ("30_formatted_whitespace_before_numeric.js", include_bytes!("data/javascript/30_formatted_whitespace_before_numeric.js").as_slice()),
        ("31_final_boss.js", include_bytes!("data/javascript/31_final_boss.js").as_slice()),
    ];
    assert_eq!(examples.len(), 31);
    for (name, program) in examples {
        let mut state = constraint.start();
        let mut mask = vec![0; constraint.mask_bit_len().div_ceil(32)];
        for (offset, &byte) in program.iter().enumerate() {
            state.fill_mask(&mut mask);
            let id = byte as u32;
            assert_ne!(mask[(id / 32) as usize] & (1 << (id % 32)), 0,
                "{name}, byte {offset} ({byte:#x}): masked out");
            assert!(state.commit_token(id), "{name}, byte {offset}: commit rejected");
        }
        assert!(state.is_accepting(), "{name}: final state not accepting");
    }
}

#[test]
fn documented_glrm_blocks_import() {
    let docs = [
        ("GUIDE.md", include_str!("../GUIDE.md")),
        ("docs/content/reference.md", include_str!("../docs/content/reference.md")),
        ("docs/glrm-format.md", include_str!("../docs/glrm-format.md")),
    ];
    let mut count = 0;
    for (name, doc) in docs {
        let doc = doc.replace("\r\n", "\n");
        for part in doc.split("\n\x60\x60\x60glrm\n").skip(1) {
            let (source, _) = part.split_once("\n\x60\x60\x60").expect("closed fence");
            Grammar::from_glrm(source).unwrap_or_else(|e| panic!("{name}: {e}\n{source}"));
            count += 1;
        }
    }
    assert_eq!(count, 7);
}

#[test]
fn shipped_example_sources_import() {
    // Extract actual source literals rather than maintaining duplicate grammars.
    let sources = [
        ("examples/generate.rs", include_str!("../examples/generate.rs"), "from_glrm(r#\"", "\"#"),
        ("examples/tool_calls.rs", include_str!("../examples/tool_calls.rs"), "from_glrm(r#\"", "\"#"),
        ("bindings/cpp/examples/generate.cpp", include_str!("../bindings/cpp/examples/generate.cpp"), "from_glrm(R\"glrm(", ")glrm\""),
        ("bindings/node/examples/generate.mjs", include_str!("../bindings/node/examples/generate.mjs"), "fromGlrm(\x60", "\x60"),
        ("bindings/node/examples/tool_calls.cjs", include_str!("../bindings/node/examples/tool_calls.cjs"), "fromGlrm(\x60", "\x60"),
        ("python/examples_module_expr.py", include_str!("../python/examples_module_expr.py"), "from_glrm(\n    '", "'"),
        ("GUIDE.md arithmetic", include_str!("../GUIDE.md"), "from_glrm(r'''", "'''"),
        ("GUIDE.md call", include_str!("../GUIDE.md"), "from_glrm('''", "'''"),
        ("reference arithmetic", include_str!("../docs/content/reference.md"), "from_glrm(r'''", "'''"),
        ("reference call", include_str!("../docs/content/reference.md"), "from_glrm('''", "'''"),
        ("README.md", include_str!("../README.md"), "from_glrm('''", "'''"),
        ("usage details", include_str!("../docs/content/usage-details.md"), "from_glrm('''", "'''"),
        ("token sets", include_str!("../docs/token-sets.md"), "from_glrm('''", "'''"),
        ("C++ README", include_str!("../bindings/cpp/README.md"), "from_glrm(R\"(", ")\""),
        ("C++ details", include_str!("../docs/content/cpp-details.md"), "from_glrm(R\"(", ")\""),
        ("Node README", include_str!("../bindings/node/README.md"), "fromGlrm(\x60", "\x60"),
        ("Node details", include_str!("../docs/content/javascript-details.md"), "fromGlrm(\x60", "\x60"),
        ("template parser", include_str!("../docs/design/template-parser.md"), "from_glrm(r#\"", "\"#"),
        ("template composition", include_str!("../docs/design/template-parser-composition.md"), "r#\"glrm 1;", "\"#"),
    ];
    let mut count = 0;
    for (name, code, open, close) in sources {
        let code = code.replace("\r\n", "\n");
        let mut found = 0;
        for part in code.split(open).skip(1) {
            let (literal, _) = part.split_once(close).expect("closed example literal");
            let source = if open.ends_with("glrm 1;") { format!("glrm 1;{literal}") }
                else { literal.to_owned() };
            Grammar::from_glrm(&source).unwrap_or_else(|e| panic!("{name}: {e}\n{source}"));
            found += 1;
            count += 1;
        }
        assert!(found > 0, "{name}: example literal not found");
    }
    assert_eq!(count, 20);
}

#[test]
fn developer_example_literal_grammars_import() {
    let sources = [
        ("composition_dynamic_control", include_str!("../examples/dev/composition_dynamic_control.rs"), 8),
        ("composition_semantic_probe", include_str!("../examples/dev/composition_semantic_probe.rs"), 8),
        ("composition_static_probe", include_str!("../examples/dev/composition_static_probe.rs"), 6),
        ("composition_tbm_probe", include_str!("../examples/dev/composition_tbm_probe.rs"), 4),
        ("composition_virtual_control", include_str!("../examples/dev/composition_virtual_control.rs"), 2),
        ("core_batch_bench", include_str!("../examples/dev/core_batch_bench.rs"), 6),
        ("core_batch_scaling", include_str!("../examples/dev/core_batch_scaling.rs"), 3),
        ("exact_token_range_bench", include_str!("../examples/dev/exact_token_range_bench.rs"), 3),
        ("template_composition", include_str!("../examples/dev/template_composition.rs"), 1),
    ];
    let strings = regex::Regex::new(r##"(?s)r#"(.*?)"#|("(?:\\.|[^"\\])*")"##).unwrap();
    let start = regex::Regex::new(r"^\s*(glrm\s+1;|start\s+\w+|extern\s+(token|grammar))").unwrap();
    let rule = regex::Regex::new(r"\bnt\s+\w+\s*=").unwrap();
    for (name, code, expected) in sources {
        let mut count = 0;
        for caps in strings.captures_iter(code) {
            let source = if let Some(raw) = caps.get(1) { raw.as_str().to_owned() }
                else if let Ok(value) = serde_json::from_str::<String>(caps.get(2).unwrap().as_str()) { value }
                else { continue };
            if !start.is_match(&source) || !rule.is_match(&source) { continue; }
            count += 1;
            if name == "core_batch_bench" && (source.contains("extern CHILD;") || source.contains("extern MARK;")) {
                // These are unreachable historical comparison branches.
                assert!(Grammar::from_glrm(&source).is_err());
            } else if name == "core_batch_scaling" && source == "start root nt root=E0" {
                // A fragment used by the generator after emitting the externs.
                Grammar::from_glrm(format!("extern token E0\n{source}")).unwrap();
            } else {
                Grammar::from_glrm(&source).unwrap_or_else(|e| panic!("{name}: {e}\n{source}"));
            }
        }
        assert_eq!(count, expected, "{name}: example inventory changed");
    }
}
