//! Differential harness for mixed retained/fresh compiles.
//!
//! A `TokenTrie` request over an assembly that has retained work keeps those
//! parts and compiles only the remaining modules. This ignored suite compares
//! every route against an all-`ParserDwa` compile and records whether any
//! fallback was used.
#![allow(clippy::needless_range_loop)]
use crate::test_modes::MaskMode;

use super::*;
use crate::__private::VocabExt as _;

const PARSER_DWA: MaskMode = MaskMode::ParserDwa;
const TOKEN_TRIE: MaskMode = MaskMode::A;

fn vocab() -> Vocab {
    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"x".to_vec()),
        (3, b"y".to_vec()),
        (4, b"c".to_vec()),
        (5, b"p".to_vec()),
        (6, b"q".to_vec()),
        (7, b"ab".to_vec()),
        (8, b"xy".to_vec()),
        (9, b"xay".to_vec()),
        (10, b"xby".to_vec()),
        (11, b"\"".to_vec()),
        (12, b"\"a\"".to_vec()),
        (13, b"\"ab\"".to_vec()),
    ]);
    vocab.prepare_for_compile();
    vocab
}

#[derive(Clone, Copy)]
enum Kind {
    Glrm(&'static str),
    Ebnf(&'static str),
    Lark(&'static str),
    JsonSchema(&'static str),
}

impl Kind {
    fn module(self) -> Module {
        match self {
            Kind::Glrm(source) => Module::new(&Grammar::from_glrm(source).unwrap()),
            Kind::Ebnf(source) => Module::new(&Grammar::from_ebnf(source).unwrap()),
            Kind::Lark(source) => Module::new(&Grammar::from_lark(source).unwrap()),
            Kind::JsonSchema(source) => Module::new(&Grammar::from_json_schema(source).unwrap()),
        }
    }
}

/// One module node. Slot and token children must have a higher index than
/// their parent, so the assembly can be built from the leaves up.
struct NodeSpec {
    kind: Kind,
    slots: &'static [(&'static str, usize)],
    tokens: &'static [(&'static str, u32)],
}

struct AssemblySpec {
    name: &'static str,
    nodes: Vec<NodeSpec>,
    root: usize,
    /// `(label, node indices precompiled with ParserDwa)`.
    subsets: Vec<(&'static str, Vec<usize>)>,
}

const fn node(kind: Kind) -> NodeSpec {
    NodeSpec { kind, slots: &[], tokens: &[] }
}

const HOST: &str = r#"glrm 1; start root; extern grammar child; nt root = "x" child "y";"#;
const MIDDLE: &str = r#"glrm 1; start root; extern grammar leaf; nt root = "p" leaf "q";"#;
const TWO_EXTERNS: &str = r#"glrm 1; start root; extern grammar LEFT; extern grammar RIGHT; nt root = "x" LEFT "c" RIGHT "y";"#;
const TOKEN_HOST: &str = r#"glrm 1; start root; extern token MARK; nt root = "x" MARK "y";"#;

fn assemblies() -> Vec<AssemblySpec> {
    vec![
        AssemblySpec {
            name: "glrm_parent_ebnf_child",
            nodes: vec![
                NodeSpec {
                    kind: Kind::Glrm(HOST),
                    slots: &[("child", 1)],
                    tokens: &[],
                },
                node(Kind::Ebnf(r#"start ::= "a" "b""#)),
            ],
            root: 0,
            subsets: vec![
                ("none", vec![]),
                ("child", vec![1]),
                ("parent", vec![0]),
                ("all", vec![0, 1]),
            ],
        },
        AssemblySpec {
            name: "depth2_glrm_glrm_ebnf",
            nodes: vec![
                NodeSpec {
                    kind: Kind::Glrm(HOST),
                    slots: &[("child", 1)],
                    tokens: &[],
                },
                NodeSpec {
                    kind: Kind::Glrm(MIDDLE),
                    slots: &[("leaf", 2)],
                    tokens: &[],
                },
                node(Kind::Ebnf(r#"start ::= "a" | "b""#)),
            ],
            root: 0,
            subsets: vec![
                ("none", vec![]),
                ("child", vec![1, 2]),
                ("parent", vec![0]),
                ("all", vec![0, 1, 2]),
            ],
        },
        AssemblySpec {
            name: "two_externs_reused_child",
            nodes: vec![
                NodeSpec {
                    kind: Kind::Glrm(TWO_EXTERNS),
                    slots: &[("LEFT", 1), ("RIGHT", 1)],
                    tokens: &[],
                },
                node(Kind::Ebnf(r#"start ::= "a" | "b""#)),
            ],
            root: 0,
            subsets: vec![
                ("none", vec![]),
                ("child", vec![1]),
                ("parent", vec![0]),
                ("all", vec![0, 1]),
            ],
        },
        AssemblySpec {
            name: "token_binding",
            nodes: vec![NodeSpec {
                kind: Kind::Glrm(TOKEN_HOST),
                slots: &[],
                tokens: &[("MARK", 0)],
            }],
            root: 0,
            subsets: vec![("none", vec![]), ("parent", vec![0])],
        },
        AssemblySpec {
            name: "json_schema_child",
            nodes: vec![
                NodeSpec {
                    kind: Kind::Glrm(HOST),
                    slots: &[("child", 1)],
                    tokens: &[],
                },
                node(Kind::JsonSchema(r#"{"type":"string","maxLength":2}"#)),
            ],
            root: 0,
            subsets: vec![
                ("none", vec![]),
                ("child", vec![1]),
                ("parent", vec![0]),
                ("all", vec![0, 1]),
            ],
        },
        AssemblySpec {
            name: "glrm_parent_lark_child",
            nodes: vec![
                NodeSpec {
                    kind: Kind::Glrm(HOST),
                    slots: &[("child", 1)],
                    tokens: &[],
                },
                node(Kind::Lark("start: \"a\" \"b\"")),
            ],
            root: 0,
            subsets: vec![
                ("none", vec![]),
                ("child", vec![1]),
                ("parent", vec![0]),
                ("all", vec![0, 1]),
            ],
        },
    ]
}

/// One precompile choice for a part in the engine matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Precompile {
    None,
    ParserDwa,
    Merged,
    MergedLr,
    A,
    B,
}

impl Precompile {
    const ALL: [Self; 6] = [
        Self::None,
        Self::ParserDwa,
        Self::Merged,
        Self::MergedLr,
        Self::A,
        Self::B,
    ];

    fn options(self) -> Option<crate::CompileOptions> {
        match self {
            Self::None => None,
            Self::ParserDwa => Some(PARSER_DWA.compile_options()),
            Self::Merged => Some(MaskMode::C.compile_options()),
            Self::MergedLr => Some(MaskMode::D.compile_options()),
            Self::A => Some(MaskMode::A.compile_options()),
            Self::B => Some(MaskMode::B.compile_options()),
        }
    }

    fn retains_key(self) -> Option<u8> {
        match self {
            Self::None | Self::MergedLr | Self::A => None,
            Self::ParserDwa => Some(Engine::ParserDwa.key()),
            Self::Merged => Some(Engine::MergedTokenTrie.key()),
            Self::B => Some(Engine::TokenTrie.key()),
        }
    }
}

fn compile_assembly(spec: &AssemblySpec, precompiled: &[usize], vocab: &Vocab) -> Module {
    let modes = (0..spec.nodes.len())
        .map(|index| {
            if precompiled.contains(&index) {
                Precompile::ParserDwa
            } else {
                Precompile::None
            }
        })
        .collect::<Vec<_>>();
    build_assembly(spec, &modes, vocab).0
}

/// Build the assembly leaves-up and return the root plus every built part, so
/// the differential can inspect retained keys and build counts.
fn build_assembly(
    spec: &AssemblySpec,
    modes: &[Precompile],
    vocab: &Vocab,
) -> (Module, Vec<Module>) {
    assert_eq!(modes.len(), spec.nodes.len());
    let mut built: Vec<Option<Module>> = (0..spec.nodes.len()).map(|_| None).collect();
    for index in (0..spec.nodes.len()).rev() {
        let node = &spec.nodes[index];
        let mut module = node.kind.module();
        if let Some(options) = modes[index].options() {
            module.precompile_with(vocab, options).unwrap_or_else(|error| {
                panic!("{}: precompile node {index} as {:?}: {error}", spec.name, modes[index])
            });
        }
        for &(slot, child) in node.slots {
            let child = built[child]
                .clone()
                .unwrap_or_else(|| panic!("{}: child {child} is not built", spec.name));
            module
                .bind(slot, child)
                .unwrap_or_else(|error| panic!("{}: bind {slot:?}: {error}", spec.name));
        }
        for &(slot, id) in node.tokens {
            module
                .bind(slot, TokenSet::from(id))
                .unwrap_or_else(|error| panic!("{}: bind token {slot:?}: {error}", spec.name));
        }
        built[index] = Some(module);
    }
    let parts = built
        .iter()
        .map(|module| module.clone().unwrap_or_else(|| panic!("{}: part missing", spec.name)))
        .collect();
    (built[spec.root].clone().unwrap(), parts)
}

fn precompile_assignments(count: usize) -> Vec<Vec<Precompile>> {
    let mut assignments = vec![Vec::new()];
    for _ in 0..count {
        assignments = assignments
            .into_iter()
            .flat_map(|prefix| {
                Precompile::ALL.into_iter().map(move |mode| {
                    let mut next = prefix.clone();
                    next.push(mode);
                    next
                })
            })
            .collect();
    }
    assignments
}

fn compile_reference(spec: &AssemblySpec, strategy: MaskMode, vocab: &Vocab) -> RuntimeConstraint {
    compile_assembly(spec, &[], vocab)
        .compile_with(vocab, strategy.compile_options())
        .unwrap_or_else(|error| panic!("{}: {strategy:?} reference compile: {error}", spec.name))
}

fn allowed_tokens(mask: &[u32]) -> Vec<u32> {
    let mut tokens = Vec::new();
    for (word_index, &word) in mask.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let bit = bits.trailing_zeros();
            tokens.push((word_index * 32 + bit as usize) as u32);
            bits &= bits - 1;
        }
    }
    tokens
}

/// Exhaustive walk to `depth`, comparing the mask and acceptance of every
/// reachable state. Returns the number of visited states.
fn compare_exhaustive(
    reference: &crate::ConstraintState<'_>,
    candidate: &crate::ConstraintState<'_>,
    depth: usize,
    label: &str,
) -> usize {
    let reference_mask = reference.mask();
    let candidate_mask = candidate.mask();
    assert_eq!(
        reference_mask, candidate_mask,
        "{label}: mask mismatch at depth {depth}",
    );
    assert_eq!(
        reference.is_accepting(),
        candidate.is_accepting(),
        "{label}: acceptance mismatch at depth {depth}",
    );
    if depth == 0 {
        return 1;
    }
    let mut visited = 1;
    for token in allowed_tokens(&reference_mask) {
        let mut next_reference = reference.clone();
        assert!(next_reference
            .commit_token(token));
        let mut next_candidate = candidate.clone();
        assert!(next_candidate
            .commit_token(token));
        visited += compare_exhaustive(&next_reference, &next_candidate, depth - 1, label);
    }
    visited
}

fn compare_against_parser_dwa_at(
    reference: &RuntimeConstraint,
    candidate: &RuntimeConstraint,
    label: &str,
    depth: usize,
) -> usize {
    compare_exhaustive(&reference.start(), &candidate.start(), depth, label)
}

fn compare_against_parser_dwa(
    reference: &RuntimeConstraint,
    candidate: &RuntimeConstraint,
    label: &str,
) -> usize {
    compare_against_parser_dwa_at(reference, candidate, label, 3)
}

#[test]
#[ignore = "differential mixed static/dynamic compile validation; run explicitly"]
fn mixed_compile_matches_parser_dwa_on_composed_assemblies() {
    let vocab = vocab();
    let mut fallbacks = Vec::<(String, String)>::new();
    let mut comparisons = 0usize;

    for spec in assemblies() {
        let reference = compile_reference(&spec, PARSER_DWA, &vocab);
        for (subset_label, precompiled) in &spec.subsets {
            // Clear anything left on this thread by an earlier compile.
            let _ = crate::public_api::module::mixed_compile_probe::take();
            let module = compile_assembly(&spec, precompiled, &vocab);
            let candidate = module
                .compile_with(&vocab, TOKEN_TRIE.compile_options())
                .unwrap_or_else(|error| {
                    panic!(
                        "{} subset {subset_label}: TokenTrie compile failed: {error}",
                        spec.name,
                    )
                });
            let fallback_errors =
                crate::public_api::module::mixed_compile_probe::take();
            let route = if precompiled.is_empty() {
                "pure-token-trie"
            } else if fallback_errors.is_empty() {
                "mixed"
            } else {
                "fallback"
            };
            for error in &fallback_errors {
                fallbacks.push((format!("{} subset {subset_label}", spec.name), error.clone()));
            }
            let label = format!(
                "{} subset {subset_label} route {route}",
                spec.name,
            );
            let visited = compare_against_parser_dwa(&reference, &candidate, &label);
            comparisons += 1;
            eprintln!(
                "[mixed-validation] assembly={} subset={subset_label} precompiled={precompiled:?} \
                 route={route} states={visited} fallback={fallback_errors:?}",
                spec.name,
            );
        }
    }

    eprintln!(
        "[mixed-validation] summary: {} assemblies, {} compiles compared, {} fallbacks",
        assemblies().len(),
        comparisons,
        fallbacks.len(),
    );
    for (label, error) in &fallbacks {
        eprintln!("[mixed-validation] fallback {label}: {error}");
    }
    assert!(
        fallbacks.is_empty(),
        "mixed compile fell back for {} subset(s): {fallbacks:?}",
        fallbacks.len(),
    );
}

fn engine_for_key(key: u8) -> Option<Engine> {
    match key {
        1 => Some(Engine::ParserDwa),
        2 => Some(Engine::TokenTrie),
        3 => Some(Engine::MergedTokenTrie),
        _ => None,
    }
}

#[test]
#[ignore = "engine/precompile differential matrix; run explicitly"]
fn precompiled_work_is_reused_across_every_engine_request() {
    let vocab = vocab();
    let compile_modes = [
        MaskMode::Auto,
        MaskMode::ParserDwa,
        MaskMode::A,
        MaskMode::B,
        MaskMode::C,
        MaskMode::D,
    ];
    let mut compiles = 0usize;
    let mut retained_parts = 0usize;

    for spec in assemblies() {
        let reference = compile_reference(&spec, PARSER_DWA, &vocab);
        for assignment in precompile_assignments(spec.nodes.len()) {
            let (module, parts) = build_assembly(&spec, &assignment, &vocab);
            for (index, mode) in assignment.iter().enumerate() {
                match mode.retains_key() {
                    Some(expected) => {
                        assert!(
                            parts[index].node.prepared.contains_key(&expected),
                            "{} node {index}: {:?} did not retain key {expected}",
                            spec.name,
                            mode,
                        );
                        retained_parts += 1;
                    }
                    None => assert!(
                        parts[index].node.prepared.is_empty(),
                        "{} node {index}: {:?} must retain nothing, kept {:?}",
                        spec.name,
                        mode,
                        parts[index].node.prepared.keys().collect::<Vec<_>>(),
                    ),
                }
            }
            let mut retained = Vec::new();
            for part in &parts {
                for &key in part.node.prepared.keys() {
                    if let Some(engine) = engine_for_key(key) {
                        retained.push((part, engine, static_build_probe::builds_for(part, engine)));
                    }
                }
            }

            for mode in compile_modes {
                let candidate = module
                    .compile_with(&vocab, mode.compile_options())
                    .unwrap_or_else(|error| {
                        panic!(
                            "{} assignment {assignment:?} mode {mode:?}: compile failed: {error}",
                            spec.name,
                        )
                    });
                for (part, engine, before) in &retained {
                    assert_eq!(
                        static_build_probe::builds_for(part, *engine),
                        *before,
                        "{} assignment {assignment:?} mode {mode:?}: retained {engine:?} was rebuilt",
                        spec.name,
                    );
                }
                let label = format!("{} assignment {assignment:?} mode {mode:?}", spec.name);
                compare_against_parser_dwa(&reference, &candidate, &label);
                compiles += 1;
            }
        }
    }
    eprintln!(
        "[engine-matrix] assemblies={} compiles={compiles} retained_parts={retained_parts}",
        assemblies().len(),
    );
}

/// A retained GLRM parent plus a non-precompiled JSON Schema child must take
/// the mixed route: the child gets a dynamic component shell without retrying
/// the whole assembly as all-static.
#[test]
fn json_schema_child_mixed_compile_has_no_fallback() {
    let vocab = vocab();
    let spec = AssemblySpec {
        name: "json_schema_child_unit",
        nodes: vec![
            NodeSpec {
                kind: Kind::Glrm(HOST),
                slots: &[("child", 1)],
                tokens: &[],
            },
            node(Kind::JsonSchema(r#"{"type":"string","maxLength":2}"#)),
        ],
        root: 0,
        subsets: vec![("parent", vec![0])],
    };
    let reference = compile_reference(&spec, PARSER_DWA, &vocab);

    let _ = crate::public_api::module::mixed_compile_probe::take();
    let candidate = compile_assembly(&spec, &[0], &vocab)
        .compile_with(&vocab, TOKEN_TRIE.compile_options())
        .unwrap_or_else(|error| panic!("{}: mixed TokenTrie compile failed: {error}", spec.name));
    let fallbacks = crate::public_api::module::mixed_compile_probe::take();
    assert!(
        fallbacks.is_empty(),
        "{}: mixed compile fell back: {fallbacks:?}",
        spec.name,
    );

    compare_against_parser_dwa(&reference, &candidate, spec.name);
}

fn json_vocab() -> Vocab {
    let entries = [
        "{", "}", "[", "]", ",", ":", "\"", "\"a\"", "\"b\"", "\"ab\"", "\"abc\"",
        "\"key\"", "a", "b", "0", "1", "12", "-1", "true", "false", "null", " ", "x", "y",
    ]
    .iter()
    .enumerate()
    .map(|(id, bytes)| (id as u32, bytes.as_bytes().to_vec()))
    .collect::<Vec<_>>();
    let vocab = Vocab::new(entries);
    vocab.prepare_for_compile();
    vocab
}

/// The JSON Schema component shell must recognize the same token language as
/// the static schema compiler it replaces; a retained parent plus this child
/// must stay on the mixed route for every shape below.
#[test]
#[ignore = "differential JSON Schema component language validation; run explicitly"]
fn json_schema_component_matches_static_schema_compiler() {
    let vocab = json_vocab();
    let schemas: [(&str, &str); 8] = [
        ("string_max_length", r#"{"type":"string","maxLength":2}"#),
        ("string_pattern", r#"{"type":"string","pattern":"^[ab]+$"}"#),
        ("integer", r#"{"type":"integer"}"#),
        ("number", r#"{"type":"number"}"#),
        ("enum", r#"{"enum":["a","b",1]}"#),
        (
            "object_required",
            r#"{"type":"object","properties":{"key":{"type":"integer"}},"required":["key"]}"#,
        ),
        (
            "array_items",
            r#"{"type":"array","items":{"type":"boolean"},"maxItems":2}"#,
        ),
        (
            "nested_object",
            r#"{"type":"object","properties":{"a":{"type":"array","items":{"type":"string","maxLength":2}},"b":{"type":"integer"}},"required":["a","b"]}"#,
        ),
    ];

    for (name, schema) in schemas {
        let spec = AssemblySpec {
            name,
            nodes: vec![
                NodeSpec {
                    kind: Kind::Glrm(HOST),
                    slots: &[("child", 1)],
                    tokens: &[],
                },
                node(Kind::JsonSchema(schema)),
            ],
            root: 0,
            subsets: vec![("parent", vec![0])],
        };
        let reference = compile_reference(&spec, PARSER_DWA, &vocab);
        let _ = crate::public_api::module::mixed_compile_probe::take();
        let candidate = compile_assembly(&spec, &[0], &vocab)
            .compile_with(&vocab, TOKEN_TRIE.compile_options())
            .unwrap_or_else(|error| panic!("{name}: mixed TokenTrie compile failed: {error}"));
        let fallbacks = crate::public_api::module::mixed_compile_probe::take();
        assert!(
            fallbacks.is_empty(),
            "{name}: mixed compile fell back: {fallbacks:?}",
        );
        let visited = compare_against_parser_dwa_at(&reference, &candidate, name, 5);
        eprintln!("[json-language] schema={name} states={visited} fallback=[]");
    }
}
