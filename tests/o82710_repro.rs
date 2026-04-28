use glrmask::{Constraint, Vocab, compile_grammar_def_json, dump_json_schema_prepared_grammar_def};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn token_allowed(mask: &[u32], id: usize) -> bool {
    let word = id / 32;
    if word >= mask.len() {
        return false;
    }
    (mask[word] >> (id % 32)) & 1 != 0
}

fn make_vocab(entries: &[&[u8]]) -> Vocab {
    let entries: Vec<(u32, Vec<u8>)> = entries
        .iter()
        .enumerate()
        .map(|(i, entry)| (i as u32, entry.to_vec()))
        .collect();
    Vocab::new(entries, None)
}

fn grammar_mismatch_predicate(grammar_json: &str, vocab: &Vocab, prefix: &[u8], token_id: u32) -> bool {
    let constraint = match compile_grammar_def_json(grammar_json, vocab) {
        Ok(value) => value,
        Err(_) => return false,
    };

    let mut mask_state = constraint.start();
    if mask_state.commit_bytes(prefix).is_err() {
        return false;
    }
    let mask_accepts = token_allowed(&mask_state.mask(), token_id as usize);

    let mut commit_state = constraint.start();
    if commit_state.commit_bytes(prefix).is_err() {
        return false;
    }
    let commit_accepts = match catch_unwind(AssertUnwindSafe(|| commit_state.commit_token(token_id))) {
        Ok(Ok(())) => true,
        Ok(Err(_)) => false,
        Err(_) => true,
    };

    !mask_accepts && commit_accepts
}

fn subset_object_schema(include_aside: bool, include_autoplay: bool, include_css_class: bool) -> String {
    let mut properties = Vec::new();
    if include_aside {
        properties.push(r#""aside": { "type": "boolean" }"#);
    }
    if include_autoplay {
        properties.push(r#""autoplay": { "type": "boolean" }"#);
    }
    if include_css_class {
        properties.push(
            r#""css_class": {
                    "type": "string",
                    "pattern": "^[\\w\\s-]+$"
                }"#,
        );
    }
    properties.push(
        r#""description": {
                    "type": "string",
                    "minLength": 0,
                    "maxLength": 5000
                }"#,
    );

    format!(
        r##"
        {{
            "type": "object",
            "properties": {{
                {}
            }},
            "required": ["id"],
            "additionalProperties": true
        }}
        "##,
        properties.join(",\n                ")
    )
}

fn subset_object_prefix(
    include_aside: bool,
    include_autoplay: bool,
    include_css_class: bool,
    repeats: usize,
    tail: &[u8],
) -> Vec<u8> {
    let mut prefix = Vec::from(b"{".as_slice());
    let mut first = true;

    if include_aside {
        prefix.extend_from_slice(if first {
            b"\"aside\": true"
        } else {
            b", \"aside\": true"
        });
        first = false;
    }
    if include_autoplay {
        prefix.extend_from_slice(if first {
            b"\"autoplay\": false"
        } else {
            b", \"autoplay\": false"
        });
        first = false;
    }
    if include_css_class {
        prefix.extend_from_slice(if first {
            b"\"css_class\": \"vimeo-video-block\""
        } else {
            b", \"css_class\": \"vimeo-video-block\""
        });
        first = false;
    }
    prefix.extend_from_slice(if first {
        b"\"description\": \""
    } else {
        b", \"description\": \""
    });
    prefix.extend(std::iter::repeat(b"This is a Vimeo video block. ".as_slice()).take(repeats).flatten().copied());
    prefix.extend_from_slice(tail);
    prefix
}

fn o82710_schema() -> &'static str {
    r##"
    {
      "maxLength": 5000,
      "minLength": 0,
      "type": "string"
    }
    "##
}

fn o82710_object_schema() -> &'static str {
        r##"
        {
            "type": "object",
            "properties": {
                "aside": { "type": "boolean" },
                "autoplay": { "type": "boolean" },
                "css_class": {
                    "type": "string",
                    "pattern": "^[\\w\\s-]+$"
                },
                "description": {
                    "type": "string",
                    "minLength": 0,
                    "maxLength": 5000
                }
            },
            "required": [],
            "additionalProperties": true
        }
        "##
}

fn o82710_step_580_prefix() -> Vec<u8> {
    let mut prefix = String::from("\"");
    prefix.push_str(&"This is a Vimeo video block. ".repeat(79));
    prefix.push_str("This is a");
    prefix.into_bytes()
}

fn o82710_object_step_580_prefix() -> Vec<u8> {
    let mut prefix = String::from(
        "{\"aside\": true, \"autoplay\": false, \"css_class\": \"vimeo-video-block\", \"description\": \"",
    );
    prefix.push_str(&"This is a Vimeo video block. ".repeat(79));
    prefix.push_str("This is a");
    prefix.into_bytes()
}

fn o82710_minimal_required_object_schema() -> &'static str {
    r##"
    {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "minLength": 0,
                "maxLength": 5000
            }
        },
        "required": ["id"],
        "additionalProperties": true
    }
    "##
}

fn o82710_minimal_required_object_prefix() -> Vec<u8> {
    let mut prefix = String::from("{\"description\": \"");
    prefix.push_str(&"This is a Vimeo video block. ".repeat(79));
    prefix.push_str("This is a");
    prefix.into_bytes()
}

#[test]
fn test_o82710_step_580_allows_disputed_token_in_small_vocab() {
    let vocab = make_vocab(&[b"'];?>\"", b" Vimeo"]);
    let constraint = Constraint::from_json_schema(o82710_schema(), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(&o82710_step_580_prefix()).unwrap();

    let mask = state.mask();
    assert!(
        token_allowed(&mask, 0),
        "expected disputed token b\"'];?>\\\"\" to be in mask"
    );
}

#[test]
fn test_o82710_step_580_allows_control_token_in_small_vocab() {
    let vocab = make_vocab(&[b"'];?>\"", b" Vimeo"]);
    let constraint = Constraint::from_json_schema(o82710_schema(), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(&o82710_step_580_prefix()).unwrap();

    let mask = state.mask();
    assert!(
        token_allowed(&mask, 1),
        "expected control token b\" Vimeo\" to be in mask"
    );
}

#[test]
fn test_o82710_step_580_commits_disputed_token_in_single_token_vocab() {
    let vocab = make_vocab(&[b"'];?>\""]);
    let constraint = Constraint::from_json_schema(o82710_schema(), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(&o82710_step_580_prefix()).unwrap();

    assert!(
        token_allowed(&state.mask(), 0),
        "expected disputed token to be in mask before commit"
    );
    state.commit_token(0).unwrap();
}

#[test]
fn test_o82710_object_step_580_allows_disputed_token_in_small_vocab() {
    let vocab = make_vocab(&[b"'];?>\"", b" Vimeo"]);
    let constraint = Constraint::from_json_schema(o82710_object_schema(), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(&o82710_object_step_580_prefix()).unwrap();

    let mask = state.mask();
    assert!(
        token_allowed(&mask, 0),
        "expected disputed token b\"'];?>\\\"\" to be in mask in object context"
    );
}

#[ignore = "known minimized native mismatch: mask rejects token that commit accepts"]
#[test]
fn test_o82710_minimal_required_object_mask_commit_mismatch() {
    let vocab = make_vocab(&[b"'];?>\""]);
    let constraint = Constraint::from_json_schema(o82710_minimal_required_object_schema(), &vocab).unwrap();
    let prefix = o82710_minimal_required_object_prefix();

    let mut mask_state = constraint.start();
    mask_state.commit_bytes(&prefix).unwrap();
    assert!(
        !token_allowed(&mask_state.mask(), 0),
        "expected minimized repro token to be absent from mask"
    );

    let mut commit_state = constraint.start();
    commit_state.commit_bytes(&prefix).unwrap();
    let commit_result = catch_unwind(AssertUnwindSafe(|| commit_state.commit_token(0)));
    match commit_result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => panic!("expected minimized repro token to commit, got {error:?}"),
        Err(_) => {}
    }
}

#[ignore = "known minimized prepared-grammar mismatch: mask rejects token that commit accepts"]
#[test]
fn test_o82710_minimal_required_object_prepared_grammar_mask_commit_mismatch() {
    let vocab = make_vocab(&[b"'];?>\""]);
    let grammar_json = dump_json_schema_prepared_grammar_def(o82710_minimal_required_object_schema()).unwrap();
    let constraint = compile_grammar_def_json(&grammar_json, &vocab).unwrap();
    let prefix = o82710_minimal_required_object_prefix();

    let mut mask_state = constraint.start();
    mask_state.commit_bytes(&prefix).unwrap();
    assert!(
        !token_allowed(&mask_state.mask(), 0),
        "expected minimized prepared-grammar repro token to be absent from mask"
    );

    let mut commit_state = constraint.start();
    commit_state.commit_bytes(&prefix).unwrap();
    let commit_result = catch_unwind(AssertUnwindSafe(|| commit_state.commit_token(0)));
    match commit_result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => panic!("expected minimized prepared-grammar repro token to commit, got {error:?}"),
        Err(_) => {}
    }
}

#[ignore = "known minimized direct-GLRM mismatch: mask rejects token that commit accepts"]
#[test]
fn test_o82710_minimal_required_object_inline_glrm_mask_commit_mismatch() {
    let grammar = r#"
start start;

internal t JSON_STRING_CHAR ::= /[^\x00-\x1f\x7f"\\]|\\["\\\/bfnrt]|\\u[0-9A-Fa-f]{4}/;
t JSON_STRING_BODY ::= JSON_STRING_CHAR* "\"";
nt json_string ::= "\"" JSON_STRING_BODY;
internal t JSON_STRING_CHAR_UPTO_256_0 ::= JSON_STRING_CHAR{0,256};
t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= JSON_STRING_CHAR_UPTO_256_0 "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= JSON_STRING_CHAR{256};
internal t JSON_STRING_CHAR_UPTO_136_3 ::= JSON_STRING_CHAR{0,136};
t JSON_STRING_CHAR_UPTO_CLOSE_4 ::= JSON_STRING_CHAR_UPTO_136_3 "\"";
nt json_string_bounded_split_5 ::= "\"" (JSON_STRING_CHAR_EXACT_256_2{0,18} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{19} JSON_STRING_CHAR_UPTO_CLOSE_4);
nt obj_open_reqmask_0_nc_0 ::= (("\"" "description\"" ": ") json_string_bounded_split_5) obj_open_reqmask_0_c_0 | (("\"" "id\"" ": ") json_string) obj_open_reqmask_0_c_1;
nt obj_open_reqmask_0_c_0 ::= ", " (("\"" "description\"" ": ") json_string_bounded_split_5) obj_open_reqmask_0_c_0 | ", " (("\"" "id\"" ": ") json_string) obj_open_reqmask_0_c_1;
nt obj_open_reqmask_0_c_1 ::= ", " (("\"" "description\"" ": ") json_string_bounded_split_5) obj_open_reqmask_0_c_1 | ;
nt start ::= "{" obj_open_reqmask_0_nc_0 "}";
"#;

    let vocab = make_vocab(&[b"'];?>\""]);
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

    let mut prefix = String::from("{\"description\": \"");
    prefix.push_str(&"This is a Vimeo video block. ".repeat(79));
    prefix.push_str("This is a");
    let prefix = prefix.into_bytes();

    let mut mask_state = constraint.start();
    mask_state.commit_bytes(&prefix).unwrap();
    assert!(
        !token_allowed(&mask_state.mask(), 0),
        "expected minimized inline-GLRM repro token to be absent from mask"
    );

    let mut commit_state = constraint.start();
    commit_state.commit_bytes(&prefix).unwrap();
    let commit_result = catch_unwind(AssertUnwindSafe(|| commit_state.commit_token(0)));
    match commit_result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => panic!("expected minimized inline-GLRM repro token to commit, got {error:?}"),
        Err(_) => {}
    }
}

#[ignore = "scanner for aggressively minimized native open-object mismatch"]
#[test]
fn scan_o82710_minimal_open_object_schema_single_token_vocab() {
    std::panic::set_hook(Box::new(|_| {}));

    let vocab = make_vocab(&[b"'];?>\""]);
    let mut found = None;

    for mask in 0u8..8 {
        let include_aside = (mask & 0b001) != 0;
        let include_autoplay = (mask & 0b010) != 0;
        let include_css_class = (mask & 0b100) != 0;
        let schema = subset_object_schema(include_aside, include_autoplay, include_css_class);
        let constraint = match Constraint::from_json_schema(&schema, &vocab) {
            Ok(value) => value,
            Err(_) => continue,
        };
        for repeats in 0..=79 {
            let prefix = subset_object_prefix(
                include_aside,
                include_autoplay,
                include_css_class,
                repeats,
                b"This is a",
            );
            let mut mask_state = constraint.start();
            if mask_state.commit_bytes(&prefix).is_err() {
                continue;
            }
            let mask_accepts = token_allowed(&mask_state.mask(), 0);

            let mut commit_state = constraint.start();
            if commit_state.commit_bytes(&prefix).is_err() {
                continue;
            }
            let commit_accepts = match catch_unwind(AssertUnwindSafe(|| commit_state.commit_token(0))) {
                Ok(Ok(())) => true,
                Ok(Err(_)) => false,
                Err(_) => true,
            };

            if !mask_accepts && commit_accepts {
                found = Some((
                    format!(
                        "aside={include_aside},autoplay={include_autoplay},css_class={include_css_class}"
                    ),
                    repeats,
                    prefix,
                ));
                break;
            }
        }
        if found.is_some() {
            break;
        }
    }

    let Some((label, repeats, prefix)) = found else {
        panic!("expected minimized open-object schema + single-token vocab to reproduce");
    };

    println!("minimal_open_object_schema_variant={label}");
    println!("minimal_open_object_repeat_count={repeats}");
    println!("minimal_open_object_prefix={:?}", String::from_utf8_lossy(&prefix));
}

#[ignore = "scanner for prepared-grammar form of minimized native mismatch"]
#[test]
fn scan_o82710_minimal_open_object_prepared_grammar() {
    std::panic::set_hook(Box::new(|_| {}));

    let vocab = make_vocab(&[b"'];?>\""]);
    let mut found = None;

    for mask in 0u8..8 {
        let include_aside = (mask & 0b001) != 0;
        let include_autoplay = (mask & 0b010) != 0;
        let include_css_class = (mask & 0b100) != 0;
        let schema = subset_object_schema(include_aside, include_autoplay, include_css_class);
        let grammar_json = match dump_json_schema_prepared_grammar_def(&schema) {
            Ok(value) => value,
            Err(_) => continue,
        };
        for repeats in 0..=79 {
            let prefix = subset_object_prefix(
                include_aside,
                include_autoplay,
                include_css_class,
                repeats,
                b"This is a",
            );
            if grammar_mismatch_predicate(&grammar_json, &vocab, &prefix, 0) {
                found = Some((repeats, prefix));
                break;
            }
        }
        if found.is_some() {
            break;
        }
    }

    let Some((repeats, prefix)) = found else {
        panic!("expected prepared grammar form to preserve minimized mismatch");
    };

    println!("minimal_prepared_grammar_repeat_count={repeats}");
    println!("minimal_prepared_grammar_prefix={:?}", String::from_utf8_lossy(&prefix));
}
