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

fn constraint_mismatch_predicate(constraint: &Constraint, prefix: &[u8], token_id: u32) -> bool {
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

fn direct_glrm_prefix_with_content(content: &[u8]) -> Vec<u8> {
    let mut prefix = b"{\"description\": \"".to_vec();
    prefix.extend_from_slice(content);
    prefix
}

fn current_inline_glrm() -> &'static str {
    r#"
start start;

t JSON_STRING_CHAR ::= /[^\x00-\x1f\x7f"\\]|\\["\\\/bfnrt]|\\u[0-9A-Fa-f]{4}/;
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
"#
}

fn fixed_chunk_object_glrm(variant: &str) -> String {
    format!(
        r#"
start start;

t C ::= /a/;
t BODY ::= C* "\"";
nt json_string ::= "\"" BODY;
internal t UPTO_256 ::= C{{0,256}};
t CLOSE_256 ::= UPTO_256 "\"";
t EXACT_256 ::= C{{256}};
nt bounded_fixed8 ::= "\"" EXACT_256{{8}} CLOSE_256;
nt bounded_fixed9 ::= "\"" EXACT_256{{9}} CLOSE_256;
nt bounded_alt_8_9 ::= "\"" (EXACT_256{{8}} CLOSE_256 | EXACT_256{{9}} CLOSE_256);
nt bounded_alt_9_8 ::= "\"" (EXACT_256{{9}} CLOSE_256 | EXACT_256{{8}} CLOSE_256);
nt start ::= "{{" (("\"" "description\"" ": ") {variant}) ", " (("\"" "id\"" ": ") json_string) "}}";
"#,
    )
}

fn explicit_8_9_object_glrm(reverse_order: bool) -> String {
    let bounded = if reverse_order {
        "(EXACT_256{9} CLOSE_256 | EXACT_256{8} CLOSE_256)"
    } else {
        "(EXACT_256{8} CLOSE_256 | EXACT_256{9} CLOSE_256)"
    };
    format!(
        r#"
start start;

t C ::= /a/;
t BODY ::= C* "\"";
nt json_string ::= "\"" BODY;
internal t UPTO_256 ::= C{{0,256}};
t CLOSE_256 ::= UPTO_256 "\"";
t EXACT_256 ::= C{{256}};
nt json_string_bounded_split_5 ::= "\"" {bounded};
nt obj_open_reqmask_0_nc_0 ::= (("\"" "description\"" ": ") json_string_bounded_split_5) obj_open_reqmask_0_c_0 | (("\"" "id\"" ": ") json_string) obj_open_reqmask_0_c_1;
nt obj_open_reqmask_0_c_0 ::= ", " (("\"" "description\"" ": ") json_string_bounded_split_5) obj_open_reqmask_0_c_0 | ", " (("\"" "id\"" ": ") json_string) obj_open_reqmask_0_c_1;
nt obj_open_reqmask_0_c_1 ::= ", " (("\"" "description\"" ": ") json_string_bounded_split_5) obj_open_reqmask_0_c_1 | ;
nt start ::= "{{" obj_open_reqmask_0_nc_0 "}}";
"#,
    )
}

fn counted_repeat_object_glrm_a_only() -> &'static str {
    r#"
start start;

t JSON_STRING_CHAR ::= /a/;
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
"#
}

fn bounded_string_only_glrm() -> &'static str {
    r#"
start start;

t C ::= /a/;
internal t UPTO_256 ::= C{0,256};
t CLOSE_256 ::= UPTO_256 "\"";
t EXACT_256 ::= C{256};
internal t UPTO_136 ::= C{0,136};
t CLOSE_136 ::= UPTO_136 "\"";
nt bounded ::= "\"" (EXACT_256{0,18} CLOSE_256 | EXACT_256{19} CLOSE_136);
nt start ::= bounded;
"#
}

fn object_close_glrm() -> &'static str {
    r#"
start start;

t C ::= /a/;
internal t UPTO_256 ::= C{0,256};
t CLOSE_256 ::= UPTO_256 "\"";
t EXACT_256 ::= C{256};
internal t UPTO_136 ::= C{0,136};
t CLOSE_136 ::= UPTO_136 "\"";
nt bounded ::= "\"" (EXACT_256{0,18} CLOSE_256 | EXACT_256{19} CLOSE_136);
nt start ::= "{" (("\"" "description\"" ": ") bounded) "}";
"#
}

fn object_required_id_nonrecursive_glrm() -> &'static str {
    r#"
start start;

t C ::= /a/;
t BODY ::= C* "\"";
nt json_string ::= "\"" BODY;
internal t UPTO_256 ::= C{0,256};
t CLOSE_256 ::= UPTO_256 "\"";
t EXACT_256 ::= C{256};
internal t UPTO_136 ::= C{0,136};
t CLOSE_136 ::= UPTO_136 "\"";
nt bounded ::= "\"" (EXACT_256{0,18} CLOSE_256 | EXACT_256{19} CLOSE_136);
nt start ::= "{" (("\"" "description\"" ": ") bounded) ", " (("\"" "id\"" ": ") json_string) "}";
"#
}

fn classify_constraint(
    constraint: &Constraint,
    prefix: &[u8],
    token: &[u8],
    token_id: u32,
    completion: Option<&[u8]>,
) -> (bool, bool, bool, bool) {
    let mut mask_state = constraint.start();
    let prefix_ok = mask_state.commit_bytes(prefix).is_ok();
    let mask_accepts = prefix_ok && token_allowed(&mask_state.mask(), token_id as usize);

    let mut commit_token_state = constraint.start();
    let commit_token_accepts = if commit_token_state.commit_bytes(prefix).is_ok() {
        match catch_unwind(AssertUnwindSafe(|| commit_token_state.commit_token(token_id))) {
            Ok(Ok(())) => true,
            Ok(Err(_)) => false,
            Err(_) => true,
        }
    } else {
        false
    };

    let mut commit_bytes_state = constraint.start();
    let commit_bytes_accepts = if commit_bytes_state.commit_bytes(prefix).is_ok() {
        commit_bytes_state.commit_bytes(token).is_ok()
    } else {
        false
    };

    let can_complete_after_token = if let Some(tail) = completion {
        let mut completion_state = constraint.start();
        if completion_state.commit_bytes(prefix).is_err() {
            false
        } else if completion_state.commit_bytes(token).is_err() {
            false
        } else {
            completion_state.commit_bytes(tail).is_ok()
        }
    } else {
        false
    };

    (
        mask_accepts,
        commit_token_accepts,
        commit_bytes_accepts,
        can_complete_after_token,
    )
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
    let vocab = make_vocab(&[b"'];?>\""]);
    let constraint = Constraint::from_glrm_grammar(current_inline_glrm(), &vocab).unwrap();

    let prefix = direct_glrm_prefix_with_content(&vec![b'a'; 2300]);

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

#[ignore = "scanner for smaller direct-GLRM prefix/content repros"]
#[test]
fn scan_o82710_minimal_required_object_inline_glrm_prefix() {
    std::panic::set_hook(Box::new(|_| {}));

    let vocab = make_vocab(&[b"'];?>\""]);
    let constraint = Constraint::from_glrm_grammar(current_inline_glrm(), &vocab).unwrap();
    let mut found = None;

    for len in 0..=2400 {
        let content = vec![b'a'; len];
        let prefix = direct_glrm_prefix_with_content(&content);
        if constraint_mismatch_predicate(&constraint, &prefix, 0) {
            found = Some(("all_a_bytes", len, prefix));
            break;
        }
    }

    if found.is_none() {
        let unit = b"This is a Vimeo video block. ";
        for repeats in 0..=79 {
            let mut content = std::iter::repeat(unit)
                .take(repeats)
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            content.extend_from_slice(b"This is a");
            let prefix = direct_glrm_prefix_with_content(&content);
            if constraint_mismatch_predicate(&constraint, &prefix, 0) {
                found = Some(("phrase_repeats", repeats, prefix));
                break;
            }
        }
    }

    let Some((label, size, prefix)) = found else {
        panic!("expected direct GLRM scanner to find a smaller reproducing prefix");
    };

    println!("direct_glrm_prefix_mode={label}");
    println!("direct_glrm_prefix_size={size}");
    println!("direct_glrm_prefix={:?}", String::from_utf8_lossy(&prefix));
}

#[ignore = "expert experiment: scan residue window around the 9th 256-byte boundary"]
#[test]
fn scan_o82710_inline_glrm_boundary_residues() {
    let vocab = make_vocab(&[b"'];?>\""]);
    let constraint = Constraint::from_glrm_grammar(current_inline_glrm(), &vocab).unwrap();
    let tail = b", \"id\": \"\"}";
    let token = b"'];?>\"";

    for len in 2296usize..=2312 {
        let prefix = direct_glrm_prefix_with_content(&vec![b'a'; len]);
        let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
            classify_constraint(&constraint, &prefix, token, 0, Some(tail));
        println!(
            "boundary_len={len} mod256={} mask={} commit_token={} commit_bytes={} complete_after_token={}",
            len % 256,
            mask_accepts,
            commit_token_accepts,
            commit_bytes_accepts,
            can_complete_after_token,
        );
    }
}

#[ignore = "expert experiment: matrix over token pre-close length at the 9th 256-byte boundary"]
#[test]
fn scan_o82710_inline_glrm_token_length_matrix() {
    let tail = b", \"id\": \"\"}";

    for body_len in 1usize..=8 {
        let token_bytes = {
            let mut token = vec![b'a'; body_len];
            token.push(b'"');
            token
        };
        let vocab = make_vocab(&[token_bytes.as_slice()]);
        let constraint = Constraint::from_glrm_grammar(r#"
    start start;

    t JSON_STRING_CHAR ::= /a/;
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
    "#, &vocab).unwrap();

        for len in 2297usize..=2305 {
            let prefix = direct_glrm_prefix_with_content(&vec![b'a'; len]);
            let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
                classify_constraint(&constraint, &prefix, &token_bytes, 0, Some(tail));
            println!(
                "matrix_body_len={} prefix_len={} mod256={} mask={} commit_token={} commit_bytes={} complete_after_token={}",
                body_len,
                len,
                len % 256,
                mask_accepts,
                commit_token_accepts,
                commit_bytes_accepts,
                can_complete_after_token,
            );
        }
    }
}

#[ignore = "expert experiment: compare fixed8, fixed9, and local alternation around the boundary"]
#[test]
fn scan_o82710_inline_glrm_fixed8_fixed9_alternatives() {
    let token = b"aaaaa\"";
    let tail = b", \"id\": \"\"}";
    let vocab = make_vocab(&[token]);
    let prefix = direct_glrm_prefix_with_content(&vec![b'a'; 2300]);

    for variant in [
        "bounded_fixed8",
        "bounded_fixed9",
        "bounded_alt_8_9",
        "bounded_alt_9_8",
    ] {
        let grammar = fixed_chunk_object_glrm(variant);
        let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
        let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
            classify_constraint(&constraint, &prefix, token, 0, Some(tail));
        println!(
            "variant={} mask={} commit_token={} commit_bytes={} complete_after_token={}",
            variant,
            mask_accepts,
            commit_token_accepts,
            commit_bytes_accepts,
            can_complete_after_token,
        );
    }
}

#[ignore = "expert experiment: explicit 8|9 alternatives versus counted-repeat lowering"]
#[test]
fn scan_o82710_inline_glrm_explicit_8_9_vs_counted_repeat() {
    let token = b"aaaaa\"";
    let tail = b", \"id\": \"\"}";
    let vocab = make_vocab(&[token]);
    let prefix = direct_glrm_prefix_with_content(&vec![b'a'; 2300]);

    for (label, grammar) in [
        ("counted_repeat", counted_repeat_object_glrm_a_only().to_string()),
        ("explicit_8_9", explicit_8_9_object_glrm(false)),
        ("explicit_9_8", explicit_8_9_object_glrm(true)),
    ] {
        let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
        let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
            classify_constraint(&constraint, &prefix, token, 0, Some(tail));
        println!(
            "explicit_compare={} mask={} commit_token={} commit_bytes={} complete_after_token={}",
            label,
            mask_accepts,
            commit_token_accepts,
            commit_bytes_accepts,
            can_complete_after_token,
        );
    }
}

#[ignore = "expert experiment: no-quote boundary crossing tokens versus close-quote token"]
#[test]
fn scan_o82710_inline_glrm_no_quote_crossing_tokens() {
    let tokens: [&[u8]; 4] = [b"aaaa", b"aaaaa", b"aaaaaa", b"aaaaa\""];
    let vocab = make_vocab(&tokens);
    let constraint = Constraint::from_glrm_grammar(r#"
start start;

t JSON_STRING_CHAR ::= /a/;
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
"#, &vocab).unwrap();
    let prefix = direct_glrm_prefix_with_content(&vec![b'a'; 2300]);
    let tail = b", \"id\": \"\"}";

    for (token_id, token) in tokens.iter().enumerate() {
        let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
            classify_constraint(&constraint, &prefix, token, token_id as u32, Some(tail));
        println!(
            "crossing_token={:?} mask={} commit_token={} commit_bytes={} complete_after_token={}",
            String::from_utf8_lossy(token),
            mask_accepts,
            commit_token_accepts,
            commit_bytes_accepts,
            can_complete_after_token,
        );
    }
}

#[ignore = "expert experiment: continuation ladder from pure string to recursive required-id object"]
#[test]
fn scan_o82710_inline_glrm_continuation_ladder() {
    let token = b"aaaaa\"";
    let vocab = make_vocab(&[token]);
    let content = vec![b'a'; 2300];

    for (label, grammar, prefix, tail) in [
        (
            "pure_bounded_string",
            bounded_string_only_glrm().to_string(),
            {
                let mut p = vec![b'"'];
                p.extend_from_slice(&content);
                p
            },
            None,
        ),
        (
            "object_immediate_close",
            object_close_glrm().to_string(),
            direct_glrm_prefix_with_content(&content),
            Some(b"}".as_slice()),
        ),
        (
            "object_required_id_nonrecursive",
            object_required_id_nonrecursive_glrm().to_string(),
            direct_glrm_prefix_with_content(&content),
            Some(b", \"id\": \"\"}".as_slice()),
        ),
        (
            "object_required_id_recursive",
            r#"
        start start;

        t JSON_STRING_CHAR ::= /a/;
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
        "#.to_string(),
            direct_glrm_prefix_with_content(&content),
            Some(b", \"id\": \"\"}".as_slice()),
        ),
    ] {
        let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
        let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
            classify_constraint(&constraint, &prefix, token, 0, tail);
        println!(
            "ladder={} mask={} commit_token={} commit_bytes={} complete_after_token={}",
            label,
            mask_accepts,
            commit_token_accepts,
            commit_bytes_accepts,
            can_complete_after_token,
        );
    }
}

#[ignore = "expert experiment: split the closing token across the boundary"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary() {
    let vocab = make_vocab(&[b"aaaaa\""]);
    let constraint = Constraint::from_glrm_grammar(r#"
        start start;

        t JSON_STRING_CHAR ::= /a/;
        t JSON_STRING_BODY ::= JSON_STRING_CHAR* "\"";
        nt json_string ::= "\"" JSON_STRING_BODY;
        internal t JSON_STRING_CHAR_UPTO_256_0 ::= JSON_STRING_CHAR{0,256};
        t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= JSON_STRING_CHAR_UPTO_256_0 "\"";
        t JSON_STRING_CHAR_EXACT_256_2 ::= JSON_STRING_CHAR{256};
        internal t JSON_STRING_CHAR_UPTO_136_3 ::= JSON_STRING_CHAR{0,136};
        t JSON_STRING_CHAR_UPTO_CLOSE_4 ::= JSON_STRING_CHAR_UPTO_136_3 "\"";
        nt json_string_bounded_split_5 ::= "\"" (JSON_STRING_CHAR_EXACT_256_2{0,18} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{19} JSON_STRING_CHAR_UPTO_CLOSE_4);
        nt obj_open_reqmask_0_nc_0 ::= json_string_bounded_split_5+ | obj_open_reqmask_0_c_1;
        nt obj_open_reqmask_0_c_1 ::= "," (json_string_bounded_split_5) obj_open_reqmask_0_c_1 | ;
        nt start ::= obj_open_reqmask_0_nc_0 "}";
    "#, &vocab).unwrap();
    let mut prefix = b"\"".to_vec();
    prefix.extend_from_slice(&vec![b'a'; 2300]);
    let tail = b"";

    let (full_mask, full_commit_token, full_commit_bytes, full_complete) =
        classify_constraint(&constraint, &prefix, [b"aaaaa\""][0], 0, Some(tail));
    println!(
        "split_full_token mask={} commit_token={} commit_bytes={} complete_after_token={}",
        full_mask,
        full_commit_token,
        full_commit_bytes,
        full_complete,
    );
    assert!(
        full_mask && full_commit_token && full_commit_bytes && full_complete
    );

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
