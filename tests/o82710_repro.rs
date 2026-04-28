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
    let token = b"aa\"";
    let vocab = Vocab::new(vec![(0, token.to_vec())], None);
    let constraint = Constraint::from_glrm_grammar(r#"
start start;
t A_EXACT ::= "a"{32};
nt start ::= (A_EXACT{4} ("a"{0,32} "\"") | A_EXACT{5}) ("a"{0,32} "\"");
    "#, &vocab).unwrap();
    let prefix = [b'a'; 159];

    let mut mask_state = constraint.start();
    mask_state.commit_bytes(&prefix).unwrap();
    let full_mask = mask_state.mask().first().map(|word| (word & 1) != 0).unwrap_or(false);

    let mut commit_token_state = constraint.start();
    commit_token_state.commit_bytes(&prefix).unwrap();
    let full_commit_token = match catch_unwind(AssertUnwindSafe(|| commit_token_state.commit_token(0))) {
        Ok(Ok(())) => true,
        Ok(Err(_)) => false,
        Err(_) => true,
    };

    let mut commit_bytes_state = constraint.start();
    commit_bytes_state.commit_bytes(&prefix).unwrap();
    let full_commit_bytes = commit_bytes_state.commit_bytes(token).is_ok();

    println!(
        "split_full_token mask={} commit_token={} commit_bytes={}",
        full_mask,
        full_commit_token,
        full_commit_bytes,
    );
    assert!(!full_mask && full_commit_token && full_commit_bytes);
}

#[ignore = "scanner for smaller counted-repeat chunk sizes in the split-token-boundary MRE"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_small_chunks() {
    for exact in [128usize, 64, 32, 16, 8, 4, 2, 1] {
        let token_body_len = 3usize;
        let prefix_len = 9 * exact - 2;
        let token = format!("{}\"", "a".repeat(token_body_len));
        let vocab = make_vocab(&[token.as_bytes()]);
        let grammar = format!(
            r#"
start start;
t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= "a"{{0,{exact}}} "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= "a"{{{exact}}};
nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_EXACT_256_2{{0,18}} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{{19}});
nt start ::= json_string_bounded_split_5+ | "," ? json_string_bounded_split_5 ;
"#,
        );
        let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
        let prefix = vec![b'a'; prefix_len];
        let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
            classify_constraint(&constraint, &prefix, token.as_bytes(), 0, Some(b""));
        println!(
            "small_chunk exact={} prefix_len={} residue={} token_len={} mask={} commit_token={} commit_bytes={} complete_after_token={}",
            exact,
            prefix_len,
            prefix_len % exact.max(1),
            token.len(),
            mask_accepts,
            commit_token_accepts,
            commit_bytes_accepts,
            can_complete_after_token,
        );
    }
}

#[ignore = "scanner for smaller exact-count and repeat-count combinations in the split-token-boundary MRE"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_parameter_search() {
    let token = b"aaa\"";
    for exact in [32usize, 24, 16, 12, 8, 6, 4] {
        for repeat_cap in 0usize..=18 {
            let full_repeat = repeat_cap + 1;
            let grammar = format!(
                r#"
start start;
t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= "a"{{0,{exact}}} "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= "a"{{{exact}}};
nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_EXACT_256_2{{0,{repeat_cap}}} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{{{full_repeat}}});
nt start ::= json_string_bounded_split_5+ | "," ? json_string_bounded_split_5 ;
"#,
            );
            let vocab = make_vocab(&[token]);
            let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();

            let mut found = None;
            let max_prefix = exact * (full_repeat + 1);
            for prefix_len in 0usize..=max_prefix {
                let prefix = vec![b'a'; prefix_len];
                let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
                    classify_constraint(&constraint, &prefix, token, 0, Some(b""));
                if !mask_accepts
                    && commit_token_accepts
                    && commit_bytes_accepts
                    && can_complete_after_token
                {
                    found = Some(prefix_len);
                    break;
                }
            }

            if let Some(prefix_len) = found {
                println!(
                    "parameter_search exact={} repeat_cap={} full_repeat={} prefix_len={} residue={} token_len={}",
                    exact,
                    repeat_cap,
                    full_repeat,
                    prefix_len,
                    prefix_len % exact.max(1),
                    token.len(),
                );
            }
        }
    }
}

#[ignore = "scanner for token body length in the smaller split-token-boundary MRE"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_token_lengths() {
    let exact = 32usize;
    let repeat_cap = 4usize;
    let full_repeat = repeat_cap + 1;

    for token_body_len in 0usize..=6 {
        let token = format!("{}\"", "a".repeat(token_body_len));
        let vocab = make_vocab(&[token.as_bytes()]);
        let grammar = format!(
            r#"
start start;
t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= "a"{{0,{exact}}} "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= "a"{{{exact}}};
nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_EXACT_256_2{{0,{repeat_cap}}} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{{{full_repeat}}});
nt start ::= json_string_bounded_split_5+ | "," ? json_string_bounded_split_5 ;
"#,
        );
        let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();

        let mut found = None;
        for prefix_len in 0usize..=(exact * (full_repeat + 1) + token_body_len + 2) {
            let prefix = vec![b'a'; prefix_len];
            let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
                classify_constraint(&constraint, &prefix, token.as_bytes(), 0, Some(b""));
            if !mask_accepts
                && commit_token_accepts
                && commit_bytes_accepts
                && can_complete_after_token
            {
                found = Some(prefix_len);
                break;
            }
        }

        println!(
            "token_length_search token_body_len={} token_len={} first_prefix={:?}",
            token_body_len,
            token.len(),
            found,
        );
    }
}

#[ignore = "scanner for under-32 exact sizes in the split-token-boundary MRE"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_under_32_search() {
    for exact in [24usize, 16, 12, 8, 6, 4, 3, 2, 1] {
        for repeat_cap in 0usize..=18 {
            let full_repeat = repeat_cap + 1;
            for token_body_len in 0usize..=8 {
                let token = format!("{}\"", "a".repeat(token_body_len));
                let vocab = make_vocab(&[token.as_bytes()]);
                let grammar = format!(
                    r#"
start start;
t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= "a"{{0,{exact}}} "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= "a"{{{exact}}};
nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_EXACT_256_2{{0,{repeat_cap}}} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{{{full_repeat}}});
nt start ::= json_string_bounded_split_5+ | "," ? json_string_bounded_split_5 ;
"#,
                );
                let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();

                let search_limit = exact * (full_repeat + 1) + token_body_len + 4;
                let mut found = None;
                for prefix_len in 0usize..=search_limit {
                    let prefix = vec![b'a'; prefix_len];
                    let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
                        classify_constraint(&constraint, &prefix, token.as_bytes(), 0, Some(b""));
                    if !mask_accepts
                        && commit_token_accepts
                        && commit_bytes_accepts
                        && can_complete_after_token
                    {
                        found = Some(prefix_len);
                        break;
                    }
                }

                if let Some(prefix_len) = found {
                    println!(
                        "under32_search exact={} repeat_cap={} full_repeat={} token_body_len={} token_len={} prefix_len={} residue={}",
                        exact,
                        repeat_cap,
                        full_repeat,
                        token_body_len,
                        token.len(),
                        prefix_len,
                        prefix_len % exact.max(1),
                    );
                }
            }
        }
    }
}

#[ignore = "focused scanner for smaller exact sizes near the current total-run boundary"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_constant_total_budget() {
    let target_totals = [128usize, 144, 160, 192];
    for exact in [24usize, 16, 12, 10, 8, 6, 5, 4, 3, 2, 1] {
        for target_total in target_totals {
            if target_total % exact != 0 {
                continue;
            }
            let full_repeat = target_total / exact;
            if full_repeat == 0 || full_repeat > 19 {
                continue;
            }
            let repeat_cap = full_repeat - 1;
            for token_body_len in 2usize..=8 {
                let token = format!("{}\"", "a".repeat(token_body_len));
                let vocab = make_vocab(&[token.as_bytes()]);
                let grammar = format!(
                    r#"
start start;
t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= "a"{{0,{exact}}} "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= "a"{{{exact}}};
nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_EXACT_256_2{{0,{repeat_cap}}} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{{{full_repeat}}});
nt start ::= json_string_bounded_split_5+ | "," ? json_string_bounded_split_5 ;
"#,
                );
                let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
                let predicted_prefix = target_total.saturating_sub(token_body_len - 1);
                let window_start = predicted_prefix.saturating_sub(4);
                let window_end = predicted_prefix + 4;

                for prefix_len in window_start..=window_end {
                    let prefix = vec![b'a'; prefix_len];
                    let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
                        classify_constraint(&constraint, &prefix, token.as_bytes(), 0, Some(b""));
                    if !mask_accepts
                        && commit_token_accepts
                        && commit_bytes_accepts
                        && can_complete_after_token
                    {
                        println!(
                            "constant_budget exact={} repeat_cap={} full_repeat={} target_total={} token_body_len={} token_len={} prefix_len={} residue={}",
                            exact,
                            repeat_cap,
                            full_repeat,
                            target_total,
                            token_body_len,
                            token.len(),
                            prefix_len,
                            prefix_len % exact.max(1),
                        );
                        break;
                    }
                }
            }
        }
    }
}

#[ignore = "focused scanner for direct exact-run then close-run sequences"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_direct_two_piece_sequences() {
    for exact in [32usize, 24, 16, 12, 10, 8, 6, 5, 4, 3, 2, 1] {
        for exact_repeats in 1usize..=19 {
            for token_body_len in 2usize..=8 {
                let token = format!("{}\"", "a".repeat(token_body_len));
                let vocab = make_vocab(&[token.as_bytes()]);
                let grammar = format!(
                    r#"
start start;
t A_UPTO_CLOSE ::= "a"{{0,{exact}}} "\"";
t A_EXACT ::= "a"{{{exact}}};
nt start ::= A_EXACT{{{exact_repeats}}} A_UPTO_CLOSE;
"#,
                );
                let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
                let predicted_prefix = (exact * exact_repeats).saturating_sub(token_body_len - 1);
                let window_start = predicted_prefix.saturating_sub(4);
                let window_end = predicted_prefix + 4;

                for prefix_len in window_start..=window_end {
                    let prefix = vec![b'a'; prefix_len];
                    let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
                        classify_constraint(&constraint, &prefix, token.as_bytes(), 0, Some(b""));
                    if !mask_accepts
                        && commit_token_accepts
                        && commit_bytes_accepts
                        && can_complete_after_token
                    {
                        println!(
                            "direct_two_piece exact={} exact_repeats={} token_body_len={} token_len={} prefix_len={} residue={}",
                            exact,
                            exact_repeats,
                            token_body_len,
                            token.len(),
                            prefix_len,
                            prefix_len % exact.max(1),
                        );
                        break;
                    }
                }
            }
        }
    }
}

#[ignore = "focused scanner for simpler start-rule shapes around the current chunk witness"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_start_rule_shapes() {
    let token = b"aaa\"";
    let vocab = make_vocab(&[token]);
    let exact = 32usize;
    let repeat_cap = 4usize;
    let full_repeat = repeat_cap + 1;

    for (label, start_rule, prefix_len) in [
        (
            "current",
            "nt start ::= json_string_bounded_split_5+ | \",\" ? json_string_bounded_split_5 ;",
            158usize,
        ),
        (
            "two_chunks",
            "nt start ::= json_string_bounded_split_5 json_string_bounded_split_5 ;",
            158usize,
        ),
        (
            "exact_then_chunk",
            "nt start ::= JSON_STRING_CHAR_EXACT_256_2{5} json_string_bounded_split_5 ;",
            158usize,
        ),
        (
            "chunk_then_close",
            "nt start ::= json_string_bounded_split_5 JSON_STRING_CHAR_UPTO_CLOSE_1 ;",
            158usize,
        ),
        (
            "five_exact_then_close",
            "nt start ::= JSON_STRING_CHAR_EXACT_256_2{5} JSON_STRING_CHAR_UPTO_CLOSE_1 ;",
            158usize,
        ),
    ] {
        let grammar = format!(
            r#"
start start;
t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= "a"{{0,{exact}}} "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= "a"{{{exact}}};
nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_EXACT_256_2{{0,{repeat_cap}}} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{{{full_repeat}}});
{start_rule}
"#,
        );
        let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
        let prefix = vec![b'a'; prefix_len];
        let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
            classify_constraint(&constraint, &prefix, token, 0, Some(b""));
        println!(
            "start_rule_shape={} mask={} commit_token={} commit_bytes={} complete_after_token={}",
            label,
            mask_accepts,
            commit_token_accepts,
            commit_bytes_accepts,
            can_complete_after_token,
        );
    }
}

#[ignore = "focused scanner for simpler left-chunk definitions in the split-boundary MRE"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_left_chunk_shapes() {
    let token = b"aaa\"";
    let vocab = make_vocab(&[token]);
    let exact = 32usize;

    for (label, chunk_def, prefix_len) in [
        (
            "current_chunk",
            "nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_EXACT_256_2{0,4} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{5});",
            158usize,
        ),
        (
            "close_or_five_exact",
            "nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{5});",
            158usize,
        ),
        (
            "close_or_exact_exact",
            "nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2 JSON_STRING_CHAR_EXACT_256_2 JSON_STRING_CHAR_EXACT_256_2 JSON_STRING_CHAR_EXACT_256_2 JSON_STRING_CHAR_EXACT_256_2);",
            158usize,
        ),
        (
            "two_close_branches_or_five_exact",
            "nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_EXACT_256_2{0,1} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{5});",
            158usize,
        ),
        (
            "four_close_branches_or_five_exact",
            "nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_EXACT_256_2{0,3} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{5});",
            158usize,
        ),
    ] {
        let grammar = format!(
            r#"
start start;
t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= "a"{{0,{exact}}} "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= "a"{{{exact}}};
{chunk_def}
nt start ::= json_string_bounded_split_5 JSON_STRING_CHAR_UPTO_CLOSE_1 ;
"#,
        );
        let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
        let prefix = vec![b'a'; prefix_len];
        let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
            classify_constraint(&constraint, &prefix, token, 0, Some(b""));
        println!(
            "left_chunk_shape={} mask={} commit_token={} commit_bytes={} complete_after_token={}",
            label,
            mask_accepts,
            commit_token_accepts,
            commit_bytes_accepts,
            can_complete_after_token,
        );
    }
}

#[ignore = "focused scanner for scale reductions in the chunk-then-close MRE family"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_chunk_then_close_scale() {
    for exact in [32usize, 24, 16, 12, 10, 8, 6, 5, 4, 3, 2, 1] {
        for token_body_len in 1usize..=6 {
            let token = format!("{}\"", "a".repeat(token_body_len));
            let vocab = make_vocab(&[token.as_bytes()]);
            let grammar = format!(
                r#"
start start;
t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= "a"{{0,{exact}}} "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= "a"{{{exact}}};
nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_EXACT_256_2{{0,4}} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{{5}});
nt start ::= json_string_bounded_split_5 JSON_STRING_CHAR_UPTO_CLOSE_1 ;
"#,
            );
            let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
            let predicted_prefix = (5 * exact).saturating_sub(token_body_len - 1);
            let window_start = predicted_prefix.saturating_sub(4);
            let window_end = predicted_prefix + 4;

            for prefix_len in window_start..=window_end {
                let prefix = vec![b'a'; prefix_len];
                let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
                    classify_constraint(&constraint, &prefix, token.as_bytes(), 0, Some(b""));
                if !mask_accepts
                    && commit_token_accepts
                    && commit_bytes_accepts
                    && can_complete_after_token
                {
                    println!(
                        "chunk_then_close_scale exact={} token_body_len={} token_len={} prefix_len={} residue={}",
                        exact,
                        token_body_len,
                        token.len(),
                        prefix_len,
                        prefix_len % exact.max(1),
                    );
                    break;
                }
            }
        }
    }
}

#[ignore = "focused scanner for inlining the left chunk directly into start"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_inline_start() {
    let token = b"aa\"";
    let vocab = make_vocab(&[token]);

    for (label, grammar, prefix_len) in [
        (
            "chunk_nonterminal",
            r#"
start start;
t A_UPTO_CLOSE ::= "a"{0,32} "\"";
t A_EXACT ::= "a"{32};
nt chunk ::= (A_EXACT{0,4} A_UPTO_CLOSE | A_EXACT{5});
nt start ::= chunk A_UPTO_CLOSE;
"#,
            159usize,
        ),
        (
            "inline_start",
            r#"
start start;
t A_UPTO_CLOSE ::= "a"{0,32} "\"";
t A_EXACT ::= "a"{32};
nt start ::= (A_EXACT{0,4} A_UPTO_CLOSE | A_EXACT{5}) A_UPTO_CLOSE;
"#,
            159usize,
        ),
    ] {
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let prefix = vec![b'a'; prefix_len];
        let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
            classify_constraint(&constraint, &prefix, token, 0, Some(b""));
        println!(
            "inline_start_shape={} mask={} commit_token={} commit_bytes={} complete_after_token={}",
            label,
            mask_accepts,
            commit_token_accepts,
            commit_bytes_accepts,
            can_complete_after_token,
        );
    }
}

#[ignore = "focused scanner for the minimal exact count in the current inline-start family"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_inline_start_exact_range() {
    let token = b"aa\"";

    for exact in 1usize..=32 {
        let vocab = make_vocab(&[token]);
        let grammar = format!(
            r#"
start start;
t A_UPTO_CLOSE ::= "a"{{0,{exact}}} "\"";
t A_EXACT ::= "a"{{{exact}}};
nt start ::= (A_EXACT{{0,4}} A_UPTO_CLOSE | A_EXACT{{5}}) A_UPTO_CLOSE;
"#,
        );
        let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
        let predicted_prefix = (5 * exact).saturating_sub(1);
        let window_start = predicted_prefix.saturating_sub(6);
        let window_end = predicted_prefix + 6;

        for prefix_len in window_start..=window_end {
            let prefix = vec![b'a'; prefix_len];
            let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
                classify_constraint(&constraint, &prefix, token, 0, Some(b""));
            if !mask_accepts
                && commit_token_accepts
                && commit_bytes_accepts
                && can_complete_after_token
            {
                println!(
                    "inline_start_exact_hit exact={} prefix_len={} residue={}",
                    exact,
                    prefix_len,
                    prefix_len % exact,
                );
                break;
            }
        }
    }
}

#[ignore = "focused scanner for the minimal repeat cap in the current inline-start family"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_inline_start_repeat_caps() {
    let token = b"aa\"";
    let vocab = make_vocab(&[token]);
    let exact = 32usize;

    for repeat_cap in 0usize..=12 {
        let grammar = format!(
            r#"
start start;
t A_UPTO_CLOSE ::= "a"{{0,{exact}}} "\"";
t A_EXACT ::= "a"{{{exact}}};
nt start ::= (A_EXACT{{0,{repeat_cap}}} A_UPTO_CLOSE | A_EXACT{{{}}}) A_UPTO_CLOSE;
"#,
            repeat_cap + 1,
        );
        let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
        let predicted_prefix = ((repeat_cap + 1) * exact).saturating_sub(1);
        let window_start = predicted_prefix.saturating_sub(6);
        let window_end = predicted_prefix + 6;

        for prefix_len in window_start..=window_end {
            let prefix = vec![b'a'; prefix_len];
            let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
                classify_constraint(&constraint, &prefix, token, 0, Some(b""));
            if !mask_accepts
                && commit_token_accepts
                && commit_bytes_accepts
                && can_complete_after_token
            {
                println!(
                    "inline_start_repeat_cap_hit repeat_cap={} prefix_len={} residue={}",
                    repeat_cap,
                    prefix_len,
                    prefix_len % exact,
                );
                break;
            }
        }
    }
}

#[ignore = "focused scanner for sparse close-side branch sets in the inline-start family"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_inline_start_sparse_branches() {
    let token = b"aa\"";
    let vocab = make_vocab(&[token]);

    for (label, start_rule) in [
        (
            "range_0_4",
            "nt start ::= (A_EXACT{0,4} A_UPTO_CLOSE | A_EXACT{5}) A_UPTO_CLOSE;",
        ),
        (
            "only_4",
            "nt start ::= (A_EXACT{4} A_UPTO_CLOSE | A_EXACT{5}) A_UPTO_CLOSE;",
        ),
        (
            "only_3_or_4",
            "nt start ::= ((A_EXACT{3} | A_EXACT{4}) A_UPTO_CLOSE | A_EXACT{5}) A_UPTO_CLOSE;",
        ),
        (
            "only_2_3_4",
            "nt start ::= ((A_EXACT{2} | A_EXACT{3} | A_EXACT{4}) A_UPTO_CLOSE | A_EXACT{5}) A_UPTO_CLOSE;",
        ),
        (
            "only_0_or_4",
            "nt start ::= ((A_EXACT{0} | A_EXACT{4}) A_UPTO_CLOSE | A_EXACT{5}) A_UPTO_CLOSE;",
        ),
    ] {
        let grammar = format!(
            r#"
start start;
t A_UPTO_CLOSE ::= "a"{{0,32}} "\"";
t A_EXACT ::= "a"{{32}};
{start_rule}
"#,
        );
        let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();

        for prefix_len in 152usize..=164 {
            let prefix = vec![b'a'; prefix_len];
            let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
                classify_constraint(&constraint, &prefix, token, 0, Some(b""));
            if !mask_accepts
                && commit_token_accepts
                && commit_bytes_accepts
                && can_complete_after_token
            {
                println!(
                    "inline_start_sparse_branches_hit label={} prefix_len={} residue={}",
                    label,
                    prefix_len,
                    prefix_len % 32,
                );
                break;
            }
        }
    }
}

#[ignore = "focused scanner for factoring the common exact-prefix in the current MRE"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_factored_prefix() {
    let token = b"aa\"";
    let vocab = make_vocab(&[token]);

    for (label, start_rule) in [
        (
            "single_branch",
            "nt start ::= (A_EXACT{4} A_UPTO_CLOSE | A_EXACT{5}) A_UPTO_CLOSE;",
        ),
        (
            "factored_prefix",
            "nt start ::= A_EXACT{4} (A_UPTO_CLOSE | A_EXACT) A_UPTO_CLOSE;",
        ),
    ] {
        let grammar = format!(
            r#"
start start;
t A_UPTO_CLOSE ::= "a"{{0,32}} "\"";
t A_EXACT ::= "a"{{32}};
{start_rule}
"#,
        );
        let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
        let prefix = vec![b'a'; 159];
        let (mask_accepts, commit_token_accepts, commit_bytes_accepts, can_complete_after_token) =
            classify_constraint(&constraint, &prefix, token, 0, Some(b""));
        println!(
            "factored_prefix_shape={} mask={} commit_token={} commit_bytes={} complete_after_token={}",
            label,
            mask_accepts,
            commit_token_accepts,
            commit_bytes_accepts,
            can_complete_after_token,
        );
    }
}

#[ignore = "focused scanner for replacing repeated exact chunks with explicit long terminals"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_explicit_long_terminals() {
    let token = b"aa\"";
    let vocab = Vocab::new(vec![(0, token.to_vec())], None);

    for (label, grammar) in [
        (
            "counted_repeat",
            r#"
start start;
t A_UPTO_CLOSE ::= "a"{0,32} "\"";
t A_EXACT ::= "a"{32};
nt start ::= (A_EXACT{4} A_UPTO_CLOSE | A_EXACT{5}) A_UPTO_CLOSE;
"#,
        ),
        (
            "explicit_long_terminals",
            r#"
start start;
t A_UPTO_CLOSE ::= "a"{0,32} "\"";
t A_128 ::= "a"{128};
t A_160 ::= "a"{160};
nt start ::= (A_128 A_UPTO_CLOSE | A_160) A_UPTO_CLOSE;
"#,
        ),
    ] {
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let prefix = [b'a'; 159];

        let mut mask_state = constraint.start();
        mask_state.commit_bytes(&prefix).unwrap();
        let mask_accepts = mask_state.mask().first().map(|word| (word & 1) != 0).unwrap_or(false);

        let mut commit_token_state = constraint.start();
        commit_token_state.commit_bytes(&prefix).unwrap();
        let commit_token_accepts = match catch_unwind(AssertUnwindSafe(|| commit_token_state.commit_token(0))) {
            Ok(Ok(())) => true,
            Ok(Err(_)) => false,
            Err(_) => true,
        };

        let mut commit_bytes_state = constraint.start();
        commit_bytes_state.commit_bytes(&prefix).unwrap();
        let commit_bytes_accepts = commit_bytes_state.commit_bytes(token).is_ok();

        println!(
            "explicit_long_terminals_shape={} mask={} commit_token={} commit_bytes={}",
            label,
            mask_accepts,
            commit_token_accepts,
            commit_bytes_accepts,
        );
    }
}

#[ignore = "focused scanner for spelling out the exact chunks explicitly"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_explicit_chunk_sequence() {
    let token = b"aa\"";
    let vocab = Vocab::new(vec![(0, token.to_vec())], None);

    for (label, grammar) in [
        (
            "counted_repeat",
            r#"
start start;
t A_UPTO_CLOSE ::= "a"{0,32} "\"";
t A_EXACT ::= "a"{32};
nt start ::= (A_EXACT{4} A_UPTO_CLOSE | A_EXACT{5}) A_UPTO_CLOSE;
"#,
        ),
        (
            "explicit_chunk_sequence",
            r#"
start start;
t A_UPTO_CLOSE ::= "a"{0,32} "\"";
t A_EXACT ::= "a"{32};
nt start ::= (
    A_EXACT A_EXACT A_EXACT A_EXACT A_UPTO_CLOSE
    | A_EXACT A_EXACT A_EXACT A_EXACT A_EXACT
) A_UPTO_CLOSE;
"#,
        ),
    ] {
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let prefix = [b'a'; 159];

        let mut mask_state = constraint.start();
        mask_state.commit_bytes(&prefix).unwrap();
        let mask_accepts = mask_state.mask().first().map(|word| (word & 1) != 0).unwrap_or(false);

        let mut commit_token_state = constraint.start();
        commit_token_state.commit_bytes(&prefix).unwrap();
        let commit_token_accepts = match catch_unwind(AssertUnwindSafe(|| commit_token_state.commit_token(0))) {
            Ok(Ok(())) => true,
            Ok(Err(_)) => false,
            Err(_) => true,
        };

        let mut commit_bytes_state = constraint.start();
        commit_bytes_state.commit_bytes(&prefix).unwrap();
        let commit_bytes_accepts = commit_bytes_state.commit_bytes(token).is_ok();

        println!(
            "explicit_chunk_sequence_shape={} mask={} commit_token={} commit_bytes={}",
            label,
            mask_accepts,
            commit_token_accepts,
            commit_bytes_accepts,
        );
    }
}

#[ignore = "focused scanner for shorter tokens in the current single-branch witness"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_single_branch_token_lengths() {
    for token_body_len in 0usize..=5 {
        let token = format!("{}\"", "a".repeat(token_body_len));
        let vocab = Vocab::new(vec![(0, token.as_bytes().to_vec())], None);
        let constraint = Constraint::from_glrm_grammar(r#"
start start;
t A_UPTO_CLOSE ::= "a"{0,32} "\"";
t A_EXACT ::= "a"{32};
nt start ::= (A_EXACT{4} A_UPTO_CLOSE | A_EXACT{5}) A_UPTO_CLOSE;
"#, &vocab).unwrap();

        let predicted_prefix = 160usize.saturating_sub(token_body_len.saturating_sub(1));
        let window_start = predicted_prefix.saturating_sub(6);
        let window_end = predicted_prefix + 6;

        for prefix_len in window_start..=window_end {
            let prefix = vec![b'a'; prefix_len];

            let mut mask_state = constraint.start();
            mask_state.commit_bytes(&prefix).unwrap();
            let mask_accepts = mask_state.mask().first().map(|word| (word & 1) != 0).unwrap_or(false);

            let mut commit_token_state = constraint.start();
            commit_token_state.commit_bytes(&prefix).unwrap();
            let commit_token_accepts = match catch_unwind(AssertUnwindSafe(|| commit_token_state.commit_token(0))) {
                Ok(Ok(())) => true,
                Ok(Err(_)) => false,
                Err(_) => true,
            };

            let mut commit_bytes_state = constraint.start();
            commit_bytes_state.commit_bytes(&prefix).unwrap();
            let commit_bytes_accepts = commit_bytes_state.commit_bytes(token.as_bytes()).is_ok();

            if !mask_accepts && commit_token_accepts && commit_bytes_accepts {
                println!(
                    "single_branch_token_length_hit token_body_len={} token_len={} prefix_len={} residue={}",
                    token_body_len,
                    token.len(),
                    prefix_len,
                    prefix_len % 32,
                );
                break;
            }
        }
    }
}

#[ignore = "focused scanner for smaller close-terminal caps in the current single-branch witness"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_single_branch_close_caps() {
    let token = b"aa\"";
    let vocab = Vocab::new(vec![(0, token.to_vec())], None);

    for close_cap in 2usize..=32 {
        let grammar = format!(
            r#"
start start;
t A_UPTO_CLOSE ::= "a"{{0,{close_cap}}} "\"";
t A_EXACT ::= "a"{{32}};
nt start ::= (A_EXACT{{4}} A_UPTO_CLOSE | A_EXACT{{5}}) A_UPTO_CLOSE;
"#,
        );
        let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();

        for prefix_len in 150usize..=159 {
            let prefix = vec![b'a'; prefix_len];

            let mut mask_state = constraint.start();
            if mask_state.commit_bytes(&prefix).is_err() {
                continue;
            }
            let mask_accepts = mask_state.mask().first().map(|word| (word & 1) != 0).unwrap_or(false);

            let mut commit_token_state = constraint.start();
            if commit_token_state.commit_bytes(&prefix).is_err() {
                continue;
            }
            let commit_token_accepts = match catch_unwind(AssertUnwindSafe(|| commit_token_state.commit_token(0))) {
                Ok(Ok(())) => true,
                Ok(Err(_)) => false,
                Err(_) => true,
            };

            let mut commit_bytes_state = constraint.start();
            if commit_bytes_state.commit_bytes(&prefix).is_err() {
                continue;
            }
            let commit_bytes_accepts = commit_bytes_state.commit_bytes(token).is_ok();

            if !mask_accepts && commit_token_accepts && commit_bytes_accepts {
                println!(
                    "single_branch_close_cap_hit close_cap={} prefix_len={} residue={}",
                    close_cap,
                    prefix_len,
                    prefix_len % 32,
                );
                break;
            }
        }
    }
}

#[ignore = "focused scanner for inlining the close-terminal expression into start"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_inline_close_expression() {
    let token = b"aa\"";
    let vocab = Vocab::new(vec![(0, token.to_vec())], None);

    for (label, grammar) in [
        (
            "named_close_terminal",
            r#"
start start;
t A_UPTO_CLOSE ::= "a"{0,32} "\"";
t A_EXACT ::= "a"{32};
nt start ::= (A_EXACT{4} A_UPTO_CLOSE | A_EXACT{5}) A_UPTO_CLOSE;
"#,
        ),
        (
            "inline_close_expression",
            r#"
start start;
t A_EXACT ::= "a"{32};
nt start ::= (A_EXACT{4} ("a"{0,32} "\"") | A_EXACT{5}) ("a"{0,32} "\"");
"#,
        ),
    ] {
        let Ok(constraint) = Constraint::from_glrm_grammar(grammar, &vocab) else {
            println!("inline_close_expression_shape={} compile=false", label);
            continue;
        };
        let prefix = [b'a'; 159];

        let mut mask_state = constraint.start();
        mask_state.commit_bytes(&prefix).unwrap();
        let mask_accepts = mask_state.mask().first().map(|word| (word & 1) != 0).unwrap_or(false);

        let mut commit_token_state = constraint.start();
        commit_token_state.commit_bytes(&prefix).unwrap();
        let commit_token_accepts = match catch_unwind(AssertUnwindSafe(|| commit_token_state.commit_token(0))) {
            Ok(Ok(())) => true,
            Ok(Err(_)) => false,
            Err(_) => true,
        };

        let mut commit_bytes_state = constraint.start();
        commit_bytes_state.commit_bytes(&prefix).unwrap();
        let commit_bytes_accepts = commit_bytes_state.commit_bytes(token).is_ok();

        println!(
            "inline_close_expression_shape={} compile=true mask={} commit_token={} commit_bytes={}",
            label,
            mask_accepts,
            commit_token_accepts,
            commit_bytes_accepts,
        );
    }
}

#[ignore = "focused scanner for fully inlining the witness grammar into start"]
#[test]
fn scan_o82710_inline_glrm_split_token_boundary_fully_inline_start() {
    let token = b"aa\"";
    let vocab = Vocab::new(vec![(0, token.to_vec())], None);

    for (label, grammar) in [
        (
            "inline_close_expression",
            r#"
start start;
t A_EXACT ::= "a"{32};
nt start ::= (A_EXACT{4} ("a"{0,32} "\"") | A_EXACT{5}) ("a"{0,32} "\"");
"#,
        ),
        (
            "fully_inline_start",
            r#"
start start;
nt start ::= (("a"{32}){4} ("a"{0,32} "\"") | ("a"{32}){5}) ("a"{0,32} "\"");
"#,
        ),
    ] {
        let Ok(constraint) = Constraint::from_glrm_grammar(grammar, &vocab) else {
            println!("fully_inline_start_shape={} compile=false", label);
            continue;
        };
        let prefix = [b'a'; 159];

        let mut mask_state = constraint.start();
        mask_state.commit_bytes(&prefix).unwrap();
        let mask_accepts = mask_state.mask().first().map(|word| (word & 1) != 0).unwrap_or(false);

        let mut commit_token_state = constraint.start();
        commit_token_state.commit_bytes(&prefix).unwrap();
        let commit_token_accepts = match catch_unwind(AssertUnwindSafe(|| commit_token_state.commit_token(0))) {
            Ok(Ok(())) => true,
            Ok(Err(_)) => false,
            Err(_) => true,
        };

        let mut commit_bytes_state = constraint.start();
        commit_bytes_state.commit_bytes(&prefix).unwrap();
        let commit_bytes_accepts = commit_bytes_state.commit_bytes(token).is_ok();

        println!(
            "fully_inline_start_shape={} compile=true mask={} commit_token={} commit_bytes={}",
            label,
            mask_accepts,
            commit_token_accepts,
            commit_bytes_accepts,
        );
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
