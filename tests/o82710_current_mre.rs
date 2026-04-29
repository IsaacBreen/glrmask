use glrmask::{Constraint, Vocab};
use serde_json::Value;
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

const DISPUTED_TOKEN_ID: u32 = 68439;
const DISPUTED_TOKEN_BYTES: &[u8] = b"'];?>\"";
const CONTROL_TOKEN_ID: u32 = 99925;
const CONTROL_TOKEN_BYTES: &[u8] = b" Vimeo";
const SPARSE_SCHEMA_GLRM: &str = r#"
start start;

internal t JSON_STRING_CHAR ::= /[^\x00-\x1f\x7f"\\]|\\["\\\/bfnrt]|\\u[0-9A-Fa-f]{4}/;
t JSON_STRING_BODY ::= JSON_STRING_CHAR* "\"";
nt json_string ::= "\"" JSON_STRING_BODY;
t JSON_INTEGER ::= /-?(0|[1-9][0-9]*)/;
t JSON_NUMBER ::= /-?(0|[1-9][0-9]*)(\.[0-9]+([eE][+-]?[0-9]+)?|[eE][+-]?[0-9]+)/;
t JSON_BOOL ::= "true" | "false";
t JSON_NULL ::= "null";
t JSON_KEY_COLON_BODY ::= JSON_STRING_CHAR* "\"";
nt json_key_colon ::= "\"" JSON_KEY_COLON_BODY ": ";
nt json_kv ::= json_key_colon json_value;
nt json_object ::= "{" ", " ~ ( json_kv* ) "}";
nt json_array ::= "[" ", " ~ ( json_value* ) "]";
nt json_value ::= json_object | json_array | json_string | JSON_NUMBER | JSON_INTEGER | JSON_BOOL | JSON_NULL;
t JSON_STRING_PATTERN_FULLMATCH_0 ::= ([ \-0-9A-Z_a-z] | "\xC2" "\x85" | "\xC2" "\xA0" | "\\" "t" | "\\" "u" "0" "0" "0" "9" | "\\" "n" | "\\" "u" "0" "0" "0" [Aa] | "\\" "u" "0" "0" "0" [Bb] | "\\" "f" | "\\" "u" "0" "0" "0" [Cc] | "\\" "r" | "\\" "u" "0" "0" "0" [Dd])+ "\"";
internal t JSON_STRING_CHAR_UPTO_256_1 ::= JSON_STRING_CHAR{0,256};
t JSON_STRING_CHAR_UPTO_CLOSE_2 ::= JSON_STRING_CHAR_UPTO_256_1 "\"";
t JSON_STRING_CHAR_EXACT_256_3 ::= JSON_STRING_CHAR{256};
internal t JSON_STRING_CHAR_UPTO_136_4 ::= JSON_STRING_CHAR{0,136};
t JSON_STRING_CHAR_UPTO_CLOSE_5 ::= JSON_STRING_CHAR_UPTO_136_4 "\"";
nt json_string_bounded_split_6 ::= "\"" (JSON_STRING_CHAR_EXACT_256_3{0,18} JSON_STRING_CHAR_UPTO_CLOSE_2 | JSON_STRING_CHAR_EXACT_256_3{19} JSON_STRING_CHAR_UPTO_CLOSE_5);
internal t AP_SHARED_KEY_COLON_7 ::= "aside\"" | "autoplay\"" | "css_class\"" | "description\"" | "id\"";
internal t AP_SHARED_KEY_COLON_8 ::= ((([ !#-[\]-~] | [\xC2-\xDF] [\x80-\xBF] | [\xE0] [\xA0-\xBF] [\x80-\xBF] | [\xE1-\xEC] [\x80-\xBF] [\x80-\xBF] | [\xED] [\x80-\x9F] [\x80-\xBF] | [\xEE\xEF] [\x80-\xBF] [\x80-\xBF] | [\xF0] [\x90-\xBF] [\x80-\xBF] [\x80-\xBF] | [\xF1-\xF3] [\x80-\xBF] [\x80-\xBF] [\x80-\xBF] | [\xF4] [\x80-\x8F] [\x80-\xBF] [\x80-\xBF]) | "\\" ["/\\bfnrt] | "\\" "u" [0-9A-Fa-f]{4})* "\"");
t AP_SHARED_KEY_COLON_9 ::= AP_SHARED_KEY_COLON_8 - AP_SHARED_KEY_COLON_7;
nt obj_open_reqmask_0_nc_0 ::= (("\"" "aside\"" ": ") JSON_BOOL) obj_open_reqmask_0_c_0 | (("\"" "autoplay\"" ": ") JSON_BOOL) obj_open_reqmask_0_c_0 | (("\"" "css_class\"" ": ") ("\"" JSON_STRING_PATTERN_FULLMATCH_0)) obj_open_reqmask_0_c_0 | (("\"" "description\"" ": ") json_string_bounded_split_6) obj_open_reqmask_0_c_0 | (("\"" AP_SHARED_KEY_COLON_9 ": ") json_value) obj_open_reqmask_0_c_0 | (("\"" "id\"" ": ") json_value) obj_open_reqmask_0_c_1;
nt obj_open_reqmask_0_c_0 ::= ", " (("\"" "aside\"" ": ") JSON_BOOL) obj_open_reqmask_0_c_0 | ", " (("\"" "autoplay\"" ": ") JSON_BOOL) obj_open_reqmask_0_c_0 | ", " (("\"" "css_class\"" ": ") ("\"" JSON_STRING_PATTERN_FULLMATCH_0)) obj_open_reqmask_0_c_0 | ", " (("\"" "description\"" ": ") json_string_bounded_split_6) obj_open_reqmask_0_c_0 | ", " (("\"" AP_SHARED_KEY_COLON_9 ": ") json_value) obj_open_reqmask_0_c_0 | ", " (("\"" "id\"" ": ") json_value) obj_open_reqmask_0_c_1;
nt obj_open_reqmask_0_c_1 ::= ", " (("\"" "aside\"" ": ") JSON_BOOL) obj_open_reqmask_0_c_1 | ", " (("\"" "autoplay\"" ": ") JSON_BOOL) obj_open_reqmask_0_c_1 | ", " (("\"" "css_class\"" ": ") ("\"" JSON_STRING_PATTERN_FULLMATCH_0)) obj_open_reqmask_0_c_1 | ", " (("\"" "description\"" ": ") json_string_bounded_split_6) obj_open_reqmask_0_c_1 | ", " (("\"" AP_SHARED_KEY_COLON_9 ": ") json_value) obj_open_reqmask_0_c_1 | ;
nt start ::= "{" obj_open_reqmask_0_nc_0 "}";
"#;
const MINIMIZED_INLINE_GLRM_CANDIDATE: &str = r#"
start start;

internal t JSON_STRING_CHAR ::= /[^\x00-\x1f\x7f"\\]|\\["\\\/bfnrt]|\\u[0-9A-Fa-f]{4}/;
t JSON_STRING_BODY ::= JSON_STRING_CHAR* "\"";
nt json_string ::= "\"" JSON_STRING_BODY;
t JSON_BOOL ::= "true" | "false";
t JSON_STRING_PATTERN_FULLMATCH_0 ::= ([ \-0-9A-Z_a-z] | "\xC2" "\x85" | "\xC2" "\xA0" | "\\" "t" | "\\" "u" "0" "0" "0" "9" | "\\" "n" | "\\" "u" "0" "0" "0" [Aa] | "\\" "u" "0" "0" "0" [Bb] | "\\" "f" | "\\" "u" "0" "0" "0" [Cc] | "\\" "r" | "\\" "u" "0" "0" "0" [Dd])+ "\"";
internal t JSON_STRING_CHAR_UPTO_256_1 ::= JSON_STRING_CHAR{0,256};
t JSON_STRING_CHAR_UPTO_CLOSE_2 ::= JSON_STRING_CHAR_UPTO_256_1 "\"";
t JSON_STRING_CHAR_EXACT_256_3 ::= JSON_STRING_CHAR{256};
internal t JSON_STRING_CHAR_UPTO_136_4 ::= JSON_STRING_CHAR{0,136};
t JSON_STRING_CHAR_UPTO_CLOSE_5 ::= JSON_STRING_CHAR_UPTO_136_4 "\"";
nt json_string_bounded_split_6 ::= "\"" (JSON_STRING_CHAR_EXACT_256_3{0,18} JSON_STRING_CHAR_UPTO_CLOSE_2 | JSON_STRING_CHAR_EXACT_256_3{19} JSON_STRING_CHAR_UPTO_CLOSE_5);
nt obj_open_reqmask_0_nc_0 ::= (("\"" "description\"" ": ") json_string_bounded_split_6) obj_open_reqmask_0_c_0 | (("\"" "id\"" ": ") json_string) obj_open_reqmask_0_c_1;
nt obj_open_reqmask_0_c_0 ::= ", " (("\"" "description\"" ": ") json_string_bounded_split_6) obj_open_reqmask_0_c_0 | ", " (("\"" "id\"" ": ") json_string) obj_open_reqmask_0_c_1;
nt obj_open_reqmask_0_c_1 ::= ", " (("\"" "description\"" ": ") json_string_bounded_split_6) obj_open_reqmask_0_c_1 | ;
nt start ::= "{" obj_open_reqmask_0_nc_0 "}";
"#;

fn token_allowed(mask: &[u32], id: usize) -> bool {
    let word = id / 32;
    if word >= mask.len() {
        return false;
    }
    ((mask[word] >> (id % 32)) & 1) != 0
}

fn classify_constraint(
    constraint: &Constraint,
    prefix: &[u8],
    token_id: u32,
    token_bytes: &[u8],
) -> (bool, bool, bool) {
    let mut mask_state = constraint.start();
    mask_state.commit_bytes(prefix).unwrap();
    let mask_accepts = token_allowed(&mask_state.mask(), token_id as usize);

    let mut commit_token_state = constraint.start();
    commit_token_state.commit_bytes(prefix).unwrap();
    let commit_token_accepts = match catch_unwind(AssertUnwindSafe(|| commit_token_state.commit_token(token_id))) {
        Ok(Ok(())) => true,
        Ok(Err(_)) => false,
        Err(_) => true,
    };

    let mut commit_bytes_state = constraint.start();
    commit_bytes_state.commit_bytes(prefix).unwrap();
    let commit_bytes_accepts = commit_bytes_state.commit_bytes(token_bytes).is_ok();

    (mask_accepts, commit_token_accepts, commit_bytes_accepts)
}

fn decode_hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => 10 + (byte - b'a'),
        b'A'..=b'F' => 10 + (byte - b'A'),
        _ => panic!("invalid hex nibble: {byte}"),
    }
}

fn decode_hex_bytes(value: &str) -> Vec<u8> {
    let bytes = value.as_bytes();
    assert_eq!(bytes.len() % 2, 0, "hex string must have even length");
    bytes
        .chunks_exact(2)
        .map(|chunk| (decode_hex_nibble(chunk[0]) << 4) | decode_hex_nibble(chunk[1]))
        .collect()
}

fn load_llama3_full_vocab() -> Vocab {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".cache/vocab_cache/llama3_vocab.json");
    let raw = fs::read_to_string(&path).unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    let json: Value = serde_json::from_str(&raw).expect("llama3 vocab json should parse");
    let object = json.as_object().expect("llama3 vocab json should be an object");

    let mut entries = object
        .iter()
        .map(|(id, hex)| {
            let token_id = id.parse::<u32>().expect("token id should parse");
            let hex = hex.as_str().expect("token bytes should be a hex string");
            (token_id, decode_hex_bytes(hex))
        })
        .collect::<Vec<_>>();
    entries.sort_by_key(|(token_id, _)| *token_id);
    Vocab::new(entries, None)
}

fn single_token_vocab() -> Vocab {
    Vocab::new(vec![(0, DISPUTED_TOKEN_BYTES.to_vec())], None)
}

fn reduced_two_token_vocab() -> Vocab {
    Vocab::new(
        vec![
            (DISPUTED_TOKEN_ID, DISPUTED_TOKEN_BYTES.to_vec()),
            (CONTROL_TOKEN_ID, CONTROL_TOKEN_BYTES.to_vec()),
        ],
        None,
    )
}

fn exact_benchmark_schema() -> &'static str {
    r##"{
        "$schema": "http://json-schema.org/draft-04/schema#",
        "additionalProperties": true,
        "id": "http://schemas.triniti.io/json-schema/triniti/canvas/mixin/vimeo-video-block/1-0-0.json#",
        "properties": {
            "aside": {
                "description": "When true it means this block represents a portion of a document whose content is only indirectly related to the document's main content. Asides are frequently presented as sidebars or call-out boxes.",
                "pbj": { "rule": "single", "type": "boolean" },
                "type": "boolean"
            },
            "autoplay": {
                "pbj": { "rule": "single", "type": "boolean" },
                "type": "boolean"
            },
            "css_class": {
                "description": "In rendering environments that support HTML the css_class can be appended to the dom elements' class attribute.",
                "pattern": "^[\\w\\s-]+$",
                "pbj": { "rule": "single", "type": "string" },
                "type": "string"
            },
            "description": {
                "maxLength": 5000,
                "minLength": 0,
                "pbj": { "rule": "single", "type": "text" },
                "type": "string"
            },
            "etag": {
                "pattern": "^[\\w\\.:-]+$",
                "pbj": { "rule": "single", "type": "string" },
                "type": "string"
            },
            "id": {
                "pattern": "^\\d+$",
                "pbj": { "rule": "single", "type": "string" },
                "type": "string"
            },
            "loop": {
                "pbj": { "rule": "single", "type": "boolean" },
                "type": "boolean"
            },
            "poster_image_ref": {
                "description": "A reference to an image asset to use as the poster that will override what is provided by vimeo.",
                "pattern": "^[\\w\\/\\.:-]+$",
                "pbj": { "rule": "single", "type": "identifier" },
                "type": "string"
            },
            "show_byline": {
                "description": "Whether or not to show the byline (eg \"from Dick Tracy\") in the thumbnail.",
                "pbj": { "rule": "single", "type": "boolean" },
                "type": "boolean"
            },
            "show_portrait": {
                "description": "Whether or not to show the portrait (profile image) in the thumbnail.",
                "pbj": { "rule": "single", "type": "boolean" },
                "type": "boolean"
            },
            "show_title": {
                "description": "Whether or not to show the video title in the thumbnail.",
                "pbj": { "rule": "single", "type": "boolean" },
                "type": "boolean"
            },
            "title": {
                "maxLength": 255,
                "minLength": 0,
                "pbj": { "rule": "single", "type": "string" },
                "type": "string"
            },
            "updated_date": {
                "_format": "date-time",
                "description": "Represents an update that occurred on the node this block is attached to. DOES NOT indicate an update to the block itself. eg an article with a twitter block with updated_date means that the article was updated to include that twitter block.",
                "pbj": { "rule": "single", "type": "date-time" },
                "type": "string"
            },
            "user_id": {
                "pattern": "^[\\w\\.-]+$",
                "pbj": { "rule": "single", "type": "string" },
                "type": "string"
            },
            "user_name": {
                "pattern": "^[\\s\\w\\.-]+$",
                "pbj": { "rule": "single", "type": "string" },
                "type": "string"
            }
        },
        "required": ["id"],
        "type": "object"
    }"##
}

fn sparse_current_schema() -> &'static str {
    r##"{
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
        "required": ["id"],
        "additionalProperties": true
    }"##
}

fn benchmark_prefix_at_discrepancy() -> Vec<u8> {
    let mut prefix = Vec::from(
        b"{\"aside\": true, \"autoplay\": false, \"css_class\": \"vimeo-video-block\", \"description\": \"".as_slice(),
    );
    prefix.extend(std::iter::repeat(b"This is a Vimeo video block. ".as_slice()).take(79).flatten().copied());
    prefix.extend_from_slice(b"This is a");
    prefix
}

fn description_only_prefix() -> Vec<u8> {
    description_only_prefix_with_ascii_repeat(2300)
}

fn description_only_prefix_with_ascii_repeat(content_len: usize) -> Vec<u8> {
    let mut prefix = Vec::from(b"{\"description\": \"".as_slice());
    prefix.extend(std::iter::repeat(b'a').take(content_len));
    prefix
}

fn aside_description_prefix() -> Vec<u8> {
    let mut prefix = Vec::from(b"{\"aside\": true, \"description\": \"".as_slice());
    prefix.extend(std::iter::repeat(b"This is a Vimeo video block. ".as_slice()).take(79).flatten().copied());
    prefix.extend_from_slice(b"This is a");
    prefix
}

fn fixed_order_subset_prefix(include_aside: bool, include_autoplay: bool, include_css_class: bool) -> Vec<u8> {
    let mut prefix = b"{".to_vec();
    let mut first = true;
    if include_aside {
        prefix.extend_from_slice(if first { b"\"aside\": true" } else { b", \"aside\": true" });
        first = false;
    }
    if include_autoplay {
        prefix.extend_from_slice(if first { b"\"autoplay\": false" } else { b", \"autoplay\": false" });
        first = false;
    }
    if include_css_class {
        prefix.extend_from_slice(
            if first {
                b"\"css_class\": \"vimeo-video-block\""
            } else {
                b", \"css_class\": \"vimeo-video-block\""
            },
        );
        first = false;
    }
    prefix.extend_from_slice(if first { b"\"description\": \"" } else { b", \"description\": \"" });
    prefix.extend(std::iter::repeat(b"This is a Vimeo video block. ".as_slice()).take(79).flatten().copied());
    prefix.extend_from_slice(b"This is a");
    prefix
}

fn fixed_order_subset_glrm(include_aside: bool, include_autoplay: bool, include_css_class: bool) -> String {
    let mut pieces = Vec::new();
    if include_aside {
        pieces.push("(\"\\\"\" \"aside\\\"\" \": \") JSON_BOOL".to_string());
    }
    if include_autoplay {
        pieces.push("(\"\\\"\" \"autoplay\\\"\" \": \") JSON_BOOL".to_string());
    }
    if include_css_class {
        pieces.push("(\"\\\"\" \"css_class\\\"\" \": \") (\"\\\"\" JSON_STRING_PATTERN_FULLMATCH_0)".to_string());
    }
    pieces.push("(\"\\\"\" \"description\\\"\" \": \") json_string_bounded_split_6".to_string());
    pieces.push("(\"\\\"\" \"id\\\"\" \": \") json_string".to_string());

    let mut body = String::new();
    for (index, piece) in pieces.iter().enumerate() {
        if index > 0 {
            body.push_str(" \", \" ");
        }
        body.push_str(piece);
    }

    format!(
        r#"
start start;

internal t JSON_STRING_CHAR ::= /[^\x00-\x1f\x7f"\\]|\\["\\\/bfnrt]|\\u[0-9A-Fa-f]{{4}}/;
t JSON_STRING_BODY ::= JSON_STRING_CHAR* "\"";
nt json_string ::= "\"" JSON_STRING_BODY;
t JSON_BOOL ::= "true" | "false";
t JSON_STRING_PATTERN_FULLMATCH_0 ::= ([ \-0-9A-Z_a-z] | "\xC2" "\x85" | "\xC2" "\xA0" | "\\" "t" | "\\" "u" "0" "0" "0" "9" | "\\" "n" | "\\" "u" "0" "0" "0" [Aa] | "\\" "u" "0" "0" "0" [Bb] | "\\" "f" | "\\" "u" "0" "0" "0" [Cc] | "\\" "r" | "\\" "u" "0" "0" "0" [Dd])+ "\"";
internal t JSON_STRING_CHAR_UPTO_256_1 ::= JSON_STRING_CHAR{{0,256}};
t JSON_STRING_CHAR_UPTO_CLOSE_2 ::= JSON_STRING_CHAR_UPTO_256_1 "\"";
t JSON_STRING_CHAR_EXACT_256_3 ::= JSON_STRING_CHAR{{256}};
internal t JSON_STRING_CHAR_UPTO_136_4 ::= JSON_STRING_CHAR{{0,136}};
t JSON_STRING_CHAR_UPTO_CLOSE_5 ::= JSON_STRING_CHAR_UPTO_136_4 "\"";
nt json_string_bounded_split_6 ::= "\"" (JSON_STRING_CHAR_EXACT_256_3{{0,18}} JSON_STRING_CHAR_UPTO_CLOSE_2 | JSON_STRING_CHAR_EXACT_256_3{{19}} JSON_STRING_CHAR_UPTO_CLOSE_5);
nt start ::= "{{" {body} "}}";
"#,
    )
}

fn recursive_known_subset_glrm(
    include_aside: bool,
    include_autoplay: bool,
    include_css_class: bool,
    css_class_uses_pattern: bool,
) -> String {
    let mut known_branches = Vec::new();
    if include_aside {
        known_branches.push("((\"\\\"\" \"aside\\\"\" \": \") JSON_BOOL)".to_string());
    }
    if include_autoplay {
        known_branches.push("((\"\\\"\" \"autoplay\\\"\" \": \") JSON_BOOL)".to_string());
    }
    if include_css_class {
        let value = if css_class_uses_pattern {
            "(\"\\\"\" JSON_STRING_PATTERN_FULLMATCH_0)"
        } else {
            "json_string"
        };
        known_branches.push(format!("((\"\\\"\" \"css_class\\\"\" \": \") {value})"));
    }
    known_branches.push("((\"\\\"\" \"description\\\"\" \": \") json_string_bounded_split_6)".to_string());

    let nc_known = known_branches
        .iter()
        .map(|branch| format!("{branch} obj_open_reqmask_0_c_0"))
        .collect::<Vec<_>>()
        .join(" | ");
    let c0_known = known_branches
        .iter()
        .map(|branch| format!("\", \" {branch} obj_open_reqmask_0_c_0"))
        .collect::<Vec<_>>()
        .join(" | ");
    let c1_known = known_branches
        .iter()
        .map(|branch| format!("\", \" {branch} obj_open_reqmask_0_c_1"))
        .collect::<Vec<_>>()
        .join(" | ");

    format!(
        r#"
start start;

internal t JSON_STRING_CHAR ::= /[^\x00-\x1f\x7f"\\]|\\["\\\/bfnrt]|\\u[0-9A-Fa-f]{{4}}/;
t JSON_STRING_BODY ::= JSON_STRING_CHAR* "\"";
nt json_string ::= "\"" JSON_STRING_BODY;
t JSON_BOOL ::= "true" | "false";
t JSON_STRING_PATTERN_FULLMATCH_0 ::= ([ \-0-9A-Z_a-z] | "\xC2" "\x85" | "\xC2" "\xA0" | "\\" "t" | "\\" "u" "0" "0" "0" "9" | "\\" "n" | "\\" "u" "0" "0" "0" [Aa] | "\\" "u" "0" "0" "0" [Bb] | "\\" "f" | "\\" "u" "0" "0" "0" [Cc] | "\\" "r" | "\\" "u" "0" "0" "0" [Dd])+ "\"";
internal t JSON_STRING_CHAR_UPTO_256_1 ::= JSON_STRING_CHAR{{0,256}};
t JSON_STRING_CHAR_UPTO_CLOSE_2 ::= JSON_STRING_CHAR_UPTO_256_1 "\"";
t JSON_STRING_CHAR_EXACT_256_3 ::= JSON_STRING_CHAR{{256}};
internal t JSON_STRING_CHAR_UPTO_136_4 ::= JSON_STRING_CHAR{{0,136}};
t JSON_STRING_CHAR_UPTO_CLOSE_5 ::= JSON_STRING_CHAR_UPTO_136_4 "\"";
nt json_string_bounded_split_6 ::= "\"" (JSON_STRING_CHAR_EXACT_256_3{{0,18}} JSON_STRING_CHAR_UPTO_CLOSE_2 | JSON_STRING_CHAR_EXACT_256_3{{19}} JSON_STRING_CHAR_UPTO_CLOSE_5);
nt obj_open_reqmask_0_nc_0 ::= {nc_known} | (("\"" "id\"" ": ") json_string) obj_open_reqmask_0_c_1;
nt obj_open_reqmask_0_c_0 ::= {c0_known} | ", " (("\"" "id\"" ": ") json_string) obj_open_reqmask_0_c_1;
nt obj_open_reqmask_0_c_1 ::= {c1_known} | ;
nt start ::= "{{" obj_open_reqmask_0_nc_0 "}}";
"#,
    )
}

fn desc_id_reqmask_variant(
    start_allows_id_first: bool,
    pre_id_allows_desc_repeat: bool,
    post_id_allows_desc_repeat: bool,
) -> String {
    let start_desc = "((\"\\\"\" \"description\\\"\" \": \") json_string_bounded_split_6) obj_open_reqmask_0_c_0";
    let start_id = "((\"\\\"\" \"id\\\"\" \": \") json_string) obj_open_reqmask_0_c_1";

    let nc0 = if start_allows_id_first {
        format!("{start_desc} | {start_id}")
    } else {
        start_desc.to_string()
    };

    let c0 = if pre_id_allows_desc_repeat {
        format!("\", \" {start_desc} | \", \" {start_id}")
    } else {
        format!("\", \" {start_id}")
    };

    let c1 = if post_id_allows_desc_repeat {
        format!("\", \" {start_desc} | ")
    } else {
        String::from(";")
    };

    let c1_rule = if post_id_allows_desc_repeat {
        format!("nt obj_open_reqmask_0_c_1 ::= {c1};")
    } else {
        String::from("nt obj_open_reqmask_0_c_1 ::= ;")
    };

    format!(
        r#"
start start;

internal t JSON_STRING_CHAR ::= /[^\x00-\x1f\x7f"\\]|\\["\\\/bfnrt]|\\u[0-9A-Fa-f]{{4}}/;
t JSON_STRING_BODY ::= JSON_STRING_CHAR* "\"";
nt json_string ::= "\"" JSON_STRING_BODY;
internal t JSON_STRING_CHAR_UPTO_256_1 ::= JSON_STRING_CHAR{{0,256}};
t JSON_STRING_CHAR_UPTO_CLOSE_2 ::= JSON_STRING_CHAR_UPTO_256_1 "\"";
t JSON_STRING_CHAR_EXACT_256_3 ::= JSON_STRING_CHAR{{256}};
internal t JSON_STRING_CHAR_UPTO_136_4 ::= JSON_STRING_CHAR{{0,136}};
t JSON_STRING_CHAR_UPTO_CLOSE_5 ::= JSON_STRING_CHAR_UPTO_136_4 "\"";
nt json_string_bounded_split_6 ::= "\"" (JSON_STRING_CHAR_EXACT_256_3{{0,18}} JSON_STRING_CHAR_UPTO_CLOSE_2 | JSON_STRING_CHAR_EXACT_256_3{{19}} JSON_STRING_CHAR_UPTO_CLOSE_5);
nt obj_open_reqmask_0_nc_0 ::= {nc0};
nt obj_open_reqmask_0_c_0 ::= {c0};
{c1_rule}
nt start ::= "{{" obj_open_reqmask_0_nc_0 "}}";
"#,
    )
}

#[ignore = "diagnostic for more aggressive recursive o82710 inline-GLRM minimization"]
#[test]
fn scan_o82710_inline_glrm_more_minimal_variants() {
    let vocab = reduced_two_token_vocab();

    for (label, grammar, prefix) in [
        (
            "description_id_only",
            r#"
start start;

internal t JSON_STRING_CHAR ::= /[^\x00-\x1f\x7f"\\]|\\["\\\/bfnrt]|\\u[0-9A-Fa-f]{4}/;
t JSON_STRING_BODY ::= JSON_STRING_CHAR* "\"";
nt json_string ::= "\"" JSON_STRING_BODY;
internal t JSON_STRING_CHAR_UPTO_256_1 ::= JSON_STRING_CHAR{0,256};
t JSON_STRING_CHAR_UPTO_CLOSE_2 ::= JSON_STRING_CHAR_UPTO_256_1 "\"";
t JSON_STRING_CHAR_EXACT_256_3 ::= JSON_STRING_CHAR{256};
internal t JSON_STRING_CHAR_UPTO_136_4 ::= JSON_STRING_CHAR{0,136};
t JSON_STRING_CHAR_UPTO_CLOSE_5 ::= JSON_STRING_CHAR_UPTO_136_4 "\"";
nt json_string_bounded_split_6 ::= "\"" (JSON_STRING_CHAR_EXACT_256_3{0,18} JSON_STRING_CHAR_UPTO_CLOSE_2 | JSON_STRING_CHAR_EXACT_256_3{19} JSON_STRING_CHAR_UPTO_CLOSE_5);
nt start ::= "{" "\"description\": " json_string_bounded_split_6 ", " "\"id\": " json_string "}";
"#,
            description_only_prefix(),
        ),
        (
            "aside_description_id",
            r#"
start start;

internal t JSON_STRING_CHAR ::= /[^\x00-\x1f\x7f"\\]|\\["\\\/bfnrt]|\\u[0-9A-Fa-f]{4}/;
t JSON_STRING_BODY ::= JSON_STRING_CHAR* "\"";
nt json_string ::= "\"" JSON_STRING_BODY;
t JSON_BOOL ::= "true" | "false";
internal t JSON_STRING_CHAR_UPTO_256_1 ::= JSON_STRING_CHAR{0,256};
t JSON_STRING_CHAR_UPTO_CLOSE_2 ::= JSON_STRING_CHAR_UPTO_256_1 "\"";
t JSON_STRING_CHAR_EXACT_256_3 ::= JSON_STRING_CHAR{256};
internal t JSON_STRING_CHAR_UPTO_136_4 ::= JSON_STRING_CHAR{0,136};
t JSON_STRING_CHAR_UPTO_CLOSE_5 ::= JSON_STRING_CHAR_UPTO_136_4 "\"";
nt json_string_bounded_split_6 ::= "\"" (JSON_STRING_CHAR_EXACT_256_3{0,18} JSON_STRING_CHAR_UPTO_CLOSE_2 | JSON_STRING_CHAR_EXACT_256_3{19} JSON_STRING_CHAR_UPTO_CLOSE_5);
nt start ::= "{" "\"aside\": " JSON_BOOL ", " "\"description\": " json_string_bounded_split_6 ", " "\"id\": " json_string "}";
"#,
            aside_description_prefix(),
        ),
        (
            "no_shared_branch",
            r#"
start start;

internal t JSON_STRING_CHAR ::= /[^\x00-\x1f\x7f"\\]|\\["\\\/bfnrt]|\\u[0-9A-Fa-f]{4}/;
t JSON_STRING_BODY ::= JSON_STRING_CHAR* "\"";
nt json_string ::= "\"" JSON_STRING_BODY;
t JSON_BOOL ::= "true" | "false";
t JSON_STRING_PATTERN_FULLMATCH_0 ::= ([ \-0-9A-Z_a-z] | "\xC2" "\x85" | "\xC2" "\xA0" | "\\" "t" | "\\" "u" "0" "0" "0" "9" | "\\" "n" | "\\" "u" "0" "0" "0" [Aa] | "\\" "u" "0" "0" "0" [Bb] | "\\" "f" | "\\" "u" "0" "0" "0" [Cc] | "\\" "r" | "\\" "u" "0" "0" "0" [Dd])+ "\"";
internal t JSON_STRING_CHAR_UPTO_256_1 ::= JSON_STRING_CHAR{0,256};
t JSON_STRING_CHAR_UPTO_CLOSE_2 ::= JSON_STRING_CHAR_UPTO_256_1 "\"";
t JSON_STRING_CHAR_EXACT_256_3 ::= JSON_STRING_CHAR{256};
internal t JSON_STRING_CHAR_UPTO_136_4 ::= JSON_STRING_CHAR{0,136};
t JSON_STRING_CHAR_UPTO_CLOSE_5 ::= JSON_STRING_CHAR_UPTO_136_4 "\"";
nt json_string_bounded_split_6 ::= "\"" (JSON_STRING_CHAR_EXACT_256_3{0,18} JSON_STRING_CHAR_UPTO_CLOSE_2 | JSON_STRING_CHAR_EXACT_256_3{19} JSON_STRING_CHAR_UPTO_CLOSE_5);
nt obj_open_reqmask_0_nc_0 ::= (("\"" "aside\"" ": ") JSON_BOOL) obj_open_reqmask_0_c_0 | (("\"" "autoplay\"" ": ") JSON_BOOL) obj_open_reqmask_0_c_0 | (("\"" "css_class\"" ": ") ("\"" JSON_STRING_PATTERN_FULLMATCH_0)) obj_open_reqmask_0_c_0 | (("\"" "description\"" ": ") json_string_bounded_split_6) obj_open_reqmask_0_c_0 | (("\"" "id\"" ": ") json_string) obj_open_reqmask_0_c_1;
nt obj_open_reqmask_0_c_0 ::= ", " (("\"" "aside\"" ": ") JSON_BOOL) obj_open_reqmask_0_c_0 | ", " (("\"" "autoplay\"" ": ") JSON_BOOL) obj_open_reqmask_0_c_0 | ", " (("\"" "css_class\"" ": ") ("\"" JSON_STRING_PATTERN_FULLMATCH_0)) obj_open_reqmask_0_c_0 | ", " (("\"" "description\"" ": ") json_string_bounded_split_6) obj_open_reqmask_0_c_0 | ", " (("\"" "id\"" ": ") json_string) obj_open_reqmask_0_c_1;
nt obj_open_reqmask_0_c_1 ::= ", " (("\"" "aside\"" ": ") JSON_BOOL) obj_open_reqmask_0_c_1 | ", " (("\"" "autoplay\"" ": ") JSON_BOOL) obj_open_reqmask_0_c_1 | ", " (("\"" "css_class\"" ": ") ("\"" JSON_STRING_PATTERN_FULLMATCH_0)) obj_open_reqmask_0_c_1 | ", " (("\"" "description\"" ": ") json_string_bounded_split_6) obj_open_reqmask_0_c_1 | ;
nt start ::= "{" obj_open_reqmask_0_nc_0 "}";
"#,
            benchmark_prefix_at_discrepancy(),
        ),
        (
            "no_post_id_knowns",
            r#"
start start;

internal t JSON_STRING_CHAR ::= /[^\x00-\x1f\x7f"\\]|\\["\\\/bfnrt]|\\u[0-9A-Fa-f]{4}/;
t JSON_STRING_BODY ::= JSON_STRING_CHAR* "\"";
nt json_string ::= "\"" JSON_STRING_BODY;
t JSON_BOOL ::= "true" | "false";
t JSON_STRING_PATTERN_FULLMATCH_0 ::= ([ \-0-9A-Z_a-z] | "\xC2" "\x85" | "\xC2" "\xA0" | "\\" "t" | "\\" "u" "0" "0" "0" "9" | "\\" "n" | "\\" "u" "0" "0" "0" [Aa] | "\\" "u" "0" "0" "0" [Bb] | "\\" "f" | "\\" "u" "0" "0" "0" [Cc] | "\\" "r" | "\\" "u" "0" "0" "0" [Dd])+ "\"";
internal t JSON_STRING_CHAR_UPTO_256_1 ::= JSON_STRING_CHAR{0,256};
t JSON_STRING_CHAR_UPTO_CLOSE_2 ::= JSON_STRING_CHAR_UPTO_256_1 "\"";
t JSON_STRING_CHAR_EXACT_256_3 ::= JSON_STRING_CHAR{256};
internal t JSON_STRING_CHAR_UPTO_136_4 ::= JSON_STRING_CHAR{0,136};
t JSON_STRING_CHAR_UPTO_CLOSE_5 ::= JSON_STRING_CHAR_UPTO_136_4 "\"";
nt json_string_bounded_split_6 ::= "\"" (JSON_STRING_CHAR_EXACT_256_3{0,18} JSON_STRING_CHAR_UPTO_CLOSE_2 | JSON_STRING_CHAR_EXACT_256_3{19} JSON_STRING_CHAR_UPTO_CLOSE_5);
internal t AP_SHARED_KEY_COLON_7 ::= "aside\"" | "autoplay\"" | "css_class\"" | "description\"" | "id\"";
internal t AP_SHARED_KEY_COLON_8 ::= ((([ !#-[\]-~] | [\xC2-\xDF] [\x80-\xBF] | [\xE0] [\xA0-\xBF] [\x80-\xBF] | [\xE1-\xEC] [\x80-\xBF] [\x80-\xBF] | [\xED] [\x80-\x9F] [\x80-\xBF] | [\xEE\xEF] [\x80-\xBF] [\x80-\xBF] | [\xF0] [\x90-\xBF] [\x80-\xBF] [\x80-\xBF] | [\xF1-\xF3] [\x80-\xBF] [\x80-\xBF] [\x80-\xBF] | [\xF4] [\x80-\x8F] [\x80-\xBF] [\x80-\xBF]) | "\\" ["/\\bfnrt] | "\\" "u" [0-9A-Fa-f]{4})* "\"");
t AP_SHARED_KEY_COLON_9 ::= AP_SHARED_KEY_COLON_8 - AP_SHARED_KEY_COLON_7;
nt json_value ::= json_string | JSON_BOOL;
nt obj_open_reqmask_0_nc_0 ::= (("\"" "aside\"" ": ") JSON_BOOL) obj_open_reqmask_0_c_0 | (("\"" "autoplay\"" ": ") JSON_BOOL) obj_open_reqmask_0_c_0 | (("\"" "css_class\"" ": ") ("\"" JSON_STRING_PATTERN_FULLMATCH_0)) obj_open_reqmask_0_c_0 | (("\"" "description\"" ": ") json_string_bounded_split_6) obj_open_reqmask_0_c_0 | (("\"" AP_SHARED_KEY_COLON_9 ": ") json_value) obj_open_reqmask_0_c_0 | (("\"" "id\"" ": ") json_string) obj_open_reqmask_0_c_1;
nt obj_open_reqmask_0_c_0 ::= ", " (("\"" "aside\"" ": ") JSON_BOOL) obj_open_reqmask_0_c_0 | ", " (("\"" "autoplay\"" ": ") JSON_BOOL) obj_open_reqmask_0_c_0 | ", " (("\"" "css_class\"" ": ") ("\"" JSON_STRING_PATTERN_FULLMATCH_0)) obj_open_reqmask_0_c_0 | ", " (("\"" "description\"" ": ") json_string_bounded_split_6) obj_open_reqmask_0_c_0 | ", " (("\"" AP_SHARED_KEY_COLON_9 ": ") json_value) obj_open_reqmask_0_c_0 | ", " (("\"" "id\"" ": ") json_string) obj_open_reqmask_0_c_1;
nt obj_open_reqmask_0_c_1 ::= ;
nt start ::= "{" obj_open_reqmask_0_nc_0 "}";
"#,
            benchmark_prefix_at_discrepancy(),
        ),
    ] {
        let Ok(constraint) = Constraint::from_glrm_grammar(grammar, &vocab) else {
            println!("variant={label} compile=false");
            continue;
        };
        let (mask_accepts, commit_token_accepts, commit_bytes_accepts) =
            classify_constraint(&constraint, &prefix, DISPUTED_TOKEN_ID, DISPUTED_TOKEN_BYTES);
        println!(
            "variant={label} mask={} commit_token={} commit_bytes={}",
            mask_accepts,
            commit_token_accepts,
            commit_bytes_accepts,
        );
    }
}

#[ignore = "diagnostic for fixed-order subset minimization of the current o82710 inline GLRM"]
#[test]
fn scan_o82710_fixed_order_field_subsets() {
    let vocab = reduced_two_token_vocab();

    for mask in 0u8..8 {
        let include_aside = (mask & 0b001) != 0;
        let include_autoplay = (mask & 0b010) != 0;
        let include_css_class = (mask & 0b100) != 0;
        let label = format!(
            "aside={} autoplay={} css_class={}",
            include_aside, include_autoplay, include_css_class
        );
        let grammar = fixed_order_subset_glrm(include_aside, include_autoplay, include_css_class);
        let prefix = fixed_order_subset_prefix(include_aside, include_autoplay, include_css_class);
        let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
        let (mask_accepts, commit_token_accepts, commit_bytes_accepts) =
            classify_constraint(&constraint, &prefix, DISPUTED_TOKEN_ID, DISPUTED_TOKEN_BYTES);
        println!(
            "fixed_order_subset={} mask={} commit_token={} commit_bytes={}",
            label,
            mask_accepts,
            commit_token_accepts,
            commit_bytes_accepts,
        );
    }
}

#[ignore = "diagnostic for recursive known-field subset minimization of the current o82710 inline GLRM"]
#[test]
fn scan_o82710_recursive_known_field_subsets() {
    let vocab = reduced_two_token_vocab();

    for mask in 0u8..8 {
        let include_aside = (mask & 0b001) != 0;
        let include_autoplay = (mask & 0b010) != 0;
        let include_css_class = (mask & 0b100) != 0;
        for css_class_uses_pattern in [true, false] {
            let label = format!(
                "aside={} autoplay={} css_class={} css_pattern={}",
                include_aside, include_autoplay, include_css_class, css_class_uses_pattern
            );
            let grammar = recursive_known_subset_glrm(
                include_aside,
                include_autoplay,
                include_css_class,
                css_class_uses_pattern,
            );
            let prefix = fixed_order_subset_prefix(include_aside, include_autoplay, include_css_class);
            let Ok(constraint) = Constraint::from_glrm_grammar(&grammar, &vocab) else {
                println!("recursive_subset={} compile=false", label);
                continue;
            };
            let (mask_accepts, commit_token_accepts, commit_bytes_accepts) =
                classify_constraint(&constraint, &prefix, DISPUTED_TOKEN_ID, DISPUTED_TOKEN_BYTES);
            println!(
                "recursive_subset={} mask={} commit_token={} commit_bytes={}",
                label,
                mask_accepts,
                commit_token_accepts,
                commit_bytes_accepts,
            );
        }
    }
}

#[ignore = "diagnostic for desc/id reqmask structure minimization"]
#[test]
fn scan_o82710_desc_id_reqmask_structure() {
    let vocab = reduced_two_token_vocab();
    let prefix = description_only_prefix();

    for start_allows_id_first in [false, true] {
        for pre_id_allows_desc_repeat in [false, true] {
            for post_id_allows_desc_repeat in [false, true] {
                let grammar = desc_id_reqmask_variant(
                    start_allows_id_first,
                    pre_id_allows_desc_repeat,
                    post_id_allows_desc_repeat,
                );
                let label = format!(
                    "start_id_first={} pre_desc_repeat={} post_desc_repeat={}",
                    start_allows_id_first,
                    pre_id_allows_desc_repeat,
                    post_id_allows_desc_repeat,
                );
                let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
                let (mask_accepts, commit_token_accepts, commit_bytes_accepts) =
                    classify_constraint(&constraint, &prefix, DISPUTED_TOKEN_ID, DISPUTED_TOKEN_BYTES);
                println!(
                    "desc_id_variant={} mask={} commit_token={} commit_bytes={}",
                    label,
                    mask_accepts,
                    commit_token_accepts,
                    commit_bytes_accepts,
                );
            }
        }
    }
}

#[ignore = "diagnostic for desc-only prefix length minimization against the current recursive-core witness"]
#[test]
fn scan_o82710_description_only_ascii_prefix_lengths() {
    let vocab = reduced_two_token_vocab();
    let constraint = Constraint::from_glrm_grammar(MINIMIZED_INLINE_GLRM_CANDIDATE, &vocab).unwrap();
    let token_content_len = DISPUTED_TOKEN_BYTES.len() - 1;

    for content_len in [252usize, 508, 764, 1020, 1276, 1532, 1788, 2044, 2300] {
        let prefix = description_only_prefix_with_ascii_repeat(content_len);
        let (mask_accepts, commit_token_accepts, commit_bytes_accepts) =
            classify_constraint(&constraint, &prefix, DISPUTED_TOKEN_ID, DISPUTED_TOKEN_BYTES);
        println!(
            "desc_ascii_len={} close_len={} close_mod_256={} mask={} commit_token={} commit_bytes={}",
            content_len,
            content_len + token_content_len,
            (content_len + token_content_len) % 256,
            mask_accepts,
            commit_token_accepts,
            commit_bytes_accepts,
        );
    }
}

#[ignore = "expensive full-vocab benchmark witness for the current false negative"]
#[test]
fn test_o82710_exact_schema_full_llama3_vocab_false_negative() {
    let vocab = load_llama3_full_vocab();
    let constraint = Constraint::from_json_schema(exact_benchmark_schema(), &vocab).unwrap();
    let prefix = benchmark_prefix_at_discrepancy();

    let (mask_accepts, commit_token_accepts, commit_bytes_accepts) =
        classify_constraint(&constraint, &prefix, DISPUTED_TOKEN_ID, DISPUTED_TOKEN_BYTES);

    assert!(
        !mask_accepts && commit_token_accepts && commit_bytes_accepts,
        "expected current benchmark witness to remain a full-vocab false negative; got mask={mask_accepts} commit_token={commit_token_accepts} commit_bytes={commit_bytes_accepts}",
    );
}

#[ignore = "current discrepancy MRE stage 1: exact schema with single-token vocab"]
#[test]
fn test_o82710_exact_schema_single_token_vocab_does_not_reproduce() {
    let vocab = single_token_vocab();
    let constraint = Constraint::from_json_schema(exact_benchmark_schema(), &vocab).unwrap();
    let prefix = benchmark_prefix_at_discrepancy();

    let (mask_accepts, commit_token_accepts, commit_bytes_accepts) =
        classify_constraint(&constraint, &prefix, 0, DISPUTED_TOKEN_BYTES);

    assert!(
        mask_accepts && commit_token_accepts && commit_bytes_accepts,
        "expected the single-token reduction to eliminate the false negative; got mask={mask_accepts} commit_token={commit_token_accepts} commit_bytes={commit_bytes_accepts}",
    );
}

#[ignore = "current discrepancy MRE stage 2: exact schema with reduced two-token vocab"]
#[test]
fn test_o82710_exact_schema_two_token_vocab_false_negative() {
    let vocab = reduced_two_token_vocab();
    let constraint = Constraint::from_json_schema(exact_benchmark_schema(), &vocab).unwrap();
    let prefix = benchmark_prefix_at_discrepancy();

    let (mask_accepts, commit_token_accepts, commit_bytes_accepts) =
        classify_constraint(&constraint, &prefix, DISPUTED_TOKEN_ID, DISPUTED_TOKEN_BYTES);

    assert!(
        !mask_accepts && commit_token_accepts && commit_bytes_accepts,
        "expected exact-schema two-token witness to remain a false negative; got mask={mask_accepts} commit_token={commit_token_accepts} commit_bytes={commit_bytes_accepts}",
    );
}

#[ignore = "current discrepancy MRE stage 3: simplified schema with reduced two-token vocab"]
#[test]
fn test_o82710_sparse_schema_two_token_vocab_false_negative() {
    let vocab = reduced_two_token_vocab();
    let constraint = Constraint::from_json_schema(sparse_current_schema(), &vocab).unwrap();
    let prefix = benchmark_prefix_at_discrepancy();

    let (mask_accepts, commit_token_accepts, commit_bytes_accepts) =
        classify_constraint(&constraint, &prefix, DISPUTED_TOKEN_ID, DISPUTED_TOKEN_BYTES);

    assert!(
        !mask_accepts && commit_token_accepts && commit_bytes_accepts,
        "expected sparse-schema two-token witness to remain a false negative; got mask={mask_accepts} commit_token={commit_token_accepts} commit_bytes={commit_bytes_accepts}",
    );
}

#[ignore = "current discrepancy MRE stage 4: direct inline GLRM witness"]
#[test]
fn test_o82710_sparse_schema_inline_glrm_two_token_vocab_false_negative() {
    let vocab = reduced_two_token_vocab();
    let constraint = Constraint::from_glrm_grammar(SPARSE_SCHEMA_GLRM, &vocab).unwrap();
    let prefix = benchmark_prefix_at_discrepancy();

    let (mask_accepts, commit_token_accepts, commit_bytes_accepts) =
        classify_constraint(&constraint, &prefix, DISPUTED_TOKEN_ID, DISPUTED_TOKEN_BYTES);

    assert!(
        !mask_accepts && commit_token_accepts && commit_bytes_accepts,
        "expected inline-GLRM two-token witness to remain a false negative; got mask={mask_accepts} commit_token={commit_token_accepts} commit_bytes={commit_bytes_accepts}",
    );
}

#[ignore = "current discrepancy MRE stage 5: recursively minimized inline GLRM candidate"]
#[test]
fn test_o82710_minimized_inline_glrm_candidate_two_token_vocab_false_negative() {
    let vocab = reduced_two_token_vocab();
    let constraint = Constraint::from_glrm_grammar(MINIMIZED_INLINE_GLRM_CANDIDATE, &vocab).unwrap();
    let prefix = description_only_prefix();

    let (mask_accepts, commit_token_accepts, commit_bytes_accepts) =
        classify_constraint(&constraint, &prefix, DISPUTED_TOKEN_ID, DISPUTED_TOKEN_BYTES);

    assert!(
        !mask_accepts && commit_token_accepts && commit_bytes_accepts,
        "expected minimized inline-GLRM candidate to remain a false negative; got mask={mask_accepts} commit_token={commit_token_accepts} commit_bytes={commit_bytes_accepts}",
    );
}