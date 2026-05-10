use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use glrmask::{Constraint, Vocab};

const GITHUB_TRIVIAL_O70256_GLRM: &str = r#"start start;

internal t JSON_STRING_CHAR ::= /[^\x00-\x1f\x7f"\\]|\\["\\\/bfnrt]|\\u[0-9A-Fa-f]{4}/;
t JSON_STRING_BODY ::= JSON_STRING_CHAR* "\"";
internal t JSON_STRING_MIDDLE ::= JSON_STRING_CHAR*;
internal t JSON_STRING_MIDDLE_END ::= JSON_STRING_MIDDLE "\"";
nt json_string ::= "\"" JSON_STRING_BODY;
t JSON_INTEGER ::= /-?(0|[1-9][0-9]*)/;
t JSON_NUMBER ::= /-?(0|[1-9][0-9]*)(\.[0-9]+([eE][+-]?[0-9]+)?|[eE][+-]?[0-9]+)/;
internal t JSON_NONNEG_INTEGER ::= /(0|[1-9][0-9]*)/;
internal t JSON_NONNEG_NUMBER ::= /(0|[1-9][0-9]*)(\.[0-9]+([eE][+-]?[0-9]+)?|[eE][+-]?[0-9]+)/;
t JSON_BOOL ::= "true" | "false";
t JSON_NULL ::= "null";
t JSON_KEY_COLON_BODY ::= JSON_STRING_CHAR* "\"";
nt json_key_colon ::= "\"" JSON_KEY_COLON_BODY ": ";
nt json_kv ::= json_key_colon json_value;
nt json_object ::= "{" ", " ~ ( json_kv* ) "}";
nt json_array ::= "[" ", " ~ ( json_value* ) "]";
nt json_value ::= json_object | json_array | json_string | JSON_NUMBER | JSON_INTEGER | JSON_BOOL | JSON_NULL;
t JSON_ENUM_STRING_0 ::= "AD\"" | "AE\"" | "AF\"" | "AG\"" | "AI\"" | "AL\"" | "AM\"" | "AN\"" | "AO\"" | "AQ\"" | "AR\"" | "AS\"" | "AT\"" | "AU\"" | "AW\"" | "AZ\"" | "BA\"" | "BB\"" | "BD\"" | "BE\"" | "BF\"" | "BG\"" | "BH\"" | "BI\"" | "BJ\"" | "BM\"" | "BN\"" | "BO\"" | "BR\"" | "BS\"" | "BT\"" | "BV\"" | "BW\"" | "BY\"" | "BZ\"" | "CA\"" | "CC\"" | "CD\"" | "CF\"" | "CG\"" | "CH\"" | "CI\"" | "CK\"" | "CL\"" | "CM\"" | "CN\"" | "CO\"" | "CR\"" | "CU\"" | "CV\"" | "CX\"" | "CY\"" | "CZ\"" | "DE\"" | "DJ\"" | "DK\"" | "DM\"" | "DO\"" | "DZ\"" | "EC\"" | "EE\"" | "EG\"" | "EH\"" | "ER\"" | "ES\"" | "ET\"" | "FI\"" | "FJ\"" | "FK\"" | "FM\"" | "FO\"" | "FR\"" | "GA\"" | "GB\"" | "GD\"" | "GE\"" | "GF\"" | "GH\"" | "GI\"" | "GL\"" | "GM\"" | "GN\"" | "GP\"" | "GQ\"" | "GR\"" | "GS\"" | "GT\"" | "GU\"" | "GW\"" | "GY\"" | "HK\"" | "HM\"" | "HN\"" | "HR\"" | "HT\"" | "HU\"" | "ID\"" | "IE\"" | "IL\"" | "IN\"" | "IO\"" | "IQ\"" | "IR\"" | "IS\"" | "IT\"" | "JM\"" | "JO\"" | "JP\"" | "KE\"" | "KG\"" | "KH\"" | "KI\"" | "KM\"" | "KN\"" | "KP\"" | "KR\"" | "KW\"" | "KY\"" | "KZ\"" | "LA\"" | "LB\"" | "LC\"" | "LI\"" | "LK\"" | "LR\"" | "LS\"" | "LT\"" | "LU\"" | "LV\"" | "LY\"" | "MA\"" | "MC\"" | "MD\"" | "MG\"" | "MH\"" | "MK\"" | "ML\"" | "MM\"" | "MN\"" | "MO\"" | "MP\"" | "MQ\"" | "MR\"" | "MS\"" | "MT\"" | "MU\"" | "MV\"" | "MW\"" | "MX\"" | "MY\"" | "MZ\"" | "NA\"" | "NC\"" | "NE\"" | "NF\"" | "NG\"" | "NI\"" | "NL\"" | "NO\"" | "NP\"" | "NU\"" | "NZ\"" | "OM\"" | "PA\"" | "PE\"" | "PF\"" | "PG\"" | "PH\"" | "PK\"" | "PL\"" | "PM\"" | "PN\"" | "PR\"" | "PT\"" | "PW\"" | "PY\"" | "QA\"" | "RE\"" | "RO\"" | "RU\"" | "RW\"" | "SA\"" | "SB\"" | "SC\"" | "SD\"" | "SE\"" | "SG\"" | "SH\"" | "SI\"" | "SJ\"" | "SK\"" | "SL\"" | "SM\"" | "SN\"" | "SO\"" | "SR\"" | "ST\"" | "SV\"" | "SY\"" | "SZ\"" | "TC\"" | "TD\"" | "TF\"" | "TG\"" | "TH\"" | "TJ\"" | "TK\"" | "TM\"" | "TN\"" | "TO\"" | "TP\"" | "TR\"" | "TT\"" | "TV\"" | "TW\"" | "TZ\"" | "UA\"" | "UG\"" | "UM\"" | "US\"" | "UY\"" | "UZ\"" | "VE\"" | "VG\"" | "VI\"" | "VN\"" | "VU\"" | "WF\"" | "WS\"" | "YE\"" | "YT\"" | "YU\"" | "ZA\"" | "ZM\"" | "ZW\"";
nt start ::= "\"" JSON_ENUM_STRING_0;
"#;

fn llama3_vocab_path() -> PathBuf {
    std::env::var_os("GLRMASK_LLAMA3_VOCAB_JSON")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(".cache/vocab_cache/llama3_vocab.json"))
}

fn load_llama3_vocab() -> Vocab {
    let path = llama3_vocab_path();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed to read Llama 3 vocab from {}: {err}", path.display()));
    let id_to_hex: BTreeMap<u32, String> = serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("failed to parse Llama 3 vocab JSON from {}: {err}", path.display()));
    let entries = id_to_hex
        .into_iter()
        .map(|(token_id, hex)| {
            let bytes = hex_to_bytes(&hex)
                .unwrap_or_else(|err| panic!("invalid hex bytes for token {token_id} in {}: {err}", path.display()));
            (token_id, bytes)
        })
        .collect();
    Vocab::new(entries, None)
}

fn hex_to_bytes(hex: &str) -> Result<Vec<u8>, String> {
    if hex.len() % 2 != 0 {
        return Err(format!("odd hex length {}", hex.len()));
    }

    (0..hex.len())
        .step_by(2)
        .map(|idx| u8::from_str_radix(&hex[idx..idx + 2], 16).map_err(|err| err.to_string()))
        .collect()
}

fn assert_release_benchmark() {
    if cfg!(debug_assertions) {
        panic!("github_trivial_o70256_build must be run in release/bench mode, e.g. `cargo bench --bench github_trivial_o70256_build`");
    }
}

fn configure_benchmark_environment() {
    // Avoid Rayon scheduling noise unless the benchmark itself is edited.
    unsafe {
        std::env::set_var("GLRMASK_COMPILE_THREADS", "1");
        std::env::set_var("RAYON_NUM_THREADS", "1");
    }
}

fn run_one_profiled_build(vocab: &Vocab) {
    eprintln!("[bench][github_trivial_o70256_build] diagnostic_build=1 compile_profile=1");
    unsafe {
        std::env::set_var("GLRMASK_PROFILE_COMPILE", "1");
        std::env::set_var("GLRMASK_PROFILE_COMPILE_SUMMARY", "1");
    }
    let constraint = Constraint::from_glrm_grammar(GITHUB_TRIVIAL_O70256_GLRM, vocab)
        .expect("Github_trivial---o70256 GLRM grammar should compile");
    black_box(constraint);
    unsafe {
        std::env::remove_var("GLRMASK_PROFILE_COMPILE");
        std::env::remove_var("GLRMASK_PROFILE_COMPILE_SUMMARY");
    }
    eprintln!("[bench][github_trivial_o70256_build] diagnostic_build=done compile_profile=0");
}

fn bench_github_trivial_o70256_build(c: &mut Criterion) {
    assert_release_benchmark();
    configure_benchmark_environment();

    let vocab = load_llama3_vocab();
    assert_eq!(vocab.len(), 128_002, "expected the full Llama 3 vocabulary");
    eprintln!(
        "[bench][github_trivial_o70256_build] vocab_tokens={} rayon_threads=1 profile_once=1",
        vocab.len()
    );
    run_one_profiled_build(&vocab);

    c.bench_function("github_trivial_o70256_glrmask_build_llama3", |b| {
        b.iter(|| {
            let constraint = Constraint::from_glrm_grammar(black_box(GITHUB_TRIVIAL_O70256_GLRM), black_box(&vocab))
                .expect("Github_trivial---o70256 GLRM grammar should compile");
            black_box(constraint);
        });
    });
}

criterion_group!(benches, bench_github_trivial_o70256_build);
criterion_main!(benches);
