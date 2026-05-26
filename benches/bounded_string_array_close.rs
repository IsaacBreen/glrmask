#[path = "support/cfa_sweep.rs"]
mod cfa_sweep;

use criterion::{
    BatchSize,
    BenchmarkId,
    Criterion,
    black_box,
    criterion_group,
    criterion_main,
};
use glrmask::Constraint;

const ARRAY_CLOSE_BYTES: &[u8] = b"\"]";
const OBJECT_MRE_CLOSE_BYTES: &[u8] = b"\"],";

const PATTERN_ID_SCHEMA: &str = r#"{"type":"array","items":{"type":"string","pattern":"^[A-Fa-f\\d]{24}$"},"maxItems":1000}"#;
const PLAIN_STRING_SCHEMA: &str = r#"{"type":"array","items":{"type":"string"},"maxItems":1000}"#;
const LITERAL_X_ENUM_SCHEMA: &str = r#"{"type":"array","items":{"enum":["x"]},"maxItems":1000}"#;
const O9881_TWO_ARRAYS_GROUP_TAGS_SCHEMA: &str = r#"{"type":"object","properties":{"experienceUserIds":{"type":"array","items":{"type":"string","pattern":"^[A-Fa-f\\d]{24}$"},"maxItems":1000},"experienceEndpointIds":{"type":"array","items":{"type":"string","pattern":"^[A-Fa-f\\d]{24}$"},"maxItems":1000},"groupTags":{"type":"object"}}}"#;
const FULL_ITEM_O9881_SCHEMA: &str = r#"{"title":"Experience Group","description":"Schema for a single Experience Group","type":"object","properties":{"id":{"type":"string","pattern":"^[A-Fa-f\\d]{24}$"},"experienceGroupId":{"type":"string","pattern":"^[A-Fa-f\\d]{24}$"},"applicationId":{"type":"string","pattern":"^[A-Fa-f\\d]{24}$"},"creationDate":{"type":"string","format":"date-time"},"lastUpdated":{"type":"string","format":"date-time"},"name":{"type":"string","minLength":1,"maxLength":255},"description":{"type":"string","maxLength":32767},"experienceUserIds":{"type":"array","items":{"type":"string","pattern":"^[A-Fa-f\\d]{24}$"},"maxItems":1000},"experienceEndpointIds":{"type":"array","items":{"type":"string","pattern":"^[A-Fa-f\\d]{24}$"},"maxItems":1000},"groupTags":{"type":"object","patternProperties":{"^[0-9a-zA-Z_-]{1,255}$":{"type":"string","minLength":1,"maxLength":255}},"additionalProperties":false},"deviceIds":{"type":"array","items":{"type":"string","pattern":"^[A-Fa-f\\d]{24}$"},"maxItems":1000},"deviceTags":{"type":"array","items":{"type":"object","properties":{"key":{"type":"string","pattern":"^[0-9a-zA-Z_-]{1,255}$"},"value":{"type":"string","minLength":1,"maxLength":255}},"additionalProperties":false},"maxItems":100},"deviceQueryJson":{"type":["string","null"],"maxLength":8192},"parentId":{"oneOf":[{"type":"string","pattern":"^[A-Fa-f\\d]{24}$"},{"type":"null"}]}}}"#;
const FULL_O9881_SCHEMA: &str = r#"{"$schema":"http://json-schema.org/draft-04/schema#","type":"object","properties":{"items":{"type":"array","items":{"title":"Experience Group","description":"Schema for a single Experience Group","type":"object","properties":{"id":{"type":"string","pattern":"^[A-Fa-f\\d]{24}$"},"experienceGroupId":{"type":"string","pattern":"^[A-Fa-f\\d]{24}$"},"applicationId":{"type":"string","pattern":"^[A-Fa-f\\d]{24}$"},"creationDate":{"type":"string","format":"date-time"},"lastUpdated":{"type":"string","format":"date-time"},"name":{"type":"string","minLength":1,"maxLength":255},"description":{"type":"string","maxLength":32767},"experienceUserIds":{"type":"array","items":{"type":"string","pattern":"^[A-Fa-f\\d]{24}$"},"maxItems":1000},"experienceEndpointIds":{"type":"array","items":{"type":"string","pattern":"^[A-Fa-f\\d]{24}$"},"maxItems":1000},"groupTags":{"type":"object","patternProperties":{"^[0-9a-zA-Z_-]{1,255}$":{"type":"string","minLength":1,"maxLength":255}},"additionalProperties":false},"deviceIds":{"type":"array","items":{"type":"string","pattern":"^[A-Fa-f\\d]{24}$"},"maxItems":1000},"deviceTags":{"type":"array","items":{"type":"object","properties":{"key":{"type":"string","pattern":"^[0-9a-zA-Z_-]{1,255}$"},"value":{"type":"string","minLength":1,"maxLength":255}},"additionalProperties":false},"maxItems":100},"deviceQueryJson":{"type":["string","null"],"maxLength":8192},"parentId":{"oneOf":[{"type":"string","pattern":"^[A-Fa-f\\d]{24}$"},{"type":"null"}]}}}},"count":{"type":"integer"},"totalCount":{"type":"integer"},"perPage":{"type":"integer"},"page":{"type":"integer"},"filter":{"type":"string"},"filterField":{"type":"string"},"sortField":{"type":"string"},"sortDirection":{"type":"string","enum":["asc","desc","ASC","DESC",""]},"applicationId":{"type":"string","pattern":"^[A-Fa-f\\d]{24}$"}}}"#;

const PATTERN_PREFIX: &[u8] = br#"["507f1f77bcf86cd799439016", "507f1f77bcf86cd799439017"#;
const PLAIN_STRING_PREFIX: &[u8] = br#"["507f1f77bcf86cd799439016", "507f1f77bcf86cd799439017"#;
const LITERAL_X_ENUM_PREFIX: &[u8] = br#"["x", "x"#;
const O9881_TWO_ARRAYS_GROUP_TAGS_PREFIX: &[u8] = br#"{"experienceUserIds": ["507f1f77bcf86cd799439016", "507f1f77bcf86cd799439017"], "experienceEndpointIds": ["507f1f77bcf86cd799439016", "507f1f77bcf86cd799439017"#;
const FULL_ITEM_O9881_PREFIX: &[u8] = br#"{"experienceUserIds": ["507f1f77bcf86cd799439014", "507f1f77bcf86cd799439015"], "experienceEndpointIds": ["507f1f77bcf86cd799439016", "507f1f77bcf86cd799439017"#;
const FULL_O9881_STEP182_PREFIX: &[u8] = br#"{"items": [{"id": "507f1f77bcf86cd799439011", "experienceGroupId": "507f1f77bcf86cd799439012", "applicationId": "507f1f77bcf86cd799439013", "creationDate": "2022-01-01T12:00:00.000Z", "lastUpdated": "2022-01-01T12:00:00.000Z", "name": "Example Experience Group", "description": "This is an example experience group.", "experienceUserIds": ["507f1f77bcf86cd799439014", "507f1f77bcf86cd799439015"], "experienceEndpointIds": ["507f1f77bcf86cd799439016", "507f1f77bcf86cd799439017"#;

struct BenchCase {
    name: &'static str,
    schema: &'static str,
    prefix: &'static [u8],
    close_token_bytes: &'static [u8],
}

const CASES: &[BenchCase] = &[
    BenchCase {
        name: "pattern_id_string",
        schema: PATTERN_ID_SCHEMA,
        prefix: PATTERN_PREFIX,
        close_token_bytes: ARRAY_CLOSE_BYTES,
    },
    BenchCase {
        name: "plain_string",
        schema: PLAIN_STRING_SCHEMA,
        prefix: PLAIN_STRING_PREFIX,
        close_token_bytes: ARRAY_CLOSE_BYTES,
    },
    BenchCase {
        name: "literal_x_enum",
        schema: LITERAL_X_ENUM_SCHEMA,
        prefix: LITERAL_X_ENUM_PREFIX,
        close_token_bytes: ARRAY_CLOSE_BYTES,
    },
    BenchCase {
        name: "o9881_two_arrays_group_tags",
        schema: O9881_TWO_ARRAYS_GROUP_TAGS_SCHEMA,
        prefix: O9881_TWO_ARRAYS_GROUP_TAGS_PREFIX,
        close_token_bytes: OBJECT_MRE_CLOSE_BYTES,
    },
    BenchCase {
        name: "full_item_o9881",
        schema: FULL_ITEM_O9881_SCHEMA,
        prefix: FULL_ITEM_O9881_PREFIX,
        close_token_bytes: OBJECT_MRE_CLOSE_BYTES,
    },
    BenchCase {
        name: "full_o9881_step182",
        schema: FULL_O9881_SCHEMA,
        prefix: FULL_O9881_STEP182_PREFIX,
        close_token_bytes: OBJECT_MRE_CLOSE_BYTES,
    },
];

fn close_token_id(vocab: &glrmask::Vocab, close_token_bytes: &[u8]) -> u32 {
    vocab
        .entries
        .iter()
        .find_map(|(&token_id, bytes)| (bytes.as_slice() == close_token_bytes).then_some(token_id))
        .unwrap_or_else(|| panic!("Llama vocab is missing token bytes {:?}", close_token_bytes))
}

fn prepare_state<'a>(
    schema: &str,
    vocab: &'a glrmask::Vocab,
    prefix: &[u8],
    close_token_bytes: &[u8],
) -> glrmask::ConstraintState<'a> {
    let constraint = Box::leak(Box::new(Constraint::from_json_schema(schema, vocab).unwrap()));
    let mut state = constraint.start();
    state.commit_bytes(prefix).unwrap();
    let close_token_id = close_token_id(vocab, close_token_bytes);
    let mut verify = state.clone();
    verify.commit_token(close_token_id).unwrap();
    state
}

fn bench_bounded_string_array_close(c: &mut Criterion) {
    cfa_sweep::assert_release_benchmark("bounded_string_array_close");
    let vocab = cfa_sweep::load_llama3_vocab();
    assert_eq!(vocab.len(), 128_002, "expected the full Llama 3 vocabulary");
    let mask_words = (vocab.len() + 31) / 32;
    let mut group = c.benchmark_group("bounded_string_array_close");

    for case in CASES {
        let close_token_id = close_token_id(&vocab, case.close_token_bytes);
        let prepared = prepare_state(case.schema, &vocab, case.prefix, case.close_token_bytes);
        group.bench_with_input(BenchmarkId::new("commit_close", case.name), case, |b, _| {
            b.iter_batched(
                || prepared.clone(),
                |mut state| {
                    state.commit_token(black_box(close_token_id)).unwrap();
                    black_box(state);
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("mask_then_commit", case.name), case, |b, _| {
            b.iter_batched(
                || (prepared.clone(), vec![0u32; mask_words]),
                |(mut state, mut mask_buf)| {
                    state.fill_mask(&mut mask_buf);
                    state.commit_token(black_box(close_token_id)).unwrap();
                    black_box((&state, &mask_buf));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("build_constraint", case.name), case, |b, case| {
            b.iter_batched(
                || vocab.clone(),
                |fresh_vocab| {
                    cfa_sweep::clear_compile_caches();
                    let constraint = Constraint::from_json_schema(black_box(case.schema), black_box(&fresh_vocab)).unwrap();
                    black_box(constraint.num_parser_states());
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_bounded_string_array_close);
criterion_main!(benches);
