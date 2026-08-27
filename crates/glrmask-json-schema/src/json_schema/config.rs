/// Import-time configuration kept intentionally small.
///
/// These are the only user-visible JSON Schema importer knobs in the rewrite.
/// Most affect grammar shape only. Patterned string bounds are always
/// semantic. `preserve_pattern_max_length` is retained for configuration/API
/// compatibility, but disabling it must never enlarge the accepted language.
#[derive(Debug, Clone)]
pub struct JsonSchemaConfig {
    pub llguidance_compat: bool,
    pub coerce_one_of_to_any_of: bool,
    pub repeat_chunk_size: usize,
    pub string_repeat_chunk_size: usize,
    pub terminalize_bounded_string_max: usize,
    pub preserve_pattern_max_length: bool,
    pub pattern_max_length_complexity_limit: usize,
    pub pattern_max_length_hard_complexity_limit: usize,
    pub split_complex_patterns: bool,
    pub value_merging: MergeFamily,
    pub key_merging: MergeFamily,
    pub object_merging: ObjectMergeConfig,
    /// Optional non-vocabulary sentinel used as an external dynamic-value
    /// subgrammar at nested JSON value positions (object property values and
    /// array items). The schema root itself never gains this alternative.
    ///
    /// This is intentionally not environment-configurable: callers must bind
    /// the sentinel to an actual compiled child constraint through the facade
    /// API, so an unbound placeholder can never leak into a user grammar.
    pub dynamic_value_token_id: Option<u32>,
    /// Optional non-vocabulary sentinel for an unrestricted JavaScript
    /// condition expression used by schema-aware conditional values. This is
    /// meaningful only when `dynamic_value_token_id` is also configured.
    pub dynamic_condition_token_id: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuoteMerge {
    pub merge_open: bool,
    pub merge_close: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergeFamily {
    pub generic: QuoteMerge,
    pub literal: QuoteMerge,
    pub pattern: QuoteMerge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectMergeConfig {
    pub closed_objects: bool,
    pub open_objects: bool,
}

impl Default for JsonSchemaConfig {
    fn default() -> Self {
        let split_open_merge_close = QuoteMerge { merge_open: false, merge_close: true };
        let merge_both_quotes = QuoteMerge { merge_open: true, merge_close: true };
        let merge_open_split_close = QuoteMerge { merge_open: true, merge_close: false };
        Self {
            llguidance_compat: false,
            coerce_one_of_to_any_of: true,
            // Sticky: do not change this default to tune TBM. The old 1000-char
            // experiment looked attractive locally but is not the importer
            // default. This warning itself should never be removed in the future.
            repeat_chunk_size: 50,
            // Keep bounded-string chunking independent from array/generic repeat
            // chunking. A 64-byte tokenizer token can span three 50-character
            // chunks when it starts part-way through the first one; using 64 for
            // string chunks removes that avoidable terminal-path depth.
            string_repeat_chunk_size: 64,
            // Sticky: do not change this default to tune TBM. Broader bounded
            // string terminalization can look attractive on individual schemas
            // but creates severe build fallout. This warning itself should
            // never be removed in the future.
            terminalize_bounded_string_max: 50,
            // Legacy compatibility knob. Finite patterned-string maxLength is
            // now always enforced; retain the field/env so older callers do not
            // break when constructing importer configuration.
            preserve_pattern_max_length: true,
            // Static regex-HIR budget for choosing specialized patterned-string
            // lowering strategies. It must not decide whether maxLength is
            // semantically enforced.
            pattern_max_length_complexity_limit: 8_000,
            // Exact pattern/maxLength intersections can have a deterministic
            // state space proportional to the product of two large automata.
            // Above this conservative structural budget, reject the schema
            // explicitly instead of risking an unbounded compile or silently
            // dropping the finite length constraint.
            pattern_max_length_hard_complexity_limit: 1_000_000,
            split_complex_patterns: false,
            value_merging: MergeFamily {
                generic: split_open_merge_close,
                literal: merge_both_quotes,
                pattern: merge_open_split_close,
            },
            key_merging: MergeFamily {
                generic: split_open_merge_close,
                literal: merge_both_quotes,
                pattern: split_open_merge_close,
            },
            object_merging: ObjectMergeConfig { closed_objects: false, open_objects: false },
            dynamic_value_token_id: None,
            dynamic_condition_token_id: None,
        }
    }
}

impl JsonSchemaConfig {
    pub fn from_env() -> Self {
        let mut config = Self::default();
        config.llguidance_compat = super::string::json_string_compat_mode() == super::string::JsonStringCompatMode::LlGuidanceNative;
        config.coerce_one_of_to_any_of = read_bool(
            "GLRMASK_JSON_SCHEMA_COERCE_ONE_OF_TO_ANY_OF",
        )
        .unwrap_or(config.coerce_one_of_to_any_of);
        config.repeat_chunk_size = read_usize("GLRMASK_JSON_SCHEMA_REPEAT_CHUNK")
            .unwrap_or(config.repeat_chunk_size)
            .max(1);
        config.string_repeat_chunk_size = read_usize(
            "GLRMASK_JSON_SCHEMA_STRING_REPEAT_CHUNK",
        )
        .unwrap_or(config.string_repeat_chunk_size)
        .max(1);
        config.terminalize_bounded_string_max = read_usize(
            "GLRMASK_JSON_SCHEMA_TERMINALIZE_BOUNDED_STRING_MAX",
        )
        .unwrap_or(config.terminalize_bounded_string_max);
        config.preserve_pattern_max_length = read_bool(
            "GLRMASK_JSON_SCHEMA_PRESERVE_PATTERN_MAX_LENGTH",
        )
        .unwrap_or(config.preserve_pattern_max_length);
        config.pattern_max_length_complexity_limit = read_usize(
            "GLRMASK_JSON_SCHEMA_PATTERN_MAX_LENGTH_COMPLEXITY_LIMIT",
        )
        .unwrap_or(config.pattern_max_length_complexity_limit);
        config.pattern_max_length_hard_complexity_limit = read_usize(
            "GLRMASK_JSON_SCHEMA_PATTERN_MAX_LENGTH_HARD_COMPLEXITY_LIMIT",
        )
        .unwrap_or(config.pattern_max_length_hard_complexity_limit);
        config.split_complex_patterns = read_bool(
            "GLRMASK_JSON_SCHEMA_SPLIT_COMPLEX_PATTERNS",
        )
        .unwrap_or(config.split_complex_patterns);
        config.value_merging.generic = read_quote_merge(
            "GLRMASK_JSON_SCHEMA_VALUE_MERGE_OPEN",
            "GLRMASK_JSON_SCHEMA_VALUE_MERGE_CLOSE",
            config.value_merging.generic,
        );
        config.value_merging.literal = read_quote_merge(
            "GLRMASK_JSON_SCHEMA_LITERAL_VALUE_MERGE_OPEN",
            "GLRMASK_JSON_SCHEMA_LITERAL_VALUE_MERGE_CLOSE",
            config.value_merging.literal,
        );
        config.value_merging.pattern = read_quote_merge(
            "GLRMASK_JSON_SCHEMA_PATTERN_VALUE_MERGE_OPEN",
            "GLRMASK_JSON_SCHEMA_PATTERN_VALUE_MERGE_CLOSE",
            config.value_merging.pattern,
        );

        config.key_merging.generic = read_quote_merge(
            "GLRMASK_JSON_SCHEMA_KEY_MERGE_OPEN",
            "GLRMASK_JSON_SCHEMA_KEY_MERGE_CLOSE",
            config.key_merging.generic,
        );
        config.key_merging.literal = read_quote_merge(
            "GLRMASK_JSON_SCHEMA_LITERAL_KEY_MERGE_OPEN",
            "GLRMASK_JSON_SCHEMA_LITERAL_KEY_MERGE_CLOSE",
            config.key_merging.literal,
        );
        config.key_merging.pattern = read_quote_merge(
            "GLRMASK_JSON_SCHEMA_PATTERN_KEY_MERGE_OPEN",
            "GLRMASK_JSON_SCHEMA_PATTERN_KEY_MERGE_CLOSE",
            config.key_merging.pattern,
        );

        config.object_merging.closed_objects = read_bool(
            "GLRMASK_JSON_SCHEMA_MERGE_CLOSED_OBJECTS",
        ).unwrap_or(config.object_merging.closed_objects);
        config.object_merging.open_objects = read_bool(
            "GLRMASK_JSON_SCHEMA_MERGE_OPEN_OBJECTS",
        ).unwrap_or(config.object_merging.open_objects);

        config
    }
}

fn read_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse().ok()
}

fn read_quote_merge(open_name: &str, close_name: &str, default: QuoteMerge) -> QuoteMerge {
    QuoteMerge {
        merge_open: read_bool(open_name).unwrap_or(default.merge_open),
        merge_close: read_bool(close_name).unwrap_or(default.merge_close),
    }
}

fn read_bool(name: &str) -> Option<bool> {
    let value = std::env::var(name).ok()?;
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}
