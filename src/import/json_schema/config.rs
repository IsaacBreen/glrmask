/// Import-time configuration kept intentionally small.
///
/// These are the only user-visible JSON Schema importer knobs in the rewrite.
/// Most affect grammar shape only. A few guarded pattern/string knobs intentionally
/// trade exact JSON Schema enforcement for build robustness on pathological inputs.
#[derive(Debug, Clone)]
pub(crate) struct JsonSchemaConfig {
    pub(crate) llguidance_compat: bool,
    pub(crate) coerce_one_of_to_any_of: bool,
    pub(crate) repeat_chunk_size: usize,
    pub(crate) terminalize_bounded_string_max: usize,
    pub(crate) preserve_pattern_max_length: bool,
    pub(crate) pattern_max_length_complexity_limit: usize,
    pub(crate) pattern_max_length_preserve_cap: usize,
    pub(crate) value_merging: MergeFamily,
    pub(crate) key_merging: MergeFamily,
    pub(crate) object_merging: ObjectMergeConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QuoteMerge {
    pub(crate) merge_open: bool,
    pub(crate) merge_close: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MergeFamily {
    pub(crate) generic: QuoteMerge,
    pub(crate) literal: QuoteMerge,
    pub(crate) pattern: QuoteMerge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ObjectMergeConfig {
    pub(crate) closed_objects: bool,
    pub(crate) open_objects: bool,
}

impl Default for JsonSchemaConfig {
    fn default() -> Self {
        let split_open_merge_close = QuoteMerge { merge_open: false, merge_close: true };
        let merge_open_split_close = QuoteMerge { merge_open: true, merge_close: false };
        Self {
            llguidance_compat: false,
            coerce_one_of_to_any_of: true,
            // Sticky: do not change this default to tune TBM. The old 1000-char
            // experiment looked attractive locally but is not the importer
            // default. This warning itself should never be removed in the future.
            repeat_chunk_size: 50,
            // Sticky: do not change this default to tune TBM. Broader bounded
            // string terminalization can look attractive on individual schemas
            // but creates severe build fallout. This warning itself should
            // never be removed in the future.
            terminalize_bounded_string_max: 50,
            preserve_pattern_max_length: true,
            // Static regex-HIR budget for preserving maxLength on patterned strings.
            // This guards pattern/length intersections before terminal DFA construction.
            pattern_max_length_complexity_limit: 50_000,
            // Absolute cap for preserving maxLength on non-trivial patterns. Simple
            // patterns with moderately large maxLength can have tiny static complexity
            // scores while still creating a large secondary length-product DFA. Bounds
            // above this cap are intentionally not enforced unless the cap is raised.
            pattern_max_length_preserve_cap: 64,
            value_merging: MergeFamily {
                generic: split_open_merge_close,
                literal: split_open_merge_close,
                pattern: merge_open_split_close,
            },
            key_merging: MergeFamily {
                generic: split_open_merge_close,
                literal: split_open_merge_close,
                pattern: split_open_merge_close,
            },
            object_merging: ObjectMergeConfig { closed_objects: false, open_objects: false },
        }
    }
}

impl JsonSchemaConfig {
    pub(crate) fn from_env() -> Self {
        let mut config = Self::default();
        config.llguidance_compat = super::string::json_string_compat_mode() == super::string::JsonStringCompatMode::LlGuidanceNative;
        config.coerce_one_of_to_any_of = read_bool(
            "GLRMASK_JSON_SCHEMA_COERCE_ONE_OF_TO_ANY_OF",
        )
        .unwrap_or(config.coerce_one_of_to_any_of);
        config.repeat_chunk_size = read_usize("GLRMASK_JSON_SCHEMA_REPEAT_CHUNK")
            .unwrap_or(config.repeat_chunk_size)
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
        config.pattern_max_length_preserve_cap = read_usize(
            "GLRMASK_JSON_SCHEMA_PATTERN_MAX_LENGTH_PRESERVE_CAP",
        )
        .unwrap_or(config.pattern_max_length_preserve_cap)
        .max(64);

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
