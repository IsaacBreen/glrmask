# Symbol-level priority ledger

| Priority | Symbol/file pattern | Reason | Action |
|---:|---|---|---|
| 3 | `src/import/json_schema/diagnostics.rs:8` `type ImportResult` | ordinary internal symbol | keep near current phase |
| 1 | `src/import/json_schema/diagnostics.rs:11` `struct SchemaImportError` | central phase-boundary type | document and keep prominent |
| 1 | `src/import/json_schema/diagnostics.rs:15` `impl SchemaImportError` | central phase-boundary type | document and keep prominent |
| 3 | `src/import/json_schema/diagnostics.rs:16` `fn new` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/diagnostics.rs:20` `fn at` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/diagnostics.rs:24` `fn message` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/diagnostics.rs:29` `impl From` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/diagnostics.rs:30` `fn from` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:10` `fn singleton_all_of_ref_without_siblings` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:27` `fn one_of_mixes_ref_and_inline_branches` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:40` `fn schema_is_null_only_inline_branch` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:57` `fn collect_all_ref_pointers` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:72` `fn local_id_alias` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:90` `fn load_document` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:126` `fn collect_definitions` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:170` `fn collect_ref_targets` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:258` `fn load_schema_at` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:267` `fn load_object_schema` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:295` `fn load_assertions` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:327` `fn validate_supported_keys` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:340` `fn is_unsupported_validation_key` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:356` `fn load_types` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:380` `fn parse_type_name` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:393` `fn load_enum_values` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:403` `fn load_schema_array` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:421` `fn load_schema_member` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:432` `fn should_load_object_assertion` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:445` `fn should_load_array_assertion` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:452` `fn should_load_string_assertion` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:459` `fn should_load_number_assertion` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:467` `fn type_mentions` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:471` `fn load_object_keywords` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:532` `fn load_array_keywords` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:585` `fn load_string_keywords` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:594` `fn load_number_keywords` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:636` `fn read_usize_keyword` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:648` `fn read_f64_keyword` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:658` `fn read_string_keyword` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/load/mod.rs:668` `fn escape_pointer_segment` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/array/mod.rs:9` `fn lower_array` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/array/mod.rs:45` `fn should_terminalize_bounded_scalar_array` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/array/mod.rs:49` `fn array_body` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/array/mod.rs:79` `fn bounded_homogeneous_array_exprnfa` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/array/mod.rs:118` `fn bounded_homogeneous_array_terminal` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/array/mod.rs:150` `fn unbounded_homogeneous_array_terminal` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/array/mod.rs:162` `fn lower_tuple_array_body` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/array/mod.rs:211` `fn fixed_array_items` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/array/mod.rs:225` `fn tuple_tail_items` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/array/mod.rs:253` `fn bounded_array_object_item_candidate` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:26` `const JSON_VALUE_RULE` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:27` `const JSON_OBJECT_RULE` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:28` `const JSON_ARRAY_RULE` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:29` `const JSON_STRING_RULE` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:30` `const JSON_STRING_CHAR_RULE` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:31` `const JSON_ITEM_SEPARATOR_RULE` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:32` `const JSON_KEY_SEPARATOR_RULE` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:33` `const JSON_INTEGER_RULE` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:34` `const JSON_NUMBER_RULE` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:35` `const JSON_BOOL_RULE` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:36` `const JSON_NULL_RULE` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:37` `const JSON_ADDITIONAL_KEY_COLON_SHARED_RULE` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:38` `const JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_RULE` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:40` `const JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_NT_RULE` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:42` `const MAX_SHARED_ADDITIONAL_EXCLUSION_KEYS` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:43` `const STRING_ENUM_REGEX_MIN_VALUES` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:44` `const STRING_ENUM_REGEX_MIN_ENCODED_BYTES` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:46` `fn lower_document` | ordinary internal symbol | keep near current phase |
| 1 | `src/import/json_schema/lower/mod.rs:54` `struct Lowerer` | central phase-boundary type | document and keep prominent |
| 3 | `src/import/json_schema/lower/mod.rs:76` `fn new` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:111` `fn finish` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:117` `fn install_json_builtins` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:189` `fn item_separator_expr` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:193` `fn key_separator_expr` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:197` `fn separator_regex` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:204` `fn json_string_char_regex` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:208` `fn lower_schema` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:217` `fn lower_ref` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:237` `fn resolve_ref_target` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:245` `fn lower_assertions` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:306` `fn selected_types` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:326` `fn lower_for_type` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:358` `fn inferred_constrained_types` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:375` `fn lower_untyped_single_family_assertions` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:406` `fn lower_json_literal` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:453` `fn add_nonterminal_rule` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:463` `fn add_terminal_rule` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:473` `fn add_internal_terminal_rule` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:483` `fn fresh_rule_name` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:494` `fn large_string_enum_regex_literals` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:544` `fn string_enum_regex` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:555` `fn factored_small_string_enum_expr` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:576` `fn collect_shared_ap_exclusion_plan` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:591` `fn collect_shared_ap_exclusions_from_schema` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:643` `fn normalize_local_ref` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:655` `fn is_local_fragment_alias` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:659` `fn is_absolute_self_ref_alias` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:663` `fn r` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:667` `fn lit` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:671` `fn lit_bytes` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:675` `fn seq` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:683` `fn choice` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/mod.rs:698` `fn never` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/number/mod.rs:10` `const MAX_EXPLICIT_INTEGER_RANGE` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/number/mod.rs:11` `const MAX_EXPLICIT_INTEGER_MULTIPLES` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/number/mod.rs:14` `fn lower_number` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/number/mod.rs:59` `fn lower_integer` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/number/mod.rs:107` `fn integer_lower_bound` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/number/mod.rs:119` `fn integer_upper_bound` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/number/mod.rs:131` `fn integer_satisfies_multiple` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/number/mod.rs:139` `fn bounded_integer_multiple_choice` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/number/mod.rs:162` `fn ceil_div_i64` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/number/mod.rs:168` `fn integer_multiple_expr` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/number/mod.rs:172` `fn positive_integer_multiple_value` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/number/mod.rs:180` `fn positive_integer_multiple_i64` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/number/mod.rs:185` `fn power_of_ten_multiple_regex` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/number/mod.rs:206` `fn decimal_multiple_regex` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/number/mod.rs:212` `struct DecimalStep` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/number/mod.rs:218` `fn parse_decimal_step` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/lower/number/mod.rs:247` `fn decimal_fraction_regex` | ordinary internal symbol | keep near current phase |
| 2 | `src/import/json_schema/lower/object/mod.rs:22` `const POINT_PATH_PATTERN` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:23` `const LARGE_OBJECT_LITERAL_KEY_TRIE_MIN_ITEMS` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:24` `const SNOWPLOW_CONTEXTS_PATTERN` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:25` `const SNOWPLOW_UNSTRUCT_EVENT_PATTERN` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:26` `const SNOWPLOW_KEY_TRIE_PREFIX_SPLIT_BYTES` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:28` `struct ObjectItem` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:36` `const ANYOF_FIXED_OBJECT_EXPR_NFA_MAX_STATES` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:38` `struct AnyOfFixedObjectItem` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:44` `struct AnyOfFixedObjectVariant` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:48` `struct AnyOfObjectVariant` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:57` `struct AnyOfFixedObjectState` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:64` `enum AnyOfObjectPhase` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:71` `enum ShadowOwnerState` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:82` `struct AnyOfObjectState` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:90` `fn is_obviously_object_valued_property` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:102` `fn obvious_object_valued_property_count` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:110` `fn is_unconstrained_open_object_schema` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:119` `impl AnyOfFixedObjectVariant` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:120` `fn len` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:124` `fn advance_cursor` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:135` `fn close_allowed` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:139` `fn legal_next_keys` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:150` `fn value_expr_for_key` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:158` `impl AnyOfObjectVariant` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:159` `fn len` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:163` `fn advance_cursor` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:174` `fn close_allowed` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:178` `fn legal_next_keys` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:189` `fn value_expr_for_key` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:196` `fn has_required_items` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:202` `fn lower_object` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:206` `fn lower_object_requiring_any_property` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:214` `fn lower_object_with_exclusive_properties` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:223` `fn try_lower_closed_object_any_of_variants` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:251` `fn try_lower_open_object_any_of_variants` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:277` `fn try_lower_ref_string_path_object_any_of` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:332` `fn resolve_branch_schema` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:339` `fn is_path_recursive_open_object_branch` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:386` `fn lower_object_internal` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:677` `fn dynamic_pair_list_body` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:711` `fn schema_has_huge_bounded_string` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:728` `fn try_lower_pattern_map_pair_list_object` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:804` `fn try_lower_wrapper_pattern_map_anyof_value` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:832` `fn lower_fixed_object_body_exprnfa` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:938` `fn collect_closed_any_of_object_variant` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1017` `fn collect_open_any_of_object_variant` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1024` `fn collect_open_any_of_object_variant_inner` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1138` `fn add_expr_nfa_symbol_path` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1160` `fn split_additional_key_colon_transition` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1171` `fn is_shared_additional_key_colon_choice` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1186` `fn is_shared_additional_key_colon_base_ref` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1190` `fn is_shared_additional_key_colon_addback` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1206` `fn split_object_pair_symbols` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1217` `fn split_object_pair_symbol_paths` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1227` `fn lower_closed_any_of_object_variants_expr_nfa` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1318` `fn is_json_value_expr` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1322` `fn is_json_string_expr` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1326` `fn is_json_string_constrained_expr` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1330` `fn non_string_json_value_expr` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1340` `fn invalid_residual_value_for_owner` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1356` `fn select_shadow_owner_for_variant` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1399` `fn shadow_owner_suppresses_close` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1416` `fn shadow_owner_can_take_additional` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1429` `fn advance_shadow_owner_on_key` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1473` `fn lower_open_any_of_object_variants_expr_nfa` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1822` `fn lower_fixed_object_body_exprnfa_without_group` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1962` `fn split_literal_key_symbol` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1985` `fn object_pair_path_symbols` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:1994` `fn lower_snowplow_large_pattern_object_key_trie` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:2163` `fn lower_large_optional_open_object_fused_prefix_chain` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:2217` `fn lower_large_closed_object_prefix_chain` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:2253` `fn lower_large_closed_object_fixed_pair_loop` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:2327` `fn lower_required_prefix_open_object_pair_loop` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:2429` `fn lower_property_item` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:2456` `fn lower_object_property_value_schema` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:2498` `fn object_with_required_synthetic_properties` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:2548` `fn is_ref_string_open_object_branch` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:2584` `fn all_of_has_explicit_object_only_type` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:2588` `fn schema_has_explicit_object_only_type` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:2602` `fn is_plain_array_branch` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:2627` `fn is_string_schema` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:2652` `fn property_matches_pattern` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:2656` `fn pattern_schema_for_property` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:2673` `fn single_numeric_property_type` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/object/mod.rs:2684` `fn has_non_numeric_assertions` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:19` `fn lower_string` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:37` `fn lower_inline_bounded_array_string_item_expr` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:83` `fn lower_constrained_string_terminal_expr` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:133` `fn lower_string_expr` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:157` `fn should_split_bounded_string` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:165` `fn string_char_exact_ref` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:188` `fn string_char_upto_ref` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:211` `fn string_char_upto_close_ref` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:225` `fn string_char_exact_open_ref` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:235` `fn string_char_upto_wrapped_ref` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:242` `fn split_string_exact_expr` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:261` `fn split_string_upto_close_expr` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:289` `fn lower_split_bounded_string` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:303` `fn split_bounded_string_terminal_expr` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:344` `fn lower_string_literal` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:352` `fn lower_literal_key_colon` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:356` `fn lower_literal_key_colon_with_prefix` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:369` `fn lower_pattern_key_colon_expr` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:373` `fn pattern_key_colon_full_language` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:381` `fn lower_pattern_key_colon_terminal` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:408` `fn pattern_overlapping_literal_keys` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:432` `fn pattern_local_overlapping_literal_keys` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:453` `fn shared_pattern_overlap_literal_rule` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:475` `fn lower_pattern_key_colon_appearance` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:510` `fn lower_additional_key_colon` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:552` `fn use_shared_additional_key_colon` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:556` `fn lower_additional_key_colon_expanded_addback` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:590` `fn lower_additional_key_colon_literal_only` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:613` `fn shared_additional_excluded_key_colon` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:634` `fn shared_additional_key_colon_base` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:673` `fn lower_pattern_key_colon_addback` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:714` `fn string_body_for_length` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:728` `fn repeat_exact_string_char` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:756` `fn string_pattern_as_body_regex` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:764` `fn preprocess_ascii_shorthand` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:806` `fn string_pattern_hir_as_body_regex` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:847` `fn string_pattern_branch_as_body_regex` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:856` `fn lower_string_pattern_branch_parts` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:862` `fn wrap_lowered_string_pattern_branch` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:872` `fn quoted_string_body_regex` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:876` `const DATE_FORMAT_BODY_REGEX` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:877` `const DATE_TIME_FORMAT_BODY_REGEX` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:879` `fn recognized_string_format_body_regex` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:905` `fn pattern_key_colon_regex` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:910` `fn strip_outer_captures` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:919` `fn strip_outer_start_anchor` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:934` `fn strip_outer_end_anchor` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:949` `fn strip_outer_anchors` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:988` `fn is_start_look` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:992` `fn is_end_look` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:996` `fn lower_decoded_regex_hir_to_json_body_regex` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1035` `fn lower_decoded_repetition_to_json_body_regex` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1054` `fn lower_decoded_class_to_json_body_regex` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1123` `fn utf8_sequence_to_regex_string` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1135` `fn unicode_range_to_utf8_regex_string` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1146` `fn is_unicode_decimal_digit_class` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1170` `fn is_dot_like_unicode_class` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1182` `fn push_safe_raw_char_ranges` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1205` `fn decoded_class_contains` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1219` `fn class_contains_general_non_ascii_non_whitespace` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1225` `fn regex_char_class_range` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1235` `fn escape_regex_class_char` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1245` `fn json_body_char_regex_for_decoded_char` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1255` `fn json_string_body_char_regex` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1259` `fn json_string_body_non_ascii_non_whitespace_regex` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1263` `fn json_string_body_dot_regex` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1267` `fn is_safe_raw_json_string_char` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1271` `fn property_name_matches_pattern` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1278` `fn is_regex_compile_limit_error` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1282` `fn string_value_satisfies_schema` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1317` `fn preprocess_ascii_shorthand_rewrites_generic_word_shorthand` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1323` `fn preprocess_ascii_shorthand_preserves_escaped_word_shorthand` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1328` `fn lowered_bounded_free_text_pattern_rejects_leading_space_slash` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/lower/string/mod.rs:1337` `fn lowered_optional_decimal_pattern_rejects_backslash_digit_string` | hotspot implementation symbol | split or proof-comment in follow-up |
| 3 | `src/import/json_schema/mod.rs:55` `fn schema_to_named_grammar` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/mod.rs:62` `fn simplify_grammar_enabled` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/mod.rs:67` `fn lower_exact_subtractions_enabled` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/mod.rs:72` `fn promote_literal_choices_enabled` | ordinary internal symbol | keep near current phase |
| 2 | `src/import/json_schema/normalize/combinators.rs:25` `fn lower_any_of` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:86` `fn try_merge_single_object_any_of_with_siblings` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:110` `fn lower_one_of` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:120` `fn lower_all_of` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:217` `fn try_lower_single_ref_with_object_siblings` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:265` `fn inline_all_of_refs` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:272` `fn inline_all_of_refs_for_any_of_factoring` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:286` `fn schema_transitively_refs_pointer` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:356` `fn inline_all_of_ref_target` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:385` `fn try_inline_object_like_all_of_target` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:400` `fn try_rewrite_all_of_object_choice_target` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:449` `fn inline_refs_in_all_of_branch` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:468` `fn try_merge_all_of_single_ref_object_branches` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:505` `fn drop_subsumed_open_object_any_of_branches` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:555` `fn object_branch_resolved` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:565` `fn object_schema_subsumes` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:612` `fn schema_subsumes` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:691` `fn all_of_intersection_terminal_safe` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:727` `fn explicit_all_of_type_intersection` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:748` `fn untyped_single_family_assertion` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:781` `fn family_overlaps_types` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:790` `fn drop_vacuous_untyped_family_branches` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:809` `fn flatten_pure_all_of_branches` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:825` `fn collapse_pure_single_choice_branches` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:838` `fn try_factor_required_property_any_of` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:884` `fn try_factor_closed_object_variant_any_of` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:989` `fn try_factor_mutually_exclusive_property_not_any_of` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1041` `fn mutually_exclusive_property_not_branch` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1078` `fn single_required_object_not_name` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1110` `fn single_required_object_branch_name` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1143` `fn closed_object_variant_branch` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1176` `fn open_object_any_of_covers_json_object` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1211` `fn object_branch` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1236` `fn property_schema_by_name` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1244` `fn schema_subsumption_key` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1251` `fn pure_any_of_assertions` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1265` `fn broad_string_assertions` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1288` `fn string_literal_values` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1311` `fn schemas_shape_equivalent` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1332` `fn option_schemas_shape_equivalent` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1340` `fn option_objects_shape_equivalent` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1348` `fn object_schemas_shape_equivalent` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1371` `fn additional_properties_shape_equivalent` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1385` `fn option_arrays_shape_equivalent` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1401` `fn option_strings_shape_equivalent` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1417` `fn option_numbers_shape_equivalent` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1435` `fn schema_slices_shape_equivalent` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1443` `fn sibling_assertion_schema` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1452` `fn branch_with_siblings` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1464` `fn push_object_only_type_into_branch` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1489` `fn schema_contains_ref` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1502` `fn schema_has_explicit_object_only_type` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1512` `fn try_merge_all_of_objects` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1521` `fn plain_object_schema` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1542` `enum ChoiceKind` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1547` `fn pure_choice_branch` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1570` `fn distribute_all_of_over_single_object_choice` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1613` `fn schema_is_object_like` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1617` `fn merge_all_of_object_like_schema` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1658` `fn object_like_schema` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1703` `fn merge_all_of_array_like_schema` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1742` `fn plain_array_schema` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1765` `fn array_is_bounds_only` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1770` `fn merge_array_bounds` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1780` `fn merge_two_objects` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1810` `fn merge_property_schemas` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1820` `fn is_vacuous_json_value_schema` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1850` `fn is_vacuous_object_schema` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1874` `fn merge_additional_properties` | hotspot implementation symbol | split or proof-comment in follow-up |
| 2 | `src/import/json_schema/normalize/combinators.rs:1894` `fn all_of_schema` | hotspot implementation symbol | split or proof-comment in follow-up |
| 1 | `src/import/json_schema/options.rs:10` `struct JsonSchemaConfig` | central phase-boundary type | document and keep prominent |
| 3 | `src/import/json_schema/options.rs:19` `struct QuoteMerge` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/options.rs:25` `struct MergeFamily` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/options.rs:32` `struct ObjectMergeConfig` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/options.rs:37` `impl Default` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/options.rs:38` `fn default` | ordinary internal symbol | keep near current phase |
| 1 | `src/import/json_schema/options.rs:66` `impl JsonSchemaConfig` | central phase-boundary type | document and keep prominent |
| 3 | `src/import/json_schema/options.rs:67` `fn from_env` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/options.rs:120` `fn read_usize` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/options.rs:124` `fn read_quote_merge` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/options.rs:131` `fn read_bool` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/options.rs:143` `fn simplify_grammar_enabled` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/options.rs:153` `fn lower_exact_subtractions_enabled` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/options.rs:170` `fn promote_literal_choices_enabled` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/schema/array.rs:5` `struct ArraySchema` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/schema/array.rs:12` `impl Default` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/schema/array.rs:13` `fn default` | ordinary internal symbol | keep near current phase |
| 1 | `src/import/json_schema/schema/assertions.rs:12` `struct SchemaAssertions` | central phase-boundary type | document and keep prominent |
| 1 | `src/import/json_schema/schema/assertions.rs:26` `impl SchemaAssertions` | central phase-boundary type | document and keep prominent |
| 3 | `src/import/json_schema/schema/assertions.rs:27` `fn is_empty` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/schema/assertions.rs:41` `fn has_value_assertions_without_combinators` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/schema/assertions.rs:51` `fn clone_without_combinators` | ordinary internal symbol | keep near current phase |
| 1 | `src/import/json_schema/schema/document.rs:10` `struct SchemaDocument` | central phase-boundary type | document and keep prominent |
| 3 | `src/import/json_schema/schema/document.rs:18` `struct SchemaDefinition` | ordinary internal symbol | keep near current phase |
| 1 | `src/import/json_schema/schema/mod.rs:21` `struct Schema` | central phase-boundary type | document and keep prominent |
| 3 | `src/import/json_schema/schema/mod.rs:28` `enum SchemaKind` | ordinary internal symbol | keep near current phase |
| 1 | `src/import/json_schema/schema/mod.rs:39` `impl Schema` | central phase-boundary type | document and keep prominent |
| 3 | `src/import/json_schema/schema/mod.rs:40` `fn any` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/schema/mod.rs:44` `fn never` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/schema/mod.rs:48` `fn assertions` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/schema/object.rs:7` `struct ObjectSchema` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/schema/object.rs:16` `impl Default` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/schema/object.rs:17` `fn default` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/schema/object.rs:30` `struct PropertySchema` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/schema/object.rs:36` `struct PatternPropertySchema` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/schema/object.rs:42` `enum AdditionalProperties` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/schema/scalar.rs:3` `enum SchemaType` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/schema/scalar.rs:19` `struct StringSchema` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/schema/scalar.rs:32` `struct NumberSchema` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:16` `struct EnvVarGuard` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:21` `impl EnvVarGuard` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:22` `fn set` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:30` `fn unset` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:39` `impl Drop` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:40` `fn drop` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:52` `fn start_expr` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:62` `fn exact_subtraction_lowering_env_var_defaults_true_and_accepts_falsey_values` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:77` `fn contains_separated_sequence` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:105` `fn contains_expr_nfa` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:133` `fn count_rules_with_prefix` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:137` `fn byte_vocab` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:145` `fn schema_accepts_bytes` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:153` `fn parser_path_count_after_bytes` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:163` `fn contains_exclude` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:187` `fn contains_ref_with_prefix` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:218` `fn find_all_pop1_stackshifts` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:235` `fn recursive_array_additional_properties_schema_does_not_reproduce_all_pop1_stackshifts` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:271` `fn contains_intersect` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:295` `fn contains_intersect_with_separated_sequence` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:332` `fn contains_ref_named` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:363` `fn contains_literal_bytes` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:398` `fn closed_object_lowers_to_prefix_chain_body` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:418` `fn large_optional_closed_object_uses_expr_nfa_body` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:438` `fn required_prefix_open_object_uses_pair_loop_body` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:465` `fn open_additional_map_min_properties_requires_dynamic_pair` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:483` `fn closed_fixed_object_min_properties_requires_one_optional_after_required` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:503` `fn closed_fixed_object_min_max_properties_exactly_one_optional` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:521` `fn closed_fixed_object_max_properties_caps_optional_after_required` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:540` `fn open_additional_map_max_properties_emits_bounded_dynamic_body` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:554` `fn required_property_covered_by_pattern_properties_is_synthesized` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:571` `fn required_property_matching_multiple_patterns_applies_all_pattern_schemas` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:590` `fn required_property_not_covered_by_closed_object_lowers_to_empty_language` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:607` `fn fixed_property_still_intersects_matching_pattern_property` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:628` `fn open_no_pattern_object_lowers_to_expr_nfa_body` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:647` `fn large_optional_open_object_uses_fused_prefix_chain_rules` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:668` `fn large_optional_open_object_allow_any_scalars_uses_expr_nfa_body` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:687` `fn large_optional_open_object_allow_any_object_valued_at_16_uses_expr_nfa_body` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:715` `fn large_optional_open_object_allow_any_object_valued_at_32_uses_expr_nfa_body` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:743` `fn large_required_open_object_does_not_use_fused_prefix_chain_rules` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:763` `fn pattern_property_object_still_uses_separated_sequence` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:778` `fn large_optional_open_object_with_pattern_properties_uses_fused_prefix_chain_rules` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:803` `fn allof_drops_vacuous_untyped_object_branch_for_typed_property` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:826` `fn large_snowplow_like_pattern_property_object_uses_expr_nfa_body` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:852` `fn shared_additional_key_colon_terminal_is_emitted_once` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:882` `fn additional_properties_factoring_uses_shared_key_colon_terminal` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:905` `fn huge_shared_additional_exclusion_set_uses_expanded_literal_addback` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:925` `fn shared_additional_excluded_key_skips_closed_object_keys` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:961` `fn arrays_use_item_schema_and_min_max_items` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:976` `fn bounded_object_arrays_use_exprnfa_rule` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:998` `fn bounded_pattern_string_arrays_use_terminal_rule` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1018` `fn large_bounded_pattern_string_arrays_do_not_use_terminal_rule` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1038` `fn unbounded_plain_string_arrays_use_terminal_rule` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1054` `fn prefix_items_lower_with_no_tail` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1077` `fn legacy_tuple_items_use_additional_items_tail` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1099` `fn plain_items_ignore_additional_items_without_tuple` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1113` `fn map_shaped_min_properties_lowers_as_bounded_pattern_map` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1128` `fn small_bounded_string_pattern_ignores_length_bounds` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1155` `fn large_bounded_string_pattern_ignores_length_bounds` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1184` `fn string_pattern_lowers_ascii_digit_subranges` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1198` `fn terminalized_dot_pattern_lowers_utf8_lead_byte_alternatives` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1219` `fn json_string_char_terminal_requires_valid_utf8_sequences` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1230` `fn medium_bounded_string_uses_split_chunk_rules_by_default` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1257` `fn bounded_pattern_map_respects_min_and_max_properties` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1273` `fn unsupported_nonredundant_max_properties_broadens` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1288` `fn unsupported_nonredundant_min_properties_broadens` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1304` `fn oversized_pattern_properties_overlap_check_broadens` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1331` `fn medium_bounded_string_terminalizes_with_env_override` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1359` `fn moderately_bounded_string_terminalizes_by_default` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1379` `fn split_bounded_string_chunks_do_not_overlap_at_boundary` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1398` `fn very_large_bounded_string_still_uses_split_chunk_rules` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1422` `fn decoded_string_patterns_are_matched_against_json_string_bodies` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1451` `fn uuid_format_lowers_to_constrained_terminal` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1474` `fn date_time_format_lowers_to_constrained_terminal` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1498` `fn date_format_lowers_to_constrained_terminal` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1522` `fn email_format_lowers_to_constrained_terminal` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1545` `fn email_format_with_large_max_length_does_not_preserve_length_envelope` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1570` `fn hostname_ipv4_ipv6_formats_lower_to_constrained_terminals` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1599` `fn uri_format_lowers_to_constrained_terminal` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1622` `fn string_pattern_is_intersected_with_format` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1638` `fn object_nonterminals_reference_terminalized_key_and_string_patterns` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1687` `fn overlapping_literal_and_pattern_keys_still_lower_with_shared_factoring` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1707` `fn json_separators_are_canonical_space_separated` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1724` `fn legacy_id_metadata_is_accepted` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1743` `fn local_ref_to_property_schema_is_loaded` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1757` `fn default_object_named_properties_is_not_scanned_for_ref_targets` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1772` `fn property_named_definitions_is_not_definition_container` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1790` `fn unknown_format_is_ignored_as_annotation` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1801` `fn date_time_string_value_satisfaction_filters_invalid_literals` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1815` `fn date_string_value_satisfaction_filters_invalid_literals` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1829` `fn uuid_string_value_satisfaction_filters_invalid_literals` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1844` `fn email_string_value_satisfaction_filters_invalid_literals` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1857` `fn host_string_value_satisfaction_filters_invalid_literals` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1882` `fn uri_string_value_satisfaction_filters_invalid_literals` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1899` `fn unknown_metadata_keys_are_ignored` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1911` `fn conditional_keywords_are_ignored_for_broad_lowering` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1934` `fn oneof_lowers_as_choice` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1947` `fn oneof_single_ref_wrapper_is_supported` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1962` `fn fragment_id_ref_alias_lowers` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1983` `fn absolute_root_id_self_ref_lowers` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:1998` `fn oneof_ref_and_null_is_supported` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2020` `fn oneof_mixed_ref_and_inline_errors` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2042` `fn unsupported_not_shape_errors` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2053` `fn anyof_property_not_mutual_exclusion_lowers_as_exclusive_group` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2084` `fn enum_and_const_lower_to_exact_json_literals` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2095` `fn string_const_splits_open_quote_from_literal_body` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2107` `fn object_const_uses_json_separator_rules` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2123` `fn large_string_enum_at_root_uses_raw_regex` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2135` `fn small_string_enum_at_root_uses_factored_suffix_choice` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2156` `fn snowplow_style_string_enum_uses_factored_suffix_choice` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2181` `fn patterned_string_enum_does_not_use_raw_regex_fast_path` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2197` `fn mixed_type_enum_does_not_use_raw_regex_fast_path` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2206` `fn integer_power_of_ten_multiple_lowers_to_regex` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2215` `fn unbounded_integer_multiple_of_three_lowers_broadly` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2223` `fn lower_bounded_integer_multiple_of_twelve_lowers_to_range` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2234` `fn bounded_integer_multiple_of_sixteen_lowers_without_enumerating_large_range` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2250` `fn non_integer_integer_multiple_of_remains_unsupported` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2257` `fn finite_integer_range_multiple_lowers_to_literals` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2271` `fn bounded_number_lowers_to_range_regex_not_plain_json_number` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2285` `fn large_bounded_integer_lowers_to_range_regex_not_plain_json_integer` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2299` `fn number_integer_union_uses_json_number_once` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2309` `fn anyof_lowers_to_choice` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2323` `fn anyof_allows_sibling_assertions` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2337` `fn anyof_required_property_object_factors_into_single_expr_nfa_body` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2360` `fn anyof_required_sets_with_object_sibling_type_do_not_allow_non_objects` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2385` `fn anyof_closed_object_variants_factor_into_single_expr_nfa_body` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2422` `fn anyof_required_property_factoring_falls_back_for_nontrivial_branch` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2445` `fn anyof_open_objects_with_disjoint_optional_properties_collapses_to_json_object` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2472` `fn unconstrained_object_collapses_to_json_object` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2485` `fn empty_properties_object_collapses_to_json_object` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2499` `fn constrained_open_objects_do_not_collapse_to_json_object` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2523` `fn anyof_open_objects_with_shared_optional_property_does_not_collapse_to_json_object` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2549` `fn anyof_nested_object_allof_refs_factor_into_single_body` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2627` `fn pattern_map_anyof_open_objects_with_disjoint_optional_properties_collapses_value_to_json_object` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2664` `fn anyof_closed_object_variant_factoring_falls_back_for_two_variant_properties` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2693` `fn anyof_closed_object_variant_factoring_falls_back_for_mismatched_common_schema` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2722` `fn anyof_closed_object_variants_with_shared_required_prefix_use_exact_variant_nfa` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2755` `fn anyof_untyped_closed_object_variants_keep_non_object_alternatives` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2791` `fn anyof_untyped_closed_object_variants_with_sibling_required_use_exact_variant_nfa` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2830` `fn anyof_explicit_object_variants_do_not_add_non_object_alternatives` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2857` `fn untyped_plain_object_assertions_keep_non_object_alternatives` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2882` `fn explicit_plain_object_assertions_remain_object_only` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2897` `fn untyped_object_and_array_assertions_do_not_take_plain_object_fallback` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2911` `fn anyof_required_property_factoring_falls_back_for_unknown_required_name` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2933` `fn allof_merges_plain_object_branches` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2958` `fn allof_merges_array_ref_with_min_items_assertion` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:2980` `fn allof_merges_array_bounds_before_ref_branch` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3002` `fn allof_array_min_max_items_merge_clamps_bounds` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3025` `fn allof_array_merge_preserves_non_array_type_union_guard` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3042` `fn allof_flattens_nested_object_allof_before_intersect` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3072` `fn allof_collapses_single_anyof_ref_before_intersect` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3104` `fn recursive_ref_in_allof_is_not_inlined` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3133` `fn allof_drops_vacuous_json_value_property_when_refined` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3168` `fn allof_drops_vacuous_object_property_when_refined` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3200` `fn allof_distributes_over_object_anyof_before_lowering` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3233` `fn allof_ref_to_nested_object_oneof_with_siblings_lowers` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3287` `fn unsafe_allof_object_ref_intersection_broadens_to_choice` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3312` `fn unsafe_allof_array_separated_sequence_broadens_to_choice` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3331` `fn terminal_safe_allof_keeps_intersection` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3346` `fn oneof_object_branches_with_root_type_object_and_required_anyof_lowers` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3394` `fn open_object_anyof_uses_single_object_body_nfa` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3482` `fn array_items_anyof_allof_ref_alias_variants_lower_to_shared_open_object_body` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3577` `fn sibling_pattern_addback_subtracts_local_pattern_language_for_o10297_shape` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3625` `fn anyof_drops_subsumed_open_object_branch_for_o83993_shape` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3662` `fn anyof_drops_recursive_open_object_branches_subsumed_by_base_node` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3745` `fn anyof_does_not_drop_open_object_branch_that_widens_base_property` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3771` `fn shadow_author_author_path_schema` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3799` `fn shadow_owner_owned_object_close_suppresses_residual_duplicate` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3808` `fn shadow_owner_missing_required_key_keeps_residual_open_branch` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3815` `fn shadow_owner_invalid_owner_fixed_type_keeps_residual_open_branch` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3822` `fn shadow_owner_invalid_date_time_string_keeps_residual_string_subtraction` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3832` `fn shadow_owner_out_of_order_fixed_fields_keep_residual_open_branch` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3842` `fn shadow_owner_skips_residual_with_unsafe_additional_constraints` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3871` `fn shadow_owner_allows_unsupported_optional_owner_fields` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3903` `fn shadow_owner_ref_branch_context_uses_factored_open_object_body` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3962` `fn single_anyof_object_ref_with_sibling_properties_merges_before_lowering` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:3989` `fn ref_with_sibling_assertions_is_intersected` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:4005` `fn singleton_allof_ref_without_siblings_reuses_ref_rule` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:4031` `fn singleton_allof_ref_with_noop_object_siblings_reuses_ref_rule` | ordinary internal symbol | keep near current phase |
| 3 | `src/import/json_schema/tests/mod.rs:4059` `fn singleton_allof_ref_with_restrictive_additional_properties_skips_fast_path` | ordinary internal symbol | keep near current phase |
