# Normalization function catalog

Functions that perform schema algebra, factoring, and merge logic.

| Line | Symbol | Priority | Publication action |
|---:|---|---:|---|
| 24 | `impl<'a> Lowerer<'a> {` | 1 | high-prominence boundary; document before modifying |
| 25 | `pub(crate) fn lower_any_of(` | 1 | high-prominence boundary; document before modifying |
| 86 | `fn try_merge_single_object_any_of_with_siblings(` | 2 | hotspot helper; split or proof-comment in follow-up |
| 110 | `pub(crate) fn lower_one_of(&mut self, assertions: &SchemaAssertions) -> ImportResult<GrammarExpr> {` | 1 | high-prominence boundary; document before modifying |
| 120 | `pub(crate) fn lower_all_of(&mut self, assertions: &SchemaAssertions) -> ImportResult<GrammarExpr> {` | 1 | high-prominence boundary; document before modifying |
| 217 | `fn try_lower_single_ref_with_object_siblings(` | 2 | hotspot helper; split or proof-comment in follow-up |
| 265 | `fn inline_all_of_refs(&self, branches: &[Schema]) -> ImportResult<Vec<Schema>> {` | 1 | high-prominence boundary; document before modifying |
| 272 | `fn inline_all_of_refs_for_any_of_factoring(` | 2 | hotspot helper; split or proof-comment in follow-up |
| 286 | `fn schema_transitively_refs_pointer(` | 3 | keep in current phase, add exactness comment if semantic |
| 356 | `fn inline_all_of_ref_target(&self, pointer: &str, fallback: &Schema) -> ImportResult<Schema> {` | 1 | high-prominence boundary; document before modifying |
| 385 | `fn try_inline_object_like_all_of_target(&self, target: &Schema) -> ImportResult<Option<Schema>> {` | 1 | high-prominence boundary; document before modifying |
| 400 | `fn try_rewrite_all_of_object_choice_target(&self, target: &Schema) -> ImportResult<Option<Schema>> {` | 1 | high-prominence boundary; document before modifying |
| 449 | `fn inline_refs_in_all_of_branch(&self, branch: &Schema) -> ImportResult<Schema> {` | 1 | high-prominence boundary; document before modifying |
| 468 | `fn try_merge_all_of_single_ref_object_branches(` | 2 | hotspot helper; split or proof-comment in follow-up |
| 505 | `fn drop_subsumed_open_object_any_of_branches(` | 2 | hotspot helper; split or proof-comment in follow-up |
| 555 | `fn object_branch_resolved<'schema>(` | 3 | keep in current phase, add exactness comment if semantic |
| 565 | `fn object_schema_subsumes(` | 3 | keep in current phase, add exactness comment if semantic |
| 612 | `fn schema_subsumes(` | 3 | keep in current phase, add exactness comment if semantic |
| 691 | `fn all_of_intersection_terminal_safe(expr: &GrammarExpr) -> bool {` | 3 | keep in current phase, add exactness comment if semantic |
| 727 | `fn explicit_all_of_type_intersection(branches: &[Schema]) -> Option<BTreeSet<SchemaType>> {` | 1 | high-prominence boundary; document before modifying |
| 748 | `fn untyped_single_family_assertion(schema: &Schema) -> Option<SchemaType> {` | 1 | high-prominence boundary; document before modifying |
| 781 | `fn family_overlaps_types(family: SchemaType, types: &BTreeSet<SchemaType>) -> bool {` | 1 | high-prominence boundary; document before modifying |
| 790 | `fn drop_vacuous_untyped_family_branches(branches: Vec<Schema>) -> Option<Vec<Schema>> {` | 1 | high-prominence boundary; document before modifying |
| 809 | `fn flatten_pure_all_of_branches(branches: Vec<Schema>) -> Vec<Schema> {` | 1 | high-prominence boundary; document before modifying |
| 825 | `fn collapse_pure_single_choice_branches(branches: Vec<Schema>) -> Vec<Schema> {` | 1 | high-prominence boundary; document before modifying |
| 838 | `fn try_factor_required_property_any_of(` | 2 | hotspot helper; split or proof-comment in follow-up |
| 884 | `fn try_factor_closed_object_variant_any_of(` | 2 | hotspot helper; split or proof-comment in follow-up |
| 989 | `fn try_factor_mutually_exclusive_property_not_any_of(` | 2 | hotspot helper; split or proof-comment in follow-up |
| 1041 | `fn mutually_exclusive_property_not_branch(schema: &Schema) -> Option<(&PropertySchema, String)> {` | 1 | high-prominence boundary; document before modifying |
| 1078 | `fn single_required_object_not_name(schema: &Schema) -> Option<&str> {` | 1 | high-prominence boundary; document before modifying |
| 1110 | `fn single_required_object_branch_name(schema: &Schema) -> Option<&str> {` | 1 | high-prominence boundary; document before modifying |
| 1143 | `fn closed_object_variant_branch(schema: &Schema) -> Option<&ObjectSchema> {` | 1 | high-prominence boundary; document before modifying |
| 1176 | `pub(crate) fn open_object_any_of_covers_json_object(branches: &[Schema]) -> bool {` | 1 | high-prominence boundary; document before modifying |
| 1211 | `fn object_branch(schema: &Schema) -> Option<&ObjectSchema> {` | 1 | high-prominence boundary; document before modifying |
| 1236 | `fn property_schema_by_name<'a>(object: &'a ObjectSchema, name: &str) -> Option<&'a Schema> {` | 1 | high-prominence boundary; document before modifying |
| 1244 | `fn schema_subsumption_key(schema: &Schema) -> ImportResult<String> {` | 1 | high-prominence boundary; document before modifying |
| 1251 | `fn pure_any_of_assertions(assertions: &SchemaAssertions) -> bool {` | 1 | high-prominence boundary; document before modifying |
| 1265 | `fn broad_string_assertions(assertions: &SchemaAssertions) -> Option<&super::super::schema::StringSchema> {` | 1 | high-prominence boundary; document before modifying |
| 1288 | `fn string_literal_values(assertions: &SchemaAssertions) -> Option<Vec<&serde_json::Value>> {` | 1 | high-prominence boundary; document before modifying |
| 1311 | `fn schemas_shape_equivalent(left: &Schema, right: &Schema) -> bool {` | 1 | high-prominence boundary; document before modifying |
| 1332 | `fn option_schemas_shape_equivalent(left: Option<&Schema>, right: Option<&Schema>) -> bool {` | 1 | high-prominence boundary; document before modifying |
| 1340 | `fn option_objects_shape_equivalent(left: Option<&ObjectSchema>, right: Option<&ObjectSchema>) -> bool {` | 1 | high-prominence boundary; document before modifying |
| 1348 | `fn object_schemas_shape_equivalent(left: &ObjectSchema, right: &ObjectSchema) -> bool {` | 1 | high-prominence boundary; document before modifying |
| 1371 | `fn additional_properties_shape_equivalent(` | 2 | hotspot helper; split or proof-comment in follow-up |
| 1385 | `fn option_arrays_shape_equivalent(` | 3 | keep in current phase, add exactness comment if semantic |
| 1401 | `fn option_strings_shape_equivalent(` | 3 | keep in current phase, add exactness comment if semantic |
| 1417 | `fn option_numbers_shape_equivalent(` | 3 | keep in current phase, add exactness comment if semantic |
| 1435 | `fn schema_slices_shape_equivalent(left: &[Schema], right: &[Schema]) -> bool {` | 1 | high-prominence boundary; document before modifying |
| 1443 | `fn sibling_assertion_schema(assertions: &SchemaAssertions) -> Option<Schema> {` | 1 | high-prominence boundary; document before modifying |
| 1452 | `fn branch_with_siblings(branch: Schema, siblings: Option<Schema>) -> Schema {` | 1 | high-prominence boundary; document before modifying |
| 1464 | `fn push_object_only_type_into_branch(branch: &Schema) -> Option<Schema> {` | 1 | high-prominence boundary; document before modifying |
| 1489 | `fn schema_contains_ref(schema: &Schema) -> bool {` | 1 | high-prominence boundary; document before modifying |
| 1502 | `fn schema_has_explicit_object_only_type(schema: &Schema) -> bool {` | 1 | high-prominence boundary; document before modifying |
| 1512 | `pub(crate) fn try_merge_all_of_objects(branches: &[Schema]) -> Option<ObjectSchema> {` | 1 | high-prominence boundary; document before modifying |
| 1521 | `fn plain_object_schema(schema: &Schema) -> Option<&ObjectSchema> {` | 1 | high-prominence boundary; document before modifying |
| 1542 | `enum ChoiceKind {` | 3 | keep in current phase, add exactness comment if semantic |
| 1547 | `fn pure_choice_branch(schema: &Schema) -> Option<(ChoiceKind, &[Schema])> {` | 1 | high-prominence boundary; document before modifying |
| 1570 | `fn distribute_all_of_over_single_object_choice(` | 3 | keep in current phase, add exactness comment if semantic |
| 1613 | `fn schema_is_object_like(schema: &Schema) -> bool {` | 1 | high-prominence boundary; document before modifying |
| 1617 | `fn merge_all_of_object_like_schema(branches: &[Schema]) -> Option<Schema> {` | 1 | high-prominence boundary; document before modifying |
| 1658 | `fn object_like_schema(schema: &Schema) -> Option<Schema> {` | 1 | high-prominence boundary; document before modifying |
| 1703 | `fn merge_all_of_array_like_schema(branches: &[Schema]) -> Option<Schema> {` | 1 | high-prominence boundary; document before modifying |
| 1742 | `fn plain_array_schema(schema: &Schema) -> Option<(&ArraySchema, bool)> {` | 1 | high-prominence boundary; document before modifying |
| 1765 | `fn array_is_bounds_only(array: &ArraySchema) -> bool {` | 1 | high-prominence boundary; document before modifying |
| 1770 | `fn merge_array_bounds(left: &mut ArraySchema, right: &ArraySchema) {` | 1 | high-prominence boundary; document before modifying |
| 1780 | `fn merge_two_objects(left: &ObjectSchema, right: &ObjectSchema) -> ObjectSchema {` | 1 | high-prominence boundary; document before modifying |
| 1810 | `fn merge_property_schemas(left: Schema, right: Schema) -> Schema {` | 1 | high-prominence boundary; document before modifying |
| 1820 | `fn is_vacuous_json_value_schema(schema: &Schema) -> bool {` | 1 | high-prominence boundary; document before modifying |
| 1850 | `fn is_vacuous_object_schema(schema: &Schema) -> bool {` | 1 | high-prominence boundary; document before modifying |
| 1874 | `fn merge_additional_properties(` | 2 | hotspot helper; split or proof-comment in follow-up |
| 1894 | `pub(crate) fn all_of_schema(left: Schema, right: Schema) -> Schema {` | 1 | high-prominence boundary; document before modifying |
