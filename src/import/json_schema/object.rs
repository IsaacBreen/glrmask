use std::collections::BTreeSet;

use crate::grammar::expr_nfa::ExprNfaBuilder;
use crate::import::ast::GrammarExpr;

use super::ast::{
    AdditionalProperties, ObjectSchema, PatternPropertySchema,
    PropertySchema, Schema, SchemaAssertions, SchemaKind, SchemaType,
};
use super::combinators::all_of_schema;
use super::error::{ImportResult, SchemaImportError};
use super::lower::{choice, lit, r, seq, Lowerer, JSON_VALUE_RULE};
use super::string::property_name_matches_pattern;

struct ObjectItem {
    pair: GrammarExpr,
    required: bool,
    satisfies_any_group: bool,
    exclusive_group: bool,
}

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_object(&mut self, schema: &ObjectSchema) -> ImportResult<GrammarExpr> {
        self.lower_object_internal(schema, None, None)
    }

    pub(crate) fn lower_object_requiring_any_property(
        &mut self,
        schema: &ObjectSchema,
        any_required_names: &BTreeSet<String>,
    ) -> ImportResult<GrammarExpr> {
        self.lower_object_internal(schema, Some(any_required_names), None)
    }

    pub(crate) fn lower_object_with_exclusive_properties(
        &mut self,
        schema: &ObjectSchema,
        exclusive_names: &BTreeSet<String>,
        require_one: bool,
    ) -> ImportResult<GrammarExpr> {
        self.lower_object_internal(schema, None, Some((exclusive_names, require_one)))
    }

    fn lower_object_internal(
        &mut self,
        schema: &ObjectSchema,
        any_required_names: Option<&BTreeSet<String>>,
        exclusive_group: Option<(&BTreeSet<String>, bool)>,
    ) -> ImportResult<GrammarExpr> {
        let normalized = self.object_with_required_synthetic_properties(schema)?;
        let fixed_names = normalized
            .properties
            .iter()
            .map(|property| property.name.clone())
            .collect::<BTreeSet<_>>();
        let mut items = normalized
            .properties
            .iter()
            .map(|property| {
                let required = normalized.required.contains(&property.name);
                self.lower_property_pair(property, &normalized.pattern_properties)
                    .map(|pair| ObjectItem {
                        pair,
                        required,
                        satisfies_any_group: any_required_names
                            .is_some_and(|names| names.contains(&property.name)),
                        exclusive_group: exclusive_group
                            .is_some_and(|(names, _)| names.contains(&property.name)),
                    })
            })
            .collect::<ImportResult<Vec<_>>>()?;

        if normalized.pattern_properties.is_empty() && !normalized.properties.is_empty() {
            let tail_pair = match &normalized.additional_properties {
                AdditionalProperties::Deny => None,
                AdditionalProperties::AllowAny => Some(seq(vec![
                    self.lower_additional_key_colon(&fixed_names, &[])?,
                    r(JSON_VALUE_RULE),
                ])),
                AdditionalProperties::Schema(value_schema) => {
                    let value = self.lower_schema(value_schema)?;
                    Some(seq(vec![
                        self.lower_additional_key_colon(&fixed_names, &[])?,
                        value,
                    ]))
                }
            };
            return self.lower_fixed_object_body_exprnfa(
                &items,
                tail_pair,
                any_required_names.is_some(),
                exclusive_group.is_some_and(|(_, require_one)| require_one),
            );
        }

        if any_required_names.is_some() || exclusive_group.is_some() {
            return Err(SchemaImportError::new(
                "grouped anyOf object factoring requires fixed object properties without patternProperties"
                    .to_string(),
            ));
        }

        let pattern_keys = normalized
            .pattern_properties
            .iter()
            .map(|pattern_property| pattern_property.pattern.clone())
            .collect::<Vec<_>>();

        let mut tail_pairs = Vec::new();
        for pattern_property in &normalized.pattern_properties {
            let key = if fixed_names.is_empty() {
                self.lower_pattern_key_colon(&pattern_property.pattern)?
            } else {
                self.lower_pattern_key_colon_excluding(&pattern_property.pattern, &fixed_names)?
            };
            let value = self.lower_schema(&pattern_property.schema)?;
            tail_pairs.push(seq(vec![key, value]));
        }

        match &normalized.additional_properties {
            AdditionalProperties::AllowAny => {
                tail_pairs.push(seq(vec![
                    self.lower_additional_key_colon(&fixed_names, &pattern_keys)?,
                    r(JSON_VALUE_RULE),
                ]));
            }
            AdditionalProperties::Deny => {}
            AdditionalProperties::Schema(value_schema) => {
                let value = self.lower_schema(value_schema)?;
                tail_pairs.push(seq(vec![
                    self.lower_additional_key_colon(&fixed_names, &pattern_keys)?,
                    value,
                ]));
            }
        }

        if !tail_pairs.is_empty() {
            items.push(ObjectItem {
                pair: GrammarExpr::RepeatOne(Box::new(choice(tail_pairs))),
                required: false,
                satisfies_any_group: false,
                exclusive_group: false,
            });
        }

        let body = if items.is_empty() {
            GrammarExpr::Epsilon
        } else {
            GrammarExpr::SeparatedSequence {
                items: items.into_iter().map(|item| (item.pair, item.required)).collect(),
                separator: Box::new(self.item_separator_expr()),
                allow_empty: true,
            }
        };

        Ok(seq(vec![lit("{"), body, lit("}")]))
    }

    fn lower_fixed_object_body_exprnfa(
        &mut self,
        items: &[ObjectItem],
        tail_pair: Option<GrammarExpr>,
        any_group_required: bool,
        exclusive_require_one: bool,
    ) -> ImportResult<GrammarExpr> {
        if items.iter().all(|item| !item.satisfies_any_group && !item.exclusive_group) {
            return self.lower_fixed_object_body_exprnfa_without_group(items, tail_pair);
        }

        let mut builder = ExprNfaBuilder::new();
        let mut states = vec![[[[0u32; 2]; 2]; 2]; items.len() + 1];
        for state_set in &mut states {
            for separator_seen in 0..=1 {
                for any_group_seen in 0..=1 {
                    for exclusive_seen in 0..=1 {
                        state_set[separator_seen][any_group_seen][exclusive_seen] = builder.add_state();
                    }
                }
            }
        }
        states[0][0][0][0] = builder.start_state();

        for (index, item) in items.iter().enumerate() {
            let separator_pair = seq(vec![self.item_separator_expr(), item.pair.clone()]);
            for separator_seen in 0..=1 {
                for any_group_seen in 0..=1 {
                    for exclusive_seen in 0..=1 {
                        if !item.required {
                            builder.add_epsilon(
                                states[index][separator_seen][any_group_seen][exclusive_seen],
                                states[index + 1][separator_seen][any_group_seen][exclusive_seen],
                            );
                        }
                        if item.exclusive_group && exclusive_seen == 1 {
                            continue;
                        }
                        let next_any_group_seen =
                            usize::from(any_group_seen == 1 || item.satisfies_any_group);
                        let next_exclusive_seen =
                            usize::from(exclusive_seen == 1 || item.exclusive_group);
                        let transition_expr = if separator_seen == 0 {
                            item.pair.clone()
                        } else {
                            separator_pair.clone()
                        };
                        builder.add_transition(
                            states[index][separator_seen][any_group_seen][exclusive_seen],
                            transition_expr,
                            states[index + 1][1][next_any_group_seen][next_exclusive_seen],
                        );
                    }
                }
            }
        }

        for separator_seen in 0..=1 {
            for any_group_seen in 0..=1 {
                if any_group_required && any_group_seen == 0 {
                    continue;
                }
                for exclusive_seen in 0..=1 {
                    if exclusive_require_one && exclusive_seen == 0 {
                        continue;
                    }
                    builder.set_accepting(states[items.len()][separator_seen][any_group_seen][exclusive_seen]);
                }
            }
        }

        if let Some(tail_pair_expr) = tail_pair {
            for any_group_seen in 0..=1 {
                for exclusive_seen in 0..=1 {
                    let tail_state = builder.add_state();
                    if (!any_group_required || any_group_seen == 1)
                        && (!exclusive_require_one || exclusive_seen == 1)
                    {
                        builder.set_accepting(tail_state);
                    }
                    builder.add_transition(
                        states[items.len()][0][any_group_seen][exclusive_seen],
                        tail_pair_expr.clone(),
                        tail_state,
                    );
                    builder.add_transition(
                        states[items.len()][1][any_group_seen][exclusive_seen],
                        seq(vec![self.item_separator_expr(), tail_pair_expr.clone()]),
                        tail_state,
                    );
                    builder.add_transition(
                        tail_state,
                        seq(vec![self.item_separator_expr(), tail_pair_expr.clone()]),
                        tail_state,
                    );
                }
            }
        }

        let rule_name = self.fresh_rule_name("json_closed_object_body");
        let body = GrammarExpr::ExprNFA(Box::new(builder.build().into_determinized_and_minimized()));
        self.add_nonterminal_rule(&rule_name, body);

        Ok(seq(vec![lit("{"), r(&rule_name), lit("}")]))
    }

    fn lower_fixed_object_body_exprnfa_without_group(
        &mut self,
        items: &[ObjectItem],
        tail_pair: Option<GrammarExpr>,
    ) -> ImportResult<GrammarExpr> {
        let mut builder = ExprNfaBuilder::new();
        let mut states = vec![[0u32; 2]; items.len() + 1];
        states[0][0] = builder.start_state();
        states[0][1] = builder.add_state();
        for state_pair in states.iter_mut().skip(1) {
            state_pair[0] = builder.add_state();
            state_pair[1] = builder.add_state();
        }

        for (index, item) in items.iter().enumerate() {
            let separator_pair = seq(vec![self.item_separator_expr(), item.pair.clone()]);
            if !item.required {
                builder.add_epsilon(states[index][0], states[index + 1][0]);
                builder.add_epsilon(states[index][1], states[index + 1][1]);
            }
            builder.add_transition(states[index][0], item.pair.clone(), states[index + 1][1]);
            builder.add_transition(states[index][1], separator_pair, states[index + 1][1]);
        }

        builder.set_accepting(states[items.len()][0]);
        builder.set_accepting(states[items.len()][1]);

        if let Some(tail_pair_expr) = tail_pair {
            let tail_state = builder.add_state();
            builder.set_accepting(tail_state);
            builder.add_transition(states[items.len()][0], tail_pair_expr.clone(), tail_state);
            builder.add_transition(
                states[items.len()][1],
                seq(vec![self.item_separator_expr(), tail_pair_expr.clone()]),
                tail_state,
            );
            builder.add_transition(
                tail_state,
                seq(vec![self.item_separator_expr(), tail_pair_expr]),
                tail_state,
            );
        }

        let rule_name = self.fresh_rule_name("json_closed_object_body");
        let body = GrammarExpr::ExprNFA(Box::new(builder.build().into_determinized_and_minimized()));
        self.add_nonterminal_rule(&rule_name, body);

        Ok(seq(vec![lit("{"), r(&rule_name), lit("}")]))
    }

    fn lower_property_pair(
        &mut self,
        property: &PropertySchema,
        pattern_properties: &[PatternPropertySchema],
    ) -> ImportResult<GrammarExpr> {
        let key = self.lower_literal_key_colon(&property.name);
        let mut effective_schema = property.schema.clone();
        for pattern_property in pattern_properties {
            if property_matches_pattern(&pattern_property.pattern, &property.name)? {
                let pattern_schema = pattern_schema_for_property(&effective_schema, &pattern_property.schema);
                effective_schema = all_of_schema(effective_schema, pattern_schema);
            }
        }
        let value = self.lower_schema(&effective_schema)?;
        Ok(seq(vec![key, value]))
    }

    fn object_with_required_synthetic_properties(
        &self,
        schema: &ObjectSchema,
    ) -> ImportResult<ObjectSchema> {
        let mut normalized = schema.clone();
        let mut known = normalized
            .properties
            .iter()
            .map(|property| property.name.clone())
            .collect::<BTreeSet<_>>();

        for required_name in &schema.required {
            if known.contains(required_name) {
                continue;
            }

            let synthetic_schema = match &schema.additional_properties {
                AdditionalProperties::AllowAny => Schema::any(format!("<required:{required_name}>")),
                AdditionalProperties::Schema(schema) => schema.as_ref().clone(),
                AdditionalProperties::Deny => {
                    return Err(SchemaImportError::new(format!(
                        "required property {required_name:?} is not listed in properties and additionalProperties is false"
                    )));
                }
            };
            normalized.properties.push(PropertySchema {
                name: required_name.clone(),
                schema: synthetic_schema,
            });
            known.insert(required_name.clone());
        }

        Ok(normalized)
    }
}

fn property_matches_pattern(pattern: &str, property_name: &str) -> ImportResult<bool> {
    property_name_matches_pattern(pattern, property_name)
}

fn pattern_schema_for_property(property_schema: &Schema, pattern_schema: &Schema) -> Schema {
    let Some(property_type) = single_numeric_property_type(property_schema) else {
        return pattern_schema.clone();
    };

    let SchemaKind::Assertions(assertions) = &pattern_schema.kind else {
        return pattern_schema.clone();
    };
    if assertions.types.is_some() || assertions.number.is_none() || has_non_numeric_assertions(assertions) {
        return pattern_schema.clone();
    }

    let mut typed = assertions.as_ref().clone();
    typed.types = Some(vec![property_type]);
    Schema::assertions(pattern_schema.location.clone(), typed)
}

fn single_numeric_property_type(property_schema: &Schema) -> Option<SchemaType> {
    let SchemaKind::Assertions(assertions) = &property_schema.kind else {
        return None;
    };
    match assertions.types.as_deref() {
        Some([SchemaType::Integer]) => Some(SchemaType::Integer),
        Some([SchemaType::Number]) => Some(SchemaType::Number),
        _ => None,
    }
}

fn has_non_numeric_assertions(assertions: &SchemaAssertions) -> bool {
    assertions.const_value.is_some()
        || assertions.enum_values.is_some()
        || assertions.object.is_some()
        || assertions.array.is_some()
        || assertions.string.is_some()
        || !assertions.any_of.is_empty()
        || !assertions.one_of.is_empty()
        || !assertions.all_of.is_empty()
}
