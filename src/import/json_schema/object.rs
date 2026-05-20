use std::collections::BTreeSet;

use crate::import::ast::GrammarExpr;

use super::ast::{
    AdditionalProperties, ObjectSchema, PatternPropertySchema,
    PropertySchema, Schema, SchemaAssertions, SchemaKind, SchemaType,
};
use super::combinators::all_of_schema;
use super::error::{ImportResult, SchemaImportError};
use super::lower::{choice, lit, r, seq, Lowerer, JSON_VALUE_RULE};
use super::string::property_name_matches_pattern;

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_object(&mut self, schema: &ObjectSchema) -> ImportResult<GrammarExpr> {
        let normalized = self.object_with_required_synthetic_properties(schema)?;
        let fixed_names = normalized
            .properties
            .iter()
            .map(|property| property.name.clone())
            .collect::<BTreeSet<_>>();
        let pattern_keys = normalized
            .pattern_properties
            .iter()
            .map(|pattern_property| pattern_property.pattern.clone())
            .collect::<Vec<_>>();

        let mut items = normalized
            .properties
            .iter()
            .map(|property| {
                let required = normalized.required.contains(&property.name);
                self.lower_property_pair(property, &normalized.pattern_properties)
                    .map(|pair| (pair, required))
            })
            .collect::<ImportResult<Vec<_>>>()?;

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
                let mut pattern_key_exclusions = Vec::new();
                for pattern in &pattern_keys {
                    pattern_key_exclusions.push(self.lower_pattern_key_colon_terminal(pattern)?);
                }
                tail_pairs.push(seq(vec![
                    self.lower_additional_key_colon(&fixed_names, &pattern_key_exclusions),
                    r(JSON_VALUE_RULE),
                ]));
            }
            AdditionalProperties::Deny => {}
            AdditionalProperties::Schema(value_schema) => {
                let mut pattern_key_exclusions = Vec::new();
                for pattern in &pattern_keys {
                    pattern_key_exclusions.push(self.lower_pattern_key_colon_terminal(pattern)?);
                }
                let value = self.lower_schema(value_schema)?;
                tail_pairs.push(seq(vec![
                    self.lower_additional_key_colon(&fixed_names, &pattern_key_exclusions),
                    value,
                ]));
            }
        }

        if !tail_pairs.is_empty() {
            items.push((
                GrammarExpr::RepeatOne(Box::new(choice(tail_pairs))),
                false,
            ));
        }

        let body = if items.is_empty() {
            GrammarExpr::Epsilon
        } else {
            GrammarExpr::SeparatedSequence {
                items,
                separator: Box::new(self.item_separator_expr()),
                allow_empty: true,
            }
        };

        Ok(seq(vec![lit("{"), body, lit("}")]))
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
