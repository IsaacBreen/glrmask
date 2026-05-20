use std::collections::BTreeSet;

use crate::import::ast::GrammarExpr;

use super::ast::{AdditionalProperties, ObjectSchema, PropertySchema, Schema};
use super::error::{ImportResult, SchemaImportError};
use super::lower::{choice, lit, r, seq, Lowerer, JSON_VALUE_RULE};

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_object(&mut self, schema: &ObjectSchema) -> ImportResult<GrammarExpr> {
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
                self.lower_property_pair(property).map(|pair| (pair, required))
            })
            .collect::<ImportResult<Vec<_>>>()?;

        let mut tail_pairs = Vec::new();
        for pattern_property in &normalized.pattern_properties {
            let key = self.lower_pattern_key_colon(&pattern_property.pattern)?;
            let value = self.lower_schema(&pattern_property.schema)?;
            tail_pairs.push(seq(vec![key, value]));
        }

        match &normalized.additional_properties {
            AdditionalProperties::AllowAny => {
                tail_pairs.push(seq(vec![
                    self.lower_additional_key_colon(&fixed_names),
                    r(JSON_VALUE_RULE),
                ]));
            }
            AdditionalProperties::Deny => {}
            AdditionalProperties::Schema(value_schema) => {
                let value = self.lower_schema(value_schema)?;
                tail_pairs.push(seq(vec![self.lower_additional_key_colon(&fixed_names), value]));
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

    fn lower_property_pair(&mut self, property: &PropertySchema) -> ImportResult<GrammarExpr> {
        let key = self.lower_literal_key_colon(&property.name);
        let value = self.lower_schema(&property.schema)?;
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
