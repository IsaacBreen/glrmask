use crate::import::ast::GrammarExpr;

use super::ast::{
    AdditionalProperties, ObjectSchema, Schema, SchemaAssertions, SchemaKind,
    SchemaType,
};
use super::error::{ImportResult, SchemaImportError};
use super::lower::{choice, r, Lowerer, JSON_VALUE_RULE};

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_any_of(&mut self, assertions: &SchemaAssertions) -> ImportResult<GrammarExpr> {
        reject_sibling_assertions("anyOf", assertions)?;
        let alternatives = assertions
            .any_of
            .iter()
            .map(|branch| self.lower_schema(branch))
            .collect::<ImportResult<Vec<_>>>()?;
        Ok(choice(alternatives))
    }

    pub(crate) fn lower_one_of(&mut self, assertions: &SchemaAssertions) -> ImportResult<GrammarExpr> {
        reject_sibling_assertions("oneOf", assertions)?;
        let alternatives = assertions
            .one_of
            .iter()
            .map(|branch| self.lower_schema(branch))
            .collect::<ImportResult<Vec<_>>>()?;
        Ok(choice(alternatives))
    }

    pub(crate) fn lower_all_of(&mut self, assertions: &SchemaAssertions) -> ImportResult<GrammarExpr> {
        let mut branches = assertions.all_of.clone();
        let siblings = assertions.clone_without_combinators();
        if siblings.has_value_assertions_without_combinators() {
            branches.push(Schema::assertions("<allOf-siblings>", siblings));
        }

        if branches.is_empty() {
            return Ok(r(JSON_VALUE_RULE));
        }
        if let Some(object) = try_merge_all_of_objects(&branches) {
            return self.lower_object(&object);
        }

        let mut lowered = branches
            .iter()
            .map(|branch| self.lower_schema(branch))
            .collect::<ImportResult<Vec<_>>>()?;
        if lowered.is_empty() {
            return Ok(r(JSON_VALUE_RULE));
        }
        let first = lowered.remove(0);
        Ok(lowered.into_iter().fold(first, |left, right| GrammarExpr::Intersect {
            expr: Box::new(left),
            intersect: Box::new(right),
        }))
    }
}

fn reject_sibling_assertions(name: &str, assertions: &SchemaAssertions) -> ImportResult<()> {
    if assertions.has_value_assertions_without_combinators() {
        Err(SchemaImportError::new(format!(
            "{name} with sibling assertions is unsupported by the simple importer"
        )))
    } else {
        Ok(())
    }
}

fn try_merge_all_of_objects(branches: &[Schema]) -> Option<ObjectSchema> {
    let mut objects = branches.iter().map(plain_object_schema).collect::<Option<Vec<_>>>()?;
    let mut merged = objects.remove(0).clone();
    for object in objects {
        merged = merge_two_objects(&merged, object);
    }
    Some(merged)
}

fn plain_object_schema(schema: &Schema) -> Option<&ObjectSchema> {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return None;
    };
    if assertions.const_value.is_some()
        || assertions.enum_values.is_some()
        || !assertions.any_of.is_empty()
        || !assertions.one_of.is_empty()
        || !assertions.all_of.is_empty()
    {
        return None;
    }
    if let Some(types) = &assertions.types {
        if !types.iter().all(|schema_type| *schema_type == SchemaType::Object) {
            return None;
        }
    }
    assertions.object.as_ref()
}

fn merge_two_objects(left: &ObjectSchema, right: &ObjectSchema) -> ObjectSchema {
    let mut merged = left.clone();

    for required in &right.required {
        merged.required.insert(required.clone());
    }

    for property in &right.properties {
        if let Some(existing) = merged.properties.iter_mut().find(|candidate| candidate.name == property.name) {
            existing.schema = all_of_schema(existing.schema.clone(), property.schema.clone());
        } else {
            merged.properties.push(property.clone());
        }
    }

    merged.pattern_properties.extend(right.pattern_properties.clone());
    let additional_properties = merge_additional_properties(
        &merged.additional_properties,
        &right.additional_properties,
    );
    merged.additional_properties = additional_properties;
    merged
}

fn merge_additional_properties(
    left: &AdditionalProperties,
    right: &AdditionalProperties,
) -> AdditionalProperties {
    match (left, right) {
        (AdditionalProperties::Deny, _) | (_, AdditionalProperties::Deny) => AdditionalProperties::Deny,
        (AdditionalProperties::AllowAny, AdditionalProperties::AllowAny) => AdditionalProperties::AllowAny,
        (AdditionalProperties::Schema(schema), AdditionalProperties::AllowAny)
        | (AdditionalProperties::AllowAny, AdditionalProperties::Schema(schema)) => {
            AdditionalProperties::Schema(schema.clone())
        }
        (AdditionalProperties::Schema(left), AdditionalProperties::Schema(right)) => {
            AdditionalProperties::Schema(Box::new(all_of_schema(
                left.as_ref().clone(),
                right.as_ref().clone(),
            )))
        }
    }
}

fn all_of_schema(left: Schema, right: Schema) -> Schema {
    Schema::assertions(
        "<merged-allOf-property>",
        SchemaAssertions {
            all_of: vec![left, right],
            ..SchemaAssertions::default()
        },
    )
}
