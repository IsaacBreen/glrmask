use crate::import::ast::GrammarExpr;

use super::ast::{ArraySchema, SchemaKind};
use super::error::ImportResult;
use super::lower::{choice, lit, never, seq, Lowerer};

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_array(&mut self, schema: &ArraySchema) -> ImportResult<GrammarExpr> {
        if schema.max_items.is_some_and(|max| max < schema.min_items) {
            return Ok(never());
        }

        let body = if schema.prefix_items.is_empty() {
            let item = self.lower_schema(&schema.items)?;
            self.array_body(item, schema.min_items, schema.max_items)
        } else {
            self.lower_tuple_array_body(schema)?
        };
        Ok(seq(vec![lit("["), body, lit("]")]))
    }

    fn array_body(&self, item: GrammarExpr, min: usize, max: Option<usize>) -> GrammarExpr {
        match max {
            Some(0) => GrammarExpr::Epsilon,
            Some(max) => GrammarExpr::SeparatedSequence {
                items: vec![(
                    GrammarExpr::RepeatRange { expr: Box::new(item), min, max },
                    min > 0,
                )],
                separator: Box::new(self.item_separator_expr()),
                allow_empty: min == 0,
            },
            None if min == 0 => GrammarExpr::SeparatedSequence {
                items: vec![(GrammarExpr::RepeatOne(Box::new(item)), false)],
                separator: Box::new(self.item_separator_expr()),
                allow_empty: true,
            },
            None => {
                let mut items = (0..min)
                    .map(|_| (item.clone(), true))
                    .collect::<Vec<_>>();
                items.push((GrammarExpr::RepeatOne(Box::new(item)), false));
                GrammarExpr::SeparatedSequence {
                    items,
                    separator: Box::new(self.item_separator_expr()),
                    allow_empty: false,
                }
            }
        }
    }

    fn lower_tuple_array_body(&mut self, schema: &ArraySchema) -> ImportResult<GrammarExpr> {
        let tail_allowed = !matches!(schema.items.kind, SchemaKind::Never)
            && schema.max_items.is_none_or(|max| max > schema.prefix_items.len());

        let effective_max = if tail_allowed {
            schema.max_items
        } else {
            Some(schema.max_items.unwrap_or(schema.prefix_items.len()).min(schema.prefix_items.len()))
        };
        if effective_max.is_some_and(|max| max < schema.min_items) {
            return Ok(never());
        }

        let prefix_len = schema.prefix_items.len();
        let finite_prefix_max = effective_max.unwrap_or(prefix_len).min(prefix_len);
        let finite_prefix_min = schema.min_items.min(finite_prefix_max);
        let mut alternatives = Vec::new();

        for len in finite_prefix_min..=finite_prefix_max {
            if len >= schema.min_items {
                alternatives.push(self.fixed_array_items(&schema.prefix_items[..len])?);
            }
        }

        if tail_allowed {
            let tail_max = schema.max_items.map(|max| max.saturating_sub(prefix_len));
            let tail_min = schema.min_items.saturating_sub(prefix_len).max(1);
            let tail = self.lower_schema(&schema.items)?;
            if tail_max != Some(0) {
                let mut items = Vec::new();
                for prefix in &schema.prefix_items {
                    items.push((self.lower_schema(prefix)?, true));
                }
                items.extend(self.tuple_tail_items(tail, tail_min, tail_max));
                alternatives.push(GrammarExpr::SeparatedSequence {
                    items,
                    separator: Box::new(self.item_separator_expr()),
                    allow_empty: false,
                });
            }
        }

        if alternatives.is_empty() {
            Ok(GrammarExpr::Epsilon)
        } else {
            Ok(choice(alternatives))
        }
    }

    fn fixed_array_items(&mut self, items: &[super::ast::Schema]) -> ImportResult<GrammarExpr> {
        if items.is_empty() {
            return Ok(GrammarExpr::Epsilon);
        }
        Ok(GrammarExpr::SeparatedSequence {
            items: items
                .iter()
                .map(|schema| self.lower_schema(schema).map(|expr| (expr, true)))
                .collect::<ImportResult<Vec<_>>>()?,
            separator: Box::new(self.item_separator_expr()),
            allow_empty: false,
        })
    }

    fn tuple_tail_items(
        &self,
        item: GrammarExpr,
        required_min: usize,
        max: Option<usize>,
    ) -> Vec<(GrammarExpr, bool)> {
        match max {
            Some(0) => Vec::new(),
            Some(max) => vec![(
                GrammarExpr::RepeatRange { expr: Box::new(item), min: required_min, max },
                required_min > 0,
            )],
            None if required_min == 0 => vec![(GrammarExpr::RepeatOne(Box::new(item)), false)],
            None => vec![
                (
                    GrammarExpr::RepeatRange {
                        expr: Box::new(item.clone()),
                        min: required_min,
                        max: required_min,
                    },
                    true,
                ),
                (GrammarExpr::RepeatOne(Box::new(item)), false),
            ],
        }
    }
}
