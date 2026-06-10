use crate::grammar::expr_nfa::ExprNfaBuilder;
use crate::import::ast::{GrammarExpr, Quantifier};

use super::ast::{ArraySchema, SchemaKind};
use super::error::ImportResult;
use super::lower::{choice, lit, never, seq, Lowerer};

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_array(&mut self, schema: &ArraySchema) -> ImportResult<GrammarExpr> {
        if schema.max_items.is_some_and(|max| max < schema.min_items) {
            return Ok(never());
        }

        if schema.prefix_items.is_empty()
            && let Some(max) = schema.max_items
            && max >= 2
        {
            if bounded_array_object_item_candidate(&schema.items) {
                let item = self.lower_schema(&schema.items)?;
                return Ok(self.bounded_homogeneous_array_exprnfa(item, schema.min_items, max));
            }
            if self.should_terminalize_bounded_scalar_array(max)
                && let Some(item) = self.lower_inline_bounded_array_string_item_expr(&schema.items)?
            {
                return Ok(self.bounded_homogeneous_array_terminal(item, schema.min_items, max));
            }
        }
        if schema.prefix_items.is_empty()
            && schema.min_items == 0
            && schema.max_items.is_none()
            && let Some(item) = self.lower_inline_bounded_array_string_item_expr(&schema.items)?
        {
            return Ok(self.unbounded_homogeneous_array_terminal(item));
        }

        let body = if schema.prefix_items.is_empty() {
            let item = self.lower_schema(&schema.items)?;
            self.array_body(item, schema.min_items, schema.max_items)
        } else {
            self.lower_tuple_array_body(schema)?
        };
        Ok(seq(vec![lit("["), body, lit("]")]))
    }

    fn should_terminalize_bounded_scalar_array(&self, max_items: usize) -> bool {
        max_items <= self.config.repeat_chunk_size.max(2)
    }

    fn array_body(&self, item: GrammarExpr, min: usize, max: Option<usize>) -> GrammarExpr {
        match max {
            Some(0) => GrammarExpr::Epsilon,
            Some(max) => GrammarExpr::SeparatedSequence {
                items: vec![(
                    item,
                    Some(Quantifier::Range(min, Some(max))),
                )],
                separator: Box::new(self.item_separator_expr()),
                allow_empty: min == 0,
            },
            None if min == 0 => {
                let separator_item = seq(vec![self.item_separator_expr(), item.clone()]);
                choice(vec![
                    GrammarExpr::Epsilon,
                    seq(vec![
                        item,
                        GrammarExpr::Quantified(Box::new(separator_item), Quantifier::ZeroPlus),
                    ]),
                ])
            }
            None => {
                let separator_item = seq(vec![self.item_separator_expr(), item.clone()]);
                let mut parts = vec![item.clone()];
                for _ in 1..min {
                    parts.push(separator_item.clone());
                }
                parts.push(GrammarExpr::Quantified(Box::new(separator_item), Quantifier::ZeroPlus));
                seq(parts)
            }
        }
    }

    fn bounded_homogeneous_array_exprnfa(
        &mut self,
        item: GrammarExpr,
        min: usize,
        max: usize,
    ) -> GrammarExpr {
        let mut builder = ExprNfaBuilder::new();
        let accept_state = builder.add_state();
        let mut item_states = Vec::with_capacity(max + 1);
        item_states.push(builder.start_state());
        for _ in 0..max {
            item_states.push(builder.add_state());
        }
        builder.set_accepting(accept_state);

        for &state in item_states.iter().skip(min) {
            builder.add_transition(state, lit("]"), accept_state);
        }

        for count in 0..max {
            let transition = if count == 0 {
                item.clone()
            } else {
                seq(vec![self.item_separator_expr(), item.clone()])
            };
            builder.add_transition(item_states[count], transition, item_states[count + 1]);
        }

        let rule_name = self.fresh_rule_name("bounded_array");
        self.add_nonterminal_rule(
            &rule_name,
            GrammarExpr::ExprNFA(Box::new(builder.build().into_determinized_and_minimized())),
        );
        seq(vec![lit("["), super::lower::r(&rule_name)])
    }

    fn bounded_homogeneous_array_terminal(
        &mut self,
        item: GrammarExpr,
        min: usize,
        max: usize,
    ) -> GrammarExpr {
        let separator_item = seq(vec![self.item_separator_expr(), item.clone()]);
        let body = if min == 0 {
            GrammarExpr::Quantified(Box::new(seq(vec![
                item,
                GrammarExpr::Quantified(Box::new(separator_item), Quantifier::Range(0, Some(max - 1))),
            ])), Quantifier::Optional)
        } else {
            seq(vec![
                item,
                GrammarExpr::Quantified(Box::new(separator_item), Quantifier::Range(min - 1, Some(max - 1))),
            ])
        };

        let rule_name = self.fresh_rule_name("bounded_scalar_array");
        self.add_terminal_rule(&rule_name, seq(vec![lit("["), body, lit("]")]));
        super::lower::r(&rule_name)
    }

    fn unbounded_homogeneous_array_terminal(&mut self, item: GrammarExpr) -> GrammarExpr {
        let separator_item = seq(vec![self.item_separator_expr(), item.clone()]);
        let body = GrammarExpr::Quantified(Box::new(seq(vec![
            item,
            GrammarExpr::Quantified(Box::new(GrammarExpr::Quantified(Box::new(separator_item), Quantifier::OnePlus)), Quantifier::Optional),
        ])), Quantifier::Optional);

        let rule_name = self.fresh_rule_name("unbounded_scalar_array");
        self.add_terminal_rule(&rule_name, seq(vec![lit("["), body, lit("]")]));
        super::lower::r(&rule_name)
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
                    items.push((self.lower_schema(prefix)?, None));
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
                .map(|schema| self.lower_schema(schema).map(|expr| (expr, None)))
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
    ) -> Vec<(GrammarExpr, Option<Quantifier>)> {
        match max {
            Some(0) => Vec::new(),
            Some(max) => vec![(
                item,
                Some(Quantifier::Range(required_min, Some(max))),
            )],
            None if required_min == 0 => vec![(item, Some(Quantifier::ZeroPlus))],
            None => vec![(item, Some(Quantifier::Range(required_min, None)))],
        }
    }
}

fn bounded_array_object_item_candidate(schema: &super::ast::Schema) -> bool {
    match &schema.kind {
        SchemaKind::Assertions(assertions) => assertions.object.is_some(),
        _ => false,
    }
}
