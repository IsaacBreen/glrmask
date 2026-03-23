#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::compiler::glr::parser::{ParserGSS, stacks_finished};
use crate::ds::leveled_gss::LeveledGSSSummary;

use super::super::constraint::Constraint;
use super::super::state::ConstraintStateSummary;

#[derive(Debug, Clone)]
pub struct AmbiguousConstraintState<'a> {
    pub(crate) constraint: &'a Constraint,
    pub(crate) state: BTreeMap<u32, ParserGSS>,
}

impl<'a> AmbiguousConstraintState<'a> {
    pub fn summary(&self) -> ConstraintStateSummary {
        let mut summary = ConstraintStateSummary {
            tokenizer_state_count: self.state.len(),
            ..ConstraintStateSummary::default()
        };

        for gss in self.state.values() {
            if gss.is_empty() {
                continue;
            }

            summary.nonempty_tokenizer_state_count += 1;
            let gss_summary: LeveledGSSSummary = gss.summary();
            summary.parser_top_values_total += gss_summary.top_values_count;
            summary.parser_top_values_max = summary
                .parser_top_values_max
                .max(gss_summary.top_values_count);
            summary.parser_upperbranch_nodes_total += gss_summary.upperbranch_nodes;
            summary.parser_upperbranch_nodes_max = summary
                .parser_upperbranch_nodes_max
                .max(gss_summary.upperbranch_nodes);
            summary.parser_interface_nodes_total += gss_summary.interface_nodes;
            summary.parser_interface_nodes_max = summary
                .parser_interface_nodes_max
                .max(gss_summary.interface_nodes);
            summary.parser_lower_nodes_total += gss_summary.lower_nodes;
            summary.parser_lower_nodes_max = summary
                .parser_lower_nodes_max
                .max(gss_summary.lower_nodes);
            summary.parser_unique_nodes_total += gss_summary.total_unique_nodes;
            summary.parser_unique_nodes_max = summary
                .parser_unique_nodes_max
                .max(gss_summary.total_unique_nodes);
            summary.parser_total_edges_total += gss_summary.total_edges;
            summary.parser_accumulator_instances_total += gss_summary.accumulator_instances;
            summary.parser_max_depth = summary.parser_max_depth.max(gss_summary.max_depth);
        }

        summary
    }

    pub fn is_complete(&self) -> bool {
        let initial_tsid = self.constraint.tokenizer.initial_state();
        let Some(stack) = self.state.get(&initial_tsid) else {
            return false;
        };
        !stack.is_empty() && stacks_finished(&self.constraint.table, stack)
    }

    pub fn is_finished(&self) -> bool {
        self.is_complete()
    }
}
