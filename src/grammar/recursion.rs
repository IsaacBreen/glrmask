// src/grammar/recursion.rs

// Standard library imports that might be needed in a grammar module
use std::collections::HashMap;
use std::fmt; // For debug formatting if needed elsewhere

// --- Assumed supporting data structures for the grammar ---

// RuleId is typically an index into the grammar's rule vector
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuleId(pub usize);

impl From<usize> for RuleId {
    fn from(id: usize) -> Self {
        RuleId(id)
    }
}

// Represents a symbol in a production: either a terminal or a non-terminal (rule)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Symbol {
    Terminal(String),
    NonTerminal(RuleId),
}

// A single production rule (e.g., A -> B C D)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Production {
    pub symbols: Vec<Symbol>,
    // Add other fields as necessary, e.g., semantic actions, precedence.
    // For this context, `symbols` is sufficient.
}

// Represents a non-terminal rule with its name and alternative productions
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    pub name: String,
    pub productions: Vec<Production>,
    // Add other rule properties, e.g., nullability, first set, etc.
}

// The main grammar structure holding all rules
pub struct Grammar {
    pub rules: Vec<Rule>,
    pub rule_names: HashMap<String, RuleId>, // Map rule names to their RuleId
    // Add other grammar-level properties like start rule, terminals, etc.
}

impl Grammar {
    // Constructor for the Grammar
    pub fn new() -> Self {
        Grammar {
            rules: Vec::new(),
            rule_names: HashMap::new(),
        }
    }

    // A method to add a rule to the grammar, useful for testing/building
    pub fn add_rule(&mut self, name: String, productions: Vec<Production>) -> RuleId {
        let id = RuleId(self.rules.len());
        self.rules.push(Rule { name: name.clone(), productions });
        self.rule_names.insert(name, id);
        id
    }

    // This method is crucial for the indirect recursion resolution.
    // Its detailed implementation is not part of this change,
    // but its signature and high-level purpose are necessary for the context.
    // It takes productions of `substituted_rule_id` and "injects" them
    // into productions of `target_rule_id` where `substituted_rule_id` appears.
    fn substitute(&mut self, target_rule_id: RuleId, substituted_rule_id: RuleId) {
        // Placeholder for actual substitution logic.
        // This is a simplified example; a real implementation would be more complex
        // and handle various cases (e.g., multiple occurrences of substituted_rule_id,
        // empty productions, etc.).
        let mut new_productions_for_target = Vec::new();
        let target_rule_name = self.rules[target_rule_id.0].name.clone();
        let substituted_rule_name = self.rules[substituted_rule_id.0].name.clone();

        println!(
            "DEBUG: Substituting rule '{}' into rule '{}'",
            substituted_rule_name,
            target_rule_name
        );

        let productions_to_process = self.rules[target_rule_id.0].productions.clone();

        for prod in productions_to_process {
            let mut current_expansions: Vec<Vec<Symbol>> = vec![Vec::new()];

            for symbol in prod.symbols {
                if let Symbol::NonTerminal(id) = symbol {
                    if id == substituted_rule_id {
                        let mut next_expansions = Vec::new();
                        for sub_prod in &self.rules[substituted_rule_id.0].productions {
                            for current_expansion_prefix in &current_expansions {
                                let mut new_expansion = current_expansion_prefix.clone();
                                new_expansion.extend(sub_prod.symbols.iter().cloned());
                                next_expansions.push(new_expansion);
                            }
                        }
                        current_expansions = next_expansions;
                    } else {
                        for expansion in &mut current_expansions {
                            expansion.push(symbol.clone());
                        }
                    }
                } else {
                    for expansion in &mut current_expansions {
                        expansion.push(symbol.clone());
                    }
                }
            }
            // Add the resulting expanded productions
            for expansion in current_expansions {
                new_productions_for_target.push(Production { symbols: expansion });
            }
        }
        self.rules[target_rule_id.0].productions = new_productions_for_target;
    }


    /// Resolves indirect recursion for a set of mutually recursive rules (an SCC).
    /// It uses substitution to make all recursion direct, allowing it to be resolved
    /// by the direct recursion resolver.
    fn resolve_indirect_recursion(&mut self, scc: &[RuleId]) {
        // To resolve indirect recursion, we use substitution. For a set of
        // mutually recursive rules {A, B, C}, we can make recursion direct on A
        // by substituting B and C into A.
        // The order of substitutions can drastically affect the number of new
        // rules created, which is the primary cause of the performance issue.
        // A good heuristic is to substitute smaller rules (fewer productions)
        // into bigger ones. We accomplish this by sorting the rules in the SCC
        // by their number of productions before processing.
        let mut ordered_scc = scc.to_vec();
        ordered_scc.sort_by_key(|id| self.rules[id.0].productions.len());

        for i in 0..ordered_scc.len() {
            let rule_i_id = ordered_scc[i];
            for j in 0..i {
                let rule_j_id = ordered_scc[j];
                self.substitute(rule_i_id, rule_j_id);
            }
        }
    }

    // Additional methods for grammar processing would go here, e.g.:
    // pub fn find_strongly_connected_components(&self) -> Vec<Vec<RuleId>> { ... }
    // pub fn resolve_direct_recursion(&mut self, rule_id: RuleId) { ... }
    // pub fn left_factor(&mut self) { ... }
    // pub fn compute_first_sets(&self) { ... }
    // pub fn compute_follow_sets(&self) { ... }
}
