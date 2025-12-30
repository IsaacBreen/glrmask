// //! EBNF Choice Factoring Preprocessor
//! 
//! This module implements choice factoring at the EBNF text level, matching the
//! behavior of scripts/optimize_factor_choices.py. It operates BEFORE parsing the
//! EBNF into internal productions, allowing it to factor the original grammar structure.

use std::collections::{HashMap, HashSet};
use regex::Regex;

/// Parse EBNF text into rules and directives, preserving order
/// Returns (rules as Vec for order preservation, rules as HashMap for lookup, directives)
pub fn parse_ebnf_simple(ebnf_text: &str) -> (Vec<(String, String)>, HashMap<String, String>, Vec<String>) {
    let mut rules_vec = Vec::new();
    let mut rules_map = HashMap::new();
    let mut directives = Vec::new();
    
    let rule_regex = Regex::new(r"^\s*(\S+)\s*::=\s*(.+?)\s*;\s*$").unwrap();
    
    for line in ebnf_text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        
        // Preserve directives (lines starting with #)
        if line.starts_with('#') {
            directives.push(line.to_string());
            continue;
        }
        
        if let Some(caps) = rule_regex.captures(line) {
            let name = caps.get(1).unwrap().as_str().to_string();
            let body = caps.get(2).unwrap().as_str().to_string();
            rules_vec.push((name.clone(), body.clone()));
            rules_map.insert(name, body);
        }
    }
    
    (rules_vec, rules_map, directives)
}

/// Find all rule names referenced in a rule body
fn find_refs(body: &str, all_rules: &HashMap<String, String>) -> HashSet<String> {
    let mut refs = HashSet::new();
    
    // Match non-terminal references (alphanumeric + underscore)
    let nt_regex = Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b").unwrap();
    for cap in nt_regex.captures_iter(body) {
        let name = cap.get(1).unwrap().as_str();
        if all_rules.contains_key(name) {
            refs.insert(name.to_string());
        }
    }
    
    refs
}

/// Find all recursive rules (rules that reference themselves transitively)
fn find_recursive_rules(rules: &HashMap<String, String>) -> HashSet<String> {
    let mut recursive = HashSet::new();
    
    // Build adjacency list
    let adj: HashMap<String, Vec<String>> = rules.iter()
        .map(|(name, body)| (name.clone(), find_refs(body, rules).into_iter().collect()))
        .collect();
    
    // For each rule, do DFS to check if it can reach itself
    for start in rules.keys() {
        let mut stack = vec![(start.clone(), vec![start.clone()])];
        let mut found = false;
        
        while let Some((node, path)) = stack.pop() {
            if found { break; }
            
            if let Some(neighbors) = adj.get(&node) {
                for neighbor in neighbors {
                    if neighbor == start {
                        recursive.insert(start.clone());
                        found = true;
                        break;
                    }
                    
                    if !path.contains(neighbor) {
                        let mut new_path = path.clone();
                        new_path.push(neighbor.clone());
                        stack.push((neighbor.clone(), new_path));
                    }
                }
            }
        }
    }
    
    recursive
}

/// Split a rule body into top-level choices, handling nested parentheses
fn get_choices(body: &str) -> Vec<String> {
    let body = body.trim();
    
    // Strip outer parens if present
    let body = if body.starts_with('(') && body.ends_with(')') {
        let inner = &body[1..body.len()-1];
        // Check if balanced
        let mut depth = 0;
        let mut balanced = true;
        for c in inner.chars() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth < 0 {
                        balanced = false;
                        break;
                    }
                },
                _ => {}
            }
        }
        if balanced && depth == 0 { inner } else { body }
    } else {
        body
    };
    
    // Split on '|' at depth 0
    let mut choices = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    let mut in_quote = false;
    let mut quote_char = ' ';
    
    for c in body.chars() {
        match c {
            '\'' | '"' if !in_quote => {
                in_quote = true;
                quote_char = c;
                current.push(c);
            }
            c if in_quote && c == quote_char => {
                in_quote = false;
                current.push(c);
            }
            '(' if !in_quote => {
                depth += 1;
                current.push(c);
            }
            ')' if !in_quote => {
                depth -= 1;
                current.push(c);
            }
            '|' if !in_quote && depth == 0 => {
                choices.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(c),
        }
    }
    
    if !current.is_empty() {
        choices.push(current.trim().to_string());
    }
    
    choices
}

/// Check if a subtree is safe (doesn't reference recursive rules)
fn is_safe_subtree_deep(
    name: &str,
    rules: &HashMap<String, String>,
    recursive_rules: &HashSet<String>,
    visited: &mut HashSet<String>,
) -> bool {
    if visited.contains(name) {
        return true; // Cycle, assume safe
    }
    visited.insert(name.to_string());
    
    if recursive_rules.contains(name) {
        return false;
    }
    
    // If it's a terminal (uppercase), it's safe
    if name.chars().next().map_or(false, |c| c.is_uppercase()) {
        return true;
    }
    
    // Check if this rule exists
    if let Some(body) = rules.get(name) {
        let refs = find_refs(body, rules);
        for ref_name in refs {
            if !is_safe_subtree_deep(&ref_name, rules, recursive_rules, visited) {
                return false;
            }
        }
    }
    
    true
}

fn is_safe_subtree(alt: &str, rules: &HashMap<String, String>, recursive_rules: &HashSet<String>) -> bool {
    let refs = find_refs(alt, rules);
    for ref_name in refs {
        let mut visited = HashSet::new();
        if !is_safe_subtree_deep(&ref_name, rules, recursive_rules, &mut visited) {
            return false;
        }
    }
    true
}

/// Extract tail pattern from alternatives like '"key" : _tail'
fn extract_tail(alt: &str) -> Option<(String, String)> {
    // Match ':' followed by a single rule reference at the end
    let tail_regex = Regex::new(r"\s*':'\s*([a-zA-Z_][a-zA-Z0-9_]*|\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\))\s*$").unwrap();
    
    if let Some(cap) = tail_regex.find(alt) {
        let tail_part = &alt[cap.start()..];
        let head = alt[..cap.start()].trim();
        
        // Extract tail name
        let tail_name_regex = Regex::new(r"[a-zA-Z_][a-zA-Z0-9_]*").unwrap();
        if let Some(tail_cap) = tail_name_regex.find(tail_part) {
            let tail = tail_cap.as_str().to_string();
            return Some((head.to_string(), tail));
        }
    }
    
    None
}

/// Factor choices in EBNF rules
pub fn factor_ebnf_choices(ebnf_text: &str) -> String {
    let (rules_vec, rules_map, directives) = parse_ebnf_simple(ebnf_text);
    let recursive_rules = find_recursive_rules(&rules_map);
    
    let mut new_rules: Vec<(String, String)> = Vec::new();
    let mut helper_counter = 0;
    let mut factor_cache: HashMap<String, String> = HashMap::new();
    
    // Iterate over rules_vec to preserve original order
    for (name, body) in &rules_vec {
        // Only factor internal rules starting with '_' (except '_json')
        if !name.starts_with('_') || name.starts_with("_json") {
            new_rules.push((name.clone(), body.clone()));
            continue;
        }
        
        let choices = get_choices(body);
        if choices.len() < 2 {
            new_rules.push((name.clone(), body.clone()));
            continue;
        }
        
        // Separate safe and unsafe alternatives
        let mut safe_alts = Vec::new();
        let mut unsafe_alts = Vec::new();
        
        for alt in &choices {
            if is_safe_subtree(alt, &rules_map, &recursive_rules) {
                safe_alts.push(alt.clone());
            } else {
                unsafe_alts.push(alt.clone());
            }
        }
        
        let mut final_choices = Vec::new();
        
        // 1. Group safe alternatives
        if safe_alts.len() > 1 {
            let term_body = safe_alts.join(" | ");
            
            if let Some(existing_name) = factor_cache.get(&term_body) {
                final_choices.push(existing_name.clone());
            } else {
                let mut term_name = format!("__{}_safe", name);
                
                // Check for name collisions and add suffix if needed
                let mut collision_idx = 1;
                let base_name = term_name.clone();
                while new_rules.iter().any(|(n, _)| n == &term_name) || rules_map.contains_key(&term_name) {
                    term_name = format!("{}_{}", base_name, collision_idx);
                    collision_idx += 1;
                }
                
                new_rules.push((term_name.clone(), format!("( {} )", term_body)));
                factor_cache.insert(term_body, term_name.clone());
                final_choices.push(term_name);
                helper_counter += 1;
            }
        } else if safe_alts.len() == 1 {
            final_choices.push(safe_alts[0].clone());
        }
        
        // 2. Group unsafe alternatives by tail
        let mut tail_groups: HashMap<String, Vec<String>> = HashMap::new();
        let mut others = Vec::new();
        
        for alt in &unsafe_alts {
            if let Some((head, tail)) = extract_tail(alt) {
                if is_safe_subtree(&head, &rules_map, &recursive_rules) {
                    tail_groups.entry(tail).or_insert_with(Vec::new).push(head);
                    continue;
                }
            }
            others.push(alt.clone());
        }
        
        // Create helper NTs for each tail group (even single heads!)
        for (tail, heads) in tail_groups {
            let term_body = heads.join(" | ");
            
            if let Some(existing_name) = factor_cache.get(&term_body) {
                final_choices.push(format!("{} ':' {}", existing_name, tail));
            } else {
                let mut term_name = if heads.len() == 1 {
                    let clean_head = heads[0].replace(|c: char| !c.is_alphanumeric(), "").to_lowercase();
                    format!("__key_{}", clean_head)
                } else {
                    let clean_tail = tail.replace(|c: char| !c.is_alphanumeric(), "").to_lowercase();
                    format!("__keys_for_{}", clean_tail)
                };
                
                // Check for name collisions and add suffix if needed
                let mut collision_idx = 1;
                let base_name = term_name.clone();
                while new_rules.iter().any(|(n, _)| n == &term_name) || rules_map.contains_key(&term_name) {
                    term_name = format!("{}_{}", base_name, collision_idx);
                    collision_idx += 1;
                }
                
                new_rules.push((term_name.clone(), format!("( {} )", term_body)));
                factor_cache.insert(term_body, term_name.clone());
                final_choices.push(format!("{} ':' {}", term_name, tail));
                helper_counter += 1;
            }
        }
        
        // Add other unsafe alternatives
        final_choices.extend(others);
        
        // Reconstruct rule body
        let new_body = if final_choices.len() == 1 {
            final_choices[0].clone()
        } else {
            format!("( {} )", final_choices.join(" | "))
        };
        
        new_rules.push((name.clone(), new_body));
    }
    
    // Generate output EBNF - start with directives
    let mut output = String::new();
    for directive in &directives {
        output.push_str(directive);
        output.push('\n');
    }
    if !directives.is_empty() {
        output.push('\n');
    }
    
    for (name, body) in &new_rules {
        output.push_str(&format!("{} ::= {} ;\n", name, body));
    }
    
    crate::debug!(4, "EBNF factoring: {} rules -> {} rules ({} helpers created)", rules_vec.len(), new_rules.len(), helper_counter);
    output
}