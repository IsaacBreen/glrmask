#!/usr/bin/env python3
"""
Convert Leaf _def Rules to Terminals.

Strategy:
1. Find _def rules that are "leaves" - their subtree contains only:
   - Literals ('...')
   - JSON_BOOL, JSON_NULL, JSON_INTEGER, JSON_STRING (primitives)
   - Simple enum choices of literals
2. Expand these leaves into enumerated terminals (capital letter names)
3. Replace references to these _def rules with the new terminal

This moves complex structures into the tokenizer (DFA) instead of parser.
"""

import re
import sys
from collections import defaultdict

def parse_ebnf(content):
    """Parse EBNF into a dict of rules."""
    rules = {}
    rule_order = []
    
    rule_re = re.compile(r'^\s*(\S+)\s*::=\s*(.+?)\s*;\s*$')
    
    for line in content.split('\n'):
        if not line.strip() or line.strip().startswith('#'):
            continue
        m = rule_re.match(line)
        if m:
            name = m.group(1)
            body = m.group(2)
            rules[name] = body
            rule_order.append(name)
    
    return rules, rule_order

def find_refs(body):
    """Find all rule references in a rule body."""
    refs = set()
    # Match rule names (exclude literals in quotes)
    # This is a simple heuristic
    for m in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', body):
        ref = m.group(1)
        # Skip if it's inside quotes (check character before)
        start = m.start()
        pre = body[:start]
        # Count unescaped quotes before this position
        quote_count = pre.count("'") - pre.count("\\'")
        if quote_count % 2 == 0:  # We're outside quotes
            refs.add(ref)
    return refs

def is_simple_leaf(name, body, rules, visited=None):
    """
    Check if a _def rule is a simple leaf that can be converted to a terminal.
    A simple leaf:
    1. Only contains literals, JSON_BOOL, JSON_NULL, JSON_INTEGER
    2. Does NOT contain JSON_STRING, _json_*, other _def/_mem that aren't simple
    3. Has finite enumerable expansions
    """
    if visited is None:
        visited = set()
    
    if name in visited:
        return False  # Circular reference
    visited.add(name)
    
    refs = find_refs(body)
    
    for ref in refs:
        if ref == name:
            continue
        # JSON_STRING has infinite possibilities - not simple
        if ref == 'JSON_STRING':
            return False
        if ref == 'JSON_NUMBER':
            return False  # Infinite
        # These are OK for simple leaves
        if ref in ['JSON_BOOL', 'JSON_NULL', 'JSON_INTEGER']:
            continue
        # Check for primitives we define
        if ref in ['HEX', 'DIGITS', 'EXPONENT', 'WS', 'STRING_CHARS', 'STRING_CHAR', 'ESCAPE_SEQ']:
            return False  # These involve complex patterns
        # _json_* means arbitrary JSON - not simple
        if ref.startswith('_json'):
            return False
        # Other _def or _mem - check recursively
        if ref.startswith('_'):
            if ref not in rules:
                return False
            if not is_simple_leaf(ref, rules[ref], rules, visited):
                return False
    
    # Check for repeat patterns - these can lead to infinite expansions
    # We only allow simple optional and small finite choices
    if '( _mem' in body or '( _pv' in body:
        # Check if it's a repeat pattern like ( _mem123 ( ',' _mem123 )* )?
        # These can have many combinations
        # For now, skip these to be conservative
        return False
    
    return True

def optimize_leaf_terminals(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()
    
    rules, rule_order = parse_ebnf(content)
    
    # Find candidate leaf _def rules
    candidates = []
    for name in rule_order:
        if name.startswith('_def'):
            body = rules[name]
            if is_simple_leaf(name, body, rules):
                candidates.append(name)
    
    print(f"Found {len(candidates)} candidate leaf _def rules")
    if candidates:
        print(f"Examples: {candidates[:10]}")
    
    # For each candidate, make it a terminal by renaming
    terminal_map = {}
    for name in candidates:
        # Create terminal name (capital letters)
        term_name = name.upper().lstrip('_')
        terminal_map[name] = term_name
    
    # Update rule names and references
    new_rules = {}
    new_order = []
    
    for name in rule_order:
        body = rules[name]
        
        # Replace references to leaf _def with terminal names
        for old_name, new_name in terminal_map.items():
            body = re.sub(rf'\b{re.escape(old_name)}\b', new_name, body)
        
        # Rename if this rule is a terminal candidate
        if name in terminal_map:
            new_name = terminal_map[name]
            new_rules[new_name] = body
            new_order.append(new_name)
        else:
            new_rules[name] = body
            new_order.append(name)
    
    # Write output
    with open(output_file, 'w') as f:
        for name in new_order:
            f.write(f"{name} ::= {new_rules[name]} ;\n")
    
    print(f"Converted {len(terminal_map)} _def rules to terminals")
    print(f"Output: {len(new_order)} rules to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python optimize_leaf_terminals.py input.ebnf output.ebnf")
        sys.exit(1)
    optimize_leaf_terminals(sys.argv[1], sys.argv[2])
