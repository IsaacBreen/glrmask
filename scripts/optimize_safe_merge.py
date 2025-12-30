#!/usr/bin/env python3
"""
Safe Merge Optimization for EBNF grammars.

Strategy:
1. Identify "leaf" _def rules that don't reference any _mem or other _def rules.
   These are terminal-like (e.g., choices of JSON_BOOL, literals, _alt refs).
2. Inline these leaf _def rules into their parent _mem rules.
3. Repeat until no more safe merges are possible.

This reduces rule count without causing terminal explosion.
"""

import re
import sys

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
    # Match rule names: _def123, _mem123, _pv123, _alt123, JSON_*, _json_*
    return set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', body))

def is_leaf_def(name, body, rules):
    """
    Check if a _def rule is "leaf" (doesn't reference _mem or other complex _def).
    A leaf _def contains only:
    - Literals ('...')
    - JSON_* terminals
    - _alt* references (these are usually simple choice groups)
    - _pv* references (property values - can be complex, skip these)
    - Other _def that are themselves leaves (recursive check needed)
    
    For safety, we only inline _def rules that don't reference any _mem.
    """
    if not name.startswith('_def'):
        return False
    
    refs = find_refs(body)
    
    for ref in refs:
        # Skip self and standard terminals
        if ref == name:
            continue
        if ref.startswith('JSON_') or ref in ['HEX', 'DIGITS', 'EXPONENT', 'WS', 'STRING_CHARS', 'STRING_CHAR', 'ESCAPE_SEQ']:
            continue
        # _alt rules are typically simple choices
        if ref.startswith('_alt'):
            continue
        # _pv rules can be arrays - check if they're simple
        if ref.startswith('_pv'):
            # For now, skip _pv to be safe
            return False
        # Other _def references - check if they're leaves too (but avoid deep recursion)
        if ref.startswith('_def'):
            # For simplicity, don't inline _def that references other _def
            return False
        # _mem references mean this is not a leaf
        if ref.startswith('_mem'):
            return False
        if ref.startswith('_json'):
            return False
    
    return True

def inline_rule(body, rule_name, rule_body):
    """Replace references to rule_name with rule_body (wrapped in parens)."""
    # Wrap the replacement in parens to preserve grouping
    replacement = f"( {rule_body} )"
    
    # Use word boundary regex
    pattern = rf'\b{re.escape(rule_name)}\b'
    return re.sub(pattern, replacement, body)

def optimize_safe_merge(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()
    
    rules, rule_order = parse_ebnf(content)
    
    # Find all leaf _def rules
    leaf_defs = {}
    for name, body in rules.items():
        if is_leaf_def(name, body, rules):
            leaf_defs[name] = body
    
    print(f"Found {len(leaf_defs)} leaf _def rules to inline")
    if leaf_defs:
        print(f"Examples: {list(leaf_defs.keys())[:5]}")
    
    # Inline leaf _defs into all rules that reference them
    for leaf_name, leaf_body in leaf_defs.items():
        for rule_name in rule_order:
            if rule_name == leaf_name:
                continue
            if rule_name in rules:
                old_body = rules[rule_name]
                if leaf_name in old_body:
                    rules[rule_name] = inline_rule(old_body, leaf_name, leaf_body)
    
    # Remove the inlined leaf _def rules from output
    output_rules = [(name, rules[name]) for name in rule_order if name not in leaf_defs]
    
    # Write output
    with open(output_file, 'w') as f:
        for name, body in output_rules:
            f.write(f"{name} ::= {body} ;\n")
    
    print(f"Output: {len(output_rules)} rules (removed {len(leaf_defs)} leaf defs)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python optimize_safe_merge.py input.ebnf output.ebnf")
        sys.exit(1)
    optimize_safe_merge(sys.argv[1], sys.argv[2])
