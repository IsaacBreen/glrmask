#!/usr/bin/env python3
"""
Terminal Conversion Optimization for EBNF grammars.

Strategy:
1. Find patterns like '"key"' ':' in _mem rules
2. Create proper TERMINALS (capital-letter names) for these patterns
3. Replace the patterns with terminal references

This moves key-colon parsing to the tokenizer (DFA) instead of parser (GLR).
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

def make_terminal_name(key_string):
    """Convert a key string to a valid terminal name (CAPITAL_CASE)."""
    # Remove quotes and special chars, convert to uppercase
    clean = key_string.strip('"\'')
    # Replace dots, dashes, etc with underscores
    clean = re.sub(r'[^a-zA-Z0-9]', '_', clean)
    # Remove consecutive underscores
    clean = re.sub(r'_+', '_', clean)
    clean = clean.strip('_')
    return f"KEY_{clean.upper()}_COLON"

def optimize_terminals(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()
    
    rules, rule_order = parse_ebnf(content)
    
    # Find all '"key"' ':' patterns in _mem rules
    # Pattern: '\"...\"' ':'
    key_colon_pattern = re.compile(r"'\"([^\"]+)\"'\s*':'")
    
    # Collect all unique key-colon patterns
    key_colons = {}
    for name, body in rules.items():
        if name.startswith('_mem') or name.startswith('_json'):
            for m in key_colon_pattern.finditer(body):
                key = m.group(1)
                full_match = m.group(0)  # e.g., '"apq"' ':'
                if key not in key_colons:
                    key_colons[key] = make_terminal_name(key)
    
    print(f"Found {len(key_colons)} unique key-colon patterns")
    print(f"Examples: {list(key_colons.items())[:5]}")
    
    # Create terminal definitions
    new_terminals = {}
    for key, term_name in key_colons.items():
        # Terminal definition: KEY_APQ_COLON ::= '"apq":' ;
        # Note: combining into single literal
        new_terminals[term_name] = f"'\"{key}:\"'"
    
    # Replace patterns in rules
    for name in rule_order:
        body = rules[name]
        for key, term_name in key_colons.items():
            # Replace '"key"' ':' with terminal reference
            pattern = f"'\"{key}\"'\\s*':'"
            body = re.sub(pattern, term_name, body)
        rules[name] = body
    
    # Write output
    with open(output_file, 'w') as f:
        # Write original rules first
        for name in rule_order:
            f.write(f"{name} ::= {rules[name]} ;\n")
        
        # Write new terminal definitions
        f.write("\n# Generated Key-Colon Terminals\n")
        for term_name, term_def in sorted(new_terminals.items()):
            f.write(f"{term_name} ::= {term_def} ;\n")
    
    print(f"Output: {len(rule_order)} rules + {len(new_terminals)} new terminals")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python optimize_terminals.py input.ebnf output.ebnf")
        sys.exit(1)
    optimize_terminals(sys.argv[1], sys.argv[2])
