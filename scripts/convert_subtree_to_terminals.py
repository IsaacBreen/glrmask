#!/usr/bin/env python3
"""
Convert a subtree to terminals by capitalizing all rule names.

Usage: python convert_subtree_to_terminals.py grammar.ebnf root_rule [root_rule2 ...]

This will:
1. Extract the full subtree for each root rule
2. Convert all nonterminals in those subtrees to terminals (capitalize names)
3. Update all references in the grammar
4. Output the modified grammar
"""

import re
import sys
from collections import OrderedDict

def parse_ebnf(content):
    """Parse EBNF into a dict of rules."""
    rules = OrderedDict()
    rule_re = re.compile(r'^\s*(\S+)\s*::=\s*(.+?)\s*;\s*$')
    
    for line in content.split('\n'):
        if not line.strip() or line.strip().startswith('#'):
            continue
        m = rule_re.match(line)
        if m:
            name = m.group(1)
            body = m.group(2)
            rules[name] = body
    
    return rules

def find_refs(body, rules):
    """Find all rule references in a rule body."""
    refs = []
    for m in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', body):
        ref = m.group(1)
        if ref in rules:
            refs.append(ref)
    return refs

def get_subtree(root, rules, visited=None):
    """Get all rules in the subtree rooted at 'root'."""
    if visited is None:
        visited = set()
    
    if root in visited:
        return visited
    if root not in rules:
        return visited
    
    visited.add(root)
    
    refs = find_refs(rules[root], rules)
    for ref in refs:
        get_subtree(ref, rules, visited)
    
    return visited

def to_terminal_name(name):
    """Convert a nonterminal name to terminal (capitalize)."""
    # Remove leading underscore, uppercase
    if name.startswith('_'):
        return name[1:].upper()
    return name.upper()

def convert_subtree_to_terminals(input_file, output_file, root_rules):
    with open(input_file, 'r') as f:
        content = f.read()
    
    rules = parse_ebnf(content)
    
    # Collect all rules in subtrees that should become terminals
    to_convert = set()
    for root in root_rules:
        if root not in rules:
            print(f"Warning: Rule '{root}' not found")
            continue
        subtree = get_subtree(root, rules)
        to_convert.update(subtree)
    
    # Don't convert standard terminals (already capitalized) or JSON_*
    to_convert = {r for r in to_convert 
                  if r.startswith('_') 
                  and not r.startswith('_json')}
    
    print(f"Converting {len(to_convert)} rules to terminals:")
    for r in sorted(to_convert)[:20]:
        print(f"  {r} -> {to_terminal_name(r)}")
    if len(to_convert) > 20:
        print(f"  ... and {len(to_convert) - 20} more")
    
    # Create rename map
    rename_map = {r: to_terminal_name(r) for r in to_convert}
    
    # Apply renaming to all rules
    new_rules = OrderedDict()
    for name, body in rules.items():
        # Rename references in body
        for old, new in rename_map.items():
            body = re.sub(rf'\b{re.escape(old)}\b', new, body)
        
        # Rename the rule itself if needed
        if name in rename_map:
            new_rules[rename_map[name]] = body
        else:
            new_rules[name] = body
    
    # Write output
    with open(output_file, 'w') as f:
        for name, body in new_rules.items():
            f.write(f"{name} ::= {body} ;\n")
    
    print(f"\nOutput: {len(new_rules)} rules to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python convert_subtree_to_terminals.py input.ebnf output.ebnf root_rule [root_rule2 ...]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    root_rules = sys.argv[3:]
    
    convert_subtree_to_terminals(input_file, output_file, root_rules)
