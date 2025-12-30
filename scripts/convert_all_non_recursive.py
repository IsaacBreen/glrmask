#!/usr/bin/env python3
"""
Convert ALL non-recursive nonterminals to terminals.

Usage: python convert_all_non_recursive.py input.ebnf output.ebnf
"""

import re
import sys
from collections import OrderedDict

def parse_ebnf(content):
    rules = OrderedDict()
    rule_re = re.compile(r'^\s*(\S+)\s*::=\s*(.+?)\s*;\s*$')
    for line in content.split('\n'):
        if not line.strip() or line.strip().startswith('#'):
            continue
        m = rule_re.match(line)
        if m:
            rules[m.group(1)] = m.group(2)
    return rules

def find_refs(body, rules):
    refs = set()
    for m in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', body):
        ref = m.group(1)
        if ref in rules:
            refs.add(ref)
    return refs

def find_recursive_rules(rules):
    adj = {name: list(find_refs(body, rules)) for name, body in rules.items()}
    
    recursive = set()
    for start_node in rules:
        stack = [(start_node, [start_node])]
        found = False
        while stack and not found:
            node, path = stack.pop()
            for neighbor in adj.get(node, []):
                if neighbor == start_node:
                    recursive.add(start_node)
                    found = True
                    break
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
    return recursive

def to_terminal_name(name):
    if name.startswith('_'):
        return name[1:].upper()
    return name.upper()

def convert_all_non_recursive(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()
    
    rules = parse_ebnf(content)
    recursive = find_recursive_rules(rules)
    
    # Logic for Convertibility:
    # A rule can be a Terminal if and only if it does not reference any Non-Terminal.
    # Recursive rules are by definition Non-Terminals.
    # Therefore, a rule is convertible if its entire subtree contains NO recursive rules.
    
    def subtree_has_recursive_ref(name, visited=None):
        if visited is None:
            visited = set()
        if name in visited:
            return False
        visited.add(name)
        
        # If this rule itself is recursive, it's a poison pill
        if name in recursive:
            return True
            
        if name not in rules:
            return False
            
        refs = find_refs(rules[name], rules)
        for ref in refs:
            if subtree_has_recursive_ref(ref, visited):
                return True
        return False
    
    to_convert = set()
    for name in rules:
        # Only consider rules starting with _ (symbolic nonterminals)
        if name.startswith('_'):
             # If it doesn't touch any recursive rule, it can be a terminal
            if not subtree_has_recursive_ref(name):
                to_convert.add(name)
    
    print(f"Recursive rules: {len(recursive)}")
    print(f"Non-recursive nonterminals to convert: {len(to_convert)}")
    
    # Create rename map
    rename_map = {r: to_terminal_name(r) for r in to_convert}
    
    # Apply renaming
    new_rules = OrderedDict()
    for name, body in rules.items():
        # Rename references
        for old, new in rename_map.items():
            body = re.sub(rf'\b{re.escape(old)}\b', new, body)
        
        # Rename rule itself if applicable
        if name in rename_map:
            new_rules[rename_map[name]] = body
        else:
            new_rules[name] = body
    
    with open(output_file, 'w') as f:
        for name, body in new_rules.items():
            f.write(f"{name} ::= {body} ;\n")
    
    print(f"Output: {len(new_rules)} rules to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_all_non_recursive.py input.ebnf output.ebnf")
        sys.exit(1)
    convert_all_non_recursive(sys.argv[1], sys.argv[2])
