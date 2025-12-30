#!/usr/bin/env python3
"""
Extract subtree for a given rule from an EBNF grammar.

Usage: python ebnf_subtree.py grammar.ebnf rule_name [--depth N]

This recursively follows all references and prints the full subtree.
"""

import re
import sys
from collections import OrderedDict

def parse_ebnf(content):
    """Parse EBNF into a dict of rules."""
    rules = {}
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
        # Check if it's a known rule
        if ref in rules:
            refs.append(ref)
    return refs

def get_subtree(root, rules, max_depth=None, visited=None, depth=0):
    """Get all rules in the subtree rooted at 'root'."""
    if visited is None:
        visited = OrderedDict()
    
    if max_depth is not None and depth > max_depth:
        return visited
    
    if root in visited:
        return visited
    
    if root not in rules:
        return visited
    
    body = rules[root]
    visited[root] = body
    
    # Find referenced rules
    refs = find_refs(body, rules)
    for ref in refs:
        get_subtree(ref, rules, max_depth, visited, depth + 1)
    
    return visited

def main():
    if len(sys.argv) < 3:
        print("Usage: python ebnf_subtree.py grammar.ebnf rule_name [--depth N]")
        sys.exit(1)
    
    grammar_file = sys.argv[1]
    rule_name = sys.argv[2]
    max_depth = None
    
    if '--depth' in sys.argv:
        idx = sys.argv.index('--depth')
        if idx + 1 < len(sys.argv):
            max_depth = int(sys.argv[idx + 1])
    
    with open(grammar_file, 'r') as f:
        content = f.read()
    
    rules = parse_ebnf(content)
    
    if rule_name not in rules:
        print(f"Error: Rule '{rule_name}' not found in grammar")
        sys.exit(1)
    
    subtree = get_subtree(rule_name, rules, max_depth)
    
    print(f"# Subtree for {rule_name} ({len(subtree)} rules)")
    print()
    for name, body in subtree.items():
        print(f"{name} ::= {body} ;")

if __name__ == "__main__":
    main()
