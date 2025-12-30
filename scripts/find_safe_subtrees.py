#!/usr/bin/env python3
"""
Find all _def subtrees that can safely be converted to terminals.

A subtree is "safe" if it doesn't reference:
- JSON_STRING (infinite possibilities)
- JSON_NUMBER (infinite)
- _json_* rules (arbitrary JSON)
- Any rule that itself references these

Usage: python find_safe_subtrees.py grammar.ebnf
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
    refs = set()
    for m in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', body):
        ref = m.group(1)
        if ref in rules:
            refs.add(ref)
    return refs

def is_safe_subtree(root, rules, memo=None):
    """
    Check if the subtree rooted at 'root' is safe to convert to terminals.
    """
    if memo is None:
        memo = {}
    
    if root in memo:
        return memo[root]
    
    if root not in rules:
        # Unknown rule - check if it's a forbidden builtin
        if root in ['JSON_STRING', 'JSON_NUMBER']:
            return False
        if root.startswith('_json'):
            return False
        # Assume other builtins (JSON_BOOL, JSON_NULL, JSON_INTEGER, etc.) are safe
        return True
    
    # Temporarily mark as safe to handle cycles
    memo[root] = True
    
    body = rules[root]
    refs = find_refs(body, rules)
    
    for ref in refs:
        # Direct unsafe references
        if ref in ['JSON_STRING', 'JSON_NUMBER']:
            memo[root] = False
            return False
        if ref.startswith('_json'):
            memo[root] = False
            return False
        # Recursively check
        if not is_safe_subtree(ref, rules, memo):
            memo[root] = False
            return False
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python find_safe_subtrees.py grammar.ebnf")
        sys.exit(1)
    
    with open(sys.argv[1], 'r') as f:
        content = f.read()
    
    rules = parse_ebnf(content)
    memo = {}
    
    # Find all safe _def roots
    safe_roots = []
    for name in rules:
        if name.startswith('_def'):
            if is_safe_subtree(name, rules, memo):
                safe_roots.append(name)
    
    print(f"Found {len(safe_roots)} safe _def subtrees to convert:")
    for r in safe_roots:
        print(f"  {r}")

if __name__ == "__main__":
    main()
