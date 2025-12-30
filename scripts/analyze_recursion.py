import re
import sys
from collections import defaultdict

def parse_ebnf(content):
    rules = {}
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
    # Build adjacency list
    adj = {name: find_refs(body, rules) for name, body in rules.items()}
    
    recursive = set()
    
    def is_recursive(start_node):
        stack = [(start_node, [start_node])]
        while stack:
            node, path = stack.pop()
            for neighbor in adj.get(node, []):
                if neighbor == start_node:
                    return True
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
        return False

    for name in rules:
        if is_recursive(name):
            recursive.add(name)
            
    return recursive

def main():
    with open(sys.argv[1], 'r') as f:
        content = f.read()
    rules = parse_ebnf(content)
    recursive = find_recursive_rules(rules)
    
    non_recursive = set(rules.keys()) - recursive
    
    print(f"Total rules: {len(rules)}")
    print(f"Recursive rules: {len(recursive)}")
    print(f"Non-recursive rules: {len(non_recursive)}")
    
    print("\nRecursive rules sample:")
    for r in sorted(list(recursive))[:20]:
        print(f"  {r}")
        
    # How many of these non-recursive rules are currently nonterminals?
    nr_nonterminals = [r for r in non_recursive if r.startswith('_')]
    print(f"\nNon-recursive nonterminals (Potential Terminals): {len(nr_nonterminals)}")

if __name__ == "__main__":
    main()
