import re
import sys

def simulate_radical_conversion(filename):
    with open(filename, 'r') as f:
        content = f.read()

    rules = {}
    rule_re = re.compile(r'^\s*(\S+)\s*::=\s*(.+?)\s*;\s*$')
    for line in content.split('\n'):
        if not line.strip() or line.strip().startswith('#'):
            continue
        m = rule_re.match(line)
        if m:
            rules[m.group(1)] = m.group(2)

    # Adjacency list for recursion analysis
    def find_refs(body):
        return set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', body))

    adj = {name: [r for r in find_refs(body) if r in rules] for name, body in rules.items()}
    
    # Find recursive rules
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

    # In a radical conversion:
    # 1. Any non-recursive rule becomes a Terminal (if referenced by a recursive rule or root).
    # 2. Recursive rules remain Nonterminals.
    # 3. We want to see how many "Terminals" the Parser (the recursive core) still sees.

    parser_side_terminals = set()
    
    # The parser sees any Terminal or Non-recursive-Renamed-To-Terminal referenced by a recursive rule
    for name in recursive:
        body = rules[name]
        
        # Symbolic terminals already exist
        words = re.findall(r'\b[A-Z][A-Z0-9_]*\b', body)
        for w in words:
            parser_side_terminals.add(w)
            
        # Literals inside recursive rules remain as terminals
        literals = re.findall(r"'([^']*)'", body)
        for lit in literals:
            parser_side_terminals.add(f"'{lit}'")
            
        # Non-recursive rules referenced by recursive rules become NEW terminals
        refs = find_refs(body)
        for ref in refs:
            if ref in rules and ref not in recursive:
                parser_side_terminals.add(f"TERM_{ref.upper().lstrip('_')}")

    # Also count what the root sees (if root is non-recursive)
    root_body = rules.get('root', '')
    if 'root' not in recursive:
        # If root is non-recursive, it's basically the whole grammar as one terminal? 
        # No, root is usually the starting point.
        pass

    print(f"Number of 'Recursive' Nonterminals: {len(recursive)}")
    print(f"Number of 'Terminals' the recursive core would see: {len(parser_side_terminals)}")
    print(f"List of those terminals: {sorted(list(parser_side_terminals))}")

if __name__ == "__main__":
    simulate_radical_conversion(sys.argv[1])
