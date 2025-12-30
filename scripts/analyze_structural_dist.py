import re
import sys

def analyze_structural_distribution(filename):
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
    
    # DFS to find recursive rules
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

    # Categorize literals
    recursive_literals = set()
    non_recursive_literals = set()

    for name, body in rules.items():
        literals = re.findall(r"'([^']*)'", body)
        if name in recursive:
            for lit in literals:
                recursive_literals.add(lit)
        else:
            for lit in literals:
                non_recursive_literals.add(lit)

    # Referenced symbolic terminals
    symbolic_terminals = {n for n in rules if n[0].isupper()}
    
    recursive_symbolic_refs = set()
    non_recursive_symbolic_refs = set()

    for name, body in rules.items():
        words = re.findall(r'\b[A-Z][A-Z0-9_]*\b', body)
        if name in recursive:
            for w in words:
                if w in symbolic_terminals:
                    recursive_symbolic_refs.add(w)
        else:
            for w in words:
                if w in symbolic_terminals:
                    non_recursive_symbolic_refs.add(w)

    print(f"Total Unique Literals: {len(recursive_literals | non_recursive_literals)}")
    print(f"Literals found in RECURSIVE rules: {len(recursive_literals)}")
    print(f"Literals found EXCLUSIVELY in non-recursive rules: {len(non_recursive_literals - recursive_literals)}")
    print(f"\nSymbolic terminals found in RECURSIVE rules: {len(recursive_symbolic_refs)}")
    print(f"Symbolic terminals found EXCLUSIVELY in non-recursive rules: {len(non_recursive_symbolic_refs - recursive_symbolic_refs)}")

if __name__ == "__main__":
    analyze_structural_distribution(sys.argv[1])
