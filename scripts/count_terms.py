import re
import sys

def count_referenced_terminals(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    rules = {}
    terminals = set()
    nonterminals = set()
    
    rule_re = re.compile(r'^\s*(\S+)\s*::=\s*(.+?)\s*;\s*$')
    
    for line in lines:
        m = rule_re.match(line)
        if m:
            name = m.group(1)
            body = m.group(2)
            rules[name] = body
            if name[0].isupper():
                terminals.add(name)
            else:
                nonterminals.add(name)

    referenced_symbolic = set()
    referenced_literals = set()
    
    for name in nonterminals:
        body = rules[name]
        # Find all symbolic terminals referenced
        words = re.findall(r'\b[A-Z][A-Z0-9_]*\b', body)
        for word in words:
            if word in terminals:
                referenced_symbolic.add(word)
        
        # Find all literals referenced
        literals = re.findall(r"'([^']*)'", body)
        for lit in literals:
            referenced_literals.add(lit)

    print(f"Referenced symbolic terminals: {len(referenced_symbolic)}")
    print(f"Referenced literal terminals: {len(referenced_literals)}")
    print(f"Total unique referenced terminals: {len(referenced_symbolic) + len(referenced_literals)}")
    
    # Optional: list them for verification if requested
    # print(f"Symbolic: {sorted(list(referenced_symbolic))}")


if __name__ == "__main__":
    count_referenced_terminals(sys.argv[1])
