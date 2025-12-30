import re
import sys

def detailed_count(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    rules = {}
    symbolic_terminals = set()
    nonterminals = set()
    all_literals = set()
    
    rule_re = re.compile(r'^\s*(\S+)\s*::=\s*(.+?)\s*;\s*$')
    
    for line in lines:
        m = rule_re.match(line)
        if m:
            name = m.group(1)
            body = m.group(2)
            rules[name] = body
            if name[0].isupper():
                symbolic_terminals.add(name)
            else:
                nonterminals.add(name)
            
            # Extract all literals from this body
            literals = re.findall(r"'([^']*)'", body)
            for lit in literals:
                all_literals.add(lit)

    appearing_symbolic = set()
    appearing_literals = set()

    for nt in nonterminals:
        body = rules[nt]
        # Check for symbolic terminals
        words = re.findall(r'\b[A-Z][A-Z0-9_]*\b', body)
        for word in words:
            if word in symbolic_terminals:
                appearing_symbolic.add(word)
        
        # Check for literals
        literals = re.findall(r"'([^']*)'", body)
        for lit in literals:
            appearing_literals.add(lit)

    non_appearing_symbolic = symbolic_terminals - appearing_symbolic
    non_appearing_literals = all_literals - appearing_literals

    print(f"SYMBOLIC TERMINALS (Start with Capital):")
    print(f"  Appearing in Nonterminals: {len(appearing_symbolic)}")
    print(f"  Non-appearing:             {len(non_appearing_symbolic)}")
    print(f"  Total:                    {len(symbolic_terminals)}")
    
    print(f"\nLITERALS ('...'):")
    print(f"  Appearing in Nonterminals: {len(appearing_literals)}")
    print(f"  Non-appearing:             {len(non_appearing_literals)}")
    print(f"  Total unique:             {len(all_literals)}")

if __name__ == "__main__":
    detailed_count(sys.argv[1])
