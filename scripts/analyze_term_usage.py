import re
import sys
from collections import Counter

def analyze_terminal_usage(filename):
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

    # Map of nonterminal -> set of unique terminals/literals it references
    usage = {}
    for name in nonterminals:
        body = rules[name]
        body_referenced = set()
        
        # Symbolic terminals
        words = re.findall(r'\b[A-Z][A-Z0-9_]*\b', body)
        for word in words:
            if word in terminals:
                body_referenced.add(word)
        
        # Literals
        literals = re.findall(r"'([^']*)'", body)
        for lit in literals:
            body_referenced.add(f"'{lit}'")
            
        usage[name] = body_referenced

    # Sort nonterminals by how many unique terminals they reference
    sorted_nt = sorted(usage.items(), key=lambda x: len(x[1]), reverse=True)

    print("Top 20 Terminal-Heavy Nonterminals:")
    for nt, refs in sorted_nt[:20]:
        print(f"{nt}: {len(refs)} unique terminals/literals")
        
    # Analyze literal distribution
    all_literals = [lit for refs in usage.values() for lit in refs if lit.startswith("'")]
    unique_literals = set(all_literals)
    print(f"\nTotal unique literals referenced by nonterminals: {len(unique_literals)}")
    
    # Count how many of these literals are unique to ONE nonterminal
    lit_counts = Counter(all_literals)
    unique_to_one = [lit for lit, count in lit_counts.items() if count == 1]
    print(f"Literals unique to exactly one nonterminal: {len(unique_to_one)} ({(len(unique_to_one)/len(unique_literals)*100):.1f}%)")

if __name__ == "__main__":
    analyze_terminal_usage(sys.argv[1])
