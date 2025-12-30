import re
import sys
from collections import Counter

def analyze_ref_counts(filename):
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

    def find_refs(body):
        return re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', body)

    all_refs = []
    for name, body in rules.items():
        all_refs.extend([r for r in find_refs(body) if r in rules])
    
    counts = Counter(all_refs)
    
    # Nonterminals with ref count 1
    nt_count_1 = [nt for nt, count in counts.items() if count == 1 and nt.startswith('_')]
    
    print(f"Total nonterminals: {len([n for n in rules if n.startswith('_')])}")
    print(f"Nonterminals with ref count 1: {len(nt_count_1)}")
    
    print("\nSample of ref count 1 nonterminals:")
    for nt in sorted(nt_count_1)[:20]:
        print(f"  {nt} (used in {[n for n, b in rules.items() if nt in find_refs(b)]})")

if __name__ == "__main__":
    analyze_ref_counts(sys.argv[1])
