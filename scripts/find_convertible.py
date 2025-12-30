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

def check_convertible(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    rules = parse_ebnf(content)
    
    # A rule is naturally convertible IF:
    # 1. It starts with _ (nonterminal)
    # 2. It only references symbols that start with ALPHABET (Terminals) or literals
    
    convertible = []
    
    for name, body in rules.items():
        if not name.startswith('_'): continue
        
        refs = find_refs(body, rules)
        
        # Check if all refs are terminals (start with capital) or if there are NO refs
        all_terminal_refs = True
        for ref in refs:
            if not ref[0].isupper():
                all_terminal_refs = False
                break
        
        if all_terminal_refs:
            convertible.append(name)
            
    print(f"Found {len(convertible)} naturally convertible nonterminals:")
    for c in convertible:
        print(f"  {c} ::= {rules[c]}")

if __name__ == "__main__":
    check_convertible(sys.argv[1])
