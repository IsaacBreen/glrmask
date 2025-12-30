#!/usr/bin/env python3
"""
Factor out terminal-convertible choices from mixed rules.

Strategy:
1. Parse grammar and identify recursive rules.
2. For each rule with top-level choices ( A | B | C ):
    a. Split into individual alternatives.
    b. Identify "Safe Alternatives": subtrees with no recursive refs.
    c. Identify "Pattern Alternatives": e.g., ' "key" ' : _json_object
    d. Rewriting:
        - Move all Safe Alternatives into a new TERMINAL rule (e.g., _MEM29_SAFE).
        - For Pattern Alternatives sharing the same tail (e.g., _json_object), 
          move their keys/heads into a new TERMINAL rule (e.g., _MEM29_KEYS_JSON).
        - Update the original rule to reference these new terminals.
"""

import re
import sys
from collections import OrderedDict, defaultdict

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

def find_recursive_rules(rules):
    adj = {name: list(find_refs(body, rules)) for name, body in rules.items()}
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
    return recursive

def get_choices(body):
    """
    Split body into top-level choices.
    Handles parentheses carefully to avoid splitting inside nested groups.
    Simple heuristic: if starts with ( and ends with ) and has |, strip outer parens.
    """
    body = body.strip()
    if body.startswith('(') and body.endswith(')'):
        # Check if parens are balanced without the outer ones
        inner = body[1:-1]
        depth = 0
        balanced = True
        for char in inner:
            if char == '(': depth += 1
            elif char == ')': depth -= 1
            if depth < 0: 
                balanced = False
                break
        if balanced:
            body = inner

    choices = []
    current = []
    depth = 0
    quote = False
    
    for char in body:
        if char == "'" or char == '"':
            quote = not quote
        if not quote:
            if char == '(': depth += 1
            elif char == ')': depth -= 1
            elif char == '|' and depth == 0:
                choices.append("".join(current).strip())
                current = []
                continue
        current.append(char)
    
    if current:
        choices.append("".join(current).strip())
        
    return choices

def is_safe_subtree(body, rules, recursive_rules):
    """Check if a rule body (alternative) references any recursive rules."""
    refs = find_refs(body, rules)
    for ref in refs:
        if ref in recursive_rules:
            return False
        # If it references a nonterminal that ISN'T recursive in itself,
        # we strictly need to check if THAT nonterminal's subtree is safe.
        # But for this specific factoring optimization, we can just check direct refs
        # if we assume we've already tried to convert bottom-up.
        # However, to be safe, let's assume if it references ANY nonterminal that isn't
        # already a Terminal-candidate (capitalized), it's unsafe for now 
        # unless we want to do deep graph analysis.
        
        # Better heuristic for this specific task:
        # If it refrences a Capitalized rule -> Safe (Terminal)
        # If it references a Recursive rule -> Unsafe
        # If it references a lowercase rule -> Check that rule?
        
        if ref not in rules: continue # Primitive or unknown
        
        # Deep check
        if not is_subtree_safe_deep(ref, rules, recursive_rules, set()):
            return False
            
    return True

def is_subtree_safe_deep(name, rules, recursive_rules, visited):
    if name in visited: return True # Cycle detected? assume managed elsewhere or safe
    visited.add(name)
    if name in recursive_rules: return False
    
    # If it's a known terminal (Capitalized), it's safe
    if name[0].isupper(): return True
    
    if name not in rules: return True # Primitive
    
    body = rules[name]
    refs = find_refs(body, rules)
    for ref in refs:
        if not is_subtree_safe_deep(ref, rules, recursive_rules, visited):
            return False
    return True

def extract_tail(alt):
    """
    Check if alt ends with ' : RECURSIVE_RULE'.
    Returns (head, tail) if match, else None.
    Example: '"key" : _json_object' -> ('"key"', '_json_object')
    """
    # Regex to find ':' followed by a single rule reference at the end
    # The grammar uses ':' (quoted colon) as the separator.
    m = re.search(r"\s*':'\s*([a-zA-Z_][a-zA-Z0-9_]*|\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\))\s*$", alt)
    if m:
        tail = m.group(1)
        # cleanup parens if like ( _json_object )
        if tail.startswith('(') and tail.endswith(')'):
            tail = tail[1:-1].strip()
            
        head = alt[:m.start()].strip()
        return head, tail
    return None

def factor_choices(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()
    
    rules = parse_ebnf(content)
    recursive_rules = find_recursive_rules(rules)
    
    new_rules = OrderedDict()
    
    # Track created factor rules to avoid duplicates
    # content_hash -> rule_name
    factor_cache = {} 
    
    for name, body in rules.items():
        # Only process Non-terminals that are mostly choices
        # And specifically ones we couldn't fully convert (so they contain recursion)
        # But we can also optimize the ones we DID convert if we want grouping.
        # Let's target rules starting with '_'
        if not name.startswith('_') or name.startswith('_json'):
            new_rules[name] = body
            continue
            
        choices = get_choices(body)
        if len(choices) < 2:
            new_rules[name] = body
            continue
            
        safe_alts = []
        unsafe_alts = []
        
        # Categorize choices
        for alt in choices:
            if is_safe_subtree(alt, rules, recursive_rules):
                safe_alts.append(alt)
            else:
                unsafe_alts.append(alt)
        
        # If everything is safe, we don't need to factor here (previous script handles it)
        # If everything is unsafe, check for common tails (Pattern Factoring)
        # Use simple heuristic: if we have mixed safe/unsafe, DEFINITELY factor.
        
        final_choices = []
        
        # 1. Handle Safe Alternatives (Group into one Terminal)
        if safe_alts:
            if len(safe_alts) > 1:
                # Create a new Terminal Rule
                term_name = name.upper() + "_SAFE"
                if term_name.startswith('_'): term_name = term_name[1:]
                
                # Deduplicate safe alts if exact match key exists? No, just keep simple
                term_body = " | ".join(safe_alts)
                
                # Check cache
                if term_body in factor_cache:
                    term_name = factor_cache[term_body]
                else:
                    new_rules[term_name] = f"( {term_body} )"
                    factor_cache[term_body] = term_name
                
                final_choices.append(term_name)
            else:
                final_choices.append(safe_alts[0])
        
        # 2. Handle Unsafe Alternatives (Group by Tail)
        tail_groups = defaultdict(list) # tail -> list of heads
        others = []
        
        for alt in unsafe_alts:
            res = extract_tail(alt)
            if res:
                head, tail = res
                if name == '_mem29':
                   print(f"DEBUG: Extracted head='{head}' tail='{tail}' from '{alt}'")
                
                # Check if head is terminal-convertible (no recursive refs)
                # Just checking if head is literals or capitalized is usually enough
                if is_safe_subtree(head, rules, recursive_rules):
                    tail_groups[tail].append(head)
                else:
                    others.append(alt)
            else:
                others.append(alt)
                
        # Process groups
        for tail, heads in tail_groups.items():
            if name == '_mem29':
               print(f"DEBUG: Grouping {tail} with {len(heads)} heads: {heads}")
            
            # Filter valid heads (must be literals or safe)
            
            # Create a combined rule for all keys pointing to this tail
            # e.g. KEYS_FOR_JSON_OBJECT ::= "enhanced" | "fleet" ...
            # Even if len(heads) == 1, this is worth it!
            # It moves the literal "key" into the tokenizer.
            
            # Generate a stable name
            # If 1 head, name after head: KEY_APQ
            # If >1 heads, name after tail: KEYS_FOR_JSON_OBJECT
            
            term_body = " | ".join(heads)
            
            if term_body in factor_cache:
                term_name = factor_cache[term_body]
            else:
                if len(heads) == 1:
                    # Name based on head logic
                    # head is likely ' "string" '
                    clean_head = re.sub(r'[^a-zA-Z0-9]', '', heads[0])
                    term_name = "KEY_" + clean_head.upper()
                else:
                    clean_tail = re.sub(r'[^a-zA-Z0-9]', '', tail)
                    term_name = "KEYS_FOR_" + clean_tail.upper()[-15:] # Truncate massive names
                
                if term_name.startswith('_'): term_name = term_name[1:]
                
                # Check collision
                idx = 1
                base_name = term_name
                while term_name in new_rules or term_name in rules:
                    term_name = f"{base_name}_{idx}"
                    idx += 1
            
                new_rules[term_name] = f"( {term_body} )"
                factor_cache[term_body] = term_name
            
            # Rewritten alt is: TERM_NAME ':' TAIL
            # Wait, the origin was HEAD ':' TAIL. 
            # If we group heads into TERM, we need to ensure the COLON is handled.
            # OPTION 1: TERM ::= HEAD -> Rule ::= TERM ':' TAIL
            # OPTION 2: TERM ::= HEAD ':' -> Rule ::= TERM TAIL (This is what user suggested!)
            # "Y ::= ... | ... " -> Then "Y : _json_object"
            # BUT user also had "X ::= 'key' : DEF" where colon is inside X.
            
            # Let's standardize: The TERMINAL shall include the COLON if possible?
            # User example: Y ::= '"apq"' | ... ; Rule ::= Y ':' _json_object
            # This keeps colon in parser? NO, user wants to simplify parser.
            # If colon is in parser, parser sees KEY + COLON + VALUE.
            # If colon is in terminal (KEY_COLON), parser sees KEY_COLON + VALUE. (2 tokens vs 3)
            # The latter is better.
            
            # But heads list currently is just the key string '"apq"'.
            # If we group them into Y, and say Y ':' tail, we are keeping colon in parser.
            # If we want colon in terminal, we must change heads to include colon.
            
            # Let's stick to user example: Y : _json_object
            # Wait, if Y = "key1" | "key2", then Y : _json_object parses as ("key1" | "key2") : _json_object.
            # This allows the parser to share the state for ":" and "_json_object".
            # This is GOOD.
            
            final_choices.append(f"{term_name} ':' {tail}")
        
        # Add remaining unsafe
        final_choices.extend(others)

        
        # Reconstruct Body
        new_body = " | ".join(final_choices)
        new_rules[name] = f"( {new_body} )"

    with open(output_file, 'w') as f:
        for name, body in new_rules.items():
            f.write(f"{name} ::= {body} ;\n")
    
    print(f"Factored grammar written to {output_file}")
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python optimize_factor_choices.py input.ebnf output.ebnf")
        sys.exit(1)
    factor_choices(sys.argv[1], sys.argv[2])
