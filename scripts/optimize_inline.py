import re
import sys

def optimize_inline(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    rules = {}
    rule_order = []
    
    # Parse rules
    rule_re = re.compile(r'^\s*(\S+)\s*::=\s*(.+?)\s*;\s*$')
    
    for line in lines:
        if not line.strip(): continue
        m = rule_re.match(line)
        if m:
            name = m.group(1)
            body = m.group(2)
            rules[name] = body
            rule_order.append(name)

    # Identify inlineable rules
    # We want to inline generated names like _mem123, _pv123, _alt123, _type123
    # We explicitly EXCLUDE _def to avoid duplication explosion.
    
    inline_pattern = re.compile(r'^(_mem|_pv|_alt|_type)\d+$')
    
    # We need to handle dependencies.
    # Simple approach: Repeat replacement pass until no more inlineable tokens found in kept rules.
    # To avoid infinite loops in case of recursion (shouldn't happen for these), we add a max pass limit.
    
    # Optimization: Perform topological sort or just repeated passes?
    # Repeated passes is easier to implement.
    
    print(f"Loaded {len(rules)} rules.")
    
    # Targets to keep:
    # Everything NOT matching inline_pattern
    # BUT we also want to inline into the targets effectively.
    # Effectively, we want final grammar to only have "kept" rules.
    # All definition bodies of "kept" rules should be expanded.
    
    kept_rules = []
    for r in rule_order:
        if not inline_pattern.match(r):
            kept_rules.append(r)
            
    print(f"Kept rules: {len(kept_rules)} (e.g. {kept_rules[:5]})")
    print(f"Inlineable candidates: {len(rules) - len(kept_rules)}")
    
    # We perform expansion on ALL rules, then discard inlineable ones.
    # Wait, if A uses B, and we inline B into A. A is now expanded.
    # If we inline A into C. C gets expanded A.
    # So we should expand "bottom up" or just repeat.
    
    # Let's count references to estimate efficiency?
    # repeated pass:
    
    for i in range(20): # Max 20 passes
        changed_count = 0
        
        # We want to substitute references in `rules` bodies.
        # Reference is a whole word matching a rule name.
        
        # Determine current inlineables that are still used?
        # Actually, let's just loop over all rules and try to expand them 
        # using the definitions from *other* inlineable rules.
        
        # Optim strategy:
        # Pre-compile regexes for each inlineable rule? Too many (3000 rules).
        
        # Invert: Iterate over inlineables, and replace them in all other rules.
        # To succeed in one pass (mostly), we should process inlineables that depend on nothing (or only kept rules) first.
        # i.e. Bottom-up.
        
        # Build dependency graph
        deps = {name: set() for name in rules}
        # Find refs
        # Tokenizer: just split by non-word chars?
        # EBNF tokens: [a-zA-Z0-9_]+
        
        token_re = re.compile(r'\b([a-zA-Z0-9_]+)\b')
        
        for name, body in rules.items():
            for m in token_re.finditer(body):
                ref = m.group(1)
                if ref in rules and ref != name and inline_pattern.match(ref):
                    deps[name].add(ref)
                    
        # Topological order
        # We want to process X before Y if Y depends on X.
        # i.e. inline X into Y.
        
        # Khan's algo
        # Sort such that if Y depends on X, X comes first.
        # Nodes: all inlineable rules.
        # Edges: Y -> X (Y uses X)
        
        # Wait, if we replace X in Y, Y body changes.
        # We want to replace X in Y. So we need definitions of X to be ready.
        # Ideally X should be fully expanded (depend only on kept rules) before we inline it into Y.
        # So yes, X comes first if X uses nothing (or kept).
        # Y uses X. So Y -> X dependency.
        # So X is leaf.
        
        # Only consider inlineable subset for sorting
        inline_set = {r for r in rules if inline_pattern.match(r)}
        
        # Filter deps to only inline_set
        graph = {r: set() for r in inline_set}
        for r in inline_set:
            if r in deps:
                graph[r] = deps[r].intersection(inline_set)
                
        # Topo sort
        sorted_inlineables = []
        visited = set()
        temp_visited = set()
        
        visit_count = 0 
        def visit(n):
            nonlocal visit_count
            visit_count += 1
            if visit_count > 100000: raise Exception("Cycle or too deep")
            if n in visited: return
            if n in temp_visited:
                # Cycle detected
                return
            temp_visited.add(n)
            for m in graph[n]:
                visit(m)
            temp_visited.remove(n)
            visited.add(n)
            sorted_inlineables.append(n)
            
        for n in inline_set:
            visit(n)
            
        # sorted_inlineables now has leaves first?
        # visit(m) called before append(n). So M appended before N.
        # M is dependency of N. So M comes before N.
        # Correct. We expand M, then N uses expanded M.
        
        print(f"Topological sort complete. {len(sorted_inlineables)} items.")
        
        # Perform substitution in order
        # However, we need to update *every usage* of M.
        # Since we ordered them, when we get to N, and N uses M, we substitute M into N.
        # But we also need to substitute M into Kept rules.
        # Or simpler:
        # Just update the definition of N using M.
        # AND update Kept rules using M.
        
        # Actually:
        # Iterate through sorted_inlineables (M).
        # For M:
        #   Get M's body.
        #   Parens wrap it (to be safe).
        #   Replace usages of M in *all other remaining rules* (referenced in sorted_inlineables AND kept_rules).
        
        # Map of current definitions
        current_rules = rules.copy()
        
        for m in sorted_inlineables:
            body = current_rules[m]
            # Wrap in group if it contains alternation or concat?
            # EBNF: A ::= ( ... ).
            # If body is `( ... )`, we don't need extra parens, but safe to add `( ... )` if not present.
            # Most generated definitions seem wrapped `( ... )` or start with `( ... )` or are `...`.
            # To be safe: `( body )`
            
            # Optimization: check if parens needed
            if not (body.strip().startswith('(') and body.strip().endswith(')')):
                 replacement = f"( {body} )"
            else:
                 replacement = body
                 
            # Regex for token replacement
            # \bM\b provided it's not inside quotes? 
            # Our tokens are simple names.
            ref_re = re.compile(rf'\b{m}\b')
            
            # Replace in all rules that might use it
            # We only need to update rules that come *after* M in sort, AND kept rules.
            # But graph only keys inlineables.
            
            # List of targets: 
            # 1. inlineables coming after M
            # 2. kept_rules
            
            # Since we iterate M in order, we can just update 'rules' dict globally?
            # But wait, scanning all rules for every M is O(N^2). 3000 * 3000 = 9M. Slow but acceptable in python.
            # Can optimize by using the reverse usage map (where is M used?)
            
            # Let's revert to reverse usage map built once?
            # But usages change as we inline!
            # E.g. A uses B. B uses C.
            # Inline C into B. B changes.
            # Inline B into A. A changes (now contains C's body).
            # If we used usage map, A uses B. A does not use C initially.
            # But after inlining C into B, A still uses B.
            # Then we inline B (expanded) into A.
            # So we don't need to know A uses C.
            # We just need:
            # 1. Expand C.
            # 2. Update B (uses C).
            # 3. Update A (uses B).
            # So for current M, we only need to update rules that *directly* reference M.
            
            # Identify immediate users
            # Scan all rules?
            # Or build "immediate usage" map at start.
            # deps[Y] contains X. => reverse: users[X] contains Y.
            
            pass # logic continued below
        
        break # We only need one pass with topological sort!
        
    # Re-build users map
    users = {r: set() for r in rules}
    for r, deps_set in deps.items():
        for d in deps_set:
            if d in users:
                users[d].add(r)
                
    # Also kept rules depend on inlineables
    for r in kept_rules:
        body = rules[r]
        for m in token_re.finditer(body):
            ref = m.group(1)
            if ref in users:
                users[ref].add(r)
    
    # Execution with progress logging
    print("Starting inlining...")
    count = 0
    total = len(sorted_inlineables)
    
    # Pre-compile regexes for all inlineables? No, too many.
    # But for a given target rule T, it might use M1, M2, M3...
    # We visit M1. We find T uses M1. We sub M1 into T.
    
    # Critical optimization:
    # Instead of compiling regex for M every time, verify if M is in T's body string before regex?
    # Actually, `users[m]` tells us T uses M *initially*.
    # But after replacements, T might change.
    
    # But we can just cache compiled regexes?
    # Or just perform string replacement if tokens are distinct?
    # Our tokens are `_name123`. They are potentially distinct enough for replace() if flanked by boundaries.
    
    for m in sorted_inlineables:
        target_list = list(users.get(m, []))
        if not target_list:
            count += 1
            continue
            
        body = rules[m]
        # Wrap
        if not (body.strip().startswith('(') and body.strip().endswith(')')):
             body = f"( {body} )"
             
        # Compile regex once for M
        ref_re = re.compile(rf'\b{m}\b')
        
        for target in target_list:
            if target not in rules: continue
            
            old_body = rules[target]
            # Fast check
            if m not in old_body: continue
            
            new_body = ref_re.sub(body, old_body)
            rules[target] = new_body
            
            # Note: We do NOT need to update `users` here for future Ms?
            # Because we sorted M topo.
            # If T uses M2, and M2 uses M1.
            # We process M1 first.
            # T uses M1? No initially.
            # M2 uses M1. So M2 updated.
            # Later we process M2.
            # T uses M2. So T updated with M2's body (which contains M1's body).
            # So logic holds.
            
        count += 1
        if count % 100 == 0:
            print(f"Inlined {count}/{len(sorted_inlineables)} rules...")

    # Output
    with open(output_file, 'w') as f:
        for name in rule_order:
            # Only write kept rules
            if name in kept_rules:
                f.write(f"{name} ::= {rules[name]} ;\n")
    
    print(f"Finished. Wrote {len(kept_rules)} rules to {output_file}.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        pass
    else:
        optimize_inline(sys.argv[1], sys.argv[2])
