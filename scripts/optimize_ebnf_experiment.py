import re
import sys

def split_toplevel(s, delimiter='|'):
    parts = []
    current = []
    depth = 0
    quote = None
    
    # iterate chars
    # We need basic tokenizer to handle quotes if | is inside quotes (unlikely in EBNF structure but safe)
    # EBNF here uses '...' or "..." (JSON_STRING uses ")
    
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        
        if quote:
            current.append(c)
            if c == quote:
                # check escape? EBNF simplistic usually
                # The file uses ' "..." '
                quote = None
        else:
            if c == "'" or c == '"':
                quote = c
                current.append(c)
            elif c == '(':
                depth += 1
                current.append(c)
            elif c == ')':
                depth -= 1
                current.append(c)
            elif c == delimiter and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(c)
        i += 1
        
    parts.append("".join(current))
    return parts

def optimize_ebnf(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    rules = {}
    rule_order = []
    
    # Parse 1 rule per line assumption (verified with view_file)
    # rule ::= ... ;
    rule_re = re.compile(r'^\s*(\S+)\s*::=\s*(.+?)\s*;\s*$')
    
    for line in lines:
        if not line.strip(): continue
        m = rule_re.match(line)
        if m:
            name = m.group(1)
            body = m.group(2)
            rules[name] = body
            rule_order.append(name)
    
    new_terminals = {} 
    mem_usage = {}

    def make_term_name(s):
        # s is the content inside quotes e.g. "apq"
        # s usually has " inside? 
        # In EBNF ' "apq" ' -> raw content is "apq"
        # We strip quotes
        clean_s = s.strip('"')
        clean = re.sub(r'[^a-zA-Z0-9]', '_', clean_s)
        return f"TERM_{clean}"

    # Scan all rules to find usage pattern
    # Pattern: ( M ( ',' M )* )
    # Note: EBNF uses single quoted comma ','
    
    usage_pattern = r"\(\s*([a-zA-Z0-9_]+)\s*\(\s*','\s*([a-zA-Z0-9_]+)\s*\)\*\s*\)"
    
    for name, body in rules.items():
        matches = re.finditer(usage_pattern, body)
        for m in matches:
            m1 = m.group(1)
            m2 = m.group(2)
            if m1 == m2:
                if m1 not in mem_usage:
                    mem_usage[m1] = []
                mem_usage[m1].append(name)
                
    modified_mem_rules = set()
    
    for mem_name in mem_usage:
        if mem_name not in rules: continue
        
        body = rules[mem_name]
        inner = body.strip()
        
        # Remove outer parens if present (standard structure)
        if inner.startswith('(') and inner.endswith(')'):
            inner = inner[1:-1].strip()
            
        alts = [x.strip() for x in split_toplevel(inner, '|')]
        
        new_alts_start = []
        new_alts_comma = []
        
        can_transform = True
        
        for alt in alts:
            # Check format: ' "string" ' : val  OR  JSON_STRING : val
            # EBNF literals are single-quoted. keys are usually double-quoted strings inside.
            # So ' "key" ' : val
            
            # Regex:
            # Group 1: key token ( '...' or JSON_STRING )
            # Group 2: remainder
            
            kv_match = re.match(r'^((?:\'[^\']*\')|JSON_STRING)\s*\':\'\s*(.+)$', alt, flags=re.DOTALL)
            if not kv_match:
                print(f"FAILED: {repr(alt)} (in {mem_name})")
                can_transform = False
                break
            
            key_token = kv_match.group(1)
            val_part = kv_match.group(2)
            
            # Create Terminals
            if key_token == 'JSON_STRING':
                term_name = "TERM_JSON_STRING_COLON"
                term_comma_name = "TERM_COMMA_JSON_STRING_COLON"
                term_def = "JSON_STRING ':'"
                term_comma_def = "',' JSON_STRING ':'"
            else:
                # Literal string ' "foo" '
                # content is "foo" (including double quotes)
                raw_full = key_token.strip("'") # "foo"
                term_name = make_term_name(raw_full) + "_COLON"
                term_comma_name = make_term_name(raw_full) + "_COMMA_COLON"
                term_def = f"{key_token} ':'"
                term_comma_def = f"',' {key_token} ':'"
            
            if term_name not in new_terminals:
                new_terminals[term_name] = term_def
            if term_comma_name not in new_terminals:
                new_terminals[term_comma_name] = term_comma_def
                
            new_alts_start.append(f"{term_name} {val_part}")
            new_alts_comma.append(f"{term_comma_name} {val_part}")
            
        if can_transform:
            print(f"Transformed {mem_name}")
            modified_mem_rules.add(mem_name)
            rules[mem_name] = "( " + " | ".join(new_alts_start) + " )"
            mem_comma_name = f"{mem_name}_comma"
            rules[mem_comma_name] = "( " + " | ".join(new_alts_comma) + " )"
            rule_order.append(mem_comma_name)

    # Update usages
    for name in rule_order:
        if name in new_terminals: continue
        body = rules[name]
        
        for mem_name in modified_mem_rules:
            mem_comma = f"{mem_name}_comma"
            escaped_mem = re.escape(mem_name)
            
            # Regex to find `( mem ( ',' mem )* )`
            # With careful matching of `','` terminal
            # Assuming usage pattern matches findings
            p = re.compile(rf'\(\s*{escaped_mem}\s*\(\s*\'\,\'\s*{escaped_mem}\s*\)\*\s*\)')
            
            if p.search(body):
                replacement = f"( {mem_name} ( {mem_comma} )* )"
                body = p.sub(replacement, body)
                
        rules[name] = body

    with open(output_file, 'w') as f:
        for name in rule_order:
            if name in rules:
                f.write(f"{name} ::= {rules[name]} ;\n")
        
        f.write("\n\n# New Terminals\n")
        for name, body in sorted(new_terminals.items()):
            f.write(f"{name} ::= {body} ;\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        pass # Handle usage error
    else:
        optimize_ebnf(sys.argv[1], sys.argv[2])
