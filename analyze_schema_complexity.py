
import json
import sys

def count_optional_props(schema, path="#"):
    counts = []
    
    if isinstance(schema, dict):
        if schema.get("type") == "object" or "properties" in schema:
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            optional_keys = [k for k in props.keys() if k not in required]
            
            counts.append({
                "path": path,
                "total": len(props),
                "required": len(required),
                "optional": len(optional_keys),
                "keys": list(props.keys())
            })
            
            for k, v in props.items():
                counts.extend(count_optional_props(v, f"{path}/{k}"))
                
        # Handle definitions
        for key in ["$defs", "definitions", "defs"]:
            if key in schema:
                for name, sub in schema[key].items():
                    counts.extend(count_optional_props(sub, f"{path}/{key}/{name}"))
                    
        # Handle allOf, anyOf, oneOf
        for key in ["allOf", "anyOf", "oneOf"]:
            if key in schema:
                for i, sub in enumerate(schema[key]):
                    counts.extend(count_optional_props(sub, f"{path}/{key}/{i}"))
                    
    return counts

def analyze_schema(filename):
    with open(filename) as f:
        data = json.load(f)
        
    schema = data.get("schema", data)
    counts = count_optional_props(schema)
    
    # Sort by optional count descending
    counts.sort(key=lambda x: x["optional"], reverse=True)
    
    print(f"Top 20 objects by optional property count:")
    for c in counts[:20]:
        print(f"{c['path']}: {c['optional']} optional (of {c['total']} total)")
        
    print("\nOptional property count distribution:")
    dist = {}
    for c in counts:
        n = c["optional"]
        dist[n] = dist.get(n, 0) + 1
        
    for n in sorted(dist.keys()):
        print(f"  {n} optional props: {dist[n]} objects")

if __name__ == "__main__":
    analyze_schema("gcg-paper/hard_schemas/data/ApolloRouter---apollo-router-2.9.0.json")
