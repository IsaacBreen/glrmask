import json
from typing import Any, Dict, List, Optional, Set, Tuple

class JsonSchemaToEbnf:
    """
    Converts a JSON Schema to an EBNF grammar string compatible with sep1.
    """
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.rules: List[str] = []
        self.rule_counter = 0
        self.definitions: Dict[str, str] = {}  # Map ref path to rule name
        self.pending_definitions: List[Tuple[str, Dict[str, Any]]] = [] # (rule_name, schema)

    def generate(self) -> str:
        """Generates the EBNF grammar string."""
        self.rules = []
        self.rule_counter = 0
        self.definitions = {}
        
        # Add standard whitespace/comment ignore rules
        self.rules.append("#![ignore(IGNORE)]")
        self.rules.append("root ::= root_schema EOF;")
        self.rules.append("EOF ::= '<|EOF|>';")
        self.rules.append("IGNORE ::= ( WS | COMMENT )+ ;")
        self.rules.append("WS ::= ( ' ' | '\\t' | '\\n' | '\\r' )+ ;")
        self.rules.append("COMMENT ::= '//' ( [^\\n\\r] )* ;")
        
        # Process definitions first if any
        if "$defs" in self.schema:
            self._process_definitions(self.schema["$defs"], "#/$defs/")
        if "definitions" in self.schema:
            self._process_definitions(self.schema["definitions"], "#/definitions/")

        # Generate root rule
        self._generate_rule_for_schema(self.schema, "root_schema")
        
        # Process any pending definitions (recursive or deferred)
        while self.pending_definitions:
            rule_name, schema = self.pending_definitions.pop(0)
            self._generate_rule_for_schema(schema, rule_name)

        return "\n".join(self.rules)

    def _process_definitions(self, defs: Dict[str, Any], prefix: str):
        for name, schema in defs.items():
            safe_name = self._sanitize(name)
            ref = f"{prefix}{name}"
            rule_name = f"def_{safe_name}_{self._next_id()}"
            self.definitions[ref] = rule_name
            self.pending_definitions.append((rule_name, schema))

    def _sanitize(self, name: str) -> str:
        import re
        return re.sub(r'[^a-zA-Z0-9_]', '_', name)

    def _next_id(self) -> int:
        self.rule_counter += 1
        return self.rule_counter

    def _generate_rule_for_schema(self, schema: Dict[str, Any], rule_name: str) -> str:
        """
        Generates a rule for the given schema and returns the rule name.
        If rule_name is provided, it uses that. Otherwise generates a new one.
        """
        
        # Handle $ref
        if "$ref" in schema:
            ref = schema["$ref"]
            if ref in self.definitions:
                # Direct alias to existing rule
                if rule_name != "root_schema" and not rule_name.startswith("def_"):
                     # If we need a specific name, create an alias
                     self.rules.append(f"{rule_name} ::= {self.definitions[ref]} ;")
                     return rule_name
                return self.definitions[ref]
            else:
                # Forward reference or external? Assume local for now
                print(f"Warning: Unresolved reference {ref}")
                # Fallback to generic json value
                self.rules.append(f"{rule_name} ::= generic_json_value ;")
                return rule_name

        # Handle const
        if "const" in schema:
            val = schema["const"]
            rule_body = self._value_to_ebnf(val)
            self.rules.append(f"{rule_name} ::= {rule_body} ;")
            return rule_name

        # Handle enum
        if "enum" in schema:
            options = [self._value_to_ebnf(val) for val in schema["enum"]]
            rule_body = " | ".join(options)
            self.rules.append(f"{rule_name} ::= {rule_body} ;")
            return rule_name

        # Handle anyOf, oneOf (treated same for grammar generation)
        if "anyOf" in schema or "oneOf" in schema:
            subschemas = schema.get("anyOf", []) or schema.get("oneOf", [])
            sub_rule_names = []
            for i, sub in enumerate(subschemas):
                sub_name = f"{rule_name}_opt{i}"
                sub_rule_names.append(self._generate_rule_for_schema(sub, sub_name))
            
            rule_body = " | ".join(sub_rule_names)
            self.rules.append(f"{rule_name} ::= {rule_body} ;")
            return rule_name

        # Handle allOf
        if "allOf" in schema:
            # Naive merge
            merged = {}
            for sub in schema["allOf"]:
                merged.update(sub) 
            return self._generate_rule_for_schema(merged, rule_name)

        # Handle type
        schema_type = schema.get("type")
        
        if schema_type == "object":
            self._generate_object_rule(schema, rule_name)
        elif schema_type == "array":
            self._generate_array_rule(schema, rule_name)
        elif schema_type == "string":
            self.rules.append(f"{rule_name} ::= STRING_LITERAL ;")
        elif schema_type == "number" or schema_type == "integer":
            self.rules.append(f"{rule_name} ::= NUMBER_LITERAL ;")
        elif schema_type == "boolean":
            self.rules.append(f"{rule_name} ::= 'true' | 'false' ;")
        elif schema_type == "null":
            self.rules.append(f"{rule_name} ::= 'null' ;")
        else:
            # No type specified? Could be anything.
            self.rules.append(f"{rule_name} ::= generic_json_value ;")

        return rule_name

    def _generate_object_rule(self, schema: Dict[str, Any], rule_name: str):
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        additional_properties = schema.get("additionalProperties", True) 

        if not properties and not additional_properties:
            # Empty object
            self.rules.append(f"{rule_name} ::= '{{' '}}' ;")
            return

        # Use a generic "member" rule that allows any of the defined properties.
        # object ::= '{' ( member (',' member)* )? '}'
        # member ::= prop1 | prop2 | ...
        
        member_options = []
        for prop_name, prop_schema in properties.items():
            safe_prop_name = self._sanitize(prop_name)
            val_rule = self._generate_rule_for_schema(prop_schema, f"{rule_name}_val_{safe_prop_name}_{self._next_id()}")
            key_str = json.dumps(prop_name)
            member_options.append(f"'{key_str}' ':' {val_rule}")
            
        if additional_properties:
             # Allow any string key
             member_options.append(f"STRING_LITERAL ':' generic_json_value")

        if not member_options:
             self.rules.append(f"{rule_name} ::= '{{' '}}' ;")
             return

        member_rule_name = f"{rule_name}_member"
        self.rules.append(f"{member_rule_name} ::= {' | '.join(member_options)} ;")
        self.rules.append(f"{rule_name} ::= '{{' ( {member_rule_name} ( ',' {member_rule_name} )* )? '}}' ;")


    def _generate_array_rule(self, schema: Dict[str, Any], rule_name: str):
        items_schema = schema.get("items", {})
        if not items_schema:
            # Generic array
            self.rules.append(f"{rule_name} ::= '[' ( generic_json_value ( ',' generic_json_value )* )? ']' ;")
            return

        item_rule = self._generate_rule_for_schema(items_schema, f"{rule_name}_item")
        self.rules.append(f"{rule_name} ::= '[' ( {item_rule} ( ',' {item_rule} )* )? ']' ;")

    def _value_to_ebnf(self, val: Any) -> str:
        if val is None:
            return "'null'"
        if isinstance(val, bool):
            return "'true'" if val else "'false'"
        if isinstance(val, (int, float)):
            return f"'{val}'"
        if isinstance(val, str):
            return f"'{json.dumps(val)[1:-1]}'" 
        return "'unknown'"

    def get_primitives(self) -> str:
        return """
STRING_LITERAL ::= '"' ( [^"\\\\] | ESCAPE_SEQUENCE )* '"' ;
ESCAPE_SEQUENCE ::= '\\\\' ( [\\'"\\\\bfnrtv] | 'x' HEX_DIGIT HEX_DIGIT | 'u' HEX_DIGIT HEX_DIGIT HEX_DIGIT HEX_DIGIT ) ;
HEX_DIGIT ::= [0-9a-fA-F] ;

NUMBER_LITERAL ::= '-'? ( '0' | [1-9] [0-9]* ) ( '.' [0-9]+ )? ( ( 'e' | 'E' ) ( '+' | '-' )? [0-9]+ )? ;

generic_json_value ::= STRING_LITERAL | NUMBER_LITERAL | generic_object | generic_array | 'true' | 'false' | 'null' ;
generic_object ::= '{' ( string_kv ( ',' string_kv )* )? '}' ;
generic_array ::= '[' ( generic_json_value ( ',' generic_json_value )* )? ']' ;
string_kv ::= STRING_LITERAL ':' generic_json_value ;
"""

def convert_schema_to_ebnf(schema: Dict[str, Any]) -> str:
    converter = JsonSchemaToEbnf(schema)
    grammar = converter.generate()
    grammar += converter.get_primitives()
    return grammar

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            schema = json.load(f)
        print(convert_schema_to_ebnf(schema))
