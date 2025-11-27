#!/usr/bin/env python3
"""
Comprehensive JSON Schema to EBNF Converter for Sep1

This converts JSON Schema (draft-07 compatible) to Sep1's EBNF format.
The goal is to generate grammars that are as permissive as the schema allows,
without trying to enforce semantic constraints that are impossible for CFGs
(like min/max bounds, uniqueItems, etc.).

Key design decisions:
1. Objects allow properties in any order (since JSON objects are unordered)
2. Required properties are not enforced in the grammar (rely on post-validation)
3. additionalProperties is respected when false
4. $ref/$defs are resolved
5. allOf is handled by merging schemas (best effort)
6. anyOf/oneOf are treated as alternations

Unsupported (intentionally - these require semantic validation, not syntax):
- minimum/maximum/exclusiveMinimum/exclusiveMaximum
- minLength/maxLength
- minItems/maxItems
- minProperties/maxProperties
- pattern (partially - could be added but regex translation is complex)
- uniqueItems
- dependencies (semantic constraint)
- if/then/else (semantic constraint)
- not (requires complement which CFGs can't express)
- format (semantic validation)
"""

import json
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import re


@dataclass
class EBNFGrammar:
    """Represents an EBNF grammar being built."""
    rules: List[str] = field(default_factory=list)
    rule_counter: int = 0
    definitions: Dict[str, str] = field(default_factory=dict)  # ref path -> rule name
    pending_refs: List[Tuple[str, Dict]] = field(default_factory=list)
    
    def new_rule_name(self, prefix: str = "r") -> str:
        self.rule_counter += 1
        return f"_{prefix}{self.rule_counter}"
    
    def add_rule(self, name: str, body: str):
        self.rules.append(f"{name} ::= {body} ;")
    
    def finalize(self) -> str:
        """Return the complete grammar as a string."""
        # Add primitive rules
        primitives = self._primitive_rules()
        return "\n".join(self.rules) + "\n" + primitives
    
    def _primitive_rules(self) -> str:
        # Sep1 requires uppercase names for terminal rules (those with character classes)
        return '''
WS ::= ( ' ' | '\\t' | '\\n' | '\\r' )* ;

JSON_STRING ::= '"' STRING_CHARS '"' ;
STRING_CHARS ::= ( STRING_CHAR | ESCAPE_SEQ )* ;
STRING_CHAR ::= [^"\\\\\\x00-\\x1f] ;
ESCAPE_SEQ ::= '\\\\' ( ["\\\\/bfnrt] | 'u' HEX HEX HEX HEX ) ;
HEX ::= [0-9a-fA-F] ;

JSON_INTEGER ::= MINUS? ( '0' | NON_ZERO_DIGIT DIGIT* ) ;
JSON_NUMBER ::= JSON_INTEGER ( '.' DIGIT+ )? ( EXPONENT )? ;
MINUS ::= '-' ;
NON_ZERO_DIGIT ::= [1-9] ;
DIGIT ::= [0-9] ;
EXPONENT ::= ( 'e' | 'E' ) ( '+' | '-' )? DIGIT+ ;

JSON_BOOL ::= 'true' | 'false' ;
JSON_NULL ::= 'null' ;

_json_value ::= _json_object | _json_array | JSON_STRING | JSON_NUMBER | JSON_BOOL | JSON_NULL ;
_json_object ::= '{' WS ( _json_kv ( ',' WS _json_kv )* )? WS '}' ;
_json_kv ::= JSON_STRING WS ':' WS _json_value ;
_json_array ::= '[' WS ( _json_value ( ',' WS _json_value )* )? WS ']' ;
'''


class JsonSchemaToEbnf:
    """Converts JSON Schema to EBNF grammar for Sep1."""
    
    def __init__(self, schema: Dict[str, Any]):
        self.root_schema = schema
        self.grammar = EBNFGrammar()
        self._resolved_refs: Dict[str, str] = {}  # ref path -> rule name
        self._generated_refs: set = set()  # refs that have been converted to rules
        self._current_ref_stack: List[str] = []  # Track current definition being processed
        
    def convert(self) -> str:
        """Convert the schema to EBNF and return as string."""
        # First, register all definitions
        self._register_definitions()
        
        # Generate main rule
        root_rule = self._convert_schema(self.root_schema, "root")
        
        # Process any pending definitions
        while self.grammar.pending_refs:
            ref_path, def_schema = self.grammar.pending_refs.pop(0)
            if ref_path not in self._generated_refs:
                self._generated_refs.add(ref_path)
                # Get the rule name (already registered)
                rule_name = self._resolved_refs.get(ref_path)
                if rule_name is None:
                    rule_name = self.grammar.new_rule_name("def")
                    self._resolved_refs[ref_path] = rule_name
                # Track current ref being processed to detect self-references
                self._current_ref_stack.append(ref_path)
                self._convert_schema(def_schema, rule_name)
                self._current_ref_stack.pop()
        
        return self.grammar.finalize()
    
    def _register_definitions(self):
        """Pre-register all $defs/definitions to handle forward references."""
        for key in ["$defs", "definitions"]:
            if key in self.root_schema:
                for name, def_schema in self.root_schema[key].items():
                    ref_path = f"#/{key}/{name}"
                    rule_name = self.grammar.new_rule_name("def")
                    self._resolved_refs[ref_path] = rule_name
                    self.grammar.pending_refs.append((ref_path, def_schema))
    
    def _resolve_ref(self, ref: str) -> Optional[str]:
        """Resolve a $ref and return the rule name."""
        if ref in self._resolved_refs:
            return self._resolved_refs[ref]
        
        # Try to resolve the reference
        if ref.startswith("#/"):
            parts = ref[2:].split("/")
            target = self.root_schema
            for part in parts:
                if isinstance(target, dict) and part in target:
                    target = target[part]
                else:
                    return None  # Can't resolve
            
            # Queue for processing
            rule_name = self.grammar.new_rule_name("ref")
            self._resolved_refs[ref] = rule_name
            self.grammar.pending_refs.append((ref, target))
            return rule_name
        
        return None  # External ref - not supported
    
    def _convert_schema(self, schema: Dict[str, Any], rule_name: str) -> str:
        """Convert a schema to EBNF rule(s). Returns the rule name."""
        
        # Handle boolean schemas
        if isinstance(schema, bool):
            if schema:
                self.grammar.add_rule(rule_name, "_json_value")
            else:
                # false schema - nothing matches. Use empty alternative.
                self.grammar.add_rule(rule_name, "'<NEVER>'")  # Should never match
            return rule_name
        
        if not isinstance(schema, dict):
            # Invalid schema
            self.grammar.add_rule(rule_name, "_json_value")
            return rule_name
        
        # Handle $ref
        if "$ref" in schema:
            ref = schema["$ref"]
            ref_rule = self._resolve_ref(ref)
            if ref_rule:
                self.grammar.add_rule(rule_name, ref_rule)
                return rule_name
            else:
                # Fallback to generic JSON
                self.grammar.add_rule(rule_name, "_json_value")
                return rule_name
        
        # Handle allOf
        if "allOf" in schema:
            merged = self._merge_all_of(schema["allOf"])
            # Also merge any sibling keys
            for key, val in schema.items():
                if key != "allOf":
                    merged[key] = val
            return self._convert_schema(merged, rule_name)
        
        # Handle anyOf / oneOf
        if "anyOf" in schema or "oneOf" in schema:
            subschemas = schema.get("anyOf") or schema.get("oneOf")
            alternatives = []
            for i, sub in enumerate(subschemas):
                sub_name = self.grammar.new_rule_name("alt")
                self._convert_schema(sub, sub_name)
                alternatives.append(sub_name)
            self.grammar.add_rule(rule_name, " | ".join(alternatives))
            return rule_name
        
        # Handle const
        if "const" in schema:
            body = self._value_to_literal(schema["const"])
            self.grammar.add_rule(rule_name, body)
            return rule_name
        
        # Handle enum
        if "enum" in schema:
            alternatives = [self._value_to_literal(val) for val in schema["enum"]]
            self.grammar.add_rule(rule_name, " | ".join(alternatives))
            return rule_name
        
        # Handle type
        schema_type = schema.get("type")
        
        if schema_type == "object":
            return self._convert_object(schema, rule_name)
        elif schema_type == "array":
            return self._convert_array(schema, rule_name)
        elif schema_type == "string":
            # Could handle format/pattern here, but for now just allow any string
            self.grammar.add_rule(rule_name, "JSON_STRING")
            return rule_name
        elif schema_type == "integer":
            self.grammar.add_rule(rule_name, "JSON_INTEGER")
            return rule_name
        elif schema_type == "number":
            self.grammar.add_rule(rule_name, "JSON_NUMBER")
            return rule_name
        elif schema_type == "boolean":
            self.grammar.add_rule(rule_name, "JSON_BOOL")
            return rule_name
        elif schema_type == "null":
            self.grammar.add_rule(rule_name, "JSON_NULL")
            return rule_name
        elif isinstance(schema_type, list):
            # Multiple types
            alternatives = []
            for t in schema_type:
                type_schema = {"type": t}
                # Copy relevant constraints
                for key in ["properties", "items", "additionalProperties", "required"]:
                    if key in schema:
                        type_schema[key] = schema[key]
                alt_name = self.grammar.new_rule_name("type")
                self._convert_schema(type_schema, alt_name)
                alternatives.append(alt_name)
            self.grammar.add_rule(rule_name, " | ".join(alternatives))
            return rule_name
        else:
            # No type specified or unknown - allow any JSON value
            self.grammar.add_rule(rule_name, "_json_value")
            return rule_name
    
    def _convert_object(self, schema: Dict[str, Any], rule_name: str) -> str:
        """Convert an object schema to EBNF."""
        properties = schema.get("properties", {})
        additional_props = schema.get("additionalProperties", True)
        pattern_props = schema.get("patternProperties", {})
        
        # If no properties defined and additional allowed, just use generic object
        if not properties and not pattern_props and additional_props is not False:
            self.grammar.add_rule(rule_name, "_json_object")
            return rule_name
        
        # If no properties and no additional allowed, empty object only
        if not properties and not pattern_props and additional_props is False:
            self.grammar.add_rule(rule_name, "'{' WS '}'")
            return rule_name
        
        # Build member alternatives
        member_alternatives = []
        
        # Add each defined property
        for prop_name, prop_schema in properties.items():
            prop_value_rule = self.grammar.new_rule_name("pv")
            self._convert_schema(prop_schema, prop_value_rule)
            
            # Escape the property name for EBNF
            escaped_name = self._escape_string_literal(prop_name)
            member_alternatives.append(f"'\"' '{escaped_name}' '\"' WS ':' WS {prop_value_rule}")
        
        # If additional properties allowed, add generic kv
        if additional_props is True:
            member_alternatives.append("_json_kv")
        elif isinstance(additional_props, dict) and additional_props:
            # additionalProperties is a schema
            additional_rule = self.grammar.new_rule_name("ap")
            self._convert_schema(additional_props, additional_rule)
            member_alternatives.append(f"JSON_STRING WS ':' WS {additional_rule}")
        
        # Create member rule
        member_rule = self.grammar.new_rule_name("mem")
        self.grammar.add_rule(member_rule, " | ".join(member_alternatives))
        
        # Object rule: { member (, member)* }
        self.grammar.add_rule(
            rule_name, 
            f"'{{' WS ( {member_rule} ( ',' WS {member_rule} )* )? WS '}}'"
        )
        return rule_name
    
    def _convert_array(self, schema: Dict[str, Any], rule_name: str) -> str:
        """Convert an array schema to EBNF."""
        items = schema.get("items")
        prefix_items = schema.get("prefixItems")  # JSON Schema draft 2020-12
        
        if items is None and prefix_items is None:
            # Any array
            self.grammar.add_rule(rule_name, "_json_array")
            return rule_name
        
        if prefix_items:
            # Tuple validation
            return self._convert_tuple_array(schema, rule_name, prefix_items)
        
        if isinstance(items, bool):
            if items:
                self.grammar.add_rule(rule_name, "_json_array")
            else:
                # Empty array only
                self.grammar.add_rule(rule_name, "'[' WS ']'")
            return rule_name
        
        if isinstance(items, dict):
            # All items must match schema
            item_rule = self.grammar.new_rule_name("item")
            self._convert_schema(items, item_rule)
            self.grammar.add_rule(
                rule_name,
                f"'[' WS ( {item_rule} ( ',' WS {item_rule} )* )? WS ']'"
            )
            return rule_name
        
        if isinstance(items, list):
            # Tuple-style (draft-07)
            return self._convert_tuple_array(schema, rule_name, items)
        
        # Fallback
        self.grammar.add_rule(rule_name, "_json_array")
        return rule_name
    
    def _convert_tuple_array(self, schema: Dict[str, Any], rule_name: str, 
                             prefix_items: List[Dict]) -> str:
        """Convert tuple-style array to EBNF."""
        additional_items = schema.get("additionalItems", schema.get("items", True))
        
        # Generate rules for each prefix item
        item_rules = []
        for i, item_schema in enumerate(prefix_items):
            item_rule = self.grammar.new_rule_name("ti")
            self._convert_schema(item_schema, item_rule)
            item_rules.append(item_rule)
        
        if not item_rules:
            if additional_items:
                self.grammar.add_rule(rule_name, "_json_array")
            else:
                self.grammar.add_rule(rule_name, "'[' WS ']'")
            return rule_name
        
        # Build the array body
        # First item, then rest with commas
        body_parts = [item_rules[0]]
        for item_rule in item_rules[1:]:
            body_parts.append(f"',' WS {item_rule}")
        
        # Add additional items if allowed
        if additional_items is True:
            body_parts.append(f"( ',' WS _json_value )*")
        elif isinstance(additional_items, dict):
            add_rule = self.grammar.new_rule_name("ai")
            self._convert_schema(additional_items, add_rule)
            body_parts.append(f"( ',' WS {add_rule} )*")
        
        body = " ".join(body_parts)
        self.grammar.add_rule(rule_name, f"'[' WS ( {body} )? WS ']'")
        return rule_name
    
    def _merge_all_of(self, subschemas: List[Dict]) -> Dict:
        """Merge allOf subschemas into a single schema (best effort).
        
        Self-referential $refs are skipped to avoid infinite recursion.
        """
        merged = {}
        merged_props = {}
        merged_required = []
        
        for sub in subschemas:
            if not isinstance(sub, dict):
                continue
            
            # Skip self-referential $refs
            if "$ref" in sub and len(sub) == 1:
                ref = sub["$ref"]
                # Check if this ref points to a definition we're currently processing
                if ref in self._current_ref_stack:
                    # Self-reference - skip it
                    continue
            
            # Merge properties
            if "properties" in sub:
                merged_props.update(sub["properties"])
            
            # Merge required
            if "required" in sub:
                merged_required.extend(sub["required"])
            
            # Copy other keys (last wins)
            for key, val in sub.items():
                if key not in ["properties", "required"]:
                    merged[key] = val
        
        if merged_props:
            merged["properties"] = merged_props
        if merged_required:
            merged["required"] = list(set(merged_required))
        
        return merged
    
    def _value_to_literal(self, val: Any) -> str:
        """Convert a JSON value to an EBNF literal."""
        if val is None:
            return "'null'"
        if isinstance(val, bool):
            return "'true'" if val else "'false'"
        if isinstance(val, int):
            return f"'{val}'"
        if isinstance(val, float):
            # Handle special cases
            s = json.dumps(val)
            return f"'{s}'"
        if isinstance(val, str):
            # Need to escape and wrap in quotes
            escaped = self._escape_string_literal(val)
            return f"'\"' '{escaped}' '\"'"
        if isinstance(val, list):
            # Full JSON array literal
            json_str = json.dumps(val, separators=(',', ':'))
            return f"'{self._escape_ebnf_string(json_str)}'"
        if isinstance(val, dict):
            # Full JSON object literal
            json_str = json.dumps(val, separators=(',', ':'))
            return f"'{self._escape_ebnf_string(json_str)}'"
        
        # Fallback
        json_str = json.dumps(val)
        return f"'{self._escape_ebnf_string(json_str)}'"
    
    def _escape_string_literal(self, s: str) -> str:
        """Escape a string for use as an EBNF literal (without outer quotes)."""
        result = []
        for c in s:
            if c == "'":
                result.append("\\'")
            elif c == "\\":
                result.append("\\\\")
            elif c == "\n":
                result.append("\\n")
            elif c == "\r":
                result.append("\\r")
            elif c == "\t":
                result.append("\\t")
            elif ord(c) < 32:
                result.append(f"\\x{ord(c):02x}")
            else:
                result.append(c)
        return "".join(result)
    
    def _escape_ebnf_string(self, s: str) -> str:
        """Escape a string for use as an EBNF string literal."""
        return s.replace("'", "\\'").replace("\\", "\\\\")


def convert_json_schema_to_ebnf(schema: Dict[str, Any]) -> str:
    """Main entry point: convert a JSON Schema to EBNF."""
    converter = JsonSchemaToEbnf(schema)
    return converter.convert()


def main():
    """CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert JSON Schema to EBNF")
    parser.add_argument("schema_file", nargs="?", help="JSON Schema file to convert")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("--test", action="store_true", help="Run tests")
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
        return
    
    if not args.schema_file:
        parser.error("schema_file is required unless --test is specified")
    
    with open(args.schema_file) as f:
        schema = json.load(f)
    
    # If it's a MaskBench file, extract the "schema" key
    if "schema" in schema and "meta" in schema:
        schema = schema["schema"]
    
    ebnf = convert_json_schema_to_ebnf(schema)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(ebnf)
        print(f"Wrote EBNF to {args.output}")
    else:
        print(ebnf)


def run_tests():
    """Run some basic tests."""
    print("Running tests...")
    
    # Test 1: Simple object
    schema1 = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }
    ebnf1 = convert_json_schema_to_ebnf(schema1)
    print("Test 1 (simple object):")
    print(ebnf1[:500])
    print()
    
    # Test 2: anyOf
    schema2 = {
        "anyOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    ebnf2 = convert_json_schema_to_ebnf(schema2)
    print("Test 2 (anyOf):")
    print(ebnf2[:500])
    print()
    
    # Test 3: enum
    schema3 = {
        "type": "string",
        "enum": ["red", "green", "blue"]
    }
    ebnf3 = convert_json_schema_to_ebnf(schema3)
    print("Test 3 (enum):")
    print(ebnf3[:500])
    print()
    
    # Test 4: Nested object
    schema4 = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"}
                }
            }
        }
    }
    ebnf4 = convert_json_schema_to_ebnf(schema4)
    print("Test 4 (nested object):")
    print(ebnf4[:500])
    print()
    
    # Test 5: $defs and $ref
    schema5 = {
        "$defs": {
            "person": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                }
            }
        },
        "type": "array",
        "items": {"$ref": "#/$defs/person"}
    }
    ebnf5 = convert_json_schema_to_ebnf(schema5)
    print("Test 5 ($defs and $ref):")
    print(ebnf5[:700])
    print()
    
    print("All tests completed!")


if __name__ == "__main__":
    main()
