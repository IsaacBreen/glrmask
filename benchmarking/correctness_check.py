import json
from typing import Dict, List, Tuple, Optional

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
except Exception:
    # Catch other errors like the attrs one
    JSONSCHEMA_AVAILABLE = False

def validate_json_output(output_str: str, schema: Dict) -> Tuple[bool, Optional[str]]:
    """
    Validates that the output string is valid JSON and conforms to the schema.
    Returns (is_valid, error_message).
    """
    try:
        # 1. Parse JSON
        data = json.loads(output_str)
    except json.JSONDecodeError as e:
        return False, f"JSON Parse Error: {e}"
        
    if JSONSCHEMA_AVAILABLE:
        try:
            # 2. Validate against schema
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as e:
            return False, f"Schema Validation Error: {e.message}"
        except Exception as e:
            return False, f"Validation Error: {e}"
    else:
        # Fallback: just check if it parsed as JSON
        pass
        
    return True, None

def decode_tokens(tokens: List[int], vocab: Dict[int, bytes]) -> str:
    """
    Decodes a list of token IDs into a string using the vocabulary.
    Handles basic byte decoding.
    """
    # Simple concatenation of bytes
    byte_stream = b""
    for tid in tokens:
        if tid in vocab:
            byte_stream += vocab[tid]
        else:
            # Placeholder or skip?
            pass
            
    # Decode utf-8
    try:
        return byte_stream.decode('utf-8')
    except UnicodeDecodeError:
        return byte_stream.decode('utf-8', errors='replace')
