import outlines
import torch
from transformers import AutoTokenizer
import inspect

print("Inspecting outlines package...")

# Try to find where the JSON generation logic lives
try:
    import outlines.generate
    print("outlines.generate found")
    print(dir(outlines.generate))
except ImportError:
    print("outlines.generate not found")

try:
    import outlines.models
    print("outlines.models found")
except ImportError:
    print("outlines.models not found")

# We are looking for the underlying FSM or LogitsProcessor
# Usually it's in outlines.fsm or outlines.processors

try:
    import outlines.fsm.json_schema
    print("outlines.fsm.json_schema found")
except ImportError:
    print("outlines.fsm.json_schema not found")

try:
    from outlines.processors import StructuredLogitsProcessor
    print("StructuredLogitsProcessor found")
except ImportError:
    print("StructuredLogitsProcessor not found")

# Let's try to build a simple JSON generator and inspect it
try:
    model_name = "gpt2"
    model = outlines.models.transformers(model_name, device="cpu")
    schema = '''{"type": "object", "properties": {"name": {"type": "string"}}}'''
    generator = outlines.generate.json(model, schema)
    
    print(f"Generator type: {type(generator)}")
    print(f"Generator attributes: {dir(generator)}")
    
    if hasattr(generator, 'logits_processor'):
        print(f"Logits processor: {generator.logits_processor}")
        print(f"Logits processor type: {type(generator.logits_processor)}")
        print(dir(generator.logits_processor))
        
except Exception as e:
    print(f"Error creating generator: {e}")
