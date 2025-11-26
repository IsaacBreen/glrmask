import sys
from pathlib import Path
import json

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from llguidance import JsonCompiler, LLTokenizer, LLInterpreter, TokenizerWrapper
    from transformers import AutoTokenizer
    print("llguidance imported successfully")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def debug_llguidance():
    # Setup tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    class HFTokenizerWithTokens:
        def __init__(self, hf_tok):
            self._hf_tok = hf_tok
            self.tokens = [hf_tok.convert_ids_to_tokens(i).encode('utf-8') if isinstance(hf_tok.convert_ids_to_tokens(i), str) else hf_tok.convert_ids_to_tokens(i) for i in range(hf_tok.vocab_size)]
            self.eos_token_id = hf_tok.eos_token_id
            self.bos_token_id = getattr(hf_tok, 'bos_token_id', None)
        
        def __getattr__(self, name):
            return getattr(self._hf_tok, name)
        
        def __call__(self, text):
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            return self._hf_tok.encode(text)
    
    wrapped_tokenizer = HFTokenizerWithTokens(hf_tokenizer)
    tokenizer_wrapper = TokenizerWrapper(wrapped_tokenizer)
    ll_tokenizer = LLTokenizer(tokenizer_wrapper)
    
    # Compile schema
    schema = {
        "type": "object", 
        "properties": {"name": {"type": "string"}}
    }
    compiler = JsonCompiler()
    compiled = compiler.compile(json.dumps(schema))
    
    # Create interpreter
    interpreter = LLInterpreter(ll_tokenizer, compiled)
    
    print("\nInterpreter attributes:")
    print(dir(interpreter))
    
    # Try to find mask method
    # interpreter.start_generation() ?
    
    # Try to call something
    try:
        print("Trying to get mask...")
        mask = interpreter.compute_mask()
        print(f"Mask type: {type(mask)}")
        print(f"Mask length: {len(mask)}")
        print(f"Mask content (first element type): {type(mask[0])}")
        print(f"Mask content: {mask}")
             
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_llguidance()
