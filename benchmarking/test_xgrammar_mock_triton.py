"""Quick test to see if we can mock triton and use XGrammar."""

import sys

# Mock triton before importing xgrammar
class MockTriton:
    """Mock triton module for CPU-only usage."""
    def __getattr__(self, name):
        def mock_fn(*args, **kwargs):
            raise NotImplementedError(f"Triton function {name} not available on Mac")
        return mock_fn

sys.modules['triton'] = MockTriton()
sys.modules['triton.language'] = MockTriton()

# Now try to import xgrammar
try:
    from xgrammar import GrammarCompiler, TokenizerInfo
    from transformers import AutoTokenizer
    
    print("✓ XGrammar imported with mocked triton")
    
    # Try to create tokenizer info
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer_info = TokenizerInfo.from_huggingface(tokenizer)
    print("✓ TokenizerInfo created")
    
    # Try to compile a simple JSON schema
    compiler = GrammarCompiler(tokenizer_info)
    print("✓ GrammarCompiler created")
    
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    compiled = compiler.compile_json_schema(schema)
    print("✓ JSON schema compiled  successfully")
    
    print("\n SUCCESS! XGrammar works with mocked triton on Mac")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
