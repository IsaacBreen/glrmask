from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
vocab = tokenizer.get_vocab()

# Sample tokens we care about
# ... existing code ...
print("\nFirst 20 tokens:")
for i in range(20):
    t_str = tokenizer.convert_ids_to_tokens(i)
    actual_str = tokenizer.convert_tokens_to_string([t_str])
    print(f"ID {i}: {repr(actual_str)}")

for s in samples:
    if s in vocab:
        tid = vocab[s]
        t_bytes = s.encode("utf-8")
        print(f"String: {repr(s)}, ID: {tid}, Bytes: {t_bytes}")
    else:
        print(f"String: {repr(s)} NOT in vocab")

# Try to find tokens by ID
ids = [tokenizer.encode(s, add_special_tokens=False) for s in samples]
print(f"IDs from encode: {ids}")

# ... existing code ...
for s, tid_list in zip(samples, ids):
    for tid in tid_list:
        decoded = tokenizer.decode([tid])
        t_str = tokenizer.convert_ids_to_tokens(tid)
        print(f"ID: {tid}, Decoded: {repr(decoded)}, Token: {repr(t_str)}")
        
        # Test conversion
        s_from_t = tokenizer.convert_tokens_to_string([t_str])
        print(f"  convert_tokens_to_string: {repr(s_from_t)}")
        
        # Try to get actual bytes if possible
        if hasattr(tokenizer, 'byte_decoder'):
            # Some tokenizers have this
            try:
                b_list = [tokenizer.byte_decoder[c] for c in t_str]
                actual_bytes = bytes(b_list)
                print(f"  Bytes from byte_decoder: {actual_bytes}")
            except Exception as e:
                print(f"  byte_decoder failed: {e}")
        else:
            # Fallback for Llama 3 / tiktoken based HF tokenizers
            # They often use a private property or specialized method
            pass
