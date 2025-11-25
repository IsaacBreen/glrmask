import json
import sys

try:
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab = tokenizer.get_vocab()
    # Save as JSON
    with open("benchmarking/gpt2_vocab.json", "w") as f:
        json.dump(vocab, f)
    print("Saved benchmarking/gpt2_vocab.json")
except ImportError:
    print("transformers not installed")
except Exception as e:
    print(f"Error: {e}")
