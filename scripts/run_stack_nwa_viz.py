#!/usr/bin/env python3
import sys
import json
import os
from pathlib import Path

# Add components path
components_path = Path("/Users/isaacbreen/Projects2/grammars2024/gcg-paper/paper/figures/components")
sys.path.append(str(components_path)) # Add components dir directly

from build_irs import IRBuilder

def main():
    root = Path("/Users/isaacbreen/Projects2/grammars2024")
    artifacts_path = root / "pipeline_artifacts.json"
    vocab_path = root / "vocab.json"
    
    if not artifacts_path.exists():
        print(f"Error: {artifacts_path} not found")
        return
        
    print(f"Loading artifacts from {artifacts_path}...")
    with open(artifacts_path) as f:
        raw_data = json.load(f)

    print(f"Loading vocab from {vocab_path}...")
    with open(vocab_path) as f:
        vocab_data = json.load(f)
        vocab_dict = vocab_data

    # We need to wrap it in a class that behaves like the `raw` object build_irs expects
    class RawWrapper:
        def __init__(self, d, v):
            self.pipeline_artifacts = d
            self.vocab = v
            self.dwa_raw = None # Not needed for forced stack path
            self.terminal_id_to_name = d.get("terminal_names", {}) # Fallback
            # It also needs original_vocab_size?
            self.tokenizer_dfa_raw = None
            self.parse_table = None
            self.productions = None
            self.terminals = None
            self.nonterminals = None
            self.internal_to_original = d.get("internal_to_original", [])
            self.max_original_llm_token_id = d.get("original_vocab_size", 50257) - 1
             
    raw_obj = RawWrapper(raw_data, vocab_dict)
            
    # However build_irs expects `PIPELINE_IR_BUILDER(raw_data_object)`?
    # Let's check constructor.
    # It takes `raw_data: "PipelineData"`.
    # PipelineData is a simple dataclass or object. 
    # I'll try to just pass the RawWrapper.
    
    builder = IRBuilder(raw_obj)
    # Inject original_vocab_size manually if needed?
    # build_irs: self.original_vocab_size = self.raw.pipeline_artifacts.get('original_vocab_size', 50257)
    # So it reads from artifacts.
    builder.original_vocab_size = raw_data.get('original_vocab_size', 50257) # Default GPT2
    
    if "terminal_dwa" not in raw_data:
        print("WARNING: terminal_dwa not in pipeline_artifacts!")
        # It might be in skeleton_dwa
        if "skeleton_dwa" in raw_data:
            print("Found skeleton_dwa, using that.")
        else:
            print("ERROR: No terminal_dwa data found. Cannot proceed.")
            return

    print("Building UnresolvedNWA IR (Stack-Driven)...")
    ir = builder.build_unresolved_nwa_ir()
    
    # Serialize to JSON
    # IR is a dataclass, use to_dict or just json.dump with default
    from dataclasses import asdict
    ir_dict = asdict(ir)
    
    output_json = root / "target/stack_nwa.json"
    output_json.parent.mkdir(exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(ir_dict, f, indent=2)
    print(f"Dumped IR to {output_json}")
    
    # Now run viewer builder
    viewer_script = components_path / "ir_to_html/nwa_viewer_builder.py"
    output_html = root / "target/stack_nwa_viewer.html"
    
    import subprocess
    cmd = [
        sys.executable, str(viewer_script),
        str(output_json), str(output_html),
        "--vocab", str(vocab_path),
        "--title", "Stack-Driven NWA"
    ]
    
    print(f"Running viewer builder: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Done! View at {output_html}")

if __name__ == "__main__":
    main()
