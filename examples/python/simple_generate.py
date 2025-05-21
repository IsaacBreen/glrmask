from __future__ import annotations

import json
import textwrap
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
# Ensure you have the _sep1 module available in your PYTHONPATH or current directory
from _sep1 import (CompiledGrammar, GrammarConstraint,
                   GrammarConstraintState, GrammarExpr as ge, RegexExpr as Regex,
                   IncrementalParser)
from transformers import (AutoModelForCausalLM, AutoTokenizer, LogitsProcessor)

# --- Helper Functions ---

def debug_print(message: str):
    """Prints a message for debugging, without a newline if part of a sequence."""
    print(message, end='; ')

def timeit(func):
    """Decorator to time a function call."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        # Print the timing information on its own line
        print(f"Debug: {func.__name__} Time: {(end_time - start_time) * 1000:.2f} ms")
        return result
    return wrapper

# --- Grammar Definition ---

def define_fruit_grammar_rules() -> List[Tuple[str, Any]]:
    """Defines the rules for a simple fruit-based natural language grammar."""

    # Helper to create a "lexical" rule: a choice of literal strings,
    # each implicitly preceded by consuming the IGNORE rule.
    def make_lexical_rule(name: str, choices: List[str]) -> Tuple[str, Any]:
        return (name, ge.choice([ge.literal(s.encode('utf-8')) for s in choices]))

    def sep(sequents: List[Any], sep: Any = ge.ref("IGNORE")) -> Any:
        sep_sequents = []
        sep_sequents.append(sequents[0])
        for s in sequents[1:]:
            sep_sequents.append(sep)
            sep_sequents.append(s)
        return ge.sequence(sep_sequents)


    # The first rule in the list is taken as the start rule by CompiledGrammar
#     rules: List[Tuple[str, Any]] = [("start_rule", ge.ref("sentences"))]

#     rules.append(("sentences", ge.sequence([ge.ref("sentence"), ge.repeat(ge.regex(Regex.eat_u8(ord('\n')))), ge.ref("sentence")])))
#     rules.append(make_lexical_rule("sentences", ["the", "apple", "is", "a", "person"]))
#     rules.append(("sentences", ge.literal("the apple is a person".encode())))
    return [("start", ge.literal("the apple is a person".encode()))]

    # IGNORE rule: optional spaces. This rule itself is not wrapped by IGNORE.
    # It allows zero or more spaces.
    rules.append(("IGNORE", ge.regex(Regex.rep(Regex.eat_u8(ord(' '))))))

    # Lexical rules (tokens of our grammar)
    rules.append(make_lexical_rule("Det", ["a", "the"]))
    rules.append(make_lexical_rule("Noun", ["apple", "banana", "person"]))
    rules.append(make_lexical_rule("Verb", ["eats", "likes", "is"]))
    rules.append(make_lexical_rule("Adj", ["tasty", "red", "happy"]))
    rules.append(make_lexical_rule("Period", ["."]))
    rules.append(make_lexical_rule("And", ["and"]))

    # Syntactic rules (combinations of other rules)
    # An NP (Noun Phrase) is a Determiner followed by a Noun, or just a Noun.
    # e.g., "the apple", "person"
    # Components (Det, Noun) already handle their own leading IGNORE.
    rules.append(("NP", ge.choice([
        sep([ge.ref("Det"), ge.ref("Noun")]),
        ge.ref("Noun")
    ])))

    # A VP (Verb Phrase) is a Verb followed by an NP, or a Verb followed by an Adjective.
    # e.g., "eats the apple", "is tasty"
    rules.append(("VP", ge.choice([
        sep([ge.ref("Verb"), ge.ref("NP")]),
        sep([ge.ref("Verb"), ge.ref("Adj")])
    ])))

    # A 'phrase' is an NP followed by a VP.
    # e.g., "the person eats an apple"
    rules.append(("phrase", sep([
        ge.ref("NP"), ge.ref("VP")
    ])))

    # A 'sentence' is a phrase ending with a Period,
    # OR a phrase followed by 'And' and another sentence (recursion).
    # e.g., "a person is happy.", "an apple is red and a banana is tasty."
    rules.append(("sentence", ge.choice([
        sep([ge.ref("phrase"), ge.ref("Period")]),
        sep([ge.ref("phrase"), ge.ref("And"), ge.ref("sentence")])
    ])))

    return rules

# --- Logits Processor ---

class GrammarConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, grammar_constraint_state: GrammarConstraintState, llm_token_to_id: Dict[bytes, int]):
        self.grammar_constraint_state = grammar_constraint_state
        self.seen_input_ids: List[int] = []
        # Create id_to_llm_token from llm_token_to_id for decoding tokens for printing
        self.id_to_llm_token: Dict[int, bytes] = {id_val: token_bytes for token_bytes, id_val in llm_token_to_id.items()}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids is typically shape [batch_size, sequence_length], batch_size=1 for generate
        current_input_ids = input_ids[0].tolist()
        new_token_ids = current_input_ids[len(self.seen_input_ids):]

        # Decode current full sequence for printing
        current_full_string_bytes = b"".join(self.id_to_llm_token.get(token_id, b'?') for token_id in current_input_ids)
        try:
            # Use errors='replace' to handle potential partial UTF-8 sequences during generation
            current_full_string_decoded = current_full_string_bytes.decode('utf-8', errors='replace')
        except Exception: # Should not be reached if errors='replace' is effective
            current_full_string_decoded = str(current_full_string_bytes) # Fallback
        print(f"Current String: {current_full_string_decoded!r}")

        print("Getting Mask...")
        # Time the get_mask operation
        mask = timeit(self.grammar_constraint_state.get_mask)()

        # Ensure mask has the same size as the vocab dimension in scores
        vocab_size = scores.shape[-1]
        if len(mask) < vocab_size:
            padding = np.zeros(vocab_size - len(mask), dtype=bool)
            mask = np.concatenate((mask, padding))
        elif len(mask) > vocab_size:
            mask = mask[:vocab_size]

        mask_ids = np.where(mask)[0]

        mask_tokens_str: List[str] = []
        for mid in mask_ids:
            tok_bytes = llm_token_id_to_token[mid]
            mask_tokens_str.append(tok_bytes.decode('utf-8'))

        print(f"Mask Info: Allowed IDs Count: {len(mask_ids)}")
        print(f"Mask Info: Allowed IDs (sample): {textwrap.shorten(str(mask_ids.tolist()), width=120)}")
        print(f"Mask Info: Allowed Tokens (sample): {mask_tokens_str[:100]}")

        # Extract and print first characters of allowed tokens for a quick glance
        first_chars = sorted(list(set(
            t_str[0] for t_str in mask_tokens_str if t_str and not t_str.endswith("(undecodable)")
        )))
        print(f"Mask Info: Allowed First Chars: {''.join(first_chars)}")
        print("-" * 30) # Separator for readability

        for token_id in new_token_ids:
            token_bytes = self.id_to_llm_token.get(token_id, b'?') # Get byte representation of the token
            try:
                token_str = token_bytes.decode('utf-8', errors='replace')
            except Exception:
                token_str = str(token_bytes)

            print(f"Committing Token: {token_str!r} (ID: {token_id})")

            # Time the commit operation
            timeit(self.grammar_constraint_state.commit)(token_id)

            print("Getting Mask...")
            # Time the get_mask operation
            mask = timeit(self.grammar_constraint_state.get_mask)()

            # Ensure mask has the same size as the vocab dimension in scores
            vocab_size = scores.shape[-1]
            if len(mask) < vocab_size:
                padding = np.zeros(vocab_size - len(mask), dtype=bool)
                mask = np.concatenate((mask, padding))
            elif len(mask) > vocab_size:
                mask = mask[:vocab_size]

            mask_ids = np.where(mask)[0]

            mask_tokens_str: List[str] = []
            for mid in mask_ids:
                tok_bytes = llm_token_id_to_token[mid]
                mask_tokens_str.append(tok_bytes.decode('utf-8'))

            print(f"Mask Info: Allowed IDs Count: {len(mask_ids)}")
            print(f"Mask Info: Allowed IDs (sample): {textwrap.shorten(str(mask_ids.tolist()), width=120)}")
            print(f"Mask Info: Allowed Tokens (sample): {mask_tokens_str[:100]}")

            # Extract and print first characters of allowed tokens for a quick glance
            first_chars = sorted(list(set(
                t_str[0] for t_str in mask_tokens_str if t_str and not t_str.endswith("(undecodable)")
            )))
            print(f"Mask Info: Allowed First Chars: {''.join(first_chars)}")
            print("-" * 30) # Separator for readability

        self.seen_input_ids = current_input_ids


        # Apply mask to scores: set scores of disallowed tokens to -infinity
        # Ensure scores are float for -np.inf
        processed_scores_np = np.where(mask, scores.cpu().to(torch.float32).numpy(), -np.inf)
        return torch.tensor(processed_scores_np, dtype=scores.dtype, device=scores.device)

# --- Text Generation Function ---

def generate_constrained_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    grammar_processor: GrammarConstrainedLogitsProcessor,
    pre_prompt_text: str,
    constrained_prompt_text: str,
    max_new_tokens: int = 50
) -> str:

    # Encode prompts. add_special_tokens=False gives more control over BOS/EOS.
    # Ensure the resulting tensors are of dtype torch.long.
    pre_prompt_ids = tokenizer.encode(
        pre_prompt_text,
        return_tensors="pt",
        add_special_tokens=False
    ).to(device=model.device, dtype=torch.long) # Ensure dtype is long

    constrained_prompt_ids = tokenizer.encode(
        constrained_prompt_text,
        return_tensors="pt",
        add_special_tokens=False
    ).to(device=model.device, dtype=torch.long) # Ensure dtype is long

    full_input_ids = torch.cat([pre_prompt_ids, constrained_prompt_ids], dim=1)

    # Configure the grammar processor for this specific generation call.
    # `seen_input_ids` are those tokens that precede the grammar-constrained part.
    grammar_processor.seen_input_ids = pre_prompt_ids[0].tolist() if pre_prompt_ids.numel() > 0 else []

    print(f"Generation Details:")
    print(f"  Pre-prompt (not grammar constrained): {pre_prompt_text!r} (IDs: {pre_prompt_ids.tolist()})")
    print(f"  Constrained prompt (start of grammar): {constrained_prompt_text!r} (IDs: {constrained_prompt_ids.tolist()})")
    print(f"  Full initial input to model: {full_input_ids.tolist()} (dtype: {full_input_ids.dtype})") # Added dtype print
    print(f"  Max new tokens to generate: {max_new_tokens}")
    print("-" * 30)

    output_ids = model.generate(
        full_input_ids,
        max_new_tokens=max_new_tokens,
        logits_processor=[grammar_logits_processor],
        # Use a pad token ID (e.g., EOS token ID if PAD is not set)
        pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
        # Using greedy search for simplicity and predictability with grammar constraints
        do_sample=False,
        num_beams=1,
    )

    print("-" * 30) # Separator after generation loop finishes
    print(f"Total output token IDs (prompt + generation): {output_ids[0].tolist()}")

    full_output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_output_text

# --- Main Execution ---

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-Coder-0.5B"
    # model_name = "gpt2" # Alternative smaller model for faster testing

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad_token to eos_token if not already set (common practice for some models)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Info: tokenizer.pad_token set to tokenizer.eos_token ({tokenizer.eos_token})")

    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # model.to('cuda') # Uncomment if a CUDA-enabled GPU is available

    # Prepare LLM token maps.
    # Tokenizers like Qwen's might use special characters (e.g., ' ' U+2581) for spaces.
    # These need to be converted to standard spaces ' ' for matching grammar literals.
    raw_tokenizer_vocab = tokenizer.get_vocab()
    processed_tokenizer_vocab: Dict[str, int] = {
        k.replace("Ġ", " "): v  # Replace Qwen's space char U+2581 with standard space ' '
        for k, v in raw_tokenizer_vocab.items()
    }

    # The GrammarConstraint expects token keys as bytes.
    llm_token_to_id: Dict[bytes, int] = {
        token_str.encode('utf-8'): token_id
        for token_str, token_id in processed_tokenizer_vocab.items()
    }
    llm_token_id_to_token = {token_id: token_str for token_str, token_id in llm_token_to_id.items()}

    max_token_id_val = 0
    if llm_token_to_id: # Check if vocab is not empty
        max_token_id_val = max(llm_token_to_id.values())
    else: # Should not happen with a valid tokenizer
        print("Warning: LLM token vocabulary is empty after processing.")

    print(f"Info: Processed tokenizer vocabulary size: {len(llm_token_to_id)}")
    print(f"Info: Max token ID in vocabulary: {max_token_id_val}")

    print("\nDefining fruit grammar...")
    grammar_rules = define_fruit_grammar_rules()
    compiled_grammar = CompiledGrammar(grammar_rules)
    print("Grammar: Compiled successfully. Rules defined:")
    for i, (name, _) in enumerate(grammar_rules):
        print(f"  Rule {i}: {name}")
    compiled_grammar.print() # This can be very verbose; uncomment for deep debugging

    # DEMO: Incremental Parser
    input_text = "the apple"
    parser_state = IncrementalParser(compiled_grammar) # Use the imported class
    print(f"Initial valid: {parser_state.is_valid()}")
    assert parser_state.is_valid()
    parser_state.feed(input_text.encode("utf-8"))
    print(f"After '{input_text}': valid={parser_state.is_valid()}")
    assert parser_state.is_valid()
    print("--- End Incremental Parser Demo ---")

    print("\nInitializing GrammarConstraint...")
    # max_token_id_val must be the highest token ID used by the tokenizer.
    grammar_constraint = GrammarConstraint(compiled_grammar, llm_token_to_id, max_token_id_val)

    # Test JSON serialization/deserialization (demonstrates persistence capability)
    print("Grammar: Serializing GrammarConstraint to JSON...")
    json_string = grammar_constraint.to_json_string()
    # print(f"Debug: Serialized JSON (first 200 chars): {json_string[:200]}...") # Optional

    print("Grammar: Deserializing GrammarConstraint from JSON...")
    grammar_constraint_from_json = GrammarConstraint.from_json_string(json_string)
    print("Grammar: Deserialized successfully.")

    # Use the deserialized constraint for the actual generation
    active_grammar_constraint = grammar_constraint_from_json

    # Initialize state and processor for the generation task
    print("\nInitializing GrammarConstraintState and LogitsProcessor for generation...")
    current_grammar_state = GrammarConstraintState(active_grammar_constraint)
    grammar_logits_processor = GrammarConstrainedLogitsProcessor(current_grammar_state, llm_token_to_id)

    # Print the initial mask
    mask = current_grammar_state.get_mask()

    mask_ids = np.where(mask)[0]

    mask_tokens_str: List[str] = []
    for mid in mask_ids:
        tok_bytes = llm_token_id_to_token[mid]
        mask_tokens_str.append(tok_bytes.decode('utf-8'))

    print(f"Mask Info: Allowed IDs Count: {len(mask_ids)}")
    print(f"Mask Info: Allowed IDs (sample): {textwrap.shorten(str(mask_ids.tolist()), width=120)}")
    print(f"Mask Info: Allowed Tokens (sample): {mask_tokens_str[:100]}")

    # Extract and print first characters of allowed tokens for a quick glance
    first_chars = sorted(list(set(
        t_str[0] for t_str in mask_tokens_str if t_str and not t_str.endswith("(undecodable)")
    )))
    print(f"Mask Info: Allowed First Chars: {''.join(first_chars)}")
    print("-" * 30) # Separator for readability



    # --- Perform Constrained Text Generation ---
    print("\n--- Starting Constrained Text Generation ---")

    # Define the starting point for generation.
    # `pre_prompt_text` is text that comes *before* grammar-constrained generation begins.
    # `constrained_prompt_text` is the initial text that *must* conform to the grammar.

    # Example 1: Start with "the " and let the model complete.
    pre_prompt = ""
    constrained_prompt = "the apple"

    # Example 2: Start completely empty (model generates from the absolute beginning of the grammar).
    # pre_prompt = ""
    # constrained_prompt = ""

    # Example 3: Start with a more complete phrase.
    # pre_prompt = ""
    # constrained_prompt = "a person eats "

    # Time the entire generation process
    generated_text_output = timeit(generate_constrained_text)(
        model,
        tokenizer,
        grammar_logits_processor,
        pre_prompt_text=pre_prompt,
        constrained_prompt_text=constrained_prompt,
        max_new_tokens=5  # Generate a short sequence
    )

    print("\n--- Generation Complete ---")
    print(f"Final Generated Text (Prompt + Completion):\n{generated_text_output}")