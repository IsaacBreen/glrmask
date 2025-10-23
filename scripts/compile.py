import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import requests

# --- Helper Functions ---
def get_vocab(url: str | None, path: Path | None, cache_dir: Path, force_download: bool) -> dict[str, int]:
    """
    Loads a vocabulary from a local path or a URL.
    The vocabulary can be a JSON dictionary (token -> id) or a JSON list of strings.
    """
    if not url and not path:
        raise ValueError("Either --vocab-url or --vocab-path must be provided.")
    if url and path:
        raise ValueError("Provide either --vocab-url or --vocab-path, not both.")

    def _load_and_process_vocab(vocab_path: Path) -> dict[str, int]:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            print(f"Loaded vocabulary is a list of strings, converting to a dictionary ({len(data)} tokens).")
            return {token: i for i, token in enumerate(data)}
        raise TypeError(f"Unsupported vocabulary format in {vocab_path}. Expected a dict[str, int] or a list[str], but got {type(data)}.")

    if path:
        print(f"Loading vocabulary from local path: {path}")
        return _load_and_process_vocab(path)

    # Handle URL download and caching
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_name = url.split("/")[-1]
    cache_path = cache_dir / file_name

    if not cache_path.exists() or force_download:
        print(f"Downloading vocabulary from: {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"Saved vocabulary to cache: {cache_path}")
        except requests.RequestException as e:
            print(f"Error downloading vocabulary: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Loading vocabulary from cache: {cache_path}")

    return _load_and_process_vocab(cache_path)

def parse_len_ranges(ranges: list[str] | None) -> tuple[set[int], int | None]:
    """
    Parses a list of string representations of integer ranges.
    e.g., ["1", "3-5", "8-"] -> ({1, 3, 4, 5}, 8)
    """
    if not ranges:
        return set(), None
    
    allowed_lengths = set()
    min_len_unbounded = None

    for r in ranges:
        if '-' in r:
            parts = r.split('-', 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid range format: {r}")
            start_str, end_str = parts
            
            if not start_str:
                raise ValueError(f"Invalid range format: {r}. Start must be specified.")

            try:
                start = int(start_str)
            except ValueError:
                raise ValueError(f"Invalid start of range in '{r}'")

            if not end_str: # e.g. "8-"
                if min_len_unbounded is not None:
                    min_len_unbounded = min(min_len_unbounded, start)
                else:
                    min_len_unbounded = start
            else: # e.g. "3-5"
                try:
                    end = int(end_str)
                except ValueError:
                    raise ValueError(f"Invalid end of range in '{r}'")
                if start > end:
                    raise ValueError(f"Invalid range: start ({start}) > end ({end}) in '{r}'")
                allowed_lengths.update(range(start, end + 1))
        else:
            try:
                allowed_lengths.add(int(r))
            except ValueError:
                raise ValueError(f"Invalid length value: {r}")
            
    return allowed_lengths, min_len_unbounded

def filter_vocab(vocab: dict[str, int], allowed_lengths: set[int], min_len_unbounded: int | None) -> dict[str, int]:
    """
    Applies filters to the vocabulary based on token byte length.
    """
    if not allowed_lengths and min_len_unbounded is None:
        return vocab

    print(f"Filtering vocabulary by token byte length...")
    
    filtered = {}
    for token_str, token_id in vocab.items():
        # This processing matches the logic in the Rust tests
        processed_str = token_str.replace("Ġ", " ").replace("ą", "\n").replace("Ċ", "\n")
        token_len = len(processed_str.encode('utf-8'))
        
        keep = False
        if token_len in allowed_lengths:
            keep = True
        elif min_len_unbounded is not None and token_len >= min_len_unbounded:
            keep = True
            
        if keep:
            filtered[token_str] = token_id
            
    print(f"  -> Filtered vocabulary from {len(vocab)} to {len(filtered)} tokens.")
    return filtered


def run_compiler(compiler_path: Path, grammar_path: Path, vocab_path: Path, output_path: Path | None, recompile: bool, disable_progress_bar: bool, save_pc0: Path | None = None, from_pc0: Path | None = None, pc0_only: bool = False):
    """
    Runs the Rust grammar-compiler CLI tool, recompiling it first by default.
    """
    # Set environment variable to enable progress bars in the Rust code.
    # This will be passed to both cargo and the final executable.
    env = os.environ.copy()
    if not disable_progress_bar:
        env["ENABLE_PROGRESS_BAR"] = "1"

    if recompile:
        print("Building compiler with 'cargo build --release'...")
        try:
            # Run without capturing output to stream compilation progress.
            subprocess.run(
                ["cargo", "build", "--release"],
                check=True,
                env=env
            )
            print("Build successful.")
        except subprocess.CalledProcessError:
            # Cargo will have already printed its error to the console.
            print("Cargo build failed.", file=sys.stderr)
            sys.exit(1)

    if not compiler_path.exists():
        print(f"Error: Compiler executable not found at '{compiler_path}'.", file=sys.stderr)
        if recompile:
             print("The build process completed but the executable is not in the expected location.", file=sys.stderr)
        else:
             print("Try running without '--no-recompile' to build it.", file=sys.stderr)
        sys.exit(1)

    command = [
        str(compiler_path),
        "--grammar", str(grammar_path),
        "--vocab", str(vocab_path),
    ]

    if pc0_only:
        command.extend(["--save-precompute0", str(save_pc0)])
        command.append("--precompute0-only")
    else:
        if output_path:
            command.extend(["--output", str(output_path)])
        if from_pc0:
            print(f"Loading from precompute0 cache: {from_pc0}")
            command.extend(["--load-precompute0", str(from_pc0)])
        if save_pc0:
            # Ensure parent directory exists
            save_pc0.parent.mkdir(parents=True, exist_ok=True)
            print(f"Will save precompute0 cache to: {save_pc0}")
            command.extend(["--save-precompute0", str(save_pc0)])

    env_str = ""
    if not disable_progress_bar:
        env_str = "ENABLE_PROGRESS_BAR=1 "

    print(f"\nRunning compiler: {env_str}{' '.join(command)}")
    try:
        # Run the compiler, passing through its output and the environment variable.
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Grammar compilation failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Could not find the compiler executable at '{compiler_path}'", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function."""
    epilog = """
Examples:
  # 1. Generate only the precompute0 cache
  python scripts/compile.py \\
    --grammar src/js.ebnf \\
    --vocab-url "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json" \\
    --save-precompute0 .cache/pc0/js_gpt2.json.gz \\
    --precompute0-only

  # 2. Generate the final constraint using the pre-built cache
  python scripts/compile.py \\
    --grammar src/js.ebnf \\
    --vocab-url "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json" \\
    --from-precompute0 .cache/pc0/js_gpt2.json.gz \\
    --output .cache/constraints/js_gpt2.json.gz

  # 3. Do everything in one step (from scratch)
  python scripts/compile.py -g src/js.ebnf -o .cache/constraints/js_gpt2.json.gz --vocab-url <URL>

  # 4. Use a small, explicit vocabulary for testing
  python scripts/compile.py \\
    -g src/js.ebnf -o .cache/constraints/js_simple.json.gz \\
    --vocab-list '{' '}' '[' ']' ',"' '":' ' "' 'true' 'false' 'null' '123'

  # 5. Filter vocabulary to include only tokens of certain byte lengths
  python scripts/compile.py \\
    -g src/js.ebnf -o .cache/constraints/js_gpt2_filtered.json.gz \\
    --vocab-url "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json" \\
    --token-len 1 3-5 8-
"""
    parser = argparse.ArgumentParser(
        description="A helper script to compile a grammar constraint file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog
    )
    parser.add_argument("-g", "--grammar", type=Path, required=True, help="Path to the EBNF grammar file.")
    parser.add_argument("-o", "--output", type=Path, help="Path for the output compressed constraint file (.json.gz).")

    vocab_source_group = parser.add_mutually_exclusive_group()
    vocab_source_group.add_argument("--vocab-url", type=str, help="URL of the JSON vocabulary file to download.")
    vocab_source_group.add_argument("--vocab-path", type=Path, help="Path to a local JSON vocabulary file.")
    parser.add_argument("--vocab-list", type=str, nargs='+', help="A list of strings to use as the vocabulary. Can be combined with --vocab-url/--vocab-path.")

    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/vocabs"), help="Directory to cache downloaded vocabularies.")
    parser.add_argument("--compiler-path", type=Path, default=Path("target/release/grammar-compiler"), help="Path to the grammar-compiler executable.")
    parser.add_argument("--no-recompile", action="store_true", help="Skip recompiling the Rust grammar-compiler executable and use the existing one.")
    parser.add_argument("--force-download", action="store_true", help="Force re-downloading the vocabulary even if it exists in the cache.")
    parser.add_argument("--no-progress-bar", action="store_true", help="Disable the progress bar output during compilation.")
    
    # Compilation mode options
    parser.add_argument("--save-precompute0", type=Path, help="Path to save a precompute0 cache (.json.gz).")
    parser.add_argument("--from-precompute0", type=Path, help="Path to load a precompute0 cache and continue compilation from it.")
    parser.add_argument("--precompute0-only", action="store_true", help="Only generate the precompute0 cache. Requires --save-precompute0.")

    # Filtering options
    parser.add_argument("--token-len", type=str, nargs='+', help="Filter vocabulary to include tokens with specific byte lengths or ranges. E.g., '1' '3-5' '8-'.")

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.precompute0_only and not args.save_precompute0:
        parser.error("--precompute0-only requires --save-precompute0")
    if args.precompute0_only and args.output:
        parser.error("--precompute0-only cannot be used with --output")
    if not args.precompute0_only and not args.output:
        parser.error("--output is required unless --precompute0-only is specified")
    if args.from_precompute0 and not args.from_precompute0.exists():
        parser.error(f"The path specified for --from-precompute0 does not exist: {args.from_precompute0}")
    if not args.vocab_url and not args.vocab_path and not args.vocab_list:
        parser.error("At least one of --vocab-url, --vocab-path, or --vocab-list must be provided.")
    if args.token_len and not (args.vocab_url or args.vocab_path):
        parser.error("--token-len can only be used with --vocab-url or --vocab-path.")

    # 1. Get the base vocabulary from a file/URL if provided
    base_vocab = {}
    if args.vocab_url or args.vocab_path:
        base_vocab = get_vocab(args.vocab_url, args.vocab_path, args.cache_dir, args.force_download)

    # 2. Apply filters to the base vocabulary
    try:
        allowed_lengths, min_len_unbounded = parse_len_ranges(args.token_len)
    except ValueError as e:
        parser.error(f"Invalid --token-len value: {e}")

    modified_vocab = filter_vocab(base_vocab, allowed_lengths, min_len_unbounded)

    # 3. Add tokens from vocab-list (filters do not apply to these)
    if args.vocab_list:
        print(f"Adding {len(args.vocab_list)} tokens from --vocab-list.")
        max_id = -1
        if modified_vocab:
            max_id = max(modified_vocab.values())
        for token in args.vocab_list:
            if token not in modified_vocab:
                max_id += 1
                modified_vocab[token] = max_id

    # Print vocabulary if it's small (less than 1000 tokens)
    if len(modified_vocab) < 1000:
        print("\n--- Small Vocabulary (< 1000 tokens) ---\n")
        # Sort by token ID for consistent output
        sorted_vocab = sorted(modified_vocab.items(), key=lambda item: item[1])
        printable_vocab = {token: id for token, id in sorted_vocab}
        print(json.dumps(printable_vocab, indent=2, ensure_ascii=False))
        print("\n-----------------------------------------\n")

    # 4. Write the (potentially modified) vocab to a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".json", encoding='utf-8') as tmp_vocab_file:
        json.dump(modified_vocab, tmp_vocab_file)
        tmp_vocab_path = Path(tmp_vocab_file.name)

    print(f"Temporary vocabulary saved to: {tmp_vocab_path}")

    # 5. Run the Rust compiler
    try:
        run_compiler(
            args.compiler_path,
            args.grammar,
            tmp_vocab_path,
            args.output,
            recompile=not args.no_recompile,
            disable_progress_bar=args.no_progress_bar,
            save_pc0=args.save_precompute0,
            from_pc0=args.from_precompute0,
            pc0_only=args.precompute0_only,
        )
    finally:
        # 6. Clean up the temporary file
        tmp_vocab_path.unlink()
        print(f"Cleaned up temporary vocabulary file.")
if __name__ == "__main__":
    main()
