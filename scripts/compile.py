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
    Loads a vocabulary from a local path or downloads it from a URL, using a cache.
    """
    if not url and not path:
        raise ValueError("Either --vocab-url or --vocab-path must be provided.")
    if url and path:
        raise ValueError("Provide either --vocab-url or --vocab-path, not both.")

    if path:
        print(f"Loading vocabulary from local path: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

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

    with open(cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def filter_vocab(vocab: dict[str, int], max_len: int | None) -> dict[str, int]:
    """
    Applies filters to the vocabulary. Currently supports filtering by max token length.
    """
    if max_len is None:
        return vocab

    print(f"Filtering vocabulary to keep tokens with byte length <= {max_len}...")
    
    filtered = {}
    for token_str, token_id in vocab.items():
        # This processing matches the logic in the Rust tests
        processed_str = token_str.replace("Ġ", " ").replace("ą", "\n").replace("Ċ", "\n")
        if len(processed_str.encode('utf-8')) <= max_len:
            filtered[token_str] = token_id
            
    print(f"  -> Filtered vocabulary from {len(vocab)} to {len(filtered)} tokens.")
    return filtered


def run_compiler(compiler_path: Path, grammar_path: Path, vocab_path: Path, output_path: Path, recompile: bool):
    """
    Runs the Rust grammar-compiler CLI tool, recompiling it first by default.
    """
    # Set environment variable to enable progress bars in the Rust code.
    # This will be passed to both cargo and the final executable.
    env = os.environ.copy()
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
        "--output", str(output_path),
    ]

    print(f"\nRunning compiler: ENABLE_PROGRESS_BAR=1 {' '.join(command)}")
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
  # Compile the JS grammar using a downloaded GPT-2 vocabulary
  python scripts/compile.py \\
    --grammar src/js.ebnf \\
    --output .cache/test_vocabs/js_constraint.json.gz \\
    --vocab-url "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json"

  # Compile using a local vocabulary and filter for short tokens
  python scripts/compile.py \\
    --grammar src/js.ebnf \\
    --output .cache/test_vocabs/js_constraint_filtered.json.gz \\
    --vocab-path .cache/test_vocabs/gpt2_vocab.json \\
    --max-token-len 10
"""
    parser = argparse.ArgumentParser(
        description="A helper script to compile a grammar constraint file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog
    )
    parser.add_argument("-g", "--grammar", type=Path, required=True, help="Path to the EBNF grammar file.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Path for the output compressed constraint file (.json.gz).")
    
    vocab_group = parser.add_mutually_exclusive_group(required=True)
    vocab_group.add_argument("--vocab-url", type=str, help="URL of the JSON vocabulary file to download.")
    vocab_group.add_argument("--vocab-path", type=Path, help="Path to a local JSON vocabulary file.")

    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/vocabs"), help="Directory to cache downloaded vocabularies.")
    parser.add_argument("--compiler-path", type=Path, default=Path("target/release/grammar-compiler"), help="Path to the grammar-compiler executable.")
    parser.add_argument("--no-recompile", action="store_true", help="Skip recompiling the Rust grammar-compiler executable and use the existing one.")
    parser.add_argument("--force-download", action="store_true", help="Force re-downloading the vocabulary even if it exists in the cache.")
    
    # Filtering options
    parser.add_argument("--max-token-len", type=int, help="Filter vocabulary to only include tokens with a byte length less than or equal to this value.")

    args = parser.parse_args()

    # 1. Get the vocabulary
    vocab = get_vocab(args.vocab_url, args.vocab_path, args.cache_dir, args.force_download)

    # 2. Apply filters
    modified_vocab = filter_vocab(vocab, args.max_token_len)

    # 3. Write the (potentially modified) vocab to a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".json", encoding='utf-8') as tmp_vocab_file:
        json.dump(modified_vocab, tmp_vocab_file)
        tmp_vocab_path = Path(tmp_vocab_file.name)

    print(f"Temporary vocabulary saved to: {tmp_vocab_path}")

    # 4. Run the Rust compiler
    try:
        run_compiler(args.compiler_path, args.grammar, tmp_vocab_path, args.output, recompile=not args.no_recompile)
    finally:
        # 5. Clean up the temporary file
        tmp_vocab_path.unlink()
        print(f"Cleaned up temporary vocabulary file.")

if __name__ == "__main__":
    main()
