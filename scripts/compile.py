import argparse
import json
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


def run_compiler(compiler_path: Path, grammar_path: Path, vocab_path: Path, output_path: Path):
    """
    Runs the Rust grammar-compiler CLI tool.
    """
    if not compiler_path.exists():
        print(f"Compiler executable not found at '{compiler_path}'.")
        print("Attempting to build it with 'cargo build --release'...")
        try:
            subprocess.run(
                ["cargo", "build", "--release"],
                check=True,
                capture_output=True,
                text=True
            )
            print("Build successful.")
        except subprocess.CalledProcessError as e:
            print("Cargo build failed:", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            sys.exit(1)

    command = [
        str(compiler_path),
        "--grammar", str(grammar_path),
        "--vocab", str(vocab_path),
        "--output", str(output_path),
    ]

    print(f"\nRunning compiler: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Grammar compilation failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Could not find the compiler executable at '{compiler_path}'", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="A helper script to compile a grammar constraint file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-g", "--grammar", type=Path, required=True, help="Path to the EBNF grammar file.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Path for the output compressed constraint file (.json.gz).")
    
    vocab_group = parser.add_mutually_exclusive_group(required=True)
    vocab_group.add_argument("--vocab-url", type=str, help="URL of the JSON vocabulary file to download.")
    vocab_group.add_argument("--vocab-path", type=Path, help="Path to a local JSON vocabulary file.")

    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/vocabs"), help="Directory to cache downloaded vocabularies.")
    parser.add_argument("--compiler-path", type=Path, default=Path("target/release/grammar-compiler"), help="Path to the compiled grammar-compiler executable.")
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
        run_compiler(args.compiler_path, args.grammar, tmp_vocab_path, args.output)
    finally:
        # 5. Clean up the temporary file
        tmp_vocab_path.unlink()
        print(f"Cleaned up temporary vocabulary file.")

if __name__ == "__main__":
    main()
