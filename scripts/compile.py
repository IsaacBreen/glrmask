#!/usr/bin/env python3
"""
Grammar Constraint Compiler

Compiles a context-free grammar and LLM vocabulary into a precomputed constraint
file for fast grammar-constrained decoding.

Usage:
    python scripts/compile.py --grammar src/js.ebnf --output constraint.json.gz \\
        --vocab-url "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json"
    
    python scripts/compile.py --grammar grammar.lark --format lark \\
        --output constraint.json.gz --vocab vocab.json

Outputs a gzip-compressed JSON file containing the deterministic weighted
automaton and vocabulary mappings needed for fast mask generation.

See README.md for more details.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import requests
from typing import Optional, List
import time

_start_time = time.time()


# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    
    @staticmethod
    def success(msg: str) -> str:
        return f"{Colors.GREEN}✓{Colors.RESET} {msg}"
    
    @staticmethod
    def error(msg: str) -> str:
        return f"{Colors.RED}✗{Colors.RESET} {msg}"
    
    @staticmethod
    def warn(msg: str) -> str:
        return f"{Colors.YELLOW}⚠{Colors.RESET} {msg}"
    
    @staticmethod
    def info(msg: str) -> str:
        return f"{Colors.CYAN}•{Colors.RESET} {msg}"
    
    @staticmethod
    def dim(msg: str) -> str:
        return f"{Colors.DIM}{msg}{Colors.RESET}"

# Global verbosity level
_verbose = False
_timings = []

def log_timing(desc: str):
    global _start_time
    now = time.time()
    elapsed = (now - _start_time) * 1000
    _timings.append((desc, elapsed))
    _start_time = now

def print_timings():
    if not _timings: return
    print(f"\n{Colors.BOLD}Python Wrapper Timings:{Colors.RESET}")
    for desc, ms in _timings:
        print(f"  {desc:<30} : {ms:.2f}ms")


def log(msg: str, force: bool = False):
    """Print a message if verbose mode is enabled or force is True."""
    if _verbose or force:
        print(msg)

def log_error(msg: str):
    """Print an error message (always shown)."""
    print(Colors.error(msg), file=sys.stderr)

# --- Helper Functions ---
def resolve_vocab_path(url: Optional[str], path: Optional[Path], vocab_list: Optional[List[str]], cache_dir: Path, force_download: bool) -> Path:
    """
    Resolves the vocabulary file path.
    - If URL: downloads to cache and returns cache path.
    - If Path: returns the path.
    - If List: writes to a temp file and returns temp path (caller must cleanup, but we use a known location or just leak for now as it's rare).
    """
    if vocab_list:
        # Create a temp file for list-based vocab
        # We effectively create a simple dict mapping
        vocab_dict = {token: i for i, token in enumerate(vocab_list)}
        
        # We use a temp file that persists until the script ends (or we'd need to manage lifecycle)
        # Using NamedTemporaryFile with delete=False is easiest, we can print a warning or try to cleanup in main
        fd, tmp_path = tempfile.mkstemp(suffix=".json", text=True)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(vocab_dict, f)
        log(Colors.info(f"Created temporary vocabulary file from list: {tmp_path}"))
        return Path(tmp_path)

    if path:
        return path

    if url:
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_name = url.split("/")[-1]
        cache_path = cache_dir / file_name

        if not cache_path.exists() or force_download:
            log(Colors.info(f"Downloading vocabulary from URL..."))
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                log(Colors.dim(f"  Cached to: {cache_path}"))
            except requests.RequestException as e:
                log_error(f"Failed to download vocabulary: {e}")
                sys.exit(1)
        else:
            log(Colors.info(f"Using cached vocabulary: {cache_path}"))
        
        return cache_path

    raise ValueError("No vocabulary source provided")


def run_compiler(compiler_path: Path, grammar_path: Path, vocab_path: Path, output_path: Optional[Path], recompile: bool, disable_progress_bar: bool, token_lens: Optional[List[str]], build_profile: str = "release", save_pc0: Optional[Path] = None, from_pc0: Optional[Path] = None, pc0_only: bool = False, format: Optional[str] = None, skip_if_up_to_date: bool = False):
    """
    Runs the Rust grammar-compiler CLI tool, recompiling it first by default.
    """
    # Set environment variable to enable progress bars in the Rust code.
    # This will be passed to both cargo and the final executable.
    env = os.environ.copy()
    if not disable_progress_bar:
        env["ENABLE_PROGRESS_BAR"] = "1"

    if skip_if_up_to_date and output_path and output_path.exists():
        inputs = [grammar_path, vocab_path]
        if from_pc0:
            inputs.append(from_pc0)
        output_mtime = output_path.stat().st_mtime
        if all(path.exists() and path.stat().st_mtime <= output_mtime for path in inputs):
            log(Colors.success(f"Output up-to-date; skipping compile: {output_path}"), force=True)
            return

    if recompile:
        build_cmd = ["cargo", "build"]
        if build_profile == "release":
            build_cmd.append("--release")
            log(Colors.info("Building compiler (release)..."))
        elif build_profile == "debug":
            log(Colors.info("Building compiler (debug)..."))
        else:
            build_cmd.extend(["--profile", build_profile])
            log(Colors.info(f"Building compiler (profile: {build_profile})..."))
        try:
            # Always stream cargo output to let user see compilation progress
            result = subprocess.run(
                build_cmd,
                check=True,
                env=env,
            )
            log(Colors.success("Build complete"))
        except subprocess.CalledProcessError:
            log_error("Cargo build failed")
            sys.exit(1)

    if not compiler_path.exists():
        log_error(f"Compiler executable not found at '{compiler_path}'")
        if recompile:
             log_error("Build completed but executable is not in expected location.")
        else:
             log_error("Try running without '--no-recompile' to build it.")
        sys.exit(1)

    command = [
        str(compiler_path),
        "--grammar", str(grammar_path),
        "--vocab", str(vocab_path),
    ]

    if format:
        command.extend(["--format", format])

    if token_lens:
        command.append("--token-len")
        command.extend(token_lens)

    if pc0_only:
        command.extend(["--save-precompute0", str(save_pc0)])
        command.append("--precompute0-only")
    else:
        if output_path:
            command.extend(["--output", str(output_path)])
        if from_pc0:
            log(Colors.info(f"Loading precompute0 cache: {from_pc0}"))
            command.extend(["--load-precompute0", str(from_pc0)])
        if save_pc0:
            # Ensure parent directory exists
            save_pc0.parent.mkdir(parents=True, exist_ok=True)
            log(Colors.dim(f"  Will save precompute0 cache to: {save_pc0}"))
            command.extend(["--save-precompute0", str(save_pc0)])

    log(Colors.dim(f"  Command: {' '.join(command)}"))
    
    try:
        # Run the compiler, passing through its output and the environment variable.
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as e:
        log_error(f"Compilation failed with exit code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        log_error(f"Could not find compiler executable at '{compiler_path}'")
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
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output from the Python wrapper.")

    vocab_source_group = parser.add_mutually_exclusive_group()
    vocab_source_group.add_argument("--vocab-url", type=str, help="URL of the JSON vocabulary file to download.")
    vocab_source_group.add_argument("--vocab-path", type=Path, help="Path to a local JSON vocabulary file.")
    parser.add_argument("--vocab-list", type=str, nargs='+', help="A list of strings to use as the vocabulary. Can be combined with --vocab-url/--vocab-path.")

    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/vocabs"), help="Directory to cache downloaded vocabularies.")
    parser.add_argument("--compiler-path", type=Path, help="Path to the grammar-compiler executable. Defaults to target/{profile}/grammar-compiler based on --build-profile.")
    parser.add_argument("--build-profile", type=str, default="release", help="Cargo build profile to use (e.g., 'release', 'debug'). Default: release.")
    parser.add_argument("--no-recompile", action="store_true", help="Skip recompiling the Rust grammar-compiler executable and use the existing one.")
    parser.add_argument("--skip-if-up-to-date", action="store_true", help="Skip compilation if output is newer than inputs.")
    parser.add_argument("--force-download", action="store_true", help="Force re-downloading the vocabulary even if it exists in the cache.")
    parser.add_argument("--no-progress-bar", action="store_true", help="Disable the progress bar output during compilation.")
    
    # Compilation mode options
    parser.add_argument("--save-precompute0", type=Path, help="Path to save a precompute0 cache (.json.gz).")
    parser.add_argument("--from-precompute0", type=Path, help="Path to load a precompute0 cache and continue compilation from it.")
    parser.add_argument("--precompute0-only", action="store_true", help="Only generate the precompute0 cache. Requires --save-precompute0.")

    # Filtering options
    parser.add_argument("--token-len", type=str, nargs='+', help="Filter vocabulary to include tokens with specific byte lengths or ranges. E.g., '1' '3-5' '8-'.")
    
    parser.add_argument("--format", type=str, choices=["ebnf", "lark"], help="Grammar format (ebnf or lark). If not specified, inferred from file extension.")

    args = parser.parse_args()
    
    # Set global verbose flag
    global _verbose
    _verbose = args.verbose

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
    if args.token_len and not (args.vocab_url or args.vocab_path or args.vocab_list):
        parser.error("--token-len can only be used with --vocab-url, --vocab-path or --vocab-list.")

    # Determine compiler path based on build profile if not explicitly provided
    if args.compiler_path is None:
        if args.build_profile == "debug":
            args.compiler_path = Path("target/debug/grammar-compiler")
        elif args.build_profile == "release":
            args.compiler_path = Path("target/release/grammar-compiler")
        else:
            args.compiler_path = Path(f"target/{args.build_profile}/grammar-compiler")

    # 1. Resolve vocabulary path
    # This might download the file if needed, but won't load it into memory if not needed.
    vocab_path = resolve_vocab_path(args.vocab_url, args.vocab_path, args.vocab_list, args.cache_dir, args.force_download)
    log_timing("Resolve Vocabulary Path")

    # 2. Run the Rust compiler
    # Filtering is now handled by the Rust compiler directly, so we just pass the args.
    try:
        run_compiler(
            args.compiler_path,
            args.grammar,
            vocab_path,
            args.output,
            recompile=not args.no_recompile,
            disable_progress_bar=args.no_progress_bar,
            token_lens=args.token_len,
            build_profile=args.build_profile,
            save_pc0=args.save_precompute0,
            from_pc0=args.from_precompute0,
            pc0_only=args.precompute0_only,
            format=args.format,
            skip_if_up_to_date=args.skip_if_up_to_date,
        )
        log_timing("Run Rust Compiler")
    finally:
        # Clean up if we created a temporary file for vocab-list
        if args.vocab_list and vocab_path.exists():
            try:
                # Basic check to avoid deleting user files: if it's in temp
                if str(vocab_path).startswith(tempfile.gettempdir()):
                    vocab_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        main()
    finally:
        print_timings()
