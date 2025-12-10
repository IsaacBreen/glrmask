"""
Common utilities for grammar-constrained decoding benchmarks.

This module provides:
- Output format definitions (BenchmarkResult dataclass)
- Tokenization utilities
- Vocabulary loading
- Statistics computation
"""

import json
import time
import gzip
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
import requests


@dataclass
class BenchmarkResult:
    """
    Unified benchmark result format for all systems.
    
    All times are in seconds unless otherwise noted.
    """
    # System identification
    system_name: str
    grammar_name: str
    vocabulary_name: str = "gpt2"
    
    # Grammar Compilation Time (GCT) - end-to-end compilation
    # This is the time from receiving vocab + grammar to having a ready-to-use constraint
    gct_samples_sec: List[float] = field(default_factory=list)
    gct_p50_sec: float = 0.0
    gct_p99_sec: float = 0.0
    gct_mean_sec: float = 0.0
    gct_min_sec: float = 0.0
    gct_max_sec: float = 0.0
    
    # Time Between Masks (TBM) - per-token mask computation
    # This is the time to compute get_mask() after commit()
    tbm_samples_us: List[float] = field(default_factory=list)  # microseconds
    tbm_p50_us: float = 0.0
    tbm_p99_us: float = 0.0
    tbm_mean_us: float = 0.0
    tbm_min_us: float = 0.0
    tbm_max_us: float = 0.0
    
    # Initial mask time (first mask without any commits)
    initial_mask_us: float = 0.0
    
    # Metadata
    num_tokens_processed: int = 0
    input_file: str = ""
    timestamp: str = ""
    error: Optional[str] = None
    
    def compute_statistics(self):
        """Compute p50, p99, mean, min, max from samples."""
        if self.gct_samples_sec:
            arr = np.array(self.gct_samples_sec)
            self.gct_p50_sec = float(np.percentile(arr, 50))
            self.gct_p99_sec = float(np.percentile(arr, 99))
            self.gct_mean_sec = float(np.mean(arr))
            self.gct_min_sec = float(np.min(arr))
            self.gct_max_sec = float(np.max(arr))
        
        if self.tbm_samples_us:
            arr = np.array(self.tbm_samples_us)
            self.tbm_p50_us = float(np.percentile(arr, 50))
            self.tbm_p99_us = float(np.percentile(arr, 99))
            self.tbm_mean_us = float(np.mean(arr))
            self.tbm_min_us = float(np.min(arr))
            self.tbm_max_us = float(np.max(arr))
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save_json(self, path: Path):
        """Save result to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: Path) -> 'BenchmarkResult':
        """Load result from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    def __str__(self) -> str:
        """Pretty print the results."""
        lines = [
            f"=== {self.system_name} on {self.grammar_name} ===",
            "",
            f"GCT (Grammar Compilation Time):",
            f"  p50:  {self.gct_p50_sec*1000:.1f} ms",
            f"  p99:  {self.gct_p99_sec*1000:.1f} ms",
            f"  mean: {self.gct_mean_sec*1000:.1f} ms",
            "",
            f"TBM (Time Between Masks):",
            f"  p50:  {self.tbm_p50_us:.1f} μs",
            f"  p99:  {self.tbm_p99_us:.1f} μs",
            f"  mean: {self.tbm_mean_us:.1f} μs",
            "",
            f"Initial mask: {self.initial_mask_us:.1f} μs",
            f"Tokens processed: {self.num_tokens_processed}",
        ]
        if self.error:
            lines.append(f"ERROR: {self.error}")
        return "\n".join(lines)


# --- GPT-2 Byte-level BPE Utilities ---

def bytes_to_unicode() -> Dict[int, str]:
    """
    Returns a mapping from byte values to unicode strings for GPT-2 byte-level BPE.
    See: https://github.com/openai/gpt-2/blob/master/src/encoder.py
    """
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

_BYTE_TO_UNICODE = bytes_to_unicode()
_UNICODE_TO_BYTE = {v: k for k, v in _BYTE_TO_UNICODE.items()}


def gpt2_token_str_to_bytes(token_str: str) -> bytes:
    """Convert a GPT-2 byte-level BPE token string to actual bytes."""
    return bytes([_UNICODE_TO_BYTE[c] for c in token_str])


def load_gpt2_vocab(cache_dir: Path = None, force_download: bool = False) -> Dict[str, int]:
    """
    Load the GPT-2 vocabulary from HuggingFace, with caching.
    
    Returns: Dict mapping token string -> token ID
    """
    if cache_dir is None:
        cache_dir = Path(".cache/vocab_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json"
    cache_path = cache_dir / "vocab.json"
    
    if not cache_path.exists() or force_download:
        print(f"Downloading GPT-2 vocab from: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
    
    with open(cache_path, encoding='utf-8') as f:
        return json.load(f)


def build_id_to_token_bytes(vocab: Dict[str, int]) -> Dict[int, bytes]:
    """
    Convert vocab from {token_str: id} to {id: token_bytes}.
    Uses GPT-2 byte-level BPE decoding.
    """
    id_to_token = {}
    for token_str, token_id in vocab.items():
        try:
            id_to_token[token_id] = gpt2_token_str_to_bytes(token_str)
        except KeyError:
            # Skip tokens with characters not in the GPT-2 byte mapping
            pass
    return id_to_token


def greedy_tokenize(text_bytes: bytes, id_to_token: Dict[int, bytes]) -> List[Tuple[int, int, int]]:
    """
    Greedy tokenizer: tokenize bytes using longest-match.
    
    Returns: List of (token_id, start_pos, end_pos)
    """
    # Build a Trie for fast prefix matching
    trie_root = {}
    for token_id, token_bytes in id_to_token.items():
        node = trie_root
        for byte_val in token_bytes:
            node = node.setdefault(byte_val, {})
        node['<ID>'] = token_id

    tokens_with_pos = []
    pos = 0
    while pos < len(text_bytes):
        node = trie_root
        longest_match_id = -1
        longest_match_len = 0
        
        for i in range(len(text_bytes) - pos):
            current_byte = text_bytes[pos + i]
            if current_byte in node:
                node = node[current_byte]
                if '<ID>' in node:
                    longest_match_id = node['<ID>']
                    longest_match_len = i + 1
            else:
                break
        
        if longest_match_len > 0:
            tokens_with_pos.append((longest_match_id, pos, pos + longest_match_len))
            pos += longest_match_len
        else:
            raise ValueError(f"Failed to tokenize. No token found for prefix: {text_bytes[pos:pos+20]!r}")
    
    return tokens_with_pos


class Timer:
    """Simple context manager for timing."""
    
    def __init__(self):
        self.elapsed = 0.0
    
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


def time_function(func, *args, **kwargs) -> Tuple[Any, float]:
    """
    Time a function call.
    
    Returns: (result, elapsed_seconds)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def load_grammar_file(path: Path) -> str:
    """Load a grammar file (plain text or gzipped)."""
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return f.read()
    else:
        return path.read_text(encoding='utf-8')


def read_json_schema(path: Path) -> dict:
    """Read a JSON schema file."""
    with open(path) as f:
        return json.load(f)
