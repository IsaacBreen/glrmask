"""Sep1 constraint system adapter for benchmarking."""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
import gzip
import subprocess
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import _sep1 as ffi
except ImportError:
    raise ImportError(
        "Sep1 Rust bindings not found. Make sure the project is built with "
        "`cargo build --release` and Python bindings are available."
    )

from benchmarking.core.base import ConstraintSystem, MaskRepresentation


class Sep1System(ConstraintSystem):
    """
    Sep1 grammar constraint system.
    
    This is our system - uses precompiled DWA with deterministic weighted automata.
    """
    
    def __init__(self, constraint: ffi.GrammarConstraint, vocab_size: int):
        """
        Initialize from precompiled constraint.
        
        Args:
            constraint: Compiled GrammarConstraint object
            vocab_size: Size of vocabulary
        """
        self.constraint = constraint
        self.vocab_size = vocab_size
        self.state = ffi.GrammarConstraintState(constraint)
        self._version = self._get_version()
    
    @classmethod
    def load(cls, grammar_path: Path, vocab: Dict[str, int], **kwargs) -> 'Sep1System':
        """
        Load Sep1 constraint.
        
        Args:
            grammar_path: Path to .ebnf grammar file OR precompiled .json/.json.gz constraint
            vocab: Vocabulary mapping
            **kwargs: Optional:
                - precompiled: If True, grammar_path is precompiled constraint (default: auto-detect)
                - compiler_path: Path to grammar-compiler binary (default: auto-detect)
        
        Returns:
            Initialized Sep1System
        """
        is_precompiled = kwargs.get('precompiled', str(grammar_path).endswith(('.json', '.json.gz')))
        
        if is_precompiled:
            # Load precompiled constraint
            constraint_json_str = cls._load_constraint_json(grammar_path)
        else:
            # Compile grammar first
            compiler_path = kwargs.get('compiler_path') or cls._find_compiler()
            constraint_json_str = cls._compile_grammar(grammar_path, vocab, compiler_path)
        
        constraint = ffi.GrammarConstraint.from_json_string(constraint_json_str)
        return cls(constraint, len(vocab))
    
    @staticmethod
    def _load_constraint_json(path: Path) -> str:
        """Load constraint JSON from file (handles .gz)."""
        if str(path).endswith('.gz'):
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                return f.read()
        else:
            return path.read_text(encoding='utf-8')
    
    @staticmethod
    def _find_compiler() -> Path:
        """Find grammar-compiler binary."""
        # Try release build first
        release_path = PROJECT_ROOT / "target" / "release" / "grammar-compiler"
        if release_path.exists():
            return release_path
        
        # Try debug build
        debug_path = PROJECT_ROOT / "target" / "debug" / "grammar-compiler"
        if debug_path.exists():
            return debug_path
        
        raise FileNotFoundError(
            "grammar-compiler not found. Build with: cargo build --release"
        )
    
    @staticmethod
    def _compile_grammar(grammar_path: Path, vocab: Dict[str, int], compiler_path: Path) -> str:
        """Compile EBNF grammar to constraint JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as vocab_file:
            json.dump(vocab, vocab_file)
            vocab_file_path = Path(vocab_file.name)
        
        with tempfile.NamedTemporaryFile(mode='r', suffix='.json', delete=False) as output_file:
            output_path = Path(output_file.name)
        
        try:
            cmd = [
                str(compiler_path),
                "--grammar", str(grammar_path),
                "--vocab", str(vocab_file_path),
                "--output", str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Grammar compilation failed: {result.stderr}")
            
            return output_path.read_text()
        
        finally:
            vocab_file_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
    
    @staticmethod
    def _get_version() -> str:
        """Get Sep1 version from Cargo.toml."""
        cargo_toml = PROJECT_ROOT / "Cargo.toml"
        if cargo_toml.exists():
            import tomli
            try:
                with open(cargo_toml, 'rb') as f:
                    data = tomli.load(f)
                    return data.get('package', {}).get('version', 'unknown')
            except:
                pass
        return 'unknown'
    
    def get_mask(self) -> MaskRepresentation:
        """Get current valid token mask."""
        # Use get_mask_bv for efficiency
        bv = self.state.get_mask_bv()
        ranges = bv.to_ranges()
        return MaskRepresentation(ranges)
    
    def commit(self, token_id: int) -> None:
        """Commit a token."""
        if not (0 <= token_id < self.vocab_size):
            raise ValueError(f"Token ID {token_id} out of range [0, {self.vocab_size})")
        
        self.state.commit(token_id)
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.state = ffi.GrammarConstraintState(self.constraint)
    
    def is_valid(self) -> bool:
        """Check if current state is valid."""
        return self.state.is_valid()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get Sep1-specific metadata."""
        return {
            "system_name": "sep1",
            "system_version": self._version,
            "vocab_size": self.vocab_size,
            "implementation": "rust",
            "features": ["DWA", "GLR", "precomputation"]
        }
