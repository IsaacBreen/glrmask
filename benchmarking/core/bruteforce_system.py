"""Bruteforce baseline system for correctness validation."""

import sys
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import _sep1 as ffi
except ImportError:
    raise ImportError("Sep1 Rust bindings required for bruteforce baseline")

from benchmarking.core.base import ConstraintSystem, MaskRepresentation
from benchmarking.core.sep1_system import Sep1System


class BruteforceSystem(ConstraintSystem):
    """
    Bruteforce baseline that checks each token individually.
    
    This is used as the ground truth for correctness validation.
    Slower than optimized systems but guaranteed to be correct.
    """
    
    def __init__(self, constraint: ffi.GrammarConstraint, vocab_size: int, show_progress: bool = False):
        """
        Initialize bruteforce system.
        
        Args:
            constraint: Compiled GrammarConstraint
            vocab_size: Size of vocabulary
            show_progress: Whether to show progress bar during get_mask
        """
        self.constraint = constraint
        self.vocab_size = vocab_size
        self.state = ffi.GrammarConstraintState(constraint)
        self.show_progress = show_progress
    
    @classmethod
    def load(cls, grammar_path: Path, vocab: Dict[str, int], **kwargs) -> 'BruteforceSystem':
        """
        Load bruteforce system.
        
        Uses same compilation as Sep1, just different runtime algorithm.
        
        Args:
            grammar_path: Path to grammar or precompiled constraint
            vocab: Vocabulary mapping
            **kwargs: Same as Sep1System.load, plus:
                - show_progress: Show progress during get_mask (default: False)
        
        Returns:
            Initialized BruteforceSystem
        """
        # Reuse Sep1's compilation logic
        sep1 = Sep1System.load(grammar_path, vocab, **kwargs)
        show_progress = kwargs.get('show_progress', False)
        return cls(sep1.constraint, sep1.vocab_size, show_progress)
    
    def get_mask(self) -> MaskRepresentation:
        """
        Get mask by checking each token individually (bruteforce).
        
        This is slow but guaranteed correct - it's the ground truth.
        """
        allowed_tokens = []
        
        iterator = range(self.vocab_size)
        if self.show_progress:
            iterator = tqdm(iterator, desc="bruteforce get_mask", leave=False)
        
        for token_id in iterator:
            # Clone state and try committing this token
            temp_state = self.state.clone()
            temp_state.commit(token_id)
            
            # Check if resulting state is valid
            if temp_state.is_valid():
                allowed_tokens.append(token_id)
        
        return MaskRepresentation.from_token_ids(allowed_tokens)
    
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
        """Get bruteforce metadata."""
        return {
            "system_name": "bruteforce",
            "system_version": "1.0",
            "vocab_size": self.vocab_size,
            "implementation": "rust",
            "baseline": True,
            "warning": "Extremely slow - for correctness validation only"
        }
