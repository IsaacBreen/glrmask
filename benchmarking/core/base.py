"""
Core base interface for all grammar constraint systems in the benchmarking framework.

This module defines the abstract interface that all systems must implement for fair comparison.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import time


class MaskRepresentation:
    """Compact range-based representation of allowed token masks."""
    
    def __init__(self, ranges: List[Tuple[int, int]]):
        """
        Initialize mask from sorted, non-overlapping ranges.
        
        Args:
            ranges: List of (start, end) inclusive ranges of allowed token IDs
        """
        self.ranges = ranges
    
    @classmethod
    def from_token_ids(cls, token_ids: List[int]) -> 'MaskRepresentation':
        """Create mask from list of individual token IDs."""
        if not token_ids:
            return cls([])
        
        token_ids = sorted(set(token_ids))
        ranges = []
        start = token_ids[0]
        end = token_ids[0]
        
        for tid in token_ids[1:]:
            if tid == end + 1:
                end = tid
            else:
                ranges.append((start, end))
                start = end = tid
        ranges.append((start, end))
        
        return cls(ranges)
    
    def to_token_ids(self) -> List[int]:
        """Expand ranges to full list of token IDs."""
        result = []
        for start, end in self.ranges:
            result.extend(range(start, end + 1))
        return result
    
    def __eq__(self, other: 'MaskRepresentation') -> bool:
        """Check if two masks are equivalent."""
        return self.ranges == other.ranges
    
    def count(self) -> int:
        """Count total number of allowed tokens."""
        return sum(end - start + 1 for start, end in self.ranges)
    
    def to_json(self) -> List[List[int]]:
        """Serialize to JSON-compatible format."""
        return [[s, e] for s, e in self.ranges]
    
    @classmethod
    def from_json(cls, data: List[List[int]]) -> 'MaskRepresentation':
        """Deserialize from JSON-compatible format."""
        return cls([(s, e) for s, e in data])


class TimingResult:
    """Container for timing measurements."""
    
    def __init__(self, operation: str, duration_ns: int, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize timing result.
        
        Args:
            operation: Name of operation (e.g., 'get_mask', 'commit')
            duration_ns: Duration in nanoseconds
            metadata: Optional additional information
        """
        self.operation = operation
        self.duration_ns = duration_ns
        self.metadata = metadata or {}


class ConstraintSystem(ABC):
    """
    Abstract base class for grammar constraint systems.
    
    All systems must implement this interface for fair benchmarking.
    """
    
    @classmethod
    @abstractmethod
    def load(cls, grammar_path: Path, vocab: Dict[str, int], **kwargs) -> 'ConstraintSystem':
        """
        Load and compile a grammar constraint.
        
        Args:
            grammar_path: Path to grammar file (format depends on system)
            vocab: Vocabulary mapping tokens (str) to IDs (int)
            **kwargs: System-specific options
        
        Returns:
            Initialized constraint system instance
        """
        pass
    
    @abstractmethod
    def get_mask(self) -> MaskRepresentation:
        """
        Get the current valid token mask.
        
        Returns:
            Mask representation indicating which tokens are currently valid
        """
        pass
    
    @abstractmethod
    def commit(self, token_id: int) -> None:
        """
        Commit a token to the current state.
        
        Args:
            token_id: ID of token to commit
        
        Raises:
            ValueError: If token_id is not currently valid
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the constraint to its initial state."""
        pass
    
    @abstractmethod
    def is_valid(self) -> bool:
        """
        Check if current state is valid for continued generation.
        
        Returns:
            True if generation can continue, False if in invalid/terminal state
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get system-specific metadata for reporting.
        
        Returns:
            Dictionary of metadata (version, config, etc.)
        """
        return {
            "system_name": self.__class__.__name__,
            "system_version": "unknown"
        }


class TimedConstraintSystem(ConstraintSystem):
    """
    Mixin that adds automatic timing instrumentation.
    
    Subclasses should implement _get_mask_impl and _commit_impl instead
    of get_mask and commit directly.
    """
    
    def __init__(self):
        self._last_get_mask_time_ns: Optional[int] = None
        self._last_commit_time_ns: Optional[int] = None
    
    @abstractmethod
    def _get_mask_impl(self) -> MaskRepresentation:
        """Implementation of get_mask to be timed."""
        pass
    
    @abstractmethod
    def _commit_impl(self, token_id: int) -> None:
        """Implementation of commit to be timed."""
        pass
    
    def get_mask(self) -> MaskRepresentation:
        """Get mask with automatic timing."""
        start = time.perf_counter_ns()
        result = self._get_mask_impl()
        end = time.perf_counter_ns()
        self._last_get_mask_time_ns = end - start
        return result
    
    def commit(self, token_id: int) -> None:
        """Commit token with automatic timing."""
        start = time.perf_counter_ns()
        self._commit_impl(token_id)
        end = time.perf_counter_ns()
        self._last_commit_time_ns = end - start
    
    def get_last_timing(self) -> Dict[str, int]:
        """Get timing of last get_mask and commit calls in nanoseconds."""
        return {
            "get_mask_ns": self._last_get_mask_time_ns,
            "commit_ns": self._last_commit_time_ns
        }
