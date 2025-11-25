"""Base interface for grammar-constrained generation systems.

This module defines the common interface that all system wrappers must implement
to enable fair benchmarking and comparison.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple, Optional
import time


@dataclass
class CompilationResult:
    """Result of grammar compilation."""
    compiled: Any  # System-specific compiled constraint
    compilation_time_sec: float
    metadata: dict  # System-specific metadata (e.g., state count, file size)


@dataclass
class MaskResult:
    """Result of get_mask operation."""
    valid_token_ids: List[int]  # List of valid token IDs
    time_sec: float
    metadata: dict = None  # Optional system-specific metadata


@dataclass
class CommitResult:
    """Result of commit operation."""
    new_state: Any  # System-specific state
    time_sec: float
    metadata: dict = None


class BaseSystem(ABC):
    """Base interface for grammar-constrained generation systems.
    
    All system wrappers must inherit from this class and implement all abstract methods.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the system name (e.g., 'sep1', 'outlines', 'xgrammar')."""
        pass
    
    @abstractmethod
    def compile_grammar(
        self,
        grammar_path: Path,
        vocab: dict[int, bytes],
        **kwargs
    ) -> CompilationResult:
        """Compile a grammar file into a system-specific constraint.
        
        Args:
            grammar_path: Path to the grammar file
            vocab: Vocabulary mapping token_id -> token_bytes
            **kwargs: System-specific compilation options
            
        Returns:
            CompilationResult with compiled constraint and timing
        """
        pass
    
    @abstractmethod
    def create_state(self, compiled: Any) -> Any:
        """Create initial state from compiled constraint.
        
        Args:
            compiled: The compiled constraint from compile_grammar
            
        Returns:
            Initial state (system-specific type)
        """
        pass
    
    @abstractmethod
    def get_mask(self, state: Any) -> MaskResult:
        """Get valid token mask for current state.
        
        Args:
            state: Current state
            
        Returns:
            MaskResult with list of valid token IDs and timing
        """
        pass
    
    @abstractmethod
    def commit(self, state: Any, token_id: int) -> CommitResult:
        """Advance state by committing a token.
        
        Args:
            state: Current state
            token_id: Token ID to commit
            
        Returns:
            CommitResult with new state and timing
        """
        pass
    
    @abstractmethod
    def supports_grammar_format(self, format: str) -> bool:
        """Check if system supports a grammar format.
        
        Args:
            format: Grammar format (e.g., 'ebnf', 'json_schema', 'regex')
            
        Returns:
            True if format is supported
        """
        pass
    
    def validate_output(
        self,
        output_tokens: List[int],
        vocab: dict[int, bytes],
        grammar_path: Path
    ) -> Tuple[bool, Optional[str]]:
        """Validate that output tokens parse correctly according to grammar.
        
        Default implementation returns (True, None). Override if system provides
        validation capabilities.
        
        Args:
            output_tokens: List of token IDs
            vocab: Vocabulary mapping
            grammar_path: Path to grammar for validation
            
        Returns:
            (is_valid, error_message)
        """
        return (True, None)
    
    def get_memory_usage(self, compiled: Any) -> int:
        """Get approximate memory usage of compiled constraint in bytes.
        
        Default implementation returns 0. Override if system can measure memory.
        
        Args:
            compiled: Compiled constraint
            
        Returns:
            Memory usage in bytes
        """
        return 0
    
    def cleanup(self):
        """Clean up any system resources.
        
        Called after benchmarking is complete. Override if cleanup is needed.
        """
        pass


def time_function(func, *args, **kwargs) -> Tuple[Any, float]:
    """Time a function call.
    
    Args:
        func: Function to call
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        (result, time_sec)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed
