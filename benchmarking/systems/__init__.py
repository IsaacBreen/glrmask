"""System wrappers for benchmarking framework."""

from benchmarking.systems.base import BaseSystem, CompilationResult, MaskResult, CommitResult
from benchmarking.systems.sep1 import Sep1System

# Import other systems as they're implemented
# from benchmarking.systems.outlines import OutlinesSystem
# from benchmarking.systems.xgrammar import XGrammarSystem
# from benchmarking.systems.llguidance import LLGuidanceSystem

__all__ = [
    'BaseSystem',
    'CompilationResult',
    'MaskResult',
    'CommitResult',
    'Sep1System',
]
