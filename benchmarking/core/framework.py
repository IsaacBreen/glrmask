import abc
import time
import dataclasses
from typing import List, Tuple, Dict, Any, Optional
import statistics
import json
from pathlib import Path

@dataclasses.dataclass
class BenchmarkConfig:
    library_name: str
    grammar_path: str
    input_file_path: str
    warmup_runs: int = 5
    measurement_runs: int = 10
    output_dir: str = "benchmark_results"

@dataclasses.dataclass
class BenchmarkResult:
    library_name: str
    grammar_name: str
    input_name: str
    total_tokens: int
    load_time_sec: float
    token_timings_sec: List[float]
    masks: List[List[Tuple[int, int]]]  # List of ranges for each token step
    error: Optional[str] = None

class AbstractBenchmarkAdapter(abc.ABC):
    """
    Abstract base class for benchmarking adapters.
    Each library (Sep1, LLGuidance, XGrammar, Outlines) must implement this.
    """

    @abc.abstractmethod
    def load_grammar(self, grammar_path: str) -> float:
        """
        Loads the grammar and returns the time taken in seconds.
        """
        pass

    @abc.abstractmethod
    def tokenize(self, input_bytes: bytes) -> List[int]:
        """
        Tokenizes the input bytes into a list of token IDs.
        This should use the same tokenizer as the model (e.g. GPT-2).
        """
        pass

    @abc.abstractmethod
    def get_mask(self) -> Tuple[List[Tuple[int, int]], float]:
        """
        Computes the valid token mask for the current state.
        Returns:
            - A list of [start, end] ranges (inclusive) representing valid tokens.
            - The time taken to compute the mask in seconds.
        """
        pass

    @abc.abstractmethod
    def commit(self, token_id: int) -> float:
        """
        Updates the state with the chosen token.
        Returns the time taken in seconds.
        """
        pass
    
    @abc.abstractmethod
    def reset(self):
        """
        Resets the state to the initial state.
        """
        pass

class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, adapter: AbstractBenchmarkAdapter):
        self.config = config
        self.adapter = adapter

    def run(self) -> BenchmarkResult:
        print(f"Running benchmark for {self.config.library_name} on {self.config.grammar_path}...")
        
        try:
            # Load Grammar
            print("  Loading grammar...")
            load_start = time.perf_counter()
            self.adapter.load_grammar(self.config.grammar_path)
            load_time = time.perf_counter() - load_start
            print(f"  Loaded in {load_time:.4f}s")

            # Load Input
            input_path = Path(self.config.input_file_path)
            input_bytes = input_path.read_bytes()
            token_ids = self.adapter.tokenize(input_bytes)
            print(f"  Input has {len(token_ids)} tokens")

            # Warmup
            print(f"  Warming up ({self.config.warmup_runs} runs)...")
            for _ in range(self.config.warmup_runs):
                self.adapter.reset()
                for token_id in token_ids:
                    self.adapter.get_mask()
                    self.adapter.commit(token_id)

            # Measurement
            print(f"  Measuring ({self.config.measurement_runs} runs)...")
            all_token_timings = []
            final_masks = []

            for run_idx in range(self.config.measurement_runs):
                self.adapter.reset()
                run_timings = []
                run_masks = []
                
                for token_id in token_ids:
                    # Measure get_mask
                    mask_ranges, mask_time = self.adapter.get_mask()
                    
                    # Measure commit
                    commit_start = time.perf_counter()
                    self.adapter.commit(token_id)
                    commit_time = time.perf_counter() - commit_start
                    
                    # We primarily care about the sum of get_mask + commit for latency
                    # But keeping them separate is useful for analysis.
                    # For the result, let's store the total constraint overhead per token.
                    run_timings.append(mask_time + commit_time)
                    
                    if run_idx == 0:
                        # Only store masks for the first run to save space/time
                        run_masks.append(mask_ranges)

                all_token_timings.append(run_timings)
                if run_idx == 0:
                    final_masks = run_masks

            # Aggregate timings (average per token position across runs)
            avg_token_timings = []
            num_tokens = len(token_ids)
            for i in range(num_tokens):
                timings_at_i = [run[i] for run in all_token_timings]
                avg_token_timings.append(statistics.mean(timings_at_i))

            return BenchmarkResult(
                library_name=self.config.library_name,
                grammar_name=Path(self.config.grammar_path).name,
                input_name=input_path.name,
                total_tokens=num_tokens,
                load_time_sec=load_time,
                token_timings_sec=avg_token_timings,
                masks=final_masks
            )

        except Exception as e:
            print(f"  Error during benchmark: {e}")
            import traceback
            traceback.print_exc()
            return BenchmarkResult(
                library_name=self.config.library_name,
                grammar_name=Path(self.config.grammar_path).name,
                input_name=Path(self.config.input_file_path).name,
                total_tokens=0,
                load_time_sec=0.0,
                token_timings_sec=[],
                masks=[],
                error=str(e)
            )

    def save_result(self, result: BenchmarkResult):
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{result.library_name}_{result.grammar_name}_{result.input_name}.json"
        output_path = output_dir / filename
        
        data = dataclasses.asdict(result)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Saved results to {output_path}")
