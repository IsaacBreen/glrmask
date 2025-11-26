# Benchmarking Framework Design

## Design Principles
1. **Separation of Concerns**: Data gathering separate from analysis
2. **Minimal Overhead**: Direct C++/Rust bindings where possible, avoid Python overhead for timing
3. **Fair Comparison**: All systems run on same machine, same inputs
4. **Comprehensive**: Both performance AND correctness validation

## Architecture

### Data Gathering Layer
```
benchmarking/
├── core/
│   ├── base.py              # Abstract interface for all systems
│   ├── mask_storage.py      # Range-based mask storage
│   └── timing.py            # High-precision timing utilities
├── systems/
│   ├── sep1_system.py       # Our system (minimal wrapper)
│   ├── bruteforce_system.py # Baseline for correctness
│   ├── llguidance_system.py # LLGuidance wrapper
│   └── ...
├── grammars/
│   ├── json.ebnf
│   ├── javascript.ebnf
│   └── sql.ebnf  
├── runners/
│   ├── collect_timings.py   # Gather per-token timings
│   └── collect_masks.py     # Gather per-token masks
└── data/                     # Output directory
    ├── timings/
    │   ├── sep1_json_run1.jsonl
    │   └── ...
    └── masks/
        ├── sep1_json_run1.jsonl
        └── ...
```

### Analysis Layer
```
analysis/
├── process_timings.py       # Load and analyze timing data
├── validate_correctness.py  # Compare masks against baseline
├── generate_tables.py       # Create paper tables
└── generate_plots.py        # Create visualizations
```

## Data Format

### Timing Data (JSONL format)
```json
{"system": "sep1", "grammar": "json", "token_id": 123, "tbm_ns": 17000, "commit_ns": 500, "step": 0}
{"system": "sep1", "grammar": "json", "token_id": 456, "tbm_ns": 14000, "commit_ns": 450, "step": 1}
...
```

### Mask Data (JSONL format with range-based storage)
```json
{"system": "sep1", "grammar": "json", "step": 0, "mask": [[0, 100], [200, 300]]}
{"system": "sep1", "grammar": "json", "step": 1, "mask": [[50, 150]]}
...
```

### Metadata
```json
{
  "run_id": "uuid",
  "timestamp": "2025-11-26T14:00:00Z",
  "system": "sep1",
  "grammar": "json",
  "input_file": "test_input.js",
  "vocab_size": 50257,
  "hardware": {
    "cpu": "M1 Max",
    "ram": "32GB"
  },
  "versions": {
    "system_version": "0.1.0"
  }
}
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. `base.py` - Abstract base class with `get_mask()`, `commit()`, `load()`
2. `mask_storage.py` - Range-based compact mask representation
3. `timing.py` - Nanosecond precision timing with minimal overhead

### Phase 2: System Adapters
1. `bruteforce_system.py` - Wrap existing bruteforce_rust_model
2. `sep1_system.py` - Direct Rust FFI with timing instrumentation
3. `llguidance_system.py` - Wrap llguidance with minimal overhead

### Phase 3: Data Collection
1. `collect_timings.py` - Run systems, record per-token timings
2. `collect_masks.py` - Run systems, record per-token masks (separate runs)
3. Generate JSONL output files

### Phase 4: Analysis
1. Load all timing data
2. Compute statistics (median, p75, p99)
3. Load all mask data
4. Compare against bruteforce baseline
5. Generate tables and plots

## Key Design Decisions

### Why JSONL?
- Each line is independent (can stream, can parallelize)
- Easy to append
- Human-readable for debugging
- Efficient to process with tools like `jq`

### Why Separate Timing and Mask Collection?
- Mask storage adds overhead (even with ranges)
- Timing needs to be pure - no extra allocations
- Can run timing-only benchmarks quickly
- Can run mask collection only when needed for validation

### Why Range-Based Mask Storage?
- Typical mask: ~150 ranges vs 50k bits
- 6.3KB → 1.2KB (80% reduction)
- Fast set operations
- Example: `[[0, 100], [500, 600]]` instead of 700 individual IDs

### Fair Timing
- Each system gets its own process (no cross-contamination)
- Warm up runs before timing
- GC disabled during critical sections where possible
- High-resolution timers (nanoseconds)
- Multiple runs for statistical significance
