# Comprehensive Benchmarking Plan

## System Format Requirements

### 1. **Outlines** (dottxt-ai/outlines)
- **Format**: JSON Schema, Regex, Context-Free Grammars (EBNF-like)
- **API**: Python library
- **Usage**:
  ```python
  import outlines
  # JSON Schema
  model = outlines.from_transformers(...)
  result = model(prompt, json_schema=my_schema)
  
  # CFG
  result = model(prompt, grammar=my_ebnf_string)
  ```
- **Notes**: High-level API, mature Python library

### 2. **XGrammar** (mlc-ai/xgrammar)
- **Format**: JSON Schema, EBNF
- **API**: Python/C++ bindings  
- **Usage**:
  ```python
  from xgrammar import GrammarCompiler
  compiled = GrammarCompiler.compile_json_schema(schema_str)
  # OR
  compiled = GrammarCompiler.compile_ebnf(ebnf_str)
  ```
- **Notes**: Fast C++ implementation, supports both JSON and EBNF

### 3. **llguidance** (Microsoft)
- **Format**: Custom grammar format (need to investigate)
- **API**: Rust library with Python bindings
- **Usage**: TBD - need to examine repository more
- **Notes**: Low-level, used by Guidance framework

### 4. **sep1** (Our system)
- **Format**: EBNF → JSON (precompiled)
- **API**: Rust with Python bindings
- **Usage**: Already have working implementation

## Grammar Selection Strategy

### Option 1: JSON Schema (SIMPLEST - START HERE)
**Rationale**: All systems support JSON Schema
**Test Cases**:
1. **Simple object**: `{"name": "string", "age": "number"}`
2. **Nested structure**: Objects with arrays and nested objects
3. **Complex schema**: From JSONSchemaBench

**Pros**:
- Universal support across all systems
- Easy to validate correctness (parse as JSON)
- Real-world use case

**Cons**:
- Not grammar-constrained in traditional sense
- Doesn't test full GLR parser capabilities

### Option 2: Programming Language Grammar (MOST RIGOROUS)
**Languages to consider**:
1. **JavaScript** (we have full grammar)
2. **Python** (download/write simplified version)
3. **SQL** (write/download)

**Challenge**: Need to either:
- Convert our EBNF to each system's format
- Write equivalent grammars manually for each

**Test approach for each language**:
- Generate small valid programs
- Validate they parse correctly  
- Measure performance

### Option 3: Hybrid Approach (RECOMMENDED)
1. **JSON Schema benchmarks** (universal, easy)
2. **One programming language** (JavaScript - we already have it)
   - For systems supporting EBNF: use our grammar
   - For systems not supporting EBNF: skip or use JSON equivalent

## Test Input Strategy

### For JSON Schema:
Generate completion tasks:
```python
prompts = [
    "Generate a user profile:",  # → {"name": "...", "age": 25}
    "Create a list of products:",  # → [{"id": 1, "name": "..."}]
    "Output the configuration:",  # → nested JSON
]
```

**Metrics**:
- Tokens until completion
- Time per token
- Total generation time
- Correctness (does it parse? does it match schema?)

### For JavaScript Grammar:
Incomplete code snippets:
```javascript
test_inputs = [
    "if (true) { if (true) { if (true) {",  # → generate closing braces and complete
    "function calculate(x",  # → complete function signature and body
    "const arr = [1,",  # → complete array literal
]
```

**Metrics**:
- Constraint computation time per token
- Memory usage
- Correct ness (does generated code parse?)

## Implementation Plan

### Phase 1: JSON Schema Benchmarks (DO THIS FIRST)
**Goal**: Get all systems working with simplest case

1. **Define 3-5 JSON schemas** of varying complexity
2. **Implement wrappers**:
   - `benchmarking/systems/outlines_wrapper.py`
   - `benchmarking/systems/xgrammar_wrapper.py`  
   - `benchmarking/systems/llguidance_wrapper.py`
   - `benchmarking/systems/sep1_wrapper.py` (update to use JSON schema)
3. **Run benchmarks**: Generate completions, measure time
4. **Validate**: Parse all outputs as JSON, check schema compliance
5. **Analyze**: Create comparison tables and plots

### Phase 2: JavaScript Grammar (IF TIME)
**Goal**: Test on real programming language

1. **Convert JavaScript grammar to each format** OR skip systems that don't support EBNF
2. **Define test inputs**: Incomplete JS code
3. **Run benchmarks**  
4. **Validate**: Use espree/acorn to parse generated JS
5. **Analyze**: Compare with JSON schema results

## Specific Next Steps

1. **RIGHT NOW**: Implement Outlines wrapper with JSON schema
2. **THEN**: Implement XGrammar wrapper  
3. **THEN**: Investigate llguidance format
4. **THEN**: Create test schemas  
5. **THEN**: Run benchmarks
6. **THEN**: Analyze and document results

## Test Schemas (Concrete)

```python
SCHEMAS = {
    "simple_user": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"}
        },
        "required": ["name", "age"]
    },
    
    "product_array": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "number"},
                "name": {"type": "string"},
                "price": {"type": "number"}
            }
        }
    },
    
    "nested_config": {
        "type": "object",
        "properties": {
            "server": {
                "type": "object",
                "properties": {
                    "host": {"type": "string"},
                    "port": {"type": "number"},
                    "ssl": {"type": "boolean"}
                }
            },
            "database": {
                "type": "object", 
                "properties": {
                    "url": {"type": "string"},
                    "credentials": {
                        "type": "object",
                        "properties": {
                            "username": {"type": "string"},
                            "password": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
}
```

## Decision: Start with JSON Schema

**Rationale**:
1. Universal support - all systems handle it
2. Easy validation - just parse JSON
3. Real-world relevance - most common use case
4. Can get comprehensive results quickly  
5. Grammar conversion problem avoided

We can add JavaScript benchmarks later if needed, but JSON Schema gives us complete comparison data NOW.
