# Module and Expr core API (v1 implementation contract)

The v1 API has three stages: `Grammar` → `Module` → `Constraint`, plus `Expr` for composing pieces. `PreparedGrammar` is removed; its role folds into `Module`. `Grammar` remains immutable and source-only without compilation settings or bindings.

```rust
// Mask generation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[non_exhaustive]
pub enum MaskStrategy {
    #[default]
    Auto,
    ParserDwa,
    TokenTrie,
    TokenClassTrie,
}
impl MaskStrategy {
    pub const FAST_BUILD: Self = Self::TokenTrie;
    pub const FAST_RUNTIME: Self = Self::ParserDwa;
}

// Composition boundary detail
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[non_exhaustive]
pub enum BoundaryMaskStrategy {
    #[default]
    Auto,
    ParserDwa,
    TokenTrie,
}
impl BoundaryMaskStrategy {
    pub const FAST_BUILD: Self = Self::TokenTrie;
    pub const FAST_RUNTIME: Self = Self::ParserDwa;
}

// Compile settings
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CompileOptions { /* private strategies */ }
impl CompileOptions {
    pub fn mask_strategy(self, strategy: MaskStrategy) -> Self;
    pub fn boundary_mask_strategy(self, strategy: BoundaryMaskStrategy) -> Self;
}

// Module binding value wrapper
pub struct Binding { /* private fields */ }
// Infallible From impls: Module, &Module, Grammar<'_>, &Grammar<'_>, Expr, &Expr, TokenSet, &TokenSet, u32

impl Module {
    pub fn new(grammar: &Grammar<'_>) -> Self;

    pub fn bind(&mut self, name: &str, value: impl Into<Binding>) -> Result<()>;
    pub fn precompile(&mut self, vocab: &Vocab) -> Result<()>;
    pub fn precompile_with(&mut self, vocab: &Vocab, options: &CompileOptions) -> Result<()>;
    pub fn compile(&self, vocab: &Vocab) -> Result<Constraint>;
    pub fn compile_with(&self, vocab: &Vocab, options: CompileOptions) -> Result<Constraint>;

    pub fn externs(&self) -> Externs<'_>;
    pub fn save(&self) -> Vec<u8>;
    pub fn load(bytes: &[u8], vocab: Option<&Vocab>) -> Result<Self>;
}

impl Grammar {
    pub fn from_glrm(source: impl AsRef<str>) -> Self;
    pub fn from_ebnf(source: impl AsRef<str>) -> Self;
    pub fn from_json_schema(source: impl AsRef<str>) -> Self;
    pub fn from_lark(source: impl AsRef<str>) -> Self;

    pub fn externs(&self) -> Externs<'_>;
    pub fn compile(&self, vocab: &Vocab) -> Result<Constraint>;
    pub fn compile_with(&self, vocab: &Vocab, options: CompileOptions) -> Result<Constraint>;
    pub fn precompile(&self, vocab: &Vocab) -> Result<Module>;
    pub fn precompile_with(&self, vocab: &Vocab, options: CompileOptions) -> Result<Module>;
}

pub struct Extern { /* private fields */ }
impl Extern {
    pub fn name(&self) -> &str;
    pub fn kind(&self) -> ExternKind;
}

pub struct Externs<'a> { /* private fields */ }
impl<'a> Externs<'a> {
    pub fn get(&self, name: &str) -> Option<Extern>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn unbound(&self) -> Externs<'a>;
    pub fn iter(&self) -> impl Iterator<Item = Extern> + '_;
}

impl Expr {
    pub fn choice(items: impl IntoIterator<Item = impl Into<Expr>>) -> Self;
    pub fn sequence(items: impl IntoIterator<Item = impl Into<Expr>>) -> Self;
    pub fn literal(bytes: impl AsRef<[u8]>) -> Self;
}
// From impls into Expr: Grammar<'_>, &Grammar<'_>, Module, &Module, TokenSet, &TokenSet, u32, &Expr
```

## Rules and semantics

### Settings
- Strategies are compile settings: `mask_strategy` (replaces `Optimization`; `Balanced` is removed) and `boundary_mask_strategy` (replaces `BoundaryTriggerDetail`; default is `Auto`). Both default to `Auto`.
- `precompile` / `precompile_with` take the same options as `compile` / `compile_with`. One value applies to the whole compile, including every composed child; cold parts build in parallel.
- `precompile` is what decides which static parts get built and stored. `compile` uses every static part that already exists: a `TokenTrie` request after a static precompile keeps the static parts and produces the same masks rather than downgrading to dynamic compilation. The compile options govern only the parts not yet built.
- `boundary_mask_strategy` is a minimum: raising it builds more later; lowering it never removes built metadata, and the best metadata built is always used.
- `MaskStrategy` explicit requests (`ParserDwa`, `TokenTrie`, `TokenClassTrie`) never fall back silently. If the engine cannot handle the grammar, compilation errors explicitly; `Auto` may fall back to `TokenTrie`.
- `TokenTrie` compiles the whole composition from source, so `precompile` retains no work for it.

### Binding and composition
- `Module::bind` mutates `self` atomically: on error the module is unchanged.
- Each extern is bound once; rebinding an already bound extern is an error.
- A child must have every extern bound before it is attached. Attaching captures a snapshot.
- Qualified `::` paths are not supported: each extern is bound directly on the module that declares it.
- `compile` takes `&self` and never modifies `self` or passed objects. It errors if any extern is unbound.

### Expr
- Parts: `Grammar`, `Module`, `Expr`, `TokenSet`. Plain strings are rejected; use `Expr::literal`.
- `Expr::literal` matches exact bytes without whitespace or ignore inserted between parts.
- Empty choice matches nothing; one-part choice simplifies to that part. Empty sequence matches epsilon.
- Open parts with unbound externs are rejected at `bind`.

### Grammar language
- `extern` declarations are allowed only at the top level of a file.
- Inline grammars (`g NAME = { ... }`) cannot refer to outer names (rules, terminals, or externs).
- `use`, `export`, `import`, and `super` are reserved keywords.
- `::` is invalid in extern names and rejected by `bind`.

### Removals
- `PreparedGrammar`, `link()`, and `requirements` (use `Module`, `compile()`, and `externs()` / `unbound()`).
- `Grammar.bind` (Grammar is source-only) and `Grammar.exact` (pass a `TokenSet` directly).
- `BoundaryTriggerDetail` (replaced by `BoundaryMaskStrategy`).
- `Optimization` (replaced by `MaskStrategy`).
- Python `clone()` (replaced by `copy()`).
- Qualified `::` path resolution.
