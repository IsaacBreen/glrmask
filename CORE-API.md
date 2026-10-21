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
    FastBuild,
    FastRuntime,
}

// Root compile options
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CompileOptions { /* private root-only end_tokens */ }
impl CompileOptions {
    pub fn end_tokens(self, ids: impl IntoIterator<Item = u32>) -> Self;
}

// Module binding value wrapper
pub struct Binding { /* private fields */ }
// Infallible From impls: Module, &Module, Grammar<'_>, &Grammar<'_>, Expr, &Expr, Token, &Token, ExactTokens, &ExactTokens

impl Module {
    pub fn new(grammar: &Grammar<'_>) -> Self;

    pub fn mask_strategy(&self) -> MaskStrategy;
    pub fn set_mask_strategy(&mut self, strategy: MaskStrategy) -> Result<()>;
    pub fn boundary_mask_strategy(&self) -> BoundaryMaskStrategy;
    pub fn set_boundary_mask_strategy(&mut self, strategy: BoundaryMaskStrategy) -> Result<()>;

    pub fn bind(&mut self, name: &str, value: impl Into<Binding>) -> Result<()>;
    pub fn precompile(&mut self, vocab: &Vocab) -> Result<()>;
    pub fn compile(&self, vocab: &Vocab) -> Result<Constraint>;
    pub fn compile_with(&self, vocab: &Vocab, options: CompileOptions) -> Result<Constraint>;

    pub fn externs(&self) -> Externs<'_>;
    pub fn save(&self) -> Vec<u8>;
    pub fn load(bytes: &[u8], vocab: Option<&Vocab>) -> Result<Self>;
}

impl<'a> Grammar<'a> {
    pub fn from_glrm(source: &'a str) -> Self;
    pub fn from_ebnf(source: &'a str) -> Self;
    pub fn from_json_schema(source: &'a str) -> Self;
    pub fn from_lark(source: &'a str) -> Self;
    pub fn from_lark_with(source: &'a str, options: LarkOptions) -> Self;

    pub fn compile(&self, vocab: &Vocab) -> Result<Constraint>;
    pub fn compile_with(&self, vocab: &Vocab, options: CompileOptions) -> Result<Constraint>;
}

pub struct Extern { /* private fields */ }
impl Extern {
    pub fn name(&self) -> &str;
    pub fn kind(&self) -> ExternKind;
    pub fn span(&self) -> Option<SourceSpan>;
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
// From impls into Expr: Grammar<'_>, &Grammar<'_>, Module, &Module, Token, &Token, ExactTokens, &ExactTokens, &Expr
```

## Rules and semantics

### Settings
- Settings are per Module: `mask_strategy` (replaces `Optimization`; `Balanced` is removed) and `boundary_mask_strategy` (replaces `BoundaryTriggerDetail`; default is `FastBuild`).
- `precompile` and `compile` build each part of the graph at that part's own settings; cold parts build in parallel.
- `mask_strategy` is resolved at compile time, so changing it never invalidates or discards precompiled work.
- `boundary_mask_strategy` is a minimum: raising it builds more later; lowering it never removes built metadata, and the best metadata built is always used.
- `MaskStrategy` explicit requests (`ParserDwa`, `TokenTrie`, `TokenClassTrie`) never fall back silently. If the engine cannot handle the grammar or cannot link with neighbouring parts, compilation errors explicitly.
- `TokenTrie` compiles the whole composition from source; `precompile` retains no work for `TokenTrie` modules. `Auto` parts join a `TokenTrie` composition; explicit `ParserDwa`/`TokenClassTrie` parts cannot be combined with `TokenTrie` parts.
- Implicit modules (a bare `Grammar` or `Expr` used as a binding or part) receive default settings.
- `end_tokens` is a root compile setting configured via `CompileOptions` (`CompileOptions::default().end_tokens(...)`). It applies only to the root of that `compile` call and is not inherited by children.

### Binding and composition
- `Module::bind` mutates `self` atomically: on error the module is unchanged.
- Each extern is bound once; rebinding an already bound extern is an error.
- A child must have every extern bound before it is attached. Attaching captures a snapshot.
- Qualified `::` paths are not supported: each extern is bound directly on the module that declares it.
- `compile` takes `&self` and never modifies `self` or passed objects. It errors if any extern is unbound.

### Expr
- Parts: `Grammar`, `Module`, `Expr`, `Token`, `ExactTokens`. Plain strings are rejected; use `Expr::literal`.
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
- `Grammar.bind` (Grammar is source-only) and `Grammar.exact` (pass `Token` or `ExactTokens` directly).
- `BoundaryTriggerDetail` (replaced by `BoundaryMaskStrategy`).
- `Optimization` (replaced by `MaskStrategy`).
- Python `clone()` (replaced by `copy()`).
- Qualified `::` path resolution.
