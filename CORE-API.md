# Module and Expr core API (implementation contract)

`Grammar` remains source-only and its direct `compile(&Vocab)` path stays intact.
`Module` is owned (no source lifetime), cloneable with independent snapshots.

```rust
pub struct ModuleOptions { /* private fields */ }
impl ModuleOptions {
    pub fn optimization(self, value: Optimization) -> Self;
    pub fn optimization_value(&self) -> Optimization;
    pub fn boundary_trigger(self, value: BoundaryTriggerDetail) -> Self;
    pub fn boundary_trigger_value(&self) -> BoundaryTriggerDetail;
}
// Both are existing compiler options. Default = Auto and None.
// ModuleOptions implements Default, Clone, Debug, PartialEq, Eq.
pub struct CompileOptions { /* private root-only end_tokens */ }
impl CompileOptions {
    pub fn end_tokens(self, ids: impl IntoIterator<Item=u32>) -> Self;
}
impl Module {
    pub fn new(grammar: &Grammar<'_>) -> Self;
    pub fn from_expr(expr: &Expr) -> Result<Self>;
    pub fn options(&self) -> &ModuleOptions;
    pub fn with_options(&self, options: ModuleOptions) -> Result<Self>;
    pub fn set_options(&mut self, options: ModuleOptions) -> Result<()>;
    pub fn bind(&mut self, path: impl AsRef<str>, value: impl IntoModuleValue) -> Result<()>;
    pub fn precompile(&mut self, vocab: &Vocab) -> Result<()>;
    pub fn compile(&self, vocab: &Vocab) -> Result<Constraint>;
    pub fn compile_with(&self, vocab: &Vocab, options: CompileOptions) -> Result<Constraint>;
    pub fn unbound(&self) -> Result<Vec<UnboundExtern>>;
    pub fn externs(&self) -> Result<Vec<UnboundExtern>>;
    pub fn save(&self) -> Vec<u8>;
    pub fn load(bytes: impl AsRef<[u8]>) -> Result<Self>;
    pub fn load_with_vocab(bytes: impl AsRef<[u8]>, vocab: &Vocab) -> Result<Self>;
}
impl Grammar<'_> {
    pub fn module(&self) -> Module;
    pub fn from_expr(expr: &Expr) -> Result<Grammar<'static>>;
    pub fn unbound(&self) -> Result<Vec<UnboundExtern>>;
    pub fn externs(&self) -> Result<Vec<UnboundExtern>>;
}
pub struct UnboundExtern {
    pub path: Vec<String>, pub kind: ExternalKind,
    pub source_span: Option<SourceSpan>,
}
impl UnboundExtern { pub fn name(&self) -> String; }
pub trait IntoExpr { fn into_expr(self) -> Expr; }
impl Expr {
    pub fn choice(items: impl IntoIterator<Item=impl IntoExpr>) -> Self;
    pub fn sequence(items: impl IntoIterator<Item=impl IntoExpr>) -> Self;
}
```

`IntoExpr` and `IntoModuleValue` accept owned/borrowed Grammar, Module, Expr;
module binding also accepts Token and ExactTokens. Heterogeneous arrays use
explicit `Expr` conversion. Expression leaf paths are `leaf0`, `leaf1`, ... in left-to-right
order; operator nodes add no path segments. Empty choice is explicitly rejected on conversion;
empty sequence denotes epsilon. Whole grammar leaves keep independent lexical scopes.

Options are local and never inherited by attached children. Changing local options
after retaining incompatible local preparation returns an error atomically; a cold
parent can change options even when a child is prepared. Unsupported LR/native
composition returns an explicit error instead of converting or recompiling children.

Save/load is versioned and retains source, local options, bindings and compiled data.
Load never compiles. ExactTokens retain their complete target vocabulary identity.

The implemented trait conversion is fallible:

```rust
pub enum ModuleValue { Module(Module), Token(Token), ExactTokens(ExactTokens) }
pub trait IntoModuleValue { fn into_module_value(self) -> Result<ModuleValue>; }
```

PreparedGrammar also has `unbound()` and `externs()` returning the canonical
`UnboundExtern` values. The legacy `Requirement` re-export and `requirements()`
methods remain available only to tests and `internal-api` tooling.

Explicit current capability boundary: standalone FastBuild Modules retain LR and
round-trip it unchanged. The compiled Module linker accepts native components;
a FastBuild node on any explicit attachment edge returns `Error::Compilation`
before preparation/final linking, including all-LR attachment graphs. It never
recompiles or converts that supplied work. Closed Lark inline schemas support all
four modes; FastBuild with both inline schemas and open imports is rejected during
preparation. Native mixed Auto/Balanced/FastRuntime attachments are supported. FastBuild with
an Exact boundary trigger is rejected atomically by with_options and at load;
Exact triggers require native modules.

`Grammar::from_expr` is implemented on `Grammar<'static>` and accepts a reference
to Expr. A raw/default/cold Module operand is source-only and is accepted; any
configured or precompiled descendant is rejected. Bindings and Expr leaves are
snapshots. Source parsing errors may remain deferred in raw Module artifacts, as
in Grammar; malformed artifact topology, options and compiled bodies are errors.

Limits: 64 attachment levels, 131072 distinct module nodes, 1048576 bindings;
diagnostic expansion is bounded at 131072 visits and 65536 slots. Runtime graph
expansion is bounded separately. Compact shared DAGs remain serializable even
when their runtime/diagnostic expansion exceeds those limits. Artifact manifest
payloads are limited to 1 GiB at load.

`Grammar::bind` also accepts Expr (owned or borrowed) directly when all operands
are source-only. Configured or prepared operands return an error atomically;
use Module::bind for those values. No explicit Grammar::from_expr wrapper is
needed for ordinary source-expression binding.
