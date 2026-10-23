#ifndef GLRMASK_HPP
#define GLRMASK_HPP
// C++17 RAII facade. Implementation and engine allocations remain in Rust.
#include "glrmask.h"
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace glrmask {
/// An original model token ID.
using TokenId = std::uint32_t;
/// Inclusive token-ID interval.
using TokenRange = glrmask_token_range; // inclusive first/last endpoints
/// A possible boundary observation.
using MarkerEvent = glrmask_marker_event;
/// Owning byte buffer.
using Bytes = std::vector<std::uint8_t>;

/// Mask computation strategy, trading build time against per-token latency.
enum class MaskStrategy : std::uint32_t {
    /// Automatic selection.
    Auto = 0,
    /// Static masks: slowest build, fastest per-token masks.
    ParserDwa = 1,
    /// Trie walk per mask: fastest build.
    TokenTrie = 2,
    /// Trie walk over token equivalence classes.
    TokenClassTrie = 3
};
/// Alias for `MaskStrategy::TokenTrie`.
inline constexpr MaskStrategy kFastBuild = MaskStrategy::TokenTrie;
/// Alias for `MaskStrategy::ParserDwa`.
inline constexpr MaskStrategy kFastRuntime = MaskStrategy::ParserDwa;

/// Boundary parser selection for crossings between composed grammars.
enum class BoundaryMaskStrategy : std::uint32_t {
    /// Automatic selection.
    Auto = 0,
    /// Static boundary parser; faster crossings.
    ParserDwa = 1,
    /// Segmented dynamic boundary token walk; faster build.
    TokenTrie = 2,
    /// Alias for `TokenTrie`.
    FastBuild = TokenTrie,
    /// Alias for `ParserDwa`.
    FastRuntime = ParserDwa
};

/// The kind of an extern slot.
enum class ExternKind {
    /// A token-level slot.
    Token,
    /// A grammar-level slot.
    Grammar
};

/// A GLRMask failure carrying the C status.
class Error : public std::runtime_error {
    glrmask_status status_;
public:
    /// Creates an error with a status and message.
    Error(glrmask_status status, const std::string& message) : std::runtime_error(message), status_(status) {}
    /// The C status that caused the failure.
    glrmask_status status() const noexcept { return status_; }
};

/// A declared extern slot.
struct Extern {
    /// The declared name.
    std::string name;
    /// The declared kind: token or grammar.
    ExternKind kind;
};

/// A snapshot view of declared externs, in declaration order.
class Externs {
    std::vector<Extern> entries_;
    std::shared_ptr<std::vector<Extern>> unbound_;
public:
    /// Creates an empty view.
    Externs() : entries_{}, unbound_{std::make_shared<std::vector<Extern>>()} {}
    /// Creates a view from entries and a separate unresolved snapshot.
    explicit Externs(std::vector<Extern> entries, std::vector<Extern> unbound = {})
        : entries_(std::move(entries)), unbound_(std::make_shared<std::vector<Extern>>(std::move(unbound))) {}

    using const_iterator = std::vector<Extern>::const_iterator;
    /// First extern.
    const_iterator begin() const noexcept { return entries_.begin(); }
    /// One past the last extern.
    const_iterator end() const noexcept { return entries_.end(); }
    /// First extern.
    const_iterator cbegin() const noexcept { return entries_.cbegin(); }
    /// One past the last extern.
    const_iterator cend() const noexcept { return entries_.cend(); }

    /// Number of externs in this view.
    std::size_t size() const noexcept { return entries_.size(); }
    /// Whether this view contains no externs.
    bool empty() const noexcept { return entries_.empty(); }

    /// The extern at `index`.
    const Extern& operator[](std::size_t index) const { return entries_[index]; }
    /// The extern at `index`; throws `std::out_of_range` if absent.
    const Extern& at(std::size_t index) const { return entries_.at(index); }

    /// The extern declared as `name`, if any.
    std::optional<Extern> get(std::string_view name) const {
        for (const auto& item : entries_) {
            if (item.name == name) return item;
        }
        return std::nullopt;
    }
    /// Whether an extern named `name` is declared.
    bool has(std::string_view name) const {
        for (const auto& item : entries_) {
            if (item.name == name) return true;
        }
        return false;
    }

    /// The same view restricted to unresolved externs.
    Externs unbound() const {
        if (!unbound_) return Externs{};
        return Externs(*unbound_, {});
    }
};

/// Compilation settings for a grammar or module assembly.
///
/// `end_tokens` applies only to the top-level sequence; the strategies apply
/// to the whole compile, including every composed child.
struct CompileOptions {
    /// Accepted end-of-sequence token IDs, admitted once the grammar is complete.
    std::vector<TokenId> end_tokens;
    /// Mask computation strategy for the whole compile.
    MaskStrategy mask_strategy = MaskStrategy::Auto;
    /// Boundary parser strategy for the whole compile.
    BoundaryMaskStrategy boundary_mask_strategy = BoundaryMaskStrategy::Auto;
    /// No end tokens and automatic strategies.
    CompileOptions() = default;
    /// The given end tokens with automatic strategies.
    CompileOptions(std::vector<TokenId> tokens) : end_tokens(std::move(tokens)) {}
    /// The matching C options struct.
    glrmask_compile_options native() const noexcept {
        return {sizeof(glrmask_compile_options), end_tokens.data(), end_tokens.size(),
                static_cast<std::uint32_t>(mask_strategy),
                static_cast<std::uint32_t>(boundary_mask_strategy)};
    }
};

/// One vocabulary entry: a model token ID and its exact bytes.
struct VocabEntry {
    /// The model token ID.
    TokenId id;
    /// The exact token bytes.
    Bytes bytes;
    /// Creates an entry from bytes.
    VocabEntry(TokenId id_, Bytes bytes_) : id(id_), bytes(std::move(bytes_)) {}
    /// Creates an entry from a string view's bytes.
    VocabEntry(TokenId id_, std::string_view bytes_) : id(id_), bytes(bytes_.begin(), bytes_.end()) {}
};

namespace detail {
inline void check(glrmask_status status) {
    if (status != GLRMASK_OK) throw Error(status, glrmask_error_message());
}
template<class H, void(*Free)(H*), glrmask_status(*Clone)(const H*, H**)>
class Owned {
    std::unique_ptr<H, decltype(Free)> handle_;
protected:
    explicit Owned(H* handle) : handle_(handle, Free) {}
public:
    Owned(const Owned& other) : handle_(nullptr, Free) {
        H* handle = nullptr; check(Clone(other.native_handle(), &handle)); handle_.reset(handle);
    }
    Owned(Owned&&) noexcept = default;
    Owned& operator=(const Owned& other) { Owned copy(other); handle_.swap(copy.handle_); return *this; }
    Owned& operator=(Owned&&) noexcept = default;
    H* native_handle() const {
        if (!handle_) throw Error(GLRMASK_CLOSED, "object has been disposed or moved from");
        return handle_.get();
    }
    void dispose() noexcept { handle_.reset(); }
};

inline Bytes take_bytes(glrmask_bytes* raw) {
    std::unique_ptr<glrmask_bytes, decltype(&glrmask_bytes_free)> owner(raw, glrmask_bytes_free);
    const std::uint8_t* data = nullptr; std::size_t len = 0;
    check(glrmask_bytes_data(raw, &data, &len));
    return len ? Bytes(data, data + len) : Bytes{};
}
inline std::vector<TokenId> take_ids(glrmask_u32s* raw) {
    std::unique_ptr<glrmask_u32s, decltype(&glrmask_u32s_free)> owner(raw, glrmask_u32s_free);
    const TokenId* data = nullptr; std::size_t len = 0;
    check(glrmask_u32s_data(raw, &data, &len));
    return len ? std::vector<TokenId>(data, data + len) : std::vector<TokenId>{};
}
inline std::vector<Extern> take_externs(glrmask_externs* raw) {
    std::unique_ptr<glrmask_externs, decltype(&glrmask_externs_free)> owner(raw, glrmask_externs_free);
    std::size_t len = 0; check(glrmask_externs_len(raw, &len));
    std::vector<Extern> result; result.reserve(len);
    for (std::size_t i = 0; i < len; ++i) {
        glrmask_extern value{}; check(glrmask_externs_get(raw, i, &value));
        Extern e; e.name.assign(value.name, value.name_len);
        e.kind = value.kind == GLRMASK_EXTERN_TOKEN ? ExternKind::Token : ExternKind::Grammar;
        result.push_back(std::move(e));
    }
    return result;
}
} // namespace detail

class Vocab;
class Grammar;
class Module;
class Expr;
class Constraint;
class ConstraintState;

/// A set of model token IDs. IDs are validated against the compile vocabulary.
class TokenSet : public detail::Owned<glrmask_token_set, glrmask_token_set_free, glrmask_token_set_clone> {
    using Base = detail::Owned<glrmask_token_set, glrmask_token_set_free, glrmask_token_set_clone>;
    explicit TokenSet(glrmask_token_set* p) : Base(p) {}
public:
    /// Creates a set from token IDs; duplicates are ignored.
    explicit TokenSet(const std::vector<TokenId>& ids) : Base(create_ids(ids)) {}
    /// Creates a set from inclusive `ranges`; overlapping or adjacent ranges merge.
    static TokenSet from_ranges(const std::vector<TokenRange>& ranges) {
        glrmask_token_set* p = nullptr; detail::check(glrmask_token_set_from_ranges(ranges.data(), ranges.size(), &p)); return TokenSet(p);
    }
private:
    static glrmask_token_set* create_ids(const std::vector<TokenId>& ids) {
        glrmask_token_set* p = nullptr; detail::check(glrmask_token_set_new(ids.data(), ids.size(), &p)); return p;
    }
};

/// The model vocabulary a constraint is compiled against.
class Vocab : public detail::Owned<glrmask_vocab, glrmask_vocab_free, glrmask_vocab_clone> {
    using Base = detail::Owned<glrmask_vocab, glrmask_vocab_free, glrmask_vocab_clone>;
    static glrmask_vocab* create(const std::vector<VocabEntry>& entries, const std::vector<TokenId>& special) {
        std::vector<glrmask_vocab_entry> native; native.reserve(entries.size());
        for (const auto& e : entries) native.push_back({e.id, e.bytes.data(), e.bytes.size()});
        glrmask_vocab* p = nullptr;
        detail::check(glrmask_vocab_new(native.data(), native.size(), special.data(), special.size(), &p)); return p;
    }
public:
    /// Creates a vocabulary from byte entries plus special token IDs.
    explicit Vocab(const std::vector<VocabEntry>& entries, const std::vector<TokenId>& special = {}) : Base(create(entries, special)) {}
    /// Number of byte-backed entries; special IDs are not counted.
    std::size_t size() const { std::size_t n = 0; detail::check(glrmask_vocab_len(native_handle(), &n)); return n; }
};

/// The source-language definition, without compilation settings.
class Grammar : public detail::Owned<glrmask_grammar, glrmask_grammar_free, glrmask_grammar_clone> {
    using Base = detail::Owned<glrmask_grammar, glrmask_grammar_free, glrmask_grammar_clone>;
    explicit Grammar(glrmask_grammar* p) : Base(p) {}
    static Grammar source(std::uint32_t format, std::string_view text) {
        glrmask_grammar* p = nullptr; detail::check(glrmask_grammar_from_source(format, text.data(), text.size(), &p)); return Grammar(p);
    }
public:
    /// Creates a grammar from EBNF source text.
    static Grammar from_ebnf(std::string_view text) { return source(GLRMASK_EBNF, text); }
    /// Creates a grammar from GLRM source.
    static Grammar from_glrm(std::string_view text) { return source(GLRMASK_GLRM, text); }
    /// Creates a grammar from a JSON Schema document.
    static Grammar from_json_schema(std::string_view text) { return source(GLRMASK_JSON_SCHEMA, text); }
    /// Creates a grammar from Lark source; named token atoms declare token externs.
    static Grammar from_lark(std::string_view text) { return source(GLRMASK_LARK, text); }
    /// A snapshot view of the grammar's declared externs, in declaration order.
    Externs externs() const {
        glrmask_externs* p = nullptr;
        detail::check(glrmask_grammar_externs(native_handle(), &p));
        return Externs(detail::take_externs(p));
    }
    /// Compiles the grammar against `vocab`.
    Constraint compile(const Vocab& vocab, const CompileOptions& options = {}) const;
    /// Precompiles the grammar against `vocab` and returns a module.
    Module precompile(const Vocab& vocab, const CompileOptions& options = {}) const;
};

/// One `Expr` operand: a grammar, module, expression, token ID or token set.
struct ExprOperand {
    glrmask_expr_operand value;
    /// From a grammar.
    ExprOperand(const Grammar&);
    /// From a module.
    ExprOperand(const Module&);
    /// From an expression.
    ExprOperand(const Expr&);
    /// From a single token ID.
    ExprOperand(TokenId);
    /// From a token set.
    ExprOperand(const TokenSet&);
};

/// An immutable unordered union or concatenation of independently scoped grammars.
class Expr : public detail::Owned<glrmask_expr, glrmask_expr_free, glrmask_expr_clone> {
    using Base = detail::Owned<glrmask_expr, glrmask_expr_free, glrmask_expr_clone>;
    explicit Expr(glrmask_expr* p) : Base(p) {}
public:
    /// Creates a union without ordered-choice priority; an empty choice matches nothing.
    static Expr choice(const std::vector<ExprOperand>& values) {
        std::vector<glrmask_expr_operand> items; items.reserve(values.size());
        for (const auto& value : values) items.push_back(value.value);
        glrmask_expr* p = nullptr; detail::check(glrmask_expr_choice(items.data(), items.size(), &p)); return Expr(p);
    }
    /// Creates a concatenation; a model token may cross any number of operand boundaries.
    static Expr sequence(const std::vector<ExprOperand>& values) {
        std::vector<glrmask_expr_operand> items; items.reserve(values.size());
        for (const auto& value : values) items.push_back(value.value);
        glrmask_expr* p = nullptr; detail::check(glrmask_expr_sequence(items.data(), items.size(), &p)); return Expr(p);
    }
    /// Matches exact bytes, with no whitespace or ignore inserted.
    static Expr literal(const std::uint8_t* data, std::size_t len) {
        glrmask_expr* p = nullptr; detail::check(glrmask_expr_literal(data, len, &p)); return Expr(p);
    }
    /// Matches the bytes of a string view.
    static Expr literal(std::string_view bytes) {
        return literal(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
    }
    /// Matches the bytes of an owning buffer.
    static Expr literal(const Bytes& bytes) {
        return literal(bytes.data(), bytes.size());
    }
};

/// A reusable grammar with bindings and any retained compilation work.
class Module : public detail::Owned<glrmask_module, glrmask_module_free, glrmask_module_clone> {
    using Base = detail::Owned<glrmask_module, glrmask_module_free, glrmask_module_clone>;
    explicit Module(glrmask_module* p) : Base(p) {}
    friend class Grammar;
    static glrmask_module* create(const Grammar& grammar) {
        glrmask_module* p = nullptr; detail::check(glrmask_module_new(grammar.native_handle(), &p)); return p;
    }
public:
    /// Creates a module from a source grammar with default settings.
    explicit Module(const Grammar& grammar) : Base(create(grammar)) {}

    /// Binds an extern to a module.
    void bind(std::string_view name, const Module& value) {
        detail::check(glrmask_module_bind_module(native_handle(), name.data(), name.size(), value.native_handle()));
    }
    /// Binds an extern to a grammar.
    void bind(std::string_view name, const Grammar& value) {
        detail::check(glrmask_module_bind_grammar(native_handle(), name.data(), name.size(), value.native_handle()));
    }
    /// Binds an extern to an expression.
    void bind(std::string_view name, const Expr& value) {
        detail::check(glrmask_module_bind_expr(native_handle(), name.data(), name.size(), value.native_handle()));
    }
    /// Binds an extern to one token ID.
    void bind(std::string_view name, TokenId value) {
        detail::check(glrmask_module_bind_token(native_handle(), name.data(), name.size(), value));
    }
    /// Binds an extern to a token set.
    void bind(std::string_view name, const TokenSet& value) {
        detail::check(glrmask_module_bind_tokens(native_handle(), name.data(), name.size(), value.native_handle()));
    }

    /// Retains vocabulary-specific compilation artifacts for this module.
    void precompile(const Vocab& vocab, const CompileOptions& options = {}) {
        const glrmask_compile_options native = options.native();
        detail::check(glrmask_module_precompile(native_handle(), vocab.native_handle(), &native));
    }
    /// Compiles the module assembly against `vocab`; every extern must be bound.
    Constraint compile(const Vocab& vocab, const CompileOptions& options = {}) const;

    /// A snapshot view of the module's declared externs, in declaration order.
    Externs externs() const {
        glrmask_externs* p_all = nullptr;
        detail::check(glrmask_module_externs(native_handle(), &p_all));
        auto all_items = detail::take_externs(p_all);

        glrmask_externs* p_unbound = nullptr;
        detail::check(glrmask_module_unbound_externs(native_handle(), &p_unbound));
        auto unbound_items = detail::take_externs(p_unbound);

        return Externs(std::move(all_items), std::move(unbound_items));
    }

    /// Serializes the module assembly and any retained precompiled artifacts.
    Bytes save() const {
        glrmask_bytes* p = nullptr; detail::check(glrmask_module_save(native_handle(), &p)); return detail::take_bytes(p);
    }
    /// Deserializes a module assembly, verifying it against `vocab` when supplied.
    static Module load(const Bytes& bytes) {
        glrmask_module* p = nullptr; detail::check(glrmask_module_load(bytes.data(), bytes.size(), &p)); return Module(p);
    }
};

/// From a grammar.
inline ExprOperand::ExprOperand(const Grammar& value) : value{0, value.native_handle()} {}
/// From a module.
inline ExprOperand::ExprOperand(const Module& value) : value{1, value.native_handle()} {}
/// From an expression.
inline ExprOperand::ExprOperand(const Expr& value) : value{2, value.native_handle()} {}
/// From a single token ID.
inline ExprOperand::ExprOperand(TokenId value) : value{3, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(value))} {}
/// From a token set.
inline ExprOperand::ExprOperand(const TokenSet& value) : value{4, value.native_handle()} {}

/// A fully runnable grammar compiled against a specific vocabulary.
class Constraint : public detail::Owned<glrmask_constraint, glrmask_constraint_free, glrmask_constraint_clone> {
    using Base = detail::Owned<glrmask_constraint, glrmask_constraint_free, glrmask_constraint_clone>;
    explicit Constraint(glrmask_constraint* p) : Base(p) {}
    friend class Grammar; friend class Module;
public:
    /// Creates independent state for one generated sequence.
    ConstraintState start() const;
    /// Number of packed `u32` words in a mask.
    std::size_t mask_len() const { std::size_t n = 0; detail::check(glrmask_constraint_mask_len(native_handle(), &n)); return n; }
    /// Sorted, distinct application marker IDs.
    std::vector<TokenId> marker_ids() const { glrmask_u32s* p = nullptr; detail::check(glrmask_constraint_marker_ids(native_handle(), &p)); return detail::take_ids(p); }
    /// Serializes the constraint; `external_vocab` omits the vocabulary.
    Bytes save() const { glrmask_bytes* p = nullptr; detail::check(glrmask_constraint_save(native_handle(), &p)); return detail::take_bytes(p); }
    /// Deserializes a constraint, optionally with the exact original vocabulary.
    static Constraint load(const Bytes& bytes, const Vocab* vocab = nullptr) {
        glrmask_constraint* p = nullptr; detail::check(glrmask_constraint_load(bytes.data(), bytes.size(), vocab ? vocab->native_handle() : nullptr, &p)); return Constraint(p);
    }
};

/// The mutable state for a single generated sequence.
class ConstraintState : public detail::Owned<glrmask_state, glrmask_state_free, glrmask_state_clone> {
    using Base = detail::Owned<glrmask_state, glrmask_state_free, glrmask_state_clone>;
    explicit ConstraintState(glrmask_state* p) : Base(p) {}
    friend class Constraint;
public:
    /// Number of packed `u32` words in a mask.
    std::size_t mask_len() const { std::size_t n = 0; detail::check(glrmask_state_mask_len(native_handle(), &n)); return n; }
    /// Writes the current mask; extra words are cleared and `output` must hold `word_count` words.
    void fill_mask(TokenId* output, std::size_t word_count) const { detail::check(glrmask_state_fill_mask(native_handle(), output, word_count)); }
    /// Writes the current mask into `output`, which must hold `mask_len()` words.
    void fill_mask(std::vector<TokenId>& output) const { fill_mask(output.data(), output.size()); }
    /// The current allowed-token mask as packed original-token-ID words.
    std::vector<TokenId> mask() const { std::vector<TokenId> out(mask_len()); fill_mask(out); return out; }
    /// Commits one model token event; unknown or forbidden IDs reject the state.
    void commit_token(TokenId id) { detail::check(glrmask_state_commit_token(native_handle(), id)); }
    void commit_tokens(const std::uint32_t* ids, std::size_t count) { detail::check(glrmask_state_commit_tokens(native_handle(), ids, count)); }
    void commit_tokens(const std::vector<std::uint32_t>& ids) { commit_tokens(ids.data(), ids.size()); }
    /// Commits raw bytes rather than a model token event.
    void commit_bytes(const std::uint8_t* bytes, std::size_t size) { detail::check(glrmask_state_commit_bytes(native_handle(), bytes, size)); }
    /// Commits the bytes of an owning buffer.
    void commit_bytes(const Bytes& bytes) { commit_bytes(bytes.data(), bytes.size()); }
    /// Commits the bytes of a string view.
    void commit_bytes(std::string_view bytes) { commit_bytes(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()); }
    /// Whether the input so far is a complete match. With end tokens configured, this becomes true only once an end token is committed.
    bool is_accepting() const { std::uint32_t value = 0; detail::check(glrmask_state_is_accepting(native_handle(), &value)); return value != 0; }
    /// Whether the prefix cannot be extended to a valid match.
    /// A read-only prefix of library-canonical forced output.
    std::vector<TokenId> forced_tokens() const { glrmask_u32s* p = nullptr; detail::check(glrmask_state_forced_tokens(native_handle(), &p)); return detail::take_ids(p); }
    /// Initial or most recent commit's marker observations, ordered by byte offset then ID.
    std::vector<MarkerEvent> marker_events() const {
        glrmask_events* raw = nullptr; detail::check(glrmask_state_marker_events(native_handle(), &raw));
        std::unique_ptr<glrmask_events, decltype(&glrmask_events_free)> owner(raw, glrmask_events_free);
        const MarkerEvent* data = nullptr; std::size_t len = 0; detail::check(glrmask_events_data(raw, &data, &len));
        return len ? std::vector<MarkerEvent>(data, data + len) : std::vector<MarkerEvent>{};
    }
};

inline Constraint Grammar::compile(const Vocab& vocab, const CompileOptions& options) const {
    const auto native = options.native(); glrmask_constraint* p = nullptr;
    detail::check(glrmask_grammar_compile(native_handle(), vocab.native_handle(), &native, &p)); return Constraint(p);
}
inline Module Grammar::precompile(const Vocab& vocab, const CompileOptions& options) const {
    const auto native = options.native(); glrmask_module* p = nullptr;
    detail::check(glrmask_grammar_precompile(native_handle(), vocab.native_handle(), &native, &p)); return Module(p);
}
inline Constraint Module::compile(const Vocab& vocab, const CompileOptions& options) const {
    const auto native = options.native(); glrmask_constraint* p = nullptr;
    detail::check(glrmask_module_compile(native_handle(), vocab.native_handle(), &native, &p)); return Constraint(p);
}
inline ConstraintState Constraint::start() const {
    glrmask_state* p = nullptr; detail::check(glrmask_constraint_start(native_handle(), &p)); return ConstraintState(p);
}
/// The library version string.
inline const char* version() noexcept { return glrmask_version(); }
} // namespace glrmask
#endif
