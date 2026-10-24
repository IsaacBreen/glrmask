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
/// Token IDs from `first` to `last`, inclusive.
using TokenRange = glrmask_token_range;
/// A marker passed by a commit.
using MarkerEvent = glrmask_marker_event;
/// A byte buffer.
using Bytes = std::vector<std::uint8_t>;

/// How masks are computed.
enum class MaskStrategy : std::uint32_t {
    /// Reuses any precompiled work; otherwise `ParserDwa`, or `TokenTrie` when that would be
    /// too large. The default.
    Auto = 0,
    /// Builds a parser DWA for the grammar. Slowest to compile, with the fastest masks.
    ParserDwa = 1,
    /// Computes each mask by walking the vocabulary. Fastest to compile, with slower masks.
    TokenTrie = 2,
    /// Same as `TokenTrie`.
    FastCompile = TokenTrie,
    /// Same as `ParserDwa`.
    FastMasks = ParserDwa
};

/// How masks are computed at grammar boundaries.
enum class BoundaryMaskStrategy : std::uint32_t {
    /// `ParserDwa`, or `TokenTrie` when that would be too large. The default.
    Auto = 0,
    /// Builds a parser DWA across each boundary. Slower to compile, with faster masks. Throws
    /// `Error` if it would be too large.
    ParserDwa = 1,
    /// Walks the vocabulary at each boundary as needed. Faster to compile, with slower masks.
    TokenTrie = 2,
    /// Same as `TokenTrie`.
    FastCompile = TokenTrie,
    /// Same as `ParserDwa`.
    FastMasks = ParserDwa
};

/// What an extern can be bound to.
enum class ExternKind {
    /// A token set or token ID.
    Token,
    /// A grammar, module, expression, token set or token ID.
    Grammar
};

/// A GLRMask error.
class Error : public std::runtime_error {
    glrmask_status status_;
public:
    /// Creates an error. GLRMask throws these; you don't need to create them.
    Error(glrmask_status status, const std::string& message) : std::runtime_error(message), status_(status) {}
    /// The C status code.
    glrmask_status status() const noexcept { return status_; }
};

/// An extern declared by a grammar: a named slot to bind later.
struct Extern {
    /// The extern's name.
    std::string name;
    /// What the extern can be bound to.
    ExternKind kind;
};

/// A grammar's or module's externs, in declaration order.
class Externs {
    std::vector<Extern> entries_;
    std::shared_ptr<std::vector<Extern>> unbound_;
public:
    /// An empty view.
    Externs() : entries_{}, unbound_{std::make_shared<std::vector<Extern>>()} {}
    /// Used by the library.
    explicit Externs(std::vector<Extern> entries, std::vector<Extern> unbound = {})
        : entries_(std::move(entries)), unbound_(std::make_shared<std::vector<Extern>>(std::move(unbound))) {}

    using const_iterator = std::vector<Extern>::const_iterator;
    /// Iterators over the externs.
    const_iterator begin() const noexcept { return entries_.begin(); }
    /// Iterators over the externs.
    const_iterator end() const noexcept { return entries_.end(); }
    /// Iterators over the externs.
    const_iterator cbegin() const noexcept { return entries_.cbegin(); }
    /// Iterators over the externs.
    const_iterator cend() const noexcept { return entries_.cend(); }

    /// The number of externs.
    std::size_t size() const noexcept { return entries_.size(); }
    /// Whether there are no externs.
    bool empty() const noexcept { return entries_.empty(); }

    /// The extern at `index`.
    const Extern& operator[](std::size_t index) const { return entries_[index]; }
    /// The extern at `index`; throws `std::out_of_range` if there is none.
    const Extern& at(std::size_t index) const { return entries_.at(index); }

    /// The extern named `name`, if any.
    std::optional<Extern> get(std::string_view name) const {
        for (const auto& item : entries_) {
            if (item.name == name) return item;
        }
        return std::nullopt;
    }
    /// Whether an extern named `name` exists.
    bool has(std::string_view name) const {
        for (const auto& item : entries_) {
            if (item.name == name) return true;
        }
        return false;
    }

    /// Only the externs that are not yet bound.
    Externs unbound() const {
        if (!unbound_) return Externs{};
        return Externs(*unbound_, {});
    }
};

/// Options for compiling a grammar or module.
struct CompileOptions {
    /// How masks are computed. Defaults to `MaskStrategy::Auto`.
    MaskStrategy mask_strategy = MaskStrategy::Auto;
    /// How masks are computed at grammar boundaries. Defaults to `BoundaryMaskStrategy::Auto`.
    BoundaryMaskStrategy boundary_mask_strategy = BoundaryMaskStrategy::Auto;
    /// Whether `MaskStrategy::TokenTrie` parses with template DFAs instead of an LR table.
    /// Slower to compile, with faster masks. Off by default.
    bool template_dfas = false;
    /// Whether `MaskStrategy::TokenTrie` merges tokens that the grammar treats the same, so
    /// masks walk fewer tokens. Requires `template_dfas`. Off by default.
    bool merge_equivalent_tokens = false;
    /// The equivalent C struct.
    glrmask_compile_options native() const noexcept {
        return {sizeof(glrmask_compile_options),
                static_cast<std::uint32_t>(mask_strategy),
                static_cast<std::uint32_t>(boundary_mask_strategy),
                static_cast<std::uint32_t>(template_dfas),
                static_cast<std::uint32_t>(merge_equivalent_tokens)};
    }
};

/// One vocabulary entry: a token ID and its exact bytes.
struct VocabEntry {
    /// The token ID.
    TokenId id;
    /// The token's exact bytes.
    Bytes bytes;
    /// Creates an entry.
    VocabEntry(TokenId id_, Bytes bytes_) : id(id_), bytes(std::move(bytes_)) {}
    /// Creates an entry.
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

/// A set of token IDs.
///
/// In a grammar, it matches any one of its tokens. Every ID must be in the
/// vocabulary.
class TokenSet : public detail::Owned<glrmask_token_set, glrmask_token_set_free, glrmask_token_set_clone> {
    using Base = detail::Owned<glrmask_token_set, glrmask_token_set_free, glrmask_token_set_clone>;
    explicit TokenSet(glrmask_token_set* p) : Base(p) {}
public:
    /// Creates a set from `ids`.
    explicit TokenSet(const std::vector<TokenId>& ids) : Base(create_ids(ids)) {}
    /// Creates a set from inclusive `ranges`.
    static TokenSet from_ranges(const std::vector<TokenRange>& ranges) {
        glrmask_token_set* p = nullptr; detail::check(glrmask_token_set_from_ranges(ranges.data(), ranges.size(), &p)); return TokenSet(p);
    }
private:
    static glrmask_token_set* create_ids(const std::vector<TokenId>& ids) {
        glrmask_token_set* p = nullptr; detail::check(glrmask_token_set_new(ids.data(), ids.size(), &p)); return p;
    }
};

/// A model's vocabulary: each token ID and its bytes.
///
/// Use each token's exact bytes, not its display form: `b" hello"`, not
/// `"Ġhello"`. IDs need not be contiguous.
class Vocab : public detail::Owned<glrmask_vocab, glrmask_vocab_free, glrmask_vocab_clone> {
    using Base = detail::Owned<glrmask_vocab, glrmask_vocab_free, glrmask_vocab_clone>;
    static glrmask_vocab* create(const std::vector<VocabEntry>& entries, const std::vector<TokenId>& special) {
        std::vector<glrmask_vocab_entry> native; native.reserve(entries.size());
        for (const auto& e : entries) native.push_back({e.id, e.bytes.data(), e.bytes.size()});
        glrmask_vocab* p = nullptr;
        detail::check(glrmask_vocab_new(native.data(), native.size(), special.data(), special.size(), &p)); return p;
    }
public:
    /// Creates a vocabulary from `entries`, plus special tokens that have no
    /// bytes.
    explicit Vocab(const std::vector<VocabEntry>& entries, const std::vector<TokenId>& special = {}) : Base(create(entries, special)) {}
    /// The number of tokens with bytes; special tokens are not counted.
    std::size_t size() const { std::size_t n = 0; detail::check(glrmask_vocab_len(native_handle(), &n)); return n; }
};

/// A grammar, written as source text or built from an `Expr`.
class Grammar : public detail::Owned<glrmask_grammar, glrmask_grammar_free, glrmask_grammar_clone> {
    using Base = detail::Owned<glrmask_grammar, glrmask_grammar_free, glrmask_grammar_clone>;
    explicit Grammar(glrmask_grammar* p) : Base(p) {}
    static Grammar source(std::uint32_t format, std::string_view text) {
        glrmask_grammar* p = nullptr; detail::check(glrmask_grammar_from_source(format, text.data(), text.size(), &p)); return Grammar(p);
    }
public:
    /// Creates a grammar from EBNF source.
    static Grammar from_ebnf(std::string_view text) { return source(GLRMASK_EBNF, text); }
    /// Creates a grammar from GLRM, GLRMask's native format.
    static Grammar from_glrm(std::string_view text) { return source(GLRMASK_GLRM, text); }
    /// Creates a grammar from a JSON Schema document.
    ///
    /// Standard keywords that GLRMask does not support are an error. Unknown
    /// keywords and `format` are ignored.
    static Grammar from_json_schema(std::string_view text) { return source(GLRMASK_JSON_SCHEMA, text); }
    /// Creates a grammar from Lark source.
    ///
    /// A token name such as `<|end|>` declares a token extern, which must be bound
    /// before compiling.
    static Grammar from_lark(std::string_view text) { return source(GLRMASK_LARK, text); }
    /// Creates a grammar from an expression.
    ///
    /// `Grammar::from_expr(Expr::sequence({grammar, eos_id}))`
    static Grammar from_expr(const Expr& value);
    /// Binds the extern `name` to a grammar.
    void bind(std::string_view name, const Grammar& value) {
        detail::check(glrmask_grammar_bind_grammar(native_handle(), name.data(), name.size(), value.native_handle()));
    }
    /// Binds the extern `name` to an expression.
    void bind(std::string_view name, const Expr& value);
    /// Binds the extern `name` to a token ID.
    void bind(std::string_view name, TokenId value) {
        detail::check(glrmask_grammar_bind_token(native_handle(), name.data(), name.size(), value));
    }
    /// Binds the extern `name` to a token set.
    void bind(std::string_view name, const TokenSet& value) {
        detail::check(glrmask_grammar_bind_tokens(native_handle(), name.data(), name.size(), value.native_handle()));
    }
    /// The grammar's externs, in declaration order.
    Externs externs() const {
        glrmask_externs* p = nullptr;
        detail::check(glrmask_grammar_externs(native_handle(), &p));
        return Externs(detail::take_externs(p));
    }
    /// Compiles the grammar for `vocab`.
    Constraint compile(const Vocab& vocab, const CompileOptions& options = {}) const;
    /// Compiles as much as possible for `vocab` now, and returns a `Module` that
    /// keeps the work.
    ///
    /// Externs may still be unbound.
    Module precompile(const Vocab& vocab, const CompileOptions& options = {}) const;
};

/// One part of an expression: a grammar, module, expression, token set or token
/// ID.
///
/// It refers to the object, which must outlive the operand.
struct ExprOperand {
    glrmask_expr_operand value;
    /// From a grammar.
    ExprOperand(const Grammar&);
    /// From a module.
    ExprOperand(const Module&);
    /// From an expression.
    ExprOperand(const Expr&);
    /// From a token ID.
    ExprOperand(TokenId);
    /// From a token set.
    ExprOperand(const TokenSet&);
};

/// A choice or sequence of grammars, modules, token sets and literals.
///
/// Each grammar keeps its own rules, including which whitespace it ignores.
class Expr : public detail::Owned<glrmask_expr, glrmask_expr_free, glrmask_expr_clone> {
    using Base = detail::Owned<glrmask_expr, glrmask_expr_free, glrmask_expr_clone>;
    explicit Expr(glrmask_expr* p) : Base(p) {}
public:
    /// Matches any one of `values`.
    static Expr choice(const std::vector<ExprOperand>& values) {
        std::vector<glrmask_expr_operand> items; items.reserve(values.size());
        for (const auto& value : values) items.push_back(value.value);
        glrmask_expr* p = nullptr; detail::check(glrmask_expr_choice(items.data(), items.size(), &p)); return Expr(p);
    }
    /// Matches `values` one after another.
    ///
    /// One token may span several values.
    static Expr sequence(const std::vector<ExprOperand>& values) {
        std::vector<glrmask_expr_operand> items; items.reserve(values.size());
        for (const auto& value : values) items.push_back(value.value);
        glrmask_expr* p = nullptr; detail::check(glrmask_expr_sequence(items.data(), items.size(), &p)); return Expr(p);
    }
    /// Matches exactly these bytes, with no whitespace skipped.
    static Expr literal(const std::uint8_t* data, std::size_t len) {
        glrmask_expr* p = nullptr; detail::check(glrmask_expr_literal(data, len, &p)); return Expr(p);
    }
    /// Matches exactly these bytes, with no whitespace skipped.
    static Expr literal(std::string_view bytes) {
        return literal(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
    }
    /// Matches exactly these bytes, with no whitespace skipped.
    static Expr literal(const Bytes& bytes) {
        return literal(bytes.data(), bytes.size());
    }
};

inline Grammar Grammar::from_expr(const Expr& value) {
    glrmask_grammar* p = nullptr; detail::check(glrmask_grammar_from_expr(value.native_handle(), &p)); return Grammar(p);
}
inline void Grammar::bind(std::string_view name, const Expr& value) {
    detail::check(glrmask_grammar_bind_expr(native_handle(), name.data(), name.size(), value.native_handle()));
}

/// A grammar you can compose with others and partly compile before its externs are bound.
class Module : public detail::Owned<glrmask_module, glrmask_module_free, glrmask_module_clone> {
    using Base = detail::Owned<glrmask_module, glrmask_module_free, glrmask_module_clone>;
    explicit Module(glrmask_module* p) : Base(p) {}
    friend class Grammar;
    static glrmask_module* create(const Grammar& grammar) {
        glrmask_module* p = nullptr; detail::check(glrmask_module_new(grammar.native_handle(), &p)); return p;
    }
public:
    /// Creates a module from `grammar`.
    explicit Module(const Grammar& grammar) : Base(create(grammar)) {}

    /// Binds the extern `name` to a module. Throws `Error` if `name` is
    /// undeclared or already bound, or the value is the wrong kind for it.
    void bind(std::string_view name, const Module& value) {
        detail::check(glrmask_module_bind_module(native_handle(), name.data(), name.size(), value.native_handle()));
    }
    /// Binds the extern `name` to a grammar.
    void bind(std::string_view name, const Grammar& value) {
        detail::check(glrmask_module_bind_grammar(native_handle(), name.data(), name.size(), value.native_handle()));
    }
    /// Binds the extern `name` to an expression.
    void bind(std::string_view name, const Expr& value) {
        detail::check(glrmask_module_bind_expr(native_handle(), name.data(), name.size(), value.native_handle()));
    }
    /// Binds the extern `name` to a token ID.
    void bind(std::string_view name, TokenId value) {
        detail::check(glrmask_module_bind_token(native_handle(), name.data(), name.size(), value));
    }
    /// Binds the extern `name` to a token set.
    void bind(std::string_view name, const TokenSet& value) {
        detail::check(glrmask_module_bind_tokens(native_handle(), name.data(), name.size(), value.native_handle()));
    }

    /// Compiles as much as possible for `vocab` now, so later compiles are fast.
    ///
    /// Externs may still be unbound. The work is kept in the module and included by
    /// `save`.
    void precompile(const Vocab& vocab, const CompileOptions& options = {}) {
        const glrmask_compile_options native = options.native();
        detail::check(glrmask_module_precompile(native_handle(), vocab.native_handle(), &native));
    }
    /// Compiles the module for `vocab`, reusing any precompiled work.
    ///
    /// Every extern must be bound.
    Constraint compile(const Vocab& vocab, const CompileOptions& options = {}) const;

    /// The module's externs, in declaration order.
    Externs externs() const {
        glrmask_externs* p_all = nullptr;
        detail::check(glrmask_module_externs(native_handle(), &p_all));
        auto all_items = detail::take_externs(p_all);

        glrmask_externs* p_unbound = nullptr;
        detail::check(glrmask_module_unbound_externs(native_handle(), &p_unbound));
        auto unbound_items = detail::take_externs(p_unbound);

        return Externs(std::move(all_items), std::move(unbound_items));
    }

    /// Saves the module, including precompiled work, as bytes.
    ///
    /// The vocabulary is not included. Bytes saved by one GLRMask version may not
    /// load in another.
    Bytes save() const {
        glrmask_bytes* p = nullptr; detail::check(glrmask_module_save(native_handle(), &p)); return detail::take_bytes(p);
    }
    /// Loads a module saved with `save`.
    ///
    /// Precompiled work applies only to the vocabulary it was built for; compiling
    /// with a different one is an error.
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

/// A grammar compiled for a vocabulary, ready to constrain generation.
///
/// Share one between threads, and call `start` for each sequence.
class Constraint : public detail::Owned<glrmask_constraint, glrmask_constraint_free, glrmask_constraint_clone> {
    using Base = detail::Owned<glrmask_constraint, glrmask_constraint_free, glrmask_constraint_clone>;
    explicit Constraint(glrmask_constraint* p) : Base(p) {}
    friend class Grammar; friend class Module;
public:
    /// Starts a new sequence.
    ConstraintState start() const;
    /// The number of `uint32_t` words in a mask.
    std::size_t mask_len() const { std::size_t n = 0; detail::check(glrmask_constraint_mask_len(native_handle(), &n)); return n; }
    /// The IDs of the grammar's markers, in ascending order.
    std::vector<TokenId> marker_ids() const { glrmask_u32s* p = nullptr; detail::check(glrmask_constraint_marker_ids(native_handle(), &p)); return detail::take_ids(p); }
    /// Saves the constraint as bytes, without the vocabulary.
    ///
    /// Bytes saved by one GLRMask version may not load in another.
    Bytes save() const { glrmask_bytes* p = nullptr; detail::check(glrmask_constraint_save(native_handle(), &p)); return detail::take_bytes(p); }
    /// Loads a constraint saved with `save`.
    ///
    /// `vocab` must be the vocabulary it was compiled for; a different one is an
    /// error.
    static Constraint load(const Bytes& bytes, const Vocab& vocab) {
        glrmask_constraint* p = nullptr; detail::check(glrmask_constraint_load(bytes.data(), bytes.size(), vocab.native_handle(), &p)); return Constraint(p);
    }
};

/// The state of one generated sequence.
///
/// Copy it to try tokens without changing the original.
class ConstraintState : public detail::Owned<glrmask_state, glrmask_state_free, glrmask_state_clone> {
    using Base = detail::Owned<glrmask_state, glrmask_state_free, glrmask_state_clone>;
    explicit ConstraintState(glrmask_state* p) : Base(p) {}
    friend class Constraint;
    static bool commit_status(glrmask_status status) {
        if (status == GLRMASK_OK) return true;
        if (status == GLRMASK_REJECTED) return false;
        detail::check(status);
        return false;
    }
public:
    /// The number of `uint32_t` words in a mask.
    std::size_t mask_len() const { std::size_t n = 0; detail::check(glrmask_state_mask_len(native_handle(), &n)); return n; }
    /// Writes the mask into `output`, setting extra words to zero.
    ///
    /// Throws `Error` if `word_count` is less than `mask_len()`.
    void fill_mask(std::uint32_t* output, std::size_t word_count) const { detail::check(glrmask_state_fill_mask(native_handle(), output, word_count)); }
    /// Writes the mask into `output`, setting extra words to zero.
    void fill_mask(std::vector<std::uint32_t>& output) const { fill_mask(output.data(), output.size()); }
    /// Whether the input committed so far is a complete match for the grammar.
    ///
    /// More input may still be allowed; see `is_terminated`.
    bool is_accepting() const { std::uint32_t value = 0; detail::check(glrmask_state_is_accepting(native_handle(), &value)); return value != 0; }
    /// The tokens allowed next, as a bitset.
    ///
    /// Token `id` is allowed when bit `id % 32` of word `id / 32` is set. No bits
    /// are set when nothing can follow: the sequence is complete, or a commit
    /// failed.
    std::vector<std::uint32_t> mask() const { std::vector<std::uint32_t> out(mask_len()); fill_mask(out); return out; }
    /// Whether nothing more can be committed.
    ///
    /// This is true when the sequence is finished, for example after its end token,
    /// and after a failed commit. `is_accepting` tells the two apart.
    ///
    /// A complete match is not terminated if more input could follow, such as
    /// whitespace the grammar ignores.
    bool is_terminated() const { std::uint32_t value = 0; detail::check(glrmask_state_is_terminated(native_handle(), &value)); return value != 0; }
    /// Commits one token.
    ///
    /// Returns `false` if it is not allowed, leaving the state unusable. To try a
    /// token safely, commit it on a clone.
    [[nodiscard]] bool commit_token(TokenId id) { return commit_status(glrmask_state_commit_token(native_handle(), id)); }
    /// Commits the tokens in order.
    ///
    /// Returns `false` at the first token that is not allowed, leaving the state
    /// unusable.
    [[nodiscard]] bool commit_tokens(const std::uint32_t* ids, std::size_t count) { return commit_status(glrmask_state_commit_tokens(native_handle(), ids, count)); }
    /// Commits the tokens in order.
    ///
    /// Returns `false` at the first token that is not allowed, leaving the state
    /// unusable.
    bool commit_tokens(const std::vector<std::uint32_t>& ids) { return commit_tokens(ids.data(), ids.size()); }
    /// Commits raw bytes instead of a token.
    ///
    /// Returns `false` if they are not allowed, leaving the state unusable. Bytes
    /// never match a token set; commit those tokens with `commit_token`.
    [[nodiscard]] bool commit_bytes(const std::uint8_t* bytes, std::size_t size) { return commit_status(glrmask_state_commit_bytes(native_handle(), bytes, size)); }
    /// Commits raw bytes instead of a token.
    ///
    /// Returns `false` if they are not allowed, leaving the state unusable.
    bool commit_bytes(const Bytes& bytes) { return commit_bytes(bytes.data(), bytes.size()); }
    /// Commits raw bytes instead of a token.
    ///
    /// Returns `false` if they are not allowed, leaving the state unusable.
    bool commit_bytes(std::string_view bytes) { return commit_bytes(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()); }
    /// The tokens that must come next, without committing them.
    ///
    /// Stops where more than one token could follow, or after 10,000 tokens or
    /// bytes. When forced text could be tokenized in several ways, the longest
    /// token is chosen at each step.
    std::vector<TokenId> forced_tokens() const { glrmask_u32s* p = nullptr; detail::check(glrmask_state_forced_tokens(native_handle(), &p)); return detail::take_ids(p); }
    /// The markers passed by the most recent commit, or by `start`, in byte order.
    ///
    /// A marker means some parse reached that point. That parse may still fail
    /// later.
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
/// The GLRMask version.
inline const char* version() noexcept { return glrmask_version(); }
} // namespace glrmask
#endif
