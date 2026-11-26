#ifndef GLRMASK_HPP
#define GLRMASK_HPP
// GLRMask C++17 API. It wraps the C API in glrmask.h.
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
/// A range of token IDs. `start` is included and `end` is not.
using TokenRange = glrmask_token_range;
/// A byte buffer.
using Bytes = std::vector<std::uint8_t>;

/// A preference for faster compiling or faster masks.
enum class MaskStrategy : std::uint32_t {
    Auto = GLRMASK_MASK_STRATEGY_AUTO,
    FastCompile = GLRMASK_MASK_STRATEGY_FAST_COMPILE,
    FastMasks = GLRMASK_MASK_STRATEGY_FAST_MASKS
};

/// What an extern can be bound to.
enum class ExternKind : std::uint32_t {
    /// A token set.
    Token = GLRMASK_EXTERN_TOKEN,
    /// A grammar, module, expression or token set.
    Grammar = GLRMASK_EXTERN_GRAMMAR
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

/// A named placeholder in a grammar.
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
    /// The equivalent C struct.
    glrmask_compile_options native() const noexcept {
        glrmask_compile_options options{};
        (void)glrmask_compile_options_init(&options, sizeof(options));
        options.mask_strategy = static_cast<std::uint32_t>(mask_strategy);
        return options;
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
inline std::vector<Extern> take_externs(glrmask_externs* raw) {
    std::unique_ptr<glrmask_externs, decltype(&glrmask_externs_free)> owner(raw, glrmask_externs_free);
    std::size_t len = 0; check(glrmask_externs_len(raw, &len));
    std::vector<Extern> result; result.reserve(len);
    for (std::size_t i = 0; i < len; ++i) {
        glrmask_extern value{}; check(glrmask_externs_get(raw, i, &value));
        Extern e; e.name.assign(value.name, value.name_len);
        e.kind = static_cast<ExternKind>(value.kind);
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
class TokenSet : public detail::Owned<glrmask_token_set, glrmask_token_set_free, glrmask_token_set_clone> {
    using Base = detail::Owned<glrmask_token_set, glrmask_token_set_free, glrmask_token_set_clone>;
    explicit TokenSet(glrmask_token_set* p) : Base(p) {}
public:
    /// Creates a set from `ids`.
    explicit TokenSet(const std::vector<TokenId>& ids) : Base(create_ids(ids)) {}
    /// Creates a set from half-open ranges, each including `start` and excluding `end`.
    static TokenSet from_ranges(const std::vector<TokenRange>& ranges) {
        glrmask_token_set* p = nullptr; detail::check(glrmask_token_set_from_ranges(ranges.data(), ranges.size(), &p)); return TokenSet(p);
    }
    /// Whether both sets contain the same IDs.
    friend bool operator==(const TokenSet& a, const TokenSet& b) {
        std::uint32_t equal = 0;
        detail::check(glrmask_token_set_equal(a.native_handle(), b.native_handle(), &equal));
        return equal != 0;
    }
    /// Whether the sets differ.
    friend bool operator!=(const TokenSet& a, const TokenSet& b) { return !(a == b); }
private:
    static glrmask_token_set* create_ids(const std::vector<TokenId>& ids) {
        glrmask_token_set* p = nullptr; detail::check(glrmask_token_set_new(ids.data(), ids.size(), &p)); return p;
    }
};

/// Each token ID and its bytes.
class Vocab : public detail::Owned<glrmask_vocab, glrmask_vocab_free, glrmask_vocab_clone> {
    using Base = detail::Owned<glrmask_vocab, glrmask_vocab_free, glrmask_vocab_clone>;
    static glrmask_vocab* create(const std::vector<VocabEntry>& entries) {
        std::vector<glrmask_vocab_entry> native; native.reserve(entries.size());
        for (const auto& e : entries) native.push_back({e.id, e.bytes.data(), e.bytes.size()});
        glrmask_vocab* p = nullptr;
        detail::check(glrmask_vocab_new(native.data(), native.size(), &p)); return p;
    }
public:

    explicit Vocab(const std::vector<VocabEntry>& entries) : Base(create(entries)) {}
};

class Grammar : public detail::Owned<glrmask_grammar, glrmask_grammar_free, glrmask_grammar_clone> {
    using Base = detail::Owned<glrmask_grammar, glrmask_grammar_free, glrmask_grammar_clone>;
    explicit Grammar(glrmask_grammar* p) : Base(p) {}
    static Grammar source(std::uint32_t format, std::string_view text) {
        glrmask_grammar* p = nullptr; detail::check(glrmask_grammar_from_source(format, text.data(), text.size(), &p)); return Grammar(p);
    }
public:
    /// Parses GLRM source.
    static Grammar from_glrm(std::string_view text) { return source(GLRMASK_GLRM, text); }
    /// Parses JSON Schema source.
    static Grammar from_json_schema(std::string_view text) { return source(GLRMASK_JSON_SCHEMA, text); }
    /// Parses Lark source.
    static Grammar from_lark(std::string_view text) { return source(GLRMASK_LARK, text); }

    static Grammar from_expr(const Expr& value);
    /// Binds the extern `name` to a grammar.
    void bind(std::string_view name, const Grammar& value) {
        detail::check(glrmask_grammar_bind_grammar(native_handle(), name.data(), name.size(), value.native_handle()));
    }
    /// Binds the extern `name` to an expression.
    void bind(std::string_view name, const Expr& value);
    /// The grammar's externs, in declaration order.
    Externs externs() const {
        glrmask_externs* p_all = nullptr;
        detail::check(glrmask_grammar_externs(native_handle(), &p_all));
        auto all_items = detail::take_externs(p_all);

        glrmask_externs* p_unbound = nullptr;
        detail::check(glrmask_grammar_unbound_externs(native_handle(), &p_unbound));
        auto unbound_items = detail::take_externs(p_unbound);

        return Externs(std::move(all_items), std::move(unbound_items));
    }
    /// Compiles the grammar for `vocab`. Fails if any extern is unbound.
    Constraint compile(const Vocab& vocab, const CompileOptions& options = {}) const;
};

/// One part of an expression: a grammar, module, expression or token set.
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
    static Expr sequence(const std::vector<ExprOperand>& values) {
        std::vector<glrmask_expr_operand> items; items.reserve(values.size());
        for (const auto& value : values) items.push_back(value.value);
        glrmask_expr* p = nullptr; detail::check(glrmask_expr_sequence(items.data(), items.size(), &p)); return Expr(p);
    }
    /// Matches bytes literally.
    static Expr literal(const std::uint8_t* data, std::size_t len) {
        glrmask_expr* p = nullptr; detail::check(glrmask_expr_literal(data, len, &p)); return Expr(p);
    }
    /// Matches bytes literally.
    static Expr literal(std::string_view bytes) {
        return literal(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
    }
    /// Matches bytes literally.
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

/// A partially compiled grammar.
class Module : public detail::Owned<glrmask_module, glrmask_module_free, glrmask_module_clone> {
    using Base = detail::Owned<glrmask_module, glrmask_module_free, glrmask_module_clone>;
    explicit Module(glrmask_module* p) : Base(p) {}
    friend class Grammar;
    static glrmask_module* create(const Grammar& grammar) {
        glrmask_module* p = nullptr; detail::check(glrmask_module_new(grammar.native_handle(), &p)); return p;
    }
public:

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
    /// Reuses any precompiled work, which must be for the same vocabulary.
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
/// From a token set.
inline ExprOperand::ExprOperand(const TokenSet& value) : value{4, value.native_handle()} {}

/// A compiled grammar, ready to use.
class Constraint : public detail::Owned<glrmask_constraint, glrmask_constraint_free, glrmask_constraint_clone> {
    using Base = detail::Owned<glrmask_constraint, glrmask_constraint_free, glrmask_constraint_clone>;
    explicit Constraint(glrmask_constraint* p) : Base(p) {}
    friend class Grammar; friend class Module;
public:
    /// Starts a new sequence.
    ConstraintState start() const;
    /// One more than the largest token ID.
    std::size_t mask_bit_len() const { std::size_t n = 0; detail::check(glrmask_constraint_mask_bit_len(native_handle(), &n)); return n; }
    /// Saves the constraint as bytes, without the vocabulary.
    ///
    /// Bytes saved by one GLRMask version may not load in another.
    Bytes save() const { glrmask_bytes* p = nullptr; detail::check(glrmask_constraint_save(native_handle(), &p)); return detail::take_bytes(p); }
    /// Loads a constraint saved with `save()`. Fails if `vocab` isn't the one
    /// it was compiled for.
    static Constraint load(const Bytes& bytes, const Vocab& vocab) {
        glrmask_constraint* p = nullptr; detail::check(glrmask_constraint_load(bytes.data(), bytes.size(), vocab.native_handle(), &p)); return Constraint(p);
    }
};

/// The state of one generated sequence.
class ConstraintState : public detail::Owned<glrmask_state, glrmask_state_free, glrmask_state_clone> {
    using Base = detail::Owned<glrmask_state, glrmask_state_free, glrmask_state_clone>;
    explicit ConstraintState(glrmask_state* p) : Base(p) {}
    friend class Constraint;
public:
    /// One more than the largest token ID.
    std::size_t mask_bit_len() const { std::size_t n = 0; detail::check(glrmask_state_mask_bit_len(native_handle(), &n)); return n; }
    /// Writes the mask into `output`, setting extra words to zero.
    ///
    /// Throws `Error` if `word_count` is less than `(mask_bit_len() + 31) / 32`.
    void fill_mask(std::uint32_t* output, std::size_t word_count) const { detail::check(glrmask_state_fill_mask(native_handle(), output, word_count)); }
    /// Writes the mask into `output`, setting extra words to zero.
    void fill_mask(std::vector<std::uint32_t>& output) const { fill_mask(output.data(), output.size()); }
    /// Whether what's committed is a complete match. If so, allow EOS (don't commit it).
    bool is_accepting() const { std::uint32_t value = 0; detail::check(glrmask_state_is_accepting(native_handle(), &value)); return value != 0; }
    /// Allowed token IDs as a packed bitmask.
    ///
    /// Token `id` is allowed when bit `id % 32` of word `id / 32` is set. No bits
    /// are set when nothing can follow: the sequence is complete, or a commit
    /// failed.
    std::vector<std::uint32_t> mask() const { std::vector<std::uint32_t> out((mask_bit_len() + 31) / 32); fill_mask(out); return out; }
    /// Whether nothing more can be committed.
    bool is_terminated() const { std::uint32_t value = 0; detail::check(glrmask_state_is_terminated(native_handle(), &value)); return value != 0; }
    /// Returns `false`, and ends the state, if the token isn't allowed.
    [[nodiscard]] bool commit_token(TokenId id) { return glrmask_state_commit_token(native_handle(), id); }
    /// Returns how many were committed. Stops at the first token that isn't allowed, which ends the state.
    [[nodiscard]] std::size_t commit_tokens(const std::uint32_t* ids, std::size_t count) { return glrmask_state_commit_tokens(native_handle(), ids, count); }
    /// Returns how many were committed. Stops at the first token that isn't allowed, which ends the state.
    [[nodiscard]] std::size_t commit_tokens(const std::vector<std::uint32_t>& ids) { return commit_tokens(ids.data(), ids.size()); }
    /// Returns `false`, and ends the state, if the bytes aren't allowed.
    [[nodiscard]] bool commit_bytes(const std::uint8_t* bytes, std::size_t size) { return glrmask_state_commit_bytes(native_handle(), bytes, size); }
    /// Returns `false`, and ends the state, if the bytes aren't allowed.
    [[nodiscard]] bool commit_bytes(const Bytes& bytes) { return commit_bytes(bytes.data(), bytes.size()); }
    /// Returns `false`, and ends the state, if the bytes aren't allowed.
    [[nodiscard]] bool commit_bytes(std::string_view bytes) { return commit_bytes(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()); }
    /// The bytes the grammar requires next. Stops where a token from a token set, the end of the grammar, or more than one byte could come next. May return only part of a long run; call again after committing.
    Bytes forced_bytes() const { glrmask_bytes* p = nullptr; detail::check(glrmask_state_forced_bytes(native_handle(), &p)); return detail::take_bytes(p); }
};

inline Constraint Grammar::compile(const Vocab& vocab, const CompileOptions& options) const {
    const auto native = options.native(); glrmask_constraint* p = nullptr;
    detail::check(glrmask_grammar_compile(native_handle(), vocab.native_handle(), &native, &p)); return Constraint(p);
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
