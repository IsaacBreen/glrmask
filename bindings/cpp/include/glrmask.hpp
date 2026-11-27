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
using TokenId = std::uint32_t;
/// A range of token IDs. `start` is included and `end` is not.
using TokenRange = glrmask_token_range;
using Bytes = std::vector<std::uint8_t>;

/// Controls how the constraint is compiled.
enum class MaskStrategy : std::uint32_t {
    /// The default; currently `FastMasks`.
    Auto = 0,
    /// Faster compilation, slower masks.
    FastCompile = 3,
    /// Faster masks, slower compilation.
    FastMasks = 4
};

/// What an extern can be bound to.
enum class ExternKind : std::uint32_t {
    /// A token set.
    Token = GLRMASK_EXTERN_TOKEN,
    /// A grammar, module, expression or token set.
    Grammar = GLRMASK_EXTERN_GRAMMAR
};

class Error : public std::runtime_error {
    glrmask_status status_;
public:
    Error(glrmask_status status, const std::string& message) : std::runtime_error(message), status_(status) {}
    /// One of `GLRMASK_SYNTAX`, `GLRMASK_INVALID_INPUT`, `GLRMASK_COMPILATION`, `GLRMASK_RESOURCE_LIMIT`, `GLRMASK_STRATEGY_UNSUPPORTED`, `GLRMASK_SERIALIZATION`, `GLRMASK_VOCAB_MISMATCH`, `GLRMASK_INTERNAL`, `GLRMASK_BUFFER_TOO_SMALL`, `GLRMASK_CLOSED`.
    glrmask_status status() const noexcept { return status_; }
};

/// A named placeholder in a grammar.
class Extern {
    std::string name_;
    ExternKind kind_;
public:
    Extern(std::string name, ExternKind kind) : name_(std::move(name)), kind_(kind) {}
    const std::string& name() const noexcept { return name_; }
    ExternKind kind() const noexcept { return kind_; }
    bool operator==(const Extern& other) const { return name_ == other.name_ && kind_ == other.kind_; }
    bool operator!=(const Extern& other) const { return !(*this == other); }
};

/// A read-only mapping of extern names to `Extern`.
class Externs {
    std::vector<Extern> entries_;
    std::shared_ptr<std::vector<Extern>> unbound_;
    friend class Grammar;
    friend class Module;
    Externs() : entries_{}, unbound_{std::make_shared<std::vector<Extern>>()} {}
    explicit Externs(std::vector<Extern> entries, std::vector<Extern> unbound = {})
        : entries_(std::move(entries)), unbound_(std::make_shared<std::vector<Extern>>(std::move(unbound))) {}

public:
    using const_iterator = std::vector<Extern>::const_iterator;
    const_iterator begin() const noexcept { return entries_.begin(); }
    const_iterator end() const noexcept { return entries_.end(); }
    const_iterator cbegin() const noexcept { return entries_.cbegin(); }
    const_iterator cend() const noexcept { return entries_.cend(); }

    std::size_t size() const noexcept { return entries_.size(); }
    bool empty() const noexcept { return entries_.empty(); }


    std::optional<Extern> get(std::string_view name) const {
        for (const auto& item : entries_) {
            if (item.name() == name) return item;
        }
        return std::nullopt;
    }
    bool has(std::string_view name) const {
        for (const auto& item : entries_) {
            if (item.name() == name) return true;
        }
        return false;
    }

    Externs unbound() const {
        if (!unbound_) return Externs{};
        return Externs(*unbound_, *unbound_);
    }
    bool operator==(const Externs& other) const {
        if (entries_ != other.entries_) return false;
        if (unbound_ && other.unbound_) return *unbound_ == *other.unbound_;
        return (!unbound_ || unbound_->empty()) && (!other.unbound_ || other.unbound_->empty());
    }
    bool operator!=(const Externs& other) const { return !(*this == other); }
};

/// Options for compiling a grammar or module.
struct CompileOptions {
    /// Defaults to `MaskStrategy::Auto`.
    MaskStrategy mask_strategy = MaskStrategy::Auto;
    glrmask_compile_options native() const noexcept {
        glrmask_compile_options options{};
        (void)glrmask_compile_options_init(&options, sizeof(options));
        options.mask_strategy = static_cast<std::uint32_t>(mask_strategy);
        return options;
    }
};

/// One vocabulary entry: a token ID and its exact bytes.
struct VocabEntry {
    TokenId id;
    Bytes bytes;
    VocabEntry(TokenId id_, Bytes bytes_) : id(id_), bytes(std::move(bytes_)) {}
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
    /// Borrows the live native handle; do not free it or retain it beyond this object.
    H* native_handle() const {
        if (!handle_) throw Error(GLRMASK_CLOSED, "object has been moved from");
        return handle_.get();
    }
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
        const char* name = nullptr; std::size_t n = 0; std::uint32_t kind = 0;
        check(glrmask_externs_name(raw, i, &name, &n));
        check(glrmask_externs_kind(raw, i, &kind));
        result.emplace_back(std::string(name, n), static_cast<ExternKind>(kind));
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
    explicit TokenSet(const std::vector<TokenId>& ids) : Base(create_ids(ids)) {}
    static TokenSet from_ranges(const std::vector<TokenRange>& ranges) {
        glrmask_token_set* p = nullptr; detail::check(glrmask_token_set_from_ranges(ranges.data(), ranges.size(), &p)); return TokenSet(p);
    }
    friend bool operator==(const TokenSet& a, const TokenSet& b) {
        std::uint32_t equal = 0;
        detail::check(glrmask_token_set_equal(a.native_handle(), b.native_handle(), &equal));
        return equal != 0;
    }
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
    /// Parses [GLRM](https://github.com/IsaacBreen/glrmask/blob/main/docs/glrm-format.md) source.
    static Grammar from_glrm(std::string_view text) { return source(GLRMASK_GLRM, text); }
    /// Parses [Lark](https://github.com/IsaacBreen/glrmask/blob/main/docs/lark-dialect.md) source.
    static Grammar from_lark(std::string_view text) { return source(GLRMASK_LARK, text); }
    /// Parses a JSON Schema given as JSON text.
    static Grammar from_json_schema(std::string_view text) { return source(GLRMASK_JSON_SCHEMA, text); }

    /// Builds a grammar from an `Expr`.
    static Grammar from_expr(const Expr& value);
    Externs externs() const {
        glrmask_externs* p_all = nullptr;
        detail::check(glrmask_grammar_externs(native_handle(), &p_all));
        auto all_items = detail::take_externs(p_all);

        glrmask_externs* p_unbound = nullptr;
        detail::check(glrmask_grammar_unbound_externs(native_handle(), &p_unbound));
        auto unbound_items = detail::take_externs(p_unbound);

        return Externs(std::move(all_items), std::move(unbound_items));
    }
    /// Compiles the grammar into a constraint.
    Constraint compile(const Vocab& vocab, const CompileOptions& options = {}) const;
};

/// One part of an expression: a grammar, module, expression or token set.
///
/// It refers to the object, which must outlive the operand.
struct ExprOperand {
    glrmask_expr_operand value;
    ExprOperand(const Grammar&);
    ExprOperand(const Module&);
    ExprOperand(const Expr&);
    ExprOperand(const TokenSet&);
};

/// A grammar expression.
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
    /// Matches `values` in order.
    static Expr sequence(const std::vector<ExprOperand>& values) {
        std::vector<glrmask_expr_operand> items; items.reserve(values.size());
        for (const auto& value : values) items.push_back(value.value);
        glrmask_expr* p = nullptr; detail::check(glrmask_expr_sequence(items.data(), items.size(), &p)); return Expr(p);
    }
    /// Matches bytes exactly.
    static Expr literal(const std::uint8_t* data, std::size_t len) {
        glrmask_expr* p = nullptr; detail::check(glrmask_expr_literal(data, len, &p)); return Expr(p);
    }
    /// Matches bytes exactly.
    static Expr literal(std::string_view bytes) {
        return literal(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
    }
    /// Matches bytes exactly.
    static Expr literal(const Bytes& bytes) {
        return literal(bytes.data(), bytes.size());
    }
};

inline Grammar Grammar::from_expr(const Expr& value) {
    glrmask_grammar* p = nullptr; detail::check(glrmask_grammar_from_expr(value.native_handle(), &p)); return Grammar(p);
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

    Externs externs() const {
        glrmask_externs* p_all = nullptr;
        detail::check(glrmask_module_externs(native_handle(), &p_all));
        auto all_items = detail::take_externs(p_all);

        glrmask_externs* p_unbound = nullptr;
        detail::check(glrmask_module_unbound_externs(native_handle(), &p_unbound));
        auto unbound_items = detail::take_externs(p_unbound);

        return Externs(std::move(all_items), std::move(unbound_items));
    }
    /// Fails if the extern is undeclared, already bound or the wrong kind.
    void bind(std::string_view name, const Module& value) {
        detail::check(glrmask_module_bind_module(native_handle(), name.data(), name.size(), value.native_handle()));
    }
    /// Fails if the extern is undeclared, already bound or the wrong kind.
    void bind(std::string_view name, const Grammar& value) {
        detail::check(glrmask_module_bind_grammar(native_handle(), name.data(), name.size(), value.native_handle()));
    }
    /// Fails if the extern is undeclared, already bound or the wrong kind.
    void bind(std::string_view name, const Expr& value) {
        detail::check(glrmask_module_bind_expr(native_handle(), name.data(), name.size(), value.native_handle()));
    }
    /// Fails if the extern is undeclared, already bound or the wrong kind.
    void bind(std::string_view name, const TokenSet& value) {
        detail::check(glrmask_module_bind_tokens(native_handle(), name.data(), name.size(), value.native_handle()));
    }

    /// Compiles what it can now, even with externs unbound. Further compilation must use the same `vocab`.
    void precompile(const Vocab& vocab, const CompileOptions& options = {}) {
        const glrmask_compile_options native = options.native();
        detail::check(glrmask_module_precompile(native_handle(), vocab.native_handle(), &native));
    }
    /// Finishes compiling the module into a constraint.
    Constraint compile(const Vocab& vocab, const CompileOptions& options = {}) const;


    Bytes save() const {
        glrmask_bytes* p = nullptr; detail::check(glrmask_module_save(native_handle(), &p)); return detail::take_bytes(p);
    }

    /// May fail if saved by a different GLRMask version.
    static Module load(const Bytes& bytes) { return load(bytes.data(), bytes.size()); }
    static Module load(const std::uint8_t* bytes, std::size_t size) {
        glrmask_module* p = nullptr; detail::check(glrmask_module_load(bytes, size, &p)); return Module(p);
    }
};

inline ExprOperand::ExprOperand(const Grammar& value) : value{0, value.native_handle()} {}
inline ExprOperand::ExprOperand(const Module& value) : value{1, value.native_handle()} {}
inline ExprOperand::ExprOperand(const Expr& value) : value{2, value.native_handle()} {}
inline ExprOperand::ExprOperand(const TokenSet& value) : value{4, value.native_handle()} {}

class Constraint : public detail::Owned<glrmask_constraint, glrmask_constraint_free, glrmask_constraint_clone> {
    using Base = detail::Owned<glrmask_constraint, glrmask_constraint_free, glrmask_constraint_clone>;
    explicit Constraint(glrmask_constraint* p) : Base(p) {}
    friend class Grammar; friend class Module;
public:
    /// Starts a new sequence.
    ConstraintState start() const;
    /// The mask's length in bits (the largest token ID plus one).
    std::size_t mask_bit_len() const { std::size_t n = 0; detail::check(glrmask_constraint_mask_bit_len(native_handle(), &n)); return n; }

    Bytes save() const { glrmask_bytes* p = nullptr; detail::check(glrmask_constraint_save(native_handle(), &p)); return detail::take_bytes(p); }
    /// May fail if saved by a different GLRMask version. Fails if `vocab` differs from the one it was compiled with.
    static Constraint load(const Bytes& bytes, const Vocab& vocab) { return load(bytes.data(), bytes.size(), vocab); }
    static Constraint load(const std::uint8_t* bytes, std::size_t size, const Vocab& vocab) {
        glrmask_constraint* p = nullptr; detail::check(glrmask_constraint_load(bytes, size, vocab.native_handle(), &p)); return Constraint(p);
    }
};

class ConstraintState : public detail::Owned<glrmask_state, glrmask_state_free, glrmask_state_clone> {
    using Base = detail::Owned<glrmask_state, glrmask_state_free, glrmask_state_clone>;
    explicit ConstraintState(glrmask_state* p) : Base(p) {}
    friend class Constraint;
public:
    /// Which tokens are allowed next, as a bitmask: token `i` is bit `i % 32` of word `i / 32`.
    std::vector<std::uint32_t> mask() const { std::vector<std::uint32_t> out((mask_bit_len() + 31) / 32); fill_mask(out); return out; }
    /// Like `mask()`, but writes the words into `output`.
    ///
    /// Throws `Error` if `word_count` is less than `(mask_bit_len() + 31) / 32`.
    void fill_mask(std::uint32_t* output, std::size_t word_count) const { detail::check(glrmask_state_fill_mask(native_handle(), output, word_count)); }
    /// Like `mask()`, but writes the words into `output`.
    void fill_mask(std::vector<std::uint32_t>& output) const { fill_mask(output.data(), output.size()); }
    /// The mask's length in bits (the largest token ID plus one).
    std::size_t mask_bit_len() const { std::size_t n = 0; detail::check(glrmask_state_mask_bit_len(native_handle(), &n)); return n; }
    /// Consumes the input. Returns true if the state is viable after the commit (this does not imply it is accepting); false means the state is dead, not that the commit was rolled back. A dead state is not accepting, is terminated, has an all-zero mask and empty forced bytes, and every later non-empty commit returns false. A dead state can also arise from a grammar whose language is empty. To try input without risking the state, clone it first.
    [[nodiscard]] bool commit_token(TokenId id) { uint32_t accepted = 0; detail::check(glrmask_state_commit_token(native_handle(), id, &accepted)); return accepted != 0; }
    /// Consumes the input. Returns true if the state is viable after the commit (this does not imply it is accepting); false means the state is dead, not that the commit was rolled back. A dead state is not accepting, is terminated, has an all-zero mask and empty forced bytes, and every later non-empty commit returns false. A dead state can also arise from a grammar whose language is empty. To try input without risking the state, clone it first. Committing empty bytes leaves the state unchanged and returns whether it is viable.
    [[nodiscard]] bool commit_bytes(const std::uint8_t* bytes, std::size_t size) { uint32_t accepted = 0; detail::check(glrmask_state_commit_bytes(native_handle(), bytes, size, &accepted)); return accepted != 0; }
    /// Consumes the input. Returns true if the state is viable after the commit (this does not imply it is accepting); false means the state is dead, not that the commit was rolled back. A dead state is not accepting, is terminated, has an all-zero mask and empty forced bytes, and every later non-empty commit returns false. A dead state can also arise from a grammar whose language is empty. To try input without risking the state, clone it first. Committing empty bytes leaves the state unchanged and returns whether it is viable.
    [[nodiscard]] bool commit_bytes(const Bytes& bytes) { return commit_bytes(bytes.data(), bytes.size()); }
    /// Consumes the input. Returns true if the state is viable after the commit (this does not imply it is accepting); false means the state is dead, not that the commit was rolled back. A dead state is not accepting, is terminated, has an all-zero mask and empty forced bytes, and every later non-empty commit returns false. A dead state can also arise from a grammar whose language is empty. To try input without risking the state, clone it first. Committing empty bytes leaves the state unchanged and returns whether it is viable.
    [[nodiscard]] bool commit_bytes(std::string_view bytes) { return commit_bytes(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()); }
    /// The bytes that must come next.
    Bytes forced_bytes() const { glrmask_bytes* p = nullptr; detail::check(glrmask_state_forced_bytes(native_handle(), &p)); return detail::take_bytes(p); }
    /// Whether the output so far could end here.
    bool is_accepting() const { std::uint32_t value = 0; detail::check(glrmask_state_is_accepting(native_handle(), &value)); return value != 0; }
    /// Whether nothing more can be committed.
    bool is_terminated() const { std::uint32_t value = 0; detail::check(glrmask_state_is_terminated(native_handle(), &value)); return value != 0; }
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
inline const char* version() noexcept { return glrmask_version(); }
} // namespace glrmask
#endif
