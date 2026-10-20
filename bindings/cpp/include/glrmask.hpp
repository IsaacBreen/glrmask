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
using TokenId = std::uint32_t;
using TokenRange = glrmask_token_range; // inclusive first/last endpoints
using MarkerEvent = glrmask_marker_event;
using Bytes = std::vector<std::uint8_t>;
enum class Optimization : std::uint32_t { Auto=0, FastBuild=1, Balanced=2, FastRuntime=3 };
enum class BoundaryTriggerDetail : std::uint32_t { None=0, Tokens=1, Exact=2 };
enum class LarkProfile : std::uint32_t { Recognition=0, Llm=1 };
enum class RequirementKind { Token, Grammar, Legacy };
class Error : public std::runtime_error {
    glrmask_status status_;
public:
    Error(glrmask_status status, const std::string& message) : std::runtime_error(message), status_(status) {}
    glrmask_status status() const noexcept { return status_; }
};
struct Token { TokenId id; explicit Token(TokenId value) : id(value) {} };
struct SourceSpan { std::uint64_t start_byte, end_byte, line, column; };
struct Requirement {
    std::string name;
    std::vector<std::string> path;
    RequirementKind kind;
    std::optional<SourceSpan> source_span;
};
using UnboundExtern = Requirement; // Legacy name retained for migration.
struct ModuleOptions {
    Optimization optimization=Optimization::Auto;
    BoundaryTriggerDetail boundary_trigger=BoundaryTriggerDetail::None;
    glrmask_module_options native() const noexcept {return {sizeof(glrmask_module_options),static_cast<std::uint32_t>(optimization),static_cast<std::uint32_t>(boundary_trigger)};}
};
struct CompileOptions {
    std::vector<TokenId> end_tokens;
    glrmask_compile_options native() const noexcept {return {sizeof(glrmask_compile_options),end_tokens.data(),end_tokens.size()};}
};
struct VocabEntry {
    TokenId id;
    Bytes bytes;
    VocabEntry(TokenId id_, Bytes bytes_) : id(id_), bytes(std::move(bytes_)) {}
    VocabEntry(TokenId id_, std::string_view bytes_) : id(id_), bytes(bytes_.begin(),bytes_.end()) {}
};
struct BuildOptions {
    Optimization optimization = Optimization::Auto;
    std::vector<TokenId> end_tokens;
    std::optional<BoundaryTriggerDetail> boundary_trigger;
    glrmask_build_options native() const noexcept {
        return {sizeof(glrmask_build_options),static_cast<std::uint32_t>(optimization),
            boundary_trigger ? static_cast<std::uint32_t>(*boundary_trigger) : GLRMASK_BOUNDARY_DEFAULT,
            end_tokens.data(),end_tokens.size()};
    }
};
struct LarkOptions {
    LarkProfile profile = LarkProfile::Recognition;
    std::map<std::string,std::vector<TokenId>> token_names;
};
namespace detail {
inline void check(glrmask_status status) {
    if (status != GLRMASK_OK) throw Error(status,glrmask_error_message());
}
template<class H, void(*Free)(H*), glrmask_status(*Clone)(const H*,H**)>
class Owned {
    std::unique_ptr<H,decltype(Free)> handle_;
protected:
    explicit Owned(H* handle) : handle_(handle,Free) {}
public:
    Owned(const Owned& other) : handle_(nullptr,Free) {
        H* handle=nullptr; check(Clone(other.native_handle(),&handle)); handle_.reset(handle);
    }
    Owned(Owned&&) noexcept = default;
    Owned& operator=(const Owned& other) { Owned copy(other); handle_.swap(copy.handle_); return *this; }
    Owned& operator=(Owned&&) noexcept = default;
    H* native_handle() const {
        if (!handle_) throw Error(GLRMASK_CLOSED,"object has been disposed or moved from");
        return handle_.get();
    }
    void dispose() noexcept { handle_.reset(); }
};
inline Bytes take_bytes(glrmask_bytes* raw) {
    std::unique_ptr<glrmask_bytes,decltype(&glrmask_bytes_free)> owner(raw,glrmask_bytes_free);
    const std::uint8_t* data=nullptr; std::size_t len=0;
    check(glrmask_bytes_data(raw,&data,&len));
    return len ? Bytes(data,data+len) : Bytes{};
}
inline std::vector<TokenId> take_ids(glrmask_u32s* raw) {
    std::unique_ptr<glrmask_u32s,decltype(&glrmask_u32s_free)> owner(raw,glrmask_u32s_free);
    const TokenId* data=nullptr; std::size_t len=0;
    check(glrmask_u32s_data(raw,&data,&len));
    return len ? std::vector<TokenId>(data,data+len) : std::vector<TokenId>{};
}
inline std::vector<Requirement> take_requirements(glrmask_requirements* raw) {
    std::unique_ptr<glrmask_requirements,decltype(&glrmask_requirements_free)> owner(raw,glrmask_requirements_free);
    std::size_t len=0; check(glrmask_requirements_len(raw,&len));
    std::vector<Requirement> result; result.reserve(len);
    for (std::size_t i=0;i<len;++i) {
        glrmask_requirement value{}; check(glrmask_requirements_get(raw,i,&value));
        Requirement r; r.name.assign(value.name,value.name_len);
        r.kind=value.kind==GLRMASK_TOKEN_REQUIREMENT ? RequirementKind::Token : RequirementKind::Grammar;
        r.kind=value.kind==0 ? RequirementKind::Token : value.kind==1 ? RequirementKind::Grammar : RequirementKind::Legacy;
        std::size_t segments=0;check(glrmask_requirements_path_len(raw,i,&segments));
        for (std::size_t j=0;j<segments;++j) {
            const char* text=nullptr;std::size_t size=0;check(glrmask_requirements_path(raw,i,j,&text,&size));
            r.path.emplace_back(text,size);
        }
        if (value.has_source_span) r.source_span=SourceSpan{value.start_byte,value.end_byte,value.line,value.column};
        result.push_back(std::move(r));
    }
    return result;
}
} // namespace detail
class Vocab;
class Grammar;
class PreparedGrammar;
class Module;
class Expr;
class Constraint;
class ConstraintState;
class ExactTokens : public detail::Owned<glrmask_exact_tokens,glrmask_exact_tokens_free,glrmask_exact_tokens_clone> {
    using Base=detail::Owned<glrmask_exact_tokens,glrmask_exact_tokens_free,glrmask_exact_tokens_clone>;
    explicit ExactTokens(glrmask_exact_tokens* p) : Base(p) {}
    friend class Vocab;
public:
    ExactTokens clone() const { return *this; }
    std::size_t size() const { std::size_t n=0; detail::check(glrmask_exact_tokens_len(native_handle(),&n)); return n; }
    bool empty() const { return size()==0; }
    std::vector<TokenId> ids() const {
        glrmask_u32s* p=nullptr; detail::check(glrmask_exact_tokens_ids(native_handle(),&p)); return detail::take_ids(p);
    }
    std::vector<TokenRange> ranges() const {
        std::size_t n=0; detail::check(glrmask_exact_tokens_ranges(native_handle(),nullptr,0,&n));
        std::vector<TokenRange> out(n); detail::check(glrmask_exact_tokens_ranges(native_handle(),out.data(),out.size(),&n)); return out;
    }
};
class Vocab : public detail::Owned<glrmask_vocab,glrmask_vocab_free,glrmask_vocab_clone> {
    using Base=detail::Owned<glrmask_vocab,glrmask_vocab_free,glrmask_vocab_clone>;
    static glrmask_vocab* create(const std::vector<VocabEntry>& entries,const std::vector<TokenId>& exact_only) {
        std::vector<glrmask_vocab_entry> native; native.reserve(entries.size());
        for (const auto& e:entries) native.push_back({e.id,e.bytes.data(),e.bytes.size()});
        glrmask_vocab* p=nullptr;
        detail::check(glrmask_vocab_new(native.data(),native.size(),exact_only.data(),exact_only.size(),&p)); return p;
    }
public:
    explicit Vocab(const std::vector<VocabEntry>& entries,const std::vector<TokenId>& exact_only={}) : Base(create(entries,exact_only)) {}
    Vocab clone() const { return *this; }
    std::size_t size() const { std::size_t n=0; detail::check(glrmask_vocab_len(native_handle(),&n)); return n; }
    ExactTokens tokens(const std::vector<TokenId>& ids) const {
        glrmask_exact_tokens* p=nullptr; detail::check(glrmask_vocab_tokens(native_handle(),ids.data(),ids.size(),&p)); return ExactTokens(p);
    }
    ExactTokens token_ranges(const std::vector<TokenRange>& ranges) const {
        glrmask_exact_tokens* p=nullptr; detail::check(glrmask_vocab_token_ranges(native_handle(),ranges.data(),ranges.size(),&p)); return ExactTokens(p);
    }
};
class Grammar : public detail::Owned<glrmask_grammar,glrmask_grammar_free,glrmask_grammar_clone> {
    using Base=detail::Owned<glrmask_grammar,glrmask_grammar_free,glrmask_grammar_clone>;
    explicit Grammar(glrmask_grammar* p) : Base(p) {}
    static Grammar source(std::uint32_t format,std::string_view text,const glrmask_lark_options* options=nullptr) {
        glrmask_grammar* p=nullptr; detail::check(glrmask_grammar_from_source(format,text.data(),text.size(),options,&p)); return Grammar(p);
    }
public:
    static Grammar from_ebnf(std::string_view text) { return source(GLRMASK_EBNF,text); }
    static Grammar from_glrm(std::string_view text) { return source(GLRMASK_GLRM,text); }
    static Grammar from_json_schema(std::string_view text) { return source(GLRMASK_JSON_SCHEMA,text); }
    static Grammar from_lark(std::string_view text,const LarkOptions& options={}) {
        std::vector<glrmask_lark_token> names;
        for (const auto& entry:options.token_names) names.push_back({entry.first.data(),entry.first.size(),entry.second.data(),entry.second.size()});
        const glrmask_lark_options native{sizeof(glrmask_lark_options),static_cast<std::uint32_t>(options.profile),names.data(),names.size()};
        return source(GLRMASK_LARK,text,&native);
    }
    static Grammar exact(const ExactTokens& tokens) {
        glrmask_grammar* p=nullptr; detail::check(glrmask_grammar_exact(tokens.native_handle(),&p)); return Grammar(p);
    }
    static Grammar from_expr(const Expr&);
    Module module() const;
    void bind(std::string_view name,const Expr& value);
    std::vector<UnboundExtern> unbound() const {return requirements();}
    std::vector<UnboundExtern> externs() const {
        glrmask_requirements* p=nullptr;detail::check(glrmask_grammar_externs(native_handle(),&p));return detail::take_requirements(p);
    }
    Grammar clone() const { return *this; }
    void bind(std::string_view name,const Grammar& value) { detail::check(glrmask_grammar_bind_grammar(native_handle(),name.data(),name.size(),value.native_handle())); }
    void bind(std::string_view name,Token value) { detail::check(glrmask_grammar_bind_token(native_handle(),name.data(),name.size(),value.id)); }
    void bind(std::string_view name,const ExactTokens& value) { detail::check(glrmask_grammar_bind_tokens(native_handle(),name.data(),name.size(),value.native_handle())); }
    std::vector<Requirement> requirements() const {
        glrmask_requirements* p=nullptr; detail::check(glrmask_grammar_requirements(native_handle(),&p)); return detail::take_requirements(p);
    }
    PreparedGrammar prepare(const Vocab&,std::optional<BoundaryTriggerDetail> boundary=std::nullopt) const;
    Constraint compile(const Vocab&,const BuildOptions& options={}) const;
};
// Operand handles are borrowed only during expression construction; the result owns snapshots.
struct ExprOperand {
    glrmask_expr_operand value;
    ExprOperand(const Grammar&);
    ExprOperand(const Module&);
    ExprOperand(const Expr&);
};
class Expr : public detail::Owned<glrmask_expr,glrmask_expr_free,glrmask_expr_clone> {
    using Base=detail::Owned<glrmask_expr,glrmask_expr_free,glrmask_expr_clone>;
    explicit Expr(glrmask_expr* p):Base(p) {}
    static Expr make(std::uint32_t kind,const std::vector<ExprOperand>& values) {
        std::vector<glrmask_expr_operand> items;items.reserve(values.size());
        for(const auto& value:values)items.push_back(value.value);
        glrmask_expr* p=nullptr;detail::check(glrmask_expr_new(kind,items.data(),items.size(),&p));return Expr(p);
    }
public:
    static Expr choice(const std::vector<ExprOperand>& values) {return make(0,values);}
    static Expr sequence(const std::vector<ExprOperand>& values) {return make(1,values);}
    Expr clone() const {return *this;}
};
class Module : public detail::Owned<glrmask_module,glrmask_module_free,glrmask_module_clone> {
    using Base=detail::Owned<glrmask_module,glrmask_module_free,glrmask_module_clone>;
    explicit Module(glrmask_module* p):Base(p) {}
    static glrmask_module* create(const Grammar& grammar,const ModuleOptions& options) {
        auto o=options.native();glrmask_module* p=nullptr;detail::check(glrmask_module_new(grammar.native_handle(),&o,&p));return p;
    }
    static glrmask_module* create(const Expr& expr,const ModuleOptions& options) {
        auto o=options.native();glrmask_module* p=nullptr;detail::check(glrmask_module_from_expr(expr.native_handle(),&o,&p));return p;
    }
public:
    explicit Module(const Grammar& grammar,const ModuleOptions& options={}):Base(create(grammar,options)) {}
    explicit Module(const Expr& expr,const ModuleOptions& options={}):Base(create(expr,options)) {}
    Module clone() const {return *this;}
    ModuleOptions options() const {
        glrmask_module_options o{};detail::check(glrmask_module_get_options(native_handle(),&o));
        return {static_cast<Optimization>(o.optimization),static_cast<BoundaryTriggerDetail>(o.boundary_trigger)};
    }
    Module with_options(const ModuleOptions& options) const {
        auto o=options.native();glrmask_module* p=nullptr;detail::check(glrmask_module_with_options(native_handle(),&o,&p));return Module(p);
    }
    void set_options(const ModuleOptions& options) {auto o=options.native();detail::check(glrmask_module_set_options(native_handle(),&o));}
    void bind(std::string_view name,const Grammar& value) {detail::check(glrmask_module_bind_grammar(native_handle(),name.data(),name.size(),value.native_handle()));}
    void bind(std::string_view name,const Module& value) {detail::check(glrmask_module_bind_module(native_handle(),name.data(),name.size(),value.native_handle()));}
    void bind(std::string_view name,const Expr& value) {detail::check(glrmask_module_bind_expr(native_handle(),name.data(),name.size(),value.native_handle()));}
    void bind(std::string_view name,Token value) {detail::check(glrmask_module_bind_token(native_handle(),name.data(),name.size(),value.id));}
    void bind(std::string_view name,const ExactTokens& value) {detail::check(glrmask_module_bind_tokens(native_handle(),name.data(),name.size(),value.native_handle()));}
    void precompile(const Vocab& vocab) {detail::check(glrmask_module_precompile(native_handle(),vocab.native_handle()));}
    Constraint compile(const Vocab&,const CompileOptions& options={}) const;
    std::vector<UnboundExtern> unbound() const {glrmask_requirements* p=nullptr;detail::check(glrmask_module_unbound(native_handle(),&p));return detail::take_requirements(p);}
    std::vector<UnboundExtern> externs() const {glrmask_requirements* p=nullptr;detail::check(glrmask_module_externs(native_handle(),&p));return detail::take_requirements(p);}
    Bytes save() const {glrmask_bytes* p=nullptr;detail::check(glrmask_module_save(native_handle(),&p));return detail::take_bytes(p);}
    static Module load(const Bytes& bytes,const Vocab* vocab=nullptr) {
        glrmask_module* p=nullptr;detail::check(glrmask_module_load(bytes.data(),bytes.size(),vocab?vocab->native_handle():nullptr,&p));return Module(p);
    }
};
inline ExprOperand::ExprOperand(const Grammar& value):value{0,value.native_handle()} {}
inline ExprOperand::ExprOperand(const Module& value):value{1,value.native_handle()} {}
inline ExprOperand::ExprOperand(const Expr& value):value{2,value.native_handle()} {}
inline Grammar Grammar::from_expr(const Expr& expr) {
    glrmask_grammar* p=nullptr;detail::check(glrmask_grammar_from_expr(expr.native_handle(),&p));return Grammar(p);
}
inline Module Grammar::module() const {return Module(*this);}
inline void Grammar::bind(std::string_view name,const Expr& value) {detail::check(glrmask_grammar_bind_expr(native_handle(),name.data(),name.size(),value.native_handle()));}
class PreparedGrammar : public detail::Owned<glrmask_prepared,glrmask_prepared_free,glrmask_prepared_clone> {
    using Base=detail::Owned<glrmask_prepared,glrmask_prepared_free,glrmask_prepared_clone>;
    explicit PreparedGrammar(glrmask_prepared* p) : Base(p) {}
    friend class Grammar;
public:
    PreparedGrammar clone() const { return *this; }
    void bind(std::string_view name,const PreparedGrammar& value) { detail::check(glrmask_prepared_bind_prepared(native_handle(),name.data(),name.size(),value.native_handle())); }
    void bind(std::string_view name,Token value) { detail::check(glrmask_prepared_bind_token(native_handle(),name.data(),name.size(),value.id)); }
    void bind(std::string_view name,const ExactTokens& value) { detail::check(glrmask_prepared_bind_tokens(native_handle(),name.data(),name.size(),value.native_handle())); }
    std::vector<Requirement> requirements() const {
        glrmask_requirements* p=nullptr; detail::check(glrmask_prepared_requirements(native_handle(),&p)); return detail::take_requirements(p);
    }
    Constraint link(const BuildOptions& options={}) const;
    Bytes save() const { glrmask_bytes* p=nullptr; detail::check(glrmask_prepared_save(native_handle(),&p)); return detail::take_bytes(p); }
    static PreparedGrammar load(const Bytes& bytes,const Vocab* vocab=nullptr) {
        glrmask_prepared* p=nullptr; detail::check(glrmask_prepared_load(bytes.data(),bytes.size(),vocab?vocab->native_handle():nullptr,&p)); return PreparedGrammar(p);
    }
};
class Constraint : public detail::Owned<glrmask_constraint,glrmask_constraint_free,glrmask_constraint_clone> {
    using Base=detail::Owned<glrmask_constraint,glrmask_constraint_free,glrmask_constraint_clone>;
    explicit Constraint(glrmask_constraint* p) : Base(p) {}
    friend class Grammar; friend class PreparedGrammar; friend class Module;
public:
    Constraint clone() const { return *this; }
    ConstraintState start() const;
    std::size_t mask_len() const { std::size_t n=0; detail::check(glrmask_constraint_mask_len(native_handle(),&n)); return n; }
    std::vector<TokenId> marker_ids() const { glrmask_u32s* p=nullptr; detail::check(glrmask_constraint_marker_ids(native_handle(),&p)); return detail::take_ids(p); }
    Bytes save(bool external_vocab=false) const { glrmask_bytes* p=nullptr; detail::check(glrmask_constraint_save(native_handle(),external_vocab?1u:0u,&p)); return detail::take_bytes(p); }
    static Constraint load(const Bytes& bytes,const Vocab* vocab=nullptr) {
        glrmask_constraint* p=nullptr; detail::check(glrmask_constraint_load(bytes.data(),bytes.size(),vocab?vocab->native_handle():nullptr,&p)); return Constraint(p);
    }
};
class ConstraintState : public detail::Owned<glrmask_state,glrmask_state_free,glrmask_state_clone> {
    using Base=detail::Owned<glrmask_state,glrmask_state_free,glrmask_state_clone>;
    explicit ConstraintState(glrmask_state* p) : Base(p) {}
    friend class Constraint;
public:
    ConstraintState clone() const { return *this; }
    std::size_t mask_len() const { std::size_t n=0; detail::check(glrmask_state_mask_len(native_handle(),&n)); return n; }
    void fill_mask(TokenId* output,std::size_t word_count) const { detail::check(glrmask_state_fill_mask(native_handle(),output,word_count)); }
    void fill_mask(std::vector<TokenId>& output) const { fill_mask(output.data(),output.size()); }
    std::vector<TokenId> mask() const { std::vector<TokenId> out(mask_len()); fill_mask(out); return out; }
    void commit_token(TokenId id) { detail::check(glrmask_state_commit_token(native_handle(),id)); }
    void commit_bytes(const std::uint8_t* bytes,std::size_t size) { detail::check(glrmask_state_commit_bytes(native_handle(),bytes,size)); }
    void commit_bytes(const Bytes& bytes) { commit_bytes(bytes.data(),bytes.size()); }
    void commit_bytes(std::string_view bytes) { commit_bytes(reinterpret_cast<const std::uint8_t*>(bytes.data()),bytes.size()); }
    bool is_accepting() const { std::uint32_t value=0; detail::check(glrmask_state_is_accepting(native_handle(),&value)); return value!=0; }
    bool is_rejected() const { std::uint32_t value=0; detail::check(glrmask_state_is_rejected(native_handle(),&value)); return value!=0; }
    std::vector<TokenId> forced_tokens() const { glrmask_u32s* p=nullptr; detail::check(glrmask_state_forced_tokens(native_handle(),&p)); return detail::take_ids(p); }
    std::vector<MarkerEvent> marker_events() const {
        glrmask_events* raw=nullptr; detail::check(glrmask_state_marker_events(native_handle(),&raw));
        std::unique_ptr<glrmask_events,decltype(&glrmask_events_free)> owner(raw,glrmask_events_free);
        const MarkerEvent* data=nullptr; std::size_t len=0; detail::check(glrmask_events_data(raw,&data,&len));
        return len ? std::vector<MarkerEvent>(data,data+len) : std::vector<MarkerEvent>{};
    }
};
inline PreparedGrammar Grammar::prepare(const Vocab& vocab,std::optional<BoundaryTriggerDetail> boundary) const {
    glrmask_prepared* p=nullptr;
    detail::check(glrmask_grammar_prepare(native_handle(),vocab.native_handle(),boundary?static_cast<std::uint32_t>(*boundary):GLRMASK_BOUNDARY_DEFAULT,&p));
    return PreparedGrammar(p);
}
inline Constraint Grammar::compile(const Vocab& vocab,const BuildOptions& options) const {
    const auto native=options.native(); glrmask_constraint* p=nullptr;
    detail::check(glrmask_grammar_compile(native_handle(),vocab.native_handle(),&native,&p)); return Constraint(p);
}
inline Constraint Module::compile(const Vocab& vocab,const CompileOptions& options) const {
    const auto native=options.native();glrmask_constraint* p=nullptr;
    detail::check(glrmask_module_compile(native_handle(),vocab.native_handle(),&native,&p));return Constraint(p);
}
inline Constraint PreparedGrammar::link(const BuildOptions& options) const {
    const auto native=options.native(); glrmask_constraint* p=nullptr;
    detail::check(glrmask_prepared_link(native_handle(),&native,&p)); return Constraint(p);
}
inline ConstraintState Constraint::start() const {
    glrmask_state* p=nullptr; detail::check(glrmask_constraint_start(native_handle(),&p)); return ConstraintState(p);
}
inline const char* version() noexcept { return glrmask_version(); }
} // namespace glrmask
#endif
