#include <glrmask.hpp>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <thread>
#if defined(__linux__)
#include <pthread.h>
#endif

using namespace glrmask;

#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string(__FILE__)+":"+std::to_string(__LINE__)+": " #x); } while(false)

template<class F> void throws(F&& fn, glrmask_status status=-1) {
    bool thrown=false; try { fn(); } catch(const Error& e) { thrown=true; CHECK(status<0 || e.status()==status); }
    CHECK(thrown);
}

bool allowed(const std::vector<TokenId>& m, TokenId id) { return id/32<m.size() && (m[id/32]&(TokenId{1}<<(id%32)))!=0; }
CompileOptions strategy_options(unsigned mode, BoundaryMaskStrategy boundary = BoundaryMaskStrategy::Auto) {
    CompileOptions options;
    options.mask_strategy = static_cast<MaskStrategy>(mode);
    options.boundary_mask_strategy = boundary;
    return options;
}
Vocab vocabulary() { return Vocab({{0,"a"},{1,"b"},{31,"ab"},{32,"a"},{65,"z"}}); }
auto simple() { return Grammar::from_ebnf(R"(start ::= "a" "b")"); }

void basic() {
    for (unsigned i=0; i<4; ++i) {
        auto v = vocabulary();
        auto g = simple();
        auto m = Module(g);
        auto c = m.compile(v, strategy_options(i));
        auto s = c.start();
        const auto initial = s.mask();
        CHECK(allowed(initial,0)); CHECK(allowed(initial,31)); CHECK(allowed(initial,32));
        CHECK(!allowed(initial,1)); CHECK(!allowed(initial,65)); CHECK(s.mask_len()==3);
        std::vector<TokenId> buffer(s.mask_len()+2, 0xffffffff);
        s.fill_mask(buffer); CHECK(std::equal(initial.begin(), initial.end(), buffer.begin()));
        CHECK(buffer[3]==0 && buffer[4]==0);
        std::vector<TokenId> short_buffer(1, 123); throws([&]{ s.fill_mask(short_buffer); }, GLRMASK_BUFFER_TOO_SMALL);
        CHECK(short_buffer[0]==123); CHECK(s.mask()==initial);
        auto copy = s; c.dispose(); g.dispose(); v.dispose();
        s.commit_token(31); CHECK(s.is_accepting()); CHECK(!copy.is_accepting());
        copy.commit_token(0); CHECK(copy.forced_tokens()==std::vector<TokenId>{1});
        CHECK(!copy.is_accepting()); copy.commit_token(1); CHECK(copy.is_accepting());
        auto moved = std::move(copy); throws([&]{ copy.mask(); }, GLRMASK_CLOSED); CHECK(moved.is_accepting());
        s.dispose(); s.dispose(); throws([&]{ s.mask(); }, GLRMASK_CLOSED);
    }
}

void exact_and_unicode() {
    const Bytes first{0xc3}, last{0xa9}, whole{0xc3, 0xa9};
    Vocab v({{0,first},{1,last},{31,whole},{32,whole},{40,Bytes{0,255}}},{99});
    const std::string source = "start ::= \"" + std::string(whole.begin(), whole.end()) + "\"";
    for (unsigned i=0; i<4; ++i) {
        auto g = Grammar::from_ebnf(source);
        auto m = Module(g);
        auto c = m.compile(v, strategy_options(i));
        auto s = c.start();
        CHECK(allowed(s.mask(),0)); CHECK(allowed(s.mask(),31));
        s.commit_bytes(first); CHECK(!s.is_accepting()); CHECK(allowed(s.mask(),1));
        s.commit_bytes(last); CHECK(s.is_accepting());

        auto host = Grammar::from_glrm(R"(glrm 1; start root; extern token T; nt root = T;)");
        auto em = Module(host);
        em.bind("T", TokenSet({31}));
        auto exact = em.compile(v, strategy_options(i));
        CHECK(allowed(exact.start().mask(),31)); CHECK(!allowed(exact.start().mask(),32));
        auto e = exact.start(); e.commit_token(31); CHECK(e.is_accepting());

        auto rm = Module(host);
        rm.bind("T", TokenSet({40}));
        auto raw = rm.compile(v, strategy_options(i)).start(); raw.commit_token(40); CHECK(raw.is_accepting());

        auto cm = Module(host);
        cm.bind("T", TokenSet({99}));
        auto control = cm.compile(v, strategy_options(i)).start(); control.commit_token(99); CHECK(control.is_accepting());
    }
    auto merged = TokenSet::from_ranges({{31,32},{32,32}});
    auto duplicate = TokenSet({31,31,32});
    auto reversed = TokenSet::from_ranges({{2,0}});
    auto empty = TokenSet::from_ranges({{5,5}});
    Vocab maximum({}, {0xffffffff});
    auto high = TokenSet::from_ranges({{0xffffffff, 0xffffffff}});
    auto none = TokenSet({});
    (void)merged; (void)duplicate; (void)high; (void)none; (void)reversed; (void)empty;
}

void end_tokens() {
    Vocab v({{0,"a"}}, {99});
    for (unsigned i=0; i<4; ++i) {
        CompileOptions o({99});
        o.mask_strategy = static_cast<MaskStrategy>(i);
        auto m = Module(Grammar::from_ebnf(R"(start ::= "a")"));
        auto s = m.compile(v, o).start();
        CHECK(!allowed(s.mask(),99)); s.commit_token(0); CHECK(!s.is_accepting()); CHECK(allowed(s.mask(),99));
        s.commit_token(99); CHECK(s.is_accepting());
        const auto m_words = s.mask(); CHECK(std::all_of(m_words.begin(), m_words.end(), [](auto w){ return w==0; }));
        CHECK(ConstraintState(s).is_accepting()); throws([&]{ s.commit_token(99); });
    }
}

void composition_and_persistence() {
    Vocab v({{0,"x"},{1,"a"},{2,"b"},{3,"y"},{4,"xay"},{5,"xby"}});
    auto host = Grammar::from_glrm(R"(glrm 1; start root; extern grammar child; nt root = "x" child "y";)");
    auto host_mod = Module(host);
    auto ext = host_mod.externs();
    CHECK(ext.size()==1 && ext[0].name=="child");
    CHECK(ext[0].kind==ExternKind::Grammar && ext[0].span.has_value());
    CHECK(host_mod.externs().unbound().size()==1);

    auto a = Grammar::from_ebnf(R"(start ::= "a")");
    auto b = Grammar::from_ebnf(R"(start ::= "b")");
    auto left = Module(host), right = Module(host);
    left.bind("child", a); right.bind("child", b);
    CHECK(left.externs().unbound().empty());
    CHECK(right.externs().unbound().empty());

    // Rebinding error
    throws([&]{ left.bind("child", b); });

    // Qualified path error
    throws([&]{ left.bind("foo::bar", b); });

    // Open-child error
    auto open_g = Grammar::from_glrm(R"(glrm 1; start root; extern grammar open_slot; nt root = open_slot;)");
    auto open_mod = Module(open_g);
    auto parent_mod = Module(host);
    throws([&]{ parent_mod.bind("child", open_mod); });

    // Precompile and save/load
    Module lp(host); lp.precompile(v);
    Module ap(a); ap.precompile(v);
    lp.bind("child", ap);
    const auto prepared_bytes = lp.save(); lp.dispose(); ap.dispose();
    auto restored = Module::load(prepared_bytes, &v);
    CHECK(restored.externs().unbound().empty());

    for (unsigned i=0; i<2; ++i) {
        auto strat = (i == 0) ? MaskStrategy::Auto : MaskStrategy::ParserDwa;
        CompileOptions options;
        options.mask_strategy = strat;
        auto lc = left.compile(v, options);
        auto rc = right.compile(v, options);
        auto pc = restored.compile(v, options);
        CHECK(allowed(lc.start().mask(),4) && !allowed(lc.start().mask(),5));
        CHECK(allowed(rc.start().mask(),5) && !allowed(rc.start().mask(),4));
        CHECK(pc.start().mask()==lc.start().mask());

        auto saved = lc.save(); auto loaded = Constraint::load(saved); CHECK(loaded.start().mask()==lc.start().mask());
        auto external = lc.save(true); throws([&]{ Constraint::load(external); });
        CHECK(Constraint::load(external, &v).start().mask()==lc.start().mask());
        Vocab wrong({{0,"wrong"},{1,"a"},{2,"b"},{3,"y"},{4,"xay"},{5,"xby"}});
        throws([&]{ Constraint::load(external, &wrong); });
    }
}

void markers_and_frontends() {
    auto v = vocabulary();
    for (unsigned i=0; i<4; ++i) {
        auto g = Grammar::from_glrm(R"(glrm 1; start root; nt root = @mark(10) "a" @mark(20) "b" @mark(30);)");
        auto m = Module(g);
        auto c = m.compile(v, strategy_options(i)); CHECK(c.marker_ids()==std::vector<TokenId>({10,20,30}));
        auto s = c.start(); CHECK(s.marker_events().size()==1 && s.marker_events()[0].byte_offset==0);
        s.commit_token(31); const auto events = s.marker_events();
        CHECK(events.size()==2 && events[0].id==20 && events[0].byte_offset==1 && events[1].id==30 && events[1].byte_offset==2);
        CHECK(ConstraintState(s).marker_events().size()==2);
        auto load = Constraint::load(c.save()); CHECK(load.marker_ids()==c.marker_ids());
        auto lm = Module(Grammar::from_lark("start: \"a\" \"b\""));
        auto l = lm.compile(v, strategy_options(i)).start(); l.commit_token(31); CHECK(l.is_accepting());
    }
    Vocab special({{0,"a"},{1,"a"}}, {99});
    for (unsigned i=0; i<4; ++i) {
        auto g = Grammar::from_lark("start: <|begin|> <|body|>");
        auto m = Module(g);
        m.bind("begin", TokenSet({99}));
        m.bind("body", TokenSet({1}));
        auto state = m.compile(special, strategy_options(i)).start();
        CHECK(allowed(state.mask(),99)); state.commit_token(99);
        CHECK(allowed(state.mask(),1) && !allowed(state.mask(),0)); state.commit_token(1); CHECK(state.is_accepting());
    }
    Vocab json({{0,"null"},{1,"true"}});
    auto js = Module(Grammar::from_json_schema(R"({"type":"null"})")).compile(json).start();
    CHECK(allowed(js.mask(),0) && !allowed(js.mask(),1)); js.commit_token(0); CHECK(js.is_accepting());
}

void threads_and_errors() {
    auto v = vocabulary(); auto c = simple().compile(v); std::atomic<bool> okay{true}; std::vector<std::thread> threads;
    for (int i=0; i<4; ++i) threads.emplace_back([&]{ try { for (int j=0; j<8; ++j) { auto s = c.start(); s.commit_token(31); if (!s.is_accepting()) okay=false; } } catch(...) { okay=false; } });
    for (auto& thread : threads) thread.join(); CHECK(okay);
    std::size_t ignored = 0; CHECK(glrmask_vocab_len(nullptr, &ignored)==GLRMASK_INVALID_ARGUMENT);
    const std::string main_error = glrmask_error_message();
    std::thread other([]{ glrmask_grammar* p = nullptr; const char invalid[] = {char(0xff)}; glrmask_grammar_from_source(0, invalid, 1, &p); }); other.join();
    CHECK(main_error==glrmask_error_message());
    throws([&]{ Vocab duplicate({{0,"a"},{0,"b"}}); }, GLRMASK_INVALID_ARGUMENT);
    throws([&]{ Vocab overlap({{0,"a"}}, {0}); }, GLRMASK_INVALID_ARGUMENT);
}

#if defined(__linux__)
int thread_stack_probe(std::size_t stack_bytes) {
    auto v = vocabulary(); std::vector<Constraint> constraints;
    for (unsigned i=0; i<4; ++i) {
        auto m = Module(simple());
        constraints.push_back(m.compile(v, strategy_options(i)));
    }
    struct Probe { std::vector<Constraint>* constraints; bool okay=false; } probe{&constraints};
    pthread_attr_t requested; CHECK(pthread_attr_init(&requested)==0);
    if (stack_bytes) CHECK(pthread_attr_setstacksize(&requested, stack_bytes)==0);
    pthread_t worker;
    CHECK(pthread_create(&worker, &requested, [](void* context)->void* {
        auto& p = *static_cast<Probe*>(context);
        pthread_attr_t actual;
        if (pthread_getattr_np(pthread_self(), &actual)==0) {
            void* base = nullptr; std::size_t size = 0, guard = 0;
            pthread_attr_getstack(&actual, &base, &size);
            pthread_attr_getguardsize(&actual, &guard);
            std::cerr<<"native worker stack base="<<base<<" bytes="<<size<<" guard="<<guard<<std::endl;
            pthread_attr_destroy(&actual);
        }
        try {
            for (auto& constraint : *p.constraints) for (int j=0; j<8; ++j) {
                auto s = constraint.start(); CHECK(allowed(s.mask(), 31));
                auto snapshot = s; s.commit_token(31); CHECK(s.is_accepting());
                snapshot.commit_token(0); CHECK(snapshot.forced_tokens()==std::vector<TokenId>{1});
                snapshot.commit_token(1); CHECK(snapshot.is_accepting());
            }
            p.okay = true;
        }
        catch (const std::exception& error) { std::cerr<<error.what()<<std::endl; }
        return nullptr;
    }, &probe)==0);
    pthread_attr_destroy(&requested); CHECK(pthread_join(worker, nullptr)==0); CHECK(probe.okay);
    return 0;
}
#endif

void emit_parity() {
    auto v = vocabulary(); std::cout << '[';
    for (unsigned i=0; i<4; ++i) {
        if (i) std::cout << ','; std::cout << "{\"mode\":" << i << ",\"states\":[";
        auto m = Module(simple());
        auto s = m.compile(v, strategy_options(i)).start();
        for (unsigned step=0; step<3; ++step) {
            if (step) { s.commit_token(step-1); std::cout << ','; }
            const auto mask_words = s.mask(); std::cout << "{\"mask\":[";
            for (std::size_t j=0; j<mask_words.size(); ++j) { if (j) std::cout << ','; std::cout << mask_words[j]; }
            std::cout << "],\"accepting\":" << (s.is_accepting()?"true":"false") << '}';
        } std::cout << "]}";
    } std::cout << "]\n";
}

Bytes read_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary); CHECK(in.good()); return Bytes(std::istreambuf_iterator<char>(in), {});
}

void artifacts(const std::filesystem::path& directory, bool check) {
    std::filesystem::create_directories(directory); auto v = vocabulary();
    for (unsigned i=0; i<4; ++i) {
        auto m = Module(simple());
        auto c = m.compile(v, strategy_options(i));
        for (bool external : {false, true}) {
            const auto path = directory/((check?"node-":"cpp-")+std::string(external?"external-":"mode-")+std::to_string(i)+".glr");
            if (check) {
                auto loaded = Constraint::load(read_bytes(path), external?&v:nullptr);
                CHECK(loaded.start().mask()==c.start().mask());
                auto s = loaded.start(); s.commit_token(31); CHECK(s.is_accepting());
            }
            else {
                const auto bytes = c.save(external); std::ofstream out(path, std::ios::binary);
                out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
                CHECK(out.good());
            }
        }
    }
}

void canonical_modules() {
    auto v = Vocab({{0,"x"},{1,"a"},{2,"y"},{3,"xay"},{4,"b"}});
    auto host = Grammar::from_glrm("glrm 1; start start; extern grammar child; nt start = \"x\" child \"y\";");
    auto parent = Module(host); parent.precompile(v);
    auto original = parent; // copy constructor is a cheap snapshot

    auto child = Module(Grammar::from_ebnf("start ::= \"a\""));
    CHECK(BoundaryMaskStrategy::FastBuild == BoundaryMaskStrategy::TokenTrie);
    CHECK(BoundaryMaskStrategy::FastRuntime == BoundaryMaskStrategy::ParserDwa);
    CHECK(static_cast<std::uint32_t>(BoundaryMaskStrategy::TokenTrie) == GLRMASK_BOUNDARY_MASK_STRATEGY_TOKEN_TRIE);
    CHECK(static_cast<std::uint32_t>(BoundaryMaskStrategy::ParserDwa) == GLRMASK_BOUNDARY_MASK_STRATEGY_PARSER_DWA);
    CompileOptions child_options = strategy_options(
        static_cast<unsigned>(MaskStrategy::TokenClassTrie), BoundaryMaskStrategy::TokenTrie);
    child.precompile(v, child_options); parent.bind("child", child);

    auto loaded = Module::load(parent.save(), &v);
    CHECK(loaded.externs().unbound().empty());
    CHECK(original.externs().unbound().at(0).name=="child");
    CHECK(loaded.externs().at(0).name=="child");

    // Retained work survives without storing per-module settings.
    auto child_bytes = child.save();
    auto child_loaded = Module::load(child_bytes, &v);
    CHECK(child_loaded.externs().empty());

    auto state = loaded.compile(v, child_options).start(); loaded.dispose(); parent.dispose(); child.dispose();
    state.commit_token(3); CHECK(state.is_accepting());

    bool failed = false; try { original.bind("missing", Grammar::from_ebnf("start ::= \"a\"")); } catch(const Error&) { failed=true; } CHECK(failed);
    CHECK(original.externs().unbound().at(0).name=="child");

    // Qualified paths rejected
    failed = false; try { original.bind("child::nested", Grammar::from_ebnf("start ::= \"a\"")); } catch(const Error&) { failed=true; } CHECK(failed);

    // Rebinding error
    auto test_mod = Module(host);
    test_mod.bind("child", Grammar::from_ebnf("start ::= \"a\""));
    failed = false; try { test_mod.bind("child", Grammar::from_ebnf("start ::= \"b\"")); } catch(const Error&) { failed=true; } CHECK(failed);

    // Expressions
    auto expr = Expr::sequence({
        Grammar::from_ebnf("start ::= \"x\""),
        Expr::choice({Grammar::from_ebnf("start ::= \"a\""), Grammar::from_ebnf("start ::= \"b\"")}),
        Grammar::from_ebnf("start ::= \"y\"")
    });
    auto host2 = Grammar::from_glrm("glrm 1; start start; extern grammar child; nt start = child;");
    auto m_expr = Module(host2);
    m_expr.bind("child", expr);
    auto c = m_expr.compile(v); auto s = c.start(); s.commit_token(3); CHECK(s.is_accepting());

    // Expr literal
    auto lit = Expr::literal("xay");
    auto m_lit = Module(host2);
    m_lit.bind("child", lit);
    auto c_lit = m_lit.compile(v);
    auto s_lit = c_lit.start(); s_lit.commit_token(3); CHECK(s_lit.is_accepting());

    // Externs ordering and unbound view
    auto multi = Grammar::from_glrm("glrm 1; start start; extern token first; extern grammar second; extern token third; nt start = first second third;");
    auto multi_mod = Module(multi);
    auto ext_all = multi_mod.externs();
    CHECK(ext_all.size() == 3);
    CHECK(ext_all[0].name == "first" && ext_all[0].kind == ExternKind::Token);
    CHECK(ext_all[1].name == "second" && ext_all[1].kind == ExternKind::Grammar);
    CHECK(ext_all[2].name == "third" && ext_all[2].kind == ExternKind::Token);

    multi_mod.bind("second", Grammar::from_ebnf("start ::= \"a\""));
    auto ext_unbound = multi_mod.externs().unbound();
    CHECK(ext_unbound.size() == 2);
    CHECK(ext_unbound[0].name == "first");
    CHECK(ext_unbound[1].name == "third");

    // Grammar::precompile is the module-returning mirror of Grammar::compile.
    auto open_host = host.precompile(v);
    auto precompiled_child = Grammar::from_ebnf("start ::= \"a\"").precompile(
        v, strategy_options(static_cast<unsigned>(MaskStrategy::ParserDwa)));
    CHECK(precompiled_child.save().size() > Module(Grammar::from_ebnf("start ::= \"a\"")).save().size());
    open_host.bind("child", precompiled_child);
    auto precompiled_state = open_host.compile(
        v, strategy_options(static_cast<unsigned>(MaskStrategy::TokenTrie))).start();
    CHECK(allowed(precompiled_state.mask(), 3));
    precompiled_state.commit_token(3); CHECK(precompiled_state.is_accepting());
}

int main(int argc, char** argv) {
    try {
#if defined(__linux__)
        if (argc==3 && std::string(argv[1])=="--thread-stack-probe") return thread_stack_probe(std::stoull(argv[2]));
#endif
        if (argc==2 && std::string(argv[1])=="--parity") { emit_parity(); return 0; }
        if (argc==3 && std::string(argv[1])=="--save") { artifacts(argv[2], false); return 0; }
        if (argc==3 && std::string(argv[1])=="--check") { artifacts(argv[2], true); return 0; }
        CHECK(argc==1);
        canonical_modules();
        basic();
        exact_and_unicode();
        end_tokens();
        composition_and_persistence();
        markers_and_frontends();
        threads_and_errors();
        std::cout << "C++ binding regression groups: 7 passed\n"; return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
