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
CompileOptions strategy_options(unsigned mode) {
    CompileOptions options;
    options.mask_strategy = (mode == 0 ? MaskStrategy::Auto : mode == 3 ? MaskStrategy::FastCompile : MaskStrategy::FastMasks);
    return options;
}
Vocab vocabulary() { return Vocab({{0,"a"},{1,"b"},{31,"ab"},{32,"a"},{65,"z"}}); }
auto simple() { return Grammar::from_lark(R"(start: "a" "b")"); }

void basic() {
    for (unsigned i=0; i<3; ++i) {
        auto v = vocabulary();
        auto g = simple();
        auto m = Module(g);
        const auto options = strategy_options(i==0?0:i==1?3:4);
        m.precompile(v, options);
        auto wrong = Vocab({{0,"z"}});
        throws([&]{ m.precompile(wrong, options); }, GLRMASK_VOCAB_MISMATCH);
        throws([&]{ m.compile(wrong, options); }, GLRMASK_VOCAB_MISMATCH);
        auto c = m.compile(v, options);
        auto s = c.start();
        const auto initial = s.mask();
        CHECK(allowed(initial,0)); CHECK(allowed(initial,31)); CHECK(allowed(initial,32));
        CHECK(!allowed(initial,1)); CHECK(!allowed(initial,65)); CHECK(s.mask_bit_len()==66); CHECK(initial.size()==(s.mask_bit_len()+31)/32);
        std::vector<TokenId> buffer((s.mask_bit_len()+31)/32+2, 0xffffffff);
        s.fill_mask(buffer); CHECK(std::equal(initial.begin(), initial.end(), buffer.begin()));
        CHECK(buffer[3]==0 && buffer[4]==0);
        std::vector<TokenId> short_buffer(1, 123); throws([&]{ s.fill_mask(short_buffer); }, GLRMASK_BUFFER_TOO_SMALL);
        CHECK(short_buffer[0]==123); CHECK(s.mask()==initial);
        auto copy = s; { auto owned = std::move(c); auto source = std::move(g); auto vocab_owner = std::move(v); }
        CHECK(!s.is_terminated());
        CHECK(s.commit_token(31)); CHECK(s.is_accepting()); CHECK(s.is_terminated()); CHECK(!copy.is_accepting());
        CHECK(copy.commit_token(0)); auto before = copy.mask(); CHECK(copy.forced_bytes()==Bytes{static_cast<uint8_t>('b')}); CHECK(copy.mask()==before);
        CHECK(!copy.is_accepting()); CHECK(copy.commit_token(1)); CHECK(copy.is_accepting()); CHECK(copy.is_terminated());
        auto moved = std::move(copy); throws([&]{ copy.mask(); }, GLRMASK_CLOSED); CHECK(moved.is_accepting());
        auto moved_s = std::move(s); throws([&]{ s.mask(); }, GLRMASK_CLOSED); CHECK(moved_s.is_accepting());
    }
}

void exact_and_unicode() {
    const Bytes first{0xc3}, last{0xa9}, whole{0xc3, 0xa9};
    Vocab v({{0,first},{1,last},{31,whole},{32,whole},{40,Bytes{0,255}}});
    const std::string source = "start: \"" + std::string(whole.begin(), whole.end()) + "\"";
    for (unsigned i=0; i<3; ++i) {
        auto g = Grammar::from_lark(source);
        auto m = Module(g);
        const auto options = strategy_options(i==0?0:i==1?3:4);
        m.precompile(v, options);
        auto wrong = Vocab({{0,"z"}});
        throws([&]{ m.precompile(wrong, options); }, GLRMASK_VOCAB_MISMATCH);
        throws([&]{ m.compile(wrong, options); }, GLRMASK_VOCAB_MISMATCH);
        auto c = m.compile(v, options);
        auto s = c.start();
        CHECK(allowed(s.mask(),0)); CHECK(allowed(s.mask(),31));
        CHECK(s.commit_bytes(first)); CHECK(!s.is_accepting()); CHECK(allowed(s.mask(),1));
        CHECK(s.commit_bytes(last)); CHECK(s.is_accepting());

        auto host = Grammar::from_glrm(R"(glrm 1; start root; extern token T; nt root = T;)");
        auto em = Module(host);
        em.bind("T", TokenSet({31}));
        auto exact = em.compile(v, strategy_options(i==0?0:i==1?3:4));
        CHECK(allowed(exact.start().mask(),31)); CHECK(!allowed(exact.start().mask(),32));
        auto e = exact.start(); CHECK(e.commit_token(31)); CHECK(e.is_accepting());

        auto rm = Module(host);
        rm.bind("T", TokenSet({40}));
        auto raw = rm.compile(v, strategy_options(i==0?0:i==1?3:4)).start(); CHECK(raw.commit_token(40)); CHECK(raw.is_accepting());

        auto cm = Module(host);
        cm.bind("T", TokenSet({99}));
        auto control = cm.compile(v, strategy_options(i==0?0:i==1?3:4)).start(); CHECK(control.commit_token(99)); CHECK(control.is_accepting());
    }
    auto merged = TokenSet::from_ranges({{31,33}});
    auto duplicate = TokenSet({31,31,32});
    throws([]{ TokenSet::from_ranges({{2,0}}); }, GLRMASK_INVALID_INPUT);
    auto empty = TokenSet::from_ranges({{5,5}});
    Vocab maximum({});
    auto high = TokenSet::from_ranges({{0xffffff, 0xffffff}});
    throws([]{ TokenSet({0xffffffff}); }, GLRMASK_INVALID_INPUT);
    throws([]{ TokenSet({4000000000u}); }, GLRMASK_INVALID_INPUT);
    auto none = TokenSet(std::vector<TokenId>{});
    (void)merged; (void)duplicate; (void)high; (void)none; (void)empty;
}

void reject_module_expr_in_grammar() {
    Vocab v({{0,"a"}});
    auto source = Grammar::from_lark("start: \"a\"");
    auto prepared = Module(source); prepared.precompile(v);
    auto host = Grammar::from_glrm("start s extern grammar item nt s = item");
    auto token = TokenSet({5});
    for (const auto& expr : {Expr::choice({token,Expr::literal("x")}), Expr::sequence({Expr::literal(""),token}), Expr::sequence({Expr::choice({token,Expr::literal("x")}),Expr::literal("")})}) {
        throws([&]{ (void)Grammar::from_expr(expr); }, GLRMASK_INVALID_INPUT);
        auto outer=Module(host); outer.bind("item",expr);
        auto state=outer.compile(v).start(); CHECK(state.commit_token(5) && state.is_accepting());
    }
    for (const auto& m : {Module(source), prepared}) {
        for (const auto& expr : {Expr::sequence({m}), Expr::sequence({Expr::choice({m}), Expr::literal("")})}) {
            throws([&]{ (void)Grammar::from_expr(expr); }, GLRMASK_INVALID_INPUT);
            auto g = host;
            CHECK(g.externs().unbound().size()==1);
            auto outer = Module(host); outer.bind("item",expr);
            auto state = outer.compile(v).start();
            CHECK(state.commit_token(0) && state.is_accepting());
        }
    }
}

void end_tokens() {
    Vocab v({{0,"a"}});
    for (unsigned i=0; i<3; ++i) {
        CompileOptions o;
        o.mask_strategy = (i==0 ? MaskStrategy::Auto : i==1 ? MaskStrategy::FastCompile : MaskStrategy::FastMasks);
        auto m = Module(Grammar::from_glrm("extern grammar VALUE start root nt root = VALUE"));
        m.bind("VALUE", Expr::sequence({Grammar::from_lark(R"(start: "a")"), TokenSet({99})}));
        auto s = m.compile(v, o).start();
        CHECK(!allowed(s.mask(),99)); CHECK(s.commit_token(0)); CHECK(!s.is_accepting()); CHECK(allowed(s.mask(),99));
        CHECK(s.commit_token(99)); CHECK(s.is_accepting());
        const auto m_words = s.mask(); CHECK(std::all_of(m_words.begin(), m_words.end(), [](auto w){ return w==0; }));
        CHECK(ConstraintState(s).is_accepting()); CHECK(!s.commit_token(99));
    }
}

void composition_and_persistence() {
    Vocab v({{0,"x"},{1,"a"},{2,"b"},{3,"y"},{4,"xay"},{5,"xby"}});
    auto host = Grammar::from_glrm(R"(glrm 1; start root; extern grammar child; nt root = "x" child "y";)");
    auto host_mod = Module(host);
    auto ext = host_mod.externs();
    CHECK(ext.size()==1 && (ext.begin()+0)->name()=="child");
    CHECK((ext.begin()+0)->kind()==ExternKind::Grammar);
    CHECK(host_mod.externs().unbound().size()==1);

    auto a = Grammar::from_lark(R"(start: "a")");
    auto b = Grammar::from_lark(R"(start: "b")");
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
    const auto prepared_bytes = lp.save(); { auto owner1 = std::move(lp); auto owner2 = std::move(ap); }
    auto restored = Module::load(prepared_bytes);
    CHECK(restored.externs().unbound().empty());

    for (unsigned i=0; i<2; ++i) {
        auto strat = (i == 0) ? MaskStrategy::Auto : MaskStrategy::FastMasks;
        CompileOptions options;
        options.mask_strategy = strat;
        auto lc = left.compile(v, options);
        auto rc = right.compile(v, options);
        auto pc = restored.compile(v, options);
        CHECK(allowed(lc.start().mask(),4) && !allowed(lc.start().mask(),5));
        CHECK(allowed(rc.start().mask(),5) && !allowed(rc.start().mask(),4));
        CHECK(pc.start().mask()==lc.start().mask());

        auto saved = lc.save(); auto loaded = Constraint::load(saved, v); CHECK(loaded.start().mask()==lc.start().mask());
        Vocab wrong({{0,"wrong"},{1,"a"},{2,"b"},{3,"y"},{4,"xay"},{5,"xby"}});
        throws([&]{ Constraint::load(saved, wrong); });
    }
}

void frontends() {
    auto v = vocabulary();
    for (unsigned i=0; i<3; ++i) {
        auto lm = Module(Grammar::from_lark("start: \"a\" \"b\""));
        auto l = lm.compile(v, strategy_options(i==0?0:i==1?3:4)).start(); CHECK(l.commit_token(31)); CHECK(l.is_accepting());
    }
    Vocab special({{0,"a"},{1,"a"}});
    for (unsigned i=0; i<3; ++i) {
        auto g = Grammar::from_lark("start: <|begin|> <|body|>");
        auto m = Module(g);
        m.bind("begin", TokenSet({99}));
        m.bind("body", TokenSet({1}));
        auto state = m.compile(special, strategy_options(i==0?0:i==1?3:4)).start();
        CHECK(allowed(state.mask(),99)); CHECK(state.commit_token(99));
        CHECK(allowed(state.mask(),1) && !allowed(state.mask(),0)); CHECK(state.commit_token(1)); CHECK(state.is_accepting());
    }
    Vocab json({{0,"null"},{1,"true"}});
    auto js = Module(Grammar::from_json_schema(R"({"type":"null"})")).compile(json).start();
    CHECK(allowed(js.mask(),0) && !allowed(js.mask(),1)); CHECK(js.commit_token(0)); CHECK(js.is_accepting());
}

void threads_and_errors() {
    auto v = vocabulary(); auto c = simple().compile(v); std::atomic<bool> okay{true}; std::vector<std::thread> threads;
    for (int i=0; i<4; ++i) threads.emplace_back([&]{ try { for (int j=0; j<8; ++j) { auto s = c.start(); CHECK(s.commit_token(31)); if (!s.is_accepting()) okay=false; } } catch(...) { okay=false; } });
    for (auto& thread : threads) thread.join(); CHECK(okay);
    std::size_t ignored = 0; CHECK(glrmask_constraint_mask_bit_len(nullptr, &ignored)==GLRMASK_INVALID_INPUT);
    const std::string main_error = glrmask_error_message();
    std::thread other([]{ glrmask_grammar* p = nullptr; const char invalid[] = {char(0xff)}; CHECK(glrmask_grammar_from_lark(invalid, 1, &p) == GLRMASK_INVALID_INPUT); }); other.join();
    CHECK(main_error==glrmask_error_message());
    throws([&]{ Vocab duplicate({{0,"a"},{0,"b"}}); }, GLRMASK_INVALID_INPUT);
    throws([]{ Grammar::from_lark("start: ("); });
    for (const auto* source : {"start: <[1]>","start: <[1,3-8]>","start: <[]>","start: <[^0]>","start: <[*]>","start: @token(7)"}) {
        throws([&]{ Grammar::from_lark(source); }, GLRMASK_SYNTAX);
    }
    throws([]{ Grammar::from_json_schema("{"); });
    throws([]{ Grammar::from_glrm("invalid source"); });
    glrmask_grammar* invalid_source = nullptr;
    CHECK(glrmask_grammar_from_lark("start: (", 9, &invalid_source) == GLRMASK_SYNTAX);
    CHECK(invalid_source == nullptr);
}

#if defined(__linux__)
int thread_stack_probe(std::size_t stack_bytes) {
    auto v = vocabulary(); std::vector<Constraint> constraints;
    for (unsigned i=0; i<3; ++i) {
        auto m = Module(simple());
        constraints.push_back(m.compile(v, strategy_options(i==0?0:i==1?3:4)));
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
                auto snapshot = s; CHECK(s.commit_token(31)); CHECK(s.is_accepting());
                CHECK(snapshot.commit_token(0)); auto before = snapshot.mask(); CHECK(snapshot.forced_bytes()==Bytes{static_cast<uint8_t>('b')}); CHECK(snapshot.mask()==before);
                CHECK(snapshot.commit_token(1)); CHECK(snapshot.is_accepting());
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

void commit_contract() {
    auto v = vocabulary();
    for (unsigned i=0; i<3; ++i) {
        auto c = Module(simple()).compile(v, strategy_options(i==0?0:i==1?3:4));
        auto s = c.start();
        CHECK(!s.commit_token(1));
        CHECK(!s.is_accepting()); CHECK(s.is_terminated());
        for(auto w:s.mask()) CHECK(w==0); CHECK(s.forced_bytes().empty());
        CHECK(!s.commit_token(0)); CHECK(!s.commit_bytes("a"));
        auto list = c.start();
        CHECK(list.commit_token(0)); CHECK(!list.commit_token(0));
        CHECK(!list.is_accepting()); CHECK(list.is_terminated());
        auto bytes = c.start();
        CHECK(!bytes.is_terminated());
        CHECK(!bytes.commit_bytes("az"));
        CHECK(bytes.is_terminated()); CHECK(!bytes.is_accepting());
        for(auto w:bytes.mask()) CHECK(w==0); CHECK(bytes.forced_bytes().empty());
        CHECK(!bytes.commit_token(1));
        auto closed = c.start();
        auto moved_closed = std::move(closed); (void)moved_closed;
        throws([&]{ (void)closed.commit_token(0); }, GLRMASK_CLOSED);
    }
    // Outcome is the viability of the post-state, including empty input and language.
    {
        auto ordinary = Module(simple());
        auto empty = Module(Grammar::from_glrm("extern token T start root nt root=T"));
        empty.bind("T", TokenSet(std::vector<TokenId>{}));
        for (auto* module : {&ordinary, &empty}) {
            for (unsigned mode : {0u, 3u, 4u}) {
                auto c = module->compile(v, strategy_options(mode));
                CHECK(c.start().is_terminated() == (module == &empty));
                for (const char* prefix : {"", "a", "ab", "z"}) {
                    for (TokenId id : {0u, 1u, 65u, 999u, UINT32_MAX}) {
                        auto s = c.start(); (void)s.commit_bytes(prefix);
                        bool result = s.commit_token(id);
                        CHECK(result == !(s.is_terminated() && !s.is_accepting()));
                    }
                    for (const char* input : {"", "a", "b", "z", "ab"}) {
                        auto s = c.start(); (void)s.commit_bytes(prefix);
                        bool result = s.commit_bytes(input);
                        CHECK(result == !(s.is_terminated() && !s.is_accepting()));
                    }
                }
            }
        }
    }
    // A negative outcome is OK with allowed=0; null state is an error.
    bool allowed=true;
    CHECK(glrmask_state_commit_token(nullptr, 0, &allowed) == GLRMASK_INVALID_INPUT && allowed == 0);
    CHECK(glrmask_state_commit_bytes(nullptr, nullptr, 0, &allowed) == GLRMASK_INVALID_INPUT && allowed == 0);
    auto c = Module(simple()).compile(v);
    glrmask_state* raw = nullptr;
    CHECK(glrmask_constraint_start(c.native_handle(), &raw) == GLRMASK_OK);
    CHECK(glrmask_state_commit_token(raw, 1, &allowed) == GLRMASK_OK && allowed == 0);
    bool accepting = 1;
    CHECK(glrmask_state_is_accepting(raw, &accepting) == GLRMASK_OK); CHECK(accepting == 0);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>("a");
    CHECK(glrmask_state_commit_bytes(raw, bytes, 1, &allowed) == GLRMASK_OK && allowed == 0);
    glrmask_state_free(raw);
}

void emit_parity() {
    auto v = vocabulary(); std::cout << '[';
    for (unsigned i=0; i<3; ++i) {
        if (i) std::cout << ','; std::cout << "{\"mode\":" << i << ",\"states\":[";
        auto m = Module(simple());
        auto s = m.compile(v, strategy_options(i==0?0:i==1?3:4)).start();
        for (unsigned step=0; step<3; ++step) {
            if (step) { CHECK(s.commit_token(step-1)); std::cout << ','; }
            const auto mask_words = s.mask(); std::cout << "{\"mask\":[";
            for (std::size_t j=0; j<mask_words.size(); ++j) { if (j) std::cout << ','; std::cout << mask_words[j]; }
            std::cout << "],\"accepting\":" << (s.is_accepting()?"true":"false")
                      << ",\"terminated\":" << (s.is_terminated()?"true":"false") << ",\"maskBitLength\":" << s.mask_bit_len() << '}';
        } std::cout << "]}";
    } std::cout << "]\n";
}

Bytes read_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary); CHECK(in.good()); return Bytes(std::istreambuf_iterator<char>(in), {});
}

void artifacts(const std::filesystem::path& directory, bool check) {
    std::filesystem::create_directories(directory); auto v = vocabulary();
    for (unsigned i=0; i<3; ++i) {
        auto m = Module(simple());
        const auto options = strategy_options(i==0?0:i==1?3:4);
        m.precompile(v, options);
        auto wrong = Vocab({{0,"z"}});
        throws([&]{ m.precompile(wrong, options); }, GLRMASK_VOCAB_MISMATCH);
        throws([&]{ m.compile(wrong, options); }, GLRMASK_VOCAB_MISMATCH);
        auto c = m.compile(v, options);
        const auto path = directory/((check?"node-mode-":"cpp-mode-")+std::to_string(i)+".glr");
        if (check) {
            auto loaded = Constraint::load(read_bytes(path), v);
            CHECK(loaded.start().mask()==c.start().mask());
            auto s = loaded.start(); CHECK(s.commit_token(31)); CHECK(s.is_accepting());
        }
        else {
            const auto bytes = c.save(); std::ofstream out(path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            CHECK(out.good());
        }
    }
}

void canonical_modules() {
    auto v = Vocab({{0,"x"},{1,"a"},{2,"y"},{3,"xay"},{4,"b"}});
    auto host = Grammar::from_glrm("glrm 1; start start; extern grammar child; nt start = \"x\" child \"y\";");
    auto parent = Module(host); parent.precompile(v);
    auto original = parent; // copy constructor is a cheap snapshot

    auto child = Module(Grammar::from_lark("start: \"a\""));
    CHECK(MaskStrategy::Auto != MaskStrategy::FastCompile);
    CompileOptions child_options = strategy_options(4);
    child.precompile(v, child_options); parent.bind("child", child);

    auto loaded = Module::load(parent.save());
    CHECK(loaded.externs().unbound().empty());
    CHECK(original.externs().unbound().get("child")->name()=="child");
    CHECK(loaded.externs().get("child")->name()=="child");

    // Retained work survives without storing per-module settings.
    auto child_bytes = child.save();
    auto child_loaded = Module::load(child_bytes);
    CHECK(child_loaded.externs().empty());

    auto state = loaded.compile(v, child_options).start(); { auto owner1 = std::move(loaded); auto owner2 = std::move(parent); auto owner3 = std::move(child); }
    CHECK(state.commit_token(3)); CHECK(state.is_accepting());

    bool failed = false; try { original.bind("missing", Grammar::from_lark("start: \"a\"")); } catch(const Error&) { failed=true; } CHECK(failed);
    CHECK(original.externs().unbound().get("child")->name()=="child");

    // Qualified paths rejected
    failed = false; try { original.bind("child::nested", Grammar::from_lark("start: \"a\"")); } catch(const Error&) { failed=true; } CHECK(failed);

    // Rebinding error
    auto test_mod = Module(host);
    test_mod.bind("child", Grammar::from_lark("start: \"a\""));
    failed = false; try { test_mod.bind("child", Grammar::from_lark("start: \"b\"")); } catch(const Error&) { failed=true; } CHECK(failed);

    // Expressions
    auto expr = Expr::sequence({
        Grammar::from_lark("start: \"x\""),
        Expr::choice({Grammar::from_lark("start: \"a\""), Grammar::from_lark("start: \"b\"")}),
        Grammar::from_lark("start: \"y\"")
    });
    auto host2 = Grammar::from_glrm("glrm 1; start start; extern grammar child; nt start = child;");
    auto m_expr = Module(host2);
    m_expr.bind("child", expr);
    auto c = m_expr.compile(v); auto s = c.start(); CHECK(s.commit_token(3)); CHECK(s.is_accepting());

    // Expr literal
    auto lit = Expr::literal("xay");
    auto m_lit = Module(host2);
    m_lit.bind("child", lit);
    auto c_lit = m_lit.compile(v);
    auto s_lit = c_lit.start(); CHECK(s_lit.commit_token(3)); CHECK(s_lit.is_accepting());

    // Externs ordering and unbound view
    auto multi = Grammar::from_glrm("glrm 1; start start; extern token first; extern grammar second; extern token third; nt start = first second third;");
    auto multi_mod = Module(multi);
    auto ext_all = multi_mod.externs();
    CHECK(ext_all.size() == 3);
    CHECK(ext_all == multi_mod.externs());
    CHECK(ext_all == ext_all.unbound());
    CHECK((ext_all.begin()+0)->name() == "first" && (ext_all.begin()+0)->kind() == ExternKind::Token);
    CHECK((ext_all.begin()+1)->name() == "second" && (ext_all.begin()+1)->kind() == ExternKind::Grammar);
    CHECK((ext_all.begin()+2)->name() == "third" && (ext_all.begin()+2)->kind() == ExternKind::Token);

    multi_mod.bind("second", Grammar::from_lark("start: \"a\""));
    auto ext_unbound = multi_mod.externs().unbound();
    CHECK(ext_unbound.size() == 2);
    CHECK(ext_unbound == ext_unbound.unbound());
    CHECK(ext_all != multi_mod.externs());
    CHECK((ext_unbound.begin()+0)->name() == "first");
    CHECK((ext_unbound.begin()+1)->name() == "third");

    // A grammar's unbound view carries its own unbound externs.
    auto grammar_externs = Grammar::from_glrm("glrm 1; start start; extern grammar one; extern grammar two; nt start = one two;");
    auto grammar_module = Module(grammar_externs);
    grammar_module.bind("one", Expr::literal("a"));
    auto grammar_unbound = grammar_module.externs().unbound();
    CHECK(grammar_unbound.size() == 1 && (grammar_unbound.begin()+0)->name() == "two");
    CHECK(grammar_externs.externs().unbound().size() == 2);
    CHECK(grammar_externs.externs().size() == 2);

    // TokenSet equality compares the normalized ID sets.
    CHECK(TokenSet({1,2}) == TokenSet::from_ranges({{1,3}}));
    CHECK(TokenSet({1,2}) != TokenSet::from_ranges({{1,2}}));
    CHECK(!(TokenSet({1,2}) != TokenSet::from_ranges({{1,3}})));

    auto open_host = Module(host); open_host.precompile(v);
    auto precompiled_child = Module(Grammar::from_lark("start: \"a\""));
    precompiled_child.precompile(v, strategy_options(4));
    CHECK(precompiled_child.save().size() > Module(Grammar::from_lark("start: \"a\"")).save().size());
    open_host.bind("child", precompiled_child);
    auto precompiled_state = open_host.compile(
        v, strategy_options(3)).start();
    CHECK(allowed(precompiled_state.mask(), 3));
    CHECK(precompiled_state.commit_token(3)); CHECK(precompiled_state.is_accepting());

    // All three cost preferences preserve the same language.
    for(auto mask : {MaskStrategy::Auto, MaskStrategy::FastCompile, MaskStrategy::FastMasks})
        {
            auto c = Grammar::from_lark("start: \"a\"").compile(v, CompileOptions{.mask_strategy=mask});
            auto state=c.start();
            CHECK(allowed(state.mask(), 1));
            CHECK(state.mask().size()==(state.mask_bit_len()+31)/32 && state.mask_bit_len()==c.mask_bit_len());
        }
    CHECK(std::hash<MaskStrategy>{}(MaskStrategy::Auto) == std::hash<MaskStrategy>{}(MaskStrategy::Auto));

}

void precompile_vocab_lock() {
    Vocab v({{0,"a"},{1,""},{2,"b"}}), wrong({{0,"b"},{1,""},{2,"a"}});
    for(auto mode : {MaskStrategy::Auto, MaskStrategy::FastCompile, MaskStrategy::FastMasks}) {
        CompileOptions options; options.mask_strategy=mode;
        auto root=Module(Grammar::from_glrm("start s extern grammar BODY extern token END nt s = BODY END"));
        root.precompile(v,options);
        auto m=Module::load(root.save());
        throws([&]{m.precompile(wrong,options);},GLRMASK_VOCAB_MISMATCH);
        m.bind("BODY",Grammar::from_lark(R"(start: "a")"));
        m.bind("END",TokenSet({1}));
        auto outer=Module(Grammar::from_glrm("start s extern grammar INNER nt s = INNER"));
        outer.bind("INNER",m);
        auto loaded=Module::load(outer.save());
        for(auto* candidate : {&m,&outer,&loaded}) {
            throws([&]{candidate->compile(wrong,options);},GLRMASK_VOCAB_MISMATCH);
            throws([&]{candidate->precompile(wrong,options);},GLRMASK_VOCAB_MISMATCH);
            auto state=candidate->compile(v,options).start();
            CHECK(state.commit_token(0)); CHECK(state.commit_token(1)); CHECK(state.is_accepting());
        }
    }
}

void direct_constraint_binding() {
    Vocab v({{0,"x"},{1,"a"},{2,"y"}});
    Module body(Grammar::from_glrm("start s extern grammar item nt s = item"));
    body.bind("item", Expr::sequence({Grammar::from_lark(R"(start: "a")"), TokenSet({63})}));
    for(auto child_mode : {MaskStrategy::Auto, MaskStrategy::FastCompile, MaskStrategy::FastMasks}) {
        CompileOptions child_options; child_options.mask_strategy=child_mode;
        auto child=body.compile(v, child_options);
        auto host=Grammar::from_glrm(R"(start s extern grammar item nt s = "x" item "y")");
        Module direct(host), wrapped(host);
        direct.bind("item",child); wrapped.bind("item",Expr::sequence({child}));
        for(auto mode : {MaskStrategy::Auto, MaskStrategy::FastCompile, MaskStrategy::FastMasks}) {
            CompileOptions options; options.mask_strategy=mode;
            auto a=direct.compile(v,options).start(), b=wrapped.compile(v,options).start();
            for(TokenId id : {0u,1u,63u,2u}) {
                CHECK(a.mask()==b.mask()); CHECK(allowed(a.mask(),id));
                CHECK(a.commit_token(id)); CHECK(b.commit_token(id));
            }
            CHECK(a.mask()==b.mask()); CHECK(a.is_accepting() && b.is_accepting());
        }
    }
}

void singleton_expression_binding() {
    Vocab v({{0,"a"}});
    auto g=Grammar::from_lark(R"(start: "a")");
    auto m=Module(g); auto c=m.compile(v);
    auto literal=Expr::literal("a"); auto tokens=TokenSet({99});
    auto parity=[&](const auto& value, const char* kind, TokenId id) {
        auto h=Grammar::from_glrm(std::string("extern ")+kind+" T start root nt root=T");
        Module direct(h); direct.bind("T",value);
        for (auto expr : {Expr::choice({value}), Expr::sequence({value})}) {
            Module wrapped(h); wrapped.bind("T",expr);
            auto a=direct.compile(v).start(), b=wrapped.compile(v).start();
            CHECK(a.mask()==b.mask()); CHECK(a.commit_token(id)); CHECK(b.commit_token(id));
            CHECK(a.mask()==b.mask() && a.is_accepting() && b.is_accepting());
        }
    };
    parity(g,"grammar",0); parity(m,"grammar",0); parity(c,"grammar",0);
    parity(literal,"grammar",0); parity(tokens,"token",99);
}

void mask_bit_domain_lengths() {
    for (std::size_t n : {0u,33u,63u,64u}) {
        std::vector<VocabEntry> entries;
        if (n) entries.emplace_back(static_cast<TokenId>(n-1),"a");
        auto v = Vocab(entries);
        auto c = Grammar::from_lark(R"(start: "a")").compile(v);
        auto s = c.start();
        CHECK(c.mask_bit_len()==n && s.mask_bit_len()==n);
        auto packed=s.mask(); std::size_t w=(n+31)/32;
        CHECK(packed.size()==w);
        if (n) CHECK(allowed(packed,static_cast<TokenId>(n-1)));
        for (std::size_t len : {w,w+2}) {
            std::vector<TokenId> out(len,0xffffffff);
            s.fill_mask(out);
            CHECK(std::equal(packed.begin(),packed.end(),out.begin()));
            CHECK(std::all_of(out.begin()+w,out.end(),[](auto x){return x==0;}));
        }
        if(w){std::vector<TokenId> out(w-1,123);throws([&]{s.fill_mask(out);},GLRMASK_BUFFER_TOO_SMALL);CHECK(std::all_of(out.begin(),out.end(),[](auto x){return x==123;}));CHECK(s.commit_token(static_cast<TokenId>(n-1)));s.fill_mask(packed);CHECK(packed==s.mask());}
    }
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
        direct_constraint_binding();
        singleton_expression_binding();
        precompile_vocab_lock();
        basic();
        mask_bit_domain_lengths();
        exact_and_unicode();
        end_tokens();
        reject_module_expr_in_grammar();
        composition_and_persistence();
        frontends();
        threads_and_errors();
        commit_contract();
        std::cout << "C++ binding regression groups: 8 passed\n"; return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
