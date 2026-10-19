#include <glrmask.hpp>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <thread>
using namespace glrmask;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string(__FILE__)+":"+std::to_string(__LINE__)+": " #x); } while(false)
template<class F> void throws(F&& fn,glrmask_status status=-1) {
    bool thrown=false; try { fn(); } catch(const Error& e) {thrown=true; CHECK(status<0 || e.status()==status);}
    CHECK(thrown);
}
bool allowed(const std::vector<TokenId>& m,TokenId id) {return id/32<m.size() && (m[id/32]&(TokenId{1}<<(id%32)))!=0;}
Vocab vocabulary() {return Vocab({{0,"a"},{1,"b"},{31,"ab"},{32,"a"},{65,"z"}});}
BuildOptions mode(unsigned i) {BuildOptions o; o.optimization=static_cast<Optimization>(i); return o;}
auto simple() {return Grammar::from_ebnf(R"(start ::= "a" "b")");}
void basic() {
    for(unsigned i=0;i<4;++i) {
        auto v=vocabulary(); auto g=simple(); auto c=g.compile(v,mode(i));
        auto s=c.start(); const auto initial=s.mask();
        CHECK(allowed(initial,0)); CHECK(allowed(initial,31)); CHECK(allowed(initial,32));
        CHECK(!allowed(initial,1)); CHECK(!allowed(initial,65)); CHECK(s.mask_len()==3);
        std::vector<TokenId> buffer(s.mask_len()+2,0xffffffff);
        s.fill_mask(buffer); CHECK(std::equal(initial.begin(),initial.end(),buffer.begin()));
        CHECK(buffer[3]==0 && buffer[4]==0);
        std::vector<TokenId> short_buffer(1,123); throws([&]{s.fill_mask(short_buffer);},GLRMASK_BUFFER_TOO_SMALL);
        CHECK(short_buffer[0]==123); CHECK(s.mask()==initial);
        auto copy=s.clone(); c.dispose(); g.dispose(); v.dispose();
        s.commit_token(31); CHECK(s.is_accepting()); CHECK(!copy.is_accepting());
        copy.commit_token(0); CHECK(copy.forced_tokens()==std::vector<TokenId>{1});
        CHECK(!copy.is_accepting()); copy.commit_token(1); CHECK(copy.is_accepting()); CHECK(!copy.is_rejected());
        auto moved=std::move(copy); throws([&]{copy.mask();},GLRMASK_CLOSED); CHECK(moved.is_accepting());
        s.dispose(); s.dispose(); throws([&]{s.mask();},GLRMASK_CLOSED);
    }
}
void exact_and_unicode() {
    const Bytes first{0xc3},last{0xa9},whole{0xc3,0xa9};
    Vocab v({{0,first},{1,last},{31,whole},{32,whole},{40,Bytes{0,255}}},{99});
    const std::string source="start ::= \""+std::string(whole.begin(),whole.end())+"\"";
    for(unsigned i=0;i<4;++i) {
        auto c=Grammar::from_ebnf(source).compile(v,mode(i)); auto s=c.start();
        CHECK(allowed(s.mask(),0)); CHECK(allowed(s.mask(),31));
        s.commit_bytes(first); CHECK(!s.is_accepting()); CHECK(allowed(s.mask(),1));
        s.commit_bytes(last); CHECK(s.is_accepting());
        auto exact=Grammar::exact(v.tokens({31})).compile(v,mode(i));
        CHECK(allowed(exact.start().mask(),31)); CHECK(!allowed(exact.start().mask(),32));
        auto e=exact.start(); e.commit_token(31); CHECK(e.is_accepting());
        auto raw=Grammar::exact(v.tokens({40})).compile(v,mode(i)).start(); raw.commit_token(40); CHECK(raw.is_accepting());
        auto control=Grammar::exact(v.tokens({99})).compile(v,mode(i)).start(); control.commit_token(99); CHECK(control.is_accepting());
    }
    auto ranges=v.token_ranges({{31,32},{32,32}}); CHECK(ranges.size()==2);
    CHECK(ranges.ranges().size()==1); CHECK(ranges.ranges()[0].first==31 && ranges.ranges()[0].last==32);
    auto ids=ranges.ids(); ranges.dispose(); CHECK(ids==std::vector<TokenId>({31,32}));
    throws([&]{v.tokens({31,31});}); throws([&]{v.token_ranges({{2,0}});});
    Vocab maximum({}, {0xffffffff}); auto high=maximum.token_ranges({{0xffffffff,0xffffffff}});
    CHECK(high.size()==1 && high.ranges()[0].last==0xffffffff);
    CHECK(v.tokens({}).empty());
}
void end_tokens() {
    Vocab v({{0,"a"}},{99});
    for(unsigned i=0;i<4;++i) {
        auto o=mode(i); o.end_tokens={99};
        auto s=Grammar::from_ebnf(R"(start ::= "a")").compile(v,o).start();
        CHECK(!allowed(s.mask(),99));s.commit_token(0);CHECK(s.is_accepting());CHECK(allowed(s.mask(),99));
        s.commit_token(99); CHECK(s.is_accepting() && !s.is_rejected());
        const auto m=s.mask(); CHECK(std::all_of(m.begin(),m.end(),[](auto w){return w==0;}));
        CHECK(s.clone().is_accepting()); throws([&]{s.commit_token(99);});
    }
}
void composition_and_persistence() {
    Vocab v({{0,"x"},{1,"a"},{2,"b"},{3,"y"},{4,"xay"},{5,"xby"}});
    auto host=Grammar::from_glrm(R"(glrm 1; start root; extern grammar child; nt root = "x" child "y";)");
    auto req=host.requirements(); CHECK(req.size()==1 && req[0].name=="child");
    CHECK(req[0].kind==RequirementKind::Grammar && req[0].source_span.has_value());
    auto a=Grammar::from_ebnf(R"(start ::= "a")"); auto b=Grammar::from_ebnf(R"(start ::= "b")");
    auto left=host.clone(),right=host.clone(); left.bind("child",a); right.bind("child",b);
    CHECK(host.requirements().size()==1 && left.requirements().empty());
    throws([&]{left.bind("child",b);}); CHECK(left.requirements().empty());
    auto lp=host.prepare(v); auto ap=a.prepare(v); lp.bind("child",ap);
    const auto prepared_bytes=lp.save(); lp.dispose(); ap.dispose();
    auto restored=PreparedGrammar::load(prepared_bytes,&v); CHECK(restored.requirements().empty());
    for(unsigned i=0;i<4;++i) {
        auto lc=left.compile(v,mode(i)); auto rc=right.compile(v,mode(i)); auto pc=restored.link(mode(i));
        CHECK(allowed(lc.start().mask(),4) && !allowed(lc.start().mask(),5));
        CHECK(allowed(rc.start().mask(),5) && !allowed(rc.start().mask(),4));
        CHECK(pc.start().mask()==lc.start().mask());
        auto saved=lc.save(); auto loaded=Constraint::load(saved); CHECK(loaded.start().mask()==lc.start().mask());
        auto external=lc.save(true); throws([&]{Constraint::load(external);});
        CHECK(Constraint::load(external,&v).start().mask()==lc.start().mask());
        Vocab wrong({{0,"wrong"},{1,"a"},{2,"b"},{3,"y"},{4,"xay"},{5,"xby"}});
        throws([&]{Constraint::load(external,&wrong);});
    }
    auto leaf=Grammar::from_glrm(R"(glrm 1; start root; extern token MARK; nt root = MARK;)");
    auto nested=host.clone(); nested.bind("child",leaf); leaf.bind("MARK",Token(2));
    CHECK(nested.requirements().size()==1 && nested.requirements()[0].name=="child::MARK");
    nested.bind("child::MARK",Token(1)); CHECK(nested.requirements().empty());
    auto s=nested.compile(v).start(); CHECK(!allowed(s.mask(),4));
    s.commit_token(0);s.commit_token(1);s.commit_token(3);CHECK(s.is_accepting());
    throws([&]{host.bind("missing",a);}); CHECK(host.requirements().size()==1);
}
void markers_and_frontends() {
    auto v=vocabulary();
    for(unsigned i=0;i<4;++i) {
        auto g=Grammar::from_glrm(R"(glrm 1; start root; nt root = @mark(10) "a" @mark(20) "b" @mark(30);)");
        auto c=g.compile(v,mode(i)); CHECK(c.marker_ids()==std::vector<TokenId>({10,20,30}));
        auto s=c.start(); CHECK(s.marker_events().size()==1 && s.marker_events()[0].byte_offset==0);
        s.commit_token(31); const auto events=s.marker_events();
        CHECK(events.size()==2 && events[0].id==20 && events[0].byte_offset==1 && events[1].id==30 && events[1].byte_offset==2);
        CHECK(s.clone().marker_events().size()==2);
        auto load=Constraint::load(c.save()); CHECK(load.marker_ids()==c.marker_ids());
        auto l=Grammar::from_lark("start: \"a\" \"b\"").compile(v,mode(i)).start();l.commit_token(31);CHECK(l.is_accepting());
    }
    Vocab special({{0,"a"},{1,"a"}},{99});
    LarkOptions named; named.profile=LarkProfile::Llm; named.token_names={{"begin",{99}},{"body",{1}}};
    for(unsigned i=0;i<4;++i) {
        auto state=Grammar::from_lark("start: <|begin|> <|body|>",named).compile(special,mode(i)).start();
        CHECK(allowed(state.mask(),99));state.commit_token(99);
        CHECK(allowed(state.mask(),1) && !allowed(state.mask(),0));state.commit_token(1);CHECK(state.is_accepting());
    }
    Vocab json({{0,"null"},{1,"true"}}); auto js=Grammar::from_json_schema(R"({"type":"null"})").compile(json).start();
    CHECK(allowed(js.mask(),0) && !allowed(js.mask(),1));js.commit_token(0);CHECK(js.is_accepting());
}
void threads_and_errors() {
    auto v=vocabulary();auto c=simple().compile(v);std::atomic<bool> okay{true};std::vector<std::thread> threads;
    for(int i=0;i<4;++i) threads.emplace_back([&]{try{for(int j=0;j<8;++j){auto s=c.start();s.commit_token(31);if(!s.is_accepting())okay=false;}}catch(...){okay=false;}});
    for(auto& thread:threads)thread.join();CHECK(okay);
    std::size_t ignored=0;CHECK(glrmask_vocab_len(nullptr,&ignored)==GLRMASK_INVALID_ARGUMENT);
    const std::string main_error=glrmask_error_message();
    std::thread other([]{glrmask_grammar* p=nullptr;const char invalid[]={char(0xff)};glrmask_grammar_from_source(0,invalid,1,nullptr,&p);});other.join();
    CHECK(main_error==glrmask_error_message());
    throws([&]{Vocab duplicate({{0,"a"},{0,"b"}});},GLRMASK_INVALID_ARGUMENT);
    throws([&]{Vocab overlap({{0,"a"}},{0});},GLRMASK_INVALID_ARGUMENT);
}
void emit_parity() {
    auto v=vocabulary(); std::cout << '[';
    for(unsigned i=0;i<4;++i) {
        if(i)std::cout << ',';std::cout << "{\"mode\":" << i << ",\"states\":[";
        auto s=simple().compile(v,mode(i)).start();
        for(unsigned step=0;step<3;++step) {
            if(step){s.commit_token(step-1);std::cout << ',';}
            const auto m=s.mask();std::cout << "{\"mask\":[";
            for(std::size_t j=0;j<m.size();++j){if(j)std::cout << ',';std::cout << m[j];}
            std::cout << "],\"accepting\":" << (s.is_accepting()?"true":"false") << ",\"rejected\":" << (s.is_rejected()?"true":"false") << '}';
        }std::cout << "]}";
    }std::cout << "]\n";
}
Bytes read_bytes(const std::filesystem::path& path) {
    std::ifstream in(path,std::ios::binary);CHECK(in.good());return Bytes(std::istreambuf_iterator<char>(in),{});
}
void artifacts(const std::filesystem::path& directory,bool check) {
    std::filesystem::create_directories(directory);auto v=vocabulary();
    for(unsigned i=0;i<4;++i) {
        auto c=simple().compile(v,mode(i));
        for(bool external:{false,true}) {
            const auto path=directory/((check?"node-":"cpp-")+std::string(external?"external-":"mode-")+std::to_string(i)+".glr");
            if(check) {auto loaded=Constraint::load(read_bytes(path),external?&v:nullptr);CHECK(loaded.start().mask()==c.start().mask());auto s=loaded.start();s.commit_token(31);CHECK(s.is_accepting());}
            else {const auto bytes=c.save(external);std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));CHECK(out.good());}
        }
    }
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--parity"){emit_parity();return 0;}
        if(argc==3 && std::string(argv[1])=="--save"){artifacts(argv[2],false);return 0;}
        if(argc==3 && std::string(argv[1])=="--check"){artifacts(argv[2],true);return 0;}
        CHECK(argc==1);
        std::cerr << "C++ regression: basic" << std::endl; basic();
        std::cerr << "C++ regression: exact_and_unicode" << std::endl; exact_and_unicode();
        std::cerr << "C++ regression: end_tokens" << std::endl; end_tokens();
        std::cerr << "C++ regression: composition_and_persistence" << std::endl; composition_and_persistence();
        std::cerr << "C++ regression: markers_and_frontends" << std::endl; markers_and_frontends();
        std::cerr << "C++ regression: threads_and_errors" << std::endl; threads_and_errors();
        std::cout << "C++ binding regression groups: 6 passed\n";return 0;
    } catch(const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
}
