#include <glrmask.hpp>
#include <stdexcept>
#include <iostream>
int main() {
    try {
        glrmask::Vocab v({{0,"a"},{31,"ab"},{1,"b"}});
        auto c=glrmask::Grammar::from_ebnf(R"(start ::= "a" "b")").compile(v);
        auto s=c.start();c.dispose();
        if ((s.mask()[0]&0x80000000u)==0) throw std::runtime_error("high-bit mask lost");
        s.commit_token(31);
        if (!s.is_accepting()) throw std::runtime_error("not accepted");
        std::cout<<"Installed C++ consumer passed\n";return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
