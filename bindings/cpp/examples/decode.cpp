#include <glrmask.hpp>
#include <iostream>
int main() {
    try {
        glrmask::Vocab vocab({{0,"hello"},{1," "},{2,"world"}});
        auto grammar=glrmask::Grammar::from_ebnf(R"(start ::= "hello" " " "world")");
        auto constraint=grammar.compile(vocab);
        auto state=constraint.start();
        std::vector<glrmask::TokenId> mask(constraint.mask_len());
        for (glrmask::TokenId token : {0u,1u,2u}) {
            state.fill_mask(mask);
            if ((mask[token/32] & (std::uint32_t{1} << (token%32))) == 0) return 2;
            state.commit_token(token);
        }
        std::cout << "GLRMask " << glrmask::version() << ": accepting=" << state.is_accepting() << '\n';
        return state.is_accepting() ? 0 : 3;
    } catch (const glrmask::Error& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
