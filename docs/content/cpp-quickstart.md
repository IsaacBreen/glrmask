```cpp
#include <glrmask.hpp>
#include <cassert>
#include <utility>
#include <vector>

int main() {
    glrmask::Vocab vocab({{0, "hello"}, {1, " "}, {2, "world"}});
    auto grammar = glrmask::Grammar::from_lark(
        R"(start: "hello" " " "world")");
    auto constraint = grammar.compile(vocab);
    auto state = constraint.start();
    std::vector<glrmask::TokenId> mask(constraint.mask_len());

    for (auto [id, expected] : std::vector<std::pair<glrmask::TokenId, glrmask::TokenId>>{
             {0, 1}, {1, 2}, {2, 4}}) {
        state.fill_mask(mask);
        assert((mask[0] & 7) == expected);
        assert(state.commit_token(id));
    }
    assert(state.is_accepting());
}
```
