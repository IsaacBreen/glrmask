```cpp
// [hidden]
#include <glrmask.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
using namespace glrmask;
constexpr TokenId EOS = 256;
std::vector<float> model(const std::vector<TokenId>& output) {
    const std::string target = R"({"color": "blue"})";
    std::vector<float> logits(257, -1.0f);
    logits[output.size() < target.size() ? static_cast<TokenId>(target[output.size()]) : EOS] = 1;
    return logits;
}
TokenId sample(const std::vector<float>& logits, const std::vector<std::uint32_t>& mask, bool accepting) {
    TokenId token = 0;
    bool found = false;
    for (TokenId id = 0; id <= EOS; ++id) {
        bool allowed = id == EOS ? accepting : ((mask[id / 32] >> (id % 32)) & 1) != 0;
        if (allowed && (!found || logits[id] > logits[token])) { token = id; found = true; }
    }
    if (!found) throw std::runtime_error("no allowed token and cannot stop");
    return token;
}
int main() {
// Replace with your tokenizer's token bytes
std::vector<VocabEntry> token_bytes;
for (TokenId id = 0; id < 256; ++id) token_bytes.emplace_back(id, Bytes{static_cast<std::uint8_t>(id)});
// [/hidden]
auto grammar = Grammar::from_json_schema(R"(
    {
        "type": "object",
        "properties": {
            "color": {
                "enum": [
                    "red", "green", "blue"
                ]
            }
        },
        "required": ["color"],
        "additionalProperties": false
    }
)");

// Token ID -> exact bytes from tokenizer
Vocab vocab(token_bytes);
auto constraint = grammar.compile(vocab);
auto state = constraint.start();

std::vector<TokenId> output;
while (true) {
    auto logits = model(output);
    auto mask = state.mask();
    auto token = sample(
        logits, mask, state.is_accepting()
    );
    if (token == EOS) break;
    if (!state.commit_token(token)) return 1;
    output.push_back(token);
}
// [hidden]
if (!state.is_accepting()) return 1;
std::cout << std::string(output.begin(), output.end()) << '\n';
}
// [/hidden]
```
