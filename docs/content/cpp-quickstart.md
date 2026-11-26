```cpp
#include <glrmask.hpp>
#include <iostream>
#include <string>
#include <vector>

int main() {
    auto grammar = glrmask::Grammar::from_json_schema(R"({
        "type": "object",
        "properties": {"color": {"enum": ["red", "green", "blue"]}},
        "required": ["color"],
        "additionalProperties": false
    })");

    // One token per byte; with a real model, use its tokenizer.
    std::vector<glrmask::VocabEntry> entries;
    for (glrmask::TokenId id = 0; id < 256; ++id)
        entries.emplace_back(id, glrmask::Bytes{static_cast<std::uint8_t>(id)});
    glrmask::Vocab vocab(entries);
    const glrmask::TokenId eos = 256;

    auto state = grammar.compile(vocab).start();
    std::string output;
    const std::string sample = R"({"color": "blue"})";
    while (true) {
        // Stand-in scores for a model that wants to produce sample.
        std::vector<float> logits(257, -1.0f);
        auto next = output.size() < sample.size()
            ? static_cast<glrmask::TokenId>(sample[output.size()]) : eos;
        logits[next] = 1.0f;
        auto mask = state.mask();
        glrmask::TokenId token = 0;
        bool found = false;
        for (glrmask::TokenId id = 0; id <= eos; ++id) {
            bool allowed = id == eos ? state.is_accepting()
                : ((mask[id / 32] >> (id % 32)) & 1) != 0;
            if (allowed && (!found || logits[id] > logits[token])) {
                token = id;
                found = true;
            }
        }
        if (!found) return 1;
        if (token == eos) break;
        if (!state.commit_token(token)) return 1;
        output += static_cast<char>(token);
    }
    std::cout << output << '\n';
}
```
