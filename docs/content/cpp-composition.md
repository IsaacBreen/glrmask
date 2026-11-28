```cpp
// [hidden]
#include <glrmask.hpp>
#include <cassert>
using namespace glrmask;
int main() {
std::vector<VocabEntry> token_bytes;
for (TokenId id=0; id<256; ++id) token_bytes.emplace_back(id, Bytes{static_cast<std::uint8_t>(id)});
// [/hidden]
// Token ID -> exact bytes from tokenizer
Vocab vocab(token_bytes);

auto weather = Grammar::from_json_schema(
    R"({
        "type": "object",
        "properties": {
            "city": {"type": "string"}
        },
        "required": ["city"],
        "additionalProperties": false
    })"
).compile(vocab);

auto parent = Grammar::from_glrm(R"(
    extern grammar ARGS
    start call
    nt call = "<weather>" ARGS
              "</weather>"
)");
Module call(parent);
call.bind("ARGS", weather);
auto constraint = call.compile(vocab);
// [hidden]
auto state = constraint.start();
if (!state.commit_bytes(R"(<weather>{"city": "Perth"}</weather>)") || !state.is_accepting()) return 1;
}
// [/hidden]
```
