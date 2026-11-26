// A generation loop: mask the model's scores, take the best allowed token,
// stop if EOS is chosen, otherwise commit it.
//
// Random numbers stand in for the model.
#include <glrmask.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main() {
    try {
        // One token per character, plus a caller-managed EOS ID.
        const std::string chars = "0123456789+*()";
        const glrmask::TokenId eos = static_cast<glrmask::TokenId>(chars.size());
        std::vector<glrmask::VocabEntry> entries;
        for (glrmask::TokenId id = 0; id < eos; ++id) entries.emplace_back(id, chars.substr(id, 1));
        glrmask::Vocab vocab(entries);

        // Arithmetic grammar.
        auto arithmetic = glrmask::Grammar::from_glrm(R"glrm(
            start sum
            t NUMBER = /[0-9]+/
            nt sum = product | sum "+" product
            nt product = atom | product "*" atom
            nt atom = NUMBER | "(" sum ")"
        )glrm");
        auto constraint = arithmetic.compile(vocab);

        std::mt19937 rng(7);
        std::uniform_real_distribution<float> model(0.0f, 1.0f);
        auto state = constraint.start();
        std::vector<std::uint32_t> mask(std::max((constraint.mask_bit_len() + 31) / 32, static_cast<std::size_t>(eos / 32 + 1)));
        std::string text;
        while (true) {
            std::vector<float> scores(eos + 1);
            for (float& score : scores) score = model(rng);  // Your model's scores go here.
            // This model grows keener to stop as the text gets longer.
            scores[eos] += 0.1f * static_cast<float>(text.size());
            state.fill_mask(mask);
            if (state.is_accepting()) mask[eos / 32] |= std::uint32_t{1} << (eos % 32);
            auto allowed = [&](glrmask::TokenId id) { return (mask[id / 32] >> (id % 32)) & 1; };
            glrmask::TokenId token = eos;
            bool found = false;
            for (glrmask::TokenId id = 0; id <= eos; ++id) {
                if (allowed(id) && (!found || scores[id] > scores[token])) {
                    token = id;
                    found = true;
                }
            }
            if (!found) return 2;
            if (token == eos) break;
            if (!state.commit_token(token)) return 2;
            text += chars[token];
        }
        if (!state.is_accepting()) return 2;
        std::cout << text << '\n';
        return 0;
    } catch (const glrmask::Error& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
