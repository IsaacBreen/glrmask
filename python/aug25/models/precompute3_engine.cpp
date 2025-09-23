#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <set>
#include <optional>
#include <string>
#include <utility>
#include <tuple>
#include <sstream>
#include <deque>
#include <algorithm>
#include <functional>
#include <memory>
#include "leveled_gss.hpp"

#include "icl_rangeset.hpp"

// Fast JSON parsing
#include <nlohmann/json.hpp>


using json = nlohmann::json;
using namespace leveled_gss;

namespace py = pybind11;


struct Acc : public std::enable_shared_from_this<Acc> {
    // terminals_union: tokenizer_state_id -> RangeSet of disallowed terminals
    std::unordered_map<int, RangeSet> terminals_union;
    // current allowed LLM mask (RangeSet)
    RangeSet llm_mask;

    bool operator==(const Acc& other) const {
        if (!(llm_mask == other.llm_mask)) return false;
        if (terminals_union.size() != other.terminals_union.size()) return false;
        for (auto const& [key, val] : terminals_union) {
            auto it = other.terminals_union.find(key);
            if (it == other.terminals_union.end() || !(val == it->second)) {
                return false;
            }
        }
        return true;
    }

    std::shared_ptr<Acc> merge(const std::shared_ptr<Acc>& other) const {
        auto n = std::make_shared<Acc>();
        // Union llm masks
        n->llm_mask = llm_mask.union_with(other->llm_mask);

        // Merge terminals_union by RangeSet union per key
        // Optimization: copy the larger map, iterate the smaller one.
        if (terminals_union.size() < other->terminals_union.size()) {
            n->terminals_union = other->terminals_union;
            for (const auto& [key, val] : terminals_union) {
                auto it = n->terminals_union.find(key);
                if (it == n->terminals_union.end()) {
                    n->terminals_union.emplace(key, val);
                } else {
                    it->second = it->second.union_with(val);
                }
            }
        } else {
            n->terminals_union = terminals_union;
            for (const auto& [key, val] : other->terminals_union) {
                auto it = n->terminals_union.find(key);
                if (it == n->terminals_union.end()) {
                    n->terminals_union.emplace(key, val);
                } else {
                    it->second = it->second.union_with(val);
                }
            }
        }
        return n;
    }
};

using leveled_gss::LeveledGSS;

class Engine {
public:
    Engine(py::object tokenizer,
           int tokenizer_initial_state,
           int tokenizer_max_state,
           py::object ignore_terminal_id_or_none,
           const std::string& json_string,
           py::dict possible_matches,             // tsid -> term_id -> _sep1.Bitset
           py::object all_internal_llm_tokens_bitset,
           py::dict internal_to_original_map)
        : tokenizer_(std::move(tokenizer)),
          tokenizer_initial_state_(tokenizer_initial_state),
          tokenizer_max_state_(tokenizer_max_state),
          pmc_(std::move(possible_matches)),
          all_internal_llm_tokens_bitset_(std::move(all_internal_llm_tokens_bitset)) {

        json data = json::parse(json_string);

        if (!ignore_terminal_id_or_none.is_none()) {
            ignore_terminal_id_ = py::cast<int>(ignore_terminal_id_or_none);
        }

        {
            py::list ranges_py = all_internal_llm_tokens_bitset_.attr("to_ranges")();
            std::vector<std::pair<unsigned long long, unsigned long long>> ranges_cpp;
            ranges_cpp.reserve(py::len(ranges_py));
            for (auto r : ranges_py) {
                py::tuple t = py::cast<py::tuple>(r);
                ranges_cpp.emplace_back(py::cast<unsigned long long>(t[0]), py::cast<unsigned long long>(t[1]));
            }
            universe_rangeset_ = RangeSet::from_ranges(ranges_cpp);
        }

        const auto& roots_map_json = data.at("precomputed3");
        for (const auto& item : roots_map_json) {
            roots_map_[item[0].get<int>()] = item[1].get<int>();
        }

        parse_parser_table_from_json(data.at("parser"));

        parse_arena(data.at("trie3_god"));

        // Pre-convert possible_matches into C++ RangeSet map (tsid -> term_id -> RangeSet)
        {
            py::module json = py::module::import("json");
            py::object sep1 = py::module::import("_sep1");
            py::object BitsetClass = sep1.attr("Bitset");

            for (auto item_handle : pmc_.attr("items")()) {
                py::tuple item = py::cast<py::tuple>(item_handle);
                int tsid = py::cast<int>(item[0]);
                py::dict inner = py::cast<py::dict>(item[1]);
                auto &dst = pmc_cpp_[tsid]; // creates entry
                for (auto inner_item_handle : inner.attr("items")()) {
                    py::tuple inner_item = py::cast<py::tuple>(inner_item_handle);
                    int term_id = py::cast<int>(inner_item[0]);
                    py::object bit = py::reinterpret_borrow<py::object>(inner_item[1]);
                    // Convert _sep1.Bitset -> RangeSet
                    py::list ranges_py = bit.attr("to_ranges")();
                    std::vector<std::pair<unsigned long long, unsigned long long>> ranges_cpp;
                    ranges_cpp.reserve(py::len(ranges_py));
                    for (auto r : ranges_py) {
                        py::tuple t = py::cast<py::tuple>(r);
                        ranges_cpp.emplace_back(py::cast<unsigned long long>(t[0]), py::cast<unsigned long long>(t[1]));
                    }
                    dst[term_id] = RangeSet::from_ranges(ranges_cpp);
                }
            }
        }

        // Pre-convert internal_to_original_map into a C++ unordered_map for fast lookups
        {
            for (auto item_handle : internal_to_original_map.attr("items")()) {
                py::tuple item = py::cast<py::tuple>(item_handle);
                unsigned long long internal_id = py::cast<unsigned long long>(item[0]);
                unsigned long long original_id = py::cast<unsigned long long>(item[1]);
                internal_to_original_map_cpp_[internal_id] = original_id;
            }
        }

        // Initialize state: one GSS per tokenizer initial state with start parser state on stack
        auto initial_acc = std::make_shared<Acc>();
        initial_acc->llm_mask = RangeSet::empty();
        // terminals_union empty by default
        std::vector<std::pair<std::vector<int>, std::shared_ptr<Acc>>> initial_stacks;
        initial_stacks.emplace_back(std::vector<int>{}, initial_acc);
        Leveled gss_with_empty_stack = Leveled::from_stacks(initial_stacks);
        Leveled init = gss_with_empty_stack.push(start_state_id_);
        state_[tokenizer_initial_state_] = std::move(init);
    }

    void commit(py::bytes token_bytes) {
        // Build terminals_map and state_map
        std::string token = token_bytes; // implicit cast via pybind11
        std::unordered_map<int, RangeSet> terminals_map; // sid -> RangeSet of matched terminals
        std::unordered_map<int, int> state_map;            // old_sid -> end_sid

        // 1) Probe tokenizer from each active state
        for (auto &kv : state_) {
            int tokenizer_sid = kv.first;
            py::tuple result = tokenizer_.attr("execute_from_state")(py::bytes(token), tokenizer_sid);
            py::object end_state_obj = result[0];
            py::object matches_obj = result[1];

            if (!end_state_obj.is_none()) {
                int end_state = py::cast<int>(end_state_obj);
                state_map[tokenizer_sid] = end_state;
            }

            std::vector<unsigned long long> matched_terminals;
            matched_terminals.reserve(32);
            for (auto tm : matches_obj) {
                py::tuple tmt = py::cast<py::tuple>(tm);
                int terminal_id = py::cast<int>(tmt[0]);
                matched_terminals.push_back(static_cast<unsigned long long>(terminal_id));
            }
            terminals_map[tokenizer_sid] = RangeSet::from_indices(matched_terminals);
        }

        // 2) Prune and map per-state GSS (rename terminals_union keys according to state_map)
        std::unordered_map<int, Leveled> temp_states;
        for (auto &kv : state_) {
            int tokenizer_sid = kv.first;
            const Leveled &gss = kv.second;

            Leveled pruned = prune_by_terminals_map(gss, terminals_map);
            if (!pruned.is_empty()) {
                Leveled mapped = apply_state_map_to_gss(pruned, state_map);
                if (!mapped.is_empty()) {
                    temp_states[tokenizer_sid] = std::move(mapped);
                }
            }
        }

        // 3) Main BFS over token bytes
        struct Item {
            int offset;
            int tokenizer_sid;
            Leveled gss;
        };

        std::deque<Item> q;
        for (auto &kv : temp_states) {
            q.push_back(Item{0, kv.first, std::move(kv.second)});
        }

        struct QItemKey {
            int offset;
            int tokenizer_sid;
            std::uintptr_t gss_ptr;

            bool operator==(const QItemKey& other) const {
                return offset == other.offset &&
                       tokenizer_sid == other.tokenizer_sid &&
                       gss_ptr == other.gss_ptr;
            }
        };

        struct QItemKeyHash {
            std::size_t operator()(const QItemKey& k) const {
                std::size_t h1 = std::hash<int>()(k.offset);
                std::size_t h2 = std::hash<int>()(k.tokenizer_sid);
                std::size_t h3 = std::hash<std::uintptr_t>()(k.gss_ptr);
                return h1 ^ (h2 << 1) ^ (h3 << 2);
            }
        };
        std::unordered_set<QItemKey, QItemKeyHash> visited_q_items;

        // new states being built
        std::unordered_map<int, std::vector<Leveled>> new_states_vec;

        while (!q.empty()) {
            Item cur = std::move(q.front());
            q.pop_front();

            QItemKey key{cur.offset, cur.tokenizer_sid, reinterpret_cast<std::uintptr_t>(cur.gss.inner.get())};
            if (visited_q_items.count(key)) continue;
            visited_q_items.insert(key);

            std::string suffix = token.substr(static_cast<size_t>(cur.offset));
            py::tuple result = tokenizer_.attr("execute_from_state")(py::bytes(suffix), cur.tokenizer_sid);
            py::object end_state_obj = result[0];
            py::object matches_obj = result[1];

            bool has_end_state = !end_state_obj.is_none();
            int end_state = has_end_state ? py::cast<int>(end_state_obj) : -1;

            // collect matches
            std::vector<std::pair<int,int>> matches; // (terminal_id, width)
            for (auto tm : matches_obj) {
                py::tuple tmt = py::cast<py::tuple>(tm);
                int terminal_id = py::cast<int>(tmt[0]);
                int width = py::cast<int>(tmt[1]);
                matches.emplace_back(terminal_id, width);
            }

            for (auto &mt : matches) {
                int terminal_id = mt.first;
                int width = mt.second;
                
                Leveled processed = cur.gss;
                if (!(ignore_terminal_id_.has_value() && terminal_id == *ignore_terminal_id_)) {
                    processed = process_token(processed, terminal_id);
                }

                if (has_end_state) {
                    // Immediate re-match disallow
                    py::object accessible = tokenizer_.attr("tokens_accessible_from_state")(end_state);
                    bool immediate = false;
                    for (auto v : accessible) {
                        if (py::cast<int>(v) == terminal_id) {
                            immediate = true; break;
                        }
                    }
                    if (immediate) {
                        processed = disallow_in_state(processed, end_state, terminal_id);
                    }
                }

                if (!processed.is_empty()) {
                    int new_offset = cur.offset + width;
                    int next_tokenizer_sid = tokenizer_initial_state_;
                    if (static_cast<size_t>(new_offset) == token.size()) {
                        new_states_vec[next_tokenizer_sid].push_back(std::move(processed));
                    } else {
                        q.push_back(Item{new_offset, next_tokenizer_sid, std::move(processed)});
                    }
                }
            }

            if (has_end_state) {
                new_states_vec[end_state].push_back(cur.gss);
            }
        }

        // 4) Merge and filter empties
        std::unordered_map<int, Leveled> merged_states;
        for (auto &kv : new_states_vec) {
            int sid = kv.first;
            const std::vector<Leveled> &lst = kv.second;
            if (lst.empty()) continue;
            Leveled merged = Leveled::merge_many(lst);
            if (!merged.is_empty()) {
                merged_states[sid] = std::move(merged);
            }
        }

        // Update internal state
        state_ = std::move(merged_states);
    }

    py::object get_mask() {
        // values: node_id -> GSS
        std::unordered_map<int, Leveled> values;
        // todo buckets: depth -> set of nodes
        std::unordered_map<int, std::set<int>> todo;
        // min-heap for depths
        std::priority_queue<int, std::vector<int>, std::greater<int>> depth_heap;

        auto enqueue = [&](int d, int n) {
            auto &bucket = todo[d];
            bool first = bucket.empty();
            bucket.insert(n);
            if (first) depth_heap.push(d);
        };

        // This memoization cache is critical for performance. It's shared across all GSSs
        // being initialized, mirroring the Python implementation's optimization.
        std::unordered_map<std::uintptr_t, std::shared_ptr<Acc>> init_acc_memo;

        // Seed with initialized accs (compute allowed llm mask from terminals_union)
        for (auto &kv : state_) {
            int sid = kv.first;
            int root = 0;
            auto it = roots_map_.find(sid);
            if (it == roots_map_.end()) continue;
            root = it->second;
            const Leveled &gss = kv.second;
            Leveled gss_initialized = initialize_gss_accs(gss, &init_acc_memo);

            auto itv = values.find(root);
            if (itv != values.end()) {
                values[root] = values[root].merge(gss_initialized);
            } else {
                values[root] = std::move(gss_initialized);
            }
            int d = arena_.at(root).max_depth;
            enqueue(d, root);
        }

        RangeSet final_mask = RangeSet::empty();

        while (!depth_heap.empty()) {
            int depth = depth_heap.top(); depth_heap.pop();
            auto &bucket = todo[depth];

            while (!bucket.empty()) {
                int node = *bucket.begin();
                bucket.erase(bucket.begin());
                auto itv = values.find(node);
                if (itv == values.end()) continue;
                Leveled gss_node = std::move(itv->second);
                values.erase(itv);

                // End-node handling
                const NodeInfo &info = arena_.at(node);
                if (info.is_end) {
                    auto reduced = gss_node.reduce_acc();
                    if (reduced) {
                        final_mask = final_mask.union_with(reduced->llm_mask);
                    }
                }

                // Traverse edges
                for (const Edge &e : info.edges) {
                    Leveled popped = gss_node.popn(e.pop);
                    if (popped.is_empty()) continue;

                    // Apply LLM limiter once per edge (not per destination) to avoid repeated
                    // expensive apply_and_prune work. This mirrors the sharing-friendly structure
                    // from the Python implementation.
                    Leveled popped_limited = intersect_llm_mask(popped, e.llm_bv_rangeset);
                    if (popped_limited.is_empty()) continue;

                    // Compute top states after pruning/limiting.
                    std::unordered_set<int> top_after = popped_limited.peek();

                    for (const DestEdge &de : e.dests) {
                        // Determine which top states to keep for this destination
                        std::unordered_set<int> keep;
                        for (int top_sid : top_after) {
                            if (contains_in_ranges(de.state_ranges, top_sid)) {
                                keep.insert(top_sid);
                            }
                        }
                        if (keep.empty()) continue;

                        Leveled child2 = popped_limited.isolate_many(keep);
                        if (child2.is_empty()) continue;

                        int dnode = de.dest_idx;
                        auto it_child = values.find(dnode);
                        if (it_child != values.end()) {
                            values[dnode] = values[dnode].merge(child2);
                        } else {
                            values[dnode] = std::move(child2);
                        }
                        enqueue(arena_.at(dnode).max_depth, dnode);
                    }
                }
            }
            // Clean up bucket
            todo.erase(depth);
        }

        // Convert internal mask indices to original
        std::vector<unsigned long long> original_indices;
        // Optimization: iterate over the map instead of expanding the rangeset,
        // as the map size is bounded by vocab size.
        original_indices.reserve(internal_to_original_map_cpp_.size());
        for (const auto& [internal_id, original_id] : internal_to_original_map_cpp_) {
            if (final_mask.contains(internal_id)) {
                original_indices.push_back(original_id);
            }
        }
        return py::cast(RangeSet::from_indices(original_indices));
    }

private:
    // -----------------------------
    // Internal data structures
    // -----------------------------
    struct ActionsOnTerminal {
        bool has_shift{false};
        int shift_state_id{0};
        std::vector<std::pair<int,int>> reduces; // (nonterminal_id, len)
    };

    struct Row {
        std::unordered_map<int, ActionsOnTerminal> actions; // terminal_id -> Actions
        std::unordered_map<int, int> gotos;                 // nonterminal_id -> state_id
    };

    struct DestEdge {
        int dest_idx;
        std::vector<std::pair<int,int>> state_ranges; // inclusive ranges
    };

    struct Edge {
        int pop;
        RangeSet llm_bv_rangeset; // icl_rangeset.RangeSet (for intersection)
        std::vector<DestEdge> dests;
    };

    struct NodeInfo {
        std::vector<Edge> edges;
        bool is_end{false};
        int max_depth{0};
    };

    using Leveled = LeveledGSS<int, Acc>;

    // -----------------------------
    // Members
    // -----------------------------
    // Tokenizer
    py::object tokenizer_;
    int tokenizer_initial_state_{0};
    int tokenizer_max_state_{0};
    std::optional<int> ignore_terminal_id_;

    // Parser
    int start_state_id_{0};
    std::unordered_map<int, Row> parser_; // state_id -> Row

    // Arena/trie
    std::unordered_map<int, NodeInfo> arena_;
    std::unordered_map<int, int> roots_map_;

    // Possible matches and universe bitset
    py::dict pmc_;
    py::object all_internal_llm_tokens_bitset_;
    // Preconverted possible matches: tsid -> term_id -> RangeSet
    std::unordered_map<int, std::unordered_map<int, RangeSet>> pmc_cpp_;

    // Universe RangeSet
    RangeSet universe_rangeset_;

    // Preconverted internal->original map
    std::unordered_map<unsigned long long, unsigned long long> internal_to_original_map_cpp_;

    // Current GLR state: tokenizer_state_id -> LeveledGSS
    std::unordered_map<int, Leveled> state_;

    // Check membership in a vector of inclusive ranges (sorted, non-overlapping).
    static bool contains_in_ranges(const std::vector<std::pair<int,int>>& ranges, int v) {
        // Binary search by upper_bound on end values
        int lo = 0;
        int hi = static_cast<int>(ranges.size()) - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            const auto& pr = ranges[mid];
            if (v < pr.first) hi = mid - 1;
            else if (v > pr.second) lo = mid + 1;
            else return true;
        }
        return false;
    }
    // -----------------------------
    // Parsing helpers
    // -----------------------------
    static int json_to_int(const json& j) {
        if (j.is_string()) {
            return std::stoi(j.get<std::string>());
        }
        return j.get<int>();
    }

    void parse_parser_table_from_json(const json& parser_data) {
        start_state_id_ = parser_data.at("start_state_id").get<int>();
        const json& table = parser_data.at("stage_7_table");

        for (const auto& row_item : table) {
            const json& row_data = row_item[1];
            int state_id = json_to_int(row_item[0]);

            Row row;

            // actions
            const json& actions_list = row_data.at("shifts_and_reduces_full");
            for (const auto& aitem : actions_list) {
                int term_id = json_to_int(aitem[0]);
                const json& action = aitem[1];
                std::string variant = action.at("variant").get<std::string>();
                ActionsOnTerminal aot;

                if (variant == "Shift") {
                    aot.has_shift = true;
                    aot.shift_state_id = py::cast<int>(action["state_id"]);
                } else if (variant == "Reduce") {
                    int nt = py::cast<int>(action["nonterminal_id"]);
                    int len = action.at("len").get<int>();
                    aot.reduces.emplace_back(nt, len);
                } else if (variant == "Split") {
                    if (action.contains("shift") && !action.at("shift").is_null()) {
                        aot.has_shift = true;
                        aot.shift_state_id = action.at("shift").get<int>();
                    }
                    const json& reduces = action.at("reduces");
                    for (const auto& len_item : reduces) {
                        int len = json_to_int(len_item[0]);
                        const json& nts = len_item[1];
                        for (const auto& nt_item : nts) {
                            int nt = json_to_int(nt_item[0]);
                            aot.reduces.emplace_back(nt, len);
                        }
                    }
                }
                row.actions[term_id] = std::move(aot);
            }

            // gotos
            const json& gotos_list = row_data.at("gotos");
            for (const auto& gitem : gotos_list) {
                int nt = json_to_int(gitem[0]);
                if (gitem[1].contains("state_id") && !gitem[1].at("state_id").is_null()) {
                    row.gotos[nt] = gitem[1].at("state_id").get<int>();
                }
            }

            parser_[state_id] = std::move(row);
        }
    }

    void parse_arena(const json& arena_json) {
        py::object sep1 = py::module::import("_sep1");
        py::object BitsetClass = sep1.attr("Bitset");

        const auto& arena_values = arena_json.at("values");
        for (const auto& item : arena_values.items()) {
            int uid = std::stoi(item.key());
            const json& node = item.value();
            NodeInfo info;

            // Determine if end node
            bool is_end = false;
            if (node.contains("value")) {
                const auto& value_obj = node.at("value");
                if (!value_obj.is_null()) {
                    if (value_obj.contains("clean_end")) {
                        is_end = value_obj.at("clean_end").get<bool>();
                    }
                }
            }
            info.is_end = is_end;

            // Max depth
            int mdepth = 0;
            if (node.contains("max_depth")) {
                mdepth = node.at("max_depth").get<int>();
            }
            info.max_depth = mdepth;

            // Children: list of ((pop, llm_bv_json), [(dest_idx, state_bv_json), ...])
            const json* children_obj_ptr = nullptr;
            if (node.contains("children_bits")) {
                children_obj_ptr = &node.at("children_bits");
            } else if (node.contains("children")) {
                children_obj_ptr = &node.at("children");
            }
            if (!children_obj_ptr) continue;

            for (const auto& entry : *children_obj_ptr) {
                const auto& edge_key = entry[0];
                int pop = edge_key[0].get<int>();
                const auto& llm_bv_json = edge_key[1];

                // Convert llm_bv_json -> _sep1.Bitset -> RangeSet
                py::object llm_bv_bitset = BitsetClass.attr("from_json_string")(llm_bv_json.dump());
                py::list llm_ranges_py = llm_bv_bitset.attr("to_ranges")();
                std::vector<std::pair<unsigned long long, unsigned long long>> llm_ranges_cpp;
                llm_ranges_cpp.reserve(py::len(llm_ranges_py));
                for (auto r : llm_ranges_py) {
                    py::tuple t = py::cast<py::tuple>(r);
                    llm_ranges_cpp.emplace_back(py::cast<unsigned long long>(t[0]), py::cast<unsigned long long>(t[1]));
                }
                RangeSet llm_bv_rs = RangeSet::from_ranges(llm_ranges_cpp);

                std::vector<DestEdge> dests;
                const auto& dest_map = entry[1];
                for (const auto& d : dest_map) {
                    int dest_idx = d[0].get<int>();
                    const auto& state_bv_json = d[1];

                    py::object state_bv = BitsetClass.attr("from_json_string")(state_bv_json.dump());
                    bool is_empty_bv = state_bv.attr("is_empty")().cast<bool>();
                    std::vector<std::pair<int,int>> ranges;
                    if (!is_empty_bv) {
                        py::object py_ranges = state_bv.attr("to_ranges")();
                        ranges.reserve(py::len(py_ranges));
                        for (auto it : py_ranges) {
                            py::tuple t = py::cast<py::tuple>(it);
                            int l = py::cast<int>(t[0]);
                            int r = py::cast<int>(t[1]);
                            ranges.emplace_back(l, r);
                        }
                    }
                    dests.push_back(DestEdge{dest_idx, std::move(ranges)});
                }

                Edge e{pop, llm_bv_rs, std::move(dests)};
                info.edges.push_back(std::move(e));
            }

            arena_[uid] = std::move(info);
        }
    }

    // -----------------------------
    // Leveled GSS transforms needed by algorithm
    // -----------------------------
    Leveled prune_by_terminals_map(const Leveled& g, const std::unordered_map<int, RangeSet>& terminals_map) {
        auto pred = [&](const std::shared_ptr<Acc>& acc) -> bool {
            RangeSet rs_empty = RangeSet::empty();
            for (auto &kv : terminals_map) {
                int sid = kv.first;
                const RangeSet &matched = kv.second;
                auto it = acc->terminals_union.find(sid);
                const RangeSet &disallowed = (it != acc->terminals_union.end()) ? it->second : rs_empty;
                RangeSet inter = matched.intersection_with(disallowed);
                bool ok = inter.is_empty();
                if (!ok) return false;
            }
            return true;
        };
        return g.prune(pred);
    }

    Leveled apply_state_map_to_gss(const Leveled& g, const std::unordered_map<int, int>& state_map) {
        auto mapper = [&](const std::shared_ptr<Acc>& acc) -> std::shared_ptr<Acc> {
            auto na = std::make_shared<Acc>();
            na->llm_mask = acc->llm_mask;
            RangeSet rs_empty = RangeSet::empty();
            // This logic must mirror the Python version to ensure terminals_union is pruned
            // and does not grow indefinitely. We iterate over the small state_map, not the
            // potentially large terminals_union.
            for (auto const& [old_sid, new_sid] : state_map) {
                auto it = acc->terminals_union.find(old_sid);
                if (it != acc->terminals_union.end()) {
                    const RangeSet& bv_source = it->second;
                    auto it_new = na->terminals_union.find(new_sid);
                    if (it_new != na->terminals_union.end()) {
                        it_new->second = it_new->second.union_with(bv_source);
                    } else {
                        na->terminals_union.emplace(new_sid, bv_source);
                    }
                }
            }
            return na;
        };
        return g.apply(mapper);
    }

    Leveled disallow_in_state(const Leveled& g, int state_id, int terminal_id) {
        auto transformer = [&](const std::shared_ptr<Acc>& acc) -> std::shared_ptr<Acc> {
            auto na = std::make_shared<Acc>();
            na->llm_mask = acc->llm_mask;
            na->terminals_union = acc->terminals_union;
            std::vector<unsigned long long> idx{static_cast<unsigned long long>(terminal_id)};
            RangeSet to_add = RangeSet::from_indices(idx);
            RangeSet rs_empty = RangeSet::empty();
            const RangeSet& curr = na->terminals_union.count(state_id) ? na->terminals_union[state_id] : rs_empty;
            na->terminals_union[state_id] = curr.union_with(to_add);
            return na;
        };
        return g.apply(transformer);
    }

    Leveled process_token(const Leveled& g, int terminal_id) {
        // heads_by_state: state_id -> vector<Leveled>
        std::unordered_map<int, std::vector<Leveled>> heads_by_state;
        std::unordered_set<int> tops = g.peek();
        for (int state_id : tops) {
            Leveled isol = g.isolate(state_id);
            heads_by_state[state_id].push_back(std::move(isol));
        }

        std::vector<Leveled> shifted_gsses;

        auto merge_many_gss = [&](const std::vector<Leveled>& lst) -> Leveled {
            return Leveled::merge_many(lst);
        };

        while (!heads_by_state.empty()) {
            // pop one element
            auto it = heads_by_state.begin();
            int state_id = it->first;
            std::vector<Leveled> gsss = std::move(it->second);
            heads_by_state.erase(it);

            // merge_many
            Leveled state_gss = merge_many_gss(gsss);

            // lookup row
            auto row_it = parser_.find(state_id);
            if (row_it == parser_.end()) continue;
            Row &row = row_it->second;

            auto act_it = row.actions.find(terminal_id);
            if (act_it == row.actions.end()) continue;
            ActionsOnTerminal &action = act_it->second;

            auto handle_shift = [&](int shift_to_state_id, const Leveled& gss_to_shift) {
                Leveled shifted = gss_to_shift.push(shift_to_state_id);
                shifted_gsses.push_back(std::move(shifted));
            };

            auto handle_reduce = [&](int nt_id, int len, const Leveled& gss_to_reduce) {
                Leveled popped_gss = gss_to_reduce.popn(len);
                std::unordered_set<int> from_states = popped_gss.peek();
                for (int from_sid : from_states) {
                    auto goto_it = parser_.find(from_sid);
                    if (goto_it == parser_.end()) continue;
                    Row &from_row = goto_it->second;
                    auto gt = from_row.gotos.find(nt_id);
                    if (gt == from_row.gotos.end()) continue;
                    int goto_state_id = gt->second;
                    Leveled goto_gss = popped_gss.isolate(from_sid).push(goto_state_id);
                    heads_by_state[goto_state_id].push_back(std::move(goto_gss));
                }
            };

            if (action.has_shift) {
                handle_shift(action.shift_state_id, state_gss);
            }
            for (auto &rd : action.reduces) {
                handle_reduce(rd.first, rd.second, state_gss);
            }
        }

        Leveled merged = Leveled::merge_many(shifted_gsses);
        return merged;
    }

    Leveled initialize_gss_accs(
        const Leveled& g,
        std::unordered_map<std::uintptr_t, std::shared_ptr<Acc>>* acc_memo
    ) {
        auto mutator = [&](const std::shared_ptr<Acc>& a) -> std::shared_ptr<Acc> {
            // Build disallowed mask as RangeSet
            RangeSet disallowed_llm_mask = RangeSet::empty();

            for (auto &kv : a->terminals_union) {
                int tsid = kv.first;
                if (tsid > tokenizer_max_state_) continue;
                auto it_ts = pmc_cpp_.find(tsid);
                if (it_ts == pmc_cpp_.end()) continue;

                const auto &term_to_llm = it_ts->second;
                // Optimization: iterate over the smaller set (terminals for this tokenizer state)
                // and do fast lookups into the RangeSet of disallowed terminals, instead of
                // expanding the RangeSet to a list of indices.
                for (const auto& [term_id, llm_rangeset] : term_to_llm) {
                    if (kv.second.contains(term_id)) {
                        disallowed_llm_mask = disallowed_llm_mask.union_with(llm_rangeset);
                    }
                }
            }

            RangeSet allowed_mask = universe_rangeset_.difference_with(disallowed_llm_mask);

            auto na = std::make_shared<Acc>();
            na->llm_mask = allowed_mask;
            // terminals_union -> empty
            return na;
        };
        return g.apply(mutator, acc_memo);
    }

    Leveled intersect_llm_mask(
        const Leveled& g,
        const RangeSet& limiter,
        std::unordered_map<std::uintptr_t, std::shared_ptr<Acc>>* acc_memo = nullptr) {
        auto mutator = [&](const std::shared_ptr<Acc>& a) -> std::shared_ptr<Acc> {
            RangeSet new_mask = a->llm_mask.intersection_with(limiter);
            bool is_empty_mask = new_mask.is_empty();
            if (is_empty_mask) return std::shared_ptr<Acc>(nullptr);
            auto na = std::make_shared<Acc>();
            na->llm_mask = new_mask;
            return na;
        };
        return g.apply_and_prune(mutator, acc_memo);
    }

};

PYBIND11_MODULE(precompute3_engine, m) {
    m.doc() = "C++ engine for precompute3_model_cpp: commit() and get_mask(), Leveled GSS implementation";

    py::class_<Engine>(m, "Engine")
        .def(py::init<py::object,int,int,py::object,const std::string&,py::dict,py::object,py::dict>(),
             py::arg("tokenizer"),
             py::arg("tokenizer_initial_state"),
             py::arg("tokenizer_max_state"),
             py::arg("ignore_terminal_id_or_none"),
             py::arg("json_string"),
             py::arg("possible_matches"),
             py::arg("all_internal_llm_tokens_bitset"),
             py::arg("internal_to_original_map"))
        .def("commit", &Engine::commit, py::arg("token_bytes"))
        .def("get_mask", &Engine::get_mask);
}
