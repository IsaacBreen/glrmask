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
#include <pybind11/gil.h>

#include <functional>
#include <memory>
#include "leveled_gss.hpp"
#include "icl_rangeset.hpp"

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

// Custom hash for std::pair<int, int> for use in unordered_map.
struct pair_hash {
    std::size_t operator () (const std::pair<int, int> &p) const {
        auto h1 = std::hash<int>{}(p.first);
        auto h2 = std::hash<int>{}(p.second);
        // A common way to combine hashes, inspired by Boost's hash_combine.
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};


class Engine {
public:
    Engine(py::object tokenizer,
           int tokenizer_initial_state,
           int tokenizer_max_state,
           py::object ignore_terminal_id_or_none,
           py::dict parser_data,                  // JSON 'parser' object with start_state_id and stage_7_table
           py::dict roots_map_py,                 // state_id -> root node id
           py::dict arena_py,                     // uid -> node dict (children carry bitset JSON)
           py::dict possible_matches,             // tsid -> term_id -> _sep1.Bitset
           py::object all_internal_llm_tokens_bitset,
           py::dict internal_to_original_map)
        : tokenizer_(std::move(tokenizer)), // Keep a ref for error messages etc.
          tokenizer_initial_state_(tokenizer_initial_state),
          tokenizer_max_state_(tokenizer_max_state),
          parser_data_(std::move(parser_data)),
          roots_map_py_(std::move(roots_map_py)),
          pmc_(std::move(possible_matches)),
          all_internal_llm_tokens_bitset_(std::move(all_internal_llm_tokens_bitset)),
          // Pre-bind Python methods to avoid repeated attr lookups
          tokenizer_execute_from_state_(tokenizer_.attr("execute_from_state")),
          tokenizer_tokens_accessible_from_state_(tokenizer_.attr("tokens_accessible_from_state"))
          {

        if (!ignore_terminal_id_or_none.is_none()) {
            ignore_terminal_id_ = py::cast<int>(ignore_terminal_id_or_none);
        }

        // Universe RangeSet for get_mask init
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

        // Parse roots_map into C++ map
        for (auto item : roots_map_py_) {
            int sid = py::cast<int>(item.first);
            int root = py::cast<int>(item.second);
            roots_map_[sid] = root;
        }

        // Parse parser table from JSON dict
        parse_parser_table_from_json(parser_data_);

        // Parse arena using JSON-encoded bitsets; convert to Bitset and RangeSet
        parse_arena(arena_py);

        // Pre-convert possible_matches into C++ RangeSet map (tsid -> term_id -> RangeSet)
        // This avoids repeated Python calls during get_mask initialization.
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
        // Caches for this commit() call.
        using ExecResult = py::tuple;
        // Key: (offset, tokenizer_sid)
        std::unordered_map<std::pair<int, int>, ExecResult, pair_hash> exec_cache;
        accessible_cache_.clear();

        // Build terminals_map and state_map
        py::buffer_info token_buf = py::buffer(token_bytes).request();
        const char* token_data = static_cast<const char*>(token_buf.ptr);
        ssize_t token_len = token_buf.size;
        std::unordered_map<int, RangeSet> terminals_map; // sid -> RangeSet of matched terminals
        std::unordered_map<int, int> state_map;            // old_sid -> end_sid

        for (const auto &kv : state_) {
            int tokenizer_sid = kv.first;
            py::tuple result = tokenizer_execute_from_state_(token_bytes, tokenizer_sid);
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
        temp_states.reserve(state_.size());
        {
            py::gil_scoped_release release; // Release GIL for C++ heavy part
            for (const auto &kv : state_) {
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
        } // GIL re-acquired here

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

            ExecResult cached_res;
            auto cache_key = std::make_pair(cur.offset, cur.tokenizer_sid);
            auto it = exec_cache.find(cache_key);
            if (it != exec_cache.end()) {
                cached_res = it->second;
            } else {
                ssize_t suffix_len = token_len - cur.offset;
                py::bytes suffix_bytes(token_data + cur.offset, suffix_len);
                cached_res = tokenizer_execute_from_state_(suffix_bytes, cur.tokenizer_sid);
                exec_cache[cache_key] = cached_res;
            }
            py::tuple result = cached_res;

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
                    const auto& accessible_set = get_accessible_tokens(end_state);
                    if (accessible_set.count(terminal_id)) {
                        processed = disallow_in_state(processed, end_state, terminal_id);
                    }
                }

                if (!processed.is_empty()) {
                    int new_offset = cur.offset + width;
                    int next_tokenizer_sid = tokenizer_initial_state_;
                    if (new_offset == token_len) {
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
        py::gil_scoped_release release; // Release GIL for the whole C++ part

        // values: node_id -> GSS
        std::unordered_map<int, Leveled> values;
        std::priority_queue<std::pair<int, int>> depth_heap; // {depth, node_id}
        std::unordered_set<int> enqueued_nodes;

        auto enqueue = [&](int d, int n) {
            if (enqueued_nodes.count(n)) {
                return;
            }
            enqueued_nodes.insert(n);
            depth_heap.push({d, n});
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
            auto [depth, node] = depth_heap.top();
            depth_heap.pop();

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
                RangeSet llm_bv = e.llm_bv_rangeset.difference_with(final_mask);
                if (llm_bv.is_empty()) continue;

                Leveled popped = gss_node.popn(e.pop);
                if (popped.is_empty()) continue;

                // Apply LLM limiter once per edge (not per destination) to avoid repeated
                // expensive apply_and_prune work. This mirrors the sharing-friendly structure
                // from the Python implementation.
                std::unordered_map<std::uintptr_t, std::shared_ptr<Acc>> acc_memo;
                Leveled popped_limited = intersect_llm_mask(popped, llm_bv, &acc_memo);
                if (popped_limited.is_empty()) continue;

                auto reduced_popped = popped_limited.reduce_acc();
                if (!reduced_popped || reduced_popped->llm_mask.is_empty()) {
                    continue;
                }

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

                    auto reduced_child = child2.reduce_acc();
                    if (!reduced_child || reduced_child->llm_mask.is_empty()) {
                        continue;
                    }

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

        // Convert internal mask indices to original
        std::vector<unsigned long long> original_indices;
        // Iterate over the final mask (much smaller than vocab) and look up original IDs.
        std::vector<unsigned long long> final_indices = final_mask.to_indices();
        original_indices.reserve(final_indices.size());
        for (unsigned long long internal_id : final_indices) {
            auto it = internal_to_original_map_cpp_.find(internal_id);
            if (it != internal_to_original_map_cpp_.end()) {
                original_indices.push_back(it->second);
            }
        }

        py::gil_scoped_acquire acquire; // Re-acquire GIL before returning to Python
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
    // We keep the main object for potential future needs (e.g., error reporting)
    // but pre-bind hot-path methods for performance.
    py::object tokenizer_;
    int tokenizer_initial_state_{0};
    int tokenizer_max_state_{0};
    std::optional<int> ignore_terminal_id_;

    // Parser
    py::dict parser_data_;
    int start_state_id_{0};
    std::unordered_map<int, Row> parser_; // state_id -> Row

    // Arena/trie
    py::dict roots_map_py_;
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

    // Pre-bound tokenizer methods
    py::object tokenizer_execute_from_state_;
    py::object tokenizer_tokens_accessible_from_state_;
    // Cache for tokens_accessible_from_state (cleared per commit)
    std::unordered_map<int, std::unordered_set<int>> accessible_cache_;

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

    // Helper for accessible tokens cache.
    // NOTE: This function must be called with the GIL released, as it will
    // temporarily re-acquire it to call into Python.
    const std::unordered_set<int>& get_accessible_tokens(int state_id) {
        auto it = accessible_cache_.find(state_id);
        if (it != accessible_cache_.end()) {
            return it->second;
        }

        py::gil_scoped_acquire acquire; // Need GIL for Python call
        py::object accessible_py = tokenizer_tokens_accessible_from_state_(state_id);
        std::unordered_set<int> accessible_set;
        accessible_set.reserve(py::len(accessible_py));
        for (auto v : accessible_py) {
            accessible_set.insert(py::cast<int>(v));
        }
        py::gil_scoped_release release; // Release again

        auto [inserted_it, success] = accessible_cache_.emplace(state_id, std::move(accessible_set));
        return inserted_it->second;
    }

    // -----------------------------
    // Parsing helpers
    // -----------------------------
    static int py_obj_to_int(py::handle obj) {
        if (py::isinstance<py::int_>(obj)) {
            return obj.cast<int>();
        }
        if (py::isinstance<py::str>(obj)) {
            return std::stoi(obj.cast<std::string>());
        }
        throw py::type_error("Expected int or string representation of int");
    }

    static bool is_py_dict(py::handle h) {
        return py::isinstance<py::dict>(h);
    }

    void parse_parser_table_from_json(const py::dict& parser_data) {
        // parser_data: { 'start_state_id': int, 'stage_7_table': list[[state_id_str, row_data_dict], ...] }
        start_state_id_ = py::cast<int>(parser_data["start_state_id"]);
        py::list table = parser_data["stage_7_table"];

        for (auto row_item : table) {
            py::tuple row_tuple = py::cast<py::tuple>(row_item);
            int state_id = py_obj_to_int(row_tuple[0]);
            py::dict row_data = py::cast<py::dict>(row_tuple[1]);

            Row row;

            // actions
            py::list actions_list = row_data["shifts_and_reduces_full"];
            for (auto aitem : actions_list) {
                py::tuple a = py::cast<py::tuple>(aitem);
                int term_id = py_obj_to_int(a[0]);
                py::dict action = py::cast<py::dict>(a[1]);

                std::string variant = py::cast<std::string>(action["variant"]);
                ActionsOnTerminal aot;

                if (variant == "Shift") {
                    aot.has_shift = true;
                    aot.shift_state_id = py::cast<int>(action["state_id"]);
                } else if (variant == "Reduce") {
                    int nt = py::cast<int>(action["nonterminal_id"]);
                    int len = py::cast<int>(action["len"]);
                    aot.reduces.emplace_back(nt, len);
                } else if (variant == "Split") {
                    py::object shift_obj = action["shift"];
                    if (!shift_obj.is_none()) {
                        aot.has_shift = true;
                        aot.shift_state_id = py::cast<int>(shift_obj);
                    }
                    py::list reduces = action["reduces"];
                    for (auto len_item : reduces) {
                        py::tuple len_tuple = py::cast<py::tuple>(len_item);
                        int len = py_obj_to_int(len_tuple[0]);
                        py::list nts = py::cast<py::list>(len_tuple[1]);
                        for (auto nt_item : nts) {
                            py::tuple nt_tuple = py::cast<py::tuple>(nt_item);
                            int nt = py_obj_to_int(nt_tuple[0]);
                            aot.reduces.emplace_back(nt, len);
                        }
                    }
                }
                row.actions[term_id] = std::move(aot);
            }

            // gotos
            py::list gotos_list = row_data["gotos"];
            for (auto gitem : gotos_list) {
                py::tuple g = py::cast<py::tuple>(gitem);
                int nt = py_obj_to_int(g[0]);
                py::dict goto_data = py::cast<py::dict>(g[1]);
                if (goto_data.contains("state_id") && !goto_data["state_id"].is_none()) {
                    row.gotos[nt] = py::cast<int>(goto_data["state_id"]);
                }
            }

            parser_[state_id] = std::move(row);
        }
    }

    void parse_arena(py::dict arena_py) {
        py::module json = py::module::import("json");
        py::object dumps = json.attr("dumps");
        py::object sep1 = py::module::import("_sep1");
        py::object BitsetClass = sep1.attr("Bitset");

        for (auto item_handle : arena_py.attr("items")()) {
            py::tuple item = py::cast<py::tuple>(item_handle);
            int uid = py::cast<int>(item[0]);
            py::dict node = py::cast<py::dict>(item[1]);
            NodeInfo info;

            // Determine if end node
            bool is_end = false;
            if (node.contains("value")) {
                py::object value_obj = node["value"];
                if (!value_obj.is_none()) {
                    py::dict value = py::cast<py::dict>(value_obj);
                    if (value.contains("clean_end")) {
                        is_end = py::cast<bool>(value["clean_end"]);
                    }
                }
            }
            info.is_end = is_end;

            // Max depth
            int mdepth = 0;
            if (node.contains("max_depth")) {
                mdepth = py::cast<int>(node["max_depth"]);
            }
            info.max_depth = mdepth;

            // Children: list of ((pop, llm_bv_json), [(dest_idx, state_bv_json), ...])
            py::object children_obj;
            if (node.contains("children_bits")) {
                children_obj = node["children_bits"];
            } else if (node.contains("children")) {
                children_obj = node["children"];
            } else {
                children_obj = py::list();
            }

            for (auto ch : children_obj) {
                py::tuple entry = py::cast<py::tuple>(ch);
                py::tuple edge_key = py::cast<py::tuple>(entry[0]);
                int pop = py::cast<int>(edge_key[0]);
                py::object llm_bv_json = edge_key[1];

                // Convert llm_bv_json -> _sep1.Bitset -> RangeSet
                py::object llm_bv_bitset = BitsetClass.attr("from_json_string")(dumps(llm_bv_json));
                py::list llm_ranges_py = llm_bv_bitset.attr("to_ranges")();
                std::vector<std::pair<unsigned long long, unsigned long long>> llm_ranges_cpp;
                llm_ranges_cpp.reserve(py::len(llm_ranges_py));
                for (auto r : llm_ranges_py) {
                    py::tuple t = py::cast<py::tuple>(r);
                    llm_ranges_cpp.emplace_back(py::cast<unsigned long long>(t[0]), py::cast<unsigned long long>(t[1]));
                }
                RangeSet llm_bv_rs = RangeSet::from_ranges(llm_ranges_cpp);

                std::vector<DestEdge> dests;
                py::object dest_map = entry[1];
                for (auto d : dest_map) {
                    py::tuple dd = py::cast<py::tuple>(d);
                    int dest_idx = py::cast<int>(dd[0]);
                    py::object state_bv_json = dd[1];

                    py::object state_bv = BitsetClass.attr("from_json_string")(dumps(state_bv_json));
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
            RangeSet to_add = RangeSet::from_singleton(static_cast<unsigned long long>(terminal_id));
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
                // Correctly iterate over the (small) set of disallowed terminals and look up
                // the corresponding LLM tokens, matching the Python implementation's logic.
                std::vector<unsigned long long> disallowed_indices = kv.second.to_indices();
                for (unsigned long long term_id_ull : disallowed_indices) {
                    int term_id = static_cast<int>(term_id_ull);
                    auto it_term = term_to_llm.find(term_id);
                    if (it_term != term_to_llm.end()) {
                        disallowed_llm_mask = disallowed_llm_mask.union_with(it_term->second);
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
        .def(py::init<py::object,int,int,py::object,py::dict,py::dict,py::dict,py::dict,py::object,py::dict>(),
             py::arg("tokenizer"),
             py::arg("tokenizer_initial_state"),
             py::arg("tokenizer_max_state"),
             py::arg("ignore_terminal_id_or_none"),
             py::arg("parser_data"),
             py::arg("roots_map_py"),
             py::arg("arena_py"),
             py::arg("possible_matches"),
             py::arg("all_internal_llm_tokens_bitset"),
             py::arg("internal_to_original_map"))
        .def("commit", &Engine::commit, py::arg("token_bytes"))
        .def("get_mask", &Engine::get_mask);
}
