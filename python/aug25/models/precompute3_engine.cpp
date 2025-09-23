#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

#include "icl_rangeset.h"

namespace py = pybind11;

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
        : tokenizer_(std::move(tokenizer)),
          tokenizer_initial_state_(tokenizer_initial_state),
          tokenizer_max_state_(tokenizer_max_state),
          parser_data_(std::move(parser_data)),
          roots_map_py_(std::move(roots_map_py)),
          internal_to_original_map_(std::move(internal_to_original_map)) {

        if (!ignore_terminal_id_or_none.is_none()) {
            ignore_terminal_id_ = py::cast<int>(ignore_terminal_id_or_none);
        }

        // Universe RangeSet for get_mask init
        universe_rangeset_ = bitset_to_rangeset(all_internal_llm_tokens_bitset);

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
        parse_pmc(possible_matches);

        // Initialize state: one GSS per tokenizer initial state with start parser state on stack
        auto initial_acc = std::make_shared<Acc>();
        initial_acc->llm_mask = RangeSet::empty();
        GSSNodePtr leaf = std::make_shared<const GSSNode>(
            std::unordered_map<int, GSSNodePtr>{}, initial_acc
        );
        state_[tokenizer_initial_state_] = gss_push(leaf, start_state_id_);
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
            RangeSet matched_rs = RangeSet::from_indices(matched_terminals);
            terminals_map[tokenizer_sid] = matched_rs;
        }

        // 2) Prune and map per-state GSS (rename terminals_union keys according to state_map)
        std::unordered_map<int, GSSNodePtr> temp_states;
        for (auto &kv : state_) {
            int tokenizer_sid = kv.first;
            const GSSNodePtr &gss = kv.second;

            GSSNodePtr pruned = prune_by_terminals_map(gss, terminals_map);
            if (!pruned->is_empty()) {
                GSSNodePtr mapped = apply_state_map_to_gss(pruned, state_map);
                if (!mapped->is_empty()) {
                    temp_states[tokenizer_sid] = std::move(mapped);
                }
            }
        }

        // 3) Main BFS over token bytes
        struct Item {
            int offset;
            int tokenizer_sid;
            GSSNodePtr gss;
        };

        std::deque<Item> q;
        for (auto &kv : temp_states) {
            q.push_back(Item{0, kv.first, std::move(kv.second)});
        }

        // new states being built
        std::unordered_map<int, std::vector<GSSNodePtr>> new_states_vec;

        while (!q.empty()) {
            Item cur = std::move(q.front());
            q.pop_front();

            std::string suffix = token.substr(static_cast<size_t>(cur.offset));
            py::tuple result = tokenizer_.attr("execute_from_state")(py::bytes(suffix), cur.tokenizer_sid);
            py::object end_state_obj = result[0];
            py::object matches_obj = result[1];

            int end_state = -1;
            bool has_end_state = !end_state_obj.is_none();
            if (has_end_state) end_state = py::cast<int>(end_state_obj);

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

                GSSNodePtr processed = cur.gss;
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

                if (!processed->is_empty()) {
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
        std::unordered_map<int, GSSNodePtr> merged_states;
        for (auto &kv : new_states_vec) {
            int sid = kv.first;
            const std::vector<GSSNodePtr> &lst = kv.second;
            if (lst.empty()) continue;
            GSSNodePtr merged = gss_merge_many(lst);
            if (!merged->is_empty()) {
                merged_states[sid] = std::move(merged);
            }
        }

        // Update internal state
        state_ = std::move(merged_states);
    }

    py::object get_mask() {
        // values: node_id -> GSS
        std::unordered_map<int, GSSNodePtr> values;
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

        // Seed with initialized accs (compute allowed llm mask from terminals_union)
        for (auto &kv : state_) {
            int sid = kv.first;
            int root = 0;
            auto it = roots_map_.find(sid);
            if (it == roots_map_.end()) continue;
            root = it->second;
            const GSSNodePtr &gss = kv.second;

            GSSNodePtr gss_initialized = initialize_gss_accs(gss);

            auto itv = values.find(root);
            if (itv != values.end()) {
                values[root] = gss_merge(values[root], gss_initialized);
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
                GSSNodePtr gss_node = std::move(itv->second);
                values.erase(itv);

                // End-node handling
                const NodeInfo &info = arena_.at(node);
                if (info.is_end) {
                    RangeSet reduced = union_llm_masks(gss_node);
                    final_mask = final_mask.union_with(reduced);
                }

                // Traverse edges
                for (const Edge &e : info.edges) {
                    GSSNodePtr popped = gss_popn(gss_node, e.pop);
                    if (popped->is_empty()) continue;

                    for (const DestEdge &de : e.dests) {
                        // Determine which top states to keep
                        std::vector<int> keep;
                        for (auto const& [state_id, _] : popped->children)
                            if (state_matches_bitset(de.state_bv, state_id))
                                keep.push_back(state_id);
                        if (keep.empty()) continue;

                        GSSNodePtr child = gss_isolate_many(popped, keep);
                        if (child->is_empty()) continue;

                        // intersect_and_prune with edge's llm_bv (as RangeSet)
                        GSSNodePtr child2 = intersect_llm_mask(child, e.llm_bv_rangeset);
                        if (child2->is_empty()) continue;

                        int dnode = de.dest_idx;
                        auto it_child = values.find(dnode);
                        if (it_child != values.end()) {
                            values[dnode] = gss_merge(it_child->second, child2);
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
        std::vector<unsigned long long> original_indices_ull;
        std::vector<unsigned long long> final_indices = final_mask.to_indices();
        for (unsigned long long i_ull : final_indices) {
            int i = static_cast<int>(i_ull);
            if (internal_to_original_map_.contains(py::int_(i))) {
                int mapped = py::cast<int>(internal_to_original_map_[py::int_(i)]);
                original_indices_ull.push_back(static_cast<unsigned long long>(mapped));
            }
        }
        py::object res = py::cast(RangeSet::from_indices(original_indices_ull));
        return res;
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
        py::object state_bv; // _sep1.Bitset
    };

    struct Edge {
        int pop;
        RangeSet llm_bv_rangeset;
        std::vector<DestEdge> dests;
    };

    struct NodeInfo {
        std::vector<Edge> edges;
        bool is_end{false};
        int max_depth{0};
    };

    struct Acc {
        // terminals_union: tokenizer_state_id -> RangeSet of disallowed terminals
        std::unordered_map<int, RangeSet> terminals_union;
        // current allowed LLM mask (RangeSet)
        RangeSet llm_mask;
    };

    struct GSSNode;
    using GSSNodePtr = std::shared_ptr<const GSSNode>;

    struct GSSNode {
        const std::unordered_map<int, GSSNodePtr> children;
        const std::shared_ptr<Acc> acc;

        mutable std::unordered_map<int, GSSNodePtr> popn_cache;

        GSSNode(std::unordered_map<int, GSSNodePtr> c, std::shared_ptr<Acc> a)
            : children(std::move(c)), acc(std::move(a)) {}

        bool is_empty() const {
            return children.empty() && !acc;
        }
    };

    // -----------------------------
    // Members
    // -----------------------------
    // Tokenizer
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
    std::unordered_map<int, std::unordered_map<int, RangeSet>> pmc_native_;
    RangeSet universe_rangeset_;

    py::dict internal_to_original_map_;

    // Current GLR state: tokenizer_state_id -> GSS
    std::unordered_map<int, GSSNodePtr> state_;

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

    static RangeSet bitset_to_rangeset(py::handle bitset_py) {
        if (bitset_py.is_none()) return RangeSet::empty();
        py::list ranges_py = bitset_py.attr("to_ranges")();
        if (ranges_py.empty()) return RangeSet::empty();

        std::vector<std::pair<unsigned long long, unsigned long long>> ranges;
        ranges.reserve(py::len(ranges_py));
        for (auto r_handle : ranges_py) {
            py::tuple t = py::cast<py::tuple>(r_handle);
            ranges.emplace_back(py::cast<unsigned long long>(t[0]), py::cast<unsigned long long>(t[1]));
        }
        return RangeSet::from_ranges(ranges);
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

    void parse_pmc(py::dict pmc_py) {
        for (auto item : pmc_py) {
            int tsid = py::cast<int>(item.first);
            py::dict inner_map = py::cast<py::dict>(item.second);
            std::unordered_map<int, RangeSet> native_inner_map;
            for (auto inner_item : inner_map) {
                int term_id = py::cast<int>(inner_item.first);
                py::object bitset = py::reinterpret_borrow<py::object>(inner_item.second);
                native_inner_map[term_id] = bitset_to_rangeset(bitset);
            }
            pmc_native_[tsid] = std::move(native_inner_map);
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
                RangeSet llm_bv_rs = bitset_to_rangeset(BitsetClass.attr("from_json_string")(dumps(llm_bv_json)));

                std::vector<DestEdge> dests;
                py::object dest_map = entry[1];
                for (auto d : dest_map) {
                    py::tuple dd = py::cast<py::tuple>(d);
                    int dest_idx = py::cast<int>(dd[0]);
                    py::handle state_bv_json = dd[1];
                    py::object state_bv = BitsetClass.attr("from_json_string")(dumps(state_bv_json));
                    dests.push_back(DestEdge{dest_idx, state_bv});
                }

                Edge e{pop, llm_bv_rs, std::move(dests)};
                info.edges.push_back(std::move(e));
            }

            arena_[uid] = std::move(info);
        }
        parse_pmc(py::cast<py::dict>(py::module::import("_sep1").attr("GrammarConstraint").attr("from_json_string")(parser_data_.attr("__str__")()).attr("possible_matches")()));
    }

    // -----------------------------
    // GSS primitives (immutable trie)
    // -----------------------------
    static GSSNodePtr GSS_EMPTY;

    static GSSNodePtr gss_merge(GSSNodePtr a, GSSNodePtr b) {
        if (a == b || b->is_empty()) return a;
        if (a->is_empty()) return b;

        std::unordered_map<int, GSSNodePtr> new_children;
        std::set<int> all_keys;
        for (const auto& kv : a->children) all_keys.insert(kv.first);
        for (const auto& kv : b->children) all_keys.insert(kv.first);

        for (int key : all_keys) {
            auto a_it = a->children.find(key);
            auto b_it = b->children.find(key);
            GSSNodePtr child_a = (a_it != a->children.end()) ? a_it->second : GSS_EMPTY;
            GSSNodePtr child_b = (b_it != b->children.end()) ? b_it->second : GSS_EMPTY;
            GSSNodePtr merged_child = gss_merge(child_a, child_b);
            if (!merged_child->is_empty()) {
                new_children[key] = merged_child;
            }
        }

        std::shared_ptr<Acc> new_acc = nullptr;
        if (a->acc && b->acc) {
            new_acc = std::make_shared<Acc>();
            new_acc->llm_mask = a->acc->llm_mask.union_with(b->acc->llm_mask);
            new_acc->terminals_union = a->acc->terminals_union;
            for (const auto& kv : b->acc->terminals_union) {
                auto it = new_acc->terminals_union.find(kv.first);
                if (it != new_acc->terminals_union.end()) {
                    it->second = it->second.union_with(kv.second);
                } else {
                    new_acc->terminals_union[kv.first] = kv.second;
                }
            }
        } else if (a->acc) {
            new_acc = a->acc;
        } else if (b->acc) {
            new_acc = b->acc;
        }

        if (new_children.empty() && !new_acc) return GSS_EMPTY;
        return std::make_shared<const GSSNode>(std::move(new_children), new_acc);
    }

    static GSSNodePtr gss_merge_many(const std::vector<GSSNodePtr>& lst) {
        if (lst.empty()) return GSS_EMPTY;
        GSSNodePtr result = lst[0];
        for (size_t i = 1; i < lst.size(); ++i) {
            result = gss_merge(result, lst[i]);
        }
        return result;
    }

    static GSSNodePtr gss_push(GSSNodePtr g, int value) {
        if (g->is_empty()) return GSS_EMPTY;
        return std::make_shared<const GSSNode>(
            std::unordered_map<int, GSSNodePtr>{{value, g}}, nullptr
        );
    }

    static GSSNodePtr gss_pop(GSSNodePtr g) {
        if (g->is_empty()) return GSS_EMPTY;
        GSSNodePtr result = GSS_EMPTY;
        for (const auto& kv : g->children) {
            result = gss_merge(result, kv.second);
        }
        return result;
    }

    static GSSNodePtr gss_popn(GSSNodePtr g, int n) {
        if (n <= 0) return g;
        if (g->is_empty()) return GSS_EMPTY;

        auto it = g->popn_cache.find(n);
        if (it != g->popn_cache.end()) return it->second;

        GSSNodePtr result = GSS_EMPTY;
        for (const auto& kv : g->children) {
            result = gss_merge(result, gss_popn(kv.second, n - 1));
        }
        g->popn_cache[n] = result;
        return result;
    }

    static GSSNodePtr gss_isolate_many(GSSNodePtr g, const std::vector<int>& values) {
        std::unordered_map<int, GSSNodePtr> new_children;
        for (int v : values) {
            auto it = g->children.find(v);
            if (it != g->children.end()) {
                new_children[v] = it->second;
            }
        }
        if (new_children.empty()) return GSS_EMPTY;
        return std::make_shared<const GSSNode>(std::move(new_children), nullptr);
    }

    // -----------------------------
    // Acc/GSS transforms needed by algorithm
    // -----------------------------
    GSSNodePtr prune_by_terminals_map(GSSNodePtr g, const std::unordered_map<int, RangeSet>& terminals_map) {
        if (g->is_empty()) return GSS_EMPTY;

        std::shared_ptr<Acc> new_acc = g->acc;
        if (g->acc) {
            bool keep = true;
            for (const auto& [sid, matched] : terminals_map) {
                auto it = g->acc->terminals_union.find(sid);
                if (it != g->acc->terminals_union.end()) {
                    if (!matched.intersection_with(it->second).is_empty()) {
                        keep = false;
                        break;
                    }
                }
            }
            if (!keep) new_acc = nullptr;
        }

        std::unordered_map<int, GSSNodePtr> new_children;
        bool changed = false;
        for (const auto& [val, child] : g->children) {
            GSSNodePtr new_child = prune_by_terminals_map(child, terminals_map);
            if (new_child != child) changed = true;
            if (!new_child->is_empty()) new_children[val] = new_child;
        }
        if (new_children.size() != g->children.size()) changed = true;

        if (!changed && new_acc == g->acc) return g;
        if (new_children.empty() && !new_acc) return GSS_EMPTY;
        return std::make_shared<const GSSNode>(std::move(new_children), new_acc);
    }

    GSSNodePtr apply_state_map_to_gss(GSSNodePtr g, const std::unordered_map<int, int>& state_map) {
        if (g->is_empty()) return GSS_EMPTY;

        std::shared_ptr<Acc> new_acc = g->acc;
        if (g->acc) {
            auto na = std::make_shared<Acc>();
            na->llm_mask = g->acc->llm_mask;
            for (const auto& [old_sid, bv] : g->acc->terminals_union) {
                auto it = state_map.find(old_sid);
                if (it != state_map.end()) {
                    int new_sid = it->second;
                    auto it2 = na->terminals_union.find(new_sid);
                    if (it2 != na->terminals_union.end()) {
                        it2->second = it2->second.union_with(bv);
                    } else {
                        na->terminals_union[new_sid] = bv;
                    }
                }
            }
            new_acc = na;
        }

        std::unordered_map<int, GSSNodePtr> new_children;
        bool changed = false;
        for (const auto& [val, child] : g->children) {
            GSSNodePtr new_child = apply_state_map_to_gss(child, state_map);
            if (new_child != child) changed = true;
            new_children[val] = new_child;
        }

        if (!changed && new_acc == g->acc) return g;
        return std::make_shared<const GSSNode>(std::move(new_children), new_acc);
    }

    GSSNodePtr disallow_in_state(GSSNodePtr g, int state_id, int terminal_id) {
        if (g->is_empty()) return GSS_EMPTY;
        RangeSet to_add = RangeSet::from_indices({(unsigned long long)terminal_id});

        std::shared_ptr<Acc> new_acc = g->acc;
        if (g->acc) {
            auto na = std::make_shared<Acc>(*g->acc);
            auto it = na->terminals_union.find(state_id);
            if (it != na->terminals_union.end()) {
                it->second = it->second.union_with(to_add);
            } else {
                na->terminals_union[state_id] = to_add;
            }
            new_acc = na;
        }

        std::unordered_map<int, GSSNodePtr> new_children;
        bool changed = false;
        for (const auto& [val, child] : g->children) {
            GSSNodePtr new_child = disallow_in_state(child, state_id, terminal_id);
            if (new_child != child) changed = true;
            new_children[val] = new_child;
        }

        if (!changed && new_acc == g->acc) return g;
        return std::make_shared<const GSSNode>(std::move(new_children), new_acc);
    }

    bool state_matches_bitset(const py::object& bitset, int sid) const {
        // Treat empty bitset as wildcard (matches all)
        bool empty = bitset.attr("is_empty")().cast<bool>();
        if (empty) return true;
        return bitset.attr("contains")(py::int_(sid)).cast<bool>();
        }

    GSSNodePtr process_token(GSSNodePtr g, int terminal_id) {
        // heads_by_state: state_id -> vector<GSS>
        std::unordered_map<int, std::vector<GSSNodePtr>> heads_by_state;
        for (const auto& [state_id, child_gss] : g->children) {
            heads_by_state[state_id].push_back(
                std::make_shared<const GSSNode>(std::unordered_map<int, GSSNodePtr>{{state_id, child_gss}}, nullptr)
            );
        }

        std::vector<GSSNodePtr> shifted_gsses;

        while (!heads_by_state.empty()) {
            // pop one element
            auto it = heads_by_state.begin(); // Inefficient, but matches Python logic
            int state_id = it->first;
            std::vector<GSSNodePtr> gsss = std::move(it->second);
            heads_by_state.erase(it);

            // merge_many
            GSSNodePtr state_gss = gss_merge_many(gsss);

            // lookup row
            auto row_it = parser_.find(state_id);
            if (row_it == parser_.end()) continue;
            Row &row = row_it->second;

            auto act_it = row.actions.find(terminal_id);
            if (act_it == row.actions.end()) continue;
            ActionsOnTerminal &action = act_it->second;

            auto handle_shift = [&](int shift_to_state_id, GSSNodePtr gss_to_shift) {
                GSSNodePtr shifted = gss_push(gss_to_shift, shift_to_state_id);
                shifted_gsses.push_back(std::move(shifted));
            };

            auto handle_reduce = [&](int nt_id, int len, GSSNodePtr gss_to_reduce) {
                GSSNodePtr popped_gss = gss_popn(gss_to_reduce, len);
                for (const auto& [from_sid, _] : popped_gss->children) {
                    auto goto_it = parser_.find(from_sid);
                    if (goto_it == parser_.end()) continue;
                    Row &from_row = goto_it->second;
                    auto gt = from_row.gotos.find(nt_id);
                    if (gt == from_row.gotos.end()) continue;
                    int goto_state_id = gt->second;

                    GSSNodePtr isolated = gss_isolate_many(popped_gss, {from_sid});
                    GSSNodePtr goto_gss = gss_push(isolated, goto_state_id);

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

        GSSNodePtr merged = gss_merge_many(shifted_gsses);
        return merged;
    }

    GSSNodePtr initialize_gss_accs(GSSNodePtr g) {
        if (g->is_empty()) return GSS_EMPTY;

        std::shared_ptr<Acc> new_acc = g->acc;
        if (g->acc) {
            RangeSet disallowed_llm_mask = RangeSet::empty();
            for (const auto& [tsid, disallowed_terminals] : g->acc->terminals_union) {
                if (tsid > tokenizer_max_state_) continue;
                auto it = pmc_native_.find(tsid);
                if (it == pmc_native_.end()) continue;
                const auto& terms_to_llm = it->second;

                for (unsigned long long terminal_id_ull : disallowed_terminals.to_indices()) {
                    int terminal_id = static_cast<int>(terminal_id_ull);
                    auto it2 = terms_to_llm.find(terminal_id);
                    if (it2 != terms_to_llm.end()) {
                        disallowed_llm_mask = disallowed_llm_mask.union_with(it2->second);
                    }
                }
            }
            RangeSet allowed_mask = universe_rangeset_.difference_with(disallowed_llm_mask);
            auto na = std::make_shared<Acc>();
            na->llm_mask = allowed_mask;
            new_acc = na;
        }

        std::unordered_map<int, GSSNodePtr> new_children;
        bool changed = false;
        for (const auto& [val, child] : g->children) {
            GSSNodePtr new_child = initialize_gss_accs(child);
            if (new_child != child) changed = true;
            new_children[val] = new_child;
        }

        if (!changed && new_acc == g->acc) return g;
        return std::make_shared<const GSSNode>(std::move(new_children), new_acc);
    }

    GSSNodePtr intersect_llm_mask(GSSNodePtr g, const RangeSet& limiter) {
        if (g->is_empty() || limiter.is_empty()) return GSS_EMPTY;

        std::shared_ptr<Acc> new_acc = g->acc;
        if (g->acc) {
            RangeSet new_mask = g->acc->llm_mask.intersection_with(limiter);
            if (new_mask.is_empty()) new_acc = nullptr;
            else if (!new_mask.operator==(g->acc->llm_mask)) {
                auto na = std::make_shared<Acc>(*g->acc);
                na->llm_mask = new_mask;
                new_acc = na;
            }
        }

        std::unordered_map<int, GSSNodePtr> new_children;
        for (const auto& [val, child] : g->children) {
            GSSNodePtr new_child = intersect_llm_mask(child, limiter);
            if (!new_child->is_empty()) new_children[val] = new_child;
        }

        if (new_children.empty() && !new_acc) return GSS_EMPTY;
        return std::make_shared<const GSSNode>(std::move(new_children), new_acc);
    }

    RangeSet union_llm_masks(GSSNodePtr g) {
        if (g->is_empty()) return RangeSet::empty();
        RangeSet r = g->acc ? g->acc->llm_mask : RangeSet::empty();
        for (const auto& [_, child] : g->children) {
            r = r.union_with(union_llm_masks(child));
        }
        return r;
    }
};

Engine::GSSNodePtr Engine::GSS_EMPTY = std::make_shared<const Engine::GSSNode>(
    std::unordered_map<int, Engine::GSSNodePtr>{}, nullptr
);

PYBIND11_MODULE(precompute3_engine, m) {
    m.doc() = "C++ engine for precompute3_model_cpp: commit() and get_mask(), fully self-contained GSS implementation";

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
