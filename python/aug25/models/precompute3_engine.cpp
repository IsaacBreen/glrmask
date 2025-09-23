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
           py::dict internal_to_original_map,
           py::object RangeSetClass)
        : tokenizer_(std::move(tokenizer)),
          tokenizer_initial_state_(tokenizer_initial_state),
          tokenizer_max_state_(tokenizer_max_state),
          parser_data_(std::move(parser_data)),
          roots_map_py_(std::move(roots_map_py)),
          pmc_(std::move(possible_matches)),
          all_internal_llm_tokens_bitset_(std::move(all_internal_llm_tokens_bitset)),
          internal_to_original_map_(std::move(internal_to_original_map)),
          RangeSetClass_(std::move(RangeSetClass)) {

        if (!ignore_terminal_id_or_none.is_none()) {
            ignore_terminal_id_ = py::cast<int>(ignore_terminal_id_or_none);
        }

        // Cache RangeSet helpers
        range_empty_ = RangeSetClass_.attr("empty");
        range_from_indices_ = RangeSetClass_.attr("from_indices");
        range_from_ranges_ = RangeSetClass_.attr("from_ranges");

        // Universe RangeSet for get_mask init
        universe_rangeset_ = range_from_ranges_(all_internal_llm_tokens_bitset_.attr("to_ranges")());

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

        // Initialize state: one GSS per tokenizer initial state with start parser state on stack
        GSS initial_gss;
        Stack st;
        st.states.push_back(start_state_id_);
        st.acc = std::make_shared<Acc>();
        st.acc->llm_mask = range_empty_(); // empty initial; get_mask will initialize
        // terminals_union is empty map by default
        initial_gss.stacks.push_back(std::move(st));
        state_[tokenizer_initial_state_] = std::move(initial_gss);
    }

    void commit(py::bytes token_bytes) {
        // Build terminals_map and state_map
        std::string token = token_bytes; // implicit cast via pybind11
        std::unordered_map<int, py::object> terminals_map; // sid -> RangeSet of matched terminals
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
            py::object matched_rs = range_from_indices_(matched_terminals);
            terminals_map[tokenizer_sid] = matched_rs;
        }

        // 2) Prune and map per-state GSS (rename terminals_union keys according to state_map)
        std::unordered_map<int, GSS> temp_states;
        for (auto &kv : state_) {
            int tokenizer_sid = kv.first;
            const GSS &gss = kv.second;

            GSS pruned = prune_by_terminals_map(gss, terminals_map);
            if (!pruned.stacks.empty()) {
                GSS mapped = apply_state_map_to_gss(pruned, state_map);
                if (!mapped.stacks.empty()) {
                    temp_states[tokenizer_sid] = std::move(mapped);
                }
            }
        }

        // 3) Main BFS over token bytes
        struct Item {
            int offset;
            int tokenizer_sid;
            GSS gss;
        };

        std::deque<Item> q;
        for (auto &kv : temp_states) {
            q.push_back(Item{0, kv.first, std::move(kv.second)});
        }

        // new states being built
        std::unordered_map<int, std::vector<GSS>> new_states_vec;

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

                GSS processed = cur.gss;
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

                if (!processed.stacks.empty()) {
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
        std::unordered_map<int, GSS> merged_states;
        for (auto &kv : new_states_vec) {
            int sid = kv.first;
            const std::vector<GSS> &lst = kv.second;
            if (lst.empty()) continue;
            GSS merged = merge_many(lst);
            if (!merged.stacks.empty()) {
                merged_states[sid] = std::move(merged);
            }
        }

        // Update internal state
        state_ = std::move(merged_states);
    }

    py::object get_mask() {
        // values: node_id -> GSS
        std::unordered_map<int, GSS> values;
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
            const GSS &gss = kv.second;

            GSS gss_initialized = initialize_gss_accs(gss);

            auto itv = values.find(root);
            if (itv != values.end()) {
                values[root] = merge(values[root], gss_initialized);
            } else {
                values[root] = std::move(gss_initialized);
            }
            int d = arena_.at(root).max_depth;
            enqueue(d, root);
        }

        py::object final_mask = range_empty_();

        while (!depth_heap.empty()) {
            int depth = depth_heap.top(); depth_heap.pop();
            auto &bucket = todo[depth];

            while (!bucket.empty()) {
                int node = *bucket.begin();
                bucket.erase(bucket.begin());
                auto itv = values.find(node);
                if (itv == values.end()) continue;
                GSS gss_node = std::move(itv->second);
                values.erase(itv);

                // End-node handling
                const NodeInfo &info = arena_.at(node);
                if (info.is_end) {
                    py::object reduced = union_llm_masks(gss_node);
                    final_mask = final_mask.attr("union")(reduced);
                }

                // Traverse edges
                for (const Edge &e : info.edges) {
                    GSS popped = popn(gss_node, e.pop);
                    if (popped.stacks.empty()) continue;

                    for (const DestEdge &de : e.dests) {
                        // Determine which top states to keep
                        std::unordered_set<int> keep;
                        for (auto &st : popped.stacks) {
                            if (st.states.empty()) continue;
                            int top = st.states.back();
                            if (state_matches_bitset(de.state_bv, top)) {
                                keep.insert(top);
                            }
                        }
                        if (keep.empty()) continue;

                        GSS child = isolate_many(popped, keep);
                        if (child.stacks.empty()) continue;

                        // intersect_and_prune with edge's llm_bv (as RangeSet)
                        GSS child2 = intersect_llm_mask(child, e.llm_bv_rangeset);
                        if (child2.stacks.empty()) continue;

                        int dnode = de.dest_idx;
                        auto it_child = values.find(dnode);
                        if (it_child != values.end()) {
                            values[dnode] = merge(it_child->second, child2);
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
        py::object idxs = final_mask.attr("to_indices")();
        for (auto iobj : idxs) {
            int i = py::cast<int>(iobj);
            if (internal_to_original_map_.contains(py::int_(i))) {
                int mapped = py::cast<int>(internal_to_original_map_[py::int_(i)]);
                original_indices.push_back(static_cast<unsigned long long>(mapped));
            }
        }
        py::object res = range_from_indices_(original_indices);
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
        py::object llm_bv_rangeset; // icl_rangeset.RangeSet (for intersection)
        std::vector<DestEdge> dests;
    };

    struct NodeInfo {
        std::vector<Edge> edges;
        bool is_end{false};
        int max_depth{0};
    };

    struct Acc {
        // terminals_union: tokenizer_state_id -> RangeSet of disallowed terminals
        std::unordered_map<int, py::object> terminals_union;
        // current allowed LLM mask (RangeSet)
        py::object llm_mask;
    };

    struct Stack {
        std::vector<int> states;
        std::shared_ptr<Acc> acc;
    };

    struct GSS {
        std::vector<Stack> stacks;
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
    py::dict pmc_;
    py::object all_internal_llm_tokens_bitset_;

    // Universe RangeSet
    py::object universe_rangeset_;

    py::dict internal_to_original_map_;

    // Current GLR state: tokenizer_state_id -> GSS
    std::unordered_map<int, GSS> state_;

    // RangeSet helpers
    py::object RangeSetClass_;
    py::object range_empty_;
    py::object range_from_indices_;
    py::object range_from_ranges_;

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
                py::object llm_bv_rs = range_from_ranges_(llm_bv_bitset.attr("to_ranges")());

                std::vector<DestEdge> dests;
                py::object dest_map = entry[1];
                for (auto d : dest_map) {
                    py::tuple dd = py::cast<py::tuple>(d);
                    int dest_idx = py::cast<int>(dd[0]);
                    py::object state_bv_json = py::reinterpret_borrow<py::object>(dd[1]);

                    py::object state_bv = BitsetClass.attr("from_json_string")(dumps(state_bv_json));
                    dests.push_back(DestEdge{dest_idx, state_bv});
                }

                Edge e{pop, llm_bv_rs, std::move(dests)};
                info.edges.push_back(std::move(e));
            }

            arena_[uid] = std::move(info);
        }
    }

    // -----------------------------
    // GSS primitives (naive vector-of-stacks)
    // -----------------------------
    static bool states_equal(const std::vector<int>& a, const std::vector<int>& b) {
        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
    }

    static std::string stack_key(const Stack& s) {
        std::string out;
        out.reserve(32 + s.states.size() * 6);
        out.append(std::to_string(reinterpret_cast<std::uintptr_t>(s.acc.get())));
        out.push_back('|');
        for (size_t i = 0; i < s.states.size(); ++i) {
            if (i) out.push_back(',');
            out.append(std::to_string(s.states[i]));
        }
        return out;
    }

    static void dedup_stacks(std::vector<Stack>& stacks) {
        std::unordered_map<std::string, size_t> seen;
        std::vector<Stack> out;
        out.reserve(stacks.size());
        for (auto &s : stacks) {
            std::string k = stack_key(s);
            if (seen.find(k) == seen.end()) {
                seen.emplace(std::move(k), out.size());
                out.push_back(std::move(s));
            }
        }
        stacks.swap(out);
    }

    static GSS merge(const GSS& a, const GSS& b) {
        GSS r;
        r.stacks.reserve(a.stacks.size() + b.stacks.size());
        r.stacks.insert(r.stacks.end(), a.stacks.begin(), a.stacks.end());
        r.stacks.insert(r.stacks.end(), b.stacks.begin(), b.stacks.end());
        dedup_stacks(r.stacks);
        return r;
    }

    static GSS merge_many(const std::vector<GSS>& lst) {
        if (lst.empty()) return GSS{};
        GSS r;
        size_t total = 0;
        for (auto &g : lst) total += g.stacks.size();
        r.stacks.reserve(total);
        for (auto &g : lst) {
            r.stacks.insert(r.stacks.end(), g.stacks.begin(), g.stacks.end());
        }
        dedup_stacks(r.stacks);
        return r;
    }

    static std::unordered_set<int> peek(const GSS& g) {
        std::unordered_set<int> s;
        for (auto &st : g.stacks) {
            if (!st.states.empty()) {
                s.insert(st.states.back());
            }
        }
        return s;
    }

    static GSS isolate(const GSS& g, int value) {
        GSS r;
        for (auto &st : g.stacks) {
            if (!st.states.empty() && st.states.back() == value) {
                r.stacks.push_back(st);
            }
        }
        return r;
    }

    static GSS isolate_many(const GSS& g, const std::unordered_set<int>& values) {
        GSS r;
        for (auto &st : g.stacks) {
            if (!st.states.empty() && values.find(st.states.back()) != values.end()) {
                r.stacks.push_back(st);
            }
        }
        return r;
    }

    static GSS push(const GSS& g, int value) {
        GSS r;
        r.stacks.reserve(g.stacks.size());
        for (auto &st : g.stacks) {
            Stack ns = st;
            ns.states.push_back(value);
            r.stacks.push_back(std::move(ns));
        }
        return r;
    }

    static GSS popn(const GSS& g, int n) {
        if (n <= 0) return g;
        GSS r;
        for (auto &st : g.stacks) {
            if (static_cast<int>(st.states.size()) >= n) {
                Stack ns;
                ns.acc = st.acc;
                ns.states.assign(st.states.begin(), st.states.end() - n);
                r.stacks.push_back(std::move(ns));
            }
        }
        return r;
    }

    // -----------------------------
    // Acc/GSS transforms needed by algorithm
    // -----------------------------
    GSS prune_by_terminals_map(const GSS& g, const std::unordered_map<int, py::object>& terminals_map) {
        GSS r;
        py::object rs_empty = range_empty_();
        for (auto &st : g.stacks) {
            bool keep = true;
            for (auto &kv : terminals_map) {
                int sid = kv.first;
                const py::object &matched = kv.second;
                auto it = st.acc->terminals_union.find(sid);
                py::object disallowed = (it != st.acc->terminals_union.end()) ? it->second : rs_empty;
                py::object inter = matched.attr("intersection")(disallowed);
                bool ok = inter.attr("is_empty")().cast<bool>();
                if (!ok) {
                    keep = false; break;
                }
            }
            if (keep) r.stacks.push_back(st);
        }
        return r;
    }

    GSS apply_state_map_to_gss(const GSS& g, const std::unordered_map<int, int>& state_map) {
        GSS r;
        py::object rs_empty = range_empty_();
        for (auto &st : g.stacks) {
            std::shared_ptr<Acc> na = std::make_shared<Acc>();
            na->llm_mask = st.acc->llm_mask;
            // merge mapped bitsets
            for (auto &kv : st.acc->terminals_union) {
                int old_sid = kv.first;
                auto it = state_map.find(old_sid);
                if (it == state_map.end()) continue;
                int new_sid = it->second;
                py::object src = kv.second;
                py::object curr = na->terminals_union.count(new_sid) ? na->terminals_union[new_sid] : rs_empty;
                na->terminals_union[new_sid] = curr.attr("union")(src);
            }
            Stack ns = st;
            ns.acc = std::move(na);
            r.stacks.push_back(std::move(ns));
        }
        return r;
    }

    GSS disallow_in_state(const GSS& g, int state_id, int terminal_id) {
        GSS r;
        std::vector<unsigned long long> idx{static_cast<unsigned long long>(terminal_id)};
        py::object to_add = range_from_indices_(idx);
        py::object rs_empty = range_empty_();

        for (auto &st : g.stacks) {
            std::shared_ptr<Acc> na = std::make_shared<Acc>(*st.acc);
            py::object curr = na->terminals_union.count(state_id) ? na->terminals_union[state_id] : rs_empty;
            py::object new_bv = curr.attr("union")(to_add);
            na->terminals_union[state_id] = new_bv;

            Stack ns = st;
            ns.acc = std::move(na);
            r.stacks.push_back(std::move(ns));
        }
        return r;
    }

    bool state_matches_bitset(const py::object& bitset, int sid) const {
        // Treat empty bitset as wildcard (matches all)
        bool empty = bitset.attr("is_empty")().cast<bool>();
        if (empty) return true;
        return bitset.attr("contains")(py::int_(sid)).cast<bool>();
        }

    GSS process_token(const GSS& g, int terminal_id) {
        // heads_by_state: state_id -> vector<GSS>
        std::unordered_map<int, std::vector<GSS>> heads_by_state;
        std::unordered_set<int> tops = peek(g);
        for (int state_id : tops) {
            GSS isol = isolate(g, state_id);
            heads_by_state[state_id].push_back(std::move(isol));
        }

        std::vector<GSS> shifted_gsses;

        while (!heads_by_state.empty()) {
            // pop one element
            auto it = heads_by_state.begin();
            int state_id = it->first;
            std::vector<GSS> gsss = std::move(it->second);
            heads_by_state.erase(it);

            // merge_many
            GSS state_gss = merge_many(gsss);

            // lookup row
            auto row_it = parser_.find(state_id);
            if (row_it == parser_.end()) continue;
            Row &row = row_it->second;

            auto act_it = row.actions.find(terminal_id);
            if (act_it == row.actions.end()) continue;
            ActionsOnTerminal &action = act_it->second;

            auto handle_shift = [&](int shift_to_state_id, const GSS& gss_to_shift) {
                GSS shifted = push(gss_to_shift, shift_to_state_id);
                shifted_gsses.push_back(std::move(shifted));
            };

            auto handle_reduce = [&](int nt_id, int len, const GSS& gss_to_reduce) {
                GSS popped_gss = popn(gss_to_reduce, len);
                std::unordered_set<int> from_states = peek(popped_gss);
                for (int from_sid : from_states) {
                    auto goto_it = parser_.find(from_sid);
                    if (goto_it == parser_.end()) continue;
                    Row &from_row = goto_it->second;
                    auto gt = from_row.gotos.find(nt_id);
                    if (gt == from_row.gotos.end()) continue;
                    int goto_state_id = gt->second;
                    GSS goto_gss = push(isolate(popped_gss, from_sid), goto_state_id);
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

        GSS merged = merge_many(shifted_gsses);
        return merged;
    }

    GSS initialize_gss_accs(const GSS& g) {
        GSS r;
        for (auto &st : g.stacks) {
            // Build disallowed mask as RangeSet
            py::object disallowed_llm_mask = range_empty_();

            for (auto &kv : st.acc->terminals_union) {
                int tsid = kv.first;
                if (tsid > tokenizer_max_state_) continue;
                if (!pmc_.contains(py::int_(tsid))) continue;

                py::dict terms_to_llm = py::cast<py::dict>(pmc_[py::int_(tsid)]);
                py::object indices = kv.second.attr("to_indices")();
                for (auto idx : indices) {
                    int terminal_id = py::cast<int>(idx);
                    if (terms_to_llm.contains(py::int_(terminal_id))) {
                        py::object bit = py::reinterpret_borrow<py::object>(terms_to_llm[py::int_(terminal_id)]);
                        py::object rs = range_from_ranges_(bit.attr("to_ranges")());
                        disallowed_llm_mask = disallowed_llm_mask.attr("union")(rs);
                    }
                }
            }

            py::object allowed_mask = universe_rangeset_.attr("difference")(disallowed_llm_mask);

            std::shared_ptr<Acc> na = std::make_shared<Acc>();
            na->llm_mask = allowed_mask;
            // terminals_union -> empty
            Stack ns = st;
            ns.acc = std::move(na);
            r.stacks.push_back(std::move(ns));
        }
        return r;
    }

    GSS intersect_llm_mask(const GSS& g, const py::object& limiter) {
        GSS r;
        for (auto &st : g.stacks) {
            py::object new_mask = st.acc->llm_mask.attr("intersection")(limiter);
            bool is_empty_mask = new_mask.attr("is_empty")().cast<bool>();
            if (is_empty_mask) continue;

            std::shared_ptr<Acc> na = std::make_shared<Acc>(*st.acc);
            na->llm_mask = new_mask;

            Stack ns = st;
            ns.acc = std::move(na);
            r.stacks.push_back(std::move(ns));
        }
        return r;
    }

    py::object union_llm_masks(const GSS& g) {
        py::object acc = range_empty_();
        for (auto &st : g.stacks) {
            acc = acc.attr("union")(st.acc->llm_mask);
        }
        return acc;
    }
};

PYBIND11_MODULE(precompute3_engine, m) {
    m.doc() = "C++ engine for precompute3_model_cpp: commit() and get_mask(), fully self-contained GSS implementation";

    py::class_<Engine>(m, "Engine")
        .def(py::init<py::object,int,int,py::object,py::dict,py::dict,py::dict,py::dict,py::object,py::dict,py::object>(),
             py::arg("tokenizer"),
             py::arg("tokenizer_initial_state"),
             py::arg("tokenizer_max_state"),
             py::arg("ignore_terminal_id_or_none"),
             py::arg("parser_data"),
             py::arg("roots_map_py"),
             py::arg("arena_py"),
             py::arg("possible_matches"),
             py::arg("all_internal_llm_tokens_bitset"),
             py::arg("internal_to_original_map"),
             py::arg("RangeSetClass"))
        .def("commit", &Engine::commit, py::arg("token_bytes"))
        .def("get_mask", &Engine::get_mask);
}
