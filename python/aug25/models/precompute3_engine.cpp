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

namespace py = pybind11;

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
    py::object llm_bv_bitset; // _sep1.Bitset
    py::object llm_bv_rangeset; // icl_rangeset.RangeSet (precomputed for intersection)
    std::vector<DestEdge> dests;
};

struct NodeInfo {
    std::vector<Edge> edges;
    bool is_end{false};
    int max_depth{0};
};

class Engine {
public:
    Engine(py::object tokenizer,
           int tokenizer_initial_state,
           int tokenizer_max_state,
           py::object ignore_terminal_id_or_none,
           py::object parser_table_obj,
           py::dict roots_map_py,
           py::dict arena_py,
           py::dict possible_matches, // tsid -> term_id -> _sep1.Bitset
           py::object all_internal_llm_tokens_bitset,
           py::dict internal_to_original_map,
           py::object PyAccClass,
           py::object RangeSetClass)
        : tokenizer_(std::move(tokenizer)),
          tokenizer_initial_state_(tokenizer_initial_state),
          tokenizer_max_state_(tokenizer_max_state),
          parser_table_obj_(std::move(parser_table_obj)),
          roots_map_py_(std::move(roots_map_py)),
          pmc_(std::move(possible_matches)),
          all_internal_llm_tokens_bitset_(std::move(all_internal_llm_tokens_bitset)),
          internal_to_original_map_(std::move(internal_to_original_map)),
          PyAccClass_(std::move(PyAccClass)),
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

        // Parse roots_map into fast C++ map
        for (auto item : roots_map_py_) {
            int sid = py::cast<int>(item.first);
            int root = py::cast<int>(item.second);
            roots_map_[sid] = root;
        }

        // Parse parser table
        parse_parser_table();

        // Parse arena
        parse_arena(arena_py);
    }

    void set_state(py::dict state) {
        state_ = std::move(state);
        // cache GSS class for static calls
        if (state_.size() > 0) {
            for (auto item : state_) {
                py::object gss = py::reinterpret_borrow<py::object>(item.second);
                gss_class_ = gss.attr("__class__");
                break;
            }
        }
    }

    py::dict commit(py::bytes token_bytes) {
        // Build terminals_map and state_map
        std::string token = token_bytes; // implicit cast via pybind11
        py::dict terminals_map;
        py::dict state_map;

        for (auto item_handle : state_) {
            int tokenizer_sid = py::cast<int>(item_handle.first);
            // py::object gss = py::reinterpret_borrow<py::object>(item.second);
            py::tuple result = tokenizer_.attr("execute_from_state")(py::bytes(token), tokenizer_sid);
            py::object end_state_obj = result[0];
            py::object matches_obj = result[1];

            if (!end_state_obj.is_none()) {
                int end_state = py::cast<int>(end_state_obj);
                state_map[py::int_(tokenizer_sid)] = py::int_(end_state);
            }

            std::vector<unsigned long long> matched_terminals;
            for (auto tm : matches_obj) {
                py::tuple tmt = py::cast<py::tuple>(tm);
                int terminal_id = py::cast<int>(tmt[0]);
                matched_terminals.push_back(static_cast<unsigned long long>(terminal_id));
            }
            py::object matched_rs = range_from_indices_(matched_terminals);
            terminals_map[py::int_(tokenizer_sid)] = matched_rs;
        }

        // Prune and map per-state GSS
        py::dict temp_states;
        // Build predicate and mapper closures in C++
        py::function predicate = make_prune_predicate(terminals_map);
        py::function mapper = make_apply_map(state_map);

        for (auto item_handle : state_.attr("items")()) {
            py::tuple item = py::cast<py::tuple>(item_handle);
            int tokenizer_sid = py::cast<int>(item[0]);
            py::object gss = py::reinterpret_borrow<py::object>(item[1]);

            py::object pruned_gss = gss.attr("prune")(predicate);
            bool is_empty = pruned_gss.attr("is_empty")().cast<bool>();
            if (!is_empty) {
                py::object mapped_gss = pruned_gss.attr("apply")(mapper);
                temp_states[py::int_(tokenizer_sid)] = mapped_gss;
                if (!gss_class_ || gss_class_.is_none()) {
                    gss_class_ = mapped_gss.attr("__class__");
                }
            }
        }

        // Main BFS over token bytes
        // Queue items: (offset, tokenizer_sid, gss)
        struct Item {
            int offset;
            int tokenizer_sid;
            py::object gss;
        };

        std::deque<Item> q;
        for (auto item_handle : temp_states.attr("items")()) {
            py::tuple item = py::cast<py::tuple>(item_handle);
            int tokenizer_sid = py::cast<int>(item[0]);
            py::object gss = py::reinterpret_borrow<py::object>(item[1]);
            q.push_back({0, tokenizer_sid, gss});
        }

        // visited set to avoid cycles: key = offset|sid|id(gss)
        std::unordered_set<std::string> visited;

        py::object py_id = py::module::import("builtins").attr("id");

        std::unordered_map<int, std::vector<py::object>> new_states_vec;

        while (!q.empty()) {
            Item cur = q.front(); q.pop_front();

            long long gss_id = py::cast<long long>(py_id(cur.gss));
            std::string key = std::to_string(cur.offset) + "|" + std::to_string(cur.tokenizer_sid) + "|" + std::to_string(gss_id);
            if (visited.find(key) != visited.end()) {
                continue;
            }
            visited.insert(key);

            std::string suffix = token.substr(static_cast<size_t>(cur.offset));
            py::tuple result = tokenizer_.attr("execute_from_state")(py::bytes(suffix), cur.tokenizer_sid);
            py::object end_state_obj = result[0];
            py::object matches_obj = result[1];

            int end_state = -1;
            bool has_end_state = !end_state_obj.is_none();
            if (has_end_state) end_state = py::cast<int>(end_state_obj);

            for (auto tm : matches_obj) {
                py::tuple tmt = py::cast<py::tuple>(tm);
                int terminal_id = py::cast<int>(tmt[0]);
                int width = py::cast<int>(tmt[1]);

                py::object processed_gss = cur.gss;
                if (!(ignore_terminal_id_.has_value() && terminal_id == *ignore_terminal_id_)) {
                    processed_gss = process_token(cur.gss, terminal_id);
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
                        processed_gss = disallow_in_state(processed_gss, end_state, terminal_id);
                    }
                }

                bool is_empty = processed_gss.attr("is_empty")().cast<bool>();
                if (!is_empty) {
                    int new_offset = cur.offset + width;
                    int next_tokenizer_sid = tokenizer_initial_state_;
                    if (static_cast<size_t>(new_offset) == token.size()) {
                        new_states_vec[next_tokenizer_sid].push_back(processed_gss);
                    } else {
                        q.push_back({new_offset, next_tokenizer_sid, processed_gss});
                    }
                }
            }

            if (has_end_state) {
                new_states_vec[end_state].push_back(cur.gss);
            }
        }

        // Merge and filter empties
        py::dict merged_states;
        for (auto &kv : new_states_vec) {
            int sid = kv.first;
            py::list lst;
            for (auto &g : kv.second) lst.append(g);
            if (lst.size() == 0) continue;
            py::object merged = gss_class_.attr("merge_many")(lst);
            bool is_empty = merged.attr("is_empty")().cast<bool>();
            if (!is_empty) {
                merged_states[py::int_(sid)] = merged;
            }
        }

        // Update internal state
        state_ = merged_states;
        return merged_states;
    }

    py::object get_mask() {
        py::dict state_map = state_;
        py::object final_mask = range_empty_();

        // values: node_id -> GSS
        std::unordered_map<int, py::object> values;
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

        // initialize_acc closure
        py::function initialize_acc = py::cpp_function([&](py::handle acc_obj) -> py::object {
            py::object acc = py::reinterpret_borrow<py::object>(acc_obj);
            py::object terminals_union = acc.attr("terminals_union");

            // Build disallowed mask as RangeSet
            py::object disallowed_llm_mask = range_empty_();

            for (auto item_handle : terminals_union.attr("items")()) {
                py::tuple item = py::cast<py::tuple>(item_handle);
                int tsid = py::cast<int>(item[0]);
                py::object disallowed_terms_rs = py::reinterpret_borrow<py::object>(item[1]);
                if (tsid > tokenizer_max_state_) continue;
                if (!pmc_.contains(py::int_(tsid))) continue;

                py::dict terms_to_llm = py::cast<py::dict>(pmc_[py::int_(tsid)]);
                py::object indices = disallowed_terms_rs.attr("to_indices")();
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
            // Build new PyAcc: terminals_union={}, llm_mask=allowed_mask
            py::dict empty_map;
            py::object new_acc = PyAccClass_(empty_map, allowed_mask);
            return new_acc;
        });

        // Seed
        for (auto item_handle : state_map.attr("items")()) {
            py::tuple item = py::cast<py::tuple>(item_handle);
            int sid = py::cast<int>(item[0]);
            int root = 0;
            auto it = roots_map_.find(sid);
            if (it == roots_map_.end()) continue;
            root = it->second;
            py::object gss = py::reinterpret_borrow<py::object>(item[1]);

            py::object gss_initialized = gss.attr("apply")(initialize_acc);

            if (values.find(root) != values.end()) {
                values[root] = values[root].attr("merge")(gss_initialized);
            } else {
                values[root] = gss_initialized;
            }
            int d = arena_.at(root).max_depth;
            enqueue(d, root);
            if (!gss_class_ || gss_class_.is_none()) {
                gss_class_ = gss.attr("__class__");
            }
        }

        while (!depth_heap.empty()) {
            int depth = depth_heap.top(); depth_heap.pop();
            auto &bucket = todo[depth];

            while (!bucket.empty()) {
                int node = *bucket.begin();
                bucket.erase(bucket.begin());
                py::object gss_node = values[node];
                values.erase(node);

                // End-node handling
                const NodeInfo &info = arena_.at(node);
                if (info.is_end) {
                    py::object reduced = gss_node.attr("reduce_acc")();
                    if (!reduced.is_none()) {
                        py::object llm_mask = reduced.attr("llm_mask");
                        final_mask = final_mask.attr("union")(llm_mask);
                    }
                }

                // Traverse edges
                for (const Edge &e : info.edges) {
                    py::object popped = gss_node.attr("popn")(py::int_(e.pop));
                    bool empty = popped.attr("is_empty")().cast<bool>();
                    if (empty) continue;

                    for (const DestEdge &de : e.dests) {
                        py::object peeked = popped.attr("peek")();
                        std::vector<int> values_to_keep;
                        for (auto sid_py : peeked) {
                            int sid = py::cast<int>(sid_py);
                            if (de.state_bv.attr("contains")(py::int_(sid)).cast<bool>()) {
                                values_to_keep.push_back(sid);
                            }
                        }
                        if (values_to_keep.empty()) continue;

                        py::list keep_list;
                        for (int v : values_to_keep) keep_list.append(py::int_(v));
                        py::object child_gss = popped.attr("isolate_many")(keep_list);
                        if (child_gss.attr("is_empty")().cast<bool>()) continue;

                        // intersect_and_prune with llm_bv (as RangeSet)
                        py::object llm_bv_rs = e.llm_bv_rangeset;
                        py::function intersect_and_prune = py::cpp_function([&, llm_bv_rs](py::handle acc_obj) -> py::object {
                            py::object acc = py::reinterpret_borrow<py::object>(acc_obj);
                            py::object new_mask = acc.attr("llm_mask").attr("intersection")(llm_bv_rs);
                            bool is_empty_mask = new_mask.attr("is_empty")().cast<bool>();
                            if (is_empty_mask) {
                                return py::none();
                            } else {
                                py::object new_acc = PyAccClass_(acc.attr("terminals_union"), new_mask);
                                return new_acc;
                            }
                        });

                        py::object child_gss2 = child_gss.attr("apply_and_prune")(intersect_and_prune);
                        if (child_gss2.attr("is_empty")().cast<bool>()) continue;

                        int dnode = de.dest_idx;
                        auto itv = values.find(dnode);
                        if (itv != values.end()) {
                            values[dnode] = itv->second.attr("merge")(child_gss2);
                        } else {
                            values[dnode] = child_gss2;
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
    // Members
    py::object tokenizer_;
    int tokenizer_initial_state_{0};
    int tokenizer_max_state_{0};
    std::optional<int> ignore_terminal_id_;
    py::object parser_table_obj_;
    py::dict roots_map_py_;

    // Parsed structures
    std::unordered_map<int, Row> parser_;
    std::unordered_map<int, NodeInfo> arena_;
    std::unordered_map<int, int> roots_map_;

    // Possible matches and universe bitset
    py::dict pmc_;
    py::object all_internal_llm_tokens_bitset_;

    // Universe RangeSet
    py::object universe_rangeset_;

    py::dict internal_to_original_map_;

    // State and type helpers
    py::dict state_;
    py::object gss_class_; // cache for static calls

    // PyAcc and RangeSet classes
    py::object PyAccClass_;
    py::object RangeSetClass_;
    py::object range_empty_;
    py::object range_from_indices_;
    py::object range_from_ranges_;

    // Helpers
    void parse_parser_table() {
        // parser_table_obj_ has: start_state_id and table (dict state_id -> Row dataclass)
        py::dict table = parser_table_obj_.attr("table");
        for (auto item_handle : table.attr("items")()) {
            py::tuple item = py::cast<py::tuple>(item_handle);
            int state_id = py::cast<int>(item[0]);
            py::object row_obj = py::reinterpret_borrow<py::object>(item[1]);
            Row row;

            py::dict actions = row_obj.attr("actions");
            for (auto aitem_handle : actions.attr("items")()) {
                py::tuple aitem = py::cast<py::tuple>(aitem_handle);
                int term_id = py::cast<int>(aitem[0]);
                py::object action = py::reinterpret_borrow<py::object>(aitem[1]);

                ActionsOnTerminal aot;
                // Shift: int
                if (py::isinstance<py::int_>(action)) {
                    aot.has_shift = true;
                    aot.shift_state_id = py::cast<int>(action);
                }
                // Reduce: has attributes 'nonterminal_id' and 'len'
                else if (py::hasattr(action, "nonterminal_id") && py::hasattr(action, "len")) {
                    int nt = py::cast<int>(action.attr("nonterminal_id"));
                    int len = py::cast<int>(action.attr("len"));
                    aot.reduces.emplace_back(nt, len);
                }
                // Split: has 'shift' and 'reduces' mapping
                else if (py::hasattr(action, "reduces")) {
                    py::object shift_obj = action.attr("shift");
                    if (!shift_obj.is_none()) {
                        aot.has_shift = true;
                        aot.shift_state_id = py::cast<int>(shift_obj);
                    }
                    py::dict reduces = action.attr("reduces");
                    for (auto len_item_handle : reduces.attr("items")()) {
                        py::tuple len_item = py::cast<py::tuple>(len_item_handle);
                        int len = py::cast<int>(len_item[0]);
                        py::dict nts = py::cast<py::dict>(len_item[1]);
                        for (auto nt_item_handle : nts) {
                            int nt = py::cast<int>(nt_item_handle.first);
                            aot.reduces.emplace_back(nt, len);
                        }
                    }
                }
                row.actions[term_id] = std::move(aot);
            }

            py::dict gotos = row_obj.attr("gotos");
            for (auto gitem_handle : gotos.attr("items")()) {
                py::tuple gitem = py::cast<py::tuple>(gitem_handle);
                int nt = py::cast<int>(gitem[0]);
                int to_state = py::cast<int>(gitem[1]);
                row.gotos[nt] = to_state;
            }

            parser_[state_id] = std::move(row);
        }
    }

    void parse_arena(py::dict arena_py) {
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

            // Children: prefer 'children_bits'
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
                py::object llm_bv_obj = edge_key[1];

                py::object llm_bv_bitset_for_edge;
                py::object llm_bv_rs_for_edge;

                std::string type_name = py::str(llm_bv_obj.get_type().attr("__name__"));

                if (type_name == "RangeSet") {
                    llm_bv_rs_for_edge = llm_bv_obj;
                    llm_bv_bitset_for_edge = py::none();
                } else { // Assume Bitset or something convertible
                    llm_bv_bitset_for_edge = llm_bv_obj;
                    llm_bv_rs_for_edge = range_from_ranges_(llm_bv_obj.attr("to_ranges")());
                }

                std::vector<DestEdge> dests;
                py::object dest_map = entry[1];
                for (auto d : dest_map) {
                    py::tuple dd = py::cast<py::tuple>(d);
                    int dest_idx = py::cast<int>(dd[0]);
                    py::object state_bv = py::reinterpret_borrow<py::object>(dd[1]);
                    dests.push_back(DestEdge{dest_idx, state_bv});
                }

                Edge e{pop, llm_bv_bitset_for_edge, llm_bv_rs_for_edge, std::move(dests)};
                info.edges.push_back(std::move(e));
            }

            arena_[uid] = std::move(info);
        }
    }

    py::function make_prune_predicate(const py::dict& terminals_map) {
        return py::cpp_function([&, terminals_map](py::handle acc_obj) -> bool {
            py::object acc = py::reinterpret_borrow<py::object>(acc_obj);
            py::object disallowed_map = acc.attr("terminals_union");
            py::object rs_empty = range_empty_();

            for (auto item_handle : terminals_map.attr("items")()) {
                py::tuple item = py::cast<py::tuple>(item_handle);
                int state_id = py::cast<int>(item[0]);
                py::object matched_bv = py::reinterpret_borrow<py::object>(item[1]);
                py::object disallowed_for_state;
                if (py::cast<py::dict>(disallowed_map).contains(py::int_(state_id))) {
                    disallowed_for_state = disallowed_map[py::int_(state_id)];
                } else {
                    disallowed_for_state = rs_empty;
                }
                py::object inter = matched_bv.attr("intersection")(disallowed_for_state);
                bool ok = inter.attr("is_empty")().cast<bool>();
                if (!ok) return false;
            }
            return true;
        });
    }

    py::function make_apply_map(const py::dict& state_map) {
        return py::cpp_function([&, state_map](py::handle acc_obj) -> py::object {
            py::object acc = py::reinterpret_borrow<py::object>(acc_obj);
            py::object old_map = acc.attr("terminals_union");
            py::dict new_bvs;
            py::object rs_empty = range_empty_();

            for (auto item_handle : state_map.attr("items")()) {
                py::tuple item = py::cast<py::tuple>(item_handle);
                int old_sid = py::cast<int>(item[0]);
                int new_sid = py::cast<int>(item[1]);
                py::object bv_source;
                if (py::cast<py::dict>(old_map).contains(py::int_(old_sid))) {
                    bv_source = old_map[py::int_(old_sid)];
                } else {
                    bv_source = rs_empty;
                }

                py::object curr = new_bvs.contains(py::int_(new_sid)) ? new_bvs[py::int_(new_sid)] : rs_empty;
                py::object merged = curr.attr("union")(bv_source);
                new_bvs[py::int_(new_sid)] = merged;
            }

            py::object new_acc = PyAccClass_(new_bvs, acc.attr("llm_mask"));
            return new_acc;
        });
    }

    py::object disallow_in_state(const py::object& gss, int state_id, int terminal_id) {
        // apply function that adds terminal_id to terminals_union[state_id]
        py::function fn = py::cpp_function([&, state_id, terminal_id](py::handle acc_obj) -> py::object {
            py::object acc = py::reinterpret_borrow<py::object>(acc_obj);
            py::dict current_map = py::cast<py::dict>(acc.attr("terminals_union"));
            py::object rs_empty = range_empty_();
            py::object curr_bv = current_map.contains(py::int_(state_id)) ? current_map[py::int_(state_id)] : rs_empty;

            std::vector<unsigned long long> idx{static_cast<unsigned long long>(terminal_id)};
            py::object to_add = range_from_indices_(idx);
            py::object new_bv = curr_bv.attr("union")(to_add);

            // Copy map
            py::dict new_map = py::cast<py::dict>(current_map.attr("copy")());
            new_map[py::int_(state_id)] = new_bv;

            py::object new_acc = PyAccClass_(new_map, acc.attr("llm_mask"));
            return new_acc;
        });

        return gss.attr("apply")(fn);
    }

    py::object process_token(const py::object& gss, int terminal_id) {
        // heads_by_state: state_id -> vector of GSS
        std::unordered_map<int, std::vector<py::object>> heads_by_state;
        py::object peeked = gss.attr("peek")();
        for (auto sid_py : peeked) {
            int state_id = py::cast<int>(sid_py);
            py::object isol = gss.attr("isolate")(py::int_(state_id));
            heads_by_state[state_id].push_back(isol);
        }

        std::vector<py::object> shifted_gsses;

        while (!heads_by_state.empty()) {
            // pop one element
            auto it = heads_by_state.begin();
            int state_id = it->first;
            std::vector<py::object> gsss = std::move(it->second);
            heads_by_state.erase(it);

            // merge_many
            py::list lst;
            for (auto &x : gsss) lst.append(x);
            py::object state_gss = gss_class_.attr("merge_many")(lst);

            // lookup row
            auto row_it = parser_.find(state_id);
            if (row_it == parser_.end()) continue;
            Row &row = row_it->second;

            auto act_it = row.actions.find(terminal_id);
            if (act_it == row.actions.end()) continue;
            ActionsOnTerminal &action = act_it->second;

            auto handle_shift = [&](int shift_to_state_id, const py::object& gss_to_shift) {
                py::object shifted = gss_to_shift.attr("push")(py::int_(shift_to_state_id));
                shifted_gsses.push_back(shifted);
            };

            auto handle_reduce = [&](int nt_id, int len, const py::object& gss_to_reduce) {
                py::object popped = gss_to_reduce.attr("popn")(py::int_(len));
                py::object from_states = popped.attr("peek")();
                for (auto from_sid_py : from_states) {
                    int from_sid = py::cast<int>(from_sid_py);
                    auto goto_it = parser_.find(from_sid);
                    if (goto_it == parser_.end()) continue;
                    Row &from_row = goto_it->second;
                    auto gt = from_row.gotos.find(nt_id);
                    if (gt == from_row.gotos.end()) continue;
                    int goto_state_id = gt->second;
                    py::object goto_gss = popped.attr("isolate")(py::int_(from_sid)).attr("push")(py::int_(goto_state_id));
                    heads_by_state[goto_state_id].push_back(goto_gss);
                }
            };

            if (action.has_shift) {
                handle_shift(action.shift_state_id, state_gss);
            }
            for (auto &rd : action.reduces) {
                handle_reduce(rd.first, rd.second, state_gss);
            }
        }

        py::list res_list;
        for (auto &x : shifted_gsses) res_list.append(x);
        py::object merged = gss_class_.attr("merge_many")(res_list);
        return merged;
    }
};

PYBIND11_MODULE(precompute3_engine, m) {
    m.doc() = "C++ engine for precompute3_model_cpp: commit() and get_mask()";

    py::class_<Engine>(m, "Engine")
        .def(py::init<py::object,int,int,py::object,py::object,py::dict,py::dict,py::dict,py::object,py::dict,py::object,py::object>(),
             py::arg("tokenizer"),
             py::arg("tokenizer_initial_state"),
             py::arg("tokenizer_max_state"),
             py::arg("ignore_terminal_id_or_none"),
             py::arg("parser_table_obj"),
             py::arg("roots_map_py"),
             py::arg("arena_py"),
             py::arg("possible_matches"),
             py::arg("all_internal_llm_tokens_bitset"),
             py::arg("internal_to_original_map"),
             py::arg("PyAccClass"),
             py::arg("RangeSetClass"))
        .def("set_state", &Engine::set_state, py::arg("state"))
        .def("commit", &Engine::commit, py::arg("token_bytes"))
        .def("get_mask", &Engine::get_mask);
}
