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
#include <memory>

namespace leveled_gss {

template <typename T, typename Acc>
struct Upper;
template <typename T, typename Acc>
struct Lower;
template <typename T, typename Acc>
struct Interface;
template <typename T, typename Acc>
struct UpperBranch;

template <typename T, typename Acc>
std::shared_ptr<Acc> _merge_optional_acc(const std::shared_ptr<Acc>& a, const std::shared_ptr<Acc>& b) {
    if (!a) return b;
    if (!b) return a;
    if (a == b) return a;
    return a->merge(b);
}

template <typename T, typename Acc>
struct Upper : public std::enable_shared_from_this<Upper<T, Acc>> {
    int max_depth;
    virtual ~Upper() = default;

    std::vector<std::shared_ptr<Upper<T, Acc>>> _all_children_upper() const {
        if (auto ub = std::dynamic_pointer_cast<const UpperBranch<T, Acc>>(this->shared_from_this())) {
            std::vector<std::shared_ptr<Upper<T, Acc>>> children;
            for (const auto& pair1 : ub->children) {
                for (const auto& pair2 : pair1.second) {
                    children.push_back(pair2.second);
                }
            }
            return children;
        }
        return {};
    }

    std::vector<std::shared_ptr<Lower<T, Acc>>> _all_children_lower() const {
        if (auto it = std::dynamic_pointer_cast<const Interface<T, Acc>>(this->shared_from_this())) {
            std::vector<std::shared_ptr<Lower<T, Acc>>> children;
            for (const auto& pair1 : it->children) {
                for (const auto& pair2 : pair1.second) {
                    children.push_back(pair2.second);
                }
            }
            return children;
        }
        return {};
    }
};

template <typename T, typename Acc>
struct Lower : public std::enable_shared_from_this<Lower<T, Acc>> {
    using ChildrenMap = std::unordered_map<T, std::unordered_map<int, std::shared_ptr<Lower<T, Acc>>>>;
    ChildrenMap children;
    bool empty;
    int max_depth;

    Lower(ChildrenMap ch, bool em) : children(std::move(ch)), empty(em) {
        int d = 0;
        for (const auto& pair1 : children) {
            for (const auto& pair2 : pair1.second) {
                d = std::max(d, pair2.second->max_depth);
            }
        }
        max_depth = d + (children.empty() ? 0 : 1);
    }

    std::vector<std::shared_ptr<Lower<T, Acc>>> _all_children() const {
        std::vector<std::shared_ptr<Lower<T, Acc>>> res;
        for (const auto& pair1 : children) {
            for (const auto& pair2 : pair1.second) {
                res.push_back(pair2.second);
            }
        }
        return res;
    }
};

template <typename T, typename Acc>
struct Interface : public Upper<T, Acc> {
    using LowerChildrenMap = std::unordered_map<T, std::unordered_map<int, std::shared_ptr<Lower<T, Acc>>>>;
    LowerChildrenMap children;
    std::shared_ptr<Acc> acc;
    std::shared_ptr<Acc> empty;

    Interface(LowerChildrenMap ch, std::shared_ptr<Acc> a, std::shared_ptr<Acc> em)
        : children(std::move(ch)), acc(std::move(a)), empty(std::move(em)) {
        int d = 0;
        for (const auto& pair1 : children) {
            for (const auto& pair2 : pair1.second) {
                d = std::max(d, pair2.second->max_depth);
            }
        }
        this->max_depth = d + (children.empty() ? 0 : 1);
    }
};

template <typename T, typename Acc>
struct UpperBranch : public Upper<T, Acc> {
    using UpperChildrenMap = std::unordered_map<T, std::unordered_map<int, std::shared_ptr<Upper<T, Acc>>>>;
    UpperChildrenMap children;
    std::shared_ptr<Acc> empty;

    UpperBranch(UpperChildrenMap ch, std::shared_ptr<Acc> em)
        : children(std::move(ch)), empty(std::move(em)) {
        int d = 0;
        for (const auto& pair1 : children) {
            for (const auto& pair2 : pair1.second) {
                d = std::max(d, pair2.second->max_depth);
            }
        }
        this->max_depth = d + (children.empty() ? 0 : 1);
    }
};

template <typename T, typename Acc>
std::shared_ptr<Upper<T, Acc>> try_promote(std::shared_ptr<UpperBranch<T, Acc>> node);

template <typename T, typename Acc>
std::shared_ptr<UpperBranch<T, Acc>> interface_to_upperbranch(std::shared_ptr<Interface<T, Acc>> it) {
    typename UpperBranch<T, Acc>::UpperChildrenMap children;
    for (const auto& [v, kids] : it->children) {
        std::unordered_map<int, std::shared_ptr<Upper<T, Acc>>> v_map;
        for (const auto& lchild_pair : kids) {
            auto lchild = lchild_pair.second;
            auto ci = std::make_shared<Interface<T, Acc>>(
                lchild->children,
                it->acc,
                (lchild->empty ? it->acc : nullptr)
            );
            v_map[ci->max_depth] = ci;
        }
        if (!v_map.empty()) {
            children[v] = v_map;
        }
    }
    
    auto new_empty = it->empty;
    if (it->children.empty()) {
        new_empty = _merge_optional_acc(it->empty, it->acc);
    }

    return std::make_shared<UpperBranch<T, Acc>>(children, new_empty);
}

template <typename T, typename Acc>
std::shared_ptr<Upper<T, Acc>> merge_upper(std::shared_ptr<Upper<T, Acc>> u1, std::shared_ptr<Upper<T, Acc>> u2);

template <typename K, typename V, typename M>
std::unordered_map<K, std::unordered_map<int, V>> _merge_children_by_depth(
    const std::unordered_map<K, std::unordered_map<int, V>>& c1,
    const std::unordered_map<K, std::unordered_map<int, V>>& c2,
    M merge_func) {
    if (&c1 == &c2) return c1;
    std::unordered_map<K, std::unordered_map<int, V>> merged_children;
    std::set<K> all_vals;
    for (const auto& kv : c1) all_vals.insert(kv.first);
    for (const auto& kv : c2) all_vals.insert(kv.first);

    for (const auto& v : all_vals) {
        std::unordered_map<int, std::vector<V>> nodes_by_depth;
        auto it1 = c1.find(v);
        if (it1 != c1.end()) {
            for (const auto& [depth, child] : it1->second) {
                nodes_by_depth[depth].push_back(child);
            }
        }
        auto it2 = c2.find(v);
        if (it2 != c2.end()) {
            for (const auto& [depth, child] : it2->second) {
                nodes_by_depth[depth].push_back(child);
            }
        }
        if (nodes_by_depth.empty()) continue;

        std::unordered_map<int, V> v_out;
        for (auto& [depth, nodes] : nodes_by_depth) {
            if (nodes.empty()) continue;
            V current = nodes[0];
            for (size_t i = 1; i < nodes.size(); ++i) {
                current = merge_func(current, nodes[i]);
            }
            v_out[current->max_depth] = current;
        }
        merged_children[v] = v_out;
    }
    return merged_children;
}

template <typename T, typename Acc>
std::shared_ptr<Upper<T, Acc>> merge_upperbranches(std::shared_ptr<UpperBranch<T, Acc>> a, std::shared_ptr<UpperBranch<T, Acc>> b) {
    if (a == b) return a;
    auto new_empty = _merge_optional_acc(a->empty, b->empty);
    auto merged_children = _merge_children_by_depth(a->children, b->children, &merge_upper<T, Acc>);
    return try_promote(std::make_shared<UpperBranch<T, Acc>>(merged_children, new_empty));
}

template <typename T, typename Acc>
std::shared_ptr<Lower<T, Acc>> merge_lower(std::shared_ptr<Lower<T, Acc>> l1, std::shared_ptr<Lower<T, Acc>> l2) {
    if (l1 == l2) return l1;
    bool new_empty = l1->empty || l2->empty;
    auto merged_children = _merge_children_by_depth(l1->children, l2->children, &merge_lower<T, Acc>);
    return std::make_shared<Lower<T, Acc>>(merged_children, new_empty);
}

template <typename T, typename Acc>
std::shared_ptr<Upper<T, Acc>> merge_interfaces(std::shared_ptr<Interface<T, Acc>> a, std::shared_ptr<Interface<T, Acc>> b) {
    bool acc_equal = (a->acc == b->acc) || (a->acc && b->acc && *a->acc == *b->acc);
    if (acc_equal || a->children == b->children) {
        auto merged_children = _merge_children_by_depth(a->children, b->children, &merge_lower<T, Acc>);
        auto new_acc = _merge_optional_acc(a->acc, b->acc);
        auto new_empty = _merge_optional_acc(a->empty, b->empty);
        return std::make_shared<Interface<T, Acc>>(merged_children, new_acc, new_empty);
    }
    return merge_upperbranches(interface_to_upperbranch(a), interface_to_upperbranch(b));
}

template <typename T, typename Acc>
std::shared_ptr<Upper<T, Acc>> merge_upper(std::shared_ptr<Upper<T, Acc>> u1, std::shared_ptr<Upper<T, Acc>> u2) {
    if (u1 == u2) return u1;
    auto i1 = std::dynamic_pointer_cast<Interface<T, Acc>>(u1);
    auto i2 = std::dynamic_pointer_cast<Interface<T, Acc>>(u2);
    auto ub1 = std::dynamic_pointer_cast<UpperBranch<T, Acc>>(u1);
    auto ub2 = std::dynamic_pointer_cast<UpperBranch<T, Acc>>(u2);

    if (i1 && i2) return merge_interfaces(i1, i2);
    if (ub1 && ub2) return merge_upperbranches(ub1, ub2);

    auto to_ub1 = ub1 ? ub1 : interface_to_upperbranch(i1);
    auto to_ub2 = ub2 ? ub2 : interface_to_upperbranch(i2);
    return merge_upperbranches(to_ub1, to_ub2);
}

template <typename T, typename Acc>
std::shared_ptr<Upper<T, Acc>> try_promote(std::shared_ptr<UpperBranch<T, Acc>> node) {
    auto all_children = node->_all_children_upper();
    if (all_children.empty()) {
        if (node->empty) {
            return std::make_shared<Interface<T, Acc>>(
                typename Interface<T, Acc>::LowerChildrenMap{},
                node->empty,
                node->empty
            );
        }
        return node;
    }

    bool all_interfaces = true;
    for (const auto& c : all_children) {
        if (!std::dynamic_pointer_cast<Interface<T, Acc>>(c)) {
            all_interfaces = false;
            break;
        }
    }
    if (!all_interfaces) return node;

    std::vector<std::shared_ptr<Acc>> accs_vec;
    if (node->empty) accs_vec.push_back(node->empty);
    for (const auto& c : all_children) {
        auto ic = std::static_pointer_cast<Interface<T, Acc>>(c);
        accs_vec.push_back(ic->acc);
        if (ic->empty) accs_vec.push_back(ic->empty);
    }

    if (!accs_vec.empty()) {
        bool all_same = true;
        for (size_t i = 1; i < accs_vec.size(); ++i) {
            bool equal = (accs_vec[i] == accs_vec[0]) || (*accs_vec[i] == *accs_vec[0]);
            if (!equal) {
                all_same = false;
                break;
            }
        }
        if (all_same) {
            auto the_acc = accs_vec[0];
            if (!the_acc) {
                return std::make_shared<UpperBranch<T, Acc>>(
                    typename UpperBranch<T, Acc>::UpperChildrenMap{}, nullptr);
            }
            typename Interface<T, Acc>::LowerChildrenMap l_children;
            for (const auto& [v, kids] : node->children) {
                std::unordered_map<int, std::shared_ptr<Lower<T, Acc>>> v_map;
                for (const auto& child_pair : kids) {
                    auto ci = std::static_pointer_cast<Interface<T, Acc>>(child_pair.second);
                    auto lower = std::make_shared<Lower<T, Acc>>(ci->children, (ci->empty != nullptr));
                    v_map[lower->max_depth] = lower;
                }
                if (!v_map.empty()) {
                    l_children[v] = v_map;
                }
            }
            return std::make_shared<Interface<T, Acc>>(l_children, the_acc, node->empty);
        }
    }

    return node;
}

template <typename T, typename Acc>
class LeveledGSS {
public:
    std::shared_ptr<Upper<T, Acc>> inner;

    LeveledGSS(std::shared_ptr<Upper<T, Acc>> in) : inner(std::move(in)) {}

    static LeveledGSS create_empty() {
        return LeveledGSS(std::make_shared<UpperBranch<T, Acc>>(
            typename UpperBranch<T, Acc>::UpperChildrenMap{}, nullptr));
    }

    bool is_empty() const {
        if (auto ub = std::dynamic_pointer_cast<UpperBranch<T, Acc>>(inner)) {
            return ub->children.empty() && !ub->empty;
        }
        return false;
    }

    LeveledGSS push(const T& value) const {
        if (is_empty()) return *this;
        if (auto it = std::dynamic_pointer_cast<Interface<T, Acc>>(inner)) {
            auto lower_node = std::make_shared<Lower<T, Acc>>(it->children, it->empty != nullptr);
            typename Interface<T, Acc>::LowerChildrenMap new_children;
            new_children[value][lower_node->max_depth] = lower_node;
            return LeveledGSS(std::make_shared<Interface<T, Acc>>(new_children, it->acc, nullptr));
        } else {
            auto ub = std::static_pointer_cast<UpperBranch<T, Acc>>(inner);
            typename UpperBranch<T, Acc>::UpperChildrenMap new_children;
            new_children[value][ub->max_depth] = ub;
            return LeveledGSS(std::make_shared<UpperBranch<T, Acc>>(new_children, nullptr));
        }
    }

    LeveledGSS popn(int n) const {
        if (n <= 0 || is_empty()) return *this;
        return LeveledGSS(_popn_upper(inner, n));
    }

    std::unordered_set<T> peek() const {
        std::unordered_set<T> keys;
        if (auto it = std::dynamic_pointer_cast<const Interface<T, Acc>>(inner)) {
            for (const auto& kv : it->children) keys.insert(kv.first);
        } else if (auto ub = std::dynamic_pointer_cast<const UpperBranch<T, Acc>>(inner)) {
            for (const auto& kv : ub->children) keys.insert(kv.first);
        }
        return keys;
    }

    LeveledGSS isolate(const T& value) const {
        if (auto it = std::dynamic_pointer_cast<Interface<T, Acc>>(inner)) {
            auto it_ch = it->children.find(value);
            if (it_ch == it->children.end()) return create_empty();
            typename Interface<T, Acc>::LowerChildrenMap new_children;
            new_children[value] = it_ch->second;
            return LeveledGSS(std::make_shared<Interface<T, Acc>>(new_children, it->acc, nullptr));
        } else if (auto ub = std::dynamic_pointer_cast<UpperBranch<T, Acc>>(inner)) {
            auto it_ch = ub->children.find(value);
            if (it_ch == ub->children.end()) return create_empty();
            typename UpperBranch<T, Acc>::UpperChildrenMap new_children;
            new_children[value] = it_ch->second;
            return LeveledGSS(try_promote(std::make_shared<UpperBranch<T, Acc>>(new_children, nullptr)));
        }
        return create_empty();
    }

    LeveledGSS isolate_many(const std::unordered_set<T>& values) const {
        if (auto it = std::dynamic_pointer_cast<Interface<T, Acc>>(inner)) {
            typename Interface<T, Acc>::LowerChildrenMap filtered_children;
            for(const T& v : values) {
                if (it->children.count(v)) filtered_children[v] = it->children.at(v);
            }
            if (filtered_children.empty()) return create_empty();
            return LeveledGSS(std::make_shared<Interface<T, Acc>>(filtered_children, it->acc, nullptr));
        } else if (auto ub = std::dynamic_pointer_cast<UpperBranch<T, Acc>>(inner)) {
            typename UpperBranch<T, Acc>::UpperChildrenMap filtered_children;
            for(const T& v : values) {
                if (ub->children.count(v)) filtered_children[v] = ub->children.at(v);
            }
            if (filtered_children.empty()) return create_empty();
            return LeveledGSS(try_promote(std::make_shared<UpperBranch<T, Acc>>(filtered_children, nullptr)));
        }
        return create_empty();
    }

    template <typename F>
    LeveledGSS apply(F&& func) const {
        return LeveledGSS(_apply(inner, func));
    }

    template <typename F>
    LeveledGSS apply_and_prune(F&& func) const {
        auto res = _apply_and_prune(inner, func);
        return res ? LeveledGSS(res) : create_empty();
    }

    template <typename F>
    LeveledGSS prune(F&& pred) const {
        auto mutator = [&](const std::shared_ptr<Acc>& a) {
            return pred(a) ? a : nullptr;
        };
        return apply_and_prune(mutator);
    }

    LeveledGSS merge(const LeveledGSS& other) const {
        return LeveledGSS(merge_upper(inner, other.inner));
    }

    static LeveledGSS merge_many(const std::vector<LeveledGSS>& gss_list) {
        if (gss_list.empty()) return create_empty();
        auto merged_inner = gss_list[0].inner;
        for (size_t i = 1; i < gss_list.size(); ++i) {
            merged_inner = merge_upper(merged_inner, gss_list[i].inner);
        }
        return LeveledGSS(merged_inner);
    }

    std::shared_ptr<Acc> reduce_acc() const {
        std::vector<std::shared_ptr<Acc>> accs;
        std::unordered_set<const void*> visited_nodes;
        std::unordered_set<const void*> visited_accs;
        std::vector<std::shared_ptr<const Upper<T, Acc>>> q;
        q.push_back(inner);

        while(!q.empty()) {
            auto node = q.back();
            q.pop_back();
            if (visited_nodes.count(node.get())) continue;
            visited_nodes.insert(node.get());

            if (auto it = std::dynamic_pointer_cast<const Interface<T, Acc>>(node)) {
                if (it->acc && visited_accs.find(it->acc.get()) == visited_accs.end()) {
                    accs.push_back(it->acc);
                    visited_accs.insert(it->acc.get());
                }
                if (it->empty && visited_accs.find(it->empty.get()) == visited_accs.end()) {
                    accs.push_back(it->empty);
                    visited_accs.insert(it->empty.get());
                }
            } else if (auto ub = std::dynamic_pointer_cast<const UpperBranch<T, Acc>>(node)) {
                if (ub->empty && visited_accs.find(ub->empty.get()) == visited_accs.end()) {
                    accs.push_back(ub->empty);
                    visited_accs.insert(ub->empty.get());
                }
                for(const auto& child : ub->_all_children_upper()) q.push_back(child);
            }
        }
        if (accs.empty()) return nullptr;
        auto res = accs[0];
        for(size_t i = 1; i < accs.size(); ++i) res = res->merge(accs[i]);
        return res;
    }

private:
    std::shared_ptr<Lower<T, Acc>> _popn_lower(std::shared_ptr<Lower<T, Acc>> node, int k) const {
        if (k == 0) return node;
        auto children = node->_all_children();
        if (children.empty()) return std::make_shared<Lower<T, Acc>>(typename Lower<T, Acc>::ChildrenMap{}, false);
        std::vector<std::shared_ptr<Lower<T, Acc>>> popped;
        for (const auto& child : children) popped.push_back(_popn_lower(child, k - 1));
        auto res = popped[0];
        for (size_t i = 1; i < popped.size(); ++i) res = merge_lower(res, popped[i]);
        return res;
    }

    std::shared_ptr<Upper<T, Acc>> _popn_upper(std::shared_ptr<Upper<T, Acc>> node, int k) const {
        if (k == 0) return node;
        if (auto it = std::dynamic_pointer_cast<Interface<T, Acc>>(node)) {
            auto children = it->_all_children_lower();
            if (children.empty()) return create_empty().inner;
            std::vector<std::shared_ptr<Lower<T, Acc>>> popped;
            for (const auto& child : children) popped.push_back(_popn_lower(child, k - 1));
            auto merged = popped[0];
            for (size_t i = 1; i < popped.size(); ++i) merged = merge_lower(merged, popped[i]);
            auto new_empty = merged->empty ? it->acc : nullptr;
            if (merged->children.empty() && !new_empty) return create_empty().inner;
            return std::make_shared<Interface<T, Acc>>(merged->children, it->acc, new_empty);
        } else if (auto ub = std::dynamic_pointer_cast<UpperBranch<T, Acc>>(node)) {
            auto children = ub->_all_children_upper();
            if (children.empty()) return create_empty().inner;
            std::vector<std::shared_ptr<Upper<T, Acc>>> popped;
            for (const auto& child : children) popped.push_back(_popn_upper(child, k - 1));
            auto merged = popped[0];
            for (size_t i = 1; i < popped.size(); ++i) merged = merge_upper(merged, popped[i]);
            return try_promote(std::static_pointer_cast<UpperBranch<T, Acc>>(merged));
        }
        return create_empty().inner;
    }

    template <typename F>
    std::shared_ptr<Upper<T, Acc>> _apply(std::shared_ptr<Upper<T, Acc>> node, F& func) const {
        if (auto it = std::dynamic_pointer_cast<Interface<T, Acc>>(node)) {
            auto new_acc = func(it->acc);
            auto new_empty = it->empty ? func(it->empty) : nullptr;
            return std::make_shared<Interface<T, Acc>>(it->children, new_acc, new_empty);
        } else if (auto ub = std::dynamic_pointer_cast<UpperBranch<T, Acc>>(node)) {
            auto new_empty = ub->empty ? func(ub->empty) : nullptr;
            typename UpperBranch<T, Acc>::UpperChildrenMap new_children;
            for (const auto& [v, kids] : ub->children) {
                for (const auto& [d, child] : kids) {
                    auto new_child = _apply(child, func);
                    new_children[v][new_child->max_depth] = new_child;
                }
            }
            return try_promote(std::make_shared<UpperBranch<T, Acc>>(new_children, new_empty));
        }
        return nullptr;
    }

    template <typename F>
    std::shared_ptr<Upper<T, Acc>> _apply_and_prune(std::shared_ptr<Upper<T, Acc>> node, F& func) const {
        if (auto it = std::dynamic_pointer_cast<Interface<T, Acc>>(node)) {
            auto new_acc = func(it->acc);
            auto new_empty = it->empty ? func(it->empty) : nullptr;
            if (!new_acc && !new_empty) return nullptr;
            if (!new_acc) {
                return try_promote(std::make_shared<UpperBranch<T, Acc>>(typename UpperBranch<T, Acc>::UpperChildrenMap{}, new_empty));
            }
            return std::make_shared<Interface<T, Acc>>(it->children, new_acc, new_empty);
        } else if (auto ub = std::dynamic_pointer_cast<UpperBranch<T, Acc>>(node)) {
            auto new_empty = ub->empty ? func(ub->empty) : nullptr;
            typename UpperBranch<T, Acc>::UpperChildrenMap new_children;
            for (const auto& [v, kids] : ub->children) {
                for (const auto& [d, child] : kids) {
                    auto new_child = _apply_and_prune(child, func);
                    if (new_child) new_children[v][new_child->max_depth] = new_child;
                }
            }
            if (new_children.empty() && !new_empty) return nullptr;
            return try_promote(std::make_shared<UpperBranch<T, Acc>>(new_children, new_empty));
        }
        return nullptr;
    }
};

} // namespace leveled_gss

namespace py = pybind11;

struct Acc : public std::enable_shared_from_this<Acc> {
    // terminals_union: tokenizer_state_id -> RangeSet of disallowed terminals
    std::unordered_map<int, py::object> terminals_union;
    // current allowed LLM mask (RangeSet)
    py::object llm_mask;

    bool operator==(const Acc& other) const {
        if (!llm_mask.attr("__eq__")(other.llm_mask).cast<bool>()) return false;
        if (terminals_union.size() != other.terminals_union.size()) return false;
        for (auto const& [key, val] : terminals_union) {
            auto it = other.terminals_union.find(key);
            if (it == other.terminals_union.end()) return false;
            if (!val.attr("__eq__")(it->second).cast<bool>()) {
                return false;
            }
        }
        return true;
    }

    std::shared_ptr<Acc> merge(const std::shared_ptr<Acc>& other) const {
        auto n = std::make_shared<Acc>();
        // Merge terminals_union by RangeSet union per key
        n->terminals_union = terminals_union;
        for (auto &kv : other->terminals_union) {
            int key = kv.first;
            const py::object &v = kv.second;
            auto it = n->terminals_union.find(key);
            if (it == n->terminals_union.end()) {
                n->terminals_union.emplace(key, v);
            } else {
                it->second = it->second.attr("union")(v);
            }
        }
        // Union llm masks
        if (llm_mask.is_none()) n->llm_mask = other->llm_mask;
        else if (other->llm_mask.is_none()) n->llm_mask = llm_mask;
        else n->llm_mask = llm_mask.attr("union")(other->llm_mask);
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
        auto initial_acc = std::make_shared<Acc>();
        initial_acc->llm_mask = range_empty_();
        // terminals_union empty by default
        Leveled init = Leveled::create_empty().push(start_state_id_);
        state_[tokenizer_initial_state_] = std::move(init);
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

        // new states being built
        std::unordered_map<int, std::vector<Leveled>> new_states_vec;

        while (!q.empty()) {
            Item cur = std::move(q.front());
            q.pop_front();

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
            Leveled merged = merge_many(lst);
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

        // Seed with initialized accs (compute allowed llm mask from terminals_union)
        for (auto &kv : state_) {
            int sid = kv.first;
            int root = 0;
            auto it = roots_map_.find(sid);
            if (it == roots_map_.end()) continue;
            root = it->second;
            const Leveled &gss = kv.second;

            Leveled gss_initialized = initialize_gss_accs(gss);

            auto itv = values.find(root);
            if (itv != values.end()) {
                values[root] = values[root].merge(gss_initialized);
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
                Leveled gss_node = std::move(itv->second);
                values.erase(itv);

                // End-node handling
                const NodeInfo &info = arena_.at(node);
                if (info.is_end) {
                    auto reduced = gss_node.reduce_acc();
                    if (reduced) {
                        final_mask = final_mask.attr("union")(reduced->llm_mask);
                    }
                }

                // Traverse edges
                for (const Edge &e : info.edges) {
                    Leveled popped = gss_node.popn(e.pop);
                    if (popped.is_empty()) continue;

                    for (const DestEdge &de : e.dests) {
                        // Determine which top states to keep
                        std::unordered_set<int> keep;
                        std::unordered_set<int> top = popped.peek();
                        for (int top_sid : top) {
                            if (state_matches_bitset(de.state_bv, top_sid)) keep.insert(top_sid);
                        }
                        if (keep.empty()) continue;

                        Leveled child = isolate_many(popped, keep);
                        if (child.is_empty()) continue;

                        // intersect_and_prune with edge's llm_bv (as RangeSet)
                        Leveled child2 = intersect_llm_mask(child, e.llm_bv_rangeset);
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

    // Current GLR state: tokenizer_state_id -> LeveledGSS
    std::unordered_map<int, Leveled> state_;

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
                    py::object state_bv_json = dd[1];

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
    // Leveled GSS transforms needed by algorithm
    // -----------------------------
    Leveled prune_by_terminals_map(const Leveled& g, const std::unordered_map<int, py::object>& terminals_map) {
        auto pred = [&](const std::shared_ptr<Acc>& acc) -> bool {
            py::object rs_empty = range_empty_();
            for (auto &kv : terminals_map) {
                int sid = kv.first;
                const py::object &matched = kv.second;
                auto it = acc->terminals_union.find(sid);
                py::object disallowed = (it != acc->terminals_union.end()) ? it->second : rs_empty;
                py::object inter = matched.attr("intersection")(disallowed);
                bool ok = inter.attr("is_empty")().cast<bool>();
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
            py::object rs_empty = range_empty_();
            for (auto &kv : acc->terminals_union) {
                int old_sid = kv.first;
                auto it = state_map.find(old_sid);
                if (it == state_map.end()) continue;
                int new_sid = it->second;
                py::object src = kv.second;
                py::object curr = na->terminals_union.count(new_sid) ? na->terminals_union[new_sid] : rs_empty;
                na->terminals_union[new_sid] = curr.attr("union")(src);
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
            py::object to_add = range_from_indices_(idx);
            py::object rs_empty = range_empty_();
            py::object curr = na->terminals_union.count(state_id) ? na->terminals_union[state_id] : rs_empty;
            py::object new_bv = curr.attr("union")(to_add);
            na->terminals_union[state_id] = new_bv;
            return na;
        };
        return g.apply(transformer);
    }

    bool state_matches_bitset(const py::object& bitset, int sid) const {
        // Treat empty bitset as wildcard (matches all)
        bool empty = bitset.attr("is_empty")().cast<bool>();
        if (empty) return true;
        return bitset.attr("contains")(py::int_(sid)).cast<bool>();
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

    Leveled initialize_gss_accs(const Leveled& g) {
        auto mutator = [&](const std::shared_ptr<Acc>& a) -> std::shared_ptr<Acc> {
            // Build disallowed mask as RangeSet
            py::object disallowed_llm_mask = range_empty_();

            for (auto &kv : a->terminals_union) {
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

            auto na = std::make_shared<Acc>();
            na->llm_mask = allowed_mask;
            // terminals_union -> empty
            return na;
        };
        return g.apply(mutator);
    }

    Leveled intersect_llm_mask(const Leveled& g, const py::object& limiter) {
        auto mutator = [&](const std::shared_ptr<Acc>& a) -> std::shared_ptr<Acc> {
            py::object new_mask = a->llm_mask.attr("intersection")(limiter);
            bool is_empty_mask = new_mask.attr("is_empty")().cast<bool>();
            if (is_empty_mask) return std::shared_ptr<Acc>(nullptr);
            auto na = std::make_shared<Acc>(*a);
            na->llm_mask = new_mask;
            return na;
        };
        return g.apply_and_prune(mutator);
    }

    // Merge many Leveled
    static Leveled merge_many(const std::vector<Leveled>& lst) {
        return Leveled::merge_many(lst);
    }

    // Convenience isolate_many wrapper for unordered_set<int>
    static Leveled isolate_many(const Leveled& g, const std::unordered_set<int>& values) {
        return g.isolate_many(values);
    }
};

PYBIND11_MODULE(precompute3_engine, m) {
    m.doc() = "C++ engine for precompute3_model_cpp: commit() and get_mask(), Leveled GSS implementation";

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
