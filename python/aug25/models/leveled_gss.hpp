#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <deque>
#include <functional>
#include <algorithm>
#include <utility>
#include <cstdint>

//
// A faithful C++ translation of the Python LeveledGSS used in precompute3_model_pure_python_standalone.py.
//
// Design:
// - Template parameters:
//     T:   stack value type (e.g., int parser state id)
//     Acc: accumulator type (shared_ptr-like), must provide:
//            std::shared_ptr<Acc> merge(const std::shared_ptr<Acc>& other) const;
//         merge should return a new accumulator that semantically combines both.
//         Pointer identity is used for fast equality where possible.
// - Immutable nodes with shared_ptr and cached _max_depth
// - Children layout mirrors Python: value -> depth -> child node
//   This allows merging and de-duplication by depth class.
// - Operations:
//     from_stacks, to_stacks, push, pop, popn, isolate, isolate_many,
//     apply, prune, apply_and_prune, merge, merge_many, peek, reduce_acc,
//     is_empty
//
// Note: To keep the template header-only, we keep everything here.
//

namespace leveled_gss {

template <typename T, typename Acc>
struct Upper;
template <typename T, typename Acc>
struct UpperBranch;
template <typename T, typename Acc>
struct Interface;
template <typename T>
struct Lower;

template <typename T, typename Acc>
using UpperPtr = std::shared_ptr<Upper<T, Acc>>;
template <typename T, typename Acc>
using UpperBranchPtr = std::shared_ptr<UpperBranch<T, Acc>>;
template <typename T, typename Acc>
using InterfacePtr = std::shared_ptr<Interface<T, Acc>>;
template <typename T>
using LowerPtr = std::shared_ptr<Lower<T>>;

template <typename T, typename Acc>
using UpperChildren = std::unordered_map<T, std::unordered_map<int, UpperPtr<T, Acc>>>;
template <typename T>
using LowerChildren = std::unordered_map<T, std::unordered_map<int, LowerPtr<T>>>;

template <typename T, typename Acc>
struct Upper {
    virtual ~Upper() = default;
    int _max_depth{0};
    virtual bool is_interface() const = 0;
    virtual UpperChildren<T, Acc>& children_mut() = 0;
    virtual const UpperChildren<T, Acc>& children() const = 0;
    // For Interface nodes, this returns nullptr; for UpperBranch nodes, may hold Optional Acc
    virtual std::shared_ptr<Acc> empty_acc() const = 0;
};

template <typename T, typename Acc>
struct UpperBranch : public Upper<T, Acc> {
    std::shared_ptr<UpperChildren<T, Acc>> _children;
    std::shared_ptr<Acc> empty; // nullptr means None

    UpperBranch(UpperChildren<T, Acc> children, std::shared_ptr<Acc> empty_acc)
        : _children(std::make_shared<UpperChildren<T, Acc>>(std::move(children))), empty(std::move(empty_acc)) {
        compute_max_depth();
    }

    UpperBranch(std::shared_ptr<UpperChildren<T, Acc>> children_ptr, std::shared_ptr<Acc> empty_acc)
        : _children(std::move(children_ptr)), empty(std::move(empty_acc)) {
        compute_max_depth();
    }

    void compute_max_depth() {
        // Compute _max_depth
        int depth = 0;
        for (auto &kv : *_children) {
            for (auto &dkv : kv.second) {
                int cd = dkv.second->_max_depth;
                if (cd + 1 > depth) depth = cd + 1;
            }
        }
        this->_max_depth = depth;
    }

    bool is_interface() const override { return false; }
    // This method is not used in a way that requires modification, but for correctness:
    UpperChildren<T, Acc>& children_mut() override { return *_children; }
    const UpperChildren<T, Acc>& children() const override { return *_children; }
    std::shared_ptr<Acc> empty_acc() const override { return empty; }
};

template <typename T>
struct Lower {
    std::shared_ptr<LowerChildren<T>> _children;
    bool empty{false};
    int _max_depth{0};

    Lower(LowerChildren<T> children, bool empty_flag)
        : _children(std::make_shared<LowerChildren<T>>(std::move(children))), empty(empty_flag) {
        compute_max_depth();
    }

    Lower(std::shared_ptr<LowerChildren<T>> children_ptr, bool empty_flag)
        : _children(std::move(children_ptr)), empty(empty_flag) {
        compute_max_depth();
    }

    void compute_max_depth() {
        int depth = 0;
        for (auto &kv : *_children) {
            for (auto &dkv : kv.second) {
                int cd = dkv.second->_max_depth;
                if (cd + 1 > depth) depth = cd + 1;
            }
        }
        _max_depth = depth;
    }
};

template <typename T, typename Acc>
struct Interface : public Upper<T, Acc> {
    std::shared_ptr<LowerChildren<T>> _children;
    std::shared_ptr<Acc> acc;
    std::shared_ptr<Acc> empty; // nullptr means None

    Interface(LowerChildren<T> children, std::shared_ptr<Acc> a, std::shared_ptr<Acc> e)
        : _children(std::make_shared<LowerChildren<T>>(std::move(children))), acc(std::move(a)), empty(std::move(e)) {
        compute_max_depth();
    }

    Interface(std::shared_ptr<LowerChildren<T>> children_ptr, std::shared_ptr<Acc> a, std::shared_ptr<Acc> e)
        : _children(std::move(children_ptr)), acc(std::move(a)), empty(std::move(e)) {
        compute_max_depth();
    }

    void compute_max_depth() {
        int depth = 0;
        for (auto &kv : *_children) {
            for (auto &dkv : kv.second) {
                int cd = dkv.second->_max_depth;
                if (cd + 1 > depth) depth = cd + 1;
            }
        }
        this->_max_depth = depth;
    }

    bool is_interface() const override { return true; }

    // For interfaces, children() must adapt: these are Lower children only for traversal
    UpperChildren<T, Acc>& children_mut() override {
        // Not meaningful for Interface; we provide a static empty ref to satisfy interface (unused).
        static UpperChildren<T, Acc> dummy;
        return dummy;
    }
    const UpperChildren<T, Acc>& children() const override {
        static UpperChildren<T, Acc> dummy;
        return dummy;
    }

    std::shared_ptr<Acc> empty_acc() const override { return empty; }
};

// Utilities

template <typename Acc>
static inline std::shared_ptr<Acc>
_merge_optional_acc(const std::shared_ptr<Acc>& a, const std::shared_ptr<Acc>& b) {
    if (!a) return b;
    if (!b) return a;
    if (a.get() == b.get()) return a;
    return a->merge(b);
}

template <typename Acc>
static inline std::shared_ptr<Acc>
_merge_acc(const std::shared_ptr<Acc>& a, const std::shared_ptr<Acc>& b) {
    if (a.get() == b.get()) return a;
    return a->merge(b);
}

template <typename NodePtr, typename MergeFunc, typename TKey>
static inline std::unordered_map<TKey, std::unordered_map<int, NodePtr>>
_merge_children_by_depth(
    const std::unordered_map<TKey, std::unordered_map<int, NodePtr>>& c1,
    const std::unordered_map<TKey, std::unordered_map<int, NodePtr>>& c2,
    MergeFunc merge_func
) {
    if (&c1 == &c2) {
        return c1;
    }
    std::unordered_map<TKey, std::unordered_map<int, NodePtr>> merged_children;
    std::unordered_set<TKey> all_keys;
    all_keys.reserve(c1.size() + c2.size());
    for (auto &kv : c1) all_keys.insert(kv.first);
    for (auto &kv : c2) all_keys.insert(kv.first);

    for (auto &v : all_keys) {
        std::unordered_map<int, std::vector<NodePtr>> nodes_by_depth;
        auto it1 = c1.find(v);
        if (it1 != c1.end()) {
            for (auto &dkv : it1->second) nodes_by_depth[dkv.first].push_back(dkv.second);
        }
        auto it2 = c2.find(v);
        if (it2 != c2.end()) {
            for (auto &dkv : it2->second) nodes_by_depth[dkv.first].push_back(dkv.second);
        }
        if (nodes_by_depth.empty()) continue;

        std::unordered_map<int, NodePtr> v_out;
        v_out.reserve(nodes_by_depth.size());
        for (auto &dv : nodes_by_depth) {
            auto &vec = dv.second;
            NodePtr merged = vec[0];
            for (size_t i = 1; i < vec.size(); ++i) {
                merged = merge_func(merged, vec[i]);
            }
            v_out[merged->_max_depth] = merged;
        }
        merged_children[v] = std::move(v_out);
    }
    return merged_children;
}

// Forward declarations
template <typename T, typename Acc>
static UpperPtr<T, Acc> merge_upper(const UpperPtr<T, Acc>& u1, const UpperPtr<T, Acc>& u2);
template <typename T, typename Acc>
static UpperPtr<T, Acc> try_promote(const UpperBranchPtr<T, Acc>& node);
template <typename T, typename Acc>
static LowerPtr<T> merge_lower(const LowerPtr<T>& l1, const LowerPtr<T>& l2);

// Convert Interface to UpperBranch (used in mixed merges)
template <typename T, typename Acc>
static UpperBranchPtr<T, Acc> interface_to_upperbranch(const InterfacePtr<T, Acc>& it) {
    auto children = std::make_shared<UpperChildren<T, Acc>>();
    for (auto &kv : *it->_children) {
        const T& v = kv.first;
        const auto& kids = kv.second;
        std::unordered_map<int, UpperPtr<T, Acc>> v_map;
        for (auto &dkv : kids) {
            auto lchild = dkv.second;
            auto ci = std::make_shared<Interface<T, Acc>>(
                lchild->_children,
                it->acc,
                lchild->empty ? it->acc : std::shared_ptr<Acc>(nullptr)
            );
            v_map[ci->_max_depth] = ci;
        }
        if (!v_map.empty()) {
            (*children)[v] = std::move(v_map);
        }
    }
    std::shared_ptr<Acc> new_empty = it->empty;
    if (it->_children->empty() && !new_empty) {
        new_empty = it->acc;
    }
    return std::make_shared<UpperBranch<T, Acc>>(children, new_empty);
}

// Try to promote an UpperBranch to Interface when possible
template <typename T, typename Acc>
static UpperPtr<T, Acc> try_promote(const UpperBranchPtr<T, Acc>& node) {
    // Collect all children
    std::vector<UpperPtr<T, Acc>> all_children;
    for (auto &kv : *node->_children) {
        for (auto &dkv : kv.second) {
            all_children.push_back(dkv.second);
        }
    }
    if (all_children.empty()) {
        if (node->empty) {
            // Canonical leaf: represent as Interface with no children, acc=node.empty
            return std::make_shared<Interface<T, Acc>>(LowerChildren<T>{}, node->empty, node->empty);
        }
        return node;
    }
    // If any child is UpperBranch, cannot promote
    for (auto &c : all_children) {
        if (!c->is_interface()) return node;
    }
    // Acc set across children + node.empty
    std::unordered_set<const Acc*> accs;
    if (node->empty) accs.insert(node->empty.get());
    for (auto &c : all_children) {
        auto ci = std::static_pointer_cast<Interface<T, Acc>>(c);
        accs.insert(ci->acc.get());
        if (ci->empty) accs.insert(ci->empty.get());
    }

    if (accs.size() <= 1) {
        // Determine the unique acc (if any)
        std::shared_ptr<Acc> the_acc;
        if (!accs.empty()) {
            // pick from children (some might be null, but if size <=1 and not empty, it's non-null)
            if (node->empty) the_acc = node->empty;
            else {
                auto ci = std::static_pointer_cast<Interface<T, Acc>>(all_children[0]);
                the_acc = ci->acc ? ci->acc : ci->empty;
            }
        } else {
            // Truly empty GSS
            return std::make_shared<UpperBranch<T, Acc>>(UpperChildren<T, Acc>{}, std::shared_ptr<Acc>(nullptr));
        }

        // Build Lower children by converting each Interface child
        auto l_children = std::make_shared<LowerChildren<T>>();
        for (auto &kv : *node->_children) {
            const T& v = kv.first;
            const auto& kids = kv.second;
            std::unordered_map<int, LowerPtr<T>> v_map;
            for (auto &dkv : kids) {
                auto ci = std::static_pointer_cast<Interface<T, Acc>>(dkv.second);
                auto lower = std::make_shared<Lower<T>>(ci->_children, (ci->empty != nullptr));
                v_map[lower->_max_depth] = lower;
            }
            if (!v_map.empty()) {
                (*l_children)[v] = std::move(v_map);
            }
        }
        return std::make_shared<Interface<T, Acc>>(std::move(l_children), the_acc, node->empty);
    }
    return node;
}

// Merge upper branches
template <typename T, typename Acc>
static UpperPtr<T, Acc> merge_upperbranches(const UpperBranchPtr<T, Acc>& a, const UpperBranchPtr<T, Acc>& b) {
    if (a.get() == b.get()) return a;
    auto new_empty = _merge_optional_acc<Acc>(a->empty, b->empty);
    auto merged_children = _merge_children_by_depth<UpperPtr<T, Acc>, std::function<UpperPtr<T, Acc>(UpperPtr<T, Acc>, UpperPtr<T, Acc>)>, T>(
        *a->_children, *b->_children,
        [](UpperPtr<T, Acc> n1, UpperPtr<T, Acc> n2) { return merge_upper<T, Acc>(n1, n2); }
    );

    bool a_unchanged = (new_empty.get() == a->empty.get()) && (merged_children == *a->_children);
    if (a_unchanged) return a;

    bool b_unchanged = (new_empty.get() == b->empty.get()) && (merged_children == *b->_children);
    if (b_unchanged) return b;

    auto ub = std::make_shared<UpperBranch<T, Acc>>(std::move(merged_children), new_empty);
    return try_promote<T, Acc>(ub);
}

// Merge interfaces; if same Acc pointer, merge Lower children only; otherwise convert and merge as branches
template <typename T, typename Acc>
static UpperPtr<T, Acc> merge_interfaces(const InterfacePtr<T, Acc>& a, const InterfacePtr<T, Acc>& b) {
    if (a.get() == b.get()) return a;

    // Prefer cheap pointer checks; defer expensive content equality to last.
    if (a->acc.get() == b->acc.get() ||
        a->_children.get() == b->_children.get() ||
        (*(a->acc) == *(b->acc))) {
        auto merged_children = _merge_children_by_depth<LowerPtr<T>, std::function<LowerPtr<T>(LowerPtr<T>, LowerPtr<T>)>, T>(
            *a->_children, *b->_children,
            [](LowerPtr<T> l1, LowerPtr<T> l2) { return merge_lower<T>(l1, l2); }
        );
        auto new_acc = _merge_acc<Acc>(a->acc, b->acc);
        auto new_empty = _merge_optional_acc<Acc>(a->empty, b->empty);

        bool a_unchanged = (new_acc.get() == a->acc.get()) &&
                           (new_empty.get() == a->empty.get()) &&
                           (merged_children == *a->_children);
        if (a_unchanged) return a;

        bool b_unchanged = (new_acc.get() == b->acc.get()) &&
                           (new_empty.get() == b->empty.get()) &&
                           (merged_children == *b->_children);
        if (b_unchanged) return b;

        return std::make_shared<Interface<T, Acc>>(merged_children, new_acc, new_empty);
    }
    auto ub1 = interface_to_upperbranch<T, Acc>(a);
    auto ub2 = interface_to_upperbranch<T, Acc>(b);
    return merge_upperbranches<T, Acc>(ub1, ub2);
}

// Merge upper nodes (polymorphic)
template <typename T, typename Acc>
static UpperPtr<T, Acc> merge_upper(const UpperPtr<T, Acc>& u1, const UpperPtr<T, Acc>& u2) {
    if (u1.get() == u2.get()) return u1;
    if (u1->is_interface() && u2->is_interface()) {
        return merge_interfaces<T, Acc>(std::static_pointer_cast<Interface<T, Acc>>(u1),
                                        std::static_pointer_cast<Interface<T, Acc>>(u2));
    }
    if (!u1->is_interface() && !u2->is_interface()) {
        return merge_upperbranches<T, Acc>(std::static_pointer_cast<UpperBranch<T, Acc>>(u1),
                                           std::static_pointer_cast<UpperBranch<T, Acc>>(u2));
    }
    auto ub1 = u1->is_interface() ? interface_to_upperbranch<T, Acc>(std::static_pointer_cast<Interface<T, Acc>>(u1))
                                  : std::static_pointer_cast<UpperBranch<T, Acc>>(u1);
    auto ub2 = u2->is_interface() ? interface_to_upperbranch<T, Acc>(std::static_pointer_cast<Interface<T, Acc>>(u2))
                                  : std::static_pointer_cast<UpperBranch<T, Acc>>(u2);
    return merge_upperbranches<T, Acc>(ub1, ub2);
}

// Merge lower nodes
template <typename T>
static LowerPtr<T> merge_lower(const LowerPtr<T>& l1, const LowerPtr<T>& l2) {
    if (l1.get() == l2.get()) return l1;
    bool new_empty = l1->empty || l2->empty;
    auto merged_children = _merge_children_by_depth<LowerPtr<T>, std::function<LowerPtr<T>(LowerPtr<T>, LowerPtr<T>)>, T>(
        *l1->_children, *l2->_children,
        [](LowerPtr<T> a, LowerPtr<T> b) { return merge_lower<T>(a, b); }
    );

    if (new_empty == l1->empty && merged_children == *l1->_children) return l1;
    if (new_empty == l2->empty && merged_children == *l2->_children) return l2;

    return std::make_shared<Lower<T>>(std::move(merged_children), new_empty);
}

// Convert lower subtree to upper subtree (with given acc)
template <typename T, typename Acc>
static UpperPtr<T, Acc> lower_to_upper(const LowerPtr<T>& l, const std::shared_ptr<Acc>& acc) {
    auto children = std::make_shared<UpperChildren<T, Acc>>();
    for (auto &kv : *l->_children) {
        const T& v = kv.first;
        const auto& kids = kv.second;
        std::unordered_map<int, UpperPtr<T, Acc>> v_map;
        for (auto &dkv : kids) {
            auto up_child = lower_to_upper<T, Acc>(dkv.second, acc);
            v_map[up_child->_max_depth] = up_child;
        }
        if (!v_map.empty()) {
            (*children)[v] = std::move(v_map);
        }
    }
    auto ub = std::make_shared<UpperBranch<T, Acc>>(std::move(children), l->empty ? acc : std::shared_ptr<Acc>(nullptr));
    return try_promote<T, Acc>(ub);
}


// LeveledGSS main type
template <typename T, typename Acc>
class LeveledGSS {
public:
    UpperPtr<T, Acc> inner;

    LeveledGSS() : inner(std::make_shared<UpperBranch<T, Acc>>(UpperChildren<T, Acc>{}, std::shared_ptr<Acc>(nullptr))) {}
    explicit LeveledGSS(const UpperPtr<T, Acc>& u) : inner(u) {}
    explicit LeveledGSS(UpperPtr<T, Acc>&& u) : inner(std::move(u)) {}

    // Rule of 5
    LeveledGSS(const LeveledGSS& other) = default;
    LeveledGSS(LeveledGSS&& other) noexcept = default;
    LeveledGSS& operator=(const LeveledGSS& other) = default;
    LeveledGSS& operator=(LeveledGSS&& other) noexcept = default;
    ~LeveledGSS() = default;

    static LeveledGSS from_stacks(const std::vector<std::pair<std::vector<T>, std::shared_ptr<Acc>>>& stacks) {
        // Canonicalize stacks: merge Acc for identical vectors
        struct VecHash {
            size_t operator()(const std::vector<T>& v) const noexcept {
                size_t h = 1469598103934665603ULL;
                for (auto &x : v) {
                    h ^= static_cast<size_t>(x) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
                }
                return h;
            }
        };
        struct VecEq {
            bool operator()(const std::vector<T>& a, const std::vector<T>& b) const noexcept {
                return a == b;
            }
        };

        std::unordered_map<std::vector<T>, std::shared_ptr<Acc>, VecHash, VecEq> merged;
        merged.reserve(stacks.size() * 2 + 1);

        for (auto &p : stacks) {
            const auto& vals = p.first;
            auto acc = p.second;
            auto it = merged.find(vals);
            if (it == merged.end()) {
                merged.emplace(vals, acc);
            } else {
                if (it->second.get() != acc.get()) {
                    it->second = it->second->merge(acc);
                }
            }
        }

        // Build trie in reversed order
        std::shared_ptr<Acc> empty_acc;
        // trie: val -> {"end": Acc*, "sub": subtrie}
        struct TrieNode {
            std::shared_ptr<Acc> end;
            std::unordered_map<T, std::shared_ptr<TrieNode>> sub;
        };
        auto root = std::make_shared<TrieNode>();

        for (auto &kv : merged) {
            const auto& vals = kv.first;
            auto acc = kv.second;
            if (vals.empty()) {
                empty_acc = acc;
                continue;
            }
            auto node = root;
            for (size_t i = 0; i < vals.size(); ++i) {
                T v = vals[vals.size() - 1 - i];
                auto &entry = node->sub[v];
                if (!entry) entry = std::make_shared<TrieNode>();
                if (i == vals.size() - 1) {
                    entry->end = acc;
                } else {
                    node = entry;
                }
            }
        }

        // Recursive builder from trie
        std::function<UpperPtr<T, Acc>(std::shared_ptr<TrieNode>, std::shared_ptr<Acc>)> build;
        build = [&](std::shared_ptr<TrieNode> d, std::shared_ptr<Acc> root_empty) -> UpperPtr<T, Acc> {
            UpperChildren<T, Acc> children;
            std::vector<UpperPtr<T, Acc>> all_child_nodes;

            for (auto &kv : d->sub) {
                const T& v = kv.first;
                auto e = kv.second;
                std::vector<UpperPtr<T, Acc>> nodes_for_v;
                auto end_acc = e->end;
                if (end_acc) {
                    auto leaf = std::make_shared<UpperBranch<T, Acc>>(UpperChildren<T, Acc>{}, end_acc);
                    nodes_for_v.push_back(try_promote<T, Acc>(leaf));
                }
                if (!e->sub.empty()) {
                    nodes_for_v.push_back(build(e, std::shared_ptr<Acc>(nullptr)));
                }
                if (!nodes_for_v.empty()) {
                    std::unordered_map<int, UpperPtr<T, Acc>> m;
                    for (auto &n : nodes_for_v) m[n->_max_depth] = n;
                    children[v] = std::move(m);
                    all_child_nodes.insert(all_child_nodes.end(), nodes_for_v.begin(), nodes_for_v.end());
                }
            }

            // Attempt promotion if all children are Interfaces with single acc
            bool all_intf = !all_child_nodes.empty();
            for (auto &c : all_child_nodes) {
                if (!c->is_interface()) { all_intf = false; break; }
            }
            if (all_intf) {
                std::unordered_set<const Acc*> accs;
                if (root_empty) accs.insert(root_empty.get());
                for (auto &c : all_child_nodes) {
                    auto ci = std::static_pointer_cast<Interface<T, Acc>>(c);
                    accs.insert(ci->acc.get());
                    if (ci->empty) accs.insert(ci->empty.get());
                }
                if (accs.size() <= 1) {
                    std::shared_ptr<Acc> the_acc;
                    if (!accs.empty()) {
                        if (root_empty) the_acc = root_empty;
                        else {
                            auto ci = std::static_pointer_cast<Interface<T, Acc>>(all_child_nodes[0]);
                            the_acc = ci->acc ? ci->acc : ci->empty;
                        }
                    } else {
                        return std::make_shared<UpperBranch<T, Acc>>(UpperChildren<T, Acc>{}, std::shared_ptr<Acc>(nullptr));
                    }

                    std::function<LowerPtr<T>(std::shared_ptr<TrieNode>)> build_lower;
                    build_lower = [&](std::shared_ptr<TrieNode> nd) -> LowerPtr<T> {
                        LowerChildren<T> l_children;
                        for (auto &kv2 : nd->sub) {
                            const T& v2 = kv2.first;
                            auto e2 = kv2.second;
                            auto sub_lower = !e2->sub.empty() ? build_lower(e2)
                                                              : std::make_shared<Lower<T>>(LowerChildren<T>{}, false);
                            auto node_for_v = std::make_shared<Lower<T>>(sub_lower->_children, (bool)e2->end);
                            std::unordered_map<int, LowerPtr<T>> m;
                            m[node_for_v->_max_depth] = node_for_v;
                            l_children[v2] = std::move(m);
                        }
                        return std::make_shared<Lower<T>>(std::move(l_children), false);
                    };

                    auto lower_tree = build_lower(d);
                    return std::make_shared<Interface<T, Acc>>(std::move(lower_tree->_children), the_acc, root_empty);
                }
            }

            return std::make_shared<UpperBranch<T, Acc>>(std::move(children), root_empty);
        };

        auto root_upper = build(root, empty_acc);
        return LeveledGSS(root_upper);
    }

    std::vector<std::pair<std::vector<T>, std::shared_ptr<Acc>>> to_stacks() const {
        std::vector<std::pair<std::vector<T>, std::shared_ptr<Acc>>> res;

        std::function<void(const LowerPtr<T>&, std::vector<T>&, const std::shared_ptr<Acc>&)> dfs_lower =
            [&](const LowerPtr<T>& l, std::vector<T>& pref, const std::shared_ptr<Acc>& acc) {
                if (l->empty) {
                    std::vector<T> out(pref.rbegin(), pref.rend());
                    res.emplace_back(std::move(out), acc);
                }
                for (auto &kv : *l->_children) {
                    T v = kv.first;
                    for (auto &dkv : kv.second) {
                        pref.push_back(v);
                        dfs_lower(dkv.second, pref, acc);
                        pref.pop_back();
                    }
                }
            };

        std::function<void(const UpperPtr<T, Acc>&, std::vector<T>&)> dfs_upper =
            [&](const UpperPtr<T, Acc>& u, std::vector<T>& pref) {
                if (!u->is_interface()) {
                    auto ub = std::static_pointer_cast<UpperBranch<T, Acc>>(u);
                    if (ub->empty) {
                        std::vector<T> out(pref.rbegin(), pref.rend());
                        res.emplace_back(std::move(out), ub->empty);
                    }
                    for (auto &kv : *ub->_children) {
                        T v = kv.first;
                        for (auto &dkv : kv.second) {
                            pref.push_back(v);
                            dfs_upper(dkv.second, pref);
                            pref.pop_back();
                        }
                    }
                } else {
                    auto it = std::static_pointer_cast<Interface<T, Acc>>(u);
                    if (it->empty) {
                        std::vector<T> out(pref.rbegin(), pref.rend());
                        res.emplace_back(std::move(out), it->empty);
                    }
                    if (it->_children->empty()) {
                        if (!it->empty) {
                            std::vector<T> out(pref.rbegin(), pref.rend());
                            res.emplace_back(std::move(out), it->acc);
                        }
                    } else {
                        for (auto &kv : *it->_children) {
                            T v = kv.first;
                            for (auto &dkv : kv.second) {
                                pref.push_back(v);
                                dfs_lower(dkv.second, pref, it->acc);
                                pref.pop_back();
                            }
                        }
                    }
                }
            };

        std::vector<T> empty;
        dfs_upper(inner, empty);
        // No sorting here; caller can canonicalize if desired.
        return res;
    }

    LeveledGSS push(const T& value) const {
        if (is_empty()) return *this;
        if (inner->is_interface()) {
            auto it = std::static_pointer_cast<Interface<T, Acc>>(inner);
            auto lower_node = std::make_shared<Lower<T>>(it->_children, (it->empty != nullptr));
            std::unordered_map<int, LowerPtr<T>> m;
            m[lower_node->_max_depth] = lower_node;
            auto new_lc = std::make_shared<LowerChildren<T>>();
            (*new_lc)[value] = std::move(m);
            auto new_interface = std::make_shared<Interface<T, Acc>>(std::move(new_lc), it->acc, std::shared_ptr<Acc>(nullptr));
            return LeveledGSS(new_interface);
        } else {
            auto ub = std::static_pointer_cast<UpperBranch<T, Acc>>(inner);
            auto ch = std::make_shared<UpperChildren<T, Acc>>();
            std::unordered_map<int, UpperPtr<T, Acc>> m;
            m[ub->_max_depth] = inner;
            (*ch)[value] = std::move(m);
            return LeveledGSS(std::make_shared<UpperBranch<T, Acc>>(std::move(ch), std::shared_ptr<Acc>(nullptr)));
        }
    }

    LeveledGSS pop() const {
        if (inner->is_interface()) {
            auto it = std::static_pointer_cast<Interface<T, Acc>>(inner);
            // Merge all lower children
            LowerPtr<T> merged;
            bool first = true;
            for (auto &kv : *it->_children) {
                for (auto &dkv : kv.second) {
                    if (first) { merged = dkv.second; first = false; }
                    else { merged = merge_lower<T>(merged, dkv.second); }
                }
            }
            if (!merged) {
                auto ub = std::make_shared<UpperBranch<T, Acc>>(UpperChildren<T, Acc>{}, std::shared_ptr<Acc>(nullptr));
                return LeveledGSS(ub);
            }
            auto merged_empty = merged->empty ? it->acc : std::shared_ptr<Acc>(nullptr);
            if (!merged_empty && merged->_children->empty()) {
                return LeveledGSS(std::make_shared<UpperBranch<T, Acc>>(UpperChildren<T, Acc>{}, merged_empty));
            } else {
                return LeveledGSS(std::make_shared<Interface<T, Acc>>(merged->_children, it->acc, merged_empty));
            }
        } else {
            auto ub = std::static_pointer_cast<UpperBranch<T, Acc>>(inner);
            UpperPtr<T, Acc> merged;
            bool first = true;
            for (auto &kv : *ub->_children) {
                for (auto &dkv : kv.second) {
                    if (first) { merged = dkv.second; first = false; }
                    else { merged = merge_upper<T, Acc>(merged, dkv.second); }
                }
            }
            if (!merged) merged = std::make_shared<UpperBranch<T, Acc>>(UpperChildren<T, Acc>{}, std::shared_ptr<Acc>(nullptr));
            UpperPtr<T, Acc> result_node = merged;
            if (!result_node->is_interface()) {
                result_node = try_promote<T, Acc>(std::static_pointer_cast<UpperBranch<T, Acc>>(result_node));
            }
            return LeveledGSS(result_node);
        }
    }

    LeveledGSS popn(int n) const {
        if (n <= 0) return *this;
        if (is_empty()) return *this;

        struct Key {
            std::uintptr_t ptr;
            int k;
            bool operator==(const Key& o) const noexcept { return ptr == o.ptr && k == o.k; }
        };
        struct KeyHash {
            size_t operator()(const Key& k) const noexcept {
                return std::hash<std::uintptr_t>()(k.ptr) ^ (k.k * 0x9e3779b97f4a7c15ULL);
            }
        };

        std::unordered_map<Key, UpperPtr<T, Acc>, KeyHash> memo_upper;
        std::unordered_map<Key, LowerPtr<T>, KeyHash> memo_lower;

        std::function<LowerPtr<T>(const LowerPtr<T>&, int)> _popn_lower =
            [&](const LowerPtr<T>& node, int k) -> LowerPtr<T> {
                if (k == 0) return node;
                Key key{reinterpret_cast<std::uintptr_t>(node.get()), k};
                auto itm = memo_lower.find(key);
                if (itm != memo_lower.end()) return itm->second;

                // If no children, popping leaves empty
                std::vector<LowerPtr<T>> all_children;
                for (auto &kv : *node->_children) for (auto &dkv : kv.second) all_children.push_back(dkv.second);
                if (all_children.empty()) {
                    auto res = std::make_shared<Lower<T>>(LowerChildren<T>{}, false);
                    memo_lower[key] = res;
                    return res;
                }

                std::vector<LowerPtr<T>> popped_children;
                popped_children.reserve(all_children.size());
                for (auto &child : all_children) popped_children.push_back(_popn_lower(child, k - 1));
                LowerPtr<T> res = popped_children[0];
                for (size_t i = 1; i < popped_children.size(); ++i) res = merge_lower<T>(res, popped_children[i]);
                memo_lower[key] = res;
                return res;
            };

        std::function<UpperPtr<T, Acc>(const UpperPtr<T, Acc>&, int)> _popn_upper =
            [&](const UpperPtr<T, Acc>& node, int k) -> UpperPtr<T, Acc> {
                if (k == 0) return node;
                Key key{reinterpret_cast<std::uintptr_t>(node.get()), k};
                auto itm = memo_upper.find(key);
                if (itm != memo_upper.end()) return itm->second;

                UpperPtr<T, Acc> res;
                if (node->is_interface()) {
                    auto it = std::static_pointer_cast<Interface<T, Acc>>(node);
                    // pop from lowers
                    std::vector<LowerPtr<T>> all_lower_children;
                    for (auto &kv : *it->_children) for (auto &dkv : kv.second) all_lower_children.push_back(dkv.second);

                    if (all_lower_children.empty()) {
                        res = std::make_shared<UpperBranch<T, Acc>>(UpperChildren<T, Acc>{}, std::shared_ptr<Acc>(nullptr));
                    } else {
                        std::vector<LowerPtr<T>> popped;
                        popped.reserve(all_lower_children.size());
                        for (auto &l : all_lower_children) popped.push_back(_popn_lower(l, k - 1));
                        LowerPtr<T> merged = popped[0];
                        for (size_t i = 1; i < popped.size(); ++i) merged = merge_lower<T>(merged, popped[i]);
                        auto new_empty = merged->empty ? it->acc : std::shared_ptr<Acc>(nullptr);
                        if (merged->_children->empty() && !new_empty) {
                            res = std::make_shared<UpperBranch<T, Acc>>(UpperChildren<T, Acc>{}, std::shared_ptr<Acc>(nullptr));
                        } else {
                            res = std::make_shared<Interface<T, Acc>>(merged->_children, it->acc, new_empty);
                        }
                    }
                } else {
                    auto ub = std::static_pointer_cast<UpperBranch<T, Acc>>(node);
                    std::vector<UpperPtr<T, Acc>> all_upper_children;
                    for (auto &kv : *ub->_children) for (auto &dkv : kv.second) all_upper_children.push_back(dkv.second);

                    if (all_upper_children.empty()) {
                        res = std::make_shared<UpperBranch<T, Acc>>(UpperChildren<T, Acc>{}, std::shared_ptr<Acc>(nullptr));
                    } else {
                        std::vector<UpperPtr<T, Acc>> popped;
                        popped.reserve(all_upper_children.size());
                        for (auto &u : all_upper_children) popped.push_back(_popn_upper(u, k - 1));
                        UpperPtr<T, Acc> merged = popped[0];
                        for (size_t i = 1; i < popped.size(); ++i) merged = merge_upper<T, Acc>(merged, popped[i]);
                        if (merged->is_interface()) {
                            res = merged;
                        } else {
                            res = try_promote<T, Acc>(std::static_pointer_cast<UpperBranch<T, Acc>>(merged));
                        }
                    }
                }
                memo_upper[key] = res;
                return res;
            };

        return LeveledGSS(_popn_upper(inner, n));
    }

    bool is_empty() const {
        if (!inner->is_interface()) {
            auto ub = std::static_pointer_cast<UpperBranch<T, Acc>>(inner);
            return ub->_children->empty() && !ub->empty;
        }
        return false;
    }

    LeveledGSS isolate(const T& value) const {
        if (!inner->is_interface()) {
            auto ub = std::static_pointer_cast<UpperBranch<T, Acc>>(inner);
            auto filtered = std::make_shared<UpperChildren<T, Acc>>();
            auto it = ub->_children->find(value);
            if (it != ub->_children->end()) (*filtered)[value] = it->second;
            auto res = std::make_shared<UpperBranch<T, Acc>>(std::move(filtered), std::shared_ptr<Acc>(nullptr));
            return LeveledGSS(try_promote<T, Acc>(res));
        } else {
            auto it = std::static_pointer_cast<Interface<T, Acc>>(inner);
            auto itc = it->_children->find(value);
            if (itc == it->_children->end()) {
                return LeveledGSS(std::make_shared<UpperBranch<T, Acc>>(UpperChildren<T, Acc>{}, std::shared_ptr<Acc>(nullptr)));
            } else {
                auto filtered = std::make_shared<LowerChildren<T>>();
                (*filtered)[value] = itc->second;
                auto res = std::make_shared<Interface<T, Acc>>(std::move(filtered), it->acc, std::shared_ptr<Acc>(nullptr));
                return LeveledGSS(res);
            }
        }
    }

    LeveledGSS isolate_none() const {
        // Keep only empty stacks (value == None)
        std::shared_ptr<Acc> empty_acc;
        if (!inner->is_interface()) {
            auto ub = std::static_pointer_cast<UpperBranch<T, Acc>>(inner);
            empty_acc = ub->empty;
        } else {
            auto it = std::static_pointer_cast<Interface<T, Acc>>(inner);
            empty_acc = it->empty;
        }
        auto new_root = std::make_shared<UpperBranch<T, Acc>>(UpperChildren<T, Acc>{}, empty_acc);
        return LeveledGSS(try_promote<T, Acc>(new_root));
    }

    LeveledGSS isolate_many(const std::unordered_set<T>& values) const {
        std::shared_ptr<Acc> new_empty = nullptr;
        if (!inner->is_interface()) {
            auto ub = std::static_pointer_cast<UpperBranch<T, Acc>>(inner);
            auto filtered_children = std::make_shared<UpperChildren<T, Acc>>();
            for (auto &kv : *ub->_children) {
                if (values.count(kv.first)) (*filtered_children)[kv.first] = kv.second;
            }
            auto res = std::make_shared<UpperBranch<T, Acc>>(std::move(filtered_children), new_empty);
            return LeveledGSS(try_promote<T, Acc>(res));
        } else {
            auto it = std::static_pointer_cast<Interface<T, Acc>>(inner);
            auto filtered_children = std::make_shared<LowerChildren<T>>();
            for (auto &kv : *it->_children) {
                if (values.count(kv.first)) (*filtered_children)[kv.first] = kv.second;
            }
            if (!filtered_children->empty()) {
                auto res = std::make_shared<Interface<T, Acc>>(std::move(filtered_children), it->acc, new_empty);
                return LeveledGSS(res);
            } else {
                auto res = std::make_shared<UpperBranch<T, Acc>>(UpperChildren<T, Acc>{}, new_empty);
                return LeveledGSS(try_promote<T, Acc>(res));
            }
        }
    }

    // Apply: Acc -> NewAcc (but we keep Acc type same for simplicity here)
    LeveledGSS apply(
        const std::function<std::shared_ptr<Acc>(const std::shared_ptr<Acc>&)>& func,
        std::unordered_map<std::uintptr_t, std::shared_ptr<Acc>>* acc_memo_ptr = nullptr
    ) const {
        std::unordered_map<std::uintptr_t, UpperPtr<T, Acc>> memo;

        std::unordered_map<std::uintptr_t, std::shared_ptr<Acc>> local_acc_memo;
        std::unordered_map<std::uintptr_t, std::shared_ptr<Acc>>& acc_memo =
            acc_memo_ptr ? *acc_memo_ptr : local_acc_memo;

        auto apply_func = [&](const std::shared_ptr<Acc>& a) -> std::shared_ptr<Acc> {
            if (!a) return std::shared_ptr<Acc>(nullptr);
            auto k = reinterpret_cast<std::uintptr_t>(a.get());
            auto it = acc_memo.find(k);
            if (it != acc_memo.end()) return it->second;
            auto r = func(a);
            acc_memo.emplace(k, r);
            return r;
        };

        std::function<UpperPtr<T, Acc>(const UpperPtr<T, Acc>&)> transform =
            [&](const UpperPtr<T, Acc>& node) -> UpperPtr<T, Acc> {
                auto nid = reinterpret_cast<std::uintptr_t>(node.get());
                auto itm = memo.find(nid);
                if (itm != memo.end()) return itm->second;

                if (node->is_interface()) {
                    auto itf = std::static_pointer_cast<Interface<T, Acc>>(node);
                    auto new_acc = apply_func(itf->acc);
                    auto new_empty = itf->empty ? apply_func(itf->empty) : std::shared_ptr<Acc>(nullptr);
                    if (new_acc.get() == itf->acc.get() && new_empty.get() == itf->empty.get()) {
                        memo[nid] = node;
                        return node;
                    }
                    auto res = std::make_shared<Interface<T, Acc>>(itf->_children, new_acc, new_empty);
                    memo[nid] = res;
                    return res;
                } else {
                    auto ub = std::static_pointer_cast<UpperBranch<T, Acc>>(node);
                    auto new_empty = ub->empty ? apply_func(ub->empty) : std::shared_ptr<Acc>(nullptr);
                    bool changed = (new_empty.get() != ub->empty.get());
                    auto new_children = std::make_shared<UpperChildren<T, Acc>>();
                    bool any_child_map_changed = false;

                    for (auto &kv : *ub->_children) {
                        const T& v = kv.first;
                        const auto& kids = kv.second;
                        std::unordered_map<int, UpperPtr<T, Acc>> new_kids;
                        bool child_map_changed = false;
                        for (auto &dkv : kv.second) {
                            auto new_child = transform(dkv.second);
                            if (new_child.get() != dkv.second.get()) child_map_changed = true;
                            new_kids[new_child->_max_depth] = new_child;
                        }

                        if (child_map_changed) {
                            changed = true;
                            any_child_map_changed = true;
                            if (!new_kids.empty()) (*new_children)[v] = std::move(new_kids);
                        } else {
                            (*new_children)[v] = kids;
                        }
                    }

                    if (!changed) {
                        memo[nid] = node;
                        return node;
                    }

                    // If only empty changed, preserve children pointer identity to trigger fast-path merges.
                    if (!any_child_map_changed) {
                        auto res = std::make_shared<UpperBranch<T, Acc>>(ub->_children, new_empty);
                        auto promoted = try_promote<T, Acc>(res);
                        memo[nid] = promoted;
                        return promoted;
                    }

                    auto res = std::make_shared<UpperBranch<T, Acc>>(std::move(new_children), new_empty);
                    auto promoted = try_promote<T, Acc>(res);
                    memo[nid] = promoted;
                    return promoted;
                }
            };

        return LeveledGSS(transform(inner));
    }

    LeveledGSS prune(const std::function<bool(const std::shared_ptr<Acc>&)>& predicate) const {
        std::unordered_map<std::uintptr_t, UpperPtr<T, Acc>> memo;

        std::function<UpperPtr<T, Acc>(const UpperPtr<T, Acc>&)> transform =
            [&](const UpperPtr<T, Acc>& node) -> UpperPtr<T, Acc> {
                auto nid = reinterpret_cast<std::uintptr_t>(node.get());
                auto itm = memo.find(nid);
                if (itm != memo.end()) return itm->second;

                if (node->is_interface()) {
                    auto itf = std::static_pointer_cast<Interface<T, Acc>>(node);
                    bool keep_acc = predicate(itf->acc);
                    bool keep_empty = (itf->empty && predicate(itf->empty));
                    auto new_empty = keep_empty ? itf->empty : std::shared_ptr<Acc>(nullptr);

                    // Fast path: if nothing changed, return original node to preserve sharing.
                    if (keep_acc && new_empty.get() == itf->empty.get()) {
                        memo[nid] = node;
                        return node;
                    }

                    if (!keep_acc && !keep_empty) {
                        memo[nid] = nullptr;
                        return nullptr;
                    }
                    if (!keep_acc && keep_empty) {
                        auto res = std::make_shared<UpperBranch<T, Acc>>(UpperChildren<T, Acc>{}, new_empty);
                        auto promoted = try_promote<T, Acc>(res);
                        memo[nid] = promoted;
                        return promoted;
                    }
                    // keep_acc
                    auto res = std::make_shared<Interface<T, Acc>>(itf->_children, itf->acc, new_empty);
                    memo[nid] = res;
                    return res;
                } else {
                    auto ub = std::static_pointer_cast<UpperBranch<T, Acc>>(node);
                    auto new_empty = (ub->empty && predicate(ub->empty)) ? ub->empty : std::shared_ptr<Acc>(nullptr);
                    bool changed = (new_empty.get() != ub->empty.get());
                    auto new_children = std::make_shared<UpperChildren<T, Acc>>();
                    bool any_child_map_changed = false;

                    for (auto &kv : *ub->_children) {
                        const T& v = kv.first;
                        const auto& kids = kv.second;
                        std::unordered_map<int, UpperPtr<T, Acc>> new_kids;
                        bool child_map_changed = false;
                        for (auto &dkv : kids) {
                            auto new_child = transform(dkv.second);
                            if (new_child.get() != dkv.second.get()) child_map_changed = true;
                            if (new_child) new_kids[new_child->_max_depth] = new_child;
                        }
                        if (new_kids.size() != kids.size()) child_map_changed = true;

                        if (child_map_changed) {
                            changed = true;
                            any_child_map_changed = true;
                            if (!new_kids.empty()) (*new_children)[v] = std::move(new_kids);
                        } else {
                            (*new_children)[v] = kids;
                        }
                    }

                    if (!changed) {
                        memo[nid] = node;
                        return node;
                    }
                    if (!any_child_map_changed) {
                        // Only empty changed; reuse children pointer to preserve identity.
                        // If children are empty too and new_empty was removed, node is pruned.
                        bool children_empty = ub->_children->empty();
                        if (children_empty && !new_empty) {
                            memo[nid] = nullptr;
                            return nullptr;
                        }
                        auto res = std::make_shared<UpperBranch<T, Acc>>(ub->_children, new_empty);
                        auto promoted = try_promote<T, Acc>(res);
                        memo[nid] = promoted;
                        return promoted;
                    }
                    if (new_children->empty() && !new_empty) {
                        memo[nid] = nullptr;
                        return nullptr;
                    }
                    auto res = std::make_shared<UpperBranch<T, Acc>>(std::move(new_children), new_empty);
                    auto promoted = try_promote<T, Acc>(res);
                    memo[nid] = promoted;
                    return promoted;
                }
            };

        auto res_inner = transform(inner);
        if (!res_inner) {
            return LeveledGSS(std::make_shared<UpperBranch<T, Acc>>(UpperChildren<T, Acc>{}, std::shared_ptr<Acc>(nullptr)));
        }
        return LeveledGSS(res_inner);
    }

    LeveledGSS apply_and_prune(
        const std::function<std::shared_ptr<Acc>(const std::shared_ptr<Acc>&)>& mutator,
        std::unordered_map<std::uintptr_t, std::shared_ptr<Acc>>* acc_cache_ptr = nullptr
    ) const {
        // mutator returns nullptr to prune stacks carrying acc; otherwise updated acc
        std::unordered_map<std::uintptr_t, UpperPtr<T, Acc>> memo;

        std::unordered_map<std::uintptr_t, std::shared_ptr<Acc>> local_acc_cache;
        std::unordered_map<std::uintptr_t, std::shared_ptr<Acc>>& acc_cache =
            acc_cache_ptr ? *acc_cache_ptr : local_acc_cache;

        auto mutate_acc = [&](const std::shared_ptr<Acc>& a) -> std::shared_ptr<Acc> {
            if (!a) return std::shared_ptr<Acc>(nullptr);
            auto k = reinterpret_cast<std::uintptr_t>(a.get());
            auto it = acc_cache.find(k);
            if (it != acc_cache.end()) return it->second;
            auto r = mutator(a);
            acc_cache.emplace(k, r);
            return r;
        };

        std::function<UpperPtr<T, Acc>(const UpperPtr<T, Acc>&)> transform =
            [&](const UpperPtr<T, Acc>& node) -> UpperPtr<T, Acc> {
                auto nid = reinterpret_cast<std::uintptr_t>(node.get());
                auto itm = memo.find(nid);
                if (itm != memo.end()) return itm->second;

                if (node->is_interface()) {
                    auto itf = std::static_pointer_cast<Interface<T, Acc>>(node);
                    auto new_acc = mutate_acc(itf->acc);
                    auto new_empty = itf->empty ? mutate_acc(itf->empty) : std::shared_ptr<Acc>(nullptr);

                    bool keep_acc = (new_acc != nullptr);
                    bool keep_empty = (new_empty != nullptr);

                    if (!keep_acc && !keep_empty) {
                        memo[nid] = nullptr;
                        return nullptr;
                    }
                    if (!keep_acc && keep_empty) {
                        auto res = std::make_shared<UpperBranch<T, Acc>>(UpperChildren<T, Acc>{}, new_empty);
                        auto promoted = try_promote<T, Acc>(res);
                        memo[nid] = promoted;
                        return promoted;
                    }
                    // keep_acc is true. Check if anything changed.
                    if (new_acc.get() == itf->acc.get() && new_empty.get() == itf->empty.get()) {
                        memo[nid] = node;
                        return node;
                    }
                    auto res = std::make_shared<Interface<T, Acc>>(itf->_children, new_acc, new_empty);
                    memo[nid] = res;
                    return res;
                } else {
                    auto ub = std::static_pointer_cast<UpperBranch<T, Acc>>(node);
                    auto new_empty = ub->empty ? mutate_acc(ub->empty) : std::shared_ptr<Acc>(nullptr);
                    bool changed = (new_empty.get() != ub->empty.get());
                    auto new_children = std::make_shared<UpperChildren<T, Acc>>();
                    bool any_child_map_changed = false;
                    for (auto &kv : *ub->_children) {
                        const T& v = kv.first;
                        const auto& kids = kv.second;
                        std::unordered_map<int, UpperPtr<T, Acc>> new_kids;
                        bool child_map_changed = false;
                        for (auto &dkv : kv.second) {
                            auto new_child = transform(dkv.second);
                            if (new_child.get() != dkv.second.get()) child_map_changed = true;
                            if (new_child) new_kids[new_child->_max_depth] = new_child;
                        }
                        if (new_kids.size() != kids.size()) child_map_changed = true;

                        if (child_map_changed) {
                            changed = true;
                            any_child_map_changed = true;
                            if (!new_kids.empty()) (*new_children)[v] = std::move(new_kids);
                        } else {
                            (*new_children)[v] = kids;
                        }
                    }

                    if (!changed) {
                        memo[nid] = node;
                        return node;
                    }

                    if (!any_child_map_changed) {
                        // Only empty changed; reuse children pointer to preserve identity.
                        bool children_empty = ub->_children->empty();
                        if (children_empty && !new_empty) {
                            memo[nid] = nullptr;
                            return nullptr;
                        }
                        auto res = std::make_shared<UpperBranch<T, Acc>>(ub->_children, new_empty);
                        auto promoted = try_promote<T, Acc>(res);
                        memo[nid] = promoted;
                        return promoted;
                    }
                    if (new_children->empty() && !new_empty) {
                        memo[nid] = nullptr;
                        return nullptr;
                    }
                    auto res = std::make_shared<UpperBranch<T, Acc>>(std::move(new_children), new_empty);
                    auto promoted = try_promote<T, Acc>(res);
                    memo[nid] = promoted;
                    return promoted;
                }
            };

        auto res_inner = transform(inner);
        if (!res_inner) {
            return LeveledGSS(std::make_shared<UpperBranch<T, Acc>>(UpperChildren<T, Acc>{}, std::shared_ptr<Acc>(nullptr)));
        }
        return LeveledGSS(res_inner);
    }

    LeveledGSS merge(const LeveledGSS& other) const {
        return LeveledGSS(merge_upper<T, Acc>(inner, other.inner));
    }

    static LeveledGSS merge_many(const std::vector<LeveledGSS>& list) {
        if (list.empty()) {
            return LeveledGSS(std::make_shared<UpperBranch<T, Acc>>(UpperChildren<T, Acc>{}, std::shared_ptr<Acc>(nullptr)));
        }
        UpperPtr<T, Acc> cur = list[0].inner;
        for (size_t i = 1; i < list.size(); ++i) {
            cur = merge_upper<T, Acc>(cur, list[i].inner);
        }
        return LeveledGSS(cur);
    }

    std::unordered_set<T> peek() const {
        std::unordered_set<T> s;
        if (!inner->is_interface()) {
            auto ub = std::static_pointer_cast<UpperBranch<T, Acc>>(inner);
            for (auto &kv : *ub->_children) s.insert(kv.first);
        } else {
            auto it = std::static_pointer_cast<Interface<T, Acc>>(inner);
            for (auto &kv : *it->_children) s.insert(kv.first);
        }
        return s;
    }

    // Reduce unique accumulators by merging; returns nullptr if none
    std::shared_ptr<Acc> reduce_acc() const {
        std::unordered_map<std::uintptr_t, bool> visited;
        std::unordered_map<std::uintptr_t, std::shared_ptr<Acc>> unique_acc_objects;

        std::deque<UpperPtr<T, Acc>> q;
        q.push_back(inner);

        while (!q.empty()) {
            auto node = q.front(); q.pop_front();
            auto nid = reinterpret_cast<std::uintptr_t>(node.get());
            if (visited[nid]) continue;
            visited[nid] = true;

            if (node->is_interface()) {
                auto it = std::static_pointer_cast<Interface<T, Acc>>(node);
                if (it->acc) unique_acc_objects[reinterpret_cast<std::uintptr_t>(it->acc.get())] = it->acc;
                if (it->empty) unique_acc_objects[reinterpret_cast<std::uintptr_t>(it->empty.get())] = it->empty;
                // No recursion into lowers (no accs there)
            } else {
                auto ub = std::static_pointer_cast<UpperBranch<T, Acc>>(node);
                if (ub->empty) unique_acc_objects[reinterpret_cast<std::uintptr_t>(ub->empty.get())] = ub->empty;
                for (auto &kv : *ub->_children) {
                    for (auto &dkv : kv.second) {
                        q.push_back(dkv.second);
                    }
                }
            }
        }

        std::shared_ptr<Acc> acc;
        for (auto &kv : unique_acc_objects) {
            if (!acc) acc = kv.second;
            else acc = acc->merge(kv.second);
        }
        return acc;
    }
};

} // namespace leveled_gss
