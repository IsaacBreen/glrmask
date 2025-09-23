#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "leveled_gss.hpp"

namespace py = pybind11;

// Add this block after the includes
namespace std {
    template <>
    struct hash<py::object> {
        size_t operator()(const py::object& obj) const {
            return py::hash(obj);
        }
    };
}

// Accumulator that wraps a Python object and calls its `merge` method.
struct PyAcc : public std::enable_shared_from_this<PyAcc> {
    py::object value;

    PyAcc(py::object v) : value(std::move(v)) {}

    std::shared_ptr<PyAcc> merge(const std::shared_ptr<PyAcc>& other) const {
        py::object new_value = value.attr("merge")(other->value);
        return std::make_shared<PyAcc>(new_value);
    }

    bool operator==(const PyAcc& other) const {
        return value.attr("__eq__")(other.value).cast<bool>();
    }
};

using GSS_T = py::object;
using GSS_Acc = PyAcc;
using CppGSS = leveled_gss::LeveledGSS<GSS_T, GSS_Acc>;

// Wrapper class to be exposed to Python
class LeveledGssPyWrapper {
public:
    CppGSS gss;

    LeveledGssPyWrapper(CppGSS g) : gss(std::move(g)) {}

    static LeveledGssPyWrapper from_stacks(const py::list& stacks_py) {
        std::vector<std::pair<std::vector<GSS_T>, std::shared_ptr<GSS_Acc>>> stacks_cpp;
        stacks_cpp.reserve(stacks_py.size());
        for (auto handle : stacks_py) {
            py::tuple t = handle.cast<py::tuple>();
            std::vector<GSS_T> vals = t[0].cast<std::vector<GSS_T>>();
            auto acc = std::make_shared<GSS_Acc>(t[1].cast<py::object>());
            stacks_cpp.emplace_back(std::move(vals), std::move(acc));
        }
        return LeveledGssPyWrapper(CppGSS::from_stacks(stacks_cpp));
    }

    py::list to_stacks() const {
        auto stacks_cpp = gss.to_stacks();
        py::list stacks_py;
        for (const auto& p : stacks_cpp) {
            stacks_py.append(py::make_tuple(p.first, p.second->value));
        }
        return stacks_py;
    }

    LeveledGssPyWrapper push(const GSS_T& value) const {
        return LeveledGssPyWrapper(gss.push(value));
    }

    LeveledGssPyWrapper pop() const {
        return LeveledGssPyWrapper(gss.pop());
    }
    
    LeveledGssPyWrapper popn(int n) const {
        return LeveledGssPyWrapper(gss.popn(n));
    }

    bool is_empty() const {
        return gss.is_empty();
    }

    LeveledGssPyWrapper isolate(py::object value) const {
        if (value.is_none()) {
            return LeveledGssPyWrapper(gss.isolate_none());
        }
        return LeveledGssPyWrapper(gss.isolate(value.cast<GSS_T>()));
    }

    LeveledGssPyWrapper isolate_many(const py::list& values) const {
        std::unordered_set<GSS_T> values_set;
        bool has_none = false;
        for (auto v_obj : values) {
            if (v_obj.is_none()) {
                has_none = true;
            } else {
                values_set.insert(v_obj.cast<GSS_T>());
            }
        }
        
        CppGSS result_gss = gss.isolate_many(values_set);
        if (has_none) {
            CppGSS none_gss = gss.isolate_none();
            result_gss = result_gss.merge(none_gss);
        }
        return LeveledGssPyWrapper(result_gss);
    }

    LeveledGssPyWrapper apply(const py::function& func) const {
        auto cpp_func = [&](const std::shared_ptr<GSS_Acc>& acc) -> std::shared_ptr<GSS_Acc> {
            py::object new_val = func(acc->value);
            return std::make_shared<GSS_Acc>(new_val);
        };
        return LeveledGssPyWrapper(gss.apply(cpp_func));
    }

    LeveledGssPyWrapper prune(const py::function& predicate) const {
        auto cpp_predicate = [&](const std::shared_ptr<GSS_Acc>& acc) -> bool {
            return predicate(acc->value).cast<bool>();
        };
        return LeveledGssPyWrapper(gss.prune(cpp_predicate));
    }

    LeveledGssPyWrapper apply_and_prune(const py::function& mutator) const {
        auto cpp_mutator = [&](const std::shared_ptr<GSS_Acc>& acc) -> std::shared_ptr<GSS_Acc> {
            py::object result = mutator(acc->value);
            if (result.is_none()) {
                return nullptr;
            }
            return std::make_shared<GSS_Acc>(result);
        };
        return LeveledGssPyWrapper(gss.apply_and_prune(cpp_mutator));
    }

    LeveledGssPyWrapper merge(const LeveledGssPyWrapper& other) const {
        return LeveledGssPyWrapper(gss.merge(other.gss));
    }

    static LeveledGssPyWrapper merge_many(const py::list& gss_list) {
        std::vector<CppGSS> gsses;
        gsses.reserve(gss_list.size());
        for (auto g_obj : gss_list) {
            gsses.push_back(g_obj.cast<LeveledGssPyWrapper>().gss);
        }
        return LeveledGssPyWrapper(CppGSS::merge_many(gsses));
    }

    py::set peek() const {
        auto tops = gss.peek();
        py::set result;
        for (const auto& t : tops) {
            result.add(t);
        }
        return result;
    }

    py::object reduce_acc() const {
        auto acc = gss.reduce_acc();
        if (acc) {
            return acc->value;
        }
        return py::none();
    }
};

PYBIND11_MODULE(leveled_gss_cpp, m) {
    m.doc() = "pybind11 wrapper for LeveledGSS C++ implementation";

    py::class_<LeveledGssPyWrapper>(m, "LeveledGssCpp")
        .def_static("from_stacks", &LeveledGssPyWrapper::from_stacks)
        .def("to_stacks", &LeveledGssPyWrapper::to_stacks)
        .def("push", &LeveledGssPyWrapper::push)
        .def("pop", &LeveledGssPyWrapper::pop)
        .def("popn", &LeveledGssPyWrapper::popn)
        .def("is_empty", &LeveledGssPyWrapper::is_empty)
        .def("isolate", &LeveledGssPyWrapper::isolate)
        .def("isolate_many", &LeveledGssPyWrapper::isolate_many)
        .def("apply", &LeveledGssPyWrapper::apply)
        .def("prune", &LeveledGssPyWrapper::prune)
        .def("apply_and_prune", &LeveledGssPyWrapper::apply_and_prune)
        .def("merge", &LeveledGssPyWrapper::merge)
        .def_static("merge_many", &LeveledGssPyWrapper::merge_many)
        .def("peek", &LeveledGssPyWrapper::peek)
        .def("reduce_acc", &LeveledGssPyWrapper::reduce_acc);
}
