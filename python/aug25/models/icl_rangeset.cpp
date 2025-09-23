#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <boost/icl/interval_set.hpp>
#include <boost/icl/interval.hpp>

#include <vector>
#include <string>
#include <sstream>
#include <utility>
#include <algorithm>

namespace py = pybind11;

class RangeSet {
public:
    using interval_type = boost::icl::discrete_interval<unsigned long long>;
    using set_type = boost::icl::interval_set<unsigned long long>;

    RangeSet() = default;

    static RangeSet empty() {
        return RangeSet();
    }

    static RangeSet from_indices(const std::vector<unsigned long long>& indices) {
        RangeSet rs;
        for (unsigned long long v : indices) {
            rs.m_set.add(interval_type::closed(v, v));
        }
        return rs;
    }

    static RangeSet from_ranges(const std::vector<std::pair<unsigned long long, unsigned long long>>& ranges) {
        RangeSet rs;
        for (auto const& pr : ranges) {
            unsigned long long l = pr.first;
            unsigned long long r = pr.second;
            if (r < l) std::swap(l, r);
            rs.m_set.add(interval_type::closed(l, r));
        }
        return rs;
    }

    bool contains(unsigned long long v) const {
        return boost::icl::contains(m_set, v);
    }

    RangeSet union_with(const RangeSet& other) const {
        RangeSet res;
        res.m_set = m_set;
        res.m_set += other.m_set;
        return res;
    }

    RangeSet intersection_with(const RangeSet& other) const {
        RangeSet res;
        res.m_set = m_set & other.m_set;
        return res;
    }

    RangeSet difference_with(const RangeSet& other) const {
        RangeSet res;
        res.m_set = m_set - other.m_set;
        return res;
    }

    bool is_empty() const {
        return m_set.empty();
    }

    std::vector<std::pair<unsigned long long, unsigned long long>> to_ranges() const {
        std::vector<std::pair<unsigned long long, unsigned long long>> out;
        out.reserve(m_set.size());
        for (auto const& itv : m_set) {
            unsigned long long l = itv.lower();
            unsigned long long r = itv.upper();
            out.emplace_back(l, r);
        }
        return out;
    }

    std::vector<unsigned long long> to_indices() const {
        std::vector<unsigned long long> out;
        out.reserve(boost::icl::cardinality(m_set));
        for (auto const& itv : m_set) {
            unsigned long long l = itv.lower();
            unsigned long long r = itv.upper();
            for (unsigned long long v = l; v <= r; ++v) {
                out.push_back(v);
            }
        }
        return out;
    }

    bool operator==(const RangeSet& other) const {
        return m_set == other.m_set;
    }

    std::string repr() const {
        std::ostringstream oss;
        oss << "RangeSet(";
        bool first = true;
        for (auto const& itv : m_set) {
            if (!first) oss << ", ";
            first = false;
            oss << "[" << itv.lower() << ", " << itv.upper() << "]";
        }
        oss << ")";
        return oss.str();
    }

private:
    set_type m_set;
};

PYBIND11_MODULE(icl_rangeset, m) {
    m.doc() = "Boost.ICL-backed RangeSet";

    py::class_<RangeSet>(m, "RangeSet")
        .def(py::init<>())
        .def_static("empty", &RangeSet::empty, "Create an empty RangeSet")
        .def_static("from_indices", &RangeSet::from_indices, py::arg("indices"),
                    "Create a RangeSet from a list of indices")
        .def_static("from_ranges", &RangeSet::from_ranges, py::arg("ranges"),
                    "Create a RangeSet from a list of inclusive ranges [[start, end], ...]")
        .def("union", &RangeSet::union_with, py::arg("other"),
             "Return the union of this and other")
        .def("intersection", &RangeSet::intersection_with, py::arg("other"),
             "Return the intersection of this and other")
        .def("difference", &RangeSet::difference_with, py::arg("other"),
             "Return the difference of this and other (self - other)")
        .def("contains", &RangeSet::contains, py::arg("index"),
             "Check if index is contained in the set")
        .def("to_ranges", &RangeSet::to_ranges,
             "Return list of inclusive ranges [[start, end], ...]")
        .def("to_indices", &RangeSet::to_indices,
             "Return list of all indices in the set")
        .def("is_empty", &RangeSet::is_empty,
             "True if the set is empty")
        .def("__repr__", &RangeSet::repr)
        .def("__eq__", &RangeSet::operator==)
        ;
}
