#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "icl_rangeset.hpp"

namespace py = pybind11;

PYBIND11_MODULE(icl_rangeset, m) {
    m.doc() = "Boost.ICL-backed RangeSet";

    py::class_<RangeSet>(m, "RangeSet")
        .def(py::init<>())
        .def_static("from_singleton", &RangeSet::from_singleton, py::arg("index"),
                    "Create a RangeSet containing exactly one index")
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
