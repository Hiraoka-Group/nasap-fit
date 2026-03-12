#include <pybind11/pybind11.h>
#include "../include/bindings.hpp"
#include "../include/NASAP_fit.hpp"

void init_NASAP_fit(pybind11::module_ &m) {
    pybind11::class_<>(m, "NASAP_fit")
        .def(pybind11::init<const std::string &>())
        .def("setName", &NASAP_fit::setName)
        .def("getName", &NASAP_fit::getName)
        .def("__repr__",
            [](const NASAP_fit &a) {
                return "<pybind11_demo.structure.NASAP_fit named '" + a.name + "'>";
            }
        );
}

void init_structure(pybind11::module_ &m) {
    pybind11::module structure = m.def_submodule("structure");

    init_NASAP_fit(structure);
}