#pragma once
#include <pybind11/pybind11.h>

// Initializes Python bindings for the nasap_fit_cpp core types.
void init_core(pybind11::module_ &m);

