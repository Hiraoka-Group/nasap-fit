#include <pybind11/pybind11.h>

#include "../../include/bindings.hpp"

PYBIND11_MODULE(_core, m) {
	init_core(m);
}