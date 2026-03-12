#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

void init_structure(pybind11::module_ &);

void init_differentialEvolution(pybind11::module_ &);

