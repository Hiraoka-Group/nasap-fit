#include <iterator>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/bindings.hpp"
#include "../include/constants.hpp"
#include "../include/NASAP_fit.hpp"

namespace py = pybind11;

namespace {
NASAP_fit::Config make_default_config() {
    NASAP_fit::Config cfg;
    cfg.QASAPFile = config::QASAPFile;
    cfg.reactNetworkFile = config::reactNetworkFile;
    cfg.species = config::species;
    cfg.constantSize = config::constantSize;
    cfg.trackedSpecies = config::trackedSpecies;
    cfg.trackedNames = std::vector<std::string>(std::begin(config::trackedNames), std::end(config::trackedNames));
    cfg.trackedIndex = std::vector<int>(std::begin(config::trackedIndex), std::end(config::trackedIndex));
    cfg.fullConc = std::vector<double>(std::begin(config::fullConc), std::end(config::fullConc));
    cfg.initConc = config::initConc;
    cfg.tolAbsError = config::tolAbsError;
    cfg.tolRelError = config::tolRelError;
    cfg.scalar = config::scalar;
    cfg.crossOver = config::crossOver;
    cfg.upperLim = config::upperLim;
    cfg.lowerLim = config::lowerLim;
    return cfg;
}
}

void init_core(pybind11::module_ &m) {
    m.doc() = "pybind11 bindings for nasap_fit_cpp";

    m.def("default_config", []() { return make_default_config(); });

    py::class_<NASAP_fit::Config>(m, "Config")
        // Config contains vector<string_view>; default-constructing it makes it unusable.
        .def(py::init([]() { return make_default_config(); }))
        .def_readwrite("QASAPFile", &NASAP_fit::Config::QASAPFile)
        .def_readwrite("reactNetworkFile", &NASAP_fit::Config::reactNetworkFile)
        .def_readwrite("species", &NASAP_fit::Config::species)
        .def_readwrite("constantSize", &NASAP_fit::Config::constantSize)
        .def_readwrite("trackedSpecies", &NASAP_fit::Config::trackedSpecies)
        .def_property_readonly("trackedNames", [](const NASAP_fit::Config& c) {
            std::vector<std::string> out;
            out.reserve(c.trackedNames.size());
            for (auto sv : c.trackedNames) out.emplace_back(sv);
            return out;
        })
        // trackedIndex/fullConc/initConc are safe as read-only
        .def_readonly("trackedIndex", &NASAP_fit::Config::trackedIndex)
        .def_readonly("fullConc", &NASAP_fit::Config::fullConc)
        .def_readonly("initConc", &NASAP_fit::Config::initConc)
        .def_readwrite("tolAbsError", &NASAP_fit::Config::tolAbsError)
        .def_readwrite("tolRelError", &NASAP_fit::Config::tolRelError)
        .def_readwrite("scalar", &NASAP_fit::Config::scalar)
        .def_readwrite("crossOver", &NASAP_fit::Config::crossOver)
        .def_readwrite("upperLim", &NASAP_fit::Config::upperLim)
        .def_readwrite("lowerLim", &NASAP_fit::Config::lowerLim);

    py::class_<NASAP_fit::OptimizeResult>(m, "OptimizeResult")
        .def_readonly("constants", &NASAP_fit::OptimizeResult::constants)
        .def_readonly("error", &NASAP_fit::OptimizeResult::error);

    py::class_<NASAP_fit::SimulationResult::ReactionProgressResult>(m, "ReactionProgressResult")
        .def_readonly("reaction_ids", &NASAP_fit::SimulationResult::ReactionProgressResult::reaction_ids)
        .def_readonly("J", &NASAP_fit::SimulationResult::ReactionProgressResult::J)
        .def_readonly("reaction_labels", &NASAP_fit::SimulationResult::ReactionProgressResult::reaction_labels);

    py::class_<NASAP_fit::SimulationResult>(m, "SimulationResult")
        .def_readonly("status", &NASAP_fit::SimulationResult::status)
        .def_readonly("timePoints", &NASAP_fit::SimulationResult::timePoints)
        .def_readonly("t", &NASAP_fit::SimulationResult::t)
        .def_readonly("y", &NASAP_fit::SimulationResult::y)
        .def_readonly("reactionProgress", &NASAP_fit::SimulationResult::reactionProgress);

    py::class_<NASAP_fit>(m, "NASAP_fit")
        .def(py::init([]() { return NASAP_fit(make_default_config()); }))
        .def(py::init<const NASAP_fit::Config&>(), py::arg("cfg"))
        .def("constants", &NASAP_fit::constants, py::return_value_policy::reference_internal)
        .def("runDE",
            py::overload_cast<int, int, double, double>(&NASAP_fit::runDE),
            py::arg("maxGen"),
            py::arg("popSize"),
            py::arg("lowerLim") = 1e-3,
            py::arg("upperLim") = 1e4)
        .def("runDE",
            py::overload_cast<std::vector<std::vector<double>>>(&NASAP_fit::runDE),
            py::arg("arg"))
        .def("runLM",
            py::overload_cast<const std::vector<double>&>(&NASAP_fit::runLM),
            py::arg("theta0"))
        .def("runLM",
            py::overload_cast<const std::vector<std::vector<double>>&>(&NASAP_fit::runLM),
            py::arg("thetaList"))
        .def("getHessian", &NASAP_fit::getHessian, py::arg("point"))
        .def("pseudoHessian", &NASAP_fit::pseudoHessian, py::arg("point"))
        .def("simulate", &NASAP_fit::simulate, py::arg("t"), py::arg("constant"), py::arg("reaction_ids"));
}