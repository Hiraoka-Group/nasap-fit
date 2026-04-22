#include <iterator>
#include <string>
#include <vector>
#include <map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/bindings.hpp"
#include "../include/constants.hpp"
#include "../include/NASAP_fit.hpp"

namespace py = pybind11;

namespace {
NASAP_fit::Config make_default_config() {
    NASAP_fit::Config cfg;
    cfg.QASAPFile = "";
    cfg.reactNetworkFile = "";
    cfg.species = 0;
    cfg.constantSize = 0;
    cfg.trackedSpecies = 0;
    cfg.trackedNames = {};
    cfg.trackedIndex = {};
    cfg.fullConc = {};
    cfg.initConc = {};
    cfg.tolAbsError = 1e-10;
    cfg.tolRelError = 1e-06;
    cfg.scalar = 0.7;
    cfg.crossOver = 0.4;
    cfg.upperLim = 1e4;
    cfg.lowerLim = 1e-3;
    cfg.cvodeMaxNumSteps = 10000;
	cfg.logLevel = NASAP_fit::LogLevel::normal;
    return cfg;
}
}

void init_core(pybind11::module_ &m) {
    m.doc() = "pybind11 bindings for nasap_fit_cpp";

    m.def("default_config", []() { return make_default_config(); });

    py::enum_<NASAP_fit::LogLevel>(m, "LogLevel")
        .value("quiet", NASAP_fit::LogLevel::quiet)
        .value("normal", NASAP_fit::LogLevel::normal)
        .value("verbose", NASAP_fit::LogLevel::verbose)
        .export_values();

    py::class_<NASAP_fit::Config>(m, "Config")
        // Config is easiest to construct from the project's defaults.
        .def(py::init([]() { return make_default_config(); }))
        .def_readwrite("QASAPFile", &NASAP_fit::Config::QASAPFile)
        .def_readwrite("reactNetworkFile", &NASAP_fit::Config::reactNetworkFile)
        .def_readwrite("species", &NASAP_fit::Config::species)
        .def_readwrite("constantSize", &NASAP_fit::Config::constantSize)
        .def_readwrite("trackedSpecies", &NASAP_fit::Config::trackedSpecies)
        .def_readwrite("trackedNames", &NASAP_fit::Config::trackedNames)
        .def_readwrite("trackedIndex", &NASAP_fit::Config::trackedIndex)
        .def_readwrite("fullConc", &NASAP_fit::Config::fullConc)
        .def_readwrite("initConc", &NASAP_fit::Config::initConc)
        .def_readwrite("tolAbsError", &NASAP_fit::Config::tolAbsError)
        .def_readwrite("tolRelError", &NASAP_fit::Config::tolRelError)
        .def_readwrite("cvodeMaxNumSteps", &NASAP_fit::Config::cvodeMaxNumSteps)
        .def_readwrite("logLevel", &NASAP_fit::Config::logLevel)
        .def_readwrite("scalar", &NASAP_fit::Config::scalar)
        .def_readwrite("crossOver", &NASAP_fit::Config::crossOver)
        .def_readwrite("upperLim", &NASAP_fit::Config::upperLim)
        .def_readwrite("lowerLim", &NASAP_fit::Config::lowerLim);

    py::class_<NASAP_fit::OptimizeResult>(m, "OptimizeResult")
        .def_readonly("constants", &NASAP_fit::OptimizeResult::constants)
        .def_readonly("error", &NASAP_fit::OptimizeResult::error);

    py::class_<NASAP_fit::TerminationCondition>(m, "TerminationCondition")
        .def(py::init<>())
        .def_readwrite("maxIter", &NASAP_fit::TerminationCondition::maxIter)
        .def_readwrite("timeLimit", &NASAP_fit::TerminationCondition::timeLimit)
        .def_readwrite("xtol", &NASAP_fit::TerminationCondition::xtol)
        .def_readwrite("ftolAbs", &NASAP_fit::TerminationCondition::ftolAbs)
        .def_readwrite("ftolRel", &NASAP_fit::TerminationCondition::ftolRel)
        .def_readwrite("targetError", &NASAP_fit::TerminationCondition::targetError)
        .def_readwrite("stall", &NASAP_fit::TerminationCondition::stall);

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
        // NASAP_fit owns non-copyable/non-movable resources (e.g., sundials::Context),
        // so avoid return-by-value factory init which requires move construction.
        .def(py::init<const NASAP_fit::Config&>(), py::arg("cfg"))
        .def("constants", &NASAP_fit::constants, py::return_value_policy::reference_internal)
		.def("termIndex", &NASAP_fit::termIndex, py::return_value_policy::reference_internal)
		.def("reactionCount", &NASAP_fit::reactionCount)
		.def("calcError", &NASAP_fit::calcError, py::arg("constant"))
        .def("runDE",
            py::overload_cast<int, double, double, const NASAP_fit::TerminationCondition&, uint64_t>(&NASAP_fit::runDE),
            py::arg("popSize"),
            py::arg("lowerLim") = 1e-3,
            py::arg("upperLim") = 1e4,
            py::arg("termCond"),
            py::arg("seed") = 1)
        .def("runDE",
            py::overload_cast<std::vector<std::vector<double>>, const NASAP_fit::TerminationCondition&, uint64_t>(&NASAP_fit::runDE),
            py::arg("arg"),
            py::arg("termCond"),
            py::arg("seed") = 1)
/*
        .def("runLM",
            py::overload_cast<const std::vector<double>&, const NASAP_fit::TerminationCondition&>(&NASAP_fit::runLM),
            py::arg("theta0"),
            py::arg("termCond"))
        .def("runLM",
            py::overload_cast<const std::vector<std::vector<double>>&, const NASAP_fit::TerminationCondition&>(&NASAP_fit::runLM),
            py::arg("thetaList"),
            py::arg("termCond"))
        .def("getHessian", &NASAP_fit::getHessian, py::arg("point"))
        .def("pseudoHessian", &NASAP_fit::pseudoHessian, py::arg("point"))
        */
        .def("simulate", &NASAP_fit::simulate, py::arg("t"), py::arg("constant"), py::arg("reaction_ids"));
}