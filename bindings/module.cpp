#include <stdexcept>
#include <utility>
#include <vector>

#include <mpi.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../include/constants.hpp"
#include "../../include/NASAP_fit.hpp"
#include "../../include/runtime_context.hpp"

namespace py = pybind11;

namespace {
void ensure_mpi_initialized() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int argc = 0;
        char** argv = nullptr;
        int provided = 0;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    }
    initialize_runtime_context();
}
}

class NASAP_fit_Engine {
public:
    explicit NASAP_fit_Engine(std::vector<std::vector<double>> qasap_data) {
        ensure_mpi_initialized();
        engine_ = std::make_unique<differentialEvolution>(qasap_data);
    }

    void optimize() { engine_->Optimize(); }

    void sort_populations_by_error() { engine_->sortPopulationsByError(); }

    void run_lm(int idx) {
        if (idx < 0 || idx >= engine_->constants().popSize) {
            throw std::out_of_range("idx is out of range");
        }
        engine_->runLM(idx);
    }

    int best_index() const { return engine_->best(); }

    std::vector<double> get_population(int idx) {
        if (idx < 0 || idx >= engine_->constants().popSize) {
            throw std::out_of_range("idx is out of range");
        }
        return engine_->getPop(idx);
    }

    double get_population_error(int idx) {
        if (idx < 0 || idx >= engine_->constants().popSize) {
            throw std::out_of_range("idx is out of range");
        }
        return engine_->getPopError(idx);
    }

    int tracked_species() const { return engine_->constants().trackedSpecies; }

    int population_size() const { return engine_->constants().popSize; }

private:
    std::unique_ptr<differentialEvolution> engine_;
};

PYBIND11_MODULE(_core, m) {
    m.doc() = "pybind11 bindings for nasap_fit_cpp";

    m.def("expected_input_columns", []() { return config::trackedSpecies + 1; });
    m.def("default_population_size", []() { return config::popSize; });

    py::class_<NASAP_fit_Engine>(m, "NASAP_fit_Engine")
        .def(py::init<std::vector<std::vector<double>>>(), py::arg("qasap_data"))
        .def("optimize", &NASAP_fit_Engine::optimize)
        .def("sort_populations_by_error", &NASAP_fit_Engine::sort_populations_by_error)
        .def("run_lm", &NASAP_fit_Engine::run_lm, py::arg("idx"))
        .def("best_index", &NASAP_fit_Engine::best_index)
        .def("get_population", &NASAP_fit_Engine::get_population, py::arg("idx"))
        .def("get_population_error", &NASAP_fit_Engine::get_population_error, py::arg("idx"))
        .def_property_readonly("tracked_species", &NASAP_fit_Engine::tracked_species)
        .def_property_readonly("population_size", &NASAP_fit_Engine::population_size);
}