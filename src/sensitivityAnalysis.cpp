#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cfloat>
#include <string>
#include <array>
#include <vector>
#include <cassert>
#include <fstream>
#include <set>
#include <queue>
#include <type_traits>
#include <iterator>
#include <chrono>

#include <Eigen/Dense>

#if __has_include(<mpi.h>)
#include <mpi.h>
#endif

#include <sunmatrix/sunmatrix_sparse.h>
#include <sunlinsol/sunlinsol_klu.h>
#include <cvodes/cvodes.h>
// #include <casadi/casadi.hpp>  // CasADi 依存は除去

#include "../include/NASAP_fit.hpp"
#include "../include/xorshift.hpp"
#include "../include/readcsv.hpp"

using std::vector;
using std::cout;
using std::endl;




void NASAP_fit::computeLMResAndJac(vector<double>& constant, Eigen::VectorXd& residual, Eigen::MatrixXd& jacobian){
    const int m = (int)QASAP.size() * cfg.trackedSpecies;
    const int n = cfg.constantSize;
    residual.resize(m);
    jacobian.resize(m, n);

    for (int i = 0; i < cfg.species; ++i) {
        NV_Ith_S(y, i) = initialState[i];
    }
    CVodeReInit(cvode_mem, 0.0, y);
    ReactionNetwork::CvodeUserData ud{ &rxnNet, constant.data(), nullptr, cfg.constantSize, nullptr };
    CVodeSetUserData(cvode_mem, (void*)&ud);
    N_Vector* uS = N_VCloneVectorArray(cfg.constantSize, y);
    for (int q = 0; q < cfg.constantSize; ++q) {
        N_VConst(0.0, uS[q]);
    }

    int flag = CVodeSensInit(cvode_mem, cfg.constantSize, CV_SIMULTANEOUS, ReactionNetwork::sensRhsCb, uS);
    assert(flag == CV_SUCCESS);
    flag = CVodeSensEEtolerances(cvode_mem);
    assert(flag == CV_SUCCESS);
    flag = CVodeSetSensErrCon(cvode_mem, SUNTRUE);
    assert(flag == CV_SUCCESS);
    flag = CVodeSetSensParams(cvode_mem, constant.data(), nullptr, nullptr);
    assert(flag == CV_SUCCESS);

    double tret = 0.0;
    bool hasReturnedSens = false;
    for (int i = 0; i < (int)QASAP.size(); ++i) {
        const double t = QASAP[i].time;
        assert(0 <= t && t <= endTime);
        if (t != tret) {
            flag = CVode(cvode_mem, t, y, &tret, CV_NORMAL);
            assert(flag >= 0);
            flag = CVodeGetSens(cvode_mem, &tret, uS);
            assert(flag == CV_SUCCESS);
            hasReturnedSens = true;
        } else if (!hasReturnedSens && t != 0.0) {
            flag = CVodeGetSens(cvode_mem, &tret, uS);
            assert(flag == CV_SUCCESS);
            hasReturnedSens = true;
        }

        const double* y_data = N_VGetArrayPointer(y);
        for (int j = 0; j < cfg.trackedSpecies; ++j) {
            const int row = i * cfg.trackedSpecies + j;
            const int speciesIndex = indexOrder[j];
            assert(0 <= speciesIndex && speciesIndex < cfg.species);

            residual[row] = y_data[speciesIndex] / cfg.fullConc[j] - QASAP[i].state[j] / 100.0;
            for (int q = 0; q < n; ++q) {
                jacobian(row, q) = NV_Ith_S(uS[q], speciesIndex) * constant[q] / cfg.fullConc[j];
            }
        }
    }

    CVodeSensFree(cvode_mem);
    N_VDestroyVectorArray(uS, cfg.constantSize);
}

vector<vector<double>> NASAP_fit::GaussNewtonHessian(const vector<double>& constant){
    validateConstants(constant);

    Eigen::VectorXd residual;
    Eigen::MatrixXd jacobian;

    vector<double> work = constant;
    computeLMResAndJac(work, residual, jacobian);

    Eigen::MatrixXd hessian = jacobian.transpose() * jacobian;
    vector<vector<double>> result(cfg.constantSize, vector<double>(cfg.constantSize, 0.0));
    for (int i = 0; i < cfg.constantSize; ++i) {
        for (int j = 0; j < cfg.constantSize; ++j) {
            result[i][j] = hessian(i, j);
        }
    }
    return result;
}


