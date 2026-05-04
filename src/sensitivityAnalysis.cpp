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

//#include <Eigen/Dense>

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

vector<vector<double>> NASAP_fit::forwardSensitivityAnalysis(vector<double>& constant){
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
    flag = CVode(cvode_mem, endTime, y, &tret, CV_NORMAL);
    assert(flag >= 0);
    flag = CVodeGetSens(cvode_mem, &tret, uS);
    assert(flag == CV_SUCCESS);

    vector<vector<double>> sensitivities(cfg.constantSize, vector<double>(cfg.species, 0.0));
    for (int q = 0; q < cfg.constantSize; ++q) {
        for (int a = 0; a < cfg.species; ++a) {
            sensitivities[q][a] = NV_Ith_S(uS[q], a);
        }
    }

    N_VDestroyVectorArray(uS, cfg.constantSize);
    return sensitivities;
}
vector<vector<double>> NASAP_fit::backwardSensitivityAnalysis(vector<double>& constant){
    for (int i = 0; i < cfg.species; ++i) {
        NV_Ith_S(y, i) = initialState[i];
    }
    CVodeReInit(cvode_mem, 0.0, y);
    ReactionNetwork::CvodeUserData ud{ &rxnNet, constant.data(), nullptr };
    CVodeSetUserData(cvode_mem, (void*)&ud);
    bool flag=CV_SUCCESS;
    flag = CVodeAdjInit(cvode_mem, 50, CV_HERMITE);
    assert(flag == CV_SUCCESS);
    return {};
}
