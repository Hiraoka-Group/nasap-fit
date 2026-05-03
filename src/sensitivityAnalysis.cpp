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
    ReactionNetwork::CvodeUserData ud{ &rxnNet, constant.data(), nullptr };
    CVodeSetUserData(cvode_mem, (void*)&ud);
    bool flag=CV_SUCCESS;
    N_Vector* uS = N_VCloneVectorArray(cfg.constantSize, y);
    flag = CVodeSensInit(cvode_mem, cfg.constantSize, CV_SIMULTANEOUS, NULL, uS);
    flag = CVodeSensEEtolerances;
    flag = CVodeSetSensErrCon;
    flag = CVodeSetSensDQMethod;
    flag = CVodeSetSensParams;
    assert(flag == CV_SUCCESS);
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
}