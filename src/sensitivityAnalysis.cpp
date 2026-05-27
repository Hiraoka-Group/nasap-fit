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
#include <cmath>

#include <Eigen/Dense>

#if defined(NASAP_USE_MPI) && NASAP_USE_MPI
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

namespace {

struct HessianAdjointData {
    ReactionNetwork* net{};
    const double* p{};
    int direction = 0;
};

inline double species_value(const double* y, int species, int idx) {
    return (idx == species) ? 1.0 : y[idx];
}

inline double tangent_value(const double* s, int species, int idx) {
    return (idx == species) ? 0.0 : s[idx];
}

void add_jacobian_transpose_product(
    const ReactionNetwork& net,
    const double* y,
    const double* p,
    const double* lambda,
    double* out
) {
    for (const auto& term : net.rhsTerms) {
        const int a = term.add_to;
        const int r = term.rateConstant;
        const int j = term.reactant1;
        const int k = term.reactant2;
        const double ckp = static_cast<double>(term.duplicacy) * p[r];
        const double la = lambda[a];

        if (j != net.species) {
            out[j] += la * ckp * species_value(y, net.species, k);
        }
        if (k != net.species) {
            out[k] += la * ckp * species_value(y, net.species, j);
        }
    }
}

//right hand side function for yB={adjoint of y, adjoint of s}
int hessianAdjointRhs(
    sunrealtype /*t*/,
    N_Vector y,
    N_Vector* yS,
    N_Vector yB,
    N_Vector yBdot,
    void* user_dataB
) {
    auto* ud = static_cast<HessianAdjointData*>(user_dataB);
    if (ud == nullptr || ud->net == nullptr || ud->p == nullptr || yS == nullptr || yS[0] == nullptr) return -1;

    const ReactionNetwork& net = *ud->net;
    const int n = net.species;
    const double* p = ud->p;
    const double pdir = p[ud->direction];
    const double* yData = N_VGetArrayPointer(y);
    const double* ySData = N_VGetArrayPointer(yS[0]);
    const double* bData = N_VGetArrayPointer(yB);
    double* bdotData = N_VGetArrayPointer(yBdot);
    if (yData == nullptr || ySData == nullptr || bData == nullptr || bdotData == nullptr) return -1;

    std::fill(bdotData, bdotData + 2 * n, 0.0);
    const double* lambdaY = bData;
    const double* lambdaS = bData + n;
    double* outY = bdotData;
    double* outS = bdotData + n;

    add_jacobian_transpose_product(net, yData, p, lambdaY, outY);
    add_jacobian_transpose_product(net, yData, p, lambdaS, outS);

    for (const auto& term : net.rhsTerms) {
        const int a = term.add_to;
        const int r = term.rateConstant;
        const int j = term.reactant1;
        const int k = term.reactant2;
        const double ckp = static_cast<double>(term.duplicacy) * p[r];
        const double la = lambdaS[a];
        const double yj = species_value(yData, n, j);
        const double yk = species_value(yData, n, k);
        const double sj = pdir * tangent_value(ySData, n, j);
        const double sk = pdir * tangent_value(ySData, n, k);

        if (j != n) {
            outY[j] += la * ckp * sk;
            if (r == ud->direction) outY[j] += la * ckp * yk;
        }
        if (k != n) {
            outY[k] += la * ckp * sj;
            if (r == ud->direction) outY[k] += la * ckp * yj;
        }
    }

    for (int i = 0; i < 2 * n; ++i) {
        bdotData[i] = -bdotData[i];
    }
    return 0;
}

//∂^{2}G/∂p_i ∂p_j を積分して得るためのCVODESのquadRhsコールバック関数
int hessianAdjointQuadRhs(
    sunrealtype /*t*/,
    N_Vector y,
    N_Vector* yS,
    N_Vector yB,
    N_Vector qBdot,
    void* user_dataB
) {
    auto* ud = static_cast<HessianAdjointData*>(user_dataB);
    if (ud == nullptr || ud->net == nullptr || ud->p == nullptr || yS == nullptr || yS[0] == nullptr) return -1;

    const ReactionNetwork& net = *ud->net;
    const int n = net.species;
    const double* p = ud->p;
    const double pdir = p[ud->direction];
    const double* yData = N_VGetArrayPointer(y);
    const double* ySData = N_VGetArrayPointer(yS[0]);
    const double* bData = N_VGetArrayPointer(yB);
    double* qdotData = N_VGetArrayPointer(qBdot);
    if (yData == nullptr || ySData == nullptr || bData == nullptr || qdotData == nullptr) return -1;

    std::fill(qdotData, qdotData + net.constantSize, 0.0);
    const double* lambdaY = bData;
    const double* lambdaS = bData + n;

    for (const auto& term : net.rhsTerms) {
        const int a = term.add_to;
        const int r = term.rateConstant;
        const int j = term.reactant1;
        const int k = term.reactant2;
        const double ckp = static_cast<double>(term.duplicacy) * p[r];
        const double yj = species_value(yData, n, j);
        const double yk = species_value(yData, n, k);
        const double sj = pdir * tangent_value(ySData, n, j);
        const double sk = pdir * tangent_value(ySData, n, k);
        const double yprod = yj * yk;
        const double sproddot = sj * yk + yj * sk;

        qdotData[r] += lambdaY[a] * ckp * yprod;
        qdotData[r] += lambdaS[a] * ckp * sproddot;
        if (r == ud->direction) {
            qdotData[r] += lambdaS[a] * ckp * yprod;
        }
    }
    return 0;
}

}



//残差ベクトルおよび残差ベクトルに対するヤコビアン行列を計算するための関数
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

            const double observedConcentration = QASAP[i].state[j] * cfg.fullConc[j] / 100.0;
            residual[row] = y_data[speciesIndex] - observedConcentration;
            for (int q = 0; q < n; ++q) {
                jacobian(row, q) = NV_Ith_S(uS[q], speciesIndex) * constant[q];
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

//SSRの、log constantに関しての二階微分を計算するための関数
vector<vector<double>> NASAP_fit::calc_hessian(const vector<double>& constant) {
    validateConstants(constant);

    const int n = cfg.constantSize;
    const int neqB = 2 * cfg.species;
    vector<vector<double>> result(n, vector<double>(n, 0.0));
    vector<double> work = constant;

    vector<vector<double>> obsY(QASAP.size(), vector<double>(cfg.species, 0.0));
    vector<vector<double>> obsS(QASAP.size(), vector<double>(cfg.species, 0.0));

    auto apply_observation_jump = [&](int obsIndex, N_Vector yB) {
        double* bData = N_VGetArrayPointer(yB);
        assert(bData != nullptr);
        double* lambdaY = bData;
        double* lambdaS = bData + cfg.species;

        for (int j = 0; j < cfg.trackedSpecies; ++j) {
            const int speciesIndex = indexOrder[j];
            assert(0 <= speciesIndex && speciesIndex < cfg.species);
            const double observedConcentration = QASAP[obsIndex].state[j] * cfg.fullConc[j] / 100.0;
            const double residual = obsY[obsIndex][speciesIndex] - observedConcentration;
            const double sensitivity = obsS[obsIndex][speciesIndex];
            lambdaY[speciesIndex] += 2.0 * sensitivity;
            lambdaS[speciesIndex] += 2.0 * residual;
        }
    };

    for (int direction = 0; direction < n; ++direction) {
        for (int i = 0; i < cfg.species; ++i) {
            NV_Ith_S(y, i) = initialState[i];
        }
        CVodeReInit(cvode_mem, 0.0, y);
        CVodeQuadReInit(cvode_mem, yQ0);

        N_Vector* uS = N_VCloneVectorArray(1, y);
        assert(uS != nullptr);
        N_VConst(0.0, uS[0]);

        ReactionNetwork::CvodeUserData ud{ &rxnNet, work.data(), nullptr, 1, &direction };
        int flag = CVodeSetUserData(cvode_mem, (void*)&ud);
        assert(flag == CV_SUCCESS);

        flag = CVodeSensInit(cvode_mem, 1, CV_SIMULTANEOUS, ReactionNetwork::sensRhsCb, uS);
        assert(flag == CV_SUCCESS);
        flag = CVodeSensEEtolerances(cvode_mem);
        assert(flag == CV_SUCCESS);
        flag = CVodeSetSensErrCon(cvode_mem, SUNTRUE);
        assert(flag == CV_SUCCESS);
        flag = CVodeSetSensParams(cvode_mem, work.data(), nullptr, &direction);
        assert(flag == CV_SUCCESS);

        flag = CVodeAdjInit(cvode_mem, 100, CV_HERMITE);
        assert(flag == CV_SUCCESS);

        double tret = 0.0;
        int ncheck = 0;
        for (int obs = 0; obs < (int)QASAP.size(); ++obs) {
            const double t = QASAP[obs].time;
            assert(0 <= t && t <= endTime);
            if (t != tret) {
                flag = CVodeF(cvode_mem, t, y, &tret, CV_NORMAL, &ncheck);
                assert(flag >= 0);
                flag = CVodeGetSens(cvode_mem, &tret, uS);
                assert(flag == CV_SUCCESS);
            }

            const double* yData = N_VGetArrayPointer(y);
            const double* sData = N_VGetArrayPointer(uS[0]);
            assert(yData != nullptr && sData != nullptr);
            for (int i = 0; i < cfg.species; ++i) {
                obsY[obs][i] = yData[i];
                obsS[obs][i] = work[direction] * sData[i];
            }
        }

        N_Vector yB = N_VNew_Serial(neqB, sunctx);
        N_Vector qB = N_VNew_Serial(n, sunctx);
        assert(yB != nullptr && qB != nullptr);
        N_VConst(0.0, yB);
        N_VConst(0.0, qB);

        double currentTime = endTime;
        int obs = (int)QASAP.size() - 1;
        while (obs >= 0 && std::abs(QASAP[obs].time - currentTime) <= 1e-12) {
            apply_observation_jump(obs, yB);
            --obs;
        }

        HessianAdjointData adjData{ &rxnNet, work.data(), direction };
        int which = -1;
        flag = CVodeCreateB(cvode_mem, CV_BDF, &which);
        assert(flag == CV_SUCCESS);
        flag = CVodeInitBS(cvode_mem, which, hessianAdjointRhs, currentTime, yB);
        assert(flag == CV_SUCCESS);
        flag = CVodeSetUserDataB(cvode_mem, which, (void*)&adjData);
        assert(flag == CV_SUCCESS);
        flag = CVodeSStolerancesB(cvode_mem, which, cfg.tolRelError, cfg.tolAbsError);
        assert(flag == CV_SUCCESS);
        flag = CVodeSetMaxNumStepsB(cvode_mem, which, cfg.cvodeMaxNumSteps);
        assert(flag == CV_SUCCESS);

        SUNLinearSolver LSB = SUNLinSol_SPGMR(yB, SUN_PREC_NONE, 0, sunctx);
        assert(LSB != nullptr);
        flag = CVodeSetLinearSolverB(cvode_mem, which, LSB, nullptr);
        assert(flag == CV_SUCCESS);

        flag = CVodeQuadInitBS(cvode_mem, which, hessianAdjointQuadRhs, qB);
        assert(flag == CV_SUCCESS);
        flag = CVodeSetQuadErrConB(cvode_mem, which, SUNTRUE);
        assert(flag == CV_SUCCESS);
        flag = CVodeQuadSStolerancesB(cvode_mem, which, cfg.tolRelError, cfg.tolAbsError);
        assert(flag == CV_SUCCESS);

        while (obs >= 0) {
            const double targetTime = QASAP[obs].time;
            if (targetTime < currentTime) {
                flag = CVodeB(cvode_mem, targetTime, CV_NORMAL);
                assert(flag >= 0);
                flag = CVodeGetB(cvode_mem, which, &currentTime, yB);
                assert(flag == CV_SUCCESS);
                flag = CVodeGetQuadB(cvode_mem, which, &currentTime, qB);
                assert(flag == CV_SUCCESS);
            }

            while (obs >= 0 && std::abs(QASAP[obs].time - currentTime) <= 1e-12) {
                apply_observation_jump(obs, yB);
                --obs;
            }
            flag = CVodeReInitB(cvode_mem, which, currentTime, yB);
            assert(flag == CV_SUCCESS);
            flag = CVodeQuadReInitB(cvode_mem, which, qB);
            assert(flag == CV_SUCCESS);
        }

        if (currentTime > 0.0) {
            flag = CVodeB(cvode_mem, 0.0, CV_NORMAL);
            assert(flag >= 0);
            flag = CVodeGetQuadB(cvode_mem, which, &currentTime, qB);
            assert(flag == CV_SUCCESS);
        }

        const double* qData = N_VGetArrayPointer(qB);
        assert(qData != nullptr);
        for (int row = 0; row < n; ++row) {
            result[row][direction] = -qData[row];
        }

        SUNLinSolFree(LSB);
        N_VDestroy(qB);
        N_VDestroy(yB);
        CVodeAdjFree(cvode_mem);
        CVodeSensFree(cvode_mem);
        N_VDestroyVectorArray(uS, 1);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double sym = 0.5 * (result[i][j] + result[j][i]);
            result[i][j] = sym;
            result[j][i] = sym;
        }
    }
    return result;
}
