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
    for (int i = 0; i < (int)QASAP.size(); ++i) {
        const double t = QASAP[i].time;
        assert(0 <= t && t <= endTime);
        if (t != tret) {
            flag = CVode(cvode_mem, t, y, &tret, CV_NORMAL);
            assert(flag >= 0);
        }
        flag = CVodeGetSens(cvode_mem, &tret, uS);
        assert(flag == CV_SUCCESS);

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


NASAP_fit::OptimizeResult NASAP_fit::runLM(const std::vector<double>& theta0, const TerminationCondition& termCond){
    const int world_rank = mpi_env.rank();
    const int n = cfg.constantSize;
    validateConstants(theta0);

    std::vector<double> theta = theta0;
    std::vector<double> logTheta(n);
    for (int i = 0; i < n; ++i) logTheta[i] = log(theta[i]);

    double bestErr = calcError(theta);
    if (!std::isfinite(bestErr)) bestErr = DBL_MAX;

    double lambda = 10.0;
    const int maxIter = (termCond.maxIter > 0) ? termCond.maxIter : 200;
    double bestSoFar = bestErr;
    int stallCount = 0;

    Eigen::VectorXd r;
    Eigen::MatrixXd J;
    Eigen::MatrixXd A;
    Eigen::VectorXd g;
    bool isChanged = true;

    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    for (int iter = 0; ; ++iter) {

        if (maxIter > 0 && iter >= maxIter) {
            if(world_rank == 0 && cfg.logLevel != LogLevel::quiet) {
                cout << "Maximum iterations reached: " << iter << " >= " << maxIter << endl;
            }
            break;
        } 
        if (termCond.timeLimit > 0) {
            double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - startTime).count();
            if (elapsed >= termCond.timeLimit) {
                if(world_rank == 0 && cfg.logLevel != LogLevel::quiet) {
                    cout << "Time limit reached: " << std::setprecision(10) << elapsed << " >= " << termCond.timeLimit << "\n";
                }
                break;
            }
        }
        if (termCond.targetError > 0 && bestErr <= termCond.targetError) {
            if(world_rank == 0 && cfg.logLevel != LogLevel::quiet) {
                cout << "Target error reached: " << bestErr << " <= " << termCond.targetError << "\n";
            }
            break;
        }

        if (world_rank == 0 && cfg.logLevel != LogLevel::quiet) {
            if (cfg.logLevel == LogLevel::verbose) {
                for (int i = 0; i < cfg.constantSize; i++) {
                    cout << std::setprecision(4) << theta[i] << ", ";
                }
                cout << endl;
                cout << "Current error: " << std::setprecision(10) << bestErr << endl;
                cout << "Iter : " << iter << ", Lambda: " << std::setprecision(10) << lambda << endl;
            } else {
                cout << "Iter : " << iter << " / " << maxIter
                     << ", error: " << std::setprecision(10) << bestErr << endl;
            }
        }

        if (isChanged) {
            computeLMResAndJac(theta, r, J);
            A = J.transpose() * J;
            g = J.transpose() * r;
        }
        Eigen::MatrixXd A_try = A;
        A_try.diagonal().array() += lambda;
        Eigen::VectorXd delta = -A_try.ldlt().solve(g);
        if (!delta.allFinite()) break;

        std::vector<double> trial(n), logTrial(n);
        for (int i = 0; i < n; ++i) {
            logTrial[i] = std::clamp(logTheta[i] + delta[i], log(cfg.lowerLim), log(cfg.upperLim));
            //logTrial[i] = logTheta[i] + delta[i];
            trial[i] = exp(logTrial[i]);
        }

        double predictedImprove = -(g.dot(delta) + 0.5 * delta.dot(A_try * delta));
        double newErr = calcError(trial);
        double actualImprove = bestErr - newErr;
        double reliabilityIndex = actualImprove / predictedImprove;

        if (newErr < bestErr) {
            isChanged = true;
            theta = std::move(trial);
            logTheta = std::move(logTrial);
            bestErr = newErr;
            if (reliabilityIndex > 0.75) {
                lambda /= 3.0;
            } else if (reliabilityIndex < 0.25) {
                lambda *= 2.0;
            }
            const double stepNorm = delta.norm();
            if (stepNorm < termCond.xtol) {
                if(world_rank == 0 && cfg.logLevel != LogLevel::quiet) {
                    cout << "XTOL reached: " << stepNorm << " < " << termCond.xtol << "\n";
                }
                break;
            }

            // ftol / stall
            if (termCond.stall > 0 || termCond.ftolAbs > 0 || termCond.ftolRel > 0) {
                //const double absImprove = bestSoFar - bestErr;
                //const double relImprove = (bestSoFar > 0) ? (absImprove / bestSoFar) : DBL_MAX;
                //const bool improvedAbs = (termCond.ftolAbs > 0) ? (absImprove > termCond.ftolAbs) : (absImprove > 0);
                //const bool improvedRel = (termCond.ftolRel > 0) ? (relImprove > termCond.ftolRel) : true;
                if (std::abs(bestSoFar - newErr) > termCond.ftolAbs + termCond.ftolRel * std::abs(newErr)) {
                    bestSoFar = bestErr;
                    stallCount = 0;
                } else {
                    stallCount++;
                }
                if (termCond.stall > 0 && stallCount >= termCond.stall){
                    if(world_rank == 0 && cfg.logLevel != LogLevel::quiet) {
                        cout << "Stall limit reached: " << stallCount << " >= " << termCond.stall << "\n";
                    }
                    break;
                }
            }
        } else {
            isChanged = false;
            lambda *= 2.0;
        }
    }
    OptimizeResult result(cfg.constantSize);
    result.constants = std::move(theta);
    result.error = bestErr;
    return result;
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
