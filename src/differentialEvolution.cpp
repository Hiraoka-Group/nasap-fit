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

#include <Eigen/Dense>
#include <mpi.h>
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunlinsol/sunlinsol_klu.h>
#include <casadi/casadi.hpp>

#include "../include/differentialEvolution.hpp"
#include "../include/constants.hpp"
#include "../include/xorshift.hpp"
#include "../include/Jacf.hpp"
#include "../include/Rhsf.hpp"

xorshift myRand(2);
int cnt=0;
extern int num_procs, proc_rank;
    
using namespace casadi;
using std::vector;
using std::cout;
using std::endl;
using std::endl;
using std::flush;
using std::sqrt;



std::vector<double> differentialEvolution::crossingOver(const std::vector<double>& baseV, const std::vector<double>& randV1, const std::vector<double>& randV2){
    std::vector<double> v(cfg.constantSize);
    int jr = myRand(cfg.constantSize);
    for (int k = 0; k < cfg.constantSize; k++) {
        //交叉
        if (jr == k || myRand.prob() < cfg.crossOver) {
            v[k] = baseV[k] * pow(randV1[k] / randV2[k], cfg.scalar);
            v[k]=std::clamp(v[k], cfg.lowerLim, cfg.upperLim);
        }
        else v[k] = baseV[k];

        assert(std::isfinite(v[k]));
    }
    return v;
}

void differentialEvolution::sortByError(vector<differentialEvolution::OptimizeResult>& populations){
    sort(
        populations.begin(),
        populations.end(),
        [](const differentialEvolution::OptimizeResult& a, const differentialEvolution::OptimizeResult& b){
            return a.error < b.error;
        }
    );
}

void differentialEvolution::validateConstants(const vector<double>& constants){
    assert(constants.size() == (size_t)cfg.constantSize);
    for(double c : constants) assert(std::isfinite(c) && c > 0);
}

//平方残差和の計算
double differentialEvolution::calcError(const std::vector<double>& constant) {
    int flag=CV_SUCCESS;

    N_Vector y = N_VNew_Serial(cfg.species, sunctx);
    for (int i = 0; i < cfg.species; ++i) {
        NV_Ith_S(y, i) = initialState[i];
    }
    
    CVodeReInit(cvode_mem, 0.0, y);
    #if USE_PREGENERATED_RHSF || USE_PREGENERATED_JACOBIAN
        CVodeSetUserData(cvode_mem, (void*)constant.data());
    #else
        static_assert(std::is_same_v<double, double>, "Non-pregenerated CVODE callbacks require double constants");
        ReactionNetwork::CvodeUserData ud{ &rxnNet, constant.data() };
        CVodeSetUserData(cvode_mem, (void*)&ud);
    #endif
    
    double SSR = 0;
    double tret=0.0;
    for (int i = 0; i < QASAP.size(); i++) {
        double t = QASAP[i].time;
        assert(0 <= t && t <= endTime);
        if(t!=tret) flag=CVode(cvode_mem, t, y, &tret, CV_NORMAL);
        if(flag!=CV_SUCCESS){
            for(int j=0;j<cfg.constantSize;j++){
                cout<<constant[j]<<", ";
            }
            cout<<endl;
            exit(1);
        }
        double* y_data = N_VGetArrayPointer(y);
        for (int j = 0; j < cfg.trackedSpecies; j++) {
            size_t idx= (size_t)cfg.trackedIndex[j];
            assert(0 <= idx && idx < (size_t)cfg.species);
            SSR += (y_data[idx] / cfg.fullConc[j] - QASAP[i].state[j]) * (y_data[idx] / cfg.fullConc[j] - QASAP[i].state[j]);
        }
    }
    return SSR;
}

static Function build_integrator(std::string name,
						const std::vector<double>& tout,
						const differentialEvolution::Constants& cfg,
						const ReactionNetwork& rxnNet) {
    // 状態変数
    SX sp = SX::sym("sp", cfg.species);
    // パラメータ（反応速度定数）
    SX logk = SX::sym("logk", cfg.constantSize);

    // 反応速度
    SX xdot = SX::zeros(cfg.species);
    //ODE
    // Build xdot from ReactionNetwork::rhsTerms (mass-action style terms)
    for (const ReactionNetwork::RhsTerm &t : rxnNet.rhsTerms) {
        // start with the rate constant multiplier
        SX term;
        if(t.reactant2 != cfg.species)
            term = t.duplicacy * exp(logk(t.rateConstant)) * sp(t.reactant1) * sp(t.reactant2);
        else{
            term = t.duplicacy * exp(logk(t.rateConstant)) * sp(t.reactant1);
        }
        // add contribution to the target species (signed by t.coeff)
        xdot(t.add_to) = xdot(t.add_to) + term;
    }
    // DAE 定義
    SXDict dae;
    dae["x"] = sp;
    dae["p"] = logk;
    dae["ode"] = xdot;

    // integrator options
    Dict opts;
    opts["abstol"] = cfg.tolAbsError;
    opts["reltol"] = cfg.tolRelError;
    opts["max_num_steps"] = INT_MAX;

    return integrator(
        name,
        "cvodes",
        dae,
        0.0,          // t0
        tout,        
        opts
    );
};

void differentialEvolution::setUpCasADiFunctions() {
    // Levenberg-Marquardt refinement of populations[idx]
    std::vector<double> t_obs;//観測時間点

    MX x0 = MX::zeros(cfg.species);
    for (int i = 0; i < cfg.species; i++) {
        x0(i) = initialState[i];
    }
    for (const auto& datum : QASAP) {
        t_obs.push_back(datum.time);
    }
    integrator_ = build_integrator("F", t_obs, cfg, rxnNet);
    MX params = MX::sym("params", cfg.constantSize);
    MXDict arg;
    arg["p"]  = params;
    arg["x0"] = x0;
    
    MXDict res = integrator_(arg);
    MX xf(res.at("xf"));

    // 残差ベクトル r(theta)
    MX r = MX::zeros(QASAP.size() * cfg.trackedSpecies);
    // 平方残差和 SSR
    MX SSR = 0;
    for (int i = 0; i < QASAP.size(); i++) {
        MX x = xf(Slice(), i);
        for (int j = 0; j < cfg.trackedSpecies; j++) {
            MX sim_val = x(cfg.trackedIndex[j]) / cfg.fullConc[j];
            MX exp_val = QASAP[i].state[j];
            r(i * cfg.trackedSpecies + j) = sim_val - exp_val;
            SSR += r(i * cfg.trackedSpecies + j) * r(i * cfg.trackedSpecies + j);
        }
    }
    MX SSR_jac = jacobian(SSR, params);
    MX SSR_hes = hessian(SSR, params);
    MX J = jacobian(r, params);

    res_fun_ = Function("res_fun_", MXIList{params}, MXIList{r});
    jac_fun_ = Function("jac_fun_", MXIList{params}, MXIList{J});
    SSR_jac_fun_ = Function("SSR_jac_fun_", MXIList{params}, MXIList{SSR_jac});
    SSR_hes_fun_ = Function("SSR_hes_fun_", MXIList{params}, MXIList{SSR_hes});
    return;
}
//ヘッセ行列の計算
std::vector<std::vector<double>> differentialEvolution::getHessian(const std::vector<double>& point){
    validateConstants(point);
    // numeric central differences for Hessian
    int n = (int)point.size();
    std::vector<std::vector<double>> H(n, std::vector<double>(n, 0.0));
    std::vector<double> logPoint(n);
    for (int i = 0; i < n; ++i) {
        logPoint[i] = log(point[i]);
    }

    std::vector<DM> arg(1);
    arg[0] = DM(logPoint);
    cout<<"Calculating Hessian..."<<endl;
    DM hdm = SSR_hes_fun_(arg).at(0);
    cout<<" done."<<endl;
    std::vector<double> hvec = hdm.nonzeros();

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> Hmat(hvec.data(), n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            H[i][j] = Hmat(i, j);
        }
    }
    return H;
}

std::vector<std::vector<double>> differentialEvolution::getHessian_parallel(const std::vector<double>& point){
    validateConstants(point);
    int n = (int)point.size();
    std::vector<std::vector<double>> H(n, std::vector<double>(n, 0.0));
    std::vector<double> logPoint(n);
    for (int i = 0; i < n; ++i) {
        logPoint[i] = log(point[i]);
    }
    double eps = 1e-5;
    std::vector<DM> arg(1);
    DM dm;
    std::vector<double> vec;
    std::vector<double> buffer(2*n*n, 0.0);
    MPI_Win win;
    MPI_Win_create(buffer.data(), sizeof(double)*2*n*n, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    for(int i=proc_rank; i<2*n; i+=num_procs){
        int idx = i/2;
        bool isUpper = (i % 2 == 0);
        DM pert = DM::zeros(n);
        pert(idx) = eps;
        if(isUpper)arg[0] = DM(logPoint) + pert;
        else arg[0] = DM(logPoint) - pert;
        dm = SSR_jac_fun_(arg).at(0);
        vec = dm.nonzeros();
        MPI_Put(vec.data(), n, MPI_DOUBLE, 0, i*n, n, MPI_DOUBLE, win);
    }
    MPI_Win_fence(0, win);
    if(proc_rank==0){
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                double f_plus = buffer[(2*i)*n + j];
                double f_minus = buffer[(2*i+1)*n + j];
                H[i][j] = (f_plus - f_minus) / (2 * eps);
            }
        }
    }
    std::vector<double> hflat(n * n, 0.0);
    if (proc_rank == 0) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                hflat[i * n + j] = H[i][j];
            }
        }
    }
    MPI_Bcast(hflat.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (proc_rank != 0) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                H[i][j] = hflat[i * n + j];
            }
        }
    }
    MPI_Win_free(&win);
    return H;

}

std::vector<std::vector<double>> differentialEvolution::pseudoHessian(const std::vector<double>& point){
    validateConstants(point);
    const int n = cfg.constantSize;
    const int m = (int)QASAP.size() * cfg.trackedSpecies;
    std::vector<DM> arg(1);
    DM jdm;
    std::vector<double> jvec;
    Eigen::MatrixXd J;
    Eigen::MatrixXd A;
    std::vector<double> logTheta(n);
    for (int i = 0; i < n; ++i) {
        logTheta[i] = std::log(point[i]);
    }
    arg[0] = DM(logTheta);
    jdm = jac_fun_(arg).at(0);
    jvec = jdm.nonzeros();
    J = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(jvec.data(), m, n);
    A = J.transpose() * J; 
    std::vector<std::vector<double>> H(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            H[i][j] = A(i, j);
        }
    }
    return H;
}

//実験データのセット
void differentialEvolution::setQASAPData(std::vector<std::vector<double>>& arg) {
    for (const std::vector<double>& vec : arg) {
        assert((int)vec.size() == cfg.trackedSpecies + 1);
        QASAP.push_back({ vec[0],std::vector<double>({vec.begin()+1, vec.begin()+cfg.trackedSpecies+1}) });
    }
}

differentialEvolution::differentialEvolution(vector<vector<double>>& arg) {
    myRand=xorshift(1+proc_rank);

    // copy config constants into instance-owned cfg
    cfg.QASAPFile = config::QASAPFile;
    cfg.reactNetworkFile = config::reactNetworkFile;
    cfg.species = config::species;
    cfg.constantSize = config::constantSize;
    cfg.trackedSpecies = config::trackedSpecies;
    cfg.trackedNames.assign(std::begin(config::trackedNames), std::end(config::trackedNames));
    cfg.trackedIndex.assign(std::begin(config::trackedIndex), std::end(config::trackedIndex));
    cfg.fullConc.assign(std::begin(config::fullConc), std::end(config::fullConc));
    cfg.tolAbsError = config::tolAbsError;
    cfg.tolRelError = config::tolRelError;
    cfg.scalar = config::scalar;
    cfg.crossOver = config::crossOver;
    cfg.upperLim = config::upperLim;
    cfg.lowerLim = config::lowerLim;

    setQASAPData(arg);
    // allocate initialState according to runtime species
    initialState.assign(cfg.species, 0.0);
    for (int i = 0; i < cfg.trackedSpecies; i++) {
        initialState[cfg.trackedIndex[i]] = cfg.fullConc[i] * QASAP[0].state[i];
    }
    endTime = arg.back()[0];

    rxnNet.build(cfg.reactNetworkFile, cfg.species, cfg.constantSize);

    //setting up jac_fun_ and res_fun_
    setUpCasADiFunctions();
    
    //seting up CVODE
    N_Vector y = N_VNew_Serial(cfg.species, sunctx);
    for (int i = 0; i < cfg.species; ++i) {
        NV_Ith_S(y, i) = initialState[i];
    }
    
    #if USE_PREGENERATED_RHSF
        int flag = CVodeInit(cvode_mem, rhsf, 0.0, y);
    #else
        int flag = CVodeInit(cvode_mem, ReactionNetwork::rhsfCb, 0.0, y);
    #endif
    assert(flag == CV_SUCCESS);

    flag = CVodeSStolerances(cvode_mem, cfg.tolRelError, cfg.tolAbsError);
    assert(flag == CV_SUCCESS);

    #if USE_PREGENERATED_JACOBIAN
        SUNMatrix J = SUNSparseMatrix(cfg.species, cfg.species, nonZeroElems, CSC_MAT, sunctx);
        SUNLinearSolver LS = SUNLinSol_KLU(y, J, sunctx);
        flag = CVodeSetLinearSolver(cvode_mem, LS, J);
        assert(flag == CV_SUCCESS);
        flag = CVodeSetJacFn(cvode_mem, JacFn);
    #else
        SUNMatrix J = SUNSparseMatrix(cfg.species, cfg.species, rxnNet.jacNonZeros, CSC_MAT, sunctx);
        SUNLinearSolver LS = SUNLinSol_KLU(y, J, sunctx);
        flag = CVodeSetLinearSolver(cvode_mem, LS, J);
        assert(flag == CV_SUCCESS);
        flag = CVodeSetJacFn(cvode_mem, ReactionNetwork::JacFnCb);
    #endif
    assert(flag == CV_SUCCESS);

    flag = CVodeSetMaxNumSteps(cvode_mem, 10000);
    assert(flag == CV_SUCCESS);
}

std::vector<differentialEvolution::OptimizeResult> differentialEvolution::Optimize(int maxGen, int popSize, double lowerLim, double upperLim) {
    auto transProb = [&](double temp, double oldE, double newE) {
        if (newE < oldE) return true;
        if (temp <= 0) return false;
        return myRand.prob() < exp((oldE - newE) / temp);
    };
    assert(maxGen > 0);
    assert(popSize > 0);
    assert(lowerLim > 0 && upperLim > 0 && lowerLim < upperLim);

    std::vector<OptimizeResult> populations;
    populations.reserve((size_t)popSize);
    for (int i = 0; i < popSize; ++i) populations.emplace_back(cfg.constantSize);
    std::vector<double> populationsFlat((size_t)popSize * (size_t)cfg.constantSize, 0.0);
    std::vector<double> populationsErrorFlat((size_t)popSize, DBL_MAX);

    // initialize owned individuals only
    for (int idx = proc_rank; idx < popSize; idx += num_procs) {
        for (int j = 0; j < cfg.constantSize; ++j) {
            double v = randbetExp(lowerLim, upperLim);
            populations[idx].constants[j] = v;
            populationsFlat[(size_t)idx * (size_t)cfg.constantSize + (size_t)j] = v;
        }
        populations[idx].error = calcError(populations[idx].constants);
        populationsErrorFlat[(size_t)idx] = populations[idx].error;
    }

    MPI_Win winConst, winErr;
    MPI_Win_create(
        populationsFlat.data(),
        (MPI_Aint)(sizeof(double) * populationsFlat.size()),
        (int)sizeof(double),
        MPI_INFO_NULL,
        MPI_COMM_WORLD,
        &winConst);
    MPI_Win_create(
        populationsErrorFlat.data(),
        (MPI_Aint)(sizeof(double) * populationsErrorFlat.size()),
        (int)sizeof(double),
        MPI_INFO_NULL,
        MPI_COMM_WORLD,
        &winErr);

    struct Triple { int j, xb, xr1, xr2; };

    for (int gen = 0; gen < maxGen; ++gen) {
        if (proc_rank == 0) cout << "current generation : " << gen << " / " << maxGen << "\n";
        double temp = 0;

        std::vector<Triple> triples;
        triples.reserve((size_t)((popSize + num_procs - 1) / num_procs));

        std::vector<int> remoteIndices;
        remoteIndices.reserve((size_t)popSize);
        auto needRemote = [&](int idx) {
            if (idx % num_procs != proc_rank) remoteIndices.push_back(idx);
        };

        for (int j = proc_rank; j < popSize; j += num_procs) {
            int xb = myRand(popSize), xr1 = myRand(popSize), xr2 = myRand(popSize);
            while (xb == xr1) { xr1 = myRand(popSize); }
            while (xb == xr2 || xr1 == xr2) { xr2 = myRand(popSize); }
            triples.push_back({j, xb, xr1, xr2});
            needRemote(xb);
            needRemote(xr1);
            needRemote(xr2);
        }

        std::sort(remoteIndices.begin(), 
            remoteIndices.end(),
            [](int left, int right){
                if (left % num_procs != right % num_procs) return (left % num_procs) < (right % num_procs);
                return left < right;
            });
        remoteIndices.erase(std::unique(remoteIndices.begin(), remoteIndices.end()), remoteIndices.end());

        MPI_Win_fence(0, winConst);
        for (int idx : remoteIndices) {
            int owner = idx % num_procs;
            MPI_Get(
                populations[idx].constants.data(),
                cfg.constantSize,
                MPI_DOUBLE,
                owner,
                (MPI_Aint)idx * (MPI_Aint)cfg.constantSize,
                cfg.constantSize,
                MPI_DOUBLE,
                winConst);
        }
        MPI_Win_fence(0, winConst);

        for (const auto& t : triples) {
            const std::vector<double>& baseV = populations[t.xb].constants;
            const std::vector<double>& randV1 = populations[t.xr1].constants;
            const std::vector<double>& randV2 = populations[t.xr2].constants;
            std::vector<double> v = crossingOver(baseV, randV1, randV2);
            double newError = calcError(v);
            if (transProb(temp, populations[t.j].error, newError) || !std::isfinite(populations[t.j].error)) {
                populations[t.j].constants = std::move(v);
                populations[t.j].error = newError;
                for (int kk = 0; kk < cfg.constantSize; ++kk) {
                    populationsFlat[(size_t)t.j * (size_t)cfg.constantSize + (size_t)kk] = populations[t.j].constants[kk];
                }
            }
        }
    }

    for (int j = proc_rank; j < popSize; j += num_procs){
        populationsErrorFlat[(size_t)j] = populations[j].error;
    }
    // broadcast all constants & errors
    for (int i=0; i<popSize; ++i) {
        int owner = i % num_procs;
        MPI_Bcast(populationsFlat.data() + (size_t)i * (size_t)cfg.constantSize, cfg.constantSize, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Bcast(&populationsErrorFlat[(size_t)i], 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        populations[i].constants.assign(populationsFlat.begin() + (size_t)i * (size_t)cfg.constantSize, populationsFlat.begin() + ((size_t)i + 1) * (size_t)cfg.constantSize);
        populations[i].error = populationsErrorFlat[(size_t)i];
    }
    sortByError(populations);

    return populations;
}

std::vector<differentialEvolution::OptimizeResult> differentialEvolution::Optimize(std::vector<std::vector<double>> arg){
    auto transProb = [&](double temp, double oldE, double newE) {
        if (newE < oldE) return true;
        if (temp <= 0) return false;
        return myRand.prob() < exp((oldE - newE) / temp);
    };
    const int n = arg.size();

    for(int i = 0; i < n; ++i) {
        validateConstants(arg[i]);
    }

    std::vector<OptimizeResult> populations;
    populations.reserve((size_t)n);
    for (int i = 0; i < n; ++i) populations.emplace_back(cfg.constantSize);
    std::vector<double> populationsFlat((size_t)n * (size_t)cfg.constantSize, 0.0);
    std::vector<double> populationsErrorFlat((size_t)n, DBL_MAX);

    // initialize with provided arg
    for (int idx = proc_rank; idx < n; idx += num_procs) {
        for (int j = 0; j < cfg.constantSize; ++j) {
            populations[idx].constants[j] = arg[idx][j];
            populationsFlat[(size_t)idx * (size_t)cfg.constantSize + (size_t)j] = arg[idx][j];
        }
        populations[idx].error = calcError(populations[idx].constants);
        populationsErrorFlat[(size_t)idx] = populations[idx].error;
    }

    MPI_Win winConst, winErr;
    MPI_Win_create(
        populationsFlat.data(),
        (MPI_Aint)(sizeof(double) * populationsFlat.size()),
        (int)sizeof(double),
        MPI_INFO_NULL,
        MPI_COMM_WORLD,
        &winConst);
    MPI_Win_create(
        populationsErrorFlat.data(),
        (MPI_Aint)(sizeof(double) * populationsErrorFlat.size()),
        (int)sizeof(double),
        MPI_INFO_NULL,
        MPI_COMM_WORLD,
        &winErr);

    struct Triple { int j, xb, xr1, xr2; };
    int maxGen = 100;

    for (int gen = 0; gen < maxGen; ++gen) {
        if (proc_rank == 0) cout << "current generation : " << gen << " / " << maxGen << "\n";
        double temp = 0;

        std::vector<Triple> triples;
        triples.reserve((size_t)((n + num_procs - 1) / num_procs));

        std::vector<int> remoteIndices;
        remoteIndices.reserve((size_t)n);
        auto needRemote = [&](int idx) {
            if (idx % num_procs != proc_rank) remoteIndices.push_back(idx);
        };

        for (int j = proc_rank; j < n; j += num_procs) {
            int xb = myRand(n), xr1 = myRand(n), xr2 = myRand(n);
            while (xb == xr1) { xr1 = myRand(n); }
            while (xb == xr2 || xr1 == xr2) { xr2 = myRand(n); }
            triples.push_back({j, xb, xr1, xr2});
            needRemote(xb);
            needRemote(xr1);
            needRemote(xr2);
        }

        std::sort(remoteIndices.begin(), 
            remoteIndices.end(),
            [](int left, int right){
                if (left % num_procs != right % num_procs) return (left % num_procs) < (right % num_procs);
                return left < right;
            });
        remoteIndices.erase(std::unique(remoteIndices.begin(), remoteIndices.end()), remoteIndices.end());

        MPI_Win_fence(0, winConst);
        for (int idx : remoteIndices) {
            int owner = idx % num_procs;
            MPI_Get(
                populations[idx].constants.data(),
                cfg.constantSize,
                MPI_DOUBLE,
                owner,
                (MPI_Aint)idx * (MPI_Aint)cfg.constantSize,
                cfg.constantSize,
                MPI_DOUBLE,
                winConst);
        }
        MPI_Win_fence(0, winConst);

        for (const auto& t : triples) {
            const std::vector<double>& baseV = populations[t.xb].constants;
            const std::vector<double>& randV1 = populations[t.xr1].constants;
            const std::vector<double>& randV2 = populations[t.xr2].constants;
            std::vector<double> v = crossingOver(baseV, randV1, randV2);
            double newError = calcError(v);
            if (transProb(temp, populations[t.j].error, newError) || !std::isfinite(populations[t.j].error)) {
                populations[t.j].constants = std::move(v);
                populations[t.j].error = newError;
                for (int kk = 0; kk < cfg.constantSize; ++kk) {
                    populationsFlat[(size_t)t.j * (size_t)cfg.constantSize + (size_t)kk] = populations[t.j].constants[kk];
                }
            }
        }
    }

    for (int j = proc_rank; j < n; j += num_procs){
        populationsErrorFlat[(size_t)j] = populations[j].error;
    }
    // broadcast all constants & errors
    for (int i=0; i<n; ++i) {
        int owner = i % num_procs;
        MPI_Bcast(populationsFlat.data() + (size_t)i * (size_t)cfg.constantSize, cfg.constantSize, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Bcast(&populationsErrorFlat[(size_t)i], 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        populations[i].constants.assign(populationsFlat.begin() + (size_t)i * (size_t)cfg.constantSize, populationsFlat.begin() + ((size_t)i + 1) * (size_t)cfg.constantSize);
        populations[i].error = populationsErrorFlat[(size_t)i];
    }
    sortByError(populations);

    MPI_Win_free(&winConst);
    MPI_Win_free(&winErr);

    return populations;
}


differentialEvolution::OptimizeResult differentialEvolution::runLM(const std::vector<double>& theta0){
    const int n = cfg.constantSize;
    const int m = (int)QASAP.size() * cfg.trackedSpecies;
    validateConstants(theta0);

    std::vector<double> theta = theta0;
    std::vector<double> logTheta(n);
    for (int i = 0; i < n; ++i) logTheta[i] = log(theta[i]);

    double bestErr = calcError(theta);
    if (!std::isfinite(bestErr)) bestErr = DBL_MAX;

    double lambda = 10.0;
    const int maxIter = 200;

    std::vector<DM> arg(1);
    DM rdm, jdm;
    std::vector<double> rvec, jvec;
    Eigen::VectorXd r;
    Eigen::MatrixXd J;
    Eigen::MatrixXd A;
    Eigen::VectorXd g;
    bool isChanged = true;

    for (int iter = 0; iter < maxIter; ++iter) {
        if (proc_rank == 0) {
            for (int i = 0; i < cfg.constantSize; i++) {
                cout << std::setprecision(4) << theta[i] << ", ";
            }
            cout << endl;
            cout << "Current error: " << std::setprecision(6) << bestErr << endl;
            cout << "Iter : " << iter << ", Lambda: " << std::setprecision(6) << lambda << endl;
        }

        if (isChanged) {
            arg[0] = DM(logTheta);
            rdm = res_fun_(arg).at(0);
            jdm = jac_fun_(arg).at(0);
            rvec = rdm.nonzeros();
            jvec = jdm.nonzeros();
            r = Eigen::Map<Eigen::VectorXd>(rvec.data(), m);
            J = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(jvec.data(), m, n);
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
        double actualImprove = bestErr - calcError(trial);
        double reliabilityIndex = actualImprove / predictedImprove;

        double newErr = calcError(trial);
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
            if (delta.norm() < 1e-6) break;
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

vector<differentialEvolution::OptimizeResult> differentialEvolution::runLM(const vector<vector<double>>& thetaList){
    const size_t n = thetaList.size();
    vector<OptimizeResult> results(n, OptimizeResult(cfg.constantSize));
    std::vector<double> populationsFlat(n * (size_t)cfg.constantSize, 0.0);
    std::vector<double> populationsErrorFlat(n, DBL_MAX);
    for(int i=0;i<n;i++){
        validateConstants(thetaList[i]);
        for(int j=0;j<cfg.constantSize;j++){
            results[i].constants[j]=thetaList[i][j];
            populationsFlat[(size_t)i * (size_t)cfg.constantSize + (size_t)j] = thetaList[i][j];
        }
    }

    for(int i=proc_rank;i<n;i+=num_procs){
        results[i] = runLM(results[i].constants);
        for(int j=0;j<cfg.constantSize;j++){
            populationsFlat[(size_t)i * (size_t)cfg.constantSize + (size_t)j] = results[i].constants[j];
        }
        populationsErrorFlat[i] = results[i].error;
    }
    for(int i=0; i<n; i++){
        MPI_Bcast(populationsFlat.data() + (size_t)i * (size_t)cfg.constantSize, cfg.constantSize, MPI_DOUBLE, i % num_procs, MPI_COMM_WORLD);
        MPI_Bcast(&populationsErrorFlat[(size_t)i], 1, MPI_DOUBLE, i % num_procs, MPI_COMM_WORLD);
        results[i].constants.assign(populationsFlat.begin() + (size_t)i * (size_t)cfg.constantSize, populationsFlat.begin() + ((size_t)i + 1) * (size_t)cfg.constantSize);
        results[i].error = populationsErrorFlat[(size_t)i];
    }
    return results;
}



void differentialEvolution::putCVODESim(const std::vector<double>& constant) {
    if(proc_rank!=0)return;
    N_Vector y = N_VNew_Serial(cfg.species, sunctx);
    for (int i = 0; i < cfg.species; ++i) {
        NV_Ith_S(y, i) = initialState[i];
    }
    CVodeReInit(cvode_mem, 0.0, y);
    #if USE_PREGENERATED_RHSF || USE_PREGENERATED_JACOBIAN
        CVodeSetUserData(cvode_mem, (void*)constant.data());
    #else
        ReactionNetwork::CvodeUserData ud{ &rxnNet, constant.data() };
        CVodeSetUserData(cvode_mem, (void*)&ud);
    #endif
    
    cout <<"Time (min),1 (%),[PdPy*4]2+ (%),Py* (%),Pd214 cage (%),"<<std::endl;
    bool flag=CV_SUCCESS;
    double tret=0.0;
    for (int i = 0; i < QASAP.size(); i++) {
        double t = QASAP[i].time;
        assert(0 <= t && t <= endTime);
        cout<<t<<", ";
        if(!(tret-0.1<t&&t<tret+0.1)) flag=CVode(cvode_mem, t, y, &tret, CV_NORMAL);
        assert(flag >= 0);

        double* y_data = N_VGetArrayPointer(y);
        for (int j = 0; j < cfg.trackedSpecies; j++) {
            size_t idx= (size_t)cfg.trackedIndex[j];
            assert(0 <= idx && idx < (size_t)cfg.species);
            cout<<y_data[idx]/cfg.fullConc[j] <<", ";
        }
        cout<<std::endl;
    }

}

