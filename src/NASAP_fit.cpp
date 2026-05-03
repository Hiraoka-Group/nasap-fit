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
using std::flush;
using std::sqrt;



std::vector<double> NASAP_fit::crossingOver(
    const std::vector<double>& baseV,
    const std::vector<double>& randV1,
    const std::vector<double>& randV2,
    uint64_t seed,
    int gen,
    int j) {
    std::vector<double> v(cfg.constantSize);

	// Deterministic crossover randomness derived from (seed, gen, j, k).
	constexpr uint64_t TAG_JR = 0xD3B54A1Du;
	constexpr uint64_t TAG_CROSS = 0xBEEF1234u;
	int jr = (int)det_rng::uniform_index(det_rng::hash64(seed, (uint64_t)gen, (uint64_t)j, TAG_JR), (uint32_t)cfg.constantSize);
    for (int k = 0; k < cfg.constantSize; k++) {
        //交叉
		double u = det_rng::u01_from_u64(det_rng::hash64(seed, (uint64_t)gen, (uint64_t)j, (uint64_t)k, TAG_CROSS));
		if (jr == k || u < cfg.crossOver) {
            v[k] = baseV[k] * pow(randV1[k] / randV2[k], cfg.scalar);
            v[k]=std::clamp(v[k], cfg.lowerLim, cfg.upperLim);
        }
        else v[k] = baseV[k];

        assert(std::isfinite(v[k]));
    }
    return v;
}

void NASAP_fit::sortByError(vector<NASAP_fit::OptimizeResult>& populations){
    sort(
        populations.begin(),
        populations.end(),
        [](const NASAP_fit::OptimizeResult& a, const NASAP_fit::OptimizeResult& b){
            return a.error < b.error;
        }
    );
}

void NASAP_fit::validateConstants(const vector<double>& constants){
    assert(constants.size() == (size_t)cfg.constantSize);
    for(double c : constants) assert(std::isfinite(c) && c > 0);
}

//平方残差和の計算
double NASAP_fit::calcError(const std::vector<double>& constant) {
    int flag=CV_SUCCESS;

    for (int i = 0; i < cfg.species; ++i) {
        NV_Ith_S(y, i) = initialState[i];
    }
    CVodeReInit(cvode_mem, 0.0, y);
    CVodeQuadReInit(cvode_mem, yQ0);
    ReactionNetwork::CvodeUserData ud{ &rxnNet, constant.data(), nullptr };
    CVodeSetUserData(cvode_mem, (void*)&ud);
    
    double SSR = 0;
    double tret=0.0;
    for (int i = 0; i < QASAP.size(); i++) {
        double t = QASAP[i].time;
        assert(0 <= t && t <= endTime);
        if(t!=tret) flag=CVode(cvode_mem, t, y, &tret, CV_NORMAL);
        if(flag!=CV_SUCCESS){
            // Always print an error message, regardless of log level.
            std::cerr << "CVode failed (flag=" << flag << ") at t=" << t
                      << " (rank=" << mpi_env.rank() << ")\n";
            if (cfg.logLevel == LogLevel::verbose) {
                for(int j=0;j<cfg.constantSize;j++){
                    std::cerr << constant[j] << ", ";
                }
                std::cerr << "\n";
            }
            exit(1);
        }
        double* y_data = N_VGetArrayPointer(y);
        for (int j = 0; j < cfg.trackedSpecies; j++) {
            size_t idx= (size_t)indexOrder[j];
            assert(0 <= idx && idx < (size_t)cfg.species);
            SSR += (y_data[idx] / cfg.fullConc[j] - QASAP[i].state[j]/100) * (y_data[idx] / cfg.fullConc[j] - QASAP[i].state[j]/100);
        }
    }
    return SSR;
}

std::vector<std::vector<double>> NASAP_fit::makeRandomPopulation(int popSize, double lower, double upper, uint64_t seed) {
    std::vector<std::vector<double>> population(popSize);
    xorshift rand(seed);
    for (int i = 0; i < popSize; i++) {
        population[i].resize(cfg.constantSize);
        for (int j = 0; j < cfg.constantSize; j++) {
            population[i][j] = rand.randbetExp(lower, upper);
        }
    }
    return population;
}
/*
static Function build_integrator(std::string name,
						const std::vector<double>& tout,
						const NASAP_fit::Config& cfg,
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

void NASAP_fit::setUpCasADiFunctions() {
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
            MX exp_val = QASAP[i].state[j]/100;
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
    */


//実験データのセット
void NASAP_fit::setQASAPData(const vector<vector<std::string>>& arg) {
    QASAP.clear();
    indexOrder.clear();
	bool isHeader = true;
	for (const auto& row : arg) {
        assert((size_t)cfg.trackedSpecies + 1 <= row.size());
		std::vector<double> row_double;
		if(isHeader){
            isHeader = false;
            for(int i=1;i<=cfg.trackedSpecies;i++){
                auto it = std::find(cfg.trackedNames.begin(), cfg.trackedNames.end(), row[i]);
                assert(it != cfg.trackedNames.end());
                int idx = std::distance(cfg.trackedNames.begin(), it);
                indexOrder.push_back(cfg.trackedIndex[idx]);
            }
        }
        else{
            for(int i=0;i<=cfg.trackedSpecies;i++){
                row_double.push_back(std::stod(row[i]));
                assert(std::stod(row[i]) >= 0);
            }
            QASAP.push_back({ row_double[0],std::vector<double>({row_double.begin()+1, row_double.begin()+cfg.trackedSpecies+1}) });
        }
	}
    endTime = QASAP.back().time;

}

NASAP_fit::NASAP_fit(const Config& arg): cfg(arg) {

    // allocate initialState according to runtime species
    initialState.assign(cfg.species, 0.0);
    for (auto p: cfg.initConc) {
        assert(0 <= p.first && p.first < cfg.species);
        assert(p.second >= 0);
        initialState[p.first] = p.second;
    }

    rxnNet.build(cfg.reactNetworkFile, cfg.species, cfg.constantSize);
    setQASAPData(read_csv(cfg.QASAPFile));
    // setUpCasADiFunctions(); // CasADi 依存を除去
    
    //seting up CVODE
    y = N_VNew_Serial(cfg.species, sunctx);

    yQ0 = N_VNew_Serial(rxnNet.data.size(), sunctx);

    for (int i = 0; i < cfg.species; ++i) {
        NV_Ith_S(y, i) = initialState[i];
    }
    
    int flag = CVodeInit(cvode_mem, ReactionNetwork::rhsfCb, 0.0, y);
    assert(flag == CV_SUCCESS);

    flag = CVodeSStolerances(cvode_mem, cfg.tolRelError, cfg.tolAbsError);
    assert(flag == CV_SUCCESS);

    // Enforce non-negativity on all state variables: y_i >= 0.
    // IMPORTANT: the constraint vector must remain alive for the lifetime of cvode_mem.
    //cvode_constraints_ = N_VNew_Serial(cfg.species, sunctx);
    //for (int i = 0; i < cfg.species; ++i) {
    //    NV_Ith_S(cvode_constraints_, i) = 1.0;
    //}
    //flag = CVodeSetConstraints(cvode_mem, cvode_constraints_);
    //assert(flag == CV_SUCCESS);

    J = SUNSparseMatrix(cfg.species, cfg.species, rxnNet.jacNonZeros, CSC_MAT, sunctx);
    LS = SUNLinSol_KLU(y, J, sunctx);
    flag = CVodeSetLinearSolver(cvode_mem, LS, J);
    assert(flag == CV_SUCCESS);
    flag = CVodeSetJacFn(cvode_mem, ReactionNetwork::JacFnCb);
    assert(flag == CV_SUCCESS);

    flag = CVodeQuadInit(cvode_mem, ReactionNetwork::quadRhsCb, yQ0);
    assert(flag == CV_SUCCESS);

    flag = CVodeSetMaxNumSteps(cvode_mem, cfg.cvodeMaxNumSteps);
    assert(flag == CV_SUCCESS);
}

NASAP_fit::~NASAP_fit() {
    CVodeFree(&cvode_mem);
    SUNLinSolFree(LS);
    SUNMatDestroy(J);
    N_VDestroy(y);
    N_VDestroy(yQ0);
}

std::vector<NASAP_fit::OptimizeResult> NASAP_fit::runDE_single(int maxGen, int popSize, double lowerLim, double upperLim, const TerminationCondition& termCond, uint64_t seed) {
    assert(popSize >= 3);
	TerminationCondition tc = termCond;
	tc.maxIter = std::min(tc.maxIter, maxGen);
    std::vector<std::vector<double>> init = makeRandomPopulation(popSize, lowerLim, upperLim, seed);
    return runDE(std::move(init), tc);
}

std::vector<NASAP_fit::OptimizeResult> NASAP_fit::runDE_single(const std::vector<std::vector<double>>& arg, const TerminationCondition& termCond, uint64_t seed) {
    const int world_size = mpi_env.size();
    const int world_rank = mpi_env.rank();
    assert(world_size == 1);

    auto transProb = [&](double temp, double oldE, double newE) {
        if (newE < oldE) return true;
        if (temp <= 0) return false;
        return false;
        //return myRand.prob() < exp((oldE - newE) / temp);
    };

    const int n = (int)arg.size();
    for (int i = 0; i < n; ++i) validateConstants(arg[i]);

    std::vector<OptimizeResult> populations;
    populations.reserve(n);
    for (int i = 0; i < n; ++i) populations.emplace_back(cfg.constantSize);

    for (int idx = 0; idx < n; ++idx) {
        populations[idx].constants = arg[idx];
        populations[idx].error = calcError(populations[idx].constants);
    }


	struct Triple { int j, xb, xr1, xr2; };
    int maxGen = 100;
    double bestSoFar = DBL_MAX;
    int stallCount = 0;
    auto startTime = std::chrono::steady_clock::now();

        for (int gen = 0; gen < termCond.maxIter; ++gen) {
		if (cfg.logLevel != LogLevel::quiet) {
			cout << "current generation : " << gen << " / " << termCond.maxIter << "\n";
		}
        double temp = 0;

        std::vector<Triple> triples;
        triples.reserve((size_t)n);
        constexpr uint64_t TAG_XB  = 0xA11CE001u;
        constexpr uint64_t TAG_XR1 = 0xA11CE002u;
        constexpr uint64_t TAG_XR2 = 0xA11CE003u;
        for (int j = 0; j < n; ++j) {
            int xb = (int)det_rng::uniform_index(det_rng::hash64(seed, (uint64_t)gen, (uint64_t)j, TAG_XB), (uint32_t)n);
            int xr1 = (int)det_rng::uniform_index(det_rng::hash64(seed, (uint64_t)gen, (uint64_t)j, TAG_XR1), (uint32_t)n);
            for (int attempt = 0; xr1 == xb; ++attempt) {
                xr1 = (int)det_rng::uniform_index(det_rng::hash64(seed, (uint64_t)gen, (uint64_t)j, TAG_XR1, (uint64_t)attempt), (uint32_t)n);
            }
            int xr2 = (int)det_rng::uniform_index(det_rng::hash64(seed, (uint64_t)gen, (uint64_t)j, TAG_XR2), (uint32_t)n);
            for (int attempt = 0; (xr2 == xb || xr2 == xr1); ++attempt) {
                xr2 = (int)det_rng::uniform_index(det_rng::hash64(seed, (uint64_t)gen, (uint64_t)j, TAG_XR2, (uint64_t)attempt), (uint32_t)n);
            }
            triples.push_back({j, xb, xr1, xr2});
        }

        for (const auto& t : triples) {
            const std::vector<double>& baseV = populations[t.xb].constants;
            const std::vector<double>& randV1 = populations[t.xr1].constants;
            const std::vector<double>& randV2 = populations[t.xr2].constants;
			std::vector<double> v = crossingOver(baseV, randV1, randV2, seed, gen, t.j);
            double newError = calcError(v);
            if (transProb(temp, populations[t.j].error, newError) || !std::isfinite(populations[t.j].error)) {
                populations[t.j].constants = std::move(v);
                populations[t.j].error = newError;
            }
        }
        
        double bestpop = DBL_MAX;
        for (int idx = 0; idx < n; idx ++) {
            bestpop = std::min(bestpop, populations[(size_t)idx].error);
        }
		if (cfg.logLevel == LogLevel::verbose) {
			cout << "best error: " << std::setprecision(10) << bestpop << "\n";
		}

        bool stop = false;
        if (termCond.targetError > 0 && bestpop <= termCond.targetError) stop = true;

        if (!stop && termCond.timeLimit > 0) {
            double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - startTime).count();
            if (elapsed >= termCond.timeLimit) stop = true;
        }
        if (!stop && termCond.stall > 0) {
            //|a-b| <= (atol + rtol * |b|) なら改善なしとみなす
            if (abs(bestSoFar - bestpop) > termCond.ftolAbs + termCond.ftolRel * abs(bestpop)) {
                bestSoFar = bestpop;
                stallCount = 0;
            } else {
                stallCount++;
            }
            if (stallCount >= termCond.stall) stop = true;
        }
        if (stop) break;
    }

    sortByError(populations);
    return populations;
}

std::vector<NASAP_fit::OptimizeResult> NASAP_fit::runDE(int popSize, double lowerLim, double upperLim, const TerminationCondition& termCond, uint64_t seed) {
    assert(popSize >= 3);
    std::vector<std::vector<double>> init = makeRandomPopulation(popSize, lowerLim, upperLim, seed);
    return runDE(std::move(init), termCond, seed);
}


std::vector<NASAP_fit::OptimizeResult> NASAP_fit::runDE(std::vector<std::vector<double>> arg, const TerminationCondition& termCond, uint64_t seed){
    assert(arg.size() >= 3);
    const int world_size = mpi_env.size();
    const int world_rank = mpi_env.rank();

    if (world_size == 1) {
        return runDE_single(arg, termCond, seed);
    }

    #if MPI_VERSION

    auto transProb = [&](double temp, double oldE, double newE) {
        if (newE < oldE) return true;
        if (temp <= 0) return false;
        //return myRand.prob() < exp((oldE - newE) / temp);
        return false;
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
    for (int idx = world_rank; idx < n; idx += world_size) {
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
    double bestSoFar = DBL_MAX;
    int stallCount = 0;
    auto startTime = std::chrono::steady_clock::now();

        for (int gen = 0; gen < termCond.maxIter; ++gen) {
		if (world_rank == 0 && cfg.logLevel != LogLevel::quiet) {
			cout << "current generation : " << gen << " / " << termCond.maxIter << "\n";
		}
        double temp = 0;

        std::vector<Triple> triples;
        triples.reserve((size_t)((n + world_size - 1) / world_size));

        std::vector<int> remoteIndices;
        remoteIndices.reserve((size_t)n);
        auto needRemote = [&](int idx) {
            if (idx % world_size != world_rank) remoteIndices.push_back(idx);
        };

        constexpr uint64_t TAG_XB  = 0xA11CE001u;
        constexpr uint64_t TAG_XR1 = 0xA11CE002u;
        constexpr uint64_t TAG_XR2 = 0xA11CE003u;
        for (int j = world_rank; j < n; j += world_size) {
            int xb = (int)det_rng::uniform_index(det_rng::hash64(seed, (uint64_t)gen, (uint64_t)j, TAG_XB), (uint32_t)n);
            int xr1 = (int)det_rng::uniform_index(det_rng::hash64(seed, (uint64_t)gen, (uint64_t)j, TAG_XR1), (uint32_t)n);
            for (int attempt = 0; xr1 == xb; ++attempt) {
                xr1 = (int)det_rng::uniform_index(det_rng::hash64(seed, (uint64_t)gen, (uint64_t)j, TAG_XR1, (uint64_t)attempt), (uint32_t)n);
            }
            int xr2 = (int)det_rng::uniform_index(det_rng::hash64(seed, (uint64_t)gen, (uint64_t)j, TAG_XR2), (uint32_t)n);
            for (int attempt = 0; (xr2 == xb || xr2 == xr1); ++attempt) {
                xr2 = (int)det_rng::uniform_index(det_rng::hash64(seed, (uint64_t)gen, (uint64_t)j, TAG_XR2, (uint64_t)attempt), (uint32_t)n);
            }
            triples.push_back({j, xb, xr1, xr2});
            needRemote(xb);
            needRemote(xr1);
            needRemote(xr2);
        }

        std::sort(remoteIndices.begin(), 
            remoteIndices.end(),
            [world_size](int left, int right){
                if (left % world_size != right % world_size) return (left % world_size) < (right % world_size);
                return left < right;
            });
        remoteIndices.erase(std::unique(remoteIndices.begin(), remoteIndices.end()), remoteIndices.end());

        MPI_Win_fence(0, winConst);
        for (int idx : remoteIndices) {
            int owner = idx % world_size;
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
			std::vector<double> v = crossingOver(baseV, randV1, randV2, seed, gen, t.j);
            double newError = calcError(v);
            if (transProb(temp, populations[t.j].error, newError) || !std::isfinite(populations[t.j].error)) {
                populations[t.j].constants = std::move(v);
                populations[t.j].error = newError;
                for (int kk = 0; kk < cfg.constantSize; ++kk) {
                    populationsFlat[(size_t)t.j * (size_t)cfg.constantSize + (size_t)kk] = populations[t.j].constants[kk];
                }
            }
        }

        // termination check (MPI islands): use best among owned indices (idx % world_size == rank)
        double localBest = DBL_MAX;
        for (int idx = world_rank; idx < n; idx += world_size) {
            localBest = std::min(localBest, populations[(size_t)idx].error);
        }
        double globalBest = DBL_MAX;
        MPI_Allreduce(&localBest, &globalBest, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        if (world_rank == 0 && cfg.logLevel == LogLevel::verbose) {
            cout << "best error: " << std::setprecision(10) << globalBest << "bestSofar: " << bestSoFar << "\n";
        }

        bool stop = false;
        if (termCond.targetError > 0 && globalBest <= termCond.targetError) {
            stop = true;
            if(world_rank == 0 && cfg.logLevel == LogLevel::verbose) {
                cout << "Target error reached: " << std::setprecision(10) << globalBest << " <= " << termCond.targetError << "\n";
            }
        }

        if (!stop && termCond.timeLimit > 0) {
            double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - startTime).count();
            if (elapsed >= termCond.timeLimit) {
                stop = true;
                if(world_rank == 0 && cfg.logLevel == LogLevel::verbose) {
                    cout << "Time limit reached: " << std::setprecision(10) << elapsed << " >= " << termCond.timeLimit << "\n";
                }
            }
        }
        if (!stop && termCond.stall > 0) {
            //|a-b| <= (atol + rtol * |b|) なら改善なしとみなす
            const double absImprove = bestSoFar - globalBest;
            const double relImprove = (bestSoFar < DBL_MAX && bestSoFar > 0) ? (absImprove / bestSoFar) : DBL_MAX;
            const bool improvedAbs = (termCond.ftolAbs > 0) ? (absImprove > termCond.ftolAbs) : (absImprove > 0);
            const bool improvedRel = (termCond.ftolRel > 0) ? (relImprove > termCond.ftolRel) : true;
            if (abs(bestSoFar - globalBest) > termCond.ftolAbs + termCond.ftolRel * abs(globalBest)) {
                bestSoFar = globalBest;
                stallCount = 0;
            } else {
                stallCount++;
            }
            if (stallCount >= termCond.stall) {
                stop = true;
                if(world_rank == 0 && cfg.logLevel == LogLevel::verbose) {
                    cout << "Stall limit reached: " << stallCount << " >= " << termCond.stall << "\n";
                }
            }
        }
        if (stop) break;
    }

    for (int j = world_rank; j < n; j += world_size){
        populationsErrorFlat[(size_t)j] = populations[j].error;
    }
    // broadcast all constants & errors
    for (int i=0; i<n; ++i) {
        int owner = i % world_size;
        MPI_Bcast(populationsFlat.data() + (size_t)i * (size_t)cfg.constantSize, cfg.constantSize, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Bcast(&populationsErrorFlat[(size_t)i], 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        populations[i].constants.assign(populationsFlat.begin() + (size_t)i * (size_t)cfg.constantSize, populationsFlat.begin() + ((size_t)i + 1) * (size_t)cfg.constantSize);
        populations[i].error = populationsErrorFlat[(size_t)i];
    }
    sortByError(populations);

    MPI_Win_free(&winConst);
    MPI_Win_free(&winErr);

    return populations;
    #endif
}

/*
NASAP_fit::OptimizeResult NASAP_fit::runLM(const std::vector<double>& theta0, const TerminationCondition& termCond){
    const int world_rank = mpi_env.rank();
    const int n = cfg.constantSize;
    const int m = (int)QASAP.size() * cfg.trackedSpecies;
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

    std::vector<DM> arg(1);
    DM rdm, jdm;
    std::vector<double> rvec, jvec;
    Eigen::VectorXd r;
    Eigen::MatrixXd J;
    Eigen::MatrixXd A;
    Eigen::VectorXd g;
    bool isChanged = true;

    const double logLower = std::log(cfg.lowerLim);
    const double logUpper = std::log(cfg.upperLim);
    const double eps = 1e-5; // log-space finite-difference step

    
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

vector<NASAP_fit::OptimizeResult> NASAP_fit::runLM(const vector<vector<double>>& thetaList, const TerminationCondition& termCond){
    const int world_size = mpi_env.size();
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

    for(int i = mpi_env.rank(); i < (int)n; i += world_size){
        results[i] = runLM(results[i].constants, termCond);
        for(int j=0;j<cfg.constantSize;j++){
            populationsFlat[(size_t)i * (size_t)cfg.constantSize + (size_t)j] = results[i].constants[j];
        }
        populationsErrorFlat[i] = results[i].error;
    }
    for(int i=0; i<n; i++){
        MPI_Bcast(populationsFlat.data() + (size_t)i * (size_t)cfg.constantSize, cfg.constantSize, MPI_DOUBLE, i % world_size, MPI_COMM_WORLD);
        MPI_Bcast(&populationsErrorFlat[(size_t)i], 1, MPI_DOUBLE, i % world_size, MPI_COMM_WORLD);
        results[i].constants.assign(populationsFlat.begin() + (size_t)i * (size_t)cfg.constantSize, populationsFlat.begin() + ((size_t)i + 1) * (size_t)cfg.constantSize);
        results[i].error = populationsErrorFlat[(size_t)i];
    }
    return results;
}
*/
//残差ベクトルのJacobianを数値微分で計算する。返り値はm行n列の2次元vectorで、mは残差ベクトルの次元、nはパラメータ（速度定数）の次元。
vector<vector<double>> NASAP_fit::calcJacobian(vector<double>& constant){
    for (int i = 0; i < cfg.species; ++i) {
        NV_Ith_S(y, i) = initialState[i];
    }
    CVodeReInit(cvode_mem, 0.0, y);
    ReactionNetwork::CvodeUserData ud{ &rxnNet, constant.data(), nullptr };
    CVodeSetUserData(cvode_mem, (void*)&ud);
    bool flag=CV_SUCCESS;
    flag = CVodeAdjInit(cvode_mem, 50, CV_HERMITE);
    assert(flag == CV_SUCCESS);

    //Perform forward integration
    

}

void NASAP_fit::putCVODESim(const std::vector<double>& constant) {
    if (mpi_env.rank() != 0) return;
    
    for (int i = 0; i < cfg.species; ++i) {
        NV_Ith_S(y, i) = initialState[i];
    }
    CVodeReInit(cvode_mem, 0.0, y);
    ReactionNetwork::CvodeUserData ud{ &rxnNet, constant.data(), nullptr };
    CVodeSetUserData(cvode_mem, (void*)&ud);
    
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
            size_t idx= (size_t)indexOrder[j];
            assert(0 <= idx && idx < (size_t)cfg.species);
            cout<<100*y_data[idx]/cfg.fullConc[j] <<", ";
        }
        cout<<std::endl;
    }
}



NASAP_fit::SimulationResult NASAP_fit::simulate(const vector<double>& t, const vector<double>& constant, const vector<int>& reaction_ids) {
    if (mpi_env.rank() != 0) return {};
    validateConstants(constant);

    for (int i = 0; i < cfg.species; ++i) {
        NV_Ith_S(y, i) = initialState[i];
    }
    vector<double> sorted_t = t;
    std::sort(sorted_t.begin(), sorted_t.end());
    assert(0 <= sorted_t.front());

    vector<int> sorted_reaction_ids = reaction_ids;
    std::sort(sorted_reaction_ids.begin(), sorted_reaction_ids.end());
    sorted_reaction_ids.erase(std::unique(sorted_reaction_ids.begin(), sorted_reaction_ids.end()), sorted_reaction_ids.end());

    for(int i=0; i<sorted_reaction_ids.size(); i++){
        int rid = sorted_reaction_ids[i];
        assert(0 <= rid && rid < rxnNet.data.size());
        NV_Ith_S(yQ0, i) = 0.0;
    }

    CVodeReInit(cvode_mem, 0.0, y);
    CVodeQuadReInit(cvode_mem, yQ0);
    ReactionNetwork::CvodeUserData ud{ &rxnNet, constant.data(), &sorted_reaction_ids };
    CVodeSetUserData(cvode_mem, (void*)&ud);
    
    SimulationResult result;
    result.t = sorted_t;
    result.timePoints = sorted_t.size();
    result.y.resize(sorted_t.size(), vector<double>(cfg.species, 0.0));
    result.reactionProgress.J.resize(sorted_t.size(), vector<double>(sorted_reaction_ids.size(), 0.0));
    result.reactionProgress.reaction_ids = sorted_reaction_ids;
    result.reactionProgress.reaction_labels.resize(sorted_reaction_ids.size(),"");
    for(size_t i=0; i<sorted_reaction_ids.size(); i++){
        int rid = sorted_reaction_ids[i];
        std::string& label = result.reactionProgress.reaction_labels[i];
        assert(0 <= rid && rid < rxnNet.data.size());
        int init = rxnNet.data[rid][0];
        int entering = rxnNet.data[rid][1];
        int product = rxnNet.data[rid][2];
        int leaving = rxnNet.data[rid][3];
        int kind = rxnNet.data[rid][4];
        int duplicacy = rxnNet.data[rid][5];
        if (entering != -1) {
           label += std::to_string(init) + " + " + std::to_string(entering) + " -> ";
        } else {
           label += std::to_string(init) + " -> ";
        }
        if (leaving != -1) {
           label += std::to_string(product) + " + " + std::to_string(leaving);
        } else {
           label += std::to_string(product);
        }
    }
    
    bool flag=CV_SUCCESS;
    double tret=0.0;
    for (size_t i = 0; i < sorted_t.size(); i++) {
        double time_point = sorted_t[i];
        assert(0 <= time_point && time_point <= endTime);
        if(!(tret-0.1<time_point&&time_point<tret+0.1)) flag=CVode(cvode_mem, time_point, y, &tret, CV_NORMAL);
        assert(flag >= 0);
        // NOTE: Some SUNDIALS builds do not export CVodeGetQuad at runtime.
        // CVodeGetQuadDky(t, k=0) returns the quadrature solution yQ(t).
        flag = CVodeGetQuadDky(cvode_mem, tret, 0, yQ0);
        assert(flag >= 0);

        double* y_data = N_VGetArrayPointer(y);
        double* q_data = N_VGetArrayPointer(yQ0);
        for (int j = 0; j < cfg.species; j++) {
            result.y[i][j] = y_data[j];
        }
        for (size_t j = 0; j < sorted_reaction_ids.size(); j++) {
            int rid = sorted_reaction_ids[j];
            assert(0 <= rid && rid < rxnNet.data.size());
            result.reactionProgress.J[i][j] = q_data[j];
        }
    }
    return result;
}
