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

//5分毎のステップ数カウント
extern int stepCount[61];


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
// simulate/calcNextStep/calcErrorDP removed — CVODE based `calcError` is used instead.

//
void differentialEvolution::addStepCountCV(const std::vector<double>& constant){
    int flag=CV_SUCCESS;


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
    
    
    
    double tret=0.0;
    long nsteps,nsteps_prev=0;

    for (int i = 0; i < endTime; i+=5) {
        double t = i + 5;
        assert(0 <= t && t <= endTime);
        if(t!=tret) flag=CVode(cvode_mem, t, y, &tret, CV_NORMAL);
        if(flag!=CV_SUCCESS){
            for(int j=0;j<cfg.constantSize;j++){
                cout<<constant[j]<<", ";
            }
            cout<<endl;
            exit(1);
        }
        CVodeGetNumSteps(cvode_mem,&nsteps);
        stepCount[i/5]+=(nsteps - nsteps_prev);
        nsteps_prev=nsteps;
    }
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
    assert((int)point.size()==cfg.constantSize);
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
    assert((int)point.size()==cfg.constantSize);
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

//エージェントのセット
void differentialEvolution::setPop(int idx, const std::vector<double>& theta){
    assert(idx>=0 && idx<cfg.popSize);
    assert((int)theta.size()==cfg.constantSize);
    bool isSame=true;
    for(int j=0;j<cfg.constantSize;j++){
        assert(std::isfinite(theta[j]));
        assert(0<theta[j]);
        if(theta[j]<cfg.lowerLim||theta[j]>cfg.upperLim){
            std::cerr<<"Warning: Out of bounds parameter value "<<theta[j]<<" at index "<<j<<".\n";
        }
        double clamped =std::clamp(theta[j], cfg.lowerLim, cfg.upperLim);
        if(populations[idx].constant[j]!=clamped){
            isSame=false;
        }
        populations[idx].constant[j]=clamped;
        populationsFlat[idx * cfg.constantSize + j] = populations[idx].constant[j];
    }
    if(!isSame){
        populations[idx].error=DBL_MAX;
        populationsErrorFlat[idx]=populations[idx].error;
    }
}

std::vector<double> differentialEvolution::getPop(int idx){
    assert(idx>=0 && idx<cfg.popSize);
    MPI_Bcast(populations[idx].constant.data(), cfg.constantSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return populations[idx].constant;
}

double differentialEvolution::getPopError(int idx){
    assert(idx>=0 && idx<cfg.popSize);
    MPI_Bcast(&populations[idx].error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(populations[idx].error==DBL_MAX){
        std::cerr<<"Warning: Error for population "<<idx<<" is not calculated yet.\n";
        evaluate();
    }
    return populations[idx].error;
}

void differentialEvolution::evaluate(){
    vector<int> uncalculated(cfg.popSize, -1);
    if(proc_rank==0){
        int count=0;
        for (int i = 0; i < cfg.popSize; i++) {
            if(populations[i].error==DBL_MAX){
                uncalculated[count]=i;
                count++;
            }
        }
        MPI_Bcast(uncalculated.data(), cfg.popSize, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else{
        MPI_Bcast(uncalculated.data(), cfg.popSize, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if(uncalculated[0]==-1) return;
    MPI_Win winConst, winErr;
    MPI_Win_create(populationsFlat.data(), sizeof(double)*cfg.popSize*cfg.constantSize, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &winConst);
    MPI_Win_create(populationsErrorFlat.data(), sizeof(double)*cfg.popSize, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &winErr);
    MPI_Win_fence(0, winConst);
    MPI_Win_fence(0, winErr);
    for(int i=proc_rank;i<cfg.popSize;i+=num_procs){
        int idx =uncalculated[i];
        if(idx!=-1){
            MPI_Get(populations[idx].constant.data(), cfg.constantSize, MPI_DOUBLE, 0, idx * cfg.constantSize, cfg.constantSize, MPI_DOUBLE, winConst);
            populations[idx].error=calcError(populations[idx].constant);
            populationsErrorFlat[idx]=populations[idx].error;
            MPI_Put(&populationsErrorFlat[idx], 1, MPI_DOUBLE, 0, idx, 1, MPI_DOUBLE, winErr);
        }
    }
    MPI_Win_fence(0, winConst);
    MPI_Win_fence(0, winErr);
    MPI_Win_free(&winConst);
    MPI_Win_free(&winErr);
    if(proc_rank==0){
        for(int i : uncalculated){
            if(i!=-1){
                populations[i].error=populationsErrorFlat[i];
            }
            else break;
        }
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
    cfg.popSize = config::popSize;
    cfg.maxGen = config::maxGen;
    cfg.tolAbsError = config::tolAbsError;
    cfg.tolRelError = config::tolRelError;
    cfg.safetyConstant = config::safetyConstant;
    cfg.scalar = config::scalar;
    cfg.crossOver = config::crossOver;
    cfg.upperLim = config::upperLim;
    cfg.lowerLim = config::lowerLim;

    setQASAPData(arg);
    // allocate initialState according to runtime species
    initialState.assign(cfg.species, 0.0);
    // allocate flat population buffers
    populationsFlat.assign(cfg.popSize * cfg.constantSize, 0.0);
    populationsErrorFlat.assign(cfg.popSize, DBL_MAX);
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
    

    populations.clear();
    populations.reserve(cfg.popSize);
    for (int i = 0; i < cfg.popSize; i++) {
        populations.emplace_back(cfg.constantSize);
        for (int j = 0; j < cfg.constantSize; j++) {
            double v = randbetExp(cfg.lowerLim, cfg.upperLim);
            populations[i].constant[j] = v;
            populationsFlat[i * cfg.constantSize + j] = v;
        }
    }
}

void differentialEvolution::Optimize(){
    auto transProb = [&](double temp, double oldE, double newE) {
        if (newE < oldE) return true;
        if (temp <= 0) return false;
        return myRand.prob() < exp((oldE - newE) / temp);
    };
    evaluate(); //初期集団の評価
    MPI_Win winConst, winErr;
    MPI_Win_create(populationsFlat.data(), sizeof(double)*cfg.popSize*cfg.constantSize, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &winConst);
    MPI_Win_create(populationsErrorFlat.data(), sizeof(double)*cfg.popSize, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &winErr);
    for (int i = 0; i < cfg.maxGen; i++) {
        if(proc_rank==0)cout<<"current generation : "<<i<<" / "<<cfg.maxGen<<"\n";
        double temp=0;
        
        vector<std::array<int, 3>> lookahead;
        vector<int> indicesToVisit;
        vector<double>errors;
        auto pushIndex = [&](int idx){
            if(idx%num_procs==proc_rank)return false;
            else{
                indicesToVisit.push_back(idx);
                return true;
            }
        };
        for (int j = proc_rank; j < cfg.popSize; j+=num_procs){
            int xb = myRand(cfg.popSize), xr1=myRand(cfg.popSize), xr2=myRand(cfg.popSize);
            while (xb == xr1) {xr1 = myRand(cfg.popSize);} 
            while (xb == xr2 || xr1 == xr2) {xr2 = myRand(cfg.popSize);} 
            lookahead.push_back({xb,xr1,xr2});
            pushIndex(xb); pushIndex(xr1); pushIndex(xr2);
        }
        std::sort(
            indicesToVisit.begin(),
            indicesToVisit.end(),
            [](int left,int right){
                if(left%num_procs==right%num_procs)return left<right;
                else return left%num_procs<right%num_procs;
            }
        );
        indicesToVisit.erase(std::unique(indicesToVisit.begin(),indicesToVisit.end()),indicesToVisit.end());
        MPI_Win_fence(0, winConst);
        MPI_Win_fence(0, winErr);
        for(int idx : indicesToVisit){
            if(idx%num_procs==proc_rank) continue;
            // fetch constants into local populations[idx].constant buffer
            MPI_Get(populations[idx].constant.data(), cfg.constantSize, MPI_DOUBLE, idx%num_procs, idx * cfg.constantSize, cfg.constantSize, MPI_DOUBLE, winConst);
            // fetch error value
            MPI_Get(&populations[idx].error, 1, MPI_DOUBLE, idx%num_procs, idx, 1, MPI_DOUBLE, winErr);
            // note: data will be available after the following fences
        }
        MPI_Win_fence(0, winConst);
        MPI_Win_fence(0, winErr);
        for(int idx : indicesToVisit){
            if(idx%num_procs==proc_rank) continue;
            errors.push_back(populations[idx].error);
        }

        for (int j = proc_rank; j < cfg.popSize; j+=num_procs) {
            //突然変異
            //ベースベクトルとその他二つのベクトルのindex
            assert(!lookahead.empty());
            auto [xb,xr1,xr2]=lookahead.back();
            lookahead.pop_back();
            //新しいベクトル、ベースベクトルとその他二つのベクトル
            std::vector<double> v,baseV,randV1,randV2;
            baseV=populations[xb].constant;
            randV1=populations[xr1].constant;
            randV2=populations[xr2].constant;
            //交叉
            v=crossingOver(baseV,randV1,randV2);
            double newError = calcError(v);
            if (transProb(temp, populations[j].error, newError) || !std::isfinite(populations[j].error)) {
                populations[j].constant = v;
                populations[j].error = newError;
                // update flat buffers for MPI
                for (int kk = 0; kk < cfg.constantSize; ++kk) populationsFlat[j * cfg.constantSize + kk] = populations[j].constant[kk];
                populationsErrorFlat[j] = newError;
            }
        }
    }
    // gather all final constants & errors to rank 0
    MPI_Win_fence(0, winConst);
    MPI_Win_fence(0, winErr);
    for(int idx=proc_rank;idx<cfg.popSize;idx+=num_procs){
        MPI_Put(populations[idx].constant.data(), cfg.constantSize, MPI_DOUBLE, 0, idx * cfg.constantSize, cfg.constantSize, MPI_DOUBLE, winConst);
        MPI_Put(&populations[idx].error, 1, MPI_DOUBLE, 0, idx, 1, MPI_DOUBLE, winErr);
    }
    MPI_Win_fence(0, winConst);
    MPI_Win_fence(0, winErr);
    
    MPI_Win_free(&winConst);
    MPI_Win_free(&winErr);
}

void differentialEvolution::runLM(int idx){
    assert(0 <= idx && idx < cfg.popSize);
    const int n = cfg.constantSize;
    const int m = (int)QASAP.size() * cfg.trackedSpecies;

    std::vector<double> theta = populations[idx].constant;
    std::vector<double> logTheta(n);
    for (int i = 0; i < n; ++i) {
        logTheta[i] = log(theta[i]);
    }
    double bestErr = populations[idx].error;
    if (!std::isfinite(bestErr)) bestErr = calcError(theta);

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
        for(int i=0; i<cfg.constantSize; i++){
            cout<<std::setprecision(4)<<theta[i]<<", ";
        }
        cout<<endl;
        cout<<"Current error: "<<std::setprecision(6)<<bestErr<<endl;
        cout<<"Iter : "<<iter<<", Lambda: "<<std::setprecision(6)<<lambda<<endl;

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

        std::vector<double> trial(n),logTrial(n);
        for (int i = 0; i < n; ++i) {
            logTrial[i] = std::clamp(logTheta[i] + delta[i], log(cfg.lowerLim), log(cfg.upperLim));
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
            if(reliabilityIndex > 0.75){
                lambda /= 3.0;
            } else if(reliabilityIndex < 0.25){
                lambda *= 2.0;
            }
            if (delta.norm() < 1e-6) break;
        } else {
            isChanged = false;
            lambda *= 2.0;
        }
    }

    populations[idx].constant = theta;
    populations[idx].error = bestErr;
    for (int i = 0; i < n; ++i) {
        populationsFlat[idx * n + i] = theta[i];
    }
    populationsErrorFlat[idx] = bestErr;
}
void differentialEvolution::runLM(vector<int>& indices){
    MPI_Win winConst, winErr;
    MPI_Win_create(populationsFlat.data(), sizeof(double)*cfg.popSize*cfg.constantSize, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &winConst);
    MPI_Win_create(populationsErrorFlat.data(), sizeof(double)*cfg.popSize, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &winErr);
    MPI_Win_fence(0, winConst);
    MPI_Win_fence(0, winErr);
    for(int i=proc_rank;i<indices.size();i+=num_procs){
        int idx =indices[i];
        MPI_Get(populations[idx].constant.data(), cfg.constantSize, MPI_DOUBLE, 0, idx * cfg.constantSize, cfg.constantSize, MPI_DOUBLE, winConst);
        runLM(idx);
        populationsErrorFlat[idx]=populations[idx].error;
        MPI_Put(populations[idx].constant.data(), cfg.constantSize, MPI_DOUBLE, 0, idx * cfg.constantSize, cfg.constantSize, MPI_DOUBLE, winConst);
        MPI_Put(&populationsErrorFlat[idx], 1, MPI_DOUBLE, 0, idx, 1, MPI_DOUBLE, winErr);
    }
    MPI_Win_fence(0, winConst);
    MPI_Win_fence(0, winErr);
    MPI_Win_free(&winConst);
    MPI_Win_free(&winErr);
    if(proc_rank==0){
        for(int i : indices){
            if(i!=-1){
                populations[i].error=populationsErrorFlat[i];
            }
            else break;
        }
    }
}
//最良個体のインデックスを返す
int differentialEvolution::best() {
    double minerror = DBL_MAX;
    int bestIdx = 0;
    if(proc_rank==0){
         for (int i = 0; i < cfg.popSize; i++) {
            if (minerror > populations[i].error) {
                minerror = populations[i].error;
                bestIdx = i;
            }
        }
    }
    MPI_Bcast(&bestIdx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return bestIdx;
}

void differentialEvolution::sortPopulationsByError(){
    sort(
        populations.begin(),
        populations.end(),
        [](const individuals& a, const individuals& b){
            return a.error < b.error;
        }
    );
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

void differentialEvolution::DEBUG() {
    std::cout<<"Population Error and Constants:\n";
    for (int i = 0; i < cfg.popSize; i++) {
        std::cout<< populations[i].error << std::endl;
        for (int j = 0; j < cfg.constantSize; j++) {
            std::cout << populations[i].constant[j] << " ";
        }
        std::cout << std::endl;
    }
}

