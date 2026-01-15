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

#include <mpi.h>
#include <cppad/cppad.hpp>
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunlinsol/sunlinsol_klu.h>

#include "../include/differentialEvolution.hpp"
#include "../include/constants.hpp"
#include "../include/xorshift.hpp"
#include "../include/ODE.hpp"
#include "../include/rhsfBuilder.hpp"
#include "../include/jacBuilder.hpp"
#include "../include/Jacf.hpp" 
#include "../include/Rhsf.hpp"

xorshift myRand(1);
int cnt=0;
extern int num_procs, proc_rank;
    
using std::vector;
using std::cout;
using std::endl;
using std::flush;
using std::sqrt;
using CppAD::sqrt;

//5分毎のステップ数カウント
extern int stepCount[61];

inline double castToDouble(const CppAD::AD<double>& x){
    return CppAD::Value(CppAD::Var2Par(x));
}
inline double castToDouble(const double& x){
    return x;
}


// Removed DP (explicit Runge-Kutta) integration routines: using CVODE instead.


std::vector<double> differentialEvolution::crossingOver(const std::vector<double>& baseV, const std::vector<double>& randV1, const std::vector<double>& randV2){
    std::vector<double> v(config::constantSize);
    int jr = myRand(config::constantSize);
    for (int k = 0; k < config::constantSize; k++) {
        //交叉
        if (jr == k || myRand.prob() < config::crossOver) {
            v[k] = baseV[k] * pow(randV1[k] / randV2[k], config::scalar);
            v[k]=std::clamp(v[k],config::lowerLim,config::upperLim);
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


    N_Vector y = N_VNew_Serial(config::species, sunctx);
    for (int i = 0; i < config::species; ++i) {
        NV_Ith_S(y, i) = initialState[i];
    }
    
    CVodeReInit(cvode_mem, 0.0, y);
    CVodeSetUserData(cvode_mem, (void*)constant.data());
    
    
    
    double tret=0.0;
    long nsteps,nsteps_prev=0;

    for (int i = 0; i < endTime; i+=5) {
        double t = i + 5;
        assert(0 <= t && t <= endTime);
        if(t!=tret) flag=CVode(cvode_mem, t, y, &tret, CV_NORMAL);
        if(flag!=CV_SUCCESS){
            for(int j=0;j<config::constantSize;j++){
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
template<class T>
T differentialEvolution::calcError(const std::vector<T>& constant) {
    int flag=CV_SUCCESS;

    N_Vector y = N_VNew_Serial(config::species, sunctx);
    for (int i = 0; i < config::species; ++i) {
        NV_Ith_S(y, i) = initialState[i];
    }
    
    CVodeReInit(cvode_mem, 0.0, y);
    CVodeSetUserData(cvode_mem, (void*)constant.data());
    
    T SSR = 0;
    double tret=0.0;
    for (int i = 0; i < QASAP.size(); i++) {
        double t = QASAP[i].time;
        assert(0 <= t && t <= endTime);
        if(t!=tret) flag=CVode(cvode_mem, t, y, &tret, CV_NORMAL);
        if(flag!=CV_SUCCESS){
            for(int j=0;j<config::constantSize;j++){
                cout<<constant[j]<<", ";
            }
            cout<<endl;
            exit(1);
        }
        double* y_data = N_VGetArrayPointer(y);
        for (int j = 0; j < config::trackedSpecies; j++) {
            size_t idx= config::trackedIndex[j];
            assert(0 <= idx && idx < config::species);
            SSR += (y_data[idx] / config::fullConc[j] - QASAP[i].state[j]) * (y_data[idx] / config::fullConc[j] - QASAP[i].state[j]);
        }
    }
    return SSR;
}

std::vector<double> differentialEvolution::getJacobian(const std::vector<double>& point){
    int n = (int)point.size();
    std::vector<double> jac(n);
    //Jacobianの実装は後回し
    return jac;
}

//ヘッセ行列の計算
std::vector<std::vector<double>> differentialEvolution::getHessian(const std::vector<double>& point){
    // numeric central differences for Hessian
    int n = (int)point.size();
    std::vector<std::vector<double>> H(n, std::vector<double>(n, 0.0));
    //Hessianの実装は後回し
    return H;
}

//実験データのセット
void differentialEvolution::setData(std::vector<std::vector<double>>& arg) {
    for (const std::vector<double>& vec : arg) {
        assert(vec.size() == config::trackedSpecies + 1);
        QASAP.push_back({ vec[0],std::vector<double>({vec.begin()+1, vec.begin()+config::trackedSpecies+1}) });
    }
}

//途中まで最適化されたエージェントのセット
void differentialEvolution::setPop(){
    std::string filename = "../optimizingData.txt";
    std::ifstream ifs(filename);
    if (!ifs) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    populations = vector<individuals>(config::popSize);
    // ensure flat buffers exist
    populationsFlat.assign(config::popSize * config::constantSize, 0.0);
    populationsErrorFlat.assign(config::popSize, DBL_MAX);
    for(int i=0; i<config::popSize; i++){
        ifs >> populations[i].error;
        populationsErrorFlat[i] = populations[i].error;
        for(int j=0; j<config::constantSize; j++){
            ifs >> populations[i].constant[j];
            populationsFlat[i * config::constantSize + j] = populations[i].constant[j];
        }
    }
}

differentialEvolution::differentialEvolution(vector<vector<double>>& arg) {
    myRand=xorshift(1+proc_rank);
    setData(arg);
    // allocate initialState according to runtime species
    initialState.assign(config::species, 0.0);
    // allocate flat population buffers
    populationsFlat.assign(config::popSize * config::constantSize, 0.0);
    populationsErrorFlat.assign(config::popSize, DBL_MAX);
    for (int i = 0; i < config::trackedSpecies; i++) {
        initialState[config::trackedIndex[i]] = config::fullConc[i]*QASAP[0].state[i];
    }
    endTime = arg.back()[0];

    //seting up CVODE
    N_Vector y = N_VNew_Serial(config::species, sunctx);
    for (int i = 0; i < config::species; ++i) {
        NV_Ith_S(y, i) = initialState[i];
    }
    
    #if USE_PREGENERATED_RHSF
        int flag = CVodeInit(cvode_mem, rhsf, 0.0, y);
    #else
        int flag = CVodeInit(cvode_mem, rhsfBuilder::rhsf, 0.0, y);
    #endif
    assert(flag == CV_SUCCESS);

    flag = CVodeSStolerances(cvode_mem, config::tolRelError, config::tolAbsError);
    assert(flag == CV_SUCCESS);

    #if USE_PREGENERATED_JACOBIAN
        SUNMatrix J = SUNSparseMatrix(config::species, config::species, nonZeroElems, CSC_MAT, sunctx);
        SUNLinearSolver LS = SUNLinSol_KLU(y, J, sunctx);
        flag = CVodeSetLinearSolver(cvode_mem, LS, J);
        assert(flag == CV_SUCCESS);
        flag = CVodeSetJacFn(cvode_mem, JacFn);
    #else
        SUNMatrix J = SUNSparseMatrix(config::species, config::species, jacBuilder::nonZeros, CSC_MAT, sunctx);
        SUNLinearSolver LS = SUNLinSol_KLU(y, J, sunctx);
        flag = CVodeSetLinearSolver(cvode_mem, LS, J);
        assert(flag == CV_SUCCESS);
        flag = CVodeSetJacFn(cvode_mem, jacBuilder::JacFn);
    #endif
    assert(flag == CV_SUCCESS);

    flag = CVodeSetMaxNumSteps(cvode_mem, 10000);
    assert(flag == CV_SUCCESS);
    

    populations = std::vector<individuals>(config::popSize);
    for (int i = proc_rank; i < config::popSize; i+=num_procs) {
        for (int j = 0; j < config::constantSize; j++) {
            double v = randbetExp(config::lowerLim, config::upperLim);
            populations[i].constant[j] = v;
            populationsFlat[i * config::constantSize + j] = v;
        }
        populationsErrorFlat[i] = DBL_MAX;
        //populations[i].error = calcError(populations[i].constant);
    }
}

void differentialEvolution::Optimize(){
    auto transProb = [&](double temp, double oldE, double newE) {
        if (newE < oldE) return true;
        if (temp <= 0) return false;
        return myRand.prob() < exp((oldE - newE) / temp);
    };
    MPI_Win winConst, winErr;
    MPI_Win_create(populationsFlat.data(), sizeof(double)*config::popSize*config::constantSize, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &winConst);
    MPI_Win_create(populationsErrorFlat.data(), sizeof(double)*config::popSize, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &winErr);
    for (int i = 0; i < config::maxGen; i++) {
        if(proc_rank==0)cout<<"current generation : "<<i<<" / "<<config::maxGen<<"\n";
        double temp=0;
        
        vector<std::array<int, 3>> lockahead;
        vector<int> indicesToVisit;
        vector<double>errors;
        auto pushIndex = [&](int idx){
            if(idx%num_procs==proc_rank)return false;
            else{
                indicesToVisit.push_back(idx);
                return true;
            }
        };
        for (int j = proc_rank; j < config::popSize; j+=num_procs){
            int xb = myRand(config::popSize), xr1=myRand(config::popSize), xr2=myRand(config::popSize);
            while (xb == xr1) {xr1 = myRand(config::popSize);}
            while (xb == xr2 || xr1 == xr2) {xr2 = myRand(config::popSize);}
            lockahead.push_back({xb,xr1,xr2});
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
            MPI_Get(populations[idx].constant.data(), config::constantSize, MPI_DOUBLE, idx%num_procs, idx * config::constantSize, config::constantSize, MPI_DOUBLE, winConst);
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

        for (int j = proc_rank; j < config::popSize; j+=num_procs) {
            //突然変異
            //ベースベクトルとその他二つのベクトルのindex
            assert(!lockahead.empty());
            auto [xb,xr1,xr2]=lockahead.back();
            lockahead.pop_back();
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
                for (int kk = 0; kk < config::constantSize; ++kk) populationsFlat[j * config::constantSize + kk] = populations[j].constant[kk];
                populationsErrorFlat[j] = newError;
            }
        }
    }
    MPI_Win_free(&winConst);
    MPI_Win_free(&winErr);
}

//最良個体の定数を返す
void differentialEvolution::best(std::vector<double>& ret, double& minerror) {
    // find local best
    minerror = DBL_MAX;
    std::vector<double> localBest(config::constantSize);
    for (auto& t : populations) {
        if (minerror > t.error) {
            minerror = t.error;
            localBest = t.constant;
        }
    }
    // gather all minerrors
    std::vector<double> all_minerrors(num_procs);
    MPI_Allgather(&minerror, 1, MPI_DOUBLE, all_minerrors.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
    // find global minimum and its owner
    double globalMin = all_minerrors[0];
    int owner = 0;
    for (int i = 1; i < num_procs; ++i) {
        if (all_minerrors[i] < globalMin) { globalMin = all_minerrors[i]; owner = i; }
    }
    minerror = globalMin;
    // owner broadcasts the best constants
    ret.assign(config::constantSize, 0.0);
    if ((int)localBest.size() != config::constantSize) localBest.resize(config::constantSize);
    if (proc_rank == owner) ret = localBest;
    MPI_Bcast(ret.data(), config::constantSize, MPI_DOUBLE, owner, MPI_COMM_WORLD);
}



void differentialEvolution::putCVODESim(const std::vector<double>& constant) {
    if(proc_rank!=0)return;
    N_Vector y = N_VNew_Serial(config::species, sunctx);
    for (int i = 0; i < config::species; ++i) {
        NV_Ith_S(y, i) = initialState[i];
    }
    CVodeReInit(cvode_mem, 0.0, y);
    CVodeSetUserData(cvode_mem, (void*)constant.data());
    
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
        for (int j = 0; j < config::trackedSpecies; j++) {
            size_t idx= config::trackedIndex[j];
            assert(0 <= idx && idx < config::species);
            cout<<y_data[idx]/config::fullConc[j] <<", ";
        }
        cout<<std::endl;
    }

}



void differentialEvolution::DEBUG() {
    std::cout<<"Population Error and Constants:\n";
    for (int i = 0; i < config::popSize; i++) {
        std::cout<< populations[i].error << std::endl;
        for (int j = 0; j < config::constantSize; j++) {
            std::cout << populations[i].constant[j] << " ";
        }
        std::cout << std::endl;
    }
}

