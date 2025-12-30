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
#include "../include/speciesAmount.hpp"
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


//シミュレーションにおける、次のステップの計算
template<class T>
differentialEvolution::stepResult<T> differentialEvolution::calcNextStep(const std::array<T, config::constantSize>& reactConst, const speciesAmount<T>& data, double stepSize) {
    stepResult<T> ret;
    speciesAmount<T> val1, val2, val3, val4,val5,val6,val7, slope1, slope2, slope3, slope4, slope5, slope6, slope7, order4_diff, order5_diff;
    double totalError, h_new;
    while(1){
        if(stepSize<=1e-12)stepSize=1e-12;
        totalError=0.0;
        val1 = data;
        diffCoeff(reactConst, val1, slope1);
        val2 = val1 + stepSize * slope1 * (1.0/5);
        diffCoeff(reactConst, val2, slope2);
        val3 = val1 + stepSize * (slope1 * (3.0/40) + slope2 * (9.0/40));
        diffCoeff(reactConst, val3, slope3);
        val4 = val1 + stepSize * (slope1 * (44.0/45) + slope2 * (-56.0/15) + slope3 * (32.0/9));
        diffCoeff(reactConst, val4, slope4);
        val5 = val1 + stepSize * (slope1 * (19372.0/6561) + slope2 * (-25360.0/2187) + slope3 * (64448.0/6561) + slope4 * (-212.0/729));
        diffCoeff(reactConst, val5, slope5);
        val6 = val1 + stepSize * (slope1 * (9017.0/3168) + slope2 * (-355.0/33) + slope3 * (46732.0/5247) + slope4 * (49.0/176) + slope5 * (-5103.0/18656));
        diffCoeff(reactConst, val6, slope6);
        order5_diff = stepSize * (slope1 * (35.0/384) + slope3 * (500.0/1113) + slope4 * (125.0/192) + slope5 * (-2187.0/6784) + slope6 * (11.0/84));
        val7 = val1 + order5_diff;
        diffCoeff(reactConst, val7, slope7);
        order4_diff = stepSize * (slope1 * (5179.0/57600) + slope3 * (7571.0/16695) + slope4 * (393.0/640) + slope5 * (-92097.0/339200) + slope6 * (187.0/2100) + slope7 * (1.0/40));
        const speciesAmount<T>& order5 = val7;

        for(int i=0; i<config::species; i++){
            double e = castToDouble(order4_diff[i]-order5_diff[i]);
            totalError+=e*e;
        }
        totalError=sqrt(totalError/config::species);
        if(totalError<=config::tolAbsError){//ステップ成功
            ret.usedStepSize=stepSize;
            ret.newState=order5;
            if(totalError==0)h_new = stepSize * 2;
            else{
                h_new=config::safetyConstant*stepSize*std::pow(config::tolAbsError/totalError,(1.0/5));
                h_new=std::clamp(h_new, stepSize*0.25, stepSize*2.0);
            }
            ret.newStepSize=h_new;
            return ret;
        }else{//ステップ失敗
        h_new=config::safetyConstant*stepSize*std::pow(config::tolAbsError/totalError,(1.0/5));
        h_new=std::clamp(h_new, stepSize*0.1, stepSize*0.9);
        stepSize=h_new;
        }
    }
}


std::array<double, config::constantSize> differentialEvolution::crossingOver(const std::array<double, config::constantSize>& baseV, const std::array<double, config::constantSize>& randV1, const std::array<double, config::constantSize>& randV2){
    std::array<double, config::constantSize> v;
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
template<class T>
std::vector<speciesAmount<T>> differentialEvolution::simulate(const std::array<T, config::constantSize>& constant){

    std::vector<speciesAmount<T>> simulation = {speciesAmount<T>()}; 
    for (int i = 0; i < config::species; i++) {
        simulation[0][i] = initialState[i];
    }
    simTime={0.0};
    double currentStepSize=0.001;
    //シミュレーション
    for (int k = 0; simTime.back() <= endTime; k++) {
        stepResult<T> res=calcNextStep(constant, simulation[k], currentStepSize);
        simulation.push_back(res.newState);
        simTime.push_back(simTime.back()+res.usedStepSize);
        currentStepSize = res.newStepSize;
    }
    return simulation;
}

template<class T>
T differentialEvolution::calcErrorDP(const std::array<T, config::constantSize>& constant) {
    if(proc_rank!=0)return 0;
    const std::vector<speciesAmount<double>>& simulation = simulate(constant);
    double SSR = 0;
    for (int i = 0; i<QASAP.size(); i++) {
        double t = QASAP[i].time;
        assert(0 <= t && t <= endTime);

        int nearestIndex = std::upper_bound(simTime.begin(),simTime.end(),t)-simTime.begin()-1;
        double nearestTime = simTime[nearestIndex];
        double difftime = t - nearestTime;
        stepResult res = calcNextStep(constant,simulation[nearestIndex],difftime);
        for (int j = 0; j < config::trackedSpecies; j++) {
            size_t idx= config::trackedIndex[j];
            assert(0 <= idx && idx < config::species);
            SSR += (res.newState[idx]/config::fullConc[j] - QASAP[i].state[j]) * (res.newState[idx]/config::fullConc[j] - QASAP[i].state[j]);
        }
    }
    return SSR;
}

//
void differentialEvolution::addStepCountCV(const std::array<double, config::constantSize>& constant){
    int flag=CV_SUCCESS;


    N_Vector y = N_VNew_Serial(config::species, sunctx);
    for (int i = 0; i < config::species; ++i) {
        NV_Ith_S(y, i) = initialState[i];
    }
    
    CVodeReInit(cvode_mem, 0.0, y);
    CVodeSetUserData(cvode_mem, (void*)&constant);
    
    
    
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
void differentialEvolution::addStepCountDP(const std::array<double, config::constantSize>& constant){
    if(proc_rank!=0)return;
    const std::vector<speciesAmount<double>>& simulation = simulate(constant);
    for (double t : simTime) {
        if(t==0)continue;
        stepCount[int(t/5)]++;
    }
}

//平方残差和の計算
template<class T>
T differentialEvolution::calcError(const std::array<T, config::constantSize>& constant) {
    int flag=CV_SUCCESS;

    N_Vector y = N_VNew_Serial(config::species, sunctx);
    for (int i = 0; i < config::species; ++i) {
        NV_Ith_S(y, i) = initialState[i];
    }
    
    CVodeReInit(cvode_mem, 0.0, y);
    CVodeSetUserData(cvode_mem, (void*)&constant);
    
    //シミュレーション
    //const std::vector<speciesAmount<T>>& simulation = simulate(constant);
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

std::vector<double> differentialEvolution::getJacobian(const std::array<double, config::constantSize>& point){
    using CppAD::AD;

    // 1. AD型配列を作成
    std::vector<AD<double>> ax_vec(point.begin(), point.end());
    // 2. 独立変数を宣言
    CppAD::Independent(ax_vec);
    std::array<AD<double>, config::constantSize> ax;
    for(int i=0; i<config::constantSize; ++i) ax[i]=ax_vec[i];

    // 3. calcError は array を受け取るのでそのままでOK
    std::vector<AD<double>> ay(1);
    ay[0] = calcError(ax);

    // 4. std::array から std::vector に変換して ADFun に渡す
    CppAD::ADFun<double> f(ax_vec, ay);

    // 5. ヤコビ行列の計算
    std::vector<double> x(point.begin(), point.end());
    std::vector<double> jac = f.Jacobian(x);

    return jac;
}

//ヘッセ行列の計算
std::vector<std::vector<double>> differentialEvolution::getHessian(const std::array<double, config::constantSize>& point){
    using CppAD::AD;

    // 1. AD型配列を作成
    std::vector<AD<double>> ax_vec(point.begin(), point.end());
    // 2. 独立変数を宣言
    CppAD::Independent(ax_vec);
    std::array<AD<double>, config::constantSize> ax;
    for(int i=0; i<config::constantSize; ++i) ax[i]=ax_vec[i];

    // 3. calcError は array を受け取るのでそのままでOK
    std::vector<AD<double>> ay(1);
    ay[0] = calcError(ax);

    // 4. std::array から std::vector に変換して ADFun に渡す
    CppAD::ADFun<double> f(ax_vec, ay);

    // 5. ヘッセ行列の計算
    std::vector<double> x(point.begin(), point.end());
    std::vector<double> hes = f.Hessian(x,0);

    // 6. 2次元配列に整形
    std::vector<std::vector<double>> H(config::constantSize, std::vector<double>(config::constantSize));
    for (int i = 0, k = 0; i < config::constantSize; ++i){
        for (int j = 0; j < config::constantSize; ++j, ++k){
            H[i][j] = hes[k];
        }
    }
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
    for(int i=0; i<config::popSize; i++){
        ifs >> populations[i].error;
        for(int j=0; j<config::constantSize; j++){
            ifs >> populations[i].constant[j];
        }
    }
}

differentialEvolution::differentialEvolution(vector<vector<double>>& arg) {
    myRand=xorshift(1+proc_rank);
    setData(arg);
    for (int i = 0; i < config::trackedSpecies; i++) {
        initialState[config::trackedIndex[i]] = config::fullConc[i]*QASAP[0].state[i];
    }
    endTime = arg.back()[0];

    //seting up CVODE
    N_Vector y = N_VNew_Serial(config::species, sunctx);
    for (int i = 0; i < config::species; ++i) {
        NV_Ith_S(y, i) = initialState[i];
    }

    int flag = CVodeInit(cvode_mem, rhsfBuilder::rhsf, 0.0, y);
    assert(flag == CV_SUCCESS);
    flag = CVodeSStolerances(cvode_mem, config::tolRelError, config::tolAbsError);
    assert(flag == CV_SUCCESS);
    SUNMatrix J = SUNSparseMatrix(config::species, config::species, jacBuilder::nonZeros, CSC_MAT, sunctx);
    SUNLinearSolver LS = SUNLinSol_KLU(y, J, sunctx);
    flag = CVodeSetLinearSolver(cvode_mem, LS, J);
    assert(flag == CV_SUCCESS);
    flag = CVodeSetJacFn(cvode_mem, jacBuilder::JacFn);
    assert(flag == CV_SUCCESS);
    flag = CVodeSetMaxNumSteps(cvode_mem, 10000);
    assert(flag == CV_SUCCESS);
    

    populations = std::vector<individuals>(config::popSize);
    for (int i = proc_rank; i < config::popSize; i+=num_procs) {
        for (int j = 0; j < config::constantSize; j++) {
            populations[i].constant[j] = randbetExp(config::lowerLim, config::upperLim);
        }
        //populations[i].error = calcError(populations[i].constant);
    }
}

void differentialEvolution::Optimize(){
    auto transProb = [&](double temp, double oldE, double newE) {
        if (newE < oldE) return true;
        if (temp <= 0) return false;
        return myRand.prob() < exp((oldE - newE) / temp);
    };
    MPI_Win win;
    MPI_Win_create(populations.data(), sizeof(individuals)*config::popSize, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    for (int i = 0; i < config::maxGen; i++) {
        if(proc_rank==0)cout<<"loop : "<<i<<"\n";
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
        MPI_Win_fence(0, win);
        for(int idx : indicesToVisit){
            assert(idx%num_procs!=proc_rank);
            MPI_Get(&populations[idx], sizeof(individuals), MPI_BYTE, idx%num_procs, sizeof(individuals)*idx, sizeof(individuals), MPI_BYTE, win);
            errors.push_back(populations[idx].error);
        }
        MPI_Win_fence(0, win);

        for (int j = proc_rank; j < config::popSize; j+=num_procs) {
            //突然変異
            //ベースベクトルとその他二つのベクトルのindex
            assert(!lockahead.empty());
            auto [xb,xr1,xr2]=lockahead.back();
            lockahead.pop_back();
            //新しいベクトル、ベースベクトルとその他二つのベクトル
            std::array<double, config::constantSize> v,baseV,randV1,randV2;
            baseV=populations[xb].constant;
            randV1=populations[xr1].constant;
            randV2=populations[xr2].constant;
            //交叉
            v=crossingOver(baseV,randV1,randV2);
            double newError = calcError(v);
            if (transProb(temp, populations[j].error, newError) || !std::isfinite(populations[j].error)) {
                populations[j].constant = v;
                populations[j].error = newError;
            }
        }
    }
    MPI_Win_free(&win);
}

//最良個体の定数を返す
std::array<double, config::constantSize> differentialEvolution::best() {
    double minerror = DBL_MAX;
    std::array<double, config::constantSize> *ret = nullptr;
    for (auto& t : populations) {
        if (minerror > t.error) {
            minerror = t.error;
            ret = &t.constant;
        }
    }
    return *ret;
}

void differentialEvolution::putSim(const std::array<double, config::constantSize>& constant) {
    if(proc_rank!=0)return;
    const std::vector<speciesAmount<double>>& simulation = simulate(constant);
    cout <<"Time (min),1 (%),[PdPy*4]2+ (%),Py* (%),Pd214 cage (%),"<<std::endl;
    for (int i = 0; i<QASAP.size(); i++) {
        double t = QASAP[i].time;
        assert(0 <= t && t <= endTime);
        cout<<t<<", ";

        int nearestIndex = std::upper_bound(simTime.begin(),simTime.end(),t)-simTime.begin()-1;
        double nearestTime = simTime[nearestIndex];
        double difftime = t - nearestTime;
        stepResult res = calcNextStep(constant,simulation[nearestIndex],difftime);
        for (int j = 0; j < config::trackedSpecies; j++) {
            size_t idx= config::trackedIndex[j];
            assert(0 <= idx && idx < config::species);
            cout<<res.newState[idx]/config::fullConc[j] <<", ";
        }
        cout<<std::endl;
    }

}

void differentialEvolution::putCVODESim(const std::array<double, config::constantSize>& constant) {
    if(proc_rank!=0)return;
    N_Vector y = N_VNew_Serial(config::species, sunctx);
    for (int i = 0; i < config::species; ++i) {
        NV_Ith_S(y, i) = initialState[i];
    }
    CVodeReInit(cvode_mem, 0.0, y);
    CVodeSetUserData(cvode_mem, (void*)&constant);
    
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

//明示的なインスタンス化
typedef CppAD::AD<double> ADdouble;
template differentialEvolution::stepResult<double> differentialEvolution::calcNextStep(const std::array<double, config::constantSize>& reactConst, const speciesAmount<double>& data, double stepSize);
template std::vector<speciesAmount<double>> differentialEvolution::simulate(const std::array<double, config::constantSize>& constant);
template double differentialEvolution::calcError(const std::array<double, config::constantSize>& constant);
template differentialEvolution::stepResult<ADdouble> differentialEvolution::calcNextStep(const std::array<ADdouble, config::constantSize>& reactConst, const speciesAmount<ADdouble>& data, double stepSize);
template std::vector<speciesAmount<ADdouble>> differentialEvolution::simulate(const std::array<ADdouble, config::constantSize>& constant);
template ADdouble differentialEvolution::calcError(const std::array<ADdouble, config::constantSize>& constant);

template double differentialEvolution::calcErrorDP(const std::array<double, config::constantSize>& constant);