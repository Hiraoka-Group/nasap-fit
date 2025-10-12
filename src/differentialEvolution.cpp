#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cfloat>
#include <string>
#include <array>
#include <vector>
#include <cassert>
#include <fstream>

#include <mpi.h>

#include "../include/differentialEvolution.hpp"
#include "../include/constants.hpp"
#include "../include/xorshift.hpp"
#include "../include/ODE.hpp"


xorshift myRand;
int cnt=0;
double timeStep=0.001;
extern int num_procs, proc_rank;

void print(std::array<double, constantSize>constant){
    std::cout<<"ar: ";
    for(int i=0; i<constantSize; i++){
        std::cout<<constant[i]<<" ";
    }
    std::cout<<std::endl;
    return;
}


//シミュレーションにおける、次のステップの計算
differentialEvolution::stepResult differentialEvolution::calcNextStep(const std::array<double, constantSize>& reactConst, const speciesAmount& data) {
    stepResult ret;
    speciesAmount val1, val2, val3, val4,val5,val6,val7, slope1, slope2, slope3, slope4,slope5,slope6,slope7, order4;
    double maxerror;
    while(1){
        maxerror=0.0;
        val1 = data;
        slope1 = diffCoeff(reactConst, val1);
        val2 = val1 + timeStep * slope1 * (1.0/5);
        slope2 = diffCoeff(reactConst, val2);
        val3 = val1 + timeStep * (slope1 * (3.0/40) + slope2 * (9.0/40));
        slope3 = diffCoeff(reactConst, val3);
        val4 = val1 + timeStep * (slope1 * (44.0/45) + slope2 * (-56.0/15) + slope3 * (32.0/9));
        slope4 = diffCoeff(reactConst, val4);
        val5 = val1 + timeStep * (slope1 * (19372.0/6561) + slope2 * (-25360.0/2187) + slope3 * (64448.0/6561) + slope4 * (-212.0/729));
        slope5 = diffCoeff(reactConst, val5);
        val6 = val1 + timeStep * (slope1 * (9017.0/3168) + slope2 * (-355.0/33) + slope3 * (46732.0/5247) + slope4 * (49.0/176) + slope5 * (-5103.0/18656));
        slope6 = diffCoeff(reactConst, val6);
        val7 = val1 + timeStep * (slope1 * (35.0/384) + slope3 * (500.0/1113) + slope4 * (125.0/192) + slope5 * (-2187.0/6784) + slope6 * (11.0/84));
        slope7 = diffCoeff(reactConst, val7);
        order4= val1 + timeStep * (slope1 * (5179.0/57600) + slope3 * (7571.0/16695) + slope4 * (393.0/640) + slope5 * (-92097.0/339200) + slope6 * (187.0/2100) + slope7 * (1.0/40));
        const speciesAmount& order5 = val7;

        for(int i=0; i<species; i++){
            double e = std::abs(order4[i]-order5[i]);
            if(maxerror < e) maxerror=e;
        }

        if(maxerror<=tolerableError){//ステップ成功
            ret.usedStepSize=timeStep;
            ret.newState=order5;
            double h_new;
            if(maxerror==0)h_new = timeStep * 2;
            else{
                h_new=safetyConstant*timeStep*std::pow(tolerableError/maxerror,(1.0/5));
                h_new=std::clamp(h_new, timeStep*0.25, timeStep*1.5);
            }
            timeStep=h_new;
            return ret;
        }else{//ステップ失敗
        timeStep=safetyConstant*timeStep*std::pow(tolerableError/maxerror,(1.0/5));
        }
    }
}


std::array<double, constantSize> differentialEvolution::crossingOver(const std::array<double, constantSize>& baseV, const std::array<double, constantSize>& randV1, const std::array<double, constantSize>& randV2){
    std::array<double, constantSize> v;
    int jr = myRand(constantSize);
    for (int k = 0; k < constantSize; k++) {
        //交叉
        if (jr == k || myRand.prob() < crossOver) {
            v[k] = baseV[k] * pow(randV1[k] / randV2[k], scalar);
            v[k]=std::clamp(v[k],lowerLim,upperLim);
        }
        else v[k] = baseV[k];

        if(!std::isfinite(v[k])){
            print(baseV);
            print(randV1);
            print(randV2);
            print(v);
        }
    }
    return v;
}

void differentialEvolution::simulate(const std::array<double, constantSize>& constant){
    simulation = {speciesAmount()}; 
    for (int i = 0; i < trackedSpecies; i++) {
        simulation[0][trackedIndex[i]] = fullConc[i]*QASAP[0].state[i];
    }
    simTime={0.0};

    //シミュレーション
    for (int k = 0; simTime.back() <= endTime; k++) {
        stepResult res=calcNextStep(constant, simulation[k]);
        simulation.push_back(res.newState);
        simTime.push_back(simTime.back()+res.usedStepSize);
    }
}

//平方残差和の計算
double differentialEvolution::calcError(const std::array<double, constantSize>& constant) {
    simulation[0] = speciesAmount(); 
    for (int i = 0; i < trackedSpecies; i++) {
        simulation[0][trackedIndex[i]] = fullConc[i]*QASAP[0].state[i];
    }
    //シミュレーション
    simulate(constant);
    double SSR = 0; 
    for (int i = 0; i < QASAP.size(); i++) {
        assert(0 <= QASAP[i].time && QASAP[i].time <= simTime.back());

        int nearestIndex = std::upper_bound(simTime.begin(),simTime.end(),QASAP[i].time)-simTime.begin()-1;
        double nearestTime = simTime[nearestIndex];
        double difftime = QASAP[i].time - nearestTime;
        timeStep=difftime;
        stepResult res = calcNextStep(constant,simulation[nearestIndex]);
        for (int j = 0; j < trackedSpecies; j++) {
            size_t idx= trackedIndex[j];
            assert(0 <= idx && idx < species);
            SSR += (QASAP[i].state[j] - res.newState[idx]/fullConc[j]) * (QASAP[i].state[j] - res.newState[idx]/fullConc[j]);
        }
    }
    return SSR;
}

//実験データのセット
void differentialEvolution::setData(std::vector<std::vector<double>>& arg) {
    for (const std::vector<double>& vec : arg) {
        assert(vec.size() == trackedSpecies + 1);
        QASAP.push_back({ vec[0],std::vector<double>({vec.begin()+1, vec.begin()+trackedSpecies+1}) });
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
    populations = std::vector<individuals>(popSize);
    for(int i=0; i<popSize; i++){
        ifs >> populations[i].error;
        for(int j=0; j<constantSize; j++){
            ifs >> populations[i].constant[j];
        }
    }
}
differentialEvolution::differentialEvolution(std::vector<std::vector<double>>& arg) {
    setData(arg);
    myRand=xorshift(1+proc_rank);
    endTime = arg.back()[0];
    simulation = std::vector<speciesAmount>(1);
    populations = std::vector<individuals>(popSize);
    for (int i = proc_rank; i < popSize; i+=num_procs) {
        for (int j = 0; j < constantSize; j++) {
            populations[i].constant[j] = randbetExp(lowerLim, upperLim);
        }
        populations[i].error = calcError(populations[i].constant);
    }
}

void differentialEvolution::Optimize() {
    std::vector<differentialEvolution::individuals> popRecvContainer(popSize);
    for (int i = 0; i < loopsNumber; i++) {
        if(proc_rank==0)std::cout<<i<<"\n";
        

        std::vector<std::array<int, 3>> lockahead;
        std::vector<int> indicesToVisit;
        auto pushIndex = [&](int idx){
            if(idx%num_procs==proc_rank)return false;
            else{
                indicesToVisit.push_back(idx);
                return true;
            }
        };
        for (int j = proc_rank; j < popSize; j+=num_procs){
            int xb = myRand(popSize), xr1=myRand(popSize), xr2=myRand(popSize);
            while (xb == xr1) {xr1 = myRand(popSize);}
            while (xb == xr2 || xr1 == xr2) {xr2 = myRand(popSize);}
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
        MPI_Win win;
        MPI_Win_create(populations.data(), sizeof(individuals)*popSize, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_fence(0, win);
        for(int idx : indicesToVisit){
            assert(idx%num_procs!=proc_rank);
            MPI_Get(&populations[idx], sizeof(individuals), MPI_BYTE, idx%num_procs, sizeof(individuals)*idx, sizeof(individuals), MPI_BYTE, win);
        }
        MPI_Win_fence(0, win);
        MPI_Win_free(&win);
        for (int j = proc_rank; j < popSize; j+=num_procs) {
            //突然変異
            //ベースベクトルとその他二つのベクトルのindex
            assert(!lockahead.empty());
            auto [xb,xr1,xr2]=lockahead.back();
            lockahead.pop_back();
            //新しいベクトル、ベースベクトルとその他二つのベクトル
            std::array<double, constantSize> v,baseV,randV1,randV2;
            baseV=populations[xb].constant;
            randV1=populations[xr1].constant;
            randV2=populations[xr2].constant;
            //交叉
            v=crossingOver(baseV,randV1,randV2);

            double newError = calcError(v);
            if (newError < populations[j].error || !std::isfinite(populations[j].error)) {
                populations[j].constant = v;
                populations[j].error = newError;
            }
        }
    }

}

//最良個体の定数を返す
std::array<double, constantSize> differentialEvolution::best() {
    double minerror = DBL_MAX;
    std::array<double, constantSize> *ret = nullptr;
    for (auto& t : populations) {
        if (minerror > t.error) {
            minerror = t.error;
            ret = &t.constant;
        }
    }
    return *ret;
}

void differentialEvolution::putSim(const std::array<double, constantSize>& constant){
    simulate(constant);
    std::cout<<"----------------------------\n";
    for (int i = 0; i < QASAP.size(); i++) {
        assert(0 <= QASAP[i].time && QASAP[i].time <= simTime.back());
        std::cout<<QASAP[i].time<<", ";

        int nearestIndex = std::upper_bound(simTime.begin(),simTime.end(),QASAP[i].time)-simTime.begin()-1;
        double nearestTime = simTime[nearestIndex];
        double difftime = QASAP[i].time - nearestTime;
        timeStep=difftime;
        stepResult res = calcNextStep(constant,simulation[nearestIndex]);
        for (int j = 0; j < trackedSpecies; j++) {
            size_t idx= trackedIndex[j];
            assert(0 <= idx && idx < species);
            std::cout<<res.newState[idx]/fullConc[j]<<", ";
        }
        std::cout<<std::endl;
    }
}

void differentialEvolution::DEBUG() {
    std::cout<<"Population Error and Constants:\n";
    for (int i = 0; i < popSize; i++) {
        std::cout<< populations[i].error << std::endl;
        for (int j = 0; j < constantSize; j++) {
            std::cout << populations[i].constant[j] << " ";
        }
        std::cout << std::endl;
    }
}