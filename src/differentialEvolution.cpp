#include <iostream>
#include <iomanip>
#include <cfloat>
#include <string>
#include <array>
#include <vector>
#include <cassert>
#include <fstream>

#include "../include/differentialEvolution.hpp"
#include "../include/constants.hpp"
#include "../include/xorshift.hpp"
#include "../include/ODE.hpp"

xorshift myRand;
int cnt=0;

//シミュレーションにおける、次のステップの計算
speciesAmount differentialEvolution::calcNextStep(const std::array<double, constantSize>& reactConst, const speciesAmount& data) {
    speciesAmount val1, val2, val3, val4, slope1, slope2, slope3, slope4;
    val1 = data;
    slope1 = diffCoeff(reactConst, val1);
    val2 = val1 + slope1 * timeStep / 2;
    slope2 = diffCoeff(reactConst, val2);
    val3 = val1 + slope2 * timeStep / 2;
    slope3 = diffCoeff(reactConst, val3);
    val4 = val1 + slope3 * timeStep;
    slope4 = diffCoeff(reactConst, val4);
    return val1 + (slope1 + 2 * slope2 + 2 * slope3 + slope4) * timeStep / 6;
}

double differentialEvolution::getSSR(const std::vector<speciesAmount>& simulatedValue) {
    double SSR = 0; 
    for (int i = 0; i < QASAP.size(); i++) {
        int nearestIndex = std::round(QASAP[i].time / timeStep);
        assert(0 <= nearestIndex && nearestIndex < simulatedValue.size());
        speciesAmount nearest = simulatedValue[nearestIndex];
        for (int j = 0; j < trackedSpecies; j++) {
            size_t idx= trackedIndex[j];
            assert(0 <= idx && idx < species);
            SSR += (QASAP[i].state[j] - nearest[idx]/fullConc[j]) * (QASAP[i].state[j] - nearest[idx]/fullConc[j]);
        }
    }
    return SSR;
}
//平方残差和の計算
double differentialEvolution::calcError(const std::array<double, constantSize>& constant) {
    simulation[0] = speciesAmount(); 
    for (int i = 0; i < trackedSpecies; i++) {
        simulation[0][trackedIndex[i]] = fullConc[i]*QASAP[0].state[i];
    }
    //シミュレーション
    for (int k = 0; k * timeStep <= endTime; k++) {
        simulation[k+1]=calcNextStep(constant, simulation[k]);
    }
    return getSSR(simulation);
}


//実験データのセット
void differentialEvolution::setData(std::vector<std::vector<double>>& arg) {
    for (const std::vector<double>& vec : arg) {
        assert(vec.size() == trackedSpecies + 1);
        QASAP.push_back({ vec[0],std::vector<double>({vec.begin()+1, vec.begin()+trackedSpecies+1}) });
    }
}
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
    endTime = arg.back()[0];
    simulation = std::vector<speciesAmount>((int)(endTime / timeStep) + 2);
}
void differentialEvolution::Optimize() {
    /*
    populations = std::vector<individuals>(popSize);
    for (int i = 0; i < popSize; i++) {
        for (int j = 0; j < constantSize; j++) {
            populations[i].constant[j] = randbetExp(lowerLim, upperLim);
        }
        populations[i].error = calcError(populations[i].constant);
    }
        */
    for (int i = 0; i < loopsNumber; i++) {
        std::cout<<i<<std::endl;
        for (int j = 0; j < popSize; j++) {
            //突然変異
            //ベースベクトルとその他二つのベクトルのindex
            int xb = myRand(popSize), xr1, xr2;
            do {xr1 = myRand(popSize);} while (xb == xr1);
            do {xr2 = myRand(popSize);} while (xb == xr1 || xr1 == xr2);
            int jr = myRand(constantSize);
            std::array<double, constantSize> v;
            for (int k = 0; k < constantSize; k++) {
                //交叉
                if (jr == k || myRand.prob() < crossOver) {
                    v[k] = populations[xb].constant[k] * exp(scalar * log(populations[xr1].constant[k] / populations[xr2].constant[k]));
                    if(v[k]<lowerLim) v[k]=lowerLim;
                    if(v[k]>upperLim) v[k]=upperLim;
                }
                else v[k] = populations[j].constant[k];
            }
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
void differentialEvolution::DEBUG() {
    for (int i = 0; i < popSize; i++) {
        std::cout<< populations[i].error << std::endl;
        for (int j = 0; j < constantSize; j++) {
            std::cout << populations[i].constant[j] << " ";
        }
        std::cout << std::endl;
    }
}