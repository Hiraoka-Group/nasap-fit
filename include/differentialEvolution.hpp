#pragma once


#include <iostream>
#include <iomanip>
#include <cfloat>
#include <string>
#include <array>
#include <vector>
#include <cassert>
#include <fstream>

#include "../include/constants.hpp"
#include "../include/xorshift.hpp"
#include "../include/ODE.hpp"


struct differentialEvolution {
	int loopsNumber = 70; //差分進化法を回す回数
private:
	double endTime; //シミュレーション終了時間
	struct individuals {
		std::array<double, constantSize>constant;
		double error = DBL_MAX;//平方残差和
	};
	static_assert(sizeof(individuals) == sizeof(double) * (constantSize + 1), "構造体のサイズが不正です");
	struct datum {
		double time;
		std::vector<double>state;
	};
	//speciesAmount, 用いたステップ幅, 推定される最適なステップ幅
	struct stepResult{
		speciesAmount newState;
		double usedStepSize;
		double newStepSize;
	};
	std::vector<datum> QASAP;  //実験データ
	std::vector<individuals>populations; //エージェントの集団
	std::vector<double>simTime;//simulationに対応する時間
	std::vector<speciesAmount>simulation; //シミュレーションに用いる配列

	//シミュレーションにおける、次のステップの計算
    stepResult calcNextStep(const std::array<double, constantSize>& reactConst, const speciesAmount& data);

	std::array<double, constantSize> crossingOver(const std::array<double, constantSize>& baseV, const std::array<double, constantSize>& randV1, const std::array<double, constantSize>& randV2);

	void simulate(const std::array<double, constantSize>& constant);
    
    double getSSR(const std::vector<speciesAmount>& simulatedValue);
	//平方残差和の計算
	double calcError(const std::array<double, constantSize>& constant);
public:
    //実験データのセット
    void setData(std::vector<std::vector<double>>& arg);

	void setPop();
	//Constructor
	differentialEvolution(std::vector<std::vector<double>>& arg);
	//最適化の実行
	void Optimize();
    //最良個体の定数を返す
	std::array<double, constantSize> best();

	void putSim(const std::array<double, constantSize>& constant);

	//デバッグ用
	void DEBUG();
};