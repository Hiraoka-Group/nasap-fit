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
	int loopsNumber = 10; //差分進化法を回す回数
	int popSize = 128; //差分進化法のエージェント数
private:
	double endTime; //シミュレーション終了時間
	struct individuals {
		std::array<double, constantSize>constant;
		double error = DBL_MAX;//平方残差和
	};
	struct datum {
		double time;
		std::vector<double>state;
	};
	std::vector<datum> QASAP;  //実験データ
	std::vector<individuals>populations; //エージェントの集団
	std::vector<speciesAmount>simulation; //シミュレーションに用いる配列

	//シミュレーションにおける、次のステップの計算
    speciesAmount calcNextStep(const std::array<double, constantSize>& reactConst, const speciesAmount& data);
    
    double getSSR(const std::vector<speciesAmount>& simulatedValue);
	//平方残差和の計算
	double calcError(const std::array<double, constantSize>& constant);
public:
    //実験データのセット
    void setData(std::vector<std::vector<double>>& arg);

	void setPop();

	differentialEvolution(std::vector<std::vector<double>>& arg);
	void Optimize();
    //最良個体の定数を返す
	std::array<double, constantSize> best();
	void DEBUG();
};