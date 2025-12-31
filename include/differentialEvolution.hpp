#pragma once


#include <iostream>
#include <iomanip>
#include <cfloat>
#include <string>
#include <array>
#include <vector>
#include <cassert>
#include <fstream>


#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_core.hpp>
#include <sunlinsol/sunlinsol_pcg.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sundials/sundials_context.hpp>

#include "../include/constants.hpp"
#include "../include/xorshift.hpp"
#include "../include/ODE.hpp"


struct differentialEvolution {
private:
	double endTime; //シミュレーション終了時間
	struct individuals {
		std::array<double, config::constantSize>constant;
		double error = DBL_MAX;//平方残差和
	};
	static_assert(sizeof(individuals) == sizeof(double) * (config::constantSize + 1), "構造体のサイズが不正です");
	struct datum {
		double time;
		std::vector<double>state;
	};
	template<class T>
	struct stepResult {
		speciesAmount<T> newState;
		double usedStepSize;
		double newStepSize;
	};
	speciesAmount<double>initialState; //初期状態
	std::vector<datum> QASAP;  //実験データ
	std::vector<individuals>populations; //エージェントの集団
	std::vector<speciesAmount<double>>simulation; //シミュレーション結果
	std::vector<double>simTime; //シミュレーション時間

	sundials::Context sunctx;
	void* cvode_mem = CVodeCreate(CV_BDF, sunctx);

	//シミュレーションにおける、次のステップの計算
	template<class T>
    stepResult<T> calcNextStep(const std::array<T, config::constantSize>& reactConst, const speciesAmount<T>& data, double stepSize);

	std::array<double, config::constantSize> crossingOver(const std::array<double, config::constantSize>& baseV, const std::array<double, config::constantSize>& randV1, const std::array<double, config::constantSize>& randV2);

	template<class T>
	std::vector<speciesAmount<T>> simulate(const std::array<T, config::constantSize>& constant);

	public:
	//平方残差和の計算
	template<class T>
	T calcErrorDP(const std::array<T, config::constantSize>& constant);

	template<class T>
	T calcError(const std::array<T, config::constantSize>& constant);
	
	void addStepCountCV(const std::array<double, config::constantSize>& constant);
	void addStepCountDP(const std::array<double, config::constantSize>& constant);

	std::vector<double> getJacobian(const std::array<double, config::constantSize>& point);
	//ヘッセ行列の計算
	std::vector<std::vector<double>> getHessian(const std::array<double, config::constantSize>& point);

    //実験データのセット
    void setData(std::vector<std::vector<double>>& arg);

	void setPop();
	//Constructor
	differentialEvolution(std::vector<std::vector<double>>& arg);
	void Optimize();
    //最良個体の定数を返す
	void best(std::array<double, config::constantSize>& ret, double& minerror);

	void putSim(const std::array<double, config::constantSize>& constant);

	void putCVODESim(const std::array<double, config::constantSize>& constant);
	void DEBUG();
};