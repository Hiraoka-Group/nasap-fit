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
#include <casadi/casadi.hpp>

#include "../include/constants.hpp"
#include "../include/xorshift.hpp"


struct differentialEvolution {
private:
	double endTime; //シミュレーション終了時間
	struct individuals {
		std::vector<double> constant;
		double error = DBL_MAX;//平方残差和
		individuals() : constant(config::constantSize) {}
	};
	struct datum {
		double time;
		std::vector<double>state;
	};
	casadi::Function integrator_;  // build_integrator の結果
    casadi::Function res_fun_;     // 残差評価用
    casadi::Function jac_fun_;     // ヤコビアン評価用

	std::vector<double> initialState; //初期状態（ランタイムサイズ）
	std::vector<datum> QASAP;  //実験データ
	std::vector<individuals>populations; //エージェントの集団
	// flat buffers for MPI-safe sharing
	std::vector<double> populationsFlat; // size: popSize * constantSize
	std::vector<double> populationsErrorFlat; // size: popSize

	sundials::Context sunctx;
	void* cvode_mem = CVodeCreate(CV_BDF, sunctx);

	// jac_fun_と res_fun_のセットアップ
	void setUpJacobian();

	std::vector<double> crossingOver(const std::vector<double>& baseV, const std::vector<double>& randV1, const std::vector<double>& randV2);

public:
	//平方残差和の計算（CVODEを用いる）
	template<class T>
	T calcError(const std::vector<T>& constant);

	void addStepCountCV(const std::vector<double>& constant);

	// ヘッセ行列の計算
	std::vector<std::vector<double>> getHessian(const std::vector<double>& point);

	// 実験データのセット
	void setData(std::vector<std::vector<double>>& arg);

	void setPop();
	// Constructor
	differentialEvolution(std::vector<std::vector<double>>& arg);

	// Levenberg-Marquardt法による最適化の実行
	void runLM(int idx);

	// 差分進化法の実行
	void Optimize();
	// 最良個体の定数を返す
	void best(std::vector<double>& ret, double& minerror);

	void putCVODESim(const std::vector<double>& constant);
	void DEBUG();
};