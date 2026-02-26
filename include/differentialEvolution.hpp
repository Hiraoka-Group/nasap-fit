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
#include "../include/reactionNetwork.hpp"


struct differentialEvolution {
private:
	double endTime; //シミュレーション終了時間
	ReactionNetwork rxnNet;
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
    casadi::Function res_fun_;     // 残差ベクトルを返す関数
    casadi::Function jac_fun_;     // ヤコビアンを返す関数
	casadi::Function SSR_jac_fun_; // 平方残差和とヤコビアンを返す関数
	casadi::Function SSR_hes_fun_; // 平方残差和とヘッセ行列を返す関数

	std::vector<double> initialState; //初期状態（ランタイムサイズ）
	std::vector<datum> QASAP;  //実験データ
	std::vector<individuals>populations; //エージェントの集団
	// flat buffers for MPI-safe sharing
	std::vector<double> populationsFlat; // size: popSize * constantSize
	std::vector<double> populationsErrorFlat; // size: popSize
	

	sundials::Context sunctx;
	void* cvode_mem = CVodeCreate(CV_BDF, sunctx);

	// jac_fun_と res_fun_のセットアップ
	void setUpCasADiFunctions();

	std::vector<double> crossingOver(const std::vector<double>& baseV, const std::vector<double>& randV1, const std::vector<double>& randV2);

public:
	// expose reaction-network metadata (e.g., kind->index map)
	const ReactionNetwork& reactionNetwork() const { return rxnNet; }

	//平方残差和の計算（CVODEを用いる）
	double calcError(const std::vector<double>& constant);

	void addStepCountCV(const std::vector<double>& constant);

	// ヘッセ行列の計算
	std::vector<std::vector<double>> getHessian(const std::vector<double>& point);

	std::vector<std::vector<double>> getHessian_parallel(const std::vector<double>& point);

	std::vector<std::vector<double>> pseudoHessian(const std::vector<double>& point);

	// 実験データのセット
	void setQASAPData(std::vector<std::vector<double>>& arg);

	void setPop(int idx, const std::vector<double>& theta);

	std::vector<double> getPop(int idx);

	double getPopError(int idx);

	void evaluate();
	// Constructor
	differentialEvolution(std::vector<std::vector<double>>& arg);

	// Levenberg-Marquardt法による最適化の実行
	void runLM(int idx);
	void runLM(std::vector<int>& indices);

	// 差分進化法の実行
	void Optimize();
	// 最良個体のインデックスを返す
	int best();

	void sortPopulationsByError();

	void putCVODESim(const std::vector<double>& constant);
	void DEBUG();
};