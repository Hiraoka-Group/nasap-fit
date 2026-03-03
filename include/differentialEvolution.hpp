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

using std::vector;

struct differentialEvolution {
	struct Config {
		std::string QASAPFile;
		std::string reactNetworkFile;
		int species = 0;//シミュレーション時 config.yamlから
		int constantSize = 0;//初期化時 config.yamlから
		int trackedSpecies = 0;//シミュレーション時 config.yamlから
		vector<std::string_view> trackedNames;//QASAPのcsv読み取り時 config.yamlから
		vector<int> trackedIndex;//QASAPのcsv読み取り時 config.yamlから
		vector<double> fullConc;//シミュレーション時 config.yamlから
		std::map<int,double> initConc;//シミュレーション時 config.yamlから
		double tolAbsError = 1e-09;//シミュレーション時
		double tolRelError = 3.2e-07;
		double scalar = 0.7;//optimize時
		double crossOver = 0.4;//optimize時
		double upperLim = 1e4;//popset時
		double lowerLim = 1e-3;//popset時
	};
	struct OptimizeResult {
		std::vector<double> constants;
		double error = DBL_MAX;
		explicit OptimizeResult(int constantSize) : constants((size_t)constantSize, 0.0) {}
	};
private:
	double endTime; //シミュレーション終了時間
	Config cfg;
	ReactionNetwork rxnNet;
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
	vector<int>indexOrder;

	sundials::Context sunctx;
	void* cvode_mem = CVodeCreate(CV_BDF, sunctx);

	vector<double> crossingOver(const vector<double>& baseV, const vector<double>& randV1, const vector<double>& randV2);

	void validateConstants(const vector<double>& constants);

	void sortByError(vector<OptimizeResult>& populations);
public:
// jac_fun_と res_fun_のセットアップ
	void setUpCasADiFunctions();

	const Config& constants() const { return cfg; }

	// expose reaction-network metadata (e.g., kind->index map)
	const ReactionNetwork& reactionNetwork() const { return rxnNet; }

	//平方残差和の計算（CVODEを用いる）
	double calcError(const vector<double>& constant);


	// ヘッセ行列の計算
	vector<vector<double>> getHessian(const vector<double>& point);

	vector<vector<double>> getHessian_parallel(const vector<double>& point);

	vector<vector<double>> pseudoHessian(const vector<double>& point);

	// 実験データのセット
	void setQASAPData(vector<vector<std::string>>& arg);

	// Constructor
	differentialEvolution(const Config& arg);

	// Levenberg-Marquardt法による最適化の実行
	OptimizeResult runLM(const vector<double>& theta0);
	vector<OptimizeResult> runLM(const vector<vector<double>>& thetaList);

	// 差分進化法の実行
	vector<OptimizeResult> Optimize(int maxGen, int popSize, double lowerLim = 1e-3, double upperLim = 1e4);

	vector<OptimizeResult> Optimize(vector<vector<double>> arg);

	void putCVODESim(const vector<double>& constant);
};