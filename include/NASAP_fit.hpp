#pragma once


#include <iostream>
#include <iomanip>
#include <cfloat>
#include <string>
#include <array>
#include <vector>
#include <cassert>
#include <fstream>
#include <cstdint>


#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_core.hpp>
#include <sunlinsol/sunlinsol_pcg.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sundials/sundials_context.hpp>

#include "../include/xorshift.hpp"
#include "../include/reactionNetwork.hpp"
#include "../include/MPIEnvironment.hpp"

using std::vector;

struct NASAP_fit {
	enum class LogLevel : uint8_t {
		quiet = 0,
		normal = 1,
		verbose = 2,
	};

	struct Config {
		std::string QASAPFile;
		std::string reactNetworkFile;
		int species = 0;//シミュレーション時 config.yamlから
		int constantSize = 0;//初期化時 config.yamlから
		int trackedSpecies = 0;//シミュレーション時 config.yamlから
		vector<std::string> trackedNames;//QASAPのcsv読み取り時 config.yamlから
		vector<int> trackedIndex;//QASAPのcsv読み取り時 config.yamlから
		vector<double> fullConc;//シミュレーション時 config.yamlから
		std::map<int,double> initConc;//シミュレーション時 config.yamlから
		double tolAbsError = 1e-09;//シミュレーション時
		double tolRelError = 3.2e-07;
		double scalar = 0.7;//optimize時
		double crossOver = 0.4;//optimize時
		double upperLim = 1e4;//popset時
		double lowerLim = 1e-3;//popset時
		// CVODEの内部ステップ数上限（CVodeSetMaxNumSteps）
		long int cvodeMaxNumSteps = 10000;
		// ログ出力レベル
		LogLevel logLevel = LogLevel::verbose;
	};
	struct OptimizeResult {
		std::vector<double> constants;
		double error = DBL_MAX;
		explicit OptimizeResult(int constantSize) : constants((size_t)constantSize, 0.0) {}
	};
	//最適化の終了条件を記述する構造体
	struct TerminationCondition {
		//最大世代数
		int maxIter=INT_MAX;
		//実行時間の上限（秒）
		double timeLimit=DBL_MAX;
		//パラメータ空間における最適解の近傍の大きさの基準値(DEでは無効)
		double xtol=0;
		//目的関数の改善率の基準値
		double ftolAbs=0;
		//目的関数の改善率の基準値
		double ftolRel=0;
		//目的関数が目標値を下回ったときに終了
		double targetError=0;
		//停滞期間の基準値（世代数）（この値を超過すると終了）
		int stall=INT_MAX;
	};
private:
	double endTime; //シミュレーション終了時間
	Config cfg;
	ReactionNetwork rxnNet;
	MpiEnvironment mpi_env;
	struct datum {
		double time;
		std::vector<double>state;
	};

	std::vector<double> initialState; //初期状態（ランタイムサイズ）
	std::vector<datum> QASAP;  //実験データ
	vector<int>indexOrder;

	sundials::Context sunctx;
	void* cvode_mem = CVodeCreate(CV_BDF, sunctx);
	N_Vector cvode_constraints_ = nullptr;
	SUNMatrix J;
    SUNLinearSolver LS;
	N_Vector y;
    N_Vector yQ0;

	vector<double> crossingOver(const vector<double>& baseV, const vector<double>& randV1, const vector<double>& randV2, uint64_t seed, int gen, int j);

	void validateConstants(const vector<double>& constants);

	void sortByError(vector<OptimizeResult>& populations);

	vector<OptimizeResult> runDE_single(int maxGen, int popSize, double lowerLim, double upperLim, const TerminationCondition& termCond, uint64_t seed = 1);
	vector<OptimizeResult> runDE_single(const vector<vector<double>>& arg, const TerminationCondition& termCond, uint64_t seed = 1);

	vector<vector<double>> makeRandomPopulation(int popSize, double lower, double upper, uint64_t seed);
public:
	// CasADi 依存の関数は除外
	#if 0
	void setUpCasADiFunctions();
	#endif

	const Config& constants() const { return cfg; }

	// Expose reaction-network term index (kind -> index)
	const std::map<std::string, int>& termIndex() const { return rxnNet.termIndex; }

	//reactionNetwork内の素反応の数を返す
	const int reactionCount() const { return rxnNet.data.size(); }

	//平方残差和の計算（CVODEを用いる）
	double calcError(const vector<double>& constant);


	// 実験データのセット
	void setQASAPData(const vector<vector<std::string>>& arg);

	// Constructor
	NASAP_fit(const Config& arg);

	~NASAP_fit();

	// Levenberg-Marquardt法（CasADi 依存のため除外）
	#if 0
	OptimizeResult runLM(const vector<double>& theta0, const TerminationCondition& termCond);
	vector<OptimizeResult> runLM(const vector<vector<double>>& thetaList, const TerminationCondition& termCond);
	#endif

	// 差分進化法の実行
	vector<OptimizeResult> runDE(int popSize, double lowerLim, double upperLim, const TerminationCondition& termCond, uint64_t seed = 1);

	vector<OptimizeResult> runDE(vector<vector<double>> arg, const TerminationCondition& termCond, uint64_t seed = 1);

	void putCVODESim(const vector<double>& constant);

	struct SimulationResult {
		struct ReactionProgressResult{
			vector<int> reaction_ids;
			//累積反応進行度
			vector<vector<double>> J; // size: [reaction_id][time_index]
			vector<std::string> reaction_labels;
		};
		std::string status; // "success"|"failed"
		int timePoints;
		vector<double> t;
		vector<vector<double>>y; // size: [species][time_index]
		ReactionProgressResult reactionProgress;
	};

	SimulationResult simulate(const vector<double>& t, const vector<double>& constant, const vector<int>& reaction_ids);
};