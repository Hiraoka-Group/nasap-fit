#include <iostream>
#include <iomanip>
#include <cfloat>
#include <string>
#include <array>
#include <vector>
#include <cassert>
#include <fstream>

#include "xorshift.hpp"
#include "ODE.hpp"
//シミュレーションの1ステップごとの時間差
constexpr double timeStep = 1.0 / (1 << 4);
//差分進化法のパラメータ
constexpr double scalar = 0.5, crossOver = 0.5;
//反応速度定数の上限下限
constexpr double upperLim = 2, lowerLim = 0.1;


class QASAP_data {
	struct datum {
		double time;
		speciesAmount state;
	};
	std::vector<datum> data;
public:;
	  QASAP_data(){}
	  QASAP_data(std::vector<std::vector<double>>arg) {
		  for (const std::vector<double>& vec : arg) {
			  assert(vec.size() == species + 1);
			  data.push_back({ vec[0], speciesAmount(std::vector<double>(vec.begin() + 1, vec.end())) });
		  }
	  }
	  double getSSR(const std::vector<speciesAmount>& simulatedValue) {
		  double SSR = 0;
		  for (int i = 0; i < data.size(); i++) {
			  int idx = data[i].time / timeStep;
			  assert(0 <= idx && idx < simulatedValue.size());
			  speciesAmount nearest = simulatedValue[idx];
			  for (int j = 0; j < species; j++) {
				  SSR += (data[i].state[j] - nearest[j]) * (data[i].state[j] - nearest[j]);
			  }
		  }
		  return SSR;
	  }
};

speciesAmount calcNextStep(const std::array<double, constantSize>& reactConst, const speciesAmount& data) {
	speciesAmount val1, val2, val3, val4, k1, k2, k3, k4;
	val1 = data;
	k1 = diffCoeff(reactConst, val1);
	val2 = val1 + k1 * timeStep / 2;
	k2 = diffCoeff(reactConst, val2);
	val3 = val1 + k2 * timeStep / 2;
	k3 = diffCoeff(reactConst, val3);
	val4 = val1 + k3 * timeStep;
	k4 = diffCoeff(reactConst, val4);
	return val1 + (k1 + 2 * k2 + 2 * k3 + k4) * timeStep / 6;
	auto t = (k1 + k2);
}

struct differentialEvolution {
	int loopsNumber = 50; //差分進化法を回す回数
	int popSize = (int)pow(2.0, constantSize); //差分進化法のエージェント数
private:;
	struct individuals {
		std::array<double, constantSize>constant;
		double error = -1;//平方残差和
	};
	QASAP_data QASAP;  //実験データ
	std::vector<individuals>populations;
	std::vector<speciesAmount>simulation;
	//平方残差和の計算
	double calcError(const std::array<double, constantSize>& constant) {
		simulation = { speciesAmount({ 1,1,0,0 }) };
		for (int k = 0; k * timeStep <= 21; k++) {
			simulation.push_back(calcNextStep(constant, simulation[k]));
		}
		return QASAP.getSSR(simulation);
	}
public:;
	differentialEvolution(std::vector<std::vector<double>> arg) {
		QASAP = QASAP_data(arg);
		popSize = 20;
	}
	void Optimize() {
		populations = std::vector<individuals>(popSize);
		for (int i = 0; i < popSize; i++) {
			for (int j = 0; j < constantSize; j++) {
				populations[i].constant[j] = randbetExp(lowerLim, upperLim);
			}
			populations[i].error = calcError(populations[i].constant);
		}
		for (int i = 0; i < loopsNumber; i++) {
			for (int j = 0; j < popSize; j++) {
				//突然変異
				//ベースベクトルとその他二つのベクトルのindex
				int xb = myRand(popSize), xr1, xr2;
				do {xr1 = myRand(popSize);} while (xb == xr1);
				do {xr2 = myRand(popSize);} while (xb == xr1 && xr1 == xr2);
				int jr = myRand(constantSize);
				std::array<double, constantSize> v;
				for (int k = 0; k < constantSize; k++) {
					//交叉
					if (jr == k || myRand.prob() < crossOver) {
						v[k] = populations[xb].constant[k] * exp(scalar * log(populations[xr1].constant[k] / populations[xr2].constant[k]));
					}
					else v[k] = populations[j].constant[k];
				}
				double newScore = calcError(v);
				if (newScore < populations[j].error) {
					populations[j].constant = v;
					populations[j].error = newScore;
					if (newScore < 0.0001) {
						//std::cout << i << std::endl;
						return;
					}
				}
			}
		}
	}
	std::array<double, constantSize> best() {
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
	void DEBUG() {
		for (int i = 0; i < popSize; i++) {
			for (int j = 0; j < constantSize; j++) {
				std::cout << populations[i].constant[j] << " ";
			}
			std::cout << std::endl;
		}
	}
};

signed main() {
	std::ifstream ifs("QASAP.txt");
	if (!ifs.is_open()) {
		std::cerr << "ファイルを開けませんでした" << std::endl;
		return 1;
	}
	std::vector<std::vector<double>>input;
	std::string line;
	while (std::getline(ifs, line)) {
		std::vector<double> numbers;
		std::string s;
		for (char c : line) {
			if (c == ' ') {
				if(s != "")numbers.push_back(stod(s));
				s = "";
			}
			else {
				s += c;
			}
		}
		if (s != "")numbers.push_back(stod(s));
		if (numbers.size() == 0)break;
		assert(numbers.size() == species + 1);
		input.push_back(numbers);
	}
	differentialEvolution diffEvo(input);
	diffEvo.Optimize();
	auto arr = diffEvo.best();
	for (auto t : arr) {
		std::cout << t << " ";
	}
	std::cout << std::endl;

	
}

/*
g++ otameshi.cpp -o otameshi && ./otameshi
*/