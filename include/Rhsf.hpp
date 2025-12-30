#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <map>
#include <charconv>

#include <nvector/nvector_serial.h>

#include "../include/constants.hpp"

namespace Rhsf {
	extern std::array<double, config::constantSize> rateConstants; //反応速度定数配列
    extern std::array<double, config::species+1> speciesData; //初期種量配列

	struct term {
		int add_to;       // 生成物のインデックス
		int duplicacy;     // 重複度
		int reactant1;     // 反応物1のインデックス
		int reactant2;     // 反応物2のインデックス、存在しない場合は -1
		int rateConstant;  // 反応速度定数のインデックス

		inline void calculate(std::span<double> ydot) const {
			ydot[add_to] += duplicacy * rateConstants[rateConstant] * speciesData[reactant1] * speciesData[reactant2];
		}

		auto operator<=>(const term&) const;
	};


	// call this function to setup Rhsf::terms from CSV data
	void makeRhsf();


	int rhsf(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data);

} // namespace Rhsf

