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

	struct term {
		int duplicacy;     // 重複度
		int reactant1;     // 反応物1のインデックス
		int reactant2;     // 反応物2のインデックス、存在しない場合は -1
		int rateConstant;  // 反応速度定数のインデックス

		// 一次反応かどうか
		bool isFirstOrder() const;

		double calculate(std::span<const double> sp, std::span<const double> k) const;
	};

	extern std::map<std::string, int> termIndex;

	// call this function to setup Rhsf::terms from CSV data
	void makeRhsf();


	int rhsf(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data);

} // namespace Rhsf

