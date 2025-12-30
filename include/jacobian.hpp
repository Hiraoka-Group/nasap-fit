#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <map>
#include <charconv>

#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_sparse.h>

#include "../include/constants.hpp"

namespace jacobian {
    extern int nonZeros; //ヤコビ行列の非ゼロ要素数

    extern std::array<double, config::constantSize> rateConstants; //反応速度定数配列
    extern std::array<double, config::species+1> speciesData; //初期種量配列

	struct term {
		int duplicacy;     // 重複度
		int reactant;     // 反応物のインデックス、存在しない場合は config::species
		int rateConstant;  // 反応速度定数のインデックス

		inline double calculate() const {
			return duplicacy * rateConstants[rateConstant] * speciesData[reactant];
		}

		auto operator<=>(const term&) const;

	};

		// call this function to setup jacobian data from CSV data
	void makeJacobian();

	int JacFn(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
}