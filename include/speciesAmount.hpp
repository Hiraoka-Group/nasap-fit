#pragma once
#include <array>
#include <vector>
#include "constants.hpp"


class speciesAmount {
	std::array<double, species>amount;
public:;
	speciesAmount();
	speciesAmount(std::vector<double> arg);
	double& operator[](int i);
	const double& operator[](int i) const;
	speciesAmount operator+(const speciesAmount& other);
	speciesAmount operator/(const double num);
	void Debug();
};
speciesAmount operator*(const speciesAmount& leftVal, const double rightVal);
speciesAmount operator*(const double leftVal, const speciesAmount& rightVal);


