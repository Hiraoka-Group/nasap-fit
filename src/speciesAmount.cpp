#include <array>
#include <vector>
#include <cassert>
#include <iostream>
#include "../include/speciesAmount.hpp"
#include "../include/ODE.hpp"


speciesAmount::speciesAmount() {
	amount.fill(0);
}
speciesAmount::speciesAmount(std::vector<double> arg) {
	assert(arg.size() == species);
	for (int i = 0; i < species; i++) {
		amount[i] = arg[i];
	}
}
inline double& speciesAmount::operator[](int i) {
	return amount[i];
}
inline const double& speciesAmount::operator[](int i) const {
	return amount[i];
}
speciesAmount speciesAmount::operator+(const speciesAmount& other) {
	speciesAmount ret;
	for (int i = 0; i < species; i++) {
		ret[i] = amount[i] + other.amount[i];
	}
	return ret;
}
speciesAmount speciesAmount::operator/(const double num) {
	speciesAmount ret;
	for (int i = 0; i < species; i++) {
		ret.amount[i] = amount[i] / num;
	}
	return ret;
}
void speciesAmount::Debug() {
	for (int i = 0; i < species; i++) {
		std::cout << amount[i] << " ";
	}
	std::cout << std::endl;
}

speciesAmount operator*(const speciesAmount& leftVal, const double rightVal) {
	speciesAmount ret;
	for (int i = 0; i < species; i++) {
		ret[i] = leftVal[i] * rightVal;
	}
	return ret;
}
speciesAmount operator*(const double leftVal, const speciesAmount& rightVal) {
	speciesAmount ret;
	for (int i = 0; i < species; i++) {
		ret[i] = leftVal * rightVal[i];
	}
	return ret;
}