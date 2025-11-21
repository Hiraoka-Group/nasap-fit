#pragma once
#include <array>
#include <vector>
#include "constants.hpp"

template<class T>
class speciesAmount {
	std::array<T, species>amount;
public:;
	speciesAmount();
	speciesAmount(std::vector<T> arg);
	T& operator[](int i);
	const T& operator[](int i) const;
	speciesAmount<T> operator+(const speciesAmount<T>& other);
	speciesAmount<T> operator/(const double num);
	void Debug();
};
template<class T>speciesAmount<T> operator*(const speciesAmount<T>& leftVal, const double rightVal);
template<class T>speciesAmount<T> operator*(const double leftVal, const speciesAmount<T>& rightVal);


