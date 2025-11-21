#include <array>
#include <vector>
#include <cassert>
#include <iostream>
#include "../include/speciesAmount.hpp"
#include "../include/ODE.hpp"

#include <cppad/cppad.hpp>


template<class T>
speciesAmount<T>::speciesAmount() {
	amount.fill(0);
}
template<class T>
speciesAmount<T>::speciesAmount(std::vector<T> arg) {
	assert(arg.size() == species);
	for (int i = 0; i < species; i++) {
		amount[i] = arg[i];
	}
}
template<class T>
inline T& speciesAmount<T>::operator[](int i) {
	return amount[i];
}
template<class T>
inline const T& speciesAmount<T>::operator[](int i) const {
	return amount[i];
}
template<class T>
speciesAmount<T> speciesAmount<T>::operator+(const speciesAmount<T>& other) {
	speciesAmount<T> ret;
	for (int i = 0; i < species; i++) {
		ret[i] = amount[i] + other.amount[i];
	}
	return ret;
}
template<class T>
speciesAmount<T> speciesAmount<T>::operator/(const double num) {
	speciesAmount<T> ret;
	for (int i = 0; i < species; i++) {
		ret.amount[i] = amount[i] / num;
	}
	return ret;
}

template<class T>
speciesAmount<T> operator*(const speciesAmount<T>& leftVal, const double rightVal) {
	speciesAmount<T> ret;
	for (int i = 0; i < species; i++) {
		ret[i] = leftVal[i] * rightVal;
	}
	return ret;
}

template<class T>
speciesAmount<T> operator*(const double leftVal, const speciesAmount<T>& rightVal) {
	speciesAmount<T> ret;
	for (int i = 0; i < species; i++) {
		ret[i] = leftVal * rightVal[i];
	}
	return ret;
}


typedef CppAD::AD<double> ADdouble;
//明示的なインスタンス化
template class speciesAmount<double>;
template speciesAmount<double> operator*(const speciesAmount<double>& leftVal, const double rightVal);
template speciesAmount<double> operator*(const double leftVal, const speciesAmount<double>& rightVal);
template class speciesAmount<ADdouble>;
template speciesAmount<ADdouble> operator*(const speciesAmount<ADdouble>& leftVal, const double rightVal);
template speciesAmount<ADdouble> operator*(const double leftVal, const speciesAmount<ADdouble>& rightVal);
