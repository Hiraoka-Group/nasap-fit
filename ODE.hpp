constexpr int species = 4;
constexpr int constantSize = 3;

class speciesAmount {
	std::array<double, species>amount;
public:;
	speciesAmount() {
		amount.fill(0);
	}
	speciesAmount(std::vector<double> arg) {
		assert(arg.size() == species);
		for (int i = 0; i < species; i++) {
			amount[i] = arg[i];
		}
	}
	inline double& operator[](int i) {
		return amount[i];
	}
	inline const double& operator[](int i) const {
		return amount[i];
	}
	speciesAmount operator+(const speciesAmount& other) {
		speciesAmount ret;
		for (int i = 0; i < species; i++) {
			ret[i] = amount[i] + other.amount[i];
		}
		return ret;
	}
	speciesAmount operator/(const double num) {
		speciesAmount ret;
		for (int i = 0; i < species; i++) {
			ret.amount[i] = amount[i] / num;
		}
		return ret;
	}
};
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


speciesAmount diffCoeff(const std::array<double, constantSize>& reactConst, const speciesAmount& status) {
	speciesAmount dxdt;
	dxdt[0] = (reactConst[1] + reactConst[2]) * status[2] - reactConst[0] * status[0] * status[1];
	dxdt[1] = reactConst[1] * status[2] - reactConst[0] * status[0] * status[1];
	dxdt[2] = reactConst[0] * status[0] * status[1] - (reactConst[1] + reactConst[2]) * status[2];
	dxdt[3] = reactConst[2] * status[2];
	return dxdt;
}