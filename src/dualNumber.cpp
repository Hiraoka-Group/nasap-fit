#include <cmath>

class dualNumber {
double real;
double dual;
public:
    dualNumber(double r=0.0, double d=1.0){
        real=r; dual=d;
    }
    operator double() const { return real; }
    double getDual() const { return dual; }
    inline dualNumber operator+(const dualNumber& b) const{
        return dualNumber(real+b.real, dual+b.dual);
    }
    inline dualNumber operator-(const dualNumber& b) const{
        return dualNumber(real-b.real, dual-b.dual);
    }
    inline dualNumber operator*(const dualNumber& b) const{
        return dualNumber(real*b.real, real*b.dual + dual*b.real);
    }
    inline dualNumber operator/(const dualNumber& b) const{
        return dualNumber(real/b.real, (dual*b.real - real*b.dual)/(b.real*b.real));
    }
    inline dualNumber operator*(const double& b) const{
        return dualNumber(real*b, dual*b);
    }
    inline dualNumber operator/(const double& b) const{
        return dualNumber(real/b, dual/b);
    }
    friend inline dualNumber operator*(const double& a, const dualNumber& b);
    friend inline dualNumber operator/(const double& a, const dualNumber& b);
    friend inline dualNumber pow(dualNumber base, dualNumber exp);
    friend inline dualNumber pow(double base, dualNumber exp);
    friend inline dualNumber pow(dualNumber base, double exp);
    friend inline dualNumber exp(dualNumber x);
    friend inline dualNumber log(dualNumber x);
    friend inline dualNumber sqrt(dualNumber x);
};
dualNumber operator*(const double& a, const dualNumber& b){
    return dualNumber(a*b.real, a*b.dual);
}
dualNumber operator/(const double& a, const dualNumber& b){
    return dualNumber(a/b.real, -a*b.dual/(b.real*b.real));
}
dualNumber pow(dualNumber base, dualNumber exp){ //x^y
    dualNumber ret;
    double temp = std::pow(base.real, exp.real - 1);
    ret.real = temp*base.real; //x^y
    ret.dual = exp.dual * (ret.real * std::log(base.real)) // dy * x^y*log(x)
        + base.dual * (exp.real * temp); // dx * y*x^(y-1)
    return ret;
}
dualNumber pow(double base, dualNumber exp){
    dualNumber ret;
    ret.real = std::pow(base, exp.real); //x^y
    ret.dual = exp.dual * (ret.real * std::log(base)); // dy * x^y*log(x)
    return ret;
}
dualNumber pow(dualNumber base, double exp){
    dualNumber ret;
    double temp = std::pow(base.real, exp - 1);
    ret.real = temp*base.real; //x^y
    ret.dual = base.dual * (exp * temp); // dx * y*x^(y-1)
    return ret;
}
dualNumber exp(dualNumber x){
    dualNumber ret;
    ret.real = std::exp(x.real);
    ret.dual = x.dual * ret.real;
    return ret;
}
dualNumber log(dualNumber x){
    dualNumber ret;
    ret.real = std::log(x.real);
    ret.dual = x.dual / x.real;
    return ret;
}
dualNumber sqrt(dualNumber x){
    dualNumber ret;
    ret.real = std::sqrt(x.real);
    ret.dual = x.dual / (2.0 * ret.real);
    return ret;
}