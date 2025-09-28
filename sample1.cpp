#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <functional>
#include <random>
using namespace std;
    


# pragma GCC optimize("O2")
double diffCoeff(double x){
    return x;
}
double timeStep=0.1;
double tolerableAbsoluteError=1e-6;
double safetyConstant=0.9;
// 処理時間を計測し、結果を表示する関数
double calcNextStep(const double& data) {
    double val1, val2, val3, val4,val5,val6,val7, slope1, slope2, slope3, slope4,slope5,slope6,slope7, order4;
    double maxerror;
    while(1){
        maxerror=0.0;
        val1 = data;
        slope1 = diffCoeff(val1);
        val2 = val1 + timeStep * slope1 * (1.0/5);
        slope2 = diffCoeff(val2);
        val3 = val1 + timeStep * (slope1 * (3.0/40) + slope2 * (9.0/40));
        slope3 = diffCoeff(val3);
        val4 = val1 + timeStep * (slope1 * (44.0/45) + slope2 * (-56.0/15) + slope3 * (32.0/9));
        slope4 = diffCoeff(val4);
        val5 = val1 + timeStep * (slope1 * (19372.0/6561) + slope2 * (-25360.0/2187) + slope3 * (64448.0/6561) + slope4 * (-212.0/729));
        slope5 = diffCoeff(val5);
        val6 = val1 + timeStep * (slope1 * (9017.0/3168) + slope2 * (-355.0/33) + slope3 * (46732.0/5247) + slope4 * (49.0/176) + slope5 * (-5103.0/18656));
        slope6 = diffCoeff(val6);
        val7 = val1 + timeStep * (slope1 * (35.0/384) + slope3 * (500.0/1113) + slope4 * (125.0/192) + slope5 * (-2187.0/6784) + slope6 * (11.0/84));
        slope7 = diffCoeff(val7);
        const double& order5 = val7;
        order4= val1 + timeStep * (slope1 * (5179.0/57600) + slope3 * (7571.0/16695) + slope4 * (393.0/640) + slope5 * (-92097.0/339200) + slope6 * (187.0/2100) + slope7 * (1.0/40));
        
            double e = std::abs(order4-order5);
            if(maxerror < e) maxerror=e;
        //timeStepの更新をスキップする場合
        //std::cout<<"maxerror: "<<maxerror<<"\n";
        return order5;

        if(maxerror<=tolerableAbsoluteError){//ステップ成功
            double h_new;
            if(maxerror==0)h_new = timeStep * 2;
            else{
                h_new=safetyConstant*timeStep*std::pow(tolerableAbsoluteError/maxerror,(1.0/5));
                h_new=std::clamp(h_new, timeStep*0.25, timeStep*1.5);
            }
            timeStep=h_new;
            return order5;
        }else{//ステップ失敗
            //timeStep=safetyConstant*timeStep*std::pow(tolerableAbsoluteError/maxerror,(1.0/5));
        }
    }
}

void simulate(){
    vector<double>simulation = {1}; 
    vector<double>simTime = {0.0};

    //シミュレーション
    for (int k = 0; simTime.back() <= 10; k++) {
        

        std::cout<<"stepsize: "<<timeStep<<"\n";
        //std::cout<<"TIME: "<<simTime.back()<<"  ";
        double  res = calcNextStep(simulation[k]);
        simulation.push_back(res);
        simTime.push_back(simTime.back()+timeStep);
        std::cout<<"TIME: "<<simTime.back()<<" "<<simulation.back()<<"\n";
        //timeStep = res.newStepSize;
    }
}

int main(){
    simulate();
}