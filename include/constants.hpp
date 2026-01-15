#pragma once
#include <string>
#include <string_view>

#define USE_PREGENERATED_RHSF (1)
#define USE_PREGENERATED_JACOBIAN (1)

namespace config {
const std::string QASAPFile = "../data/Table_S1.csv";
const std::string reactNetworkFile = "../data/classified_reactions_str.csv";

//化学種の数（ランタイムで設定可能）
inline int species = 29;
//反応速度定数の数（ランタイムで設定可能）
inline int constantSize = 8;

inline void setSpecies(int s){ species = s; }
inline void setConstantSize(int c){ constantSize = c; }

//データにより与えられる化学種の数
const int trackedSpecies=4;
//csvファイル内の化学種の名前
const std::string_view trackedNames[] = { "1 (%)", "[PdPy*4]2+ (%)", "Py* (%)", "Pd214 cage (%)" };
//データにより与えられる化学種のindex
const int trackedIndex[] = { 27, 26, 28, 25 };
//Table_S1.csvにおいて、化学種の100%にあたる濃度
const double fullConc[] = {0.0017099999999999999, 0.00085499999999999997, 0.0034199999999999999, 0.00042749999999999998};


//差分進化法のエージェント数
const int popSize = 128;
//差分進化法の最大世代数
const int maxGen = 200;

//シミュレーションの許容絶対誤差
const double tolAbsError = 1.0000000000000001e-09;
//シミュレーションの許容相対誤差
const double tolRelError = 9.9999999999999995e-07;
//適応型ルンゲクッタ法の安全係数
const double safetyConstant=0.9;
//差分進化法のパラメータ
const double scalar = 0.5, crossOver = 0.5;
//反応速度定数の上限下限
const double upperLim = 10000, lowerLim = 0.001;
}
