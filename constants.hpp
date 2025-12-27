#pragma once
#include <string>

//const std::string QASAPFile = "../data/M9L6/Table_S1.csv";

//化学種の数
constexpr int species = 29;
//反応速度定数の数
constexpr int constantSize = 8;
//Table_S1.csvにおいて、化学種の100%にあたる濃度
constexpr double fullConc[] = {1.71e-3, 8.55e-4, 3.42e-3, 4.275e-4};
//データにより与えられる化学種の数
constexpr int trackedSpecies=4;
//データにより与えられる化学種のindex
constexpr int trackedIndex[] = { 27,26,28,25 };


//差分進化法のエージェント数
constexpr int popSize = 128;

//シミュレーションの許容絶対誤差
constexpr double tolAbsError = 1e-9;
//シミュレーションの許容相対誤差
constexpr double tolRelError = 1e-6;
//適応型ルンゲクッタ法の安全係数
constexpr double safetyConstant=0.9;
//差分進化法のパラメータ
constexpr double scalar = 0.5, crossOver = 0.5;
//反応速度定数の上限下限
constexpr double upperLim = 1e4, lowerLim = 1e-3;