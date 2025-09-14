#pragma once

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


//シミュレーションの1ステップごとの時間差
constexpr double timeStep = 1.0 / (1 << 8);
//差分進化法のパラメータ
constexpr double scalar = 0.5, crossOver = 0.5;
//反応速度定数の上限下限
constexpr double upperLim = 1e2, lowerLim = 1e-3;