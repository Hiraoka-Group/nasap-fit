#include <array>

#if 0

#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_core.hpp>
#include <sunlinsol/sunlinsol_pcg.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sundials/sundials_context.hpp>

#include "../include/constants.hpp"
#include "../include/speciesAmount.hpp"
#include "../include/Rhsf.hpp"


void CVODEsimulation(speciesAmount<double>& initialState, std::array<double, constantSize>& rateConstants){
    // CVODEの初期化
    sundials::Context sunctx;
    N_Vector y0 = N_VNew_Serial(species, sunctx);
    sunindextype Nlocal = species;
    for(int i=0;i<species;i++){
        NV_Ith_S(y0,i)=initialState[i];
    }
    void* cvode_mem = CVodeCreate(CV_BDF, sunctx);

    int flag = CVodeInit(cvode_mem, rhsf, 0.0, y0);
    assert(flag == CV_SUCCESS);

    flag = CVodeSStolerances(cvode_mem, tolerableError, 1e-3);
    assert(flag == CV_SUCCESS);


    SUNLinearSolver LS = SUNLinSol_SPGMR(y0, SUN_PREC_NONE, 0, sunctx);
    SUNMatrix J = NULL;
    flag = CVodeSetLinearSolver(cvode_mem, LS, J);
    assert(flag == CV_SUCCESS);

    // シミュレーションの実行
    double t = 0.0;
    double tEnd = 10.0; // シミュレーション終了時間
    while (t < tEnd) {
        double tNext = t + 0.1; // 次の時間ステップ
        flag = CVode(cvode_mem, tNext, y0, &t, CV_NORMAL);
        assert(flag >= 0);
    }


}
#endif