#include <iostream>
#include <iomanip>
#include <cfloat>
#include <string>
#include <array>
#include <vector>
#include <cassert>
#include <fstream>
#include <chrono>

#include <mpi.h>

#include <cppad/cppad.hpp>
/*
# pragma GCC target("avx2")
# pragma GCC optimize("O2")
# pragma GCC optimize("unroll-loops")
*/

#include "../include/constants.hpp"
#include "../include/NASAP_fit.hpp"
#include "../include/readcsv.hpp"
#include "../include/MPIEnvironment.hpp"

int stepCount[61];


std::chrono::system_clock::time_point startTime,endTimeGlobal;

//rshift myRand(1);

signed main(int argc, char** argv) {
	MpiEnvironment mpi_env(argc, argv);

	NASAP_fit::Config cfg;
	cfg.QASAPFile = config::QASAPFile;
    cfg.reactNetworkFile = config::reactNetworkFile;
    cfg.species = config::species;
    cfg.constantSize = config::constantSize;
    cfg.trackedSpecies = config::trackedSpecies;
    cfg.trackedNames=vector<std::string>(std::begin(config::trackedNames), std::end(config::trackedNames));
    cfg.trackedIndex=vector<int>(std::begin(config::trackedIndex), std::end(config::trackedIndex));
    cfg.fullConc=vector<double>(std::begin(config::fullConc), std::end(config::fullConc));
    cfg.initConc = config::initConc;
    cfg.tolAbsError = config::tolAbsError;
    cfg.tolRelError = config::tolRelError;
    cfg.scalar = config::scalar;
    cfg.crossOver = config::crossOver;
    cfg.upperLim = config::upperLim;
    cfg.lowerLim = config::lowerLim;

	std::vector<std::vector<std::string>> QASAPdata = read_csv(std::string(config::QASAPFile));

	if(mpi_env.rank()==0)std::cout<<"Loaded "<<QASAPdata.size()<<" rows of data."<<std::endl;

	
	NASAP_fit diffEvo(cfg);

	startTime = std::chrono::system_clock::now();
	NASAP_fit::TerminationCondition deTerm;
	deTerm.maxIter = 50;
	auto opt = diffEvo.runDE(128, cfg.lowerLim, cfg.upperLim, deTerm);
	endTimeGlobal = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimeGlobal - startTime).count();

	if(mpi_env.rank()==0)std::cout << "Optimization took " << duration << " milliseconds." << std::endl;
	
	std::vector<double> bestConstants(config::constantSize);
	double minerror;
	// CasADi 依存の LM refinement は除外
	bestConstants = opt[0].constants;
	minerror = opt[0].error;
	std::vector<std::string>kinds(config::constantSize);
	for(const auto& [key, val] : diffEvo.termIndex()){
		kinds[val]=key;
	}
	if(mpi_env.rank()==0){
		std::cout<<"Optimized Constants:"<<std::endl;
		for(int i=0;i<config::constantSize;i++){
			std::cout<<kinds[i]<<": "<<bestConstants[i]<<std::endl;
		}
		std::cout<<std::endl;
		std::cout<<"error: "<<minerror<<std::endl;
	}
	diffEvo.putCVODESim(bestConstants);

	return 0;

}//0.0349428 7370.47 966.895 138.205 0.359602 972.249 0.00584204 0.144209

/*
g++ main.cpp -o main && ./main
cmake --build . && ./nasap_fit_cpp
*/