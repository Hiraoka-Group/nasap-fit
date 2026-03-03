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
#include "../include/differentialEvolution.hpp"
#include "../include/readcsv.hpp"

int num_procs=1;//総プロセス数
int proc_rank=0;//自分のプロセス番号

int stepCount[61];


std::chrono::system_clock::time_point startTime,endTimeGlobal;

//rshift myRand(1);

signed main(int argc, char** argv) {
	//MPI_Init(&argc, &argv);
	//MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    //MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

	differentialEvolution::Config cfg;
	cfg.QASAPFile = config::QASAPFile;
    cfg.reactNetworkFile = config::reactNetworkFile;
    cfg.species = config::species;
    cfg.constantSize = config::constantSize;
    cfg.trackedSpecies = config::trackedSpecies;
    cfg.trackedNames=vector<std::string_view>(std::begin(config::trackedNames), std::end(config::trackedNames));
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

	if(proc_rank==0)std::cout<<"Loaded "<<QASAPdata.size()<<" rows of data."<<std::endl;

	
	differentialEvolution diffEvo(cfg); // Assuming setQASAPData is a method to set the data
	diffEvo.setQASAPData(QASAPdata);
	diffEvo.setUpCasADiFunctions();
	diffEvo.runLM({3.002, 1e+04, 60.37, 2978, 4.915, 0.7673, 0.5706, 0.2084});

	startTime = std::chrono::system_clock::now();
	auto opt = diffEvo.Optimize(50, 128);
	endTimeGlobal = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimeGlobal - startTime).count();

	if(proc_rank==0)std::cout << "Optimization took " << duration << " milliseconds." << std::endl;
	
	std::vector<double> bestConstants(config::constantSize);
	double minerror;
	// LM refinement on the best solution returned by DE
	auto refined = diffEvo.runLM(opt[4].constants);
	bestConstants = refined.constants;
	minerror = refined.error;
	std::vector<std::string>kinds(config::constantSize);
	for(const auto& [key, val] : diffEvo.reactionNetwork().termIndex){
		kinds[val]=key;
	}
	if(proc_rank==0){
		std::cout<<"Optimized Constants:"<<std::endl;
		for(int i=0;i<config::constantSize;i++){
			std::cout<<kinds[i]<<": "<<bestConstants[i]<<std::endl;
		}
		std::cout<<std::endl;
		std::cout<<"error: "<<minerror<<std::endl;
	}

	MPI_Finalize();
	return 0;

	std::vector<std::vector<double>> hessianMat, hessianMatParallel;
	if(proc_rank==0)std::cout<<"Calculating Hessian Matrix at optimum..."<<std::endl;
	hessianMat =diffEvo.getHessian(bestConstants);
	hessianMatParallel = diffEvo.getHessian_parallel(bestConstants);
	if(proc_rank==0){
		for (int i = 0; i < config::constantSize; i++) {
			for (int j = 0; j < config::constantSize; j++) {
				std::cout << std::setw(15) << hessianMat[i][j] << " ";
			}
			std::cout << std::endl;	
		}
		std::cout<<"Parallel Hessian Matrix:\n";
		for (int i = 0; i < config::constantSize; i++) {
			for (int j = 0; j < config::constantSize; j++) {
				std::cout << std::setw(15) << hessianMatParallel[i][j] << " ";
			}
			std::cout << std::endl;
		}
	}

	//diffEvo.putCVODESim(bestConstants);
	
	

	MPI_Finalize();
	
}//0.0349428 7370.47 966.895 138.205 0.359602 972.249 0.00584204 0.144209

/*
g++ main.cpp -o main && ./main
cmake --build . && ./nasap_fit_cpp
*/