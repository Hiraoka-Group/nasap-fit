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
#include "../include/rhsfBuilder.hpp"
#include "../include/jacBuilder.hpp"

int num_procs=1;//総プロセス数
int proc_rank;//自分のプロセス番号

int stepCount[61];


std::chrono::system_clock::time_point startTime,endTimeGlobal;

//rshift myRand(1);

signed main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	std::vector<std::vector<std::string>> QASAPdata = read_csv(std::string(config::QASAPFile));
	std::vector<std::vector<double>> csv_data_double;
	bool isHeader = true;
	for (const auto& row : QASAPdata) {
		std::vector<double> row_double;
		for (int i=0; i<config::trackedSpecies+1; i++) {
			if(isHeader){
				isHeader=false;
				goto dontPush; //ヘッダー行をスキップ
			} 
			row_double.push_back(std::stod(row[i]));
		}
		csv_data_double.push_back(row_double);
		dontPush:;
	}
	if(proc_rank==0)std::cout<<"Loaded "<<csv_data_double.size()<<" rows of data."<<std::endl;

	
	rhsfBuilder::buildRhsf();
	jacBuilder::buildJacobian();

	differentialEvolution diffEvo(csv_data_double); // Assuming setData is a method to set the data

	startTime = std::chrono::system_clock::now();
	diffEvo.Optimize();
	endTimeGlobal = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimeGlobal - startTime).count();

	if(proc_rank==0)std::cout << "Optimization took " << duration << " milliseconds." << std::endl;
	

	
	
	std::vector<double> bestConstants(config::constantSize);
	double minerror;
	diffEvo.best(bestConstants, minerror);
	
	if(proc_rank==0){
		std::cout<<"Optimized Constants:"<<std::endl;
		std::vector<std::string>kinds(config::constantSize);
		for(const auto& [key, val] : rhsfBuilder::termIndex){
			kinds[val]=key;
		}
		for(int i=0;i<config::constantSize;i++){
			std::cout<<kinds[i]<<": "<<bestConstants[i]<<std::endl;
		}
		std::cout<<std::endl;
		std::cout<<"error: "<<minerror<<std::endl;
		diffEvo.runLM(0); //最良個体に対してLM法を実行
	}
	//diffEvo.putCVODESim(bestConstants);
	
	

	MPI_Finalize();
	
}//0.0349428 7370.47 966.895 138.205 0.359602 972.249 0.00584204 0.144209

/*
g++ main.cpp -o main && ./main
cmake --build . && ./nasap_fit_cpp
*/