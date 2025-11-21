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
#include "../include/ODE.hpp"

int num_procs;//総プロセス数
int proc_rank;//自分のプロセス番号

const extern std::string inputfile;

std::chrono::system_clock::time_point startTime,endTimeGlobal;

signed main(int argc, char** argv) {
	startTime = std::chrono::system_clock::now();
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	
	std::vector<std::vector<std::string>> csv_data = read_csv(inputfile);
	std::vector<std::vector<double>> csv_data_double;
	for (const auto& row : csv_data) {
		std::vector<double> row_double;
		for (int i=0; i<trackedSpecies+1; i++) {
			if(row[0]=="Time (min)") goto skip; //ヘッダー行をスキップ
			row_double.push_back(std::stod(row[i]));
		}
		csv_data_double.push_back(row_double);
		skip:;
	}
	differentialEvolution diffEvo(csv_data_double); // Assuming setData is a method to set the data
	//diffEvo.putSim({0.410171, 978.908, 109.259, 411.699, 0.0143831, 39.3534, 0.058751, 1.41529});
	//diffEvo.putCVODESim({0.410171, 978.908, 109.259, 411.699, 0.0143831, 39.3534, 0.058751, 1.41529});
	

	
	#if 0
	diffEvo.putSim({0.0913144, 93.7798, 0.001, 153.178, 331.427, 237.888, 0.001, 2.43555});
	diffEvo.putCVODESim({0.0913144, 93.7798, 0.001, 153.178, 331.427, 237.888, 0.001, 2.43555});
	
	#else

	diffEvo.Optimize();
	endTimeGlobal = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimeGlobal - startTime).count();
	if(proc_rank==0)std::cout << "Optimization took " << duration << " milliseconds." << std::endl;
	auto bestConstants = diffEvo.best();
	if(proc_rank==0){
		diffEvo.putSim(bestConstants);
		diffEvo.putCVODESim(bestConstants);
	}
	
	if(false){
		std::cout<<"Optimized Constants:"<<std::endl;
		auto arr = diffEvo.best();
		for (auto t : arr) {
			std::cout << t << " ";
		}
		std::cout << std::endl;
		std::cout<<"error: "<<diffEvo.calcError(arr)<<std::endl;
		if(proc_rank==0){
			std::vector<double> jac = diffEvo.getJacobian(arr);
			std::cout<<"Jacobian :"<<std::endl;
			for(double val : jac){
				std::cout<<std::setw(15)<<std::setprecision(7)<<val<<" ";
			}
			std::cout<<std::endl;
		}
		
		std::vector<std::vector<double>> hessian = diffEvo.getHessian(arr);
		if(proc_rank==0)std::cout<<"Hessian Matrix :"<<std::endl;
		if(proc_rank==0){
			for(const std::vector<double>& row : hessian){
				for(double val : row){
					std::cout<<std::setw(15)<<std::setprecision(7)<<val<<" ";
				}
				std::cout<<std::endl;
			}
		}
	}
	#endif

	MPI_Finalize();
	
}

/*
g++ main.cpp -o main && ./main
cmake --build . && ./nasap_fit_cpp
*/