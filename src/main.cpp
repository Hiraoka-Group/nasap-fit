#include <iostream>
#include <iomanip>
#include <cfloat>
#include <string>
#include <array>
#include <vector>
#include <cassert>
#include <fstream>

#include <mpi.h>

# pragma GCC target("avx2")
# pragma GCC optimize("O2")
# pragma GCC optimize("unroll-loops")

#include "../include/constants.hpp"
#include "../include/differentialEvolution.hpp"
#include "../include/readcsv.hpp"

int num_procs;//総プロセス数
int proc_rank;//自分のプロセス番号
//各プロセスのデータ数, 受信バッファ内の開始位置
std::vector<int>recvcounts,displs, recvcounts2,displs2;

void print(std::vector<int>arg){
	for(int t : arg){
		std::cout<<t<<" ";
	}
	std::cout<<std::endl;
}

signed main(int argc, char* argv[]) {
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	recvcounts.resize(num_procs);
	recvcounts2.resize(num_procs);
	displs.resize(num_procs,0);
	displs2.resize(num_procs,0);
	for(int i=0; i<num_procs; i++){
		recvcounts[i]=constantSize*(popSize/num_procs+(popSize%num_procs>i));
		recvcounts2[i]=(popSize/num_procs)+(popSize%num_procs>i);
		if(i){
			displs[i]=displs[i-1]+recvcounts[i-1];
			displs2[i]=displs2[i-1]+recvcounts2[i-1];
		}
	}
	

	std::string inputfile = "../data/Table_S1.csv";
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
	diffEvo.putSim({0.0410171, 97.8908, 10.9259, 41.1699, 0.00143831, 3.93534, 0.0058751, 0.141529});
	/*
	diffEvo.setPop();
	diffEvo.Optimize();
	
	if(proc_rank==0){
		std::cout<<"Optimized Constants:"<<std::endl;
		auto arr = diffEvo.best();
		for (auto t : arr) {
			std::cout << t << " ";
		}
		std::cout << std::endl;
		
		std::cout<<"Debug Info:"<<std::endl;
		diffEvo.DEBUG();
	}
	*/
	MPI_Finalize();
}

/*
g++ main.cpp -o main && ./main
*/