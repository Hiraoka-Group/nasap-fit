#include <iostream>
#include <iomanip>
#include <cfloat>
#include <string>
#include <array>
#include <vector>
#include <cassert>
#include <fstream>

# pragma GCC target("avx2")
# pragma GCC optimize("O2")
# pragma GCC optimize("unroll-loops")

#include "../include/constants.hpp"
#include "../include/differentialEvolution.hpp"
#include "../include/readcsv.hpp"


signed main() {
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

	diffEvo.setPop();
	diffEvo.Optimize();
	
	std::cout<<"Optimized Constants:"<<std::endl;
	auto arr = diffEvo.best();
	for (auto t : arr) {
		std::cout << t << " ";
	}
	std::cout << std::endl;
	
	std::cout<<"Debug Info:"<<std::endl;
	diffEvo.DEBUG();
	
	
}

/*
g++ main.cpp -o main && ./main
*/