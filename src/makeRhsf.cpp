#include <string>
#include <vector>
#include <span>
#include <fstream>
#include <map>

#include "../include/constants.hpp"
#include "../include/readcsv.hpp"

//extern std::string config::reactNetworkFile;


struct term {
    int addTo; // 生成物のインデックス
    int duplicacy;
    int reactant1; // 反応物1のインデックス
    int reactant2; // 反応物2のインデックス、typeが0の場合は-1
    int rateConstant;// 反応速度定数のインデックス
    //一次反応かどうか
    bool isFirstOrder() const {
        return reactant2 == -1;
    }; 

    double calculateRate(std::span<const double> sp, std::span<const double> k) const {
        if (isFirstOrder()) {
            return duplicacy * k[rateConstant] * sp[reactant1];
        } else {
            return duplicacy * k[rateConstant] * sp[reactant1] * sp[reactant2];
        }
    }
};

std::vector<term> parseReactionTerms(const std::vector<std::vector<std::string>>& data) {
    std::vector<term> terms;
    std::vector<std::map<std::string, int>> dataMap(data.size()-1);
    for(int i = 1; i < data.size(); i++){
        for(int j = 0; j < std::min(data[i].size(), data[0].size()); j++){
            dataMap[i-1][data[0][j]] = std::stoi(data[i][j]);
        }
    }
    for (const auto& row : dataMap) {
        int init = row.at("init_assem_id");
        int entering = row.at("entering_assem_id");
        int product = row.at("product_assem_id");
        int leaving = row.at("leaving_assem_id");
        int duplicacy = row.at("duplicate_count");
        int kind = row.at("kind");
    }
    return terms;
}

void makeRhsf(const std::string& filename){
    std::vector<std::vector<std::string>> data = read_csv(filename);

    
}