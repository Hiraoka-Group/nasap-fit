#include <string>
#include <cstring>
#include <vector>
#include <span>
#include <fstream>
#include <map>
#include <algorithm>
#include <charconv>

#include <iostream>

#include <nvector/nvector_serial.h>

#include "../include/constants.hpp"
#include "../include/readcsv.hpp"
#include "../include/Rhsf.hpp"

namespace rhsfBuilder{
    
    std::vector<term> terms; //反応項リスト
    std::map<std::string, int> termIndex; //反応速度定数名からindexへのマップ
    std::array<double, config::constantSize> rateConstants; //反応速度定数配列
    std::array<double, config::species+1> speciesData; //初期種量配列


    auto term::operator<=>(const term& other) const {
        if(reactant2 == other.reactant2){
            if(reactant1 == other.reactant1){
                return add_to <=> other.add_to;
            }
            return reactant1 <=> other.reactant1;
        }
        return reactant2 <=> other.reactant2;
    }

    std::string strip(std::string s){
        size_t start = s.find_first_not_of(" \t\n\r");
        size_t end = s.find_last_not_of(" \t\n\r");
        return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
    }

    bool is_stoiable(std::string s){
        s = strip(s);

        long v;
        auto res = std::from_chars(s.data(), s.data()+s.size(), v);
        return res.ec == std::errc() && res.ptr == s.data()+s.size();
    }
    void buildRhsf(){
        speciesData[config::species]=1.0; //ダミー種の初期量は1.0に設定
        std::vector<std::vector<std::string>> csv_data = read_csv(std::string(config::reactNetworkFile));
        std::map<std::tuple<int, int, int, int>,int> ODE; //key:(add_to, init,entering, kind), value: duplicacy
        terms.resize(0);
        auto addTerm = [&](int add_to, int init, int entering, int kind, int duplicate_count){
            std::tuple<int, int, int, int> key = std::make_tuple(add_to, init, entering, kind);
            if (ODE.contains(key)) ODE[key] += duplicate_count;
            else ODE[key] = duplicate_count; 
        };
        std::vector<std::map<std::string, std::string>> DataDict;
        for (size_t i = 1; i < csv_data.size(); ++i) {
            std::map<std::string, std::string> rowDict;
            for (size_t j = 0; j < std::min(csv_data[i].size(), csv_data[0].size()); ++j) {
                rowDict[strip(csv_data[0][j])] = strip(csv_data[i][j]);
            }
            DataDict.push_back(rowDict);
        }
        for(const auto& mp : DataDict){
            std::string kind=mp.at("kind");
            termIndex[kind] = 0;
        }
        int idx=0;
        for(auto& [key, val] : termIndex){
            val=idx;
            idx++;
        }
        for (const auto& mp : DataDict) {
            int init = std::stoi(mp.at("init_assem_id"));
            int entering = is_stoiable(mp.at("entering_assem_id")) ? std::stoi(mp.at("entering_assem_id")) : config::species;
            int product = std::stoi(mp.at("product_assem_id"));
            int leaving = is_stoiable(mp.at("leaving_assem_id")) ? std::stoi(mp.at("leaving_assem_id")) : config::species;
            int kind = termIndex.at(mp.at("kind"));
            int duplicacy = std::stoi(mp.at("duplicate_count"));
            
            addTerm(init, init, entering, kind, -duplicacy);
            if (entering != config::species) {
                addTerm(entering, init, entering, kind, -duplicacy);
            }
            addTerm(product, init, entering, kind, duplicacy);
            if (leaving != config::species) {
                addTerm(leaving, init, entering, kind, duplicacy);
            }
        }
        for (const auto& [key, duplicacy] : ODE) {
            int add_to = std::get<0>(key);
            int init = std::get<1>(key);
            int entering = std::get<2>(key);
            int kind = std::get<3>(key);
            rhsfBuilder::term t;
            t.add_to = add_to;
            t.duplicacy = duplicacy;
            t.reactant1 = init;
            t.reactant2 = entering;
            t.rateConstant = kind;
            rhsfBuilder::terms.push_back(t);
        }
        std::sort(rhsfBuilder::terms.begin(), rhsfBuilder::terms.end());
        
    }
    int rhsf(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data) {
        sunrealtype* sp_ptr = N_VGetArrayPointer(y);
        auto ydotData = N_VGetArrayPointer(ydot);
        // コピー: y -> speciesData (memcpy)
        std::memcpy(rhsfBuilder::speciesData.data(), sp_ptr, config::species * sizeof(double));
        // コピー: user_data(k) -> rateConstants (memcpy)
        std::memcpy(rhsfBuilder::rateConstants.data(), user_data, config::constantSize * sizeof(double));
        std::fill(ydotData, ydotData + config::species, 0.0);
        std::span<double> ydotspan(ydotData, config::species);
        for (const auto& term : rhsfBuilder::terms) {
            term.calculate(ydotspan);
        }

        return 0;
    }
}


