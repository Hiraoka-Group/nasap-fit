#include <string>
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

namespace Rhsf{

    //一次反応かどうか
    bool term::isFirstOrder() const {
        return reactant2 == -1;
    }; 

    double term::calculate(std::span<const double> sp, std::span<const double> k) const {
        if (isFirstOrder()) {
            return duplicacy * k[rateConstant] * sp[reactant1];
        } else {
            return duplicacy * k[rateConstant] * sp[reactant1] * sp[reactant2];
        }
    }

    std::vector<std::vector<term>> terms; //各生成物ごとの反応項リスト
    std::map<std::string, int> termIndex; //反応速度定数名からindexへのマップ

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
    void makeRhsf(){
        std::vector<std::vector<std::string>> csv_data = read_csv(std::string(config::reactNetworkFile));
        std::vector<std::map<std::tuple<int, int, int>,int>> ODE(config::species); //key:(init,entering, kind), value: duplicacy
        terms.resize(config::species);
        auto addTerm = [](std::map<std::tuple<int, int, int>,int> &mp,int init, int entering, int kind, int duplicate_count){
            std::tuple<int, int, int> key = std::make_tuple(init, entering, kind);
            if (mp.contains(key)) mp[key] += duplicate_count;
            else mp[key] = duplicate_count; 
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
            int entering = is_stoiable(mp.at("entering_assem_id")) ? std::stoi(mp.at("entering_assem_id")) : -1;
            int product = std::stoi(mp.at("product_assem_id"));
            int leaving = is_stoiable(mp.at("leaving_assem_id")) ? std::stoi(mp.at("leaving_assem_id")) : -1;
            int kind = termIndex.at(mp.at("kind"));
            int duplicacy = std::stoi(mp.at("duplicate_count"));
            
            addTerm(ODE[init], init, entering, kind, -duplicacy);
            if (entering != -1) {
                addTerm(ODE[entering], init, entering, kind, -duplicacy);
            }
            addTerm(ODE[product], init, entering, kind, duplicacy);
            if (leaving != -1) {
                addTerm(ODE[leaving], init, entering, kind, duplicacy);
            }
        }
        for (int i = 0; i < config::species; ++i) {
            for (const auto& [key, duplicacy] : ODE[i]) {
                int init = std::get<0>(key);
                int entering = std::get<1>(key);
                int kind = std::get<2>(key);
                Rhsf::term t;
                t.duplicacy = duplicacy;
                t.reactant1 = init;
                t.reactant2 = entering;
                t.rateConstant = kind;
                Rhsf::terms[i].push_back(t);
                //実装メモ
                //kindは文字列なので、適宜数値に変換する処理が必要
            }
        }

        
    }
    int rhsf(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data) {
        sunrealtype* sp_ptr = N_VGetArrayPointer(y);
        auto ydotData = N_VGetArrayPointer(ydot);
        std::array<double, config::constantSize> &k = *static_cast<std::array<double, config::constantSize>*>(user_data);
        std::span<const double> yspan(sp_ptr, config::species);
        std::span<const double> kspan(k.data(), k.size());
        for (int i = 0; i < config::species; ++i) {
            double sum = 0.0;
            for (const auto& term : Rhsf::terms[i]) {
                sum += term.calculate(yspan, kspan);
            }
            ydotData[i] = sum;
        }

        return 0;
    }
}


