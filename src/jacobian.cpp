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
#include "../include/jacobian.hpp"

namespace jacobian{
    int nonZeros=0;

    std::vector<std::vector<term>> terms; //反応項リスト
    std::map<std::string, int> termIndex; //反応速度定数名からindexへのマップ
    std::array<double, config::constantSize> rateConstants; //反応速度定数配列
    std::array<double, config::species+1> speciesData; //初期種量配列

    std::vector<int>idxPointer;
    std::vector<int>idxValue;


    auto term::operator<=>(const term& other) const {
        if(reactant == other.reactant){
            return rateConstant <=> other.rateConstant;
        }
        return reactant <=> other.reactant;
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
    void makeJacobian(){
        terms.clear();
        idxPointer.clear();
        idxValue.clear();
        speciesData[config::species]=1.0; //ダミー種の初期量は1.0に設定
        
        std::vector<std::vector<std::string>> csv_data = read_csv(std::string(config::reactNetworkFile));
        std::map<std::pair<int, int>,std::map<std::pair<int, int>, int>> JacMapping; //key:(row, column), (chem_idx, kind), value: duplicacy
        auto addTerm = [&](int row, int column, int chem_idx, int kind, int duplicate_count){
            std::pair<int, int> key1 = std::make_pair(row, column);
            std::pair<int, int> key2 = std::make_pair(chem_idx, kind);
            if (JacMapping.contains(key1) && JacMapping[key1].contains(key2)) JacMapping[key1][key2] += duplicate_count;
            else JacMapping[key1][key2] = duplicate_count; 
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
            
            // A + B -> C + D のとき
            // dA/dt = - k * A * B
            // dA'/dA = - k * B
            // dA'/dB = - k * A
            addTerm(init, init, entering, kind, -duplicacy);
            if(entering != config::species){
                addTerm(init, entering, init, kind, -duplicacy);
            }
            if(entering != config::species){
                // dB/dt = - k * A * B
                // dB'/dA = - k * B
                // dB'/dB = - k * A
                addTerm(entering, init, entering, kind, -duplicacy);
                addTerm(entering, entering, init, kind, -duplicacy);
            }
            // dC/dt = k * A * B
            // dC'/dA = k * B
            // dC'/dB = k * A
            addTerm(product, init, entering, kind, duplicacy);
            if(entering != config::species){
                addTerm(product, entering, init, kind, duplicacy);
            }
            if(leaving != config::species){
                // dD/dt = k * A * B
                // dD'/dA = k * B
                // dD'/dB = k * A
                addTerm(leaving, init, entering, kind, duplicacy);
                if(entering != config::species){
                    addTerm(leaving, entering, init, kind, duplicacy);
                }
            }
        }
        {
            int cnt = 0;
            nonZeros = 0;
            idxPointer.clear();
            idxValue.clear();
            terms.clear();
            idxPointer.resize(config::species + 1);
            for (int col = 0; col < config::species; ++col) {
                idxPointer[col] = cnt;
                // JacMapping のキーは (row, col) なので、全キーを走査してこの列に属するものを拾う
                for (const auto& [key1, mp] : JacMapping) {
                    int key_row = std::get<0>(key1);
                    int key_col = std::get<1>(key1);
                    if (key_col != col) continue;
                    // この非ゼロ要素は (row=key_row, col)
                    idxValue.push_back(key_row); // Ji に渡すのは行インデックス
                    std::vector<term> termList;
                    for (const auto& [key2, duplicacy] : mp) {
                        int chem_idx = std::get<0>(key2);
                        int kind = std::get<1>(key2);
                        term t;
                        t.duplicacy = duplicacy;
                        t.reactant = chem_idx;
                        t.rateConstant = kind;
                        assert(0 <= chem_idx && chem_idx <= config::species);
                        termList.push_back(t);
                    }
                    terms.push_back(termList);
                    ++cnt;
                    ++nonZeros;
                }
            }
            idxPointer[config::species] = cnt;
        }

    }
    int JacFn(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3){
        sunrealtype* sp_ptr = N_VGetArrayPointer(y);
	    sunindextype* Jp = SUNSparseMatrix_IndexPointers(Jac);
	    sunindextype* Ji = SUNSparseMatrix_IndexValues(Jac);
	    sunrealtype* Jx = SUNSparseMatrix_Data(Jac);
        for(int i=0;i<=config::species;i++) Jp[i]=idxPointer[i]; 
        for(int i=0;i<nonZeros;i++) Ji[i]=idxValue[i];

        // コピー: y -> speciesData (memcpy)
        std::memcpy(jacobian::speciesData.data(), sp_ptr, config::species * sizeof(double));
        // コピー: user_data(k) -> rateConstants (memcpy)
        std::memcpy(jacobian::rateConstants.data(), user_data, config::constantSize * sizeof(double));
        int idx=0;
        for(const auto& termList : terms){
            double val=0.0;
            for(const auto& t : termList){
                val += t.calculate();
            }
            Jx[idx]=val;
            idx++;
        }
        return 0;
    }
}