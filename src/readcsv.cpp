#include <string>
#include <vector>
#include <string_view>
#include <fstream>

#include "../include/readcsv.hpp"

std::vector<std::string>parse(std::string_view line, char delimiter){
    std::vector<std::string> result;
    std::string current;
    bool in_quotes = false;

    for (char ch : line) {
        if (ch == delimiter) {
            result.push_back(current);
            current.clear();
        } else {
            current += ch;
        }
    }
    result.push_back(current); // Add the last field
    return result;
}
std::vector<std::vector<std::string>> read_csv(const std::string& filename) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filename);
    if(file.fail()){
        throw std::runtime_error("Could not open file: " + filename);
    }
    std::string line;  
    while (std::getline(file, line)) {
        data.push_back(parse(line));
    }
    return data;
}