#pragma once 
#include <string>
#include <vector>
#include <string_view>
#include <fstream>

std::vector<std::string>parse(std::string_view line, char delimiter = ',');

std::vector<std::vector<std::string>> read_csv(const std::string& filename);