#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <functional>
#include <random>
#include <cassert>
#include <string>
#include <charconv>
using namespace std;
    
std::string strip(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\n\r");
        size_t end = s.find_last_not_of(" \t\n\r");
        return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
    }
bool is_int_from_chars(std::string s){
    s = strip(s);
    long v;
    auto res = std::from_chars(s.data(), s.data()+s.size(), v);
    return res.ec == std::errc() && res.ptr == s.data()+s.size();
}

int main() {
}