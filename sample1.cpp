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
    // Your code logic here
    cout<<"       "<<":"<<is_int_from_chars("       ")<<endl; // Testing stoi with leading/trailing spaces
    cout<<"-12345:"<<is_int_from_chars("-12345")<<endl;
    cout<<"  +6789:"<<is_int_from_chars("+6789")<<endl;
    cout<<"42abc:"<<is_int_from_chars("42abc")<<endl;
    cout<<"abc42:"<<is_int_from_chars("abc42")<<endl;
    cout<<"  123:"<<is_int_from_chars("  123")<<endl;
    cout<<"123  :"<<is_int_from_chars("123  ")<<endl;
    cout<<"0123:"<<is_int_from_chars("0123")<<endl;
    cout<<"0.0:"<<is_int_from_chars("0.0")<<endl;
    cout<<".123:"<<is_int_from_chars(".123")<<endl;
    cout<<"123.:"<<is_int_from_chars("123.")<<endl;
    cout<<"6.022e23:"<<is_int_from_chars("6.022e23")<<endl;
    return 0;
}