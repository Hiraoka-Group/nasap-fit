#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <functional>
#include <random>
#include <cassert>
using namespace std;
    


int main() {
    // Your code logic here
    int a=0;
    int b=0;
    try{
        cout << a/b << endl;
    }catch(const std::exception& e){
        std::cerr<<"Error: "<<e.what()<<std::endl;
    }
    return 0;
}