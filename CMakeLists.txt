# CMakeLists.txt

# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(nasap_fit_cpp)

# Require C++20 for std::span and other C++20 features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ソースコードを収集
file(GLOB_RECURSE SOURCES "src/*.cpp")

# MPIを探す
find_package(MPI REQUIRED)

# Eigen3を探す
find_package(Eigen3 REQUIRED)

# SUNDIALSを探す
find_package(SUNDIALS REQUIRED)
find_library(KLU_LIBRARY klu HINTS /usr/lib/x86_64-linux-gnu)

# CasADiを探す
find_package(casadi REQUIRED)

# 実行ファイルを作成
add_executable(${PROJECT_NAME} ${SOURCES})

# include ディレクトリを設定
target_include_directories(${PROJECT_NAME} 
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
        ${MPI_INCLUDE_PATH}
        /usr/local/include/cppad
        ${SUNDIALS_INCLUDE_DIR}
        /usr/include/suitesparse
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# CppADのヘッダ検索パスを追加
include_directories(/usr/local/include/cppad)

# CppADライブラリをリンク
target_link_libraries(${PROJECT_NAME} PRIVATE /usr/local/lib/libcppad_lib.so)

# MPIライブラリをリンク
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX)

# Eigen3ライブラリをリンク
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)

# SUNDIALSライブラリをリンク
target_link_libraries(${PROJECT_NAME} PRIVATE sundials_cvodes)
target_link_libraries(${PROJECT_NAME} PRIVATE /usr/local/lib/libsundials_core.so)
target_link_libraries(${PROJECT_NAME} PRIVATE /usr/local/lib/libsundials_sunlinsolklu.so)

# CasADiライブラリをリンク
target_link_libraries(${PROJECT_NAME} PRIVATE casadi)

if (KLU_LIBRARY)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${KLU_LIBRARY})
endif()