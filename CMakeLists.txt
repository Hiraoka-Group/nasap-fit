# CMakeLists.txt

# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(nasap_fit_cpp)

# ソースコードを収集
file(GLOB SOURCES "src/*.cpp")

# MPIを探す
find_package(MPI REQUIRED)

# 実行ファイルを作成
add_executable(${PROJECT_NAME} ${SOURCES})

# include ディレクトリを設定
target_include_directories(${PROJECT_NAME} 
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
        ${MPI_INCLUDE_PATH}
)

# MPIライブラリをリンク
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX)