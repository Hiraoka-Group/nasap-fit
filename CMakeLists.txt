# CMakeLists.tst

# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

file(GLOB SOURCES "src/*.cpp")

# Project name
project(nasap_fit_cpp)


add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} 
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
)
