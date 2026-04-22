# pybind11ビルド用のCMakeLists
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

option(NASAP_BUILD_EXECUTABLE "Build the nasap_fit_cpp executable" ON)
option(NASAP_BUILD_PYTHON_MODULE "Build the Python extension module (_core)" ON)

# ソースコードを収集
file(GLOB_RECURSE NASAP_ALL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

set(NASAP_EXE_SOURCES ${NASAP_ALL_SOURCES})
list(REMOVE_ITEM NASAP_EXE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/module.cpp"
)

set(NASAP_PY_SOURCES ${NASAP_ALL_SOURCES})
list(REMOVE_ITEM NASAP_PY_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"
)


# MPIを探す
find_package(MPI REQUIRED)

# Eigen3を探す
# find_package(Eigen3 REQUIRED)

# SUNDIALSを探す
find_package(SUNDIALS REQUIRED)
find_library(KLU_LIBRARY klu HINTS /usr/lib/x86_64-linux-gnu)

## CasADi は依存から除外（必要なら再導入してください）
# find_package(casadi REQUIRED)

function(nasap_apply_common_settings target_name)
    target_include_directories(${target_name}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${MPI_INCLUDE_PATH}
            /usr/include/suitesparse
    )

    if (DEFINED SUNDIALS_INCLUDE_DIRS)
        target_include_directories(${target_name} PRIVATE ${SUNDIALS_INCLUDE_DIRS})
    elseif (DEFINED SUNDIALS_INCLUDE_DIR)
        target_include_directories(${target_name} PRIVATE ${SUNDIALS_INCLUDE_DIR})
    endif()

    target_link_libraries(${target_name} PRIVATE MPI::MPI_CXX)
    # target_link_libraries(${target_name} PRIVATE Eigen3::Eigen)
    # target_link_libraries(${target_name} PRIVATE casadi)

    # SUNDIALS libraries (prefer imported targets when available)
    if (TARGET SUNDIALS::cvodes)
        target_link_libraries(${target_name} PRIVATE SUNDIALS::cvodes)
    else()
        target_link_libraries(${target_name} PRIVATE sundials_cvodes)
    endif()

    if (TARGET SUNDIALS::core)
        target_link_libraries(${target_name} PRIVATE SUNDIALS::core)
    elseif (EXISTS "/usr/local/lib/libsundials_core.so")
        target_link_libraries(${target_name} PRIVATE /usr/local/lib/libsundials_core.so)
    endif()

    if (TARGET SUNDIALS::sunlinsolklu)
        target_link_libraries(${target_name} PRIVATE SUNDIALS::sunlinsolklu)
    elseif (EXISTS "/usr/local/lib/libsundials_sunlinsolklu.so")
        target_link_libraries(${target_name} PRIVATE /usr/local/lib/libsundials_sunlinsolklu.so)
    endif()

    if (KLU_LIBRARY)
        target_link_libraries(${target_name} PRIVATE ${KLU_LIBRARY})
    endif()
endfunction()

if (NASAP_BUILD_EXECUTABLE)
    add_executable(${PROJECT_NAME} ${NASAP_EXE_SOURCES})
    nasap_apply_common_settings(${PROJECT_NAME})
endif()

if (NASAP_BUILD_PYTHON_MODULE)
    # Prefer system/venv pybind11; fall back to in-tree submodule if present.
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pybind11/CMakeLists.txt")
        add_subdirectory(pybind11)
    else()
        find_package(pybind11 CONFIG REQUIRED)
    endif()

    # Build the Python extension module that matches PYBIND11_MODULE(_core, ...)
    pybind11_add_module(_core ${NASAP_PY_SOURCES})
    nasap_apply_common_settings(_core)

    # VERSION_INFO is optionally passed by the build system; provide a sane default.
    if (NOT DEFINED EXAMPLE_VERSION_INFO)
        set(EXAMPLE_VERSION_INFO "dev")
    endif()
    target_compile_definitions(_core PRIVATE VERSION_INFO=\"${EXAMPLE_VERSION_INFO}\")
endif()