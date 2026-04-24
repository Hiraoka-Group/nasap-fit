cmake_minimum_required(VERSION 3.18)

project(nasap_fit_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =========================
# Options
# =========================
option(NASAP_BUILD_EXECUTABLE "Build native executable" OFF)
option(NASAP_BUILD_PYTHON_MODULE "Build Python extension module" ON)
option(NASAP_ENABLE_MPI "Enable MPI support" OFF)

# =========================
# Dependency roots
# =========================
# Assumed layout:
#   extern/eigen/...
#   build/deps/install/sundials/{include,lib,...}
#   build/deps/install/suitesparse/{include,lib,...}
#
# These can be overridden from the command line if needed.
set(NASAP_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/deps" CACHE PATH "Root directory for locally built dependencies")
set(NASAP_SUNDIALS_ROOT "${NASAP_DEPS_DIR}/install/sundials" CACHE PATH "Installed SUNDIALS prefix")
set(NASAP_SUITESPARSE_ROOT "${NASAP_DEPS_DIR}/install/suitesparse" CACHE PATH "Installed SuiteSparse prefix")
set(NASAP_EIGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen" CACHE PATH "Eigen header root")

# Help CMake find package config files inside the local dependency prefixes.
list(PREPEND CMAKE_PREFIX_PATH
    "${NASAP_SUNDIALS_ROOT}"
    "${NASAP_SUITESPARSE_ROOT}"
)

# =========================
# Sources
# =========================
file(GLOB_RECURSE NASAP_ALL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

set(NASAP_EXE_SOURCES ${NASAP_ALL_SOURCES})
list(REMOVE_ITEM NASAP_EXE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/module.cpp"
)

set(NASAP_PY_SOURCES ${NASAP_ALL_SOURCES})
list(REMOVE_ITEM NASAP_PY_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"
)

# =========================
# Dependencies
# =========================

# Eigen: vendored header-only dependency
# if (NOT EXISTS "${NASAP_EIGEN_ROOT}/Eigen/Core")
#     message(FATAL_ERROR
#         "Eigen headers not found at ${NASAP_EIGEN_ROOT}. "
#         "Expected ${NASAP_EIGEN_ROOT}/Eigen/Core"
#     )
# endif()

# SUNDIALS: require local installed prefix, not system /usr/local

if (DEFINED ENV{NASAP_SUNDIALS_ROOT})
    set(NASAP_SUNDIALS_ROOT "$ENV{NASAP_SUNDIALS_ROOT}" CACHE PATH "SUNDIALS root" FORCE)
endif()

if (DEFINED ENV{SUNDIALS_DIR})
    set(SUNDIALS_DIR "$ENV{SUNDIALS_DIR}" CACHE PATH "SUNDIALS_DIR" FORCE)
endif()

message(STATUS "NASAP_SUNDIALS_ROOT=${NASAP_SUNDIALS_ROOT}")
message(STATUS "SUNDIALS_DIR=${SUNDIALS_DIR}")

find_package(SUNDIALS CONFIG REQUIRED
    PATHS
      "${NASAP_SUNDIALS_ROOT}"
      "${NASAP_SUNDIALS_ROOT}/lib/cmake/sundials"
      "${SUNDIALS_DIR}"
    NO_DEFAULT_PATH
)

# Optional MPI
if (NASAP_ENABLE_MPI)
    find_package(MPI REQUIRED)
endif()

# SuiteSparse / KLU from local installed prefix
find_path(KLU_INCLUDE_DIR
    NAMES klu.h
    PATHS "${NASAP_SUITESPARSE_ROOT}/include"
    NO_DEFAULT_PATH
)

find_library(KLU_LIBRARY
    NAMES klu libklu
    PATHS "${NASAP_SUITESPARSE_ROOT}/lib" "${NASAP_SUITESPARSE_ROOT}/lib64"
    NO_DEFAULT_PATH
)

find_library(AMD_LIBRARY
    NAMES amd libamd
    PATHS "${NASAP_SUITESPARSE_ROOT}/lib" "${NASAP_SUITESPARSE_ROOT}/lib64"
    NO_DEFAULT_PATH
)

find_library(COLAMD_LIBRARY
    NAMES colamd libcolamd
    PATHS "${NASAP_SUITESPARSE_ROOT}/lib" "${NASAP_SUITESPARSE_ROOT}/lib64"
    NO_DEFAULT_PATH
)

find_library(BTF_LIBRARY
    NAMES btf libbtf
    PATHS "${NASAP_SUITESPARSE_ROOT}/lib" "${NASAP_SUITESPARSE_ROOT}/lib64"
    NO_DEFAULT_PATH
)

# If you built SUNDIALS with KLU support, these should exist.
if (NOT KLU_INCLUDE_DIR)
    message(FATAL_ERROR "KLU headers not found under ${NASAP_SUITESPARSE_ROOT}/include")
endif()

if (NOT KLU_LIBRARY)
    message(FATAL_ERROR "KLU library not found under ${NASAP_SUITESPARSE_ROOT}/lib or lib64")
endif()

# AMD/COLAMD/BTF are typically required by KLU
foreach(_lib AMD_LIBRARY COLAMD_LIBRARY BTF_LIBRARY)
    if (NOT ${_lib})
        message(FATAL_ERROR "${_lib} not found under ${NASAP_SUITESPARSE_ROOT}/lib or lib64")
    endif()
endforeach()

# =========================
# Common target settings
# =========================
function(nasap_apply_common_settings target_name)
    target_include_directories(${target_name}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${NASAP_EIGEN_ROOT}"
            "${KLU_INCLUDE_DIR}"
    )

    if (NASAP_ENABLE_MPI)
        target_link_libraries(${target_name} PRIVATE MPI::MPI_CXX)
        target_compile_definitions(${target_name} PRIVATE NASAP_USE_MPI=1)
    else()
        target_compile_definitions(${target_name} PRIVATE NASAP_USE_MPI=0)
    endif()

    # Prefer CVODES when available (quadrature/sensitivity APIs live there).
    # Fall back to CVODE only if CVODES isn't provided by the installed SUNDIALS.
    if (TARGET SUNDIALS::cvodes)
        target_link_libraries(${target_name} PRIVATE SUNDIALS::cvodes)
    elseif (TARGET SUNDIALS::cvode)
        target_link_libraries(${target_name} PRIVATE SUNDIALS::cvode)
    else()
        message(FATAL_ERROR "Neither SUNDIALS::cvodes nor SUNDIALS::cvode target was found.")
    endif()

    if (TARGET SUNDIALS::nvecserial)
        target_link_libraries(${target_name} PRIVATE SUNDIALS::nvecserial)
    endif()

    if (TARGET SUNDIALS::sunlinsolklu)
        target_link_libraries(${target_name} PRIVATE SUNDIALS::sunlinsolklu)
    endif()

    if (TARGET SUNDIALS::core)
        target_link_libraries(${target_name} PRIVATE SUNDIALS::core)
    endif()

    # Explicit SuiteSparse linkage for KLU
    target_link_libraries(${target_name}
        PRIVATE
            "${KLU_LIBRARY}"
            "${AMD_LIBRARY}"
            "${COLAMD_LIBRARY}"
            "${BTF_LIBRARY}"
    )
endfunction()

# =========================
# Native executable (optional)
# =========================
if (NASAP_BUILD_EXECUTABLE)
    add_executable(${PROJECT_NAME} ${NASAP_EXE_SOURCES})
    nasap_apply_common_settings(${PROJECT_NAME})
endif()

# =========================
# Python extension module
# =========================
if (NASAP_BUILD_PYTHON_MODULE)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_core ${NASAP_PY_SOURCES})
    nasap_apply_common_settings(_core)

    install(TARGETS _core
        LIBRARY DESTINATION ./nasap_fit
        RUNTIME DESTINATION ./nasap_fit
        ARCHIVE DESTINATION ./nasap_fit
    )

    if (NOT DEFINED EXAMPLE_VERSION_INFO)
        set(EXAMPLE_VERSION_INFO "dev")
    endif()
    target_compile_definitions(_core PRIVATE VERSION_INFO=\"${EXAMPLE_VERSION_INFO}\")
endif()