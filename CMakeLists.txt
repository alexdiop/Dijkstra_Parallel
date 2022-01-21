cmake_minimum_required(VERSION 3.20)
project(Maybe)

set(CMAKE_CXX_STANDARD 14)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

add_executable(Maybe dijkstraMPI.cpp)
