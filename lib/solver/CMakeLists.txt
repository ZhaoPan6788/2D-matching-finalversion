# Set the minimum version of cmake required for the project
cmake_minimum_required(VERSION 3.5)


# project
project(${NAME_SOLVER} LANGUAGES CXX C)


# support for c++ 11
set(CMAKE_CXX_STANDARD 11)


# source files
file(GLOB_RECURSE PETSC_SRCS ${CMAKE_CURRENT_LIST_DIR}/source/*.cpp)


# build lib
add_library(${NAME_SOLVER} STATIC ${PETSC_SRCS})


# include
target_include_directories(${NAME_SOLVER} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include/)
target_include_directories(${NAME_SOLVER} PUBLIC ${PATH_PETSC}/include)


# link
find_library(PETSc_LIB petsc ${PATH_PETSC}/lib)
target_link_libraries(${NAME_SOLVER} PUBLIC ${PETSc_LIB})

target_link_libraries(${NAME_SOLVER} PUBLIC ${MPI_LIBRARIES})

