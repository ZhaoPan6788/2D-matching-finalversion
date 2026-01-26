cmake_minimum_required(VERSION 3.5)

# project
project(${NAME_PROJECT} LANGUAGES Fortran C)


# load config from config.json
include(Configure.cmake)


# MPI
find_package(MPI)


# set compile options
# set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -heap-arrays -fpp -traceback -check bounds -g -O0")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -heap-arrays -fpp -traceback")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,stack-size=10000000")


# Folder operations
# file(REMOVE_RECURSE ${PATH_OUTPUT})

file(MAKE_DIRECTORY ${PATH_OUTPUT})
file(MAKE_DIRECTORY ${PATH_RESTART})
file(MAKE_DIRECTORY ${PATH_DIAG})
file(MAKE_DIRECTORY ${PATH_CHECK})
file(MAKE_DIRECTORY ${PATH_INIT})

file(COPY
        ${PATH_PROJECT}/config.json
    DESTINATION
        ${PATH_OUTPUT})

file(COPY
        ${PATH_PROJECT}/input/
    DESTINATION
        ${PATH_INPUT})


# compile settings
file(GLOB_RECURSE Main_SRCS ${PATH_SRC}/main.F90 onestep.F90)
add_executable(${NAME_PROJECT} ${Main_SRCS})
set(EXECUTABLE_OUTPUT_PATH ${PATH_OUTPUT})
set(CMAKE_BUILD_TYPE ${COMPILE_MODE} CACHE STRING "set build type to debug | release | other")


# add sub directory
add_subdirectory(src)
add_subdirectory(lib)


# add shared lib
add_library(${NAME_FLIB} STATIC ${SOURCE_SRC} ${SOURCE_LIB})

# definitions
target_compile_definitions(${NAME_FLIB} PUBLIC "NG_MAX=${NG_MAX}")

target_compile_definitions(${NAME_FLIB} PUBLIC "NS_MAX=${NS_MAX}")


# include
target_include_directories(${NAME_FLIB} PUBLIC ${PATH_HDF5}/include)

target_include_directories(${NAME_FLIB} PUBLIC ${PATH_PETSC_SOLVER}/include)

target_include_directories(${NAME_FLIB} PUBLIC ${PATH_SUNDIALS}/fortran)


# link HDF5
find_library(HDF5_LIB_HDF5 hdf5 ${PATH_HDF5}/lib)
target_link_libraries(${NAME_FLIB} PUBLIC ${HDF5_LIB_HDF5})

find_library(HDF5_LIB_HDF5_FORTRAN hdf5_fortran ${PATH_HDF5}/lib)
target_link_libraries(${NAME_FLIB} PUBLIC ${HDF5_LIB_HDF5_FORTRAN})

find_library(HDF5_LIB_HDF5_HL hdf5_hl ${PATH_HDF5}/lib)
target_link_libraries(${NAME_FLIB} PUBLIC ${HDF5_LIB_HDF5_HL})

find_library(HDF5_LIB_HDF5HL_FORTRAN hdf5hl_fortran ${PATH_HDF5}/lib)
target_link_libraries(${NAME_FLIB} PUBLIC ${HDF5_LIB_HDF5HL_FORTRAN})

# link sundials
find_library(SUNDIALS_SOLVER_LIB sundials_cvode ${PATH_SUNDIALS}/lib DOC "CVODE library")
target_link_libraries(${NAME_FLIB} PUBLIC ${SUNDIALS_SOLVER_LIB})

find_library(SUNDIALS_SOLVER_FLIB sundials_fcvode_mod ${PATH_SUNDIALS}/lib DOC "CVODE Fortran-C library")
target_link_libraries(${NAME_FLIB} PUBLIC ${SUNDIALS_SOLVER_FLIB})

if("${NVEC_LIB}" STREQUAL "")
  # No additional NVECTOR library necessary
else()
    find_library(SUNDIALS_NVEC_LIB sundials_nvecserial ${PATH_SUNDIALS}/lib DOC "NVECTOR library")
    target_link_libraries(${NAME_FLIB} PUBLIC ${SUNDIALS_NVEC_LIB})

    find_library(SUNDIALS_NVEC_FLIB sundials_fnvecserial_mod ${PATH_SUNDIALS}/lib DOC "NVECTOR library")
    target_link_libraries(${NAME_FLIB} PUBLIC ${SUNDIALS_NVEC_FLIB})
endif()

if("${SUNLS_LIB}" STREQUAL "")
  # No additional SUNLinearSolver library necessary
else()
    find_library(SUNDIALS_SUNLS_LIB ${PATH_SUNDIALS}/lib DOC "SUNLinearSolver library")
    target_link_libraries(${NAME_FLIB} PUBLIC ${SUNDIALS_SUNLS_LIB})

    find_library(SUNDIALS_SUNLS_FLIB ${PATH_SUNDIALS}/lib DOC "SUNLinearSolver library")
    target_link_libraries(${NAME_FLIB} PUBLIC ${SUNDIALS_SUNLS_FLIB})
endif()

# link solver and mpi
target_link_libraries(${NAME_FLIB} PUBLIC ${NAME_SOLVER})

target_link_libraries(${NAME_FLIB} PUBLIC MPI::MPI_Fortran)


# link flib to master
target_link_libraries(${NAME_PROJECT} PUBLIC ${NAME_FLIB})


# ctest
# add_subdirectory(test)
