# Global value for building project
# Priority: command line > json file > cmake

# Set the minimum version of cmake required for the project
cmake_minimum_required(VERSION 3.5)


# Load json file
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/config.json jsonConfigSourceStr)


# parse json source string
include(JSONParser.cmake)
sbeParseJson(jsonConfig jsonConfigSourceStr)


# # debug print configures parsed
# foreach(var ${jsonConfig})
#     message("${var} = ${${var}}")
# endforeach()


# switch
option(CTEST_FLAGS 
       "This variable controls whether the test is performed"
       ${jsonConfig.control.is_open_ctest})


# The variables set here do not modify the command line settings
set(COMPILE_MODE ${jsonConfig.control.compile_mode} CACHE STRING "compile mode")

set(NG_MAX ${jsonConfig.gas.ng_max} CACHE STRING "the max gas number")

set(NS_MAX ${jsonConfig.gas.ns_max} CACHE STRING "the max specy number per gas")

# lib
set(NAME_PROJECT "pic2d-mpi" CACHE STRING "project name")
set(NAME_SOLVER "petsc-solver" CACHE STRING "petsc solver name")
set(NAME_FLIB "fortran-lib" CACHE STRING "fortran lib name")

# path
set(PATH_PROJECT ${PROJECT_SOURCE_DIR} CACHE STRING "project path")
set(PATH_SCRIPT ${PATH_PROJECT}/script CACHE STRING "script path")
set(PATH_OUTPUT_TMP "run")
get_filename_component(PATH_OUTPUT_ABS ${PATH_OUTPUT_TMP} ABSOLUTE)
set(PATH_OUTPUT ${PATH_OUTPUT_ABS} CACHE STRING "output exe path")

set(PATH_TEST ${PATH_PROJECT}/test CACHE STRING "test path")
set(PATH_LIB ${PATH_PROJECT}/lib CACHE STRING "lib path")
set(PATH_SRC ${PATH_PROJECT}/src CACHE STRING "src path")

set(PATH_RESTART ${PATH_OUTPUT}/restart CACHE STRING "restart file path")
set(PATH_DIAG ${PATH_OUTPUT}/diag CACHE STRING "diag file path")
set(PATH_CHECK ${PATH_OUTPUT}/check CACHE STRING "check file path")
set(PATH_INPUT ${PATH_OUTPUT}/input CACHE STRING "input file path")
set(PATH_INIT ${PATH_OUTPUT}/init CACHE STRING "init file path")

set(PATH_HDF5 $ENV{HDF5_ENV_PATH} CACHE STRING "hdf5 lib path")
set(PATH_PETSC $ENV{PETSC_ENV_PATH} CACHE STRING "petsc lib path")
set(PATH_PETSC_SOLVER ${PATH_PROJECT}/lib/solver CACHE STRING "petsc solver lib path")
set(PATH_SUNDIALS $ENV{SUNDIALS_ENV_PATH} CACHE STRING "sundials lib path")

# clean parsed variables
sbeClearJson(jsonConfig)

# sundials 版本 使用正则表达式提取版本号
file(GLOB search_target_files "${PATH_SUNDIALS}/lib/libsundials_cvode*.so.*.*.*")
foreach(file ${search_target_files})
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" version ${file})
    if(version)
        string(REPLACE "." ";" version_list ${version})
        list(GET version_list 0 sundials_major_version)
        list(GET version_list 1 sundials_minor_version)
        list(GET version_list 2 sundials_patch_version)
        # message("文件：${file}")
        # message("版本号：${sundials_major_version}.${sundials_minor_version}.${sundials_patch_version}")
    endif()
endforeach()

# print info
# if (CTEST_FLAGS)
#     message("CTEST_FLAGS Open")
# else()
#     message("CTEST_FLAGS not Open")
# endif()

# if (PBS_FLAGS)
#     message("PBS_FLAGS Open")
# else()
#     message("PBS_FLAGS not Open")
# endif()

# message("MPI_SIZE = ${MPI_SIZE}")
# message("MPI_MODE = ${MPI_MODE}")
# message("COMPILE_MODE = ${COMPILE_MODE}")

# message("PATH_PROJECT = ${PATH_PROJECT}")
# message("PATH_SCRIPT = ${PATH_SCRIPT}")
# message("PATH_OUTPUT = ${PATH_OUTPUT}")
# message("PATH_TEST = ${PATH_TEST}")

# message("PATH_RESTART = ${PATH_RESTART}")
# message("PATH_DIAG = ${PATH_DIAG}")
# message("PATH_CHECK = ${PATH_CHECK}")
# message("PATH_INPUT = ${PATH_INPUT}")
# message("PATH_INIT = ${PATH_INIT}")

# message("PATH_HDF5 = ${PATH_HDF5}")
# message("PATH_PETSC = ${PATH_PETSC}")
# message("PATH_PETSC_SOLVER = ${PATH_PETSC_SOLVER}")
