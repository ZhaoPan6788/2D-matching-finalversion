# search fortran source code
file(GLOB_RECURSE Fortran_LIBS  mstring/*.f90
                                json/*.F90
                                json/*.inc
                                solver/interface.c)

set(SOURCE_LIB ${Fortran_LIBS} CACHE STRING "source files in /lib")


# petsc solver
add_subdirectory(solver)

