# search fortran source code
file(GLOB_RECURSE Fortran_SRCS  ./base/numerical/Numrical.F90
                                ./base/numerical/bspline.F90
                                ./base/FileName.F90
                                ./base/Constants.F90
                                ./base/ftime.F90
                                ./base/ParallelDump.F90
                                ./base/ControlFlow.F90
                                ./base/SpecyOne.F90
                                ./base/init.F90
                                ./base/Geometry.F90
                                ./base/Rezoning.F90
                                ./diag/calculate.F90
                                ./diag/grid/GridBase.F90
                                ./diag/grid/Grid1D.F90
                                ./diag/grid/Grid2D.F90
                                ./diag/grid/Grid3D.F90
                                ./diag/grid/GridControlFlow.F90
                                ./diag/diagstep.F90
                                ./diag/diagParticleField.F90
                                ./diag/Grid.f90
                                ./pic/domain/Domain.F90
                                ./pic/domain/PICCom.F90
                                ./pic/circuit/Ground.F90
                                ./pic/circuit/Voltage.F90
                                ./pic/circuit/Circuits.F90
                                ./pic/circuit/Pulse.F90
                                ./pic/circuit/transline.f90
                                ./pic/circuit/RLC.F90
                                ./pic/circuit/IMN.F90
                                ./pic/field/FieldEM.f90
                                ./pic/field/FieldOne.f90
                                ./pic/field/FieldSolver.f90
                                ./pic/field/FieldSource.f90
                                ./pic/particle/ParticleOne.f90
                                ./pic/particle/ParticleOneIndex.f90
                                ./pic/particle/ParticleBundle.f90
                                ./pic/particle/ParticleBundleIndex.f90
                                ./pic/fieldboundary/FieldBoundary2DZR.F90
                                ./pic/particleboundary/ParticleBoundary1D.f90
                                ./pic/particleboundary/ParticleMaterial.F90
                                ./pic/materials/Materials.F90
                                ./pic/materials/MaterialsCom.F90
                                ./mc/*
                                )


if (${sundials_major_version} STREQUAL "5")
    file(GLOB_RECURSE additional_files ./pic/circuit/sundials5_solver.f90)
    list(APPEND Fortran_SRCS ${additional_files})

elseif (${sundials_major_version} STREQUAL "6")
    file(GLOB_RECURSE additional_files ./pic/circuit/sundials6_solver.f90)
    list(APPEND Fortran_SRCS ${additional_files})

endif()

set(SOURCE_SRC ${Fortran_SRCS} CACHE STRING "source files in /src")
