# CTest
if (CTEST_FLAGS)
    enable_testing()

    # 01 test for mpi env
    set(testname "01_test_mpi_env")
    add_executable(${testname} test_mpi_env.F90)
    target_include_directories(${testname} PUBLIC ${PATH_PROJECT}/build)
    target_include_directories(${testname} PUBLIC ${PATH_PETSC_SOLVER}/include)
    target_link_libraries(${testname} PUBLIC ${NAME_FLIB})

    add_test(NAME ${testname} COMMAND "mpirun" "-np" "4" $<TARGET_FILE:${testname}>)

    # 02 test for file name
    set(testname "02_test_file_name")
    add_executable(${testname} test_file_name.F90)
    target_include_directories(${testname} PUBLIC ${PATH_PROJECT}/build)
    target_include_directories(${testname} PUBLIC ${PATH_PETSC_SOLVER}/include)
    target_link_libraries(${testname} PUBLIC ${NAME_FLIB})

    add_test(NAME ${testname} COMMAND "mpirun" "-np" "4" $<TARGET_FILE:${testname}>)

    # 03 test for time
    set(testname "03_test_time")
    add_executable(${testname} test_time.F90)
    target_include_directories(${testname} PUBLIC ${PATH_PROJECT}/build)
    target_include_directories(${testname} PUBLIC ${PATH_PETSC_SOLVER}/include)
    target_link_libraries(${testname} PUBLIC ${NAME_FLIB})

    add_test(NAME ${testname} COMMAND "mpirun" "-np" "4" $<TARGET_FILE:${testname}>)

    # 04 test for hdf5 dump serial
    set(testname "04_test_hdf5_dump_serial")
    add_executable(${testname} test_hdf5_dump_serial.F90)
    target_include_directories(${testname} PUBLIC ${PATH_PROJECT}/build)
    target_include_directories(${testname} PUBLIC ${PATH_PETSC_SOLVER}/include)
    target_link_libraries(${testname} PUBLIC ${NAME_FLIB})

    add_test(NAME ${testname} COMMAND "mpirun" "-np" "4" $<TARGET_FILE:${testname}>)

    # 05 test for hdf5 dump parallel
    set(testname "05_test_hdf5_dump_parallel")
    add_executable(${testname} test_hdf5_dump_parallel.F90)
    target_include_directories(${testname} PUBLIC ${PATH_PROJECT}/build)
    target_include_directories(${testname} PUBLIC ${PATH_PETSC_SOLVER}/include)
    target_link_libraries(${testname} PUBLIC ${NAME_FLIB})

    add_test(NAME ${testname} COMMAND "mpirun" "-np" "8" $<TARGET_FILE:${testname}>)

    # 06 test for domain
    set(testname "06_test_domain")
    add_executable(${testname} test_domain.F90)
    target_include_directories(${testname} PUBLIC ${PATH_PROJECT}/build)
    target_include_directories(${testname} PUBLIC ${PATH_PETSC_SOLVER}/include)
    target_link_libraries(${testname} PUBLIC ${NAME_FLIB})

    add_test(NAME ${testname} COMMAND "mpirun" "-np" "8" $<TARGET_FILE:${testname}>)

    # 07 test for field com
    set(testname "07_test_field_com")
    add_executable(${testname} test_field_com.F90)
    target_include_directories(${testname} PUBLIC ${PATH_PROJECT}/build)
    target_include_directories(${testname} PUBLIC ${PATH_PETSC_SOLVER}/include)
    target_link_libraries(${testname} PUBLIC ${NAME_FLIB})

    add_test(NAME ${testname} COMMAND "mpirun" "-np" "8" $<TARGET_FILE:${testname}>)

    # 08 test for particle com
    set(testname "08_test_particle_com")
    add_executable(${testname} test_particle_com.F90)
    target_include_directories(${testname} PUBLIC ${PATH_PROJECT}/build)
    target_include_directories(${testname} PUBLIC ${PATH_PETSC_SOLVER}/include)
    target_link_libraries(${testname} PUBLIC ${NAME_FLIB})

    add_test(NAME ${testname} COMMAND "mpirun" "-np" "8" $<TARGET_FILE:${testname}>)

    # 09 test for petsc solver
    set(testname "09_test_petsc_solver")
    add_executable(${testname} test_petsc_solver.F90)
    target_include_directories(${testname} PUBLIC ${PATH_PROJECT}/build)
    target_include_directories(${testname} PUBLIC ${PATH_PETSC_SOLVER}/include)
    target_link_libraries(${testname} PUBLIC ${NAME_FLIB})

    add_test(NAME ${testname} COMMAND "mpirun" "-np" "8" $<TARGET_FILE:${testname}>)

    # 10 test for onestep
    set(testname "10_test_one_step")
    add_executable(${testname} test_one_step.F90)
    target_include_directories(${testname} PUBLIC ${PATH_PROJECT}/build)
    target_include_directories(${testname} PUBLIC ${PATH_PETSC_SOLVER}/include)
    target_link_libraries(${testname} PUBLIC ${NAME_FLIB})

    add_test(NAME ${testname} COMMAND "mpirun" "-np" "4" $<TARGET_FILE:${testname}>)

    # 11 test for onestep
    set(testname "11_test_particle_bundle")
    add_executable(${testname} test_particle_bundle.F90)
    target_include_directories(${testname} PUBLIC ${PATH_PROJECT}/build)
    target_include_directories(${testname} PUBLIC ${PATH_PETSC_SOLVER}/include)
    target_link_libraries(${testname} PUBLIC ${NAME_FLIB})

    add_test(NAME ${testname} COMMAND "mpirun" "-np" "4" $<TARGET_FILE:${testname}>)

    # 12 test for ode
    set(testname "12_test_ode")
    add_executable(${testname} test_ode.F90)
    target_include_directories(${testname} PUBLIC ${PATH_PROJECT}/build)
    target_include_directories(${testname} PUBLIC ${PATH_PETSC_SOLVER}/include)
    target_link_libraries(${testname} PUBLIC ${NAME_FLIB})

    add_test(NAME ${testname} COMMAND "mpirun" "-np" "4" $<TARGET_FILE:${testname}>)

    # 13 test for recb
    set(testname "13_test_recb")
    add_executable(${testname} test_recb.F90)
    target_include_directories(${testname} PUBLIC ${PATH_PROJECT}/build)
    target_include_directories(${testname} PUBLIC ${PATH_PETSC_SOLVER}/include)
    target_link_libraries(${testname} PUBLIC ${NAME_FLIB})

    add_test(NAME ${testname} COMMAND $<TARGET_FILE:${testname}>)

endif()
