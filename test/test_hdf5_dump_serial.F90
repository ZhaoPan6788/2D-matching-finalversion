program fortran_mpi
    use mpi
    use ModuleParallelDump

    implicit none

    ! Variables
    type(HDF5_PDump) :: hdf5pdump
    real(8) :: x(33), y(33)
    integer(4) :: size, rank, ierr, i

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

    ! set x and y
    do i = 1, 33
        x(i) = i
        y(i) = i*i
    end do

    ! storage
    if (0 == rank) then
        call hdf5pdump%init(filename="test_hdf5_dump_serial.h5", mode="write", serial=.True.)
        call hdf5pdump%open()

        call hdf5pdump%write("/x", x)
        call hdf5pdump%write("/y", y)

        call hdf5pdump%close()
    end if

    call MPI_FINALIZE(ierr)

end program fortran_mpi
