program fortran_mpi
    use mpi
    use ModuleParallelDump

    implicit none

    ! Variables
    type(HDF5_PDump) :: hdf5pdump
    integer(4),PARAMETER :: lens = 4
    real(8) :: x(lens, lens), y(lens, lens)
    integer(4) :: i, j
    integer(4) :: size, rank, ierr

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

    ! set x and y
    do i = 1, lens
        do j = 1, lens
            x(i, j) = rank
            y(i, j) = rank
        end do
    end do

    ! storage
    call hdf5pdump%init("test_hdf5_dump_parallel.h5", mode="write")
    call hdf5pdump%open()

    call hdf5pdump%write("/x", x, [2, 4])
    call hdf5pdump%write("/y", y, [2, 4])

    call hdf5pdump%close()

    call MPI_FINALIZE(ierr)

end program fortran_mpi
