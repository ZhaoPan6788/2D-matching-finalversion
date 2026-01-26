program fortran_mpi
    use mpi

    implicit none

    integer(4) :: size, rank, ierr, i

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

    do i = 0, size-1
        if (rank == i) then
            write(*, *) "hello", rank, size
        end if

        call MPI_BARRIER(MPI_COMM_WORLD, ierr)
    end do

    call MPI_FINALIZE(ierr)

end program fortran_mpi