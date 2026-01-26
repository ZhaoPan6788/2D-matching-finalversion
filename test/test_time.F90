program fortran_mpi
    use mpi
    use ModuleFTime

    implicit none

    integer(4) :: size, rank, ierr, i
    type(SimpleTimer) :: st

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

    call st%init()
    call st%unit(TIME_UNIT_TYPE_MILLISECOND)

    do i = 1, 5
        call st%time()
        if (0 == rank) write(*, '(i4, f16.6, f16.6)') rank, st%getT(), st%getD()
        ! call st%show()
    end do

    call MPI_FINALIZE(ierr)

end program fortran_mpi