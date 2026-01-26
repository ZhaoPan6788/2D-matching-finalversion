program fortran_mpi
    use mpi
    use ModuleParticleBundle

    implicit none

    integer(4) :: size, rank, ierr, i, j, k
    integer(4), parameter :: pb_num = 16
    type(ParticleBundle) :: pb(pb_num)

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

    do i = 1, pb_num
        call pb(i)%Init(100, 10, 1000000000)
    end do

    ! do j = 1000000, 1000050
    !     do i = 1, pb_num
    !         ! do k = 1, 1000
    !             call pb(i)%Adjust(j)
    !         ! end do
    !     end do

    !     if (0 == rank) write(*, *) j
    ! end do

    ! do j = 1000050, 100010, -10000
    !     do i = 1, pb_num
    !         ! do k = 1, 1000
    !             call pb(i)%Adjust(j)
    !         ! end do
    !     end do

    !     if (0 == rank) write(*, *) j
    ! end do

    ! do j = 1000000, 1000050
    !     do i = 1, pb_num
    !         ! do k = 1, 1000
    !             call pb(i)%Adjust(j)
    !         ! end do
    !     end do

    !     if (0 == rank) write(*, *) j
    ! end do

    do i = 1, pb_num
        call pb(i)%Destroy()
    end do

    do i = 0, size-1
        if (rank == i) then
            write(*, *) "hello", rank, size
        end if

        call MPI_BARRIER(MPI_COMM_WORLD, ierr)
    end do

    call MPI_FINALIZE(ierr)

end program fortran_mpi