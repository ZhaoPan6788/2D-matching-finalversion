module sundials_ode_test
    use sundials_ode

    implicit none

    contains

        integer(c_int) function RhsFnSecond(tn, sunvec_y, sunvec_f, user_data) &
            result(ierr2) bind(C, name='RhsFnSecond')
            ! calling variables
            real(c_double), value :: tn        ! current time
            type(N_Vector)        :: sunvec_y  ! solution N_Vector
            type(N_Vector)        :: sunvec_f  ! rhs N_Vector
            type(c_ptr), value    :: user_data ! user-defined data
            ! pointers to data in SUNDIALS vectors
            real(c_double), pointer :: yvec(:)
            real(c_double), pointer :: fvec(:)

            !======= Internals ============
            ! get data arrays from SUNDIALS vectors
            yvec => FN_VGetArrayPointer(sunvec_y)
            fvec => FN_VGetArrayPointer(sunvec_f)

            fvec(1) = yvec(2)
            fvec(2) = 1.d0

            ierr2 = 0
            return

        end function RhsFnSecond


        subroutine test_simple_second_order_equation()
            type(ODESolver) :: ode
            integer(4) :: i, maxstep
            real(8), allocatable :: result(:)

            maxstep = 10000
            allocate(result(maxstep))

            call ode%init(2, 1.d0, c_funloc(RhsFnSecond))

            do i = 1, maxstep
                call ode%update()
                result(i) = ode%out(1)
            end do

            ! ! out file
            ! open(10, file="./second_order_equation_result.txt")
            !     do i = 1, maxstep
            !         write(10, '(*(es20.4, 1x))') result(i)
            !     end do
            ! close(10)

            call ode%destroy()
            deallocate(result)

        end subroutine test_simple_second_order_equation


        integer(c_int) function RhsFnIMN(tn, sunvec_y, sunvec_f, user_data) &
            result(ierr2) bind(C, name='RhsFnIMN')
            ! calling variables
            real(c_double), value :: tn        ! current time
            type(N_Vector)        :: sunvec_y  ! solution N_Vector
            type(N_Vector)        :: sunvec_f  ! rhs N_Vector
            type(c_ptr), value    :: user_data ! user-defined data
            ! pointers to data in SUNDIALS vectors
            real(c_double), pointer :: yvec(:)
            real(c_double), pointer :: fvec(:)

            real(c_double) :: pi = 3.141592653589793238d0
            real(c_double) :: Us = 200.d0
            real(c_double) :: f  = 13.56d6
            real(c_double) :: Rs = 50.d0
            real(c_double) :: Cl = 9.2677d-12
            real(c_double) :: C1 = 150.d-12
            real(c_double) :: C2 = 200.d-12
            real(c_double) :: L  = 4.3d-6
            real(c_double) :: Rm = 0.5d0

            !======= Internals ============
            ! get data arrays from SUNDIALS vectors
            yvec => FN_VGetArrayPointer(sunvec_y)
            fvec => FN_VGetArrayPointer(sunvec_f)

            fvec(1) = (Us*sin(2*pi*f*tn) - (yvec(1)-yvec(2)) / C1) / Rs
            fvec(2) = yvec(3)
            fvec(3) = ((yvec(1)-yvec(2))/C1 - yvec(2)/C2 - Rm*yvec(3) - yvec(2)/Cl) / L

            ierr2 = 0
            return

        end function RhsFnIMN


        subroutine test_imn()
            type(ODESolver) :: ode
            integer(4) :: i, maxstep
            real(8), allocatable :: result(:)

            maxstep = 50000
            allocate(result(maxstep))

            call ode%init(3, 1.d-10, c_funloc(RhsFnIMN))

            do i = 1, maxstep
                call ode%update()
                result(i) = ode%out(2) / 9.2677d-12
            end do

            ! ! out file
            ! open(10, file="./test_imn.txt")
            !     do i = 1, maxstep
            !         write(10, '(*(es20.4, 1x))') dble(i)*1.d-10, result(i)
            !     end do
            ! close(10)

            call ode%destroy()
            deallocate(result)

        end subroutine test_imn

end module sundials_ode_test


program fortran_mpi
    use mpi 
    use sundials_ode_test, only: test_simple_second_order_equation, test_imn

    implicit none

    integer(4) :: size, rank, ierr

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

    if (0 == rank) then
        write(*, *) "test ode 01 simple second oerder equation"
        call test_simple_second_order_equation()
        write(*, *) ""

        write(*, *) "test ode 02 imn"
        call test_imn()
        write(*, *) ""
    end if

    call MPI_FINALIZE(ierr)

end program fortran_mpi
