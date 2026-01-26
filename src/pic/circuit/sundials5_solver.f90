module sundials_ode
    use, intrinsic :: iso_c_binding
  
    use fcvode_mod                    ! Fortran interface to CVODE
    use fsundials_nvector_mod         ! Fortran interface to generic N_Vector
    use fnvector_serial_mod           ! Fortran interface to serial N_Vector
    use fsundials_nonlinearsolver_mod ! Fortran interface to generic SUNNonlinearSolver
    use fsunnonlinsol_fixedpoint_mod  ! Fortran interface to fixed point SUNNonlinearSolver

    implicit none

    type ODESolver
        integer(c_long) :: neq
        real(c_double), allocatable :: y(:)
        real(c_double), allocatable :: out(:)

        real(c_double) :: dt

        real(c_double) :: tstart     ! initial time
        real(c_double) :: tend       ! final time
        real(c_double) :: rtol, atol ! relative and absolute tolerance
        real(c_double) :: dtout      ! output time interval
        real(c_double) :: tout       ! output time
        real(c_double) :: tcur(1)    ! current time
        integer(c_int) :: ierr       ! error flag from C functions
        integer(c_int) :: nout       ! number of outputs
        integer(c_int) :: outstep    ! output loop counter
        type(c_ptr)    :: cvode_mem  ! CVODE memory
      
        type(N_Vector), pointer           :: sunvec_y ! sundials vector
        type(SUNNonlinearSolver), pointer :: sunnls   ! sundials fixed-point nonlinear solver
  
        logical :: is_init = .False.

        contains

            procedure :: init => initODESolver
            procedure :: update => updateODESolver
            procedure :: destroy => destroyODESolver

    end type ODESolver


    contains

        subroutine initODESolver(this, neq, dt, rhs_fun, ts, y0)
            class(ODESolver), intent(inout) :: this
            integer(c_long), intent(in) :: neq
            real(c_double), intent(in) :: dt
            real(c_double), optional, intent(in) :: ts
            real(c_double), optional, intent(in) :: y0(neq)
            type(c_funptr), intent(in), value :: rhs_fun
            integer(c_int) :: ierr

            if (neq > 0 .and. dt > 0.d0) then
                ! initialize ODE
                this%neq    = neq
                this%tstart = 0.0d0
                if (present(ts)) this%tstart = ts
                this%dt     = dt
                this%tend   = this%dt
                this%tcur   = this%tstart
                this%tout   = this%tstart
                this%dtout  = this%dt
                this%nout   = ceiling(this%tend / this%dtout)
            
                call this%destroy()
                allocate(this%y(this%neq))
                allocate(this%out(this%neq))

                this%y = 0.d0
                this%out = 0.d0
                if (present(y0)) then
                    this%y(1:neq) = y0(1:neq)
                    this%out(1:neq) = y0(1:neq)
                end if

                ! create SUNDIALS N_Vector
                this%sunvec_y => FN_VMake_Serial(this%neq, this%y)
                if (.not. associated(this%sunvec_y)) then
                    print *, 'ERROR: sunvec = NULL'
                    stop 1
                end if
            
                ! create CVode memory
                this%cvode_mem = FCVodeCreate(CV_ADAMS)
                if (.not. c_associated(this%cvode_mem)) then
                    print *, 'ERROR: cvode_mem = NULL'
                    stop 1
                end if
            
                ! initialize CVode
                ierr = FCVodeInit(this%cvode_mem, rhs_fun, this%tstart, this%sunvec_y)
                if (ierr /= 0) then
                    print *, 'Error in FCVodeInit, ierr = ', ierr, '; halting'
                    stop 1
                end if
            
                ! set relative and absolute tolerances
                this%rtol = 1.0d-6
                this%atol = 1.0d-10
            
                ierr = FCVodeSStolerances(this%cvode_mem, this%rtol, this%atol)
                if (ierr /= 0) then
                    print *, 'Error in FCVodeSStolerances, ierr = ', ierr, '; halting'
                    stop 1
                end if

                ! create fixed point nonlinear solver object
                this%sunnls => FSUNNonlinSol_FixedPoint(this%sunvec_y, 0)
                if (.not. associated(this%sunnls)) then
                    print *,'ERROR: sunnls = NULL'
                    stop 1
                end if

                ! attache nonlinear solver object to CVode
                ierr = FCVodeSetNonlinearSolver(this%cvode_mem, this%sunnls)
                if (ierr /= 0) then
                    print *, 'Error in FCVodeSetNonlinearSolver, ierr = ', ierr, '; halting'
                    stop 1
                end if

                this%is_init = .True.

            else
                write(*, *) "The input parameters of the TransLine are invalid."
                stop
            end if

        end subroutine initODESolver


        subroutine updateODESolver(this)
            class(ODESolver), intent(inout) :: this
            integer(c_int) :: ierr
            integer(c_int) :: i

            ! call CVode
            this%tout = this%tout + this%dtout
            ierr = FCVode(this%cvode_mem, this%tout, this%sunvec_y, this%tcur, CV_NORMAL)
            if (ierr /= 0) then
                print *, 'Error in FCVODE, ierr = ', ierr, '; halting'
                stop 1
            end if

            this%out(1:this%neq) = this%y(1:this%neq)

        end subroutine updateODESolver


        subroutine destroyODESolver(this)
            class(ODESolver), intent(inout) :: this
            integer(c_int) :: ierr

            if (allocated(this%y)) deallocate(this%y)
            if (allocated(this%out)) deallocate(this%out)

            ! clean up
            if (this%is_init) then
                call FCVodeFree(this%cvode_mem)
                ierr = FSUNNonLinSolFree(this%sunnls)
                call FN_VDestroy(this%sunvec_y)

                this%is_init = .False.
            end if

        end subroutine destroyODESolver

end module sundials_ode
