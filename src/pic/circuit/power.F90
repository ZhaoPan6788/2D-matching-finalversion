Module ModulePower
    use mpi
    use ModuleFileName
    use ModuleFFTW

    implicit none

    type PowerControl
        type(FileName) :: IOName
        integer(4) :: period = 0
        real(8) :: power_set = 0.d0
        real(8) :: alpha = 0.d0
        real(8) :: tol = 0.05d0

        real(8) :: power_sum = 0.d0
        integer(4) :: count = 0

        real(8), allocatable :: U_log(:)
        real(8), allocatable :: I_log(:)

        contains

            procedure :: init => initPower
            procedure :: adjust => adjustPower
            procedure :: Fwd => adjustPowerFwd
            procedure :: destroy => destroyPower

    end type PowerControl

    contains

        subroutine initPower(this, control_period, control_power, control_alpha, pname)
            class(PowerControl), intent(inout) :: this
            integer(4), intent(in) :: control_period
            real(8), intent(in) :: control_power
            real(8), optional, intent(in) :: control_alpha
            character(*), optional, intent(in) :: pname

            this%period = control_period
            this%power_set = control_power
            if (present(control_alpha)) then
                this%alpha = control_alpha
            else
                this%alpha = 10.d0
            end if
    
            if (present(pname)) then
                call this%IOName%Init(pname, CHECK_FILE_NAME)
            else
                call this%IOName%Init("power_set", CHECK_FILE_NAME)
            end if

            call this%destroy()
            allocate(this%U_log(this%period))
            allocate(this%I_log(this%period))
            this%U_log = 0.d0
            this%I_log = 0.d0
            this%power_sum = 0.d0
            this%count = 0

        end subroutine initPower


        subroutine adjustPower(this, power_now, vol_set)
            class(PowerControl), intent(inout) :: this
            real(8), intent(in) :: power_now
            real(8), intent(inout) :: vol_set
            real(8) :: vol_last
            integer(4) :: rank, ierr

            this%count = this%count + 1
            this%power_sum = this%power_sum + power_now
            if (this%count == this%period) then
                vol_last = vol_set
                this%power_sum = this%power_sum / dble(this%period)

                if (this%power_sum > 0.d0) then
                    if (this%power_sum > this%power_set) then
                        vol_set = vol_last - this%alpha * (abs(this%power_sum/this%power_set) - 1)

                    else if(this%power_sum<this%power_set) then
                        vol_set = vol_last + this%alpha * (abs(this%power_set/this%power_sum) - 1)

                    else
                        vol_set = vol_last

                    end if

                    if ((vol_set - vol_last) / vol_last > this%tol) then
                        vol_set = (1.0 + this%tol) * vol_last
                    
                    else if ((vol_last - vol_set) / vol_last > this%tol) then
                        vol_set = (1.0 - this%tol) * vol_last
                    
                    end if
                end if

                call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
                if (rank == 0) then
                    open(10, position='append', file=this%IOName%FullName%str)
                        write(10, FMt="(*(es21.14,1x))") this%power_sum, vol_set
                    close(10)
                end if

                this%count = 0
                this%power_sum = 0.d0

            end if

        end subroutine adjustPower


        subroutine adjustPowerFwd(this, U_now, I_now, power_now, dt, freq0, vol_set)
            class(PowerControl), intent(inout) :: this
            real(8), intent(in) :: U_now, I_now, power_now, dt, freq0
            real(8), intent(inout) :: vol_set
            real(8) :: vol_last
            integer(4) :: rank, ierr
            real(8), allocatable :: amp_I(:), pha_I(:), amp_U(:), pha_U(:), freq(:)
            real(8) :: freq_out, amp_out, pha_out
            complex(8) :: V_in, I_in, V_fwd, V_rev
            real(8) :: Z0 = 50.d0, P_fwd, P_rev

            this%count = this%count + 1
            this%U_log(this%count) = U_now
            this%I_log(this%count) = I_now
            this%power_sum = this%power_sum + power_now

            if (this%count == this%period) then
                vol_last = vol_set
                this%power_sum = this%power_sum / dble(this%period)

                allocate(amp_I(this%period))
                allocate(pha_I(this%period))
                allocate(amp_U(this%period))
                allocate(pha_U(this%period))
                allocate(freq(this%period))

                call get_fftw(this%period, this%I_log, this%U_log, dt, amp_I, pha_I, amp_U, pha_U, freq)

                call get_base_freq_info(this%period, freq, amp_U, pha_U, freq0, freq_out, amp_out, pha_out)
                V_in = cmplx(amp_out * cos(pha_out / 180.0 * PI), amp_out * sin(pha_out / 180.0 * PI))

                call get_base_freq_info(this%period, freq, amp_I, pha_I, freq0, freq_out, amp_out, pha_out)
                I_in = cmplx(amp_out * cos(pha_out / 180.0 * PI), amp_out * sin(pha_out / 180.0 * PI))

                V_fwd = (V_in + Z0 * I_in) / 2
                V_rev = (V_in - Z0 * I_in) / 2
                P_fwd = abs(V_fwd)**2 / 2 / Z0
                P_rev = abs(V_rev)**2 / 2 / Z0

                ! if (P_fwd / this%power_sum > 0.1) then
                    this%power_sum = P_fwd
                ! else
                    ! this%power_sum = this%power_sum * 0.5d0
                ! end if

                deallocate(amp_I)
                deallocate(pha_I)
                deallocate(amp_U)
                deallocate(pha_U)
                deallocate(freq)

                if (this%power_sum > 0.d0) then
                    if (this%power_sum > this%power_set) then
                        vol_set = vol_last - this%alpha * (abs(this%power_sum/this%power_set) - 1)

                    else if(this%power_sum<this%power_set) then
                        vol_set = vol_last + this%alpha * (abs(this%power_set/this%power_sum) - 1)

                    else
                        vol_set = vol_last

                    end if

                    if ((vol_set - vol_last) / vol_last > this%tol) then
                        vol_set = (1.0 + this%tol) * vol_last
                    
                    else if ((vol_last - vol_set) / vol_last > this%tol) then
                        vol_set = (1.0 - this%tol) * vol_last
                    
                    end if
                end if

                call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
                if (rank == 0) then
                    open(10, position='append', file=this%IOName%FullName%str)
                        ! write(10, FMt="(*(es21.14,1x))") this%power_sum, vol_set, P_fwd
                    write(10, FMt="(*(es21.14,1x))") this%power_sum, vol_set
                    close(10)
                end if

                this%count = 0
                this%power_sum = 0.d0
                this%U_log = 0.d0
                this%I_log = 0.d0

            end if

        end subroutine adjustPowerFwd


        subroutine destroyPower(this)
            class(PowerControl), intent(inout) :: this

            if (allocated(this%U_log)) deallocate(this%U_log)
            if (allocated(this%I_log)) deallocate(this%I_log)

        end subroutine destroyPower

end Module ModulePower