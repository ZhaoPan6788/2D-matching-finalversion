Module ModuleCircuitRLC
    use mpi
    use Constants
    use ModuleFileName
    use ModuleParallelDump
    use sundials_ode

    implicit none

    integer(4), save :: rlc_source_frequency_num = 0
    real(8), save, allocatable :: rlc_source_frequency(:)
    real(8), save, allocatable :: rlc_source_amplitude(:)
    real(8), save, allocatable :: rlc_source_phase(:)
    real(8), save :: rlc_source_voltage = 0.d0
    real(8), save :: rlc_electrode_voltage = 0.d0
    real(8), save :: rlc_Iconv = 0.d0
    real(8), save :: rlc_eqc = 0.d0
    real(8), save :: rlc_q0 = 0.d0
    real(8), save :: rlc_Rs = 0.d0
    real(8), save :: rlc_Cs = 0.d0
    real(8), save :: rlc_Ls = 0.d0


    type CircuitRLC
        type(FileName) :: IOName
        type(ODESolver) :: ode_rlc              ! restart

    contains

        procedure :: Init   => InitilalizationCircuitRLC
        procedure :: Update => UpdateCircuitRLC
        procedure :: Zero   => ZeroCircuitRLC

        procedure :: Dump => DumpCircuitRLC
        procedure :: Load => LoadCircuitRLC

        procedure :: Destroy => DestroyCircuitRLC

        procedure, private :: LoadCircuitRLCHDF5
        procedure, private :: LoadCircuitRLCDAT

        procedure, private :: DumpCircuitRLCHDF5
        procedure, private :: DumpCircuitRLCDAT

    end Type CircuitRLC

    Type(HDF5_PDump), private :: hdf5CircuitRLCDump

    contains

        integer(c_int) function RhsFnRLC(tn, sunvec_y, sunvec_f, user_data) &
            result(ierr2) bind(C, name='RhsFnRLC')
            ! calling variables
            real(c_double), value :: tn        ! current time
            type(N_Vector)        :: sunvec_y  ! solution N_Vector
            type(N_Vector)        :: sunvec_f  ! rhs N_Vector
            type(c_ptr), value    :: user_data ! user-defined data
            ! pointers to data in SUNDIALS vectors
            real(c_double), pointer :: yvec(:)
            real(c_double), pointer :: fvec(:)
            integer :: i

            !======= Internals ============
            ! get data arrays from SUNDIALS vectors
            yvec => FN_VGetArrayPointer(sunvec_y)
            fvec => FN_VGetArrayPointer(sunvec_f)

            associate(Nf => rlc_source_frequency_num, Vol => rlc_source_voltage, Amp => rlc_source_amplitude, &
                    Fre => rlc_source_frequency, Pha => rlc_source_phase)

                Vol = 0.d0
                do i = 1, Nf
                    Vol = Vol + Amp(i) * DSin(2 * PI * Fre(i) * tn + Pha(i) / 180.d0 * PI)
                end do

            end associate

            fvec(1) = yvec(2)
            fvec(2) = (rlc_source_voltage - (yvec(3) - rlc_q0)/rlc_eqc - yvec(1)/rlc_Cs - yvec(2)*rlc_Rs) / rlc_Ls
            fvec(3) = yvec(2) + rlc_Iconv

            ierr2 = 0

        end function RhsFnRLC


        subroutine InitilalizationCircuitRLC(this, nfreq, circuit_name)
            class(CircuitRLC), intent(inout) :: this
            integer(4), intent(in) :: nfreq
            character(*), optional, intent(in) :: circuit_name
            logical :: alive

            if (present(circuit_name)) then
                call this%IOName%Init(circuit_name, RESTART_FILE_NAME)
            else
                call this%IOName%Init("CircuitRLC", RESTART_FILE_NAME)
            end if

            call this%Destroy()

            rlc_source_frequency_num = nfreq
            allocate(rlc_source_amplitude(rlc_source_frequency_num))
            allocate(rlc_source_frequency(rlc_source_frequency_num))
            allocate(rlc_source_phase(rlc_source_frequency_num))

        end subroutine InitilalizationCircuitRLC


        subroutine UpdateCircuitRLC(this, time, electrode_voltage_boundary, Iconv, eqc, q0)
            class(CircuitRLC), intent(inout) :: this
            real(8), intent(in) :: time, electrode_voltage_boundary, Iconv, eqc, q0
            integer(4) :: i

            rlc_electrode_voltage = electrode_voltage_boundary
            rlc_Iconv = Iconv
            rlc_eqc = eqc
            rlc_q0 = q0

            call this%ode_rlc%update()

            associate(Nf => rlc_source_frequency_num, Vol => rlc_source_voltage, Amp => rlc_source_amplitude, &
                      Fre => rlc_source_frequency, Pha => rlc_source_phase)

                Vol = 0.d0
                do i = 1, Nf
                    Vol = Vol + Amp(i) * DSin(2 * PI * Fre(i) * time + Pha(i) / 180.d0 * PI)
                end do

            end associate

        end subroutine UpdateCircuitRLC


        subroutine ZeroCircuitRLC(this)
            class(CircuitRLC), intent(inout) :: this

            this%ode_rlc%y = 0.d0
            this%ode_rlc%out = 0.d0

        end subroutine ZeroCircuitRLC


        subroutine DumpCircuitRLC(this)
            class(CircuitRLC), intent(inout) :: this

            if (FILE_EXTENSION_MODE_H5 == this%IOName%ExtensionMode) then
                call this%DumpCircuitRLCHDF5()

            else if (FILE_EXTENSION_MODE_DAT == this%IOName%ExtensionMode) then
                call this%DumpCircuitRLCDAT()

            end if

        endsubroutine DumpCircuitRLC


        subroutine LoadCircuitRLC(this)
            class(CircuitRLC), intent(inout) :: this
            logical :: alive

            Inquire(file=this%IOName%FullName%str, exist=alive)
            if (alive) then
                if (FILE_EXTENSION_MODE_H5 == this%IOName%ExtensionMode) then
                    call this%LoadCircuitRLCHDF5()

                else if (FILE_EXTENSION_MODE_DAT == this%IOName%ExtensionMode) then
                    call this%LoadCircuitRLCDAT()

                end if

            end if

        end subroutine LoadCircuitRLC


        subroutine DumpCircuitRLCHDF5(this)
            class(CircuitRLC), intent(inout) :: this
            integer(4) :: size, rank, ierr, neq

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The CircuitRLC is save as hdf5 file."

            call hdf5CircuitRLCDump%init(filename=this%IOName%FullName%str, mode='write', serial=.True.)
            call hdf5CircuitRLCDump%open()

            neq = this%ode_rlc%neq
            call hdf5CircuitRLCDump%writeattr('/', 'ode_neq', neq)
            call hdf5CircuitRLCDump%writeattr('/', 'ode_tcur', this%ode_rlc%tcur(1))

            call hdf5CircuitRLCDump%write('ode_y', this%ode_rlc%y)

            call hdf5CircuitRLCDump%close()

        end subroutine DumpCircuitRLCHDF5


        subroutine DumpCircuitRLCDAT(this)
            class(CircuitRLC), intent(inout) :: this
            integer(4) :: size, rank, ierr

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The CircuitRLC is save as dat file."
            open(10, file=this%IOName%FullName%str)

                write(10, *) this%ode_rlc%neq
                write(10, *) this%ode_rlc%tcur(1)
                write(10, *) this%ode_rlc%y

            close(10)

            return
        end subroutine DumpCircuitRLCDAT


        subroutine LoadCircuitRLCHDF5(this)
            class(CircuitRLC), intent(inout) :: this
            real(8), allocatable :: Temp1D(:)
            logical :: alive
            integer(4) :: size, rank, ierr, neq
            real(8) :: ts

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The CircuitRLC is load from hdf5 file."

            call hdf5CircuitRLCDump%init(filename=this%IOName%FullName%str, mode='read')
            call hdf5CircuitRLCDump%open()

            call hdf5CircuitRLCDump%readattr('/', 'ode_neq', neq)
            this%ode_rlc%neq = neq
            call hdf5CircuitRLCDump%readattr('/', 'ode_tcur', ts)

            allocate(Temp1D(this%ode_rlc%neq))
            call hdf5CircuitRLCDump%read('ode_y', Temp1D)
            call this%ode_rlc%init(this%ode_rlc%neq, this%ode_rlc%dt, c_funloc(RhsFnRLC), ts, Temp1D)

            deallocate(Temp1D)
            call hdf5CircuitRLCDump%close()

        end subroutine LoadCircuitRLCHDF5


        subroutine LoadCircuitRLCDAT(this)
            class(CircuitRLC), intent(inout) :: this
            integer(4) :: size, rank, ierr
            real(8), allocatable :: Temp1D(:)

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The CircuitRLC is load from dat file."

            open(10, file=this%IOName%FullName%str)

                read(10, *) this%ode_rlc%neq
                read(10, *) this%ode_rlc%tcur(1)

                allocate(Temp1D(this%ode_rlc%neq))
                read(10, *) Temp1D
                call this%ode_rlc%init(this%ode_rlc%neq, this%ode_rlc%dt, c_funloc(RhsFnRLC), this%ode_rlc%tcur(1), Temp1D)

                deallocate(Temp1D)

            close(10)

        end subroutine LoadCircuitRLCDAT


        subroutine DestroyCircuitRLC(this)
            class(CircuitRLC), intent(inout) :: this

            if(allocated(rlc_source_amplitude)) deallocate(rlc_source_amplitude)
            if(allocated(rlc_source_frequency)) deallocate(rlc_source_frequency)
            if(allocated(rlc_source_phase)) deallocate(rlc_source_phase)

            call this%ode_rlc%destroy()

        end subroutine DestroyCircuitRLC

end Module ModuleCircuitRLC

