Module ModuleCircuitIMN
    use mpi
    use Constants
    use ModuleFileName
    use ModuleParallelDump
    use sundials_ode

    implicit none

    integer(4), save :: imn_source_frequency_num = 0
    real(8), save, allocatable :: imn_source_frequency(:)
    real(8), save, allocatable :: imn_source_amplitude(:)
    real(8), save, allocatable :: imn_source_phase(:)
    real(8), save :: imn_source_voltage = 0.d0
    real(8), save :: imn_electrode_current = 0.d0
    real(8), save :: imn_electrode_voltage = 0.d0
    real(8), save :: imn_Iconv = 0.d0

    real(8), save :: imn_eqc = 0.d0
    real(8), save :: imn_q0 = 0.d0

    real(8), save :: imn_Rs = 0.d0
    real(8), save :: imn_Cm1 = 0.d0
    real(8), save :: imn_Cm2 = 0.d0
    real(8), save :: imn_Lm = 0.d0
    real(8), save :: imn_Rm = 0.d0

    logical, save :: imn_is_open_stray_capacitance = .True.
    real(8), save :: imn_C_stray = 0.d0
    real(8), save :: imn_R_stray = 0.d0
    real(8), save :: imn_L_stray = 0.d0

    type CircuitIMN
        type(FileName) :: IOName
        type(ODESolver) :: ode_imn  ! restart

    contains

        procedure :: Init   => InitilalizationCircuitIMN
        procedure :: Update => UpdateCircuitIMN
        procedure :: Zero   => ZeroCircuitIMN

        procedure :: Dump => DumpCircuitIMN
        procedure :: Load => LoadCircuitIMN

        procedure :: Destroy => DestroyCircuitIMN

        procedure, private :: LoadCircuitIMNHDF5
        procedure, private :: LoadCircuitIMNDAT

        procedure, private :: DumpCircuitIMNHDF5
        procedure, private :: DumpCircuitIMNDAT

    end Type CircuitIMN

    Type(HDF5_PDump), private :: hdf5CircuitIMNDump

    contains

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
            integer :: i

            !======= Internals ============
            ! get data arrays from SUNDIALS vectors
            yvec => FN_VGetArrayPointer(sunvec_y)
            fvec => FN_VGetArrayPointer(sunvec_f)

            associate(Nf => imn_source_frequency_num, Vol => imn_source_voltage, Amp => imn_source_amplitude, &
                    Fre => imn_source_frequency, Pha => imn_source_phase)

                Vol = 0.d0
                do i = 1, Nf
                    Vol = Vol + Amp(i) * DSin(2 * PI * Fre(i) * tn + Pha(i) / 180.d0 * PI)
                end do

            end associate

            fvec(1) = (imn_source_voltage - (yvec(1)-yvec(2))/imn_Cm1) / imn_Rs
            fvec(2) = yvec(3)
            fvec(3) = ((yvec(1)-yvec(2))/imn_Cm1 - yvec(2)/imn_Cm2 - yvec(3)*imn_Rm - (yvec(6) - imn_q0)/imn_eqc) / imn_Lm

            if (imn_is_open_stray_capacitance) then
                fvec(4) = yvec(3) - yvec(5)
                fvec(5) = ((yvec(6) - imn_q0)/imn_eqc - (yvec(2)-yvec(4))/imn_C_stray - yvec(5)*imn_R_stray) / imn_L_stray

            else
                fvec(4) = 0.d0
                fvec(5) = 0.d0

            end if

            imn_electrode_current = yvec(3) - yvec(5)
            fvec(6) = imn_electrode_current + imn_Iconv

            ierr2 = 0

        end function RhsFnIMN


        subroutine InitilalizationCircuitIMN(this, nfreq, circuit_name)
            class(CircuitIMN), intent(inout) :: this
            integer(4), intent(in) :: nfreq
            character(*), optional, intent(in) :: circuit_name
            logical :: alive

            if (present(circuit_name)) then
                call this%IOName%Init(circuit_name, RESTART_FILE_NAME)
            else
                call this%IOName%Init("CircuitIMN", RESTART_FILE_NAME)
            end if

            call this%Destroy()

            imn_source_frequency_num = nfreq
            allocate(imn_source_frequency(imn_source_frequency_num))
            allocate(imn_source_amplitude(imn_source_frequency_num))
            allocate(imn_source_phase(imn_source_frequency_num))

        end subroutine InitilalizationCircuitIMN


        subroutine UpdateCircuitIMN(this, time, electrode_voltage_boundary, Iconv, eqc, q0)
            class(CircuitIMN), intent(inout) :: this
            real(8), intent(in) :: time, electrode_voltage_boundary, Iconv, eqc, q0
            integer(4) :: i

            imn_electrode_voltage = electrode_voltage_boundary
            imn_Iconv = Iconv
            imn_eqc = eqc
            imn_q0 = q0

            call this%ode_imn%update()

            associate(Nf => imn_source_frequency_num, Vol => imn_source_voltage, Amp => imn_source_amplitude, &
                      Fre => imn_source_frequency, Pha => imn_source_phase)

                Vol = 0.d0
                do i = 1, Nf
                    Vol = Vol + Amp(i) * DSin(2 * PI * Fre(i) * time + Pha(i) / 180.d0 * PI)
                end do

            end associate

        end subroutine UpdateCircuitIMN


        subroutine ZeroCircuitIMN(this)
            class(CircuitIMN), intent(inout) :: this

            this%ode_imn%y = 0.d0
            this%ode_imn%out = 0.d0

        end subroutine ZeroCircuitIMN


        subroutine DumpCircuitIMN(this)
            class(CircuitIMN), intent(inout) :: this

            if (FILE_EXTENSION_MODE_H5 == this%IOName%ExtensionMode) then
                call this%DumpCircuitIMNHDF5()

            else if (FILE_EXTENSION_MODE_DAT == this%IOName%ExtensionMode) then
                call this%DumpCircuitIMNDAT()

            end if

        endsubroutine DumpCircuitIMN


        subroutine LoadCircuitIMN(this)
            class(CircuitIMN), intent(inout) :: this
            logical :: alive

            Inquire(file=this%IOName%FullName%str, exist=alive)
            if (alive) then
                if (FILE_EXTENSION_MODE_H5 == this%IOName%ExtensionMode) then
                    call this%LoadCircuitIMNHDF5()

                else if (FILE_EXTENSION_MODE_DAT == this%IOName%ExtensionMode) then
                    call this%LoadCircuitIMNDAT()

                end if

            end if

        end subroutine LoadCircuitIMN


        subroutine DumpCircuitIMNHDF5(this)
            class(CircuitIMN), intent(inout) :: this
            integer(4) :: size, rank, ierr, neq

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The CircuitIMN is save as hdf5 file."

            call hdf5CircuitIMNDump%init(filename=this%IOName%FullName%str, mode='write', serial=.True.)
            call hdf5CircuitIMNDump%open()

            neq = this%ode_imn%neq
            call hdf5CircuitIMNDump%writeattr('/', 'ode_neq', neq)
            call hdf5CircuitIMNDump%writeattr('/', 'ode_tcur', this%ode_imn%tcur(1))

            call hdf5CircuitIMNDump%write('ode_y', this%ode_imn%y)

            call hdf5CircuitIMNDump%close()

        end subroutine DumpCircuitIMNHDF5


        subroutine DumpCircuitIMNDAT(this)
            class(CircuitIMN), intent(inout) :: this
            integer(4) :: size, rank, ierr

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The CircuitIMN is save as dat file."
            open(10, file=this%IOName%FullName%str)

                write(10, *) this%ode_imn%neq
                write(10, *) this%ode_imn%tcur(1)

                write(10, *) this%ode_imn%y

            close(10)

        end subroutine DumpCircuitIMNDAT


        subroutine LoadCircuitIMNHDF5(this)
            class(CircuitIMN), intent(inout) :: this
            real(8), allocatable :: Temp1D(:)
            logical :: alive
            integer(4) :: size, rank, ierr, neq
            real(8) :: ts

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The CircuitIMN is load from hdf5 file."

            call hdf5CircuitIMNDump%init(filename=this%IOName%FullName%str, mode='read')
            call hdf5CircuitIMNDump%open()

            call hdf5CircuitIMNDump%readattr('/', 'ode_neq', neq)
            this%ode_imn%neq = neq
            call hdf5CircuitIMNDump%readattr('/', 'ode_tcur', ts)

            allocate(Temp1D(this%ode_imn%neq))
            call hdf5CircuitIMNDump%read('ode_y', Temp1D)
            call this%ode_imn%init(this%ode_imn%neq, this%ode_imn%dt, c_funloc(RhsFnIMN), ts, Temp1D)

            deallocate(Temp1D)
            call hdf5CircuitIMNDump%close()

        end subroutine LoadCircuitIMNHDF5


        subroutine LoadCircuitIMNDAT(this)
            class(CircuitIMN), intent(inout) :: this
            integer(4) :: size, rank, ierr
            real(8), allocatable :: Temp1D(:)

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The CircuitIMN is load from dat file."

            open(10, file=this%IOName%FullName%str)

                read(10, *) this%ode_imn%neq
                read(10, *) this%ode_imn%tcur(1)

                allocate(Temp1D(this%ode_imn%neq))
                read(10, *) Temp1D
                call this%ode_imn%init(this%ode_imn%neq, this%ode_imn%dt, c_funloc(RhsFnIMN), this%ode_imn%tcur(1), Temp1D)

                deallocate(Temp1D)

            close(10)

        end subroutine LoadCircuitIMNDAT


        subroutine DestroyCircuitIMN(this)
            class(CircuitIMN), intent(inout) :: this

            if(allocated(imn_source_amplitude)) deallocate(imn_source_amplitude)
            if(allocated(imn_source_frequency)) deallocate(imn_source_frequency)
            if(allocated(imn_source_phase)) deallocate(imn_source_phase)

            call this%ode_imn%destroy()

        end subroutine DestroyCircuitIMN

end Module ModuleCircuitIMN
