Module ModuleCircuitVoltageSource
    use mpi
    use Constants
    use ModuleFileName
    use ModuleParallelDump

    implicit none

    Type CircuitVoltageSource
        Type(FileName) :: IOName
        integer(4)  :: NFrequency = 0
        real(8), allocatable :: Frequency(:)
        real(8), allocatable :: Amplitude(:)
        real(8), allocatable :: Phase(:)
        real(8) :: Voltage = 0.d0
        real(8) :: Vdc = 0.d0

    contains

        procedure :: Init   => InitilalizationCircuitVoltageSource
        procedure :: Update => UpdateCircuitVoltageSource
        procedure :: Zero   => ZeroCircuitVoltageSource

        procedure :: Dump => DumpCircuitVoltageSource
        procedure :: Load => LoadCircuitVoltageSource

        procedure :: Destroy => DestroyCircuitVoltageSource

        procedure, private :: LoadCircuitVoltageSourceHDF5
        procedure, private :: LoadCircuitVoltageSourceDAT

        procedure, private :: DumpCircuitVoltageSourceHDF5
        procedure, private :: DumpCircuitVoltageSourceDAT

    end Type CircuitVoltageSource

    Type(HDF5_PDump), private :: hdf5CircuitVoltageSourceDump

    contains
    
        subroutine InitilalizationCircuitVoltageSource(VSO, nfreq, pbname)
            Class(CircuitVoltageSource), intent(inout) :: VSO
            integer(4), intent(in) :: nfreq
            character(*), optional, intent(in) :: pbname
            logical :: alive

            if (present(pbname)) then
                call VSO%IOName%Init(pbname, RESTART_FILE_NAME)
            else
                call VSO%IOName%Init("CircuitVoltageSource", RESTART_FILE_NAME)
            end if

            call VSO%Destroy()

            VSO%NFrequency = nfreq
            Allocate(VSO%Amplitude(VSO%NFrequency))
            Allocate(VSO%Frequency(VSO%NFrequency))
            Allocate(VSO%Phase(VSO%NFrequency))

            VSO%Voltage = 0.d0
            ! VSO%Vdc = 0.d0

        end subroutine InitilalizationCircuitVoltageSource


        subroutine UpdateCircuitVoltageSource(VSO, time)
            Class(CircuitVoltageSource), intent(inout) :: VSO
            real(8), intent(in) :: time
            integer(4) :: i

            Associate(Nf => VSO%NFrequency, Vol => VSO%Voltage, Vdc => VSO%Vdc, Amp => VSO%Amplitude, &
                    Fre => VSO%Frequency, Pha => VSO%Phase)

                Vol = 0.d0
                Vol = Vdc + Vol

                do i = 1, Nf
                    Vol = Vol + Amp(i) * DSin(2 * PI * Fre(i) * time + Pha(i) / 180.d0 * PI)
                end do

            end Associate

            return
        end subroutine UpdateCircuitVoltageSource


        subroutine ZeroCircuitVoltageSource(VSO)
            Class(CircuitVoltageSource), intent(inout) :: VSO

            VSO%Voltage = 0.d0

            return
        end subroutine ZeroCircuitVoltageSource


        subroutine DumpCircuitVoltageSource(VSO)
            Class(CircuitVoltageSource), intent(inout) :: VSO

            if (FILE_EXTENSION_MODE_H5 == VSO%IOName%ExtensionMode) then
                call VSO%DumpCircuitVoltageSourceHDF5()

            else if (FILE_EXTENSION_MODE_DAT == VSO%IOName%ExtensionMode) then
                call VSO%DumpCircuitVoltageSourceDAT()

            end if

            return
        endsubroutine DumpCircuitVoltageSource


        subroutine LoadCircuitVoltageSource(VSO)
            Class(CircuitVoltageSource), intent(inout) :: VSO
            logical :: alive

            ! Inquire(file=VSO%IOName%FullName%str, exist=alive)
            ! if (alive) then
            !     if (FILE_EXTENSION_MODE_H5 == VSO%IOName%ExtensionMode) then
            !         call VSO%LoadCircuitVoltageSourceHDF5()

            !     else if (FILE_EXTENSION_MODE_DAT == VSO%IOName%ExtensionMode) then
            !         call VSO%LoadCircuitVoltageSourceDAT()

            !     end if

            ! end if

            return
        end subroutine LoadCircuitVoltageSource


        subroutine DumpCircuitVoltageSourceHDF5(VSO)
            Class(CircuitVoltageSource), intent(inout) :: VSO
            integer(4) :: size, rank, ierr

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The CircuitVoltageSource is save as hdf5 file."

            call hdf5CircuitVoltageSourceDump%init(filename=VSO%IOName%FullName%str, mode='write', serial=.True.)
            call hdf5CircuitVoltageSourceDump%open()

            call hdf5CircuitVoltageSourceDump%writeattr('/', 'NFrequency', VSO%NFrequency)
            call hdf5CircuitVoltageSourceDump%writeattr('/', 'Voltage', VSO%Voltage)
            call hdf5CircuitVoltageSourceDump%writeattr('/', 'Vdc', VSO%Vdc)

            call hdf5CircuitVoltageSourceDump%write('Frequency', VSO%Frequency)
            call hdf5CircuitVoltageSourceDump%write('Amplitude', VSO%Amplitude)
            call hdf5CircuitVoltageSourceDump%write('Phase', VSO%Phase)

            call hdf5CircuitVoltageSourceDump%close()

            return
        end subroutine DumpCircuitVoltageSourceHDF5


        subroutine DumpCircuitVoltageSourceDAT(VSO)
            Class(CircuitVoltageSource), intent(inout) :: VSO
            integer(4) :: size, rank, ierr

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The CircuitVoltageSource is save as dat file."
            open(10, file=VSO%IOName%FullName%str)

                write(10, *) VSO%NFrequency
                write(10, *) VSO%Voltage
                write(10, *) VSO%Vdc

                write(10, *) VSO%Frequency
                write(10, *) VSO%Amplitude
                write(10, *) VSO%Phase

            close(10)

            return
        end subroutine DumpCircuitVoltageSourceDAT


        subroutine LoadCircuitVoltageSourceHDF5(VSO)
            Class(CircuitVoltageSource), intent(inout) :: VSO
            real(8),ALLOCATABLE :: Temp1D(:)
            logical :: alive
            integer(4) :: size, rank, ierr

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The CircuitVoltageSource is load from hdf5 file."

            call hdf5CircuitVoltageSourceDump%init(filename=VSO%IOName%FullName%str, mode='read')
            call hdf5CircuitVoltageSourceDump%open()

            call hdf5CircuitVoltageSourceDump%readattr('/', 'NFrequency', VSO%NFrequency)
            call hdf5CircuitVoltageSourceDump%readattr('/', 'Voltage', VSO%Voltage)
            call hdf5CircuitVoltageSourceDump%readattr('/', 'Vdc', VSO%Vdc)

            call VSO%Destroy()
            Allocate(VSO%Amplitude(VSO%NFrequency))
            Allocate(VSO%Frequency(VSO%NFrequency))
            Allocate(VSO%Phase(VSO%NFrequency))

            Allocate(Temp1D, mold=VSO%Frequency)
            call hdf5CircuitVoltageSourceDump%read('Frequency', Temp1D)
            VSO%Frequency = Temp1D

            call hdf5CircuitVoltageSourceDump%read('Amplitude', Temp1D)
            VSO%Amplitude = Temp1D

            call hdf5CircuitVoltageSourceDump%read('Phase', Temp1D)
            VSO%Phase = Temp1D

            Deallocate(Temp1D)
            call hdf5CircuitVoltageSourceDump%close()

            return
        end subroutine LoadCircuitVoltageSourceHDF5


        subroutine LoadCircuitVoltageSourceDAT(VSO)
            Class(CircuitVoltageSource), intent(inout) :: VSO
            integer(4) :: size, rank, ierr

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The CircuitVoltageSource is load from dat file."

            open(10, file=VSO%IOName%FullName%str)

                read(10, *) VSO%NFrequency
                read(10, *) VSO%Voltage
                read(10, *) VSO%Vdc

                call VSO%Destroy()
                Allocate(VSO%Amplitude(VSO%NFrequency))
                Allocate(VSO%Frequency(VSO%NFrequency))
                Allocate(VSO%Phase(VSO%NFrequency))

                read(10, *) VSO%Frequency
                read(10, *) VSO%Amplitude
                read(10, *) VSO%Phase

            close(10)

            return
        end subroutine LoadCircuitVoltageSourceDAT


        subroutine DestroyCircuitVoltageSource(VSO)
            class(CircuitVoltageSource), intent(inout) :: VSO

            if(Allocated(VSO%Amplitude)) Deallocate(VSO%Amplitude)
            if(Allocated(VSO%Frequency)) Deallocate(VSO%Frequency)
            if(Allocated(VSO%Phase)) Deallocate(VSO%Phase)
            
        end subroutine DestroyCircuitVoltageSource

end Module ModuleCircuitVoltageSource

