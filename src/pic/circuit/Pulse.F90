Module ModuleCircuitPulseSource
    use mpi
    use Constants
    use ModuleFileName
    use ModuleParallelDump

    implicit none

    Type CircuitPulseSource
        Type(FileName) :: IOName
        integer(4)  :: seg = 0
        real(8), allocatable :: time(:)
        real(8), allocatable :: amplitude(:)
        real(8) :: voltage = 0.d0

    contains

        procedure :: Init   => InitilalizationCircuitPulseSource
        procedure :: Update => UpdateCircuitPulseSource
        procedure :: Zero   => ZeroCircuitPulseSource

        procedure :: Dump => DumpCircuitPulseSource
        procedure :: Load => LoadCircuitPulseSource

        procedure :: Destroy => DestroyCircuitPulseSource

        procedure, private :: LoadCircuitPulseSourceHDF5
        procedure, private :: LoadCircuitPulseSourceDAT

        procedure, private :: DumpCircuitPulseSourceHDF5
        procedure, private :: DumpCircuitPulseSourceDAT

    end Type CircuitPulseSource

    Type(HDF5_PDump), private :: hdf5CircuitPulseSourceDump

    contains
    
        subroutine InitilalizationCircuitPulseSource(this, seg, name)
            Class(CircuitPulseSource), intent(inout) :: this
            integer(4), intent(in) :: seg
            character(*), optional, intent(in) :: name
            logical :: alive
            integer(4) :: size, rank, ierr

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (present(name)) then
                call this%IOName%Init(name, RESTART_FILE_NAME)
            else
                call this%IOName%Init("CircuitPulseSource", RESTART_FILE_NAME)
            end if

            call this%Destroy()

            if (seg > 1) then
                this%seg = seg
                Allocate(this%time(this%seg))
                Allocate(this%amplitude(this%seg))

                this%Voltage = 0.d0

            else
                if (0 == rank) write(*, *) "The input parameter is invalid. seg should be greater than 0."                

            end if

            return
        end subroutine InitilalizationCircuitPulseSource


        subroutine UpdateCircuitPulseSource(this, time)
            Class(CircuitPulseSource), intent(inout) :: this
            real(8), intent(in) :: time
            integer(4) :: i

            this%voltage = 0.d0
            do i = 1, this%seg-1
                if (time >= this%time(i) .and. time < this%time(i+1)) then
                    this%voltage = this%amplitude(i) + &
                        (time-this%time(i)) / (this%time(i+1)-this%time(i)) * (this%amplitude(i+1)-this%amplitude(i))
                end if
            end do

            return
        end subroutine UpdateCircuitPulseSource


        subroutine ZeroCircuitPulseSource(this)
            Class(CircuitPulseSource), intent(inout) :: this

            this%voltage = 0.d0

            return
        end subroutine ZeroCircuitPulseSource


        subroutine DumpCircuitPulseSource(this)
            Class(CircuitPulseSource), intent(inout) :: this

            if (FILE_EXTENSION_MODE_H5 == this%IOName%ExtensionMode) then
                call this%DumpCircuitPulseSourceHDF5()

            else if (FILE_EXTENSION_MODE_DAT == this%IOName%ExtensionMode) then
                call this%DumpCircuitPulseSourceDAT()

            end if

            return
        endsubroutine DumpCircuitPulseSource


        subroutine LoadCircuitPulseSource(this)
            Class(CircuitPulseSource), intent(inout) :: this
            logical :: alive

            ! Inquire(file=this%IOName%FullName%str, exist=alive)
            ! if (alive) then
            !     if (FILE_EXTENSION_MODE_H5 == this%IOName%ExtensionMode) then
            !         call this%LoadCircuitPulseSourceHDF5()

            !     else if (FILE_EXTENSION_MODE_DAT == this%IOName%ExtensionMode) then
            !         call this%LoadCircuitPulseSourceDAT()

            !     end if

            ! end if

            return
        end subroutine LoadCircuitPulseSource


        subroutine DumpCircuitPulseSourceHDF5(this)
            Class(CircuitPulseSource), intent(inout) :: this
            integer(4) :: size, rank, ierr

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The CircuitPulseSource is save as hdf5 file."

            call hdf5CircuitPulseSourceDump%init(filename=this%IOName%FullName%str, mode='write', serial=.True.)
            call hdf5CircuitPulseSourceDump%open()

            call hdf5CircuitPulseSourceDump%writeattr('/', 'seg', this%seg)
            call hdf5CircuitPulseSourceDump%writeattr('/', 'voltage', this%voltage)

            call hdf5CircuitPulseSourceDump%write('time', this%time)
            call hdf5CircuitPulseSourceDump%write('amplitude', this%amplitude)

            call hdf5CircuitPulseSourceDump%close()

            return
        end subroutine DumpCircuitPulseSourceHDF5


        subroutine DumpCircuitPulseSourceDAT(this)
            Class(CircuitPulseSource), intent(inout) :: this
            integer(4) :: size, rank, ierr

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The CircuitPulseSource is save as dat file."
            open(10, file=this%IOName%FullName%str)

                write(10, *) this%seg
                write(10, *) this%voltage

                write(10, *) this%time
                write(10, *) this%amplitude

            close(10)

            return
        end subroutine DumpCircuitPulseSourceDAT


        subroutine LoadCircuitPulseSourceHDF5(this)
            Class(CircuitPulseSource), intent(inout) :: this
            real(8),ALLOCATABLE :: Temp1D(:)
            logical :: alive
            integer(4) :: size, rank, ierr

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The CircuitPulseSource is load from hdf5 file."

            call hdf5CircuitPulseSourceDump%init(filename=this%IOName%FullName%str, mode='read')
            call hdf5CircuitPulseSourceDump%open()

            call hdf5CircuitPulseSourceDump%readattr('/', 'seg', this%seg)
            call hdf5CircuitPulseSourceDump%readattr('/', 'voltage', this%voltage)

            call this%Destroy()
            Allocate(this%time(this%seg))
            Allocate(this%amplitude(this%seg))

            Allocate(Temp1D, mold=this%time)
            call hdf5CircuitPulseSourceDump%read('time', Temp1D)
            this%time = Temp1D

            call hdf5CircuitPulseSourceDump%read('amplitude', Temp1D)
            this%amplitude = Temp1D

            Deallocate(Temp1D)
            call hdf5CircuitPulseSourceDump%close()

            return
        end subroutine LoadCircuitPulseSourceHDF5


        subroutine LoadCircuitPulseSourceDAT(this)
            Class(CircuitPulseSource), intent(inout) :: this
            integer(4) :: size, rank, ierr

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The CircuitPulseSource is load from dat file."

            open(10, file=this%IOName%FullName%str)

                read(10, *) this%seg
                read(10, *) this%voltage

                call this%Destroy()
                Allocate(this%time(this%seg))
                Allocate(this%amplitude(this%seg))

                read(10, *) this%time
                read(10, *) this%amplitude

            close(10)

            return
        end subroutine LoadCircuitPulseSourceDAT


        subroutine DestroyCircuitPulseSource(this)
            class(CircuitPulseSource), intent(inout) :: this

            if(Allocated(this%time)) Deallocate(this%time)
            if(Allocated(this%amplitude)) Deallocate(this%amplitude)
            
        end subroutine DestroyCircuitPulseSource

end Module ModuleCircuitPulseSource

