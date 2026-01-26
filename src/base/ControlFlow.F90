module ModuleControlFlow
    use mpi
    use Constants
    use ModuleFileName
    use ModuleDomain

    implicit none

    integer(4), parameter :: load_type_vacuum = 0
    integer(4), parameter :: load_type_plasma = 1
    integer(4), save :: load_type = load_type_plasma

    integer(4), parameter :: pic_type_explicit = 0
    integer(4), parameter :: pic_type_implicit = 1
    integer(4), save :: pic_type = pic_type_implicit

    logical, save :: is_restart = .False.
    integer(4), save :: diag_type_global = 0
    integer(4), save :: diag_cycles = 0
    integer(4), save :: diag_Nt = 0
    integer(4), save :: diag_preiod = 0
    real(8), save :: diag_freq = 0.d0

    Type ControlFlow
        Type(Domain)   :: DM
        Type(FileName) :: IOName
        integer(4)  :: NRun = 0, NDiagShort = 0, NDiagLong = 0
        integer(4)  :: Ns = 0, Ng = 0           ! restart

        integer(4)  :: Timer = 0, Period = 0    ! restart
        real(8)     :: dt = 0.d0

    contains

        procedure :: Init => InitControlFlow
        procedure :: Dump => DumpControlFlow
		procedure :: Load => LoadControlFlow
        procedure :: Reset => ResetControlFlow
        procedure :: Destroy => DestroyControlFlow
        procedure :: Show => ShowControlFlow

        procedure, private :: LoadControlFlowHDF5
        procedure, private :: LoadControlFlowDAT

        procedure, private :: DumpControlFlowHDF5
        procedure, private :: DumpControlFlowDAT

    end Type ControlFlow

    Type(HDF5_PDump),private :: hdf5ControlFlowDump

contains

    subroutine InitControlFlow(CF, cfname)
        Class(ControlFlow), intent(inout) :: CF
        character(*), optional, intent(in) :: cfname

        if (present(cfname)) then
            call CF%IOName%Init(cfname, RESTART_FILE_NAME)
        else
            call CF%IOName%Init("ControlFlow", RESTART_FILE_NAME)
        end if

    end subroutine InitControlFlow


    subroutine DumpControlFlow(CF)
        Class(ControlFlow),intent(inout) :: CF

        if (FILE_EXTENSION_MODE_H5 == CF%IOName%ExtensionMode) then
            call CF%DumpControlFlowHDF5()
        
        else if (FILE_EXTENSION_MODE_DAT == CF%IOName%ExtensionMode) then
            call CF%DumpControlFlowDAT()

        end if

    endsubroutine DumpControlFlow


    subroutine LoadControlFlow(CF)
        Class(ControlFlow),intent(inout) :: CF
        logical :: alive

        Inquire(file=CF%IOName%FullName%str, exist=alive)
        if (alive) then
            if (FILE_EXTENSION_MODE_H5 == CF%IOName%ExtensionMode) then
                call CF%LoadControlFlowHDF5()

            else if (FILE_EXTENSION_MODE_DAT == CF%IOName%ExtensionMode) then
                call CF%LoadControlFlowDAT()

            end if

        end if

    end subroutine LoadControlFlow


    subroutine DumpControlFlowHDF5(CF)
        Class(ControlFlow), intent(inout) :: CF

        if (0 == CF%DM%MyId) write(*, '(a)') "The ControlFlow is save as hdf5 file."

        call hdf5ControlFlowDump%init(filename=CF%IOName%FullName%str, mode='write', serial=.True.)
        call hdf5ControlFlowDump%open()

        call hdf5ControlFlowDump%writeattr('/','NDiagShort',CF%NDiagShort)
        call hdf5ControlFlowDump%writeattr('/','NDiagLong',CF%NDiagLong)
        call hdf5ControlFlowDump%writeattr('/','Ns',CF%Ns)
        call hdf5ControlFlowDump%writeattr('/','Ng',CF%Ng)
        call hdf5ControlFlowDump%writeattr('/','Timer',CF%Timer)
        call hdf5ControlFlowDump%writeattr('/','Period',CF%Period)
        call hdf5ControlFlowDump%writeattr('/','dt',CF%dt)

        call hdf5ControlFlowDump%close()

    end subroutine DumpControlFlowHDF5


    subroutine DumpControlFlowDAT(CF)
        Class(ControlFlow), intent(inout) :: CF

        if (0 == CF%DM%MyId) write(*, '(a)') "The controlflow is save as dat file."
        open(10, file=CF%IOName%FullName%str)

            write(10, *) CF%NDiagShort
            write(10, *) CF%NDiagLong
            write(10, *) CF%Ns
            write(10, *) CF%Ng

            write(10, *) CF%Timer
            write(10, *) CF%Period
            write(10, *) CF%dt

        close(10)

    end subroutine DumpControlFlowDAT


    subroutine LoadControlFlowHDF5(CF)
        Class(ControlFlow), intent(inout) :: CF
        integer(4) :: size, rank, ierr
        integer(4) :: tmp_int
        real(8) :: tmp_real

        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

        if (0 == rank) write(*, '(a)') "The ControlFlow is load from hdf5 file."

        call hdf5ControlFlowDump%init(filename=CF%IOName%FullName%str, mode='read')
        call hdf5ControlFlowDump%open()

        call hdf5ControlFlowDump%readattr('/','NDiagShort', tmp_int)
        call hdf5ControlFlowDump%readattr('/','NDiagLong', tmp_int)
        call hdf5ControlFlowDump%readattr('/','Ns', CF%Ns)
        call hdf5ControlFlowDump%readattr('/','Ng', CF%Ng)
        call hdf5ControlFlowDump%readattr('/','Timer', CF%Timer)
        call hdf5ControlFlowDump%readattr('/','Period', CF%Period)
        call hdf5ControlFlowDump%readattr('/','dt', tmp_real)

        call hdf5ControlFlowDump%close()

    end subroutine LoadControlFlowHDF5


    subroutine LoadControlFlowDAT(CF)
        Class(ControlFlow), intent(inout) :: CF
        integer(4) :: size, rank, ierr
        integer(4) :: tmp_int
        real(8) :: tmp_real

        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

        if (0 == rank) write(*, '(a)') "The controlflow is load from dat file."
        open(10, file=CF%IOName%FullName%str)

            read(10, *) tmp_int
            read(10, *) tmp_int
            read(10, *) CF%Ns
            read(10, *) CF%Ng

            read(10, *) CF%Timer
            read(10, *) CF%Period
            read(10, *) tmp_real

        close(10)

    end subroutine LoadControlFlowDAT


    subroutine ResetControlFlow(CF)
        Class(ControlFlow), intent(inout) :: CF

        CF%NRun = 0
        CF%NDiagShort = 0
        CF%NDiagLong = 0
        CF%Ns = 0
        CF%Ng = 0

        CF%Timer = 0
        CF%Period = 0
        CF%dt = 0.d0

    end subroutine ResetControlFlow

    subroutine DestroyControlFlow(CF)
        Class(ControlFlow), intent(inout) :: CF

        call CF%DM%Destroy()

        return
    end subroutine DestroyControlFlow


    subroutine ShowControlFlow(CF)
        Class(ControlFlow), intent(inout) :: CF

        if (0 == CF%DM%MyId) then
            write(*, '(a30, i)') 'run cycles:', CF%NRun
            write(*, '(a30, i)') 'run timer:', CF%Timer
            write(*, '(a30, i)') 'run period:', CF%Period
            write(*, '(a30, es12.4)') 'time step:', CF%dt
            write(*, *) ''
        end if

    end subroutine ShowControlFlow

end module ModuleControlFlow
