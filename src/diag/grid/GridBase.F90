Module ModuleGridBase
    ! USE Constants     !等待解耦
    Use ModuleFileName  !等待解耦
    use ModuleTempCalculate
    use ModuleParallelDump
    !Use ConstantsNumerical
    !Use Numrical
    Implicit none

    !数据空间维度
    integer(4), parameter :: GRID_SPACE_DIMENSION_0D = 0
    integer(4), parameter :: GRID_SPACE_DIMENSION_1D = 1
    integer(4), parameter :: GRID_SPACE_DIMENSION_2D = 2
    integer(4), parameter :: GRID_SPACE_DIMENSION_3D = 3

    !数据时间维度
    integer(4), parameter :: GRID_TIME_DEPENDENT    = 1    !含时
    integer(4), parameter :: GRID_TIME_INDEPENDENT  = 0    !不含时

    !数据总维度
    integer(4), parameter :: GRID_DATA_DIMENSION_0D = 0
    integer(4), parameter :: GRID_DATA_DIMENSION_1D = 1
    integer(4), parameter :: GRID_DATA_DIMENSION_2D = 2
    integer(4), parameter :: GRID_DATA_DIMENSION_3D = 3
    integer(4), parameter :: GRID_DATA_DIMENSION_4D = 4

    !模拟类型
    integer(4), parameter :: SIM_MODE_STABLE        = 100   !稳态
    integer(4), parameter :: SIM_MODE_EVOLUTION     = 200   !演化

    ! 数据流转类型
    integer(4), parameter :: INPUT_TO_STACK = 1
    integer(4), parameter :: STACK_TO_FINAL = 0

    ! 数据平均类型
    integer(4), parameter :: VAL_VALUESTACK = 11
    integer(4), parameter :: VAL_VALUEFINAL = 12
    integer(4), parameter :: ADD_VALUESTACK = 21
    integer(4), parameter :: ADD_VALUEFINAL = 22

    ! 存盘类型
    ! integer(4), parameter :: DUMP_FILEMODE_HDF5     = 11
    ! integer(4), parameter :: DUMP_FILEMODE_DAT      = 12

    ! integer(4), parameter :: DUMP_NAMEMODE_STATIC   = 21
    ! integer(4), parameter :: DUMP_NAMEMODE_DYNAMIC  = 22

    integer(4), parameter :: DUMP_VALUEMODE_STACK   = 31
    integer(4), parameter :: DUMP_VALUEMODE_FINAL   = 32

    type :: GridValueInterpolation
        integer(4) :: NtRaw = -1, Nt = -1
        integer(4), allocatable :: ValueLocInt(:,:)
        real(8),    allocatable :: ValueLoc(:), ValueWeight(:,:), ValueAllWeight(:)

    contains
        procedure :: init => InitGridValueInterpolation
        procedure :: calc => calcGridValueInterpolation
        procedure :: destroy => destroyGridValueInterpolation
    end type GridValueInterpolation

    ! Integer,parameter,Private :: MaxDim=5
    Type,Abstract :: GridBase
        Type(FileName) :: IOName
        type(GridValueInterpolation) :: GVIFinal, GVIStack
        Integer(4)  :: TimeMode = -1
        !Integer(4) :: TimerStart=1,TimerEnd
        ! Integer(4)  :: Ns = 1, NsIndex = 1, NsLoop = 0, NsLoopMax = 1
        ! Integer(4)  :: Nt = 1, NtIndex = 1, NtLoop = 0, NtLoopMax = 1
            !       │ NtInner │         │          │          │
            !    ───┴──▲──────┴─────────┴──────────┴──────────┴──
            !          │           NtOuter
            !         Nt

        ! 原始：NsIndex步进 -> 抵达Ns -> NsLoop步进(无效)、NtIndex步进 -> 抵达Nt
        ! 更新：NsIndex步进 -> 抵达Ns(NsLoop+1、重置NsIndex) -> NtInnerIndex步进 -> 抵达NtInnerMax(NtInnerLoop+1、重置NtInnerIndex)
                        !  -> NtIndexOuter步进 -> 抵达NtOuterMax(完成)
        ! NtInnerIndex和NsLoop的区别：NtInnerIndex用于指示光标，Ns循环开始前+1；NsLoop用于平均，Ns循环结束才+1
        ! NtOuterIndex和NtInnerLoop的区别：NtIndexInner用于指示光标，NtInnerIndex循环开始前+1；NtInnerLoop用于平均，Ns循环结束才+1
        Integer(4)  :: Nt = 1  ! 仅用于初始化NtInnerMax和NtOuterMax
        Integer(4)  :: Ns = 1, NsIndex = 1, NsLoop = 0
        Integer(4)  :: NtInnerIndex = 1, NtInnerMax = 1, NtInnerLoop = 0
        Integer(4)  :: NtOuterIndex = 1, NtOuterMax = 1, NtOuterLoop = 0!无效，仅用于调函数

        integer(4)  :: Nx1, Nx2, Nx3        = 0
        real(8)     :: Dx1, Dx2, Dx3, Dt    = 0.d0
        character(len=99), allocatable :: VarName(:)
    contains
        !procedure InitIndex=>InitIndexGridBase
        procedure Init=>InitGridBase

        procedure ResetNs=>ResetNsGridBase
        procedure StepNs=>SteppingNsGridBase

        procedure ResetNt=>ResetNtGridBase
        procedure StepNtInner=>SteppingNtInnerGridBase
        procedure StepNtOuter=>SteppingNtOuterGridBase

        procedure(InitArrayGrid),deferred :: InitArray
        procedure(DumpGrid),deferred :: Dump
        procedure(ResetGrid),deferred :: Reset
        !procedure(RescaleNsGrid), deferred :: RescaleNs
        procedure(RescaleGrid),deferred :: Rescale
        !procedure(RescaleNtGrid), deferred :: RescaleNt
    EndType GridBase

    Abstract Interface

        Subroutine InitArrayGrid(GD)
            import GridBase
            Class(GridBase),intent(inout)  :: GD
        Endsubroutine

        subroutine DumpGrid(GD, ValueMode)
            import GridBase
            Class(GridBase),intent(inout)  :: GD
            Integer(4),intent(in) :: ValueMode
        Endsubroutine

        subroutine ResetGrid(GD)
            import GridBase
            class(GridBase),intent(inout) :: GD
        Endsubroutine

        subroutine RescaleGrid(GD, RescaleMode)
            import GridBase
            class(GridBase),intent(inout) :: GD
            Integer(4),intent(in) :: RescaleMode
        Endsubroutine

        !subroutine RescaleNtGrid(GD)
        !   import GridBase
        !   class(GridBase), intent(inout) :: GD
        !End subroutine
    EndInterface
    integer,save :: NtOuterStep = -1, NtInnerStep = -1
    integer,save :: LoopStart = 0
contains
    subroutine GridSetup(NtInner,NtOuter,LoopStartIn)
        integer(4), intent(in) :: NtInner, NtOuter
        integer(4), optional, intent(in) :: LoopStartIn
        NtInnerStep = NtInner
        NtOuterStep = NtOuter
        LoopStart = LoopStartIn
    end subroutine GridSetup

    Subroutine InitGridBase(GD, TimeMode, Nx1, Nx2, Nx3, Nt, Ns, Dx1, Dx2, Dx3, Dt)
        !import ControlFlow
        Class(GridBase), intent(inout)   :: GD
        Integer(4), intent(in)           :: TimeMode
        integer(4), intent(in), optional :: Nx1, Nx2, Nx3, Nt
        real(8),    intent(in), optional :: Dx1, Dx2, Dx3, Dt
        integer(4), intent(in) :: Ns
        ! call GD%IOName%Init('Grid', ParallelMode=FILE_PARALLEL_MODE_OPEN)
        GD%TimeMode=TimeMode
        ! If(present(UpdateMode)) Then
        !     GD%UpdateMode=UpdateMode
        ! Else
        !     GD%UpdateMode=0
        ! EndIf
        ! 临时使用，in和out全是Nt
        GD%NtOuterMax  = NtOuterStep
        GD%NtInnerMax  = NtInnerStep
        if(GD%NtOuterMax == -1 .or. GD%NtInnerMax == -1) then
            error stop "===== Error: NtInnerMax and NtOuterMax are not given! ====="
        end if
        If(Present(Nt)) then
            write(*,*) "exit Nt"
            GD%Nt  = Nt
            call GD%GVIStack%init(Nt=GD%Nt, NtRaw=GD%NtInnerMax)
            call GD%GVIFinal%init(Nt=GD%Nt, NtRaw=GD%NtOuterMax)
            call GD%GVIStack%calc
            call GD%GVIFinal%calc
        end if
        GD%Ns  = Ns
        If(Present(Dt))     GD%Dt  = Dt
        If(Present(Nx1))    GD%Nx1 = Nx1
        If(Present(Nx2))    GD%Nx2 = Nx2
        If(Present(Nx3))    GD%Nx3 = Nx3
        If(Present(Dx1))    GD%Dx1 = Dx1
        If(Present(Dx1))    GD%Dx1 = Dx1
        If(Present(Dx1))    GD%Dx1 = Dx1
        Return
    Endsubroutine InitGridBase

    subroutine ResetNsGridBase(GD)
        !import ControlFlow
        Class(GridBase),intent(inout) :: GD
        GD%NsIndex=1
        GD%NsLoop=0
    Endsubroutine ResetNsGridBase

    Subroutine SteppingNsGridBase(GD)
        Class(GridBase),intent(inout) :: GD
        Call SteppingIndex(GD%Ns,GD%NsIndex,GD%NsLoop)
        Return
    Endsubroutine SteppingNsGridBase

    subroutine ResetNtGridBase(GD)
        !import ControlFlow
        Class(GridBase),intent(inout) :: GD
        ! GD%NtIndex=1
        ! GD%NtLoop=0

        GD%NtInnerIndex = 1
        GD%NtInnerLoop  = 0

        ! GD%NtOuterIndex = 1
        ! GD%NtOuterLoop  = 0!无效，仅用于调函数
    Endsubroutine ResetNtGridBase

    ! Subroutine SteppingNtGridBase(GD)
    !     Class(GridBase),intent(inout) :: GD
    !     Call SteppingIndex(GD%Nt,GD%NtIndex,GD%NtLoop)
    !     Return
    ! Endsubroutine SteppingNtGridBase

    Subroutine SteppingNtInnerGridBase(GD)
        Class(GridBase),intent(inout) :: GD
        Call SteppingIndex(GD%NtInnerMax,GD%NtInnerIndex,GD%NtInnerLoop)
        Return
    Endsubroutine SteppingNtInnerGridBase

    Subroutine SteppingNtOuterGridBase(GD)
        Class(GridBase),intent(inout) :: GD
        Call SteppingIndex(GD%NtOuterMax,GD%NtOuterIndex,GD%NtOuterLoop)
        Return
    Endsubroutine SteppingNtOuterGridBase

    Subroutine SteppingIndex(N,NIndex,NLoop)
        Integer(4),intent(inout) :: N,NIndex,NLoop

        NIndex=NIndex+1
        If(NIndex==N+1) Then
            NLoop=NLoop+1
            NIndex=1
        ENDIf
        Return
    Endsubroutine SteppingIndex

    subroutine InitGridValueInterpolation(GVI, Nt, NtRaw)
        class(GridValueInterpolation),intent(inout) :: GVI
        integer(4), intent(in) :: Nt, NtRaw
        GVI%Nt = Nt
        GVI%NtRaw = NtRaw
        call GVI%destroy()
        allocate(GVI%ValueLoc(1:GVI%NtRaw))
        allocate(GVI%ValueLocInt(1:GVI%NtRaw,2))
        allocate(GVI%ValueWeight(1:GVI%NtRaw,2))
        allocate(GVI%ValueAllWeight(0:GVI%Nt))  ! 因为用了ceiling所以必须向左留出一格余量防止越界
        GVI%ValueLoc=-1.d0
        GVI%ValueLocInt=-1.d0
        GVI%ValueWeight=0.d0
        GVI%ValueAllWeight=0.d0
    end subroutine InitGridValueInterpolation

    subroutine CalcGridValueInterpolation(GVI)
        class(GridValueInterpolation),intent(inout) :: GVI
        integer :: i
        if(GVI%NtRaw > GVI%Nt) then
            !            Nt(j)         S2        NtRaw(i)     S1     Nt(j+1)
            !             #_________________________|___________________#
            !          LocInt(i,1)               Loc(i)            LocInt(i,2)
            !  Allweight(LocInt(i,1))+S1                    Allweight(LocInt(i,2))+S2
            do i = 1,GVI%NtRaw
                GVI%ValueLoc(i)=dble(i-1)*dble((GVI%Nt-1))/dble((GVI%NtRaw-1))+1
            end do
            do i = 1,GVI%NtRaw
                GVI%ValueLocInt(i,2)=Ceiling(GVI%ValueLoc(i))
                GVI%ValueLocInt(i,1)=Ceiling(GVI%ValueLoc(i))-1

                GVI%ValueWeight(i,1)=Dble(GVI%ValueLocInt(i,2))-GVI%ValueLoc(i) !S1
                GVI%ValueWeight(i,2)=1.d0-GVI%ValueWeight(i,1)                  !S2

                GVI%ValueAllWeight(GVI%ValueLocInt(i,1))=GVI%ValueAllWeight(GVI%ValueLocInt(i,1))+GVI%ValueWeight(i,1)
                GVI%ValueAllWeight(GVI%ValueLocInt(i,2))=GVI%ValueAllWeight(GVI%ValueLocInt(i,2))+GVI%ValueWeight(i,2)
            end do

            ! 将第一个时间步的插值格点右移一格, 避免调用时触碰到0下标, 同时反相权重, 确保外界对数值加权正确. 上面已经算好的总权重不会变
            GVI%ValueLocInt(1,1) = 1
            GVI%ValueLocInt(1,2) = 2
            GVI%ValueWeight(1,1) = 1.d0
            GVI%ValueWeight(1,2) = 0.d0
        else 
            do i = 1,GVI%NtRaw
                GVI%ValueLoc(i)=dble(i)
            end do
            do i = 1,GVI%NtRaw
                GVI%ValueLocInt(i,2)=i
                GVI%ValueLocInt(i,1)=i-1

                GVI%ValueWeight(i,1)=0.d0 !S1
                GVI%ValueWeight(i,2)=1.d0 !S2

                GVI%ValueAllWeight(GVI%ValueLocInt(i,1))=GVI%ValueAllWeight(GVI%ValueLocInt(i,1))+0.d0
                GVI%ValueAllWeight(GVI%ValueLocInt(i,2))=GVI%ValueAllWeight(GVI%ValueLocInt(i,2))+1.d0
            end do

            ! 将第一个时间步的插值格点右移一格, 避免调用时触碰到0下标, 同时反相权重, 确保外界对数值加权正确. 上面已经算好的总权重不会变
            GVI%ValueLocInt(1,1) = 1
            GVI%ValueLocInt(1,2) = 2
            GVI%ValueWeight(1,1) = 1.d0
            GVI%ValueWeight(1,2) = 0.d0
        end if
    end subroutine CalcGridValueInterpolation

    subroutine destroyGridValueInterpolation(GVI)
        class(GridValueInterpolation),intent(inout) :: GVI
        if(allocated(GVI%ValueLoc))         deallocate(GVI%ValueLoc)
        if(allocated(GVI%ValueLocInt))      deallocate(GVI%ValueLocInt)
        if(allocated(GVI%ValueWeight))      deallocate(GVI%ValueWeight)
        if(allocated(GVI%ValueAllWeight))   deallocate(GVI%ValueAllWeight)
    end subroutine destroyGridValueInterpolation

ENdModule ModuleGridBase



!
!
!
!subroutine SteppingNameGridBase(GD)
!   Class(GridBase), intent(inout) :: GD
!   Select case(GD%DumpNameMode)
!     Case(0)
!       GD%NameIndex=GD%NameIndex
!     Case(1)
!       GD%NameIndex=GD%NameIndex+1
!   ENd Select
!   Return
!End subroutine SteppingNameGridBase

!subroutine SteppingDimGridBase(GD,DimIndex)
!   Class(GridBase), intent(inout) :: GD
!   Integer(4), intent(in) :: DimIndex
!   Associate(Index=>GD%GIndex(DimIndex),Nmax=>GD%GN(DimIndex),NLoop=>GD%GLoop(DimIndex))
!   IF (DImIndex>=2) Then
!      GD%GIndex(DimIndex-1)=Mod(NLoop,GD%GN(DimIndex-1))
!
!   ENd If
!
!   End Associate
!
!
!   Return
!End subroutine SteppingDimGridBase

!subroutine RescaleGrid(GD)
!    import GridBase
!    class(GridBase),intent(inout) :: GD
!End subroutine

!subroutine UpdateGridA2(GD)
!        class(GridBase), intent(inout) :: GD
!         GD%Ns=1
!    End subroutine UpdateGridA2
!    subroutine InitGridBase(GD,NameMode,DumpNameMode,Ns,Period,Shift,Timer)
!        !import ControlFlow
!        class(GridBase), intent(inout) :: GD
!        Integer(4),intent(in),optional :: NameMode,DumpNameMode
!        Integer(4),intent(in),optional :: Ns,Shift
!        Integer(4),intent(in),optional :: Period,Timer
!        If (Present(NameMode).and.Present(DumpNameMode)) Then
!            Call InitGridBaseNameMode(GD,NameMode,DumpNameMode)
!        Else If (Present(NameMode)) Then
!            Call InitGridBaseNameMode(GD,NameMode)
!        Else
!            Call InitGridBaseNameMode(GD)
!        ENd If
!
!        If (Present(Ns).and.Present(Shift)) Then
!            Call InitGridBaseShift(GD,Ns,Shift)
!        Else If (Present(Ns)) Then
!            Call InitGridBaseShift(GD,Ns)
!        Else
!            Call InitGridBaseShift(GD)
!        ENd If
!
!        If (Present(Period).and.Present(Timer)) Then
!            Call InitGridBaseTimer(GD,Period,Timer)
!        Else If (Present(Period)) Then
!            Call InitGridBaseShift(GD,Period)
!        Else
!            Call InitGridBaseShift(GD)
!        ENd If
!
!
!    End subroutine InitGridBase
!
!    subroutine InitGridBaseNameMode(GD,NameMode,DumpNameMode)
!        !import ControlFlow
!        Class(GridBase), intent(inout) :: GD
!        Integer(4),intent(in),optional :: NameMode,DumpNameMode
!        If (present(NameMode)) Then
!            GD%NameMode=NameMode
!        Else
!            GD%NameMode=0
!        End If
!
!         If (present(DumpNameMode)) Then
!            GD%DumpNameMode=DumpNameMode
!        Else
!            GD%DumpNameMode=0
!        End If
!
!        Return
!    End subroutine InitGridBaseNameMode
!
!   subroutine InitGridN0(GD,N0,N0Index,N0Loop)
!        !import ControlFlow
!        Class(GridBase), intent(inout) :: GD
!        Integer(4),intent(in),optional :: Ns,Shift
!    If (present(Ns)) Then
!        GD%Ns=Ns
!    Else
!        GD%Ns=1
!    End If
!
!    If (present(Shift)) Then
!        GD%Shift=Shift
!    Else
!        GD%Shift=1
!    End If
!
!        Return
!    End subroutine InitGridBaseShift
!
!    subroutine InitGridBaseTimer(GD,Period,Timer)
!        !import ControlFlow
!        Class(GridBase), intent(inout) :: GD
!        Integer(4),intent(in),optional :: Period,Timer
!    If (present(Period)) Then
!        GD%Period=Period
!    Else
!        GD%Period=1
!    End If
!
!    If (present(Timer)) Then
!        GD%Timer=Timer
!    Else
!        GD%Timer=0
!    End If
!
!        Return
!    End subroutine InitGridBaseTimer
!
!subroutine UpdateGridShift(GD)
!    class(GridBase), intent(inout) :: GD
!     If (GD%Shift==GD%Ns) Then
!         GD%Shift=1
!     Else
!         GD%Shift=GD%Shift+1
!     END If
!     Return
!End subroutine UpdateGridShift
!
!subroutine UpdateGridTimer(GD)
!        class(GridBase), intent(inout) :: GD
!     If (GD%Timer==GD%Period) Then
!         GD%Timer=1
!     Else
!         GD%Timer=GD%Timer+1
!     END If
!     Return
!End subroutine UpdateGridTimer
!
!
!subroutine UpdateGridShiftTimer(GD)
!    class(GridBase), intent(inout) :: GD
!     If (GD%Shift==GD%Ns) Then
!         GD%Shift=1
!         GD%Timer=GD%Timer+1
!     Else
!         GD%Shift=GD%Shift+1
!     END If
!     Return
!End subroutine UpdateGridShiftTimer
!
!
!
!    subroutine SetGridShift(GD,Shift)
!        class(GridBase), intent(inout) :: GD
!        Integer(4),intent(in),optional :: Shift
!        GD%Shift=Shift
!End subroutine SetGridShift
!
!subroutine SetGridTimer(GD,Timer)
!        class(GridBase), intent(inout) :: GD
!        Integer(4),intent(in),optional :: Timer
!           GD%Timer=Timer
!End subroutine SetGridTimer
!
!subroutine ResetGridBase(GD)
!        class(GridBase), intent(inout) :: GD
!        Call GD%SetShift(0)
!        Call GD%SetTimer(0)
!End subroutine ResetGridBase
!subroutine SteppingIndexGridBaseSingle(GD,DimIndex)
!   Class(GridBase), intent(inout) :: GD
!   Integer(4), intent(in) :: DimIndex
!
!   If (DImIndex<1.or.DImIndex>MaxDim) Then
!       Write(*,*) "GridBase SteppingIndex Initialization Error!"
!   End If
!   Associate(Index=>GD%GIndex(DimIndex),Nmax=>GD%GN(DimIndex),NLoop=>GD%GLoop(DimIndex))
!    Index=Index+1
!    If (Index == Nmax) Then
!       NLoop = NLoop + 1
!       Index = 1
!    END If
!   End Associate
!   Return
!End subroutine SteppingIndexGridBaseSingle