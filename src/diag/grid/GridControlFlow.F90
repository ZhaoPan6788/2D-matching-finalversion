module ModuleGridControlFlow
    use ModuleGridBase
    ! use ModuleGrid0D
    use ModuleGrid1D
    use ModuleGrid2D
    use ModuleGrid3D
    ! use ModuleDomain  ! 可以解耦
    implicit none

    ! GridDim Arguments           Grid_Method_Code  DataDim
    ! 0D      (Ns)                100 200           0D
    ! 1D      (Nx1,Ns)            101 201           1D
    !         (Nt, Ns)            110 210           0D
    ! 2D      (Nx1,Nx2,Ns)        102 202           2D
    !         (Nx1,Nt, Ns)        111 211           1D
    ! 3D      (Nx1,Nx2,Nx3,Ns)    103 203           3D
    !         (Nx1,Nx2,Nt ,Ns)    112 212           2D

    type :: GridControlFlow
        integer(4)  :: Nx1, Nx2, Nx3, Nt = 0
        real(8)     :: Dx1, Dx2, Dx3, Dt = 0
        integer(4)  :: Ns = 0
        integer(4)  :: DimSpace, DimTime = 0
        integer(4)  :: Grid_Dim_Code     = 0
        integer(4)  :: Grid_Time_Code    = 0
        integer(4)  :: Grid_Method_Code  = 0
        integer(4)  :: Sim_Mode = SIM_MODE_STABLE
        ! type(Grid0D),allocatable :: G0D
        type(Grid1D),allocatable :: G1D
        type(Grid2D),allocatable :: G2D
        type(Grid3D),allocatable :: G3D
    contains
        procedure ::  Init => GridAllInit
        procedure ::  Dump => GridDump
        procedure ::  Destroy => DestroyGridControlFlow
        procedure, private :: GridUpdate0D,GridUpdate1D,GridUpdate2D
        generic :: Update =>    GridUpdate0D,&
                                GridUpdate1D,&
                                GridUpdate2D
    endtype GridControlFlow

contains
    
    ! Gird初始化
    subroutine GridAllInit(GridCF, NameInp, ExtensionMode, ParallelMode, DynamicIndex, &
                           MethodCode, Nx1, Nx2, Nx3, Nt, Ns, Dx1, Dx2, Dx3, Dt)
        class(GridControlFlow) :: GridCF
        Character(*), intent(in) :: NameInp
        integer(4), intent(in) :: ExtensionMode, ParallelMode, DynamicIndex
        integer(4), intent(in), optional :: MethodCode
        integer(4), intent(in), optional :: Nx1, Nx2, Nx3, Nt
        real(8),    intent(in), optional :: Dx1, Dx2, Dx3, Dt
        integer(4), intent(in)  :: Ns

        GridCF%DimSpace = 0
        GridCF%DimTime = 0

        if(present(Nx1)) then; GridCF%Nx1 = Nx1;  GridCF%DimSpace   = GridCF%DimSpace + 1; end if
        if(present(Nx2)) then; GridCF%Nx1 = Nx2;  GridCF%DimSpace   = GridCF%DimSpace + 1; end if
        if(present(Nx3)) then; GridCF%Nx1 = Nx3;  GridCF%DimSpace   = GridCF%DimSpace + 1; end if
        if(present(Nt))  then
            if(Nt<1) then
                error stop "=== Error: If exist Nt then Nt must > 1 ==="
            else
                GridCF%Nt  = Nt
                GridCF%DimTime    = GridCF%DimTime  + 1
            end if
        end if

        GridCF%Grid_Method_Code = GridCF%Sim_Mode + GridCF%DimTime * 10 + GridCF%DimSpace
        if(present(MethodCode)) GridCF%Grid_Method_Code = MethodCode
        GridCF%Grid_Dim_Code    = GridCF%DimTime  + GridCF%DimSpace

        GridCF%Ns     = Ns
        ! 下面这段需要严谨的错误判断
        if(present(Dx1))    GridCF%Dx1    = Dx1
        if(present(Dx2))    GridCF%Dx2    = Dx2
        if(present(Dx3))    GridCF%Dx3    = Dx3
        if(present(Dt))     GridCF%Dt     = Dt

        select case(GridCF%Grid_Dim_Code)
            ! case(GRID_DATA_DIMENSION_0D)
            !     allocate(GridCF%G0D)
            !     call GridCF%G0D%Init(Ns=Ns)
            case(GRID_DATA_DIMENSION_1D)
                allocate(GridCF%G1D)
                select case(GridCF%DimTime)
                    case(GRID_TIME_DEPENDENT)
                        call GridCF%G1D%Init(GRID_TIME_DEPENDENT,  Nt = Nt, Ns = Ns)
                    case(GRID_TIME_INDEPENDENT)
                        call GridCF%G1D%Init(GRID_TIME_INDEPENDENT,Nx1= Nx1, Ns = Ns)
                end select
                call GridCF%G1D%InitArray()
                call GridCF%G1D%IOName%Init(NameInp, PathMode=FILE_PATH_MODE_DIAG, &
                                                    ExtensionMode=ExtensionMode, &
                                                    ParallelMode=ParallelMode, &
                                                    DynamicIndex=DynamicIndex)
            case(GRID_DATA_DIMENSION_2D)
                allocate(GridCF%G2D)
                allocate(GridCF%G2D%VarName(Ns))
                select case(GridCF%DimTime)
                    case(GRID_TIME_DEPENDENT)
                        call GridCF%G2D%Init(GRID_TIME_DEPENDENT,  Nx1=Nx1, Nt =Nt,  Ns=Ns)
                    case(GRID_TIME_INDEPENDENT)
                        call GridCF%G2D%Init(GRID_TIME_INDEPENDENT,Nx1=Nx1, Nx2=Nx2, Ns=Ns)
                end select
                call GridCF%G2D%InitArray()
                call GridCF%G2D%IOName%Init(NameInp, PathMode=FILE_PATH_MODE_DIAG, &
                                                    ExtensionMode=ExtensionMode, &
                                                    ParallelMode=ParallelMode, &
                                                    DynamicIndex=DynamicIndex)
            case(GRID_DATA_DIMENSION_3D)
                allocate(GridCF%G3D)
                allocate(GridCF%G3D%VarName(Ns))
                select case(GridCF%DimTime)
                    case(GRID_TIME_DEPENDENT)
                        call GridCF%G3D%Init(GRID_TIME_DEPENDENT,  Nx1=Nx1, Nx2=Nx2, Nt =Nt, Ns=Ns)
                    case(GRID_TIME_INDEPENDENT)
                        error stop "=== Error: 3D GRID unfinished! ==="
                        call GridCF%G3D%Init(GRID_TIME_INDEPENDENT,Nx1=Nx1, Nx2=Nx2, Nx3=Nx3,Ns=Ns)
                end select
                call GridCF%G3D%InitArray()
                call GridCF%G3D%IOName%Init(NameInp, PathMode=FILE_PATH_MODE_DIAG, &
                                                    ExtensionMode=ExtensionMode, &
                                                    ParallelMode=ParallelMode, &
                                                    DynamicIndex=DynamicIndex)
            case default
                ! write(*,*) "=== Error: The number of input dimension don't meet any GRID module! ==="
                error stop "=== Error: The number of input dimension don't meet any GRID module! ==="
        end select
    end subroutine GridAllInit

    subroutine DestroyGridControlFlow(Grid)
        class(GridControlFlow), intent(inout):: Grid

        if (allocated(Grid%G1D)) then
            call Grid%G1D%destroy()
            deallocate(Grid%G1D)
        end if

        if (allocated(Grid%G2D)) then
            call Grid%G2D%destroy()
            deallocate(Grid%G2D)
        end if

        if (allocated(Grid%G3D)) then
            call Grid%G3D%destroy()
            deallocate(Grid%G3D)
        end if

    end subroutine DestroyGridControlFlow

    subroutine GridDump(Grid)
        class(GridControlFlow)  :: Grid
        write(*,*) Grid%Grid_Dim_Code
        select case(Grid%Grid_Dim_Code)
            case(GRID_DATA_DIMENSION_0D)
            case(GRID_DATA_DIMENSION_1D)
                select case(Grid%Grid_Method_Code)
                    case(101)
                        call Grid%G1D%Rescale(ADD_VALUEFINAL)
                        call Grid%G1D%Dump(DUMP_VALUEMODE_FINAL)
                        call Grid%G1D%Reset()
                    case(201)
                        call Grid%G1D%Rescale(ADD_VALUEFINAL)
                        call Grid%G1D%Dump(DUMP_VALUEMODE_FINAL)
                        call Grid%G1D%Reset()
                    ! case(自定义)
                end select
            case(GRID_DATA_DIMENSION_2D)
                select case(Grid%Grid_Method_Code)
                    case(102)
                        call Grid%G2D%Rescale(ADD_VALUEFINAL)
                        call Grid%G2D%Dump(DUMP_VALUEMODE_FINAL)
                        call Grid%G2D%Reset()
                    case(202)
                        call Grid%G2D%Rescale(ADD_VALUEFINAL)
                        call Grid%G2D%Dump(DUMP_VALUEMODE_FINAL)
                        call Grid%G2D%Reset()
                    case(111)
                        call Grid%G2D%Rescale(ADD_VALUEFINAL)
                        call Grid%G2D%Dump(DUMP_VALUEMODE_FINAL)
                        call Grid%G2D%Reset()
                    case(211)
                        call Grid%G2D%Rescale(VAL_VALUEFINAL)
                        call Grid%G2D%Dump(DUMP_VALUEMODE_FINAL)
                        call Grid%G2D%Reset()
                    ! case(自定义)
                    case default
                        error stop "=== Error: The GRID_METHOD_CODE has no matching step method! ==="
                end select
            case(GRID_DATA_DIMENSION_3D)
                select case(Grid%Grid_Method_Code)
                    case(112)
                        call Grid%G3D%Rescale(ADD_VALUEFINAL)
                        call Grid%G3D%Dump(DUMP_VALUEMODE_FINAL)
                        call Grid%G3D%Reset()
                    case(212)
                        call Grid%G3D%Rescale(VAL_VALUEFINAL)
                        call Grid%G3D%Dump(DUMP_VALUEMODE_FINAL)
                        call Grid%G3D%Reset()
                    ! case(自定义)
                    case default
                        error stop "=== Error: The GRID_METHOD_CODE has no matching step method! ==="
                end select
        end select
    end subroutine GridDump

    subroutine GridUpdate0D(Grid, Data, Dsetname)
        class(GridControlFlow)  :: Grid
        Real(8),intent(in)      :: Data
        character(*),intent(in) :: Dsetname
        select case(Grid%Grid_Method_Code)
            case(100)
                ! call Grid%G1D%Update()
            case(200)
                ! call Grid%G1D%Init(Nx1= 10, Ns = 10)
            case(110)
                ! call Grid%G1D%Update()
            case(210)
                ! call Grid%G1D%Init(Nx1= 10, Ns = 10)
        end select
    end subroutine GridUpdate0D

    subroutine GridUpdate1D(Grid, Data, Dsetname)
        class(GridControlFlow)  :: Grid
        Real(8),intent(in)      :: Data(:)
        character(*),intent(in) :: Dsetname
        select case(Grid%Grid_Method_Code)
            case(101)
                Grid%G1D%VarName(Grid%G1D%NsIndex)=trim(Dsetname)
                call Grid%G1D%Add(Data, INPUT_TO_STACK)
                If(Grid%G1D%NsIndex==1 .AND. Grid%G1D%NtInnerIndex==1) Then
                    CALL Grid%G1D%Rescale(ADD_VALUESTACK)
                    Call Grid%G1D%Add(Data, STACK_TO_FINAL)
                EndIf
            case(201)
                Grid%G1D%VarName(Grid%G1D%NsIndex)=trim(Dsetname)
                call Grid%G1D%Add(Data, INPUT_TO_STACK)
                If(Grid%G1D%NsIndex==1 .AND. Grid%G1D%NtInnerIndex==1) Then
                    CALL Grid%G1D%Rescale(ADD_VALUESTACK)
                    Call Grid%G1D%Add(Data, STACK_TO_FINAL)
                EndIf
            case(111)
                Grid%G2D%VarName(Grid%G2D%NsIndex)=trim(Dsetname)
                call Grid%G2D%Val(Data, INPUT_TO_STACK)
                If(Grid%G2D%NsIndex==1 .AND. Grid%G2D%NtInnerIndex==1) Then
                    call Grid%G2D%Rescale(VAL_VALUESTACK)
                    Call Grid%G2D%Add(Data, STACK_TO_FINAL)
                    call Grid%G2D%ResetStack
                EndIf
            case(211)
                Grid%G2D%VarName(Grid%G2D%NsIndex)=trim(Dsetname)
                call Grid%G2D%Add(Data, INPUT_TO_STACK)
                If(Grid%G2D%NsIndex==1 .AND. Grid%G2D%NtInnerIndex==1) Then
                    call Grid%G2D%Rescale(ADD_VALUESTACK)
                    call Grid%G2D%IOName%SetIndex(Grid%G2D%NtInnerLoop+LoopStart)
                    call grid%G2D%Dump(DUMP_VALUEMODE_STACK)
                    call Grid%G2D%ResetStack
                EndIf
            ! case(自定义)
            case default
                error stop "=== Error: The GRID_METHOD_CODE has no matching step method! ==="
        end select
    end subroutine GridUpdate1D

    subroutine GridUpdate2D(Grid, Data, Dsetname)
        class(GridControlFlow)  :: Grid
        Real(8),intent(in)      :: Data(:,:)
        character(*),intent(in) :: Dsetname
        select case(Grid%Grid_Method_Code)
            case(102)
                Grid%G2D%VarName(Grid%G2D%NsIndex)=trim(Dsetname)
                call Grid%G2D%Add(Data, INPUT_TO_STACK)
                If(Grid%G2D%NsIndex==1 .AND. Grid%G2D%NtInnerIndex==1) Then
                    CALL Grid%G2D%Rescale(ADD_VALUESTACK)
                    Call Grid%G2D%Add(Data, STACK_TO_FINAL)
                EndIf
            case(202)
                Grid%G2D%VarName(Grid%G2D%NsIndex)=trim(Dsetname)
                call Grid%G2D%Add(Data, INPUT_TO_STACK)
                If(Grid%G2D%NsIndex==1 .AND. Grid%G2D%NtInnerIndex==1) Then
                    CALL Grid%G2D%Rescale(ADD_VALUESTACK)
                    ! Call Grid%G2D%Add(Data, STACK_TO_FINAL)
                    call Grid%G2D%IOName%SetIndex(Grid%G2D%NtInnerLoop+LoopStart)
                    call grid%G2D%Dump(DUMP_VALUEMODE_STACK)
                    call Grid%G2D%ResetStack
                EndIf
            case(112)
                Grid%G3D%VarName(Grid%G3D%NsIndex)=trim(Dsetname)
                call Grid%G3D%Val(Data, INPUT_TO_STACK)
                If(Grid%G3D%NsIndex==1 .AND. Grid%G3D%NtInnerIndex==1) Then
                    call Grid%G3D%Rescale(VAL_VALUESTACK)
                    Call Grid%G3D%Add(Data, STACK_TO_FINAL)
                    call Grid%G3D%ResetStack
                EndIf
            case(212)
                Grid%G3D%VarName(Grid%G3D%NsIndex)=trim(Dsetname)
                call Grid%G3D%Add(Data, INPUT_TO_STACK)
                If(Grid%G3D%NsIndex==1 .AND. Grid%G3D%NtInnerIndex==1) Then
                    call Grid%G3D%Rescale(ADD_VALUESTACK)
                    call Grid%G3D%IOName%SetIndex(Grid%G3D%NtInnerLoop+LoopStart)
                    call grid%G3D%Dump(DUMP_VALUEMODE_STACK)
                    call Grid%G3D%ResetStack
                EndIf
            ! case(自定义)
            case default
                error stop "=== Error: The GRID_METHOD_CODE has no matching step method! ==="
        end select
    end subroutine GridUpdate2D
end module ModuleGridControlFlow