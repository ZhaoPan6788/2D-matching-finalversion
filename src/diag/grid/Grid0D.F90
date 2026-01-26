Module ModuleGrid0D

    Use ModuleGridBase
    Implicit none

    Type,Extends(GridBase) :: Grid0D
        Real(8),Allocatable ::  FinalValue(:)
        Real(8),Allocatable ::  StackValue(:)
    contains

        procedure :: InitArray=>InitializationArrayGrid0D

        procedure :: Dump       => DumpGrid0D
        procedure :: DumpHDF5   => DumpGrid0DHDF5
        procedure :: DumpDAT    => DumpGrid0DDAT

        procedure :: Reset      => G0D_FinalValue_Reset
        procedure :: ResetStack => G0D_StackValue_Reset
        procedure :: Rescale    => RescaleGrid0D

        procedure :: Val0D => Grid0D_Val_TimeIndependent
        procedure :: Add0D => Grid0D_Add_TimeIndependent

        generic :: Val=>Val0D
        generic :: Add=>Add0D

        !Generic :: ASSIGNMENT(=) => Val0D , Val1D
        !Generic :: Operator(+) => Add0D , Add1D

        !procedure :: Replace=>UpdateReplaceGrid0D
        !procedure :: Rescale=>RescaleGrid0D
        !procedure :: UpdateArray=>UpdateGrid0DArray
        !procedure :: InitALL=>InitializationGrid0DALL
    EndType Grid0D

Contains

    subroutine InitializationArrayGrid0D(GD)
        Implicit none
        Class(Grid0D),intent(inout)  :: GD
        If(Allocated(GD%Value)) Deallocate(GD%Value)
        If(Allocated(GD%VarName)) Deallocate(GD%VarName)
        Allocate(GD%Value(GD%Ns))
        Allocate(GD%VarName(GD%Ns))
        Call GD%Reset
        return
    endsubroutine InitializationArrayGrid0D

    subroutine DumpGrid0D(GD,NameMode)
        Implicit none
        Class(Grid0D),intent(inout)  :: GD
        Integer(4),intent(in),Optional :: NameMode
        Integer(4):: i,j,k
        open(10,file=GD%IOName)

        Write(10,FMt="(*(a,2x))") (trim(GD%VarName(i)),i=1,GD%Ns) !may be inconvenient when directly copy data to origin
        Write(10,FMt="(*(es21.14,1x))")(GD%Value(i),i=1,GD%Ns)

        close(10)
        !Write (filename, *) "Saving ", Filename, " Please wait..."
        !Write (filename, *) "Save ", Filename, "Complete!"
        return
    endsubroutine DumpGrid0D

    subroutine ResetGrid0D(GD)
        Implicit none
        Class(Grid0D),intent(inout)  :: GD
        GD%Value=0.d0
        Call GD%ResetNs()
        Call GD%ResetNt()
        return
    endsubroutine ResetGrid0D

    Subroutine RescaleGrid0D(GD)
        Implicit none
        Class(Grid0D),intent(inout)  :: GD
        !Integer(4),intent(in) :: Mode
        Associate(Val=>GD%Value,NsLoop=>GD%NsLoop)
            Val=Val/DBLE(NsLoop)
        EndAssociate
        Return
    EndSubroutine RescaleGrid0D

    Subroutine Grid0D_Val_TimeIndependent(GD,A0D)
        Implicit none
        Class(Grid0D),intent(inout)  :: GD
        Real(8),intent(in)  :: A0D
        ! Call AnyAvg(A1D(:),TempA1D(:))
        ! Associate(Val=>GD%Value,Ns=>GD%Ns,NsI=>GD%NsIndex,NtI=>GD%NtIndex)

        !     Val(:,Ns)=TempA1D
        !     Call GD%StepNs
        ! EndAssociate
        Return
    EndSubroutine Grid0D_Val_TimeIndependent

    Subroutine Grid0D_Add_TimeIndependent(GD,A0D,ReceiveMode)
        Implicit none
        Class(Grid0D),intent(inout)  :: GD
        integer,intent(in)  :: ReceiveMode
        Real(8),intent(in)  :: A0D
        select case(ReceiveMode)
            case(STACK_TO_FINAL)
                Associate(ValF=>GD%FinalValue, ValS=>GD%StackValue)
                    ValF(:)=ValF(:)+ValS(:)
                    call GD%ResetStack
                EndAssociate
            case(INPUT_TO_STACK)
                Associate(ValS=>GD%StackValue, NsI=>GD%NsIndex, NtI=>GD%NtInnerIndex, NtO=>GD%NtOuterIndex)
                    ValS(NsI)=ValS(NsI)+A0D
                    Call GD%StepNs
                    If(NsI==1) Then
                        Call GD%StepNtInner
                        if(NtI ==1) then
                            call GD%StepNtOuter
                        end if
                    EndIf
                EndAssociate
            case default
                error stop "=== Error: Add0D report that ReceiveMode does not exist! ==="
        end select
        Return
    EndSubroutine Grid0D_Add_TimeIndependent

    ! Subroutine UpdateAccumulation0D2Grid0D(GD,A0D)
    !     Implicit none
    !     Class(Grid0D),intent(inout)  :: GD
    !     Real(8),intent(in)  :: A0D
    !     Associate(Val=>GD%Value,Index=>GD%NsIndex)
    !         Val(Index)=Val(Index)+A0D
    !         ! GD%VarName(index) = NameInp
    !     EndAssociate
    !     Call GD%StepNs
    !     Return
    ! EndSubroutine UpdateAccumulation0D2Grid0D
ENdModule ModuleGrid0D

!Select case(GD%Order)
!    Case(0)
!     do i = 1, GD%GN(1)
!        Write (10, FMt="(*(es21.14,1x))") dble(i - 1)*GD%Dg(1), GD%Value(i)
!     end do
!    Case(1)
!
!End Select
!
! subroutine InitializationSizeGrid0D(GD, N, D)
!   Implicit none
!   Class(Grid0D), intent(inout)  :: GD
!   Integer(4), intent(in) :: N(:)
!   Real(8), intent(in), optional :: D(:)
!   If (Size(N)<1) Then
!       Write(*,*) "GridBase 1D Size Initialization Error!"
!   Else
!       If (present(D)) Then
!           Call GD%InitSizeN1(N(1),D(1))
!       Else
!           Call GD%InitSizeN1(N(1))
!       End If
!   End  If
!   return
!end subroutine InitializationSizeGrid0D

!subroutine SteppingDimGrid0D(GD)
!   Class(GridBase), intent(inout) :: GD
!   Select case(GD%DumpNameMode)
!     Case(0)
!       GD%NameIndex=GD%NameIndex
!     Case(1)
!       GD%NameIndex=GD%NameIndex+1
!   ENd Select
!   Return
!End subroutine SteppingDimGrid0D

!subroutine InitializationSizeN1Grid0D(GD, N1, D1)
!   Implicit none
!   Class(Grid0D), intent(inout)  :: GD
!   Integer(4), intent(in) :: N1
!   Real(8), intent(in), optional :: D1
!   GD%N1 = N1
!   If (present(D1)) Then
!      GD%D1 = D1
!   Else
!      GD%D1 = 1.d0
!   End If
!   return
!end subroutine InitializationSizeN1Grid0D