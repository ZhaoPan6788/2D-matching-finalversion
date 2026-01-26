Module ModuleGrid1D
    Use ModuleGridBase
    Implicit none!
    !  This Section is the definitions of the physicla parameters.
    !Integer(4),Parameter,Private  :: ThisGridDim=1
    Type,Extends(GridBase) :: Grid1D
        Real(8),Allocatable ::  FinalValue(:,:)  !(1:Nx,1:Ns)
        Real(8),Allocatable ::  StackValue(:,:)
    Contains
        procedure :: InitArray  =>InitializationArrayGrid1D
        procedure :: Dump       => DumpGrid1D
        procedure :: DumpHDF5   => DumpGrid1DHDF5
        procedure :: DumpDAT    => DumpGrid1DDAT

        procedure :: Reset      => G1D_FinalValue_Reset
        procedure :: ResetStack => G1D_StackValue_Reset
        procedure :: Rescale    => RescaleGrid1D

        procedure :: Val0D => Grid1D_Val_TimeDependent
        procedure :: Val1D => Grid1D_Val_TimeIndependent
        procedure :: Add0D => Grid1D_Add_TimeDependent
        procedure :: Add1D => Grid1D_Add_TimeIndependent

        procedure :: destroy => Grid1d_Destroy

        generic :: Val=>Val0D,Val1D
        generic :: Add=>Add0D,Add1D

    EndType Grid1D
Contains
    subroutine InitializationArrayGrid1D(GD)
        Implicit none
        Class(Grid1D),intent(inout)  :: GD
        If(Allocated(GD%FinalValue)) Deallocate(GD%FinalValue)
        If(Allocated(GD%StackValue)) Deallocate(GD%StackValue)
        If(Allocated(GD%VarName)) Deallocate(GD%VarName)
        Select Case(GD%TimeMode)
            Case(GRID_TIME_DEPENDENT)
                Allocate(GD%FinalValue(GD%Nt, GD%Ns))
                Allocate(GD%StackValue(GD%Nt, GD%Ns))
                Allocate(GD%VarName(GD%Ns))
            Case(GRID_TIME_INDEPENDENT)
                Allocate(GD%FinalValue(GD%Nx1,GD%Ns))
                Allocate(GD%StackValue(GD%Nx1,GD%Ns))
                Allocate(GD%VarName(GD%Ns))
        ENdSelect
        Call GD%Reset
        call GD%ResetStack
        return
    endsubroutine InitializationArrayGrid1D

    subroutine Grid1d_Destroy(GD)
        Class(Grid1D),intent(inout)  :: GD
        If(Allocated(GD%FinalValue)) Deallocate(GD%FinalValue)
        If(Allocated(GD%StackValue)) Deallocate(GD%StackValue)
        If(Allocated(GD%VarName)) Deallocate(GD%VarName)
        call GD%GVIFinal%destroy()
        call GD%GVIStack%destroy()
    endsubroutine Grid1d_Destroy

    ! subroutine DumpGrid1D(GD, ValueMode)
    !     Implicit none
    !     Class(Grid1D),intent(inout)  :: GD
    !     Integer(4),intent(in) :: ValueMode
    !     Integer(4):: i,j,k
    !     open(10,file=GD%IOName%FullName%str)
    !     Associate(Val=>GD%FinalValue, &
    !               Ns=>GD%Ns,Nt=>GD%Nt,Nx=>GD%Nx1,Ny=>GD%Nx2, &
    !               dt=>GD%Dt,dx=>GD%Dx1,dy=>GD%Dx2)
    !         Select Case(GD%TimeMode)
    !         Case(GRID_TIME_DEPENDENT)
    !             Write(10,FMt="(*(a,2x))") "t",(trim(GD%VarName(i)),i=1,GD%Ns) !may be inconvenient when directly copy data to origin
    !             do j=1,Nt
    !                 Write(10,FMt="(*(es21.14,1x))") dble(j-1)*dt,(Val(j,i),i=1,Ns)
    !             ENdDO
    !         Case(GRID_TIME_INDEPENDENT)
    !             Write(10,FMt="(*(a,2x))") "x",(trim(GD%VarName(i)),i=1,GD%Ns) !may be inconvenient when directly copy data to origin
    !             DO k=1,Nx
    !                 Write(10,FMt="(*(es21.14,1x))") dble(j-1)*dx,(Val(j,i),i=1,Ns)
    !             ENdDO
    !         ENdSelect

    !     EndAssociate
    !     close(10)
    !     return
    ! endsubroutine DumpGrid1D
    Subroutine DumpGrid1D(GD, ValueMode)
        Implicit none
        Class(Grid1D),intent(inout)  :: GD
        Integer(4),intent(in) :: ValueMode
        select case(GD%IOName%ExtensionMode)
            case(FILE_EXTENSION_MODE_H5)
                call GD%DumpHDF5(ValueMode)
            case(FILE_EXTENSION_MODE_DAT)
                call GD%DumpDAT(ValueMode)
            case default
                error stop "=== Error: Dump ExtensionMode doesn't match any method! ==="
        end select
    end subroutine DumpGrid1D

    Subroutine DumpGrid1DDAT(GD, ValueMode)
        Implicit none
        Class(Grid1D),intent(inout)  :: GD
        Integer(4):: i,j,k
        Integer(4),intent(in) :: ValueMode

        open(10,file=GD%IOName%FullName%str)
        Associate(ValF => GD%FinalValue, ValS => GD%StackValue, &
                      Ns=>GD%Ns, Nt=>GD%Nt,  Nx=>GD%Nx1, Ny=>GD%Nx2, &
                      dt=>GD%Dt, dx=>GD%Dx1, dy=>GD%Dx2)
            select case(ValueMode)
                case(DUMP_VALUEMODE_STACK)
                    select case(GD%TimeMode)
                        Case(GRID_TIME_DEPENDENT)
                            Write (10, FMt="(*(a,2x))") "T", (trim(GD%varname(k)), k=1, GD%Ns)
                            do j=1,Nt
                                Write(10,FMt="(*(es21.14,1x))") dble(j-1)*dt,(ValS(j,i),i=1,Ns)
                            end do
                        Case(GRID_TIME_INDEPENDENT)
                            Write (10, FMt="(*(a,2x))") "X", (trim(GD%varname(k)), k=1, GD%Ns)
                            DO j=1,Nx
                                Write(10,FMt="(*(es21.14,1x))") dble(j-1)*dy,(ValS(j,i),i=1,Ns)
                            end DO
                    end Select
                case(DUMP_VALUEMODE_FINAL)
                    select case(GD%TimeMode)
                        Case(GRID_TIME_DEPENDENT)
                            Write (10, FMt="(*(a,2x))") "T", (trim(GD%varname(k)), k=1, GD%Ns)
                            do j=1,Nt
                                Write(10,FMt="(*(es21.14,1x))") dble(j-1)*dt,(ValF(j,i),i=1,Ns)
                            end do
                        Case(GRID_TIME_INDEPENDENT)
                            Write (10, FMt="(*(a,2x))") "X", (trim(GD%varname(k)), k=1, GD%Ns)
                            DO j=1,Nx
                                Write(10,FMt="(*(es21.14,1x))") dble(j-1)*dy,(ValF(j,i),i=1,Ns)
                            end DO
                    end Select
            end select
        EndAssociate
        close(10)
        return
    endsubroutine DumpGrid1DDAT

    subroutine DumpGrid1DHDF5(GD, ValueMode)
        Implicit none
        Class(Grid1D),intent(inout)  :: GD
        Integer(4),intent(in) :: ValueMode
        integer(4) :: i
        type(HDF5_PDump) :: G1DPDump
        call G1DPDump%init(filename=GD%IOName%FullName%str, mode='write', serial=.True.)
        call G1DPDump%open()
        select case(ValueMode)
            case(DUMP_VALUEMODE_STACK)
                select case(GD%TimeMode)
                    case(GRID_TIME_DEPENDENT)
                        error stop "1D TIME_DEPENDENT HDF5 DUMP UNFINISHED"
                        ! do i=1,GD%Ns
                        !     call G1DPDump%write(trim(GD%VarName(i)), GD%StackValue(GD%NtInnerLoop,i))
                        ! end do
                    case(GRID_TIME_INDEPENDENT)
                        do i=1,GD%Ns
                            call G1DPDump%write(trim(GD%VarName(i)), GD%StackValue(:,i))
                        end do
                end select
            case(DUMP_VALUEMODE_FINAL)
                do i=1,GD%Ns
                    call G1DPDump%write(trim(GD%VarName(i)), GD%FinalValue(:,i))
                end do
        end select
        call G1DPDump%close()
    endsubroutine DumpGrid1DHDF5

    subroutine G1D_FinalValue_Reset(GD)
        Implicit none
        Class(Grid1D),intent(inout)  :: GD
        GD%FinalValue=0.d0
        Call GD%ResetNs()
        Call GD%ResetNt()
        return
    endsubroutine G1D_FinalValue_Reset

    subroutine G1D_StackValue_Reset(GD)
        Implicit none
        Class(Grid1D),intent(inout)  :: GD
        GD%StackValue=0.d0
        return
    endsubroutine G1D_StackValue_Reset

    Subroutine RescaleGrid1D(GD, RescaleMode)
        Implicit none
        Class(Grid1D),intent(inout)  :: GD
        Integer(4),intent(in) :: RescaleMode
        integer(4) :: i, j
        Associate(ValF => GD%FinalValue, InnerWeight => GD%GVIStack%ValueAllWeight, NsLoop => GD%NsLoop, &
            ValS => GD%StackValue, OuterWeight => GD%GVIFinal%ValueAllWeight, NtLoop => GD%NtInnerLoop)

            select case(RescaleMode)
                case(ADD_VALUESTACK)
                    ValS=ValS/DBLE(NsLoop)
                    call GD%ResetNs
                case(ADD_VALUEFINAL)
                    ValF=ValF/DBLE(NtLoop)
                    call GD%ResetNt
                case(VAL_VALUESTACK)
                    do j=1,GD%Ns
                        ValS(:,j)=ValS(:,j)/InnerWeight(1:GD%GVIStack%Nt)
                    end do 
                    call GD%ResetNs
                case(VAL_VALUEFINAL)
                    do j=1,GD%Ns
                        ValF(:,j)=ValF(:,j)/OuterWeight(1:GD%GVIFinal%Nt)
                    end do
                    call GD%ResetNt
            end select

        EndAssociate
        Return
    EndSubroutine RescaleGrid1D

    Subroutine Grid1D_Val_TimeDependent(GD,A0D,ReceiveMode)
        Implicit none
        Class(Grid1D),intent(inout)  :: GD
        integer,intent(in)  :: ReceiveMode
        Real(8),intent(in)  :: A0D
        select case(ReceiveMode)
            case(STACK_TO_FINAL)
                Associate(ValF=>GD%FinalValue, ValS=>GD%StackValue, NtI=>GD%NtInnerLoop)
                    ValF(NtI,:)=ValS(NtI,:)
                EndAssociate
            case(INPUT_TO_STACK)
                Associate(ValS=>GD%StackValue, NsI=>GD%NsIndex, NtI=>GD%NtInnerIndex, NtO=>GD%NtOuterIndex)
                    ValS(NtI,NsI)=A0D
                    Call GD%StepNs
                    If(NsI==1) Then
                        Call GD%StepNtInner
                        if(NtI ==1) then
                            call GD%StepNtOuter
                        end if
                    EndIf
                EndAssociate
            case default
                error stop "=== Error: Val0D report that ReceiveMode does not exist! ==="
        end select
        Return
    EndSubroutine Grid1D_Val_TimeDependent

    Subroutine Grid1D_Val_TimeIndependent(GD,A1D)
        Implicit none
        Class(Grid1D),intent(inout)  :: GD
        Real(8),intent(in)  :: A1D(:)
        Real(8)  :: TempA1D(Size(GD%FinalValue,1))
        ! Call AnyAvg(A1D(:),TempA1D(:))
        ! Associate(Val=>GD%Value,Ns=>GD%Ns,NsI=>GD%NsIndex,NtI=>GD%NtIndex)

        !     Val(:,Ns)=TempA1D
        !     Call GD%StepNs
        ! EndAssociate
        Return
    EndSubroutine Grid1D_Val_TimeIndependent

    Subroutine Grid1D_Add_TimeDependent(GD,A0D,ReceiveMode)
        Implicit none
        Class(Grid1D),intent(inout)  :: GD
        integer,intent(in)  :: ReceiveMode
        Real(8),intent(in)  :: A0D
        select case(ReceiveMode)
            case(STACK_TO_FINAL)
                Associate(ValF=>GD%FinalValue, ValS=>GD%StackValue)
                    ValF(:,:)=ValF(:,:)+ValS(:,:)
                EndAssociate
            case(INPUT_TO_STACK)
                Associate(ValS=>GD%StackValue, NsI=>GD%NsIndex, NtI=>GD%NtInnerIndex, NtO=>GD%NtOuterIndex)
                    ValS(NtO,NsI)=ValS(NtO,NsI)+A0D
                    Call GD%StepNs
                    If(NsI==1) Then
                        call GD%StepNtInner
                        if(NtI ==1) then
                            call GD%StepNtOuter
                        end if
                    EndIf
                EndAssociate
            case default
                error stop "=== Error: Add0D report that ReceiveMode does not exist! ==="
        end select
        Return
    EndSubroutine Grid1D_Add_TimeDependent

    Subroutine Grid1D_Add_TimeIndependent(GD,A1D,ReceiveMode)
        Implicit none
        Class(Grid1D),intent(inout)  :: GD
        integer,intent(in)  :: ReceiveMode
        Real(8),intent(in)  :: A1D(:)
        Real(8)  :: TempA1D(Size(GD%FinalValue,1))
        Call AnyAvg(A1D(:),TempA1D(:))
        select case(ReceiveMode)
            case(STACK_TO_FINAL)
                Associate(ValF=>GD%FinalValue, ValS=>GD%StackValue)
                    ValF(:,:)=ValF(:,:)+ValS(:,:)
                    call GD%ResetStack
                EndAssociate
            case(INPUT_TO_STACK)
                Associate(ValS=>GD%StackValue, NsI=>GD%NsIndex, NtI=>GD%NtInnerIndex, NtO=>GD%NtOuterIndex)
                    ValS(:,NsI)=ValS(:,NsI)+TempA1D(:)
                    Call GD%StepNs
                    If(NsI==1) Then
                        Call GD%StepNtInner
                        if(NtI ==1) then
                            call GD%StepNtOuter
                        end if
                    EndIf
                EndAssociate
            case default
                error stop "=== Error: Add1D report that ReceiveMode does not exist! ==="
        end select
        Return
    EndSubroutine Grid1D_Add_TimeIndependent

ENdModule ModuleGrid1D