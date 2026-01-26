Module ModuleGrid2D
    Use ModuleGridBase
    Implicit none!
    !  This Section is the definitions of the physicla parameters.
    Integer(4),Parameter,Private  :: ThisGridDim=2
    Type,Extends(GridBase) :: Grid2D
        Real(8),Allocatable ::  FinalValue(:,:,:)  !(1:Nx,1:Ns)
        Real(8),Allocatable ::  StackValue(:,:,:)  !(1:Nx,1:Ns)
    contains
        procedure :: InitArray  => InitializationArrayGrid2D

        procedure :: Dump       => DumpGrid2D
        procedure :: DumpHDF5   => DumpGrid2DHDF5
        procedure :: DumpDAT    => DumpGrid2DDAT
        procedure :: Reset      => G2D_FinalValue_Reset
        procedure :: ResetStack => G2D_StackValue_Reset
        procedure :: Rescale    => RescaleGrid2D

        procedure :: Val1D => Grid2D_Val_TimeDependent
        procedure :: Val2D => Grid2D_Val_TimeIndependent
        procedure :: Add1D => Grid2D_Add_TimeDependent
        procedure :: Add2D => Grid2D_Add_TimeIndependent

        procedure :: destroy => Grid2d_Destroy

        generic :: Val  =>  Val1D, Val2D
        generic :: Add  =>  Add1D, Add2D

    EndType Grid2D
Contains
    subroutine InitializationArrayGrid2D(GD)
        Implicit none
        Class(Grid2D),intent(inout)  :: GD
        If(Allocated(GD%FinalValue)) Deallocate(GD%FinalValue)
        If(Allocated(GD%StackValue)) Deallocate(GD%StackValue)
        If(Allocated(GD%VarName))    Deallocate(GD%VarName)
        Select Case(GD%TimeMode)
            Case(GRID_TIME_DEPENDENT)
                Allocate(GD%FinalValue(GD%Nx1, GD%Nt, GD%Ns))
                Allocate(GD%StackValue(GD%Nx1, GD%Nt, GD%Ns))
                Allocate(GD%VarName(GD%Ns))
            Case(GRID_TIME_INDEPENDENT)
                Allocate(GD%FinalValue(GD%Nx1, GD%Nx2,GD%Ns))
                Allocate(GD%StackValue(GD%Nx1, GD%Nx2,GD%Ns))
                Allocate(GD%VarName(GD%Ns))
            case default
                error stop "=== Error: The TimeMode of the GRID don't exist! ==="
        ENdSelect
        Call GD%Reset
        call GD%ResetStack
        return
    endsubroutine InitializationArrayGrid2D

    subroutine Grid2d_Destroy(GD)
        Class(Grid2D),intent(inout)  :: GD
        If(Allocated(GD%FinalValue)) Deallocate(GD%FinalValue)
        If(Allocated(GD%StackValue)) Deallocate(GD%StackValue)
        If(Allocated(GD%VarName)) Deallocate(GD%VarName)
        call GD%GVIFinal%destroy()
        call GD%GVIStack%destroy()
    endsubroutine Grid2d_Destroy

    Subroutine DumpGrid2D(GD, ValueMode)
        Implicit none
        Class(Grid2D),intent(inout)  :: GD
        Integer(4),intent(in) :: ValueMode
        select case(GD%IOName%ExtensionMode)
            case(FILE_EXTENSION_MODE_H5)
                call GD%DumpHDF5(ValueMode)
            case(FILE_EXTENSION_MODE_DAT)
                call GD%DumpDAT(ValueMode)
            case default
                error stop "=== Error: Dump ExtensionMode doesn't match any method! ==="
        end select
    end subroutine DumpGrid2D

    Subroutine DumpGrid2DDAT(GD, ValueMode)
        Implicit none
        Class(Grid2D),intent(inout)  :: GD
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
                            Write (10, FMt="(*(a,2x))") "X", "T", (trim(GD%varname(k)), k=1, GD%Ns)
                            do j=1,Nt
                                DO k=1,Nx
                                    Write(10,FMt="(*(es21.14,1x))") dble(k-1)*dx,dble(j-1)*dt,(ValS(k,j,i),i=1,Ns)
                                end DO
                            end do
                        Case(GRID_TIME_INDEPENDENT)
                            Write (10, FMt="(*(a,2x))") "X", "Y", (trim(GD%varname(k)), k=1, GD%Ns)
                            do j=1,Ny
                                DO k=1,Nx
                                    Write(10,FMt="(*(es21.14,1x))") dble(k-1)*dx,dble(j-1)*dy,(ValS(k,j,i),i=1,Ns)
                                end DO
                            end do
                    end Select
                case(DUMP_VALUEMODE_FINAL)
                    select case(GD%TimeMode)
                        Case(GRID_TIME_DEPENDENT)
                            Write (10, FMt="(*(a,2x))") "X", "T", (trim(GD%varname(k)), k=1, GD%Ns)
                            do j=1,Nt
                                DO k=1,Nx
                                    Write(10,FMt="(*(es21.14,1x))") dble(k-1)*dx,dble(j-1)*dt,(ValF(k,j,i),i=1,Ns)
                                end DO
                            end do
                        Case(GRID_TIME_INDEPENDENT)
                            Write (10, FMt="(*(a,2x))") "X", "Y", (trim(GD%varname(k)), k=1, GD%Ns)
                            do j=1,Ny
                                DO k=1,Nx
                                    Write(10,FMt="(*(es21.14,1x))") dble(k-1)*dx,dble(j-1)*dy,(ValF(k,j,i),i=1,Ns)
                                end DO
                            end do
                    end Select
            end select
        EndAssociate
        close(10)
        return
    endsubroutine DumpGrid2DDAT

    subroutine DumpGrid2DHDF5(GD, ValueMode)
        Implicit none
        Class(Grid2D),intent(inout)  :: GD
        Integer(4),intent(in) :: ValueMode
        integer(4) :: i
        type(HDF5_PDump) :: G2DPDump
        call G2DPDump%init(filename=GD%IOName%FullName%str, mode='write', serial=.True.)
        call G2DPDump%open()
        select case(ValueMode)
            case(DUMP_VALUEMODE_STACK)
                select case(GD%TimeMode)
                    case(GRID_TIME_DEPENDENT)
                        do i=1,GD%Ns
                            call G2DPDump%write(trim(GD%VarName(i)), GD%StackValue(:,GD%NtInnerLoop,i))
                        end do
                    case(GRID_TIME_INDEPENDENT)
                        do i=1,GD%Ns
                            call G2DPDump%write(trim(GD%VarName(i)), GD%StackValue(:,:,i))
                        end do
                end select
            case(DUMP_VALUEMODE_FINAL)
                do i=1,GD%Ns
                    call G2DPDump%write(trim(GD%VarName(i)), GD%FinalValue(:,:,i))
                end do
        end select
        call G2DPDump%close()
    endsubroutine DumpGrid2DHDF5

    subroutine G2D_FinalValue_Reset(GD)
        Implicit none
        Class(Grid2D),intent(inout)  :: GD
        GD%FinalValue=0.d0
        Call GD%ResetNs()
        Call GD%ResetNt()
        return
    endsubroutine G2D_FinalValue_Reset

    subroutine G2D_StackValue_Reset(GD)
        Implicit none
        Class(Grid2D),intent(inout)  :: GD
        GD%StackValue=0.d0
        return
    endsubroutine G2D_StackValue_Reset

    Subroutine RescaleGrid2D(GD, RescaleMode)
        Implicit none
        Class(Grid2D),intent(inout)  :: GD
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
                    do i=1,GD%Nx1
                        do j=1,GD%Ns
                            ValS(i,:,j)=ValS(i,:,j)/InnerWeight(1:GD%GVIStack%Nt)
                        end do 
                    end do
                    call GD%ResetNs
                case(VAL_VALUEFINAL)
                    do i=1,GD%Nx1
                        do j=1,GD%Ns
                            ValF(i,:,j)=ValF(i,:,j)/OuterWeight(1:GD%GVIFinal%Nt)
                        end do
                    end do
                    call GD%ResetNt
            end select

        EndAssociate
        Return
    EndSubroutine RescaleGrid2D

    Subroutine Grid2D_Val_TimeDependent(GD,A1D,ReceiveMode)
        Implicit none
        Class(Grid2D),intent(inout)  :: GD
        integer,intent(in)  :: ReceiveMode
        Real(8),intent(in)  :: A1D(:)
        Real(8)  :: TempA1D(Size(GD%FinalValue,1))
        Call AnyAvg(A1D(:),TempA1D(:))
        select case(ReceiveMode)
            case(STACK_TO_FINAL)
                Associate(ValF=>GD%FinalValue, ValS=>GD%StackValue, NtI=>GD%NtInnerLoop, Loc=>GD%GVIFinal%ValueLocInt, Weight=>GD%GVIFinal%ValueWeight)
                    ValF(:,Loc(NtI,1),:)=ValF(:,Loc(NtI,1),:)+Weight(NtI,1)*ValS(:,NtI,:)
                    ValF(:,Loc(NtI,2),:)=ValF(:,Loc(NtI,2),:)+Weight(NtI,2)*ValS(:,NtI,:)
                    ! ValF(:,NtI,:)=ValS(:,NtI,:)
                EndAssociate
            case(INPUT_TO_STACK)
                Associate(ValS=>GD%StackValue, NsI=>GD%NsIndex, NtI=>GD%NtInnerIndex, Loc=>GD%GVIStack%ValueLocInt, Weight=>GD%GVIStack%ValueWeight)
                    ValS(:,Loc(NtI,1),NsI)=ValS(:,Loc(NtI,1),NsI)+Weight(NtI,1)*TempA1D
                    ValS(:,Loc(NtI,2),NsI)=ValS(:,Loc(NtI,2),NsI)+Weight(NtI,2)*TempA1D
                    ! ValS(:,NtI,NsI)=TempA1D(:)
                    Call GD%StepNs
                    If(NsI==1) Then
                        Call GD%StepNtInner
                        if(NtI ==1) then
                            call GD%StepNtOuter
                        end if
                    EndIf
                EndAssociate
            case default
                error stop "=== Error: Val1D report that ReceiveMode does not exist! ==="
        end select
        Return
    EndSubroutine Grid2D_Val_TimeDependent

    Subroutine Grid2D_Val_TimeIndependent(GD,A2D,ReceiveMode)
        Implicit none
        Class(Grid2D),intent(inout)  :: GD
        integer,intent(in)  :: ReceiveMode
        Real(8),intent(in)  :: A2D(:,:)
        Real(8)  :: TempA2D(Size(GD%FinalValue,1),Size(GD%FinalValue,2))
        Integer(4):: i,j,k
        ! Call AnyAvg(A2D(:,:),TempA2D(:,:))
        ! select case(ReceiveMode)
        !     case(RECEIVE_TO_FINAL)
        !         Associate(ValF=>GD%FinalValue, ValS=>GD%StackValue)
        !             ValF(:,:,:)=ValS(:,:,:)
        !             ! call GD%ResetStack
        !         EndAssociate
        !     case(RECEIVE_TO_STACK)
        !         Associate(ValS=>GD%StackValue, NsI=>GD%NsIndex, NtI=>GD%NtInnerIndex, NtO=>GD%NtOuterIndex)
        !             ValS(:,:,NsI)=TempA2D(:,:)
        !             Call GD%StepNs
        !             If(NsI==1) Then
        !                 Call GD%StepNtInner
        !                 if(NtI ==1) then
        !                     call GD%StepNtOuter
        !                 end if
        !             EndIf
        !         EndAssociate
        !     case default
        !         error stop "=== Error: Val2D report that ReceiveMode does not exist! ==="
        ! end Select
        Return
    EndSubroutine Grid2D_Val_TimeIndependent

    Subroutine Grid2D_Add_TimeDependent(GD,A1D,ReceiveMode)
        Implicit none
        Class(Grid2D),intent(inout)  :: GD
        integer,intent(in)  :: ReceiveMode
        Real(8),intent(in)  :: A1D(:)
        Real(8)  :: TempA1D(Size(GD%FinalValue,1))
        Call AnyAvg(A1D(:),TempA1D(:))
        select case(ReceiveMode)
            case(STACK_TO_FINAL)
                Associate(ValF=>GD%FinalValue, ValS=>GD%StackValue)
                    ValF(:,:,:)=ValF(:,:,:)+ValS(:,:,:)
                EndAssociate
            case(INPUT_TO_STACK)
                Associate(ValS=>GD%StackValue, NsI=>GD%NsIndex, NtI=>GD%NtInnerIndex, NtO=>GD%NtOuterIndex, Loc=>GD%GVIFinal%ValueLocInt, Weight=>GD%GVIFinal%ValueWeight)
                    ValS(:,Loc(NtO,1),NsI)=ValS(:,Loc(NtO,1),NsI)+Weight(NtO,1)*TempA1D
                    ValS(:,Loc(NtO,2),NsI)=ValS(:,Loc(NtO,2),NsI)+Weight(NtO,2)*TempA1D
                    ! ValS(:,NtO,NsI)=ValS(:,NtO,NsI)+TempA1D(:)
                    Call GD%StepNs
                    If(NsI==1) Then
                        call GD%StepNtInner
                        if(NtI ==1) then
                            call GD%StepNtOuter
                        end if
                    EndIf
                EndAssociate
            case default
                error stop "=== Error: Add1D report that ReceiveMode does not exist! ==="
        end select
        Return
    EndSubroutine Grid2D_Add_TimeDependent

    Subroutine Grid2D_Add_TimeIndependent(GD,A2D,ReceiveMode)
        Implicit none
        Class(Grid2D),intent(inout)  :: GD
        integer,intent(in)  :: ReceiveMode
        Real(8),intent(in)  :: A2D(:,:)
        Real(8)  :: TempA2D(Size(GD%FinalValue,1),Size(GD%FinalValue,2))
        Integer(4):: i,j,k
        Call AnyAvg(A2D(:,:),TempA2D(:,:))
        select case(ReceiveMode)
            case(STACK_TO_FINAL)
                Associate(ValF=>GD%FinalValue, ValS=>GD%StackValue)
                    ValF(:,:,:)=ValF(:,:,:)+ValS(:,:,:)
                    call GD%ResetStack
                EndAssociate
            case(INPUT_TO_STACK)
                Associate(ValS=>GD%StackValue, NsI=>GD%NsIndex, NtI=>GD%NtInnerIndex, NtO=>GD%NtOuterIndex)
                    ValS(:,:,NsI)=ValS(:,:,NsI)+TempA2D(:,:)
                    Call GD%StepNs
                    If(NsI==1) Then
                        Call GD%StepNtInner
                        if(NtI ==1) then
                            call GD%StepNtOuter
                        end if
                    EndIf
                EndAssociate
            case default
                error stop "=== Error: Add2D report that ReceiveMode does not exist! ==="
        end select
        Return
        Return
    EndSubroutine Grid2D_Add_TimeIndependent
ENdModule ModuleGrid2D