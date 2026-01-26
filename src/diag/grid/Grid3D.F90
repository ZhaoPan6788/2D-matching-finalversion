Module ModuleGrid3D
    Use ModuleGridBase
    Implicit none!
    !  This Section is the definitions of the physicla parameters.
    Integer(4),Parameter,Private  :: ThisGridDim=3
    Type,Extends(GridBase) :: Grid3D
    Real(8),Allocatable ::  FinalValue(:,:,:,:)  !(1:Nx,1:Ns)
    Real(8),Allocatable ::  StackValue(:,:,:,:)  !(1:Nx,1:Ns)
contains
    procedure :: InitArray  => InitializationArrayGrid3D

    procedure :: Dump       => DumpGrid3D
    procedure :: DumpHDF5   => DumpGrid3DHDF5
    procedure :: DumpDAT    => DumpGrid3DDAT
    procedure :: Reset      => G3D_FinalValue_Reset
    procedure :: ResetStack => G3D_StackValue_Reset
    procedure :: Rescale    => RescaleGrid3D

    procedure :: Val2D => Grid3D_Val_TimeDependent
    procedure :: Val3D => Grid3D_Val_TimeIndependent
    procedure :: Add2D => Grid3D_Add_TimeDependent
    procedure :: Add3D => Grid3D_Add_TimeIndependent

    procedure :: destroy => Grid3d_Destroy

    generic :: Val  =>  Val2D, Val3D
    generic :: Add  =>  Add2D, Add3D

    EndType Grid3D
Contains
    subroutine InitializationArrayGrid3D(GD)
        Implicit none
        Class(Grid3D),intent(inout)  :: GD
        If(Allocated(GD%FinalValue)) Deallocate(GD%FinalValue)
        If(Allocated(GD%StackValue)) Deallocate(GD%StackValue)
        If(Allocated(GD%VarName))    Deallocate(GD%VarName)
        Select Case(GD%TimeMode)
            Case(GRID_TIME_DEPENDENT)
                Allocate(GD%FinalValue(GD%Nx1, GD%Nx2, GD%Nt, GD%Ns))
                Allocate(GD%StackValue(GD%Nx1, GD%Nx2, GD%Nt, GD%Ns))
                Allocate(GD%VarName(GD%Ns))
            Case(GRID_TIME_INDEPENDENT)
                error stop "=== Error: 3D TIME_INDEPENDENT GRID unfinished! ==="
                Allocate(GD%FinalValue(GD%Nx1, GD%Nx2, GD%Nx3,GD%Ns))
                Allocate(GD%StackValue(GD%Nx1, GD%Nx2, GD%Nx3,GD%Ns))
                Allocate(GD%VarName(GD%Ns))
            case default
                error stop "=== Error: The TimeMode of the GRID don't exist! ==="
        ENdSelect
        Call GD%Reset
        call GD%ResetStack
        return
    endsubroutine InitializationArrayGrid3D

    subroutine Grid3d_Destroy(GD)
        Class(Grid3D),intent(inout)  :: GD
        If(Allocated(GD%FinalValue)) Deallocate(GD%FinalValue)
        If(Allocated(GD%StackValue)) Deallocate(GD%StackValue)
        If(Allocated(GD%VarName)) Deallocate(GD%VarName)
        call GD%GVIFinal%destroy()
        call GD%GVIStack%destroy()
    endsubroutine Grid3d_Destroy


    Subroutine DumpGrid3D(GD, ValueMode)
        Implicit none
        Class(Grid3D),intent(inout)  :: GD
        Integer(4),intent(in) :: ValueMode
        select case(GD%IOName%ExtensionMode)
            case(FILE_EXTENSION_MODE_H5)
                call GD%DumpHDF5(ValueMode)
            case(FILE_EXTENSION_MODE_DAT)
                call GD%DumpDAT(ValueMode)
            case default
                error stop "=== Error: Dump ExtensionMode doesn't match any method! ==="
        end select
    end subroutine DumpGrid3D

    Subroutine DumpGrid3DDAT(GD, ValueMode)
        Implicit none
        Class(Grid3D),intent(inout)  :: GD
        Integer(4):: i,j,k,m
        Integer(4),intent(in) :: ValueMode

        open(10,file=GD%IOName%FullName%str)
        Associate(ValF => GD%FinalValue, ValS => GD%StackValue, &
                      Ns=>GD%Ns, Nt=>GD%Nt,  Nx=>GD%Nx1, Ny=>GD%Nx2, Nz=>GD%Nx3, &
                      dt=>GD%Dt, dx=>GD%Dx1, dy=>GD%Dx2, Dz=>GD%Dx3)
            select case(ValueMode)
                case(DUMP_VALUEMODE_STACK)
                    select case(GD%TimeMode)
                        Case(GRID_TIME_DEPENDENT)
                            Write (10, FMt="(*(a,2x))") "X", "Y", "T", (trim(GD%varname(k)), k=1, GD%Ns)
                            do j=1,Nt
                                DO k=1,Nx
                                    do m=1,Ny
                                        Write(10,FMt="(*(es21.14,1x))") dble(k-1)*dx,dble(m-1)*dy,dble(j-1)*dt,(ValS(k,m,j,i),i=1,Ns)
                                    end do
                                end DO
                            end do
                        Case(GRID_TIME_INDEPENDENT)
                            error stop "=== Error: 3D TIME_INDEPENDENT GRID unfinished! ==="
                            Write (10, FMt="(*(a,2x))") "X", "Y", "Z", (trim(GD%varname(k)), k=1, GD%Ns)
                            do m=1,Nz
                                do j=1,Ny
                                    DO k=1,Nx
                                        Write(10,FMt="(*(es21.14,1x))") dble(k-1)*dx,dble(j-1)*dy,dble(m-1)*Dz,(ValS(k,j,m,i),i=1,Ns)
                                    end DO
                                end do
                            end do
                    end Select
                case(DUMP_VALUEMODE_FINAL)
                    select case(GD%TimeMode)
                        Case(GRID_TIME_DEPENDENT)
                            Write (10, FMt="(*(a,2x))") "X", "Y", "T", (trim(GD%varname(k)), k=1, GD%Ns)
                            do j=1,Nt
                                DO k=1,Nx
                                    do m=1,Ny
                                        Write(10,FMt="(*(es21.14,1x))") dble(k-1)*dx,dble(m-1)*dy,dble(j-1)*dt,(ValF(k,m,j,i),i=1,Ns)
                                    end do
                                end DO
                            end do
                        Case(GRID_TIME_INDEPENDENT)
                            error stop "=== Error: 3D TIME_INDEPENDENT GRID unfinished! ==="
                            Write (10, FMt="(*(a,2x))") "X", "Y", "Z", (trim(GD%varname(k)), k=1, GD%Ns)
                            do m=1,Nz
                                do j=1,Ny
                                    DO k=1,Nx
                                        Write(10,FMt="(*(es21.14,1x))") dble(k-1)*dx,dble(j-1)*dy,dble(m-1)*Dz,(ValS(k,j,m,i),i=1,Ns)
                                    end DO
                                end do
                            end do
                    end Select
            end select
        EndAssociate
        close(10)
        return
    endsubroutine DumpGrid3DDAT

    subroutine DumpGrid3DHDF5(GD, ValueMode)
        Implicit none
        Class(Grid3D),intent(inout)  :: GD
        Integer(4),intent(in) :: ValueMode
        integer(4) :: i
        type(HDF5_PDump) :: G3DPDump
        call G3DPDump%init(filename=GD%IOName%FullName%str, mode='write', serial=.True.)
        call G3DPDump%open()
        select case(ValueMode)
            case(DUMP_VALUEMODE_STACK)
                select case(GD%TimeMode)
                    case(GRID_TIME_DEPENDENT)
                        do i=1,GD%Ns
                            call G3DPDump%write(trim(GD%VarName(i)), GD%StackValue(:,:,GD%NtInnerLoop,i))
                        end do
                    case(GRID_TIME_INDEPENDENT)
                        do i=1,GD%Ns
                            call G3DPDump%write(trim(GD%VarName(i)), GD%StackValue(:,:,:,i))
                        end do
                end select
            case(DUMP_VALUEMODE_FINAL)
                do i=1,GD%Ns
                    call G3DPDump%write(trim(GD%VarName(i)), GD%FinalValue(:,:,:,i))
                end do
        end select
        call G3DPDump%close()
    endsubroutine DumpGrid3DHDF5

    subroutine G3D_FinalValue_Reset(GD)
        Implicit none
        Class(Grid3D),intent(inout)  :: GD
        GD%FinalValue=0.d0
        Call GD%ResetNs()
        Call GD%ResetNt()
        return
    endsubroutine G3D_FinalValue_Reset

    subroutine G3D_StackValue_Reset(GD)
        Implicit none
        Class(Grid3D),intent(inout)  :: GD
        GD%StackValue=0.d0
        return
    endsubroutine G3D_StackValue_Reset

    Subroutine RescaleGrid3D(GD, RescaleMode)
        Implicit none
        Class(Grid3D),intent(inout)  :: GD
        Integer(4),intent(in) :: RescaleMode
        integer(4) :: i, j, k
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
                        do k=1,GD%Nx2
                            do j=1,GD%Ns
                                ValS(i,k,:,j)=ValS(i,k,:,j)/InnerWeight(1:GD%GVIStack%Nt)
                            end do 
                        end do
                    end do
                    call GD%ResetNs
                case(VAL_VALUEFINAL)
                    do i=1,GD%Nx1
                        do k=1,GD%Nx2
                            do j=1,GD%Ns
                                ValF(i,k,:,j)=ValF(i,k,:,j)/OuterWeight(1:GD%GVIFinal%Nt)
                            end do
                        end do
                    end do
                    call GD%ResetNt
            end select

        EndAssociate
        Return
    EndSubroutine RescaleGrid3D

    Subroutine Grid3D_Val_TimeDependent(GD,A2D,ReceiveMode)
        Implicit none
        Class(Grid3D),intent(inout)  :: GD
        integer,intent(in)  :: ReceiveMode
        Real(8),intent(in)  :: A2D(:,:)
        Real(8)  :: TempA2D(Size(GD%FinalValue,1),Size(GD%FinalValue,2))
        Call AnyAvg(A2D(:,:),TempA2D(:,:))
        select case(ReceiveMode)
            case(STACK_TO_FINAL)
                Associate(ValF=>GD%FinalValue, ValS=>GD%StackValue, NtI=>GD%NtInnerLoop, Loc=>GD%GVIFinal%ValueLocInt, Weight=>GD%GVIFinal%ValueWeight)
                    ValF(:,:,Loc(NtI,1),:)=ValF(:,:,Loc(NtI,1),:)+Weight(NtI,1)*ValS(:,:,NtI,:)
                    ValF(:,:,Loc(NtI,2),:)=ValF(:,:,Loc(NtI,2),:)+Weight(NtI,2)*ValS(:,:,NtI,:)
                EndAssociate
            case(INPUT_TO_STACK)
                Associate(ValS=>GD%StackValue, NsI=>GD%NsIndex, NtI=>GD%NtInnerIndex, Loc=>GD%GVIStack%ValueLocInt, Weight=>GD%GVIStack%ValueWeight)
                    ValS(:,:,Loc(NtI,1),NsI)=ValS(:,:,Loc(NtI,1),NsI)+Weight(NtI,1)*TempA2D
                    ValS(:,:,Loc(NtI,2),NsI)=ValS(:,:,Loc(NtI,2),NsI)+Weight(NtI,2)*TempA2D
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
    EndSubroutine Grid3D_Val_TimeDependent

    Subroutine Grid3D_Val_TimeIndependent(GD,A2D,ReceiveMode)
        Implicit none
        Class(Grid3D),intent(inout)  :: GD
        integer,intent(in)  :: ReceiveMode
        Real(8),intent(in)  :: A2D(:,:,:)
        Real(8)  :: TempA3D(Size(GD%FinalValue,1),Size(GD%FinalValue,2),Size(GD%FinalValue,3))
        Integer(4):: i,j,k
        error stop "=== Error: 3D GRID unfinished! ==="
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
    EndSubroutine Grid3D_Val_TimeIndependent

    Subroutine Grid3D_Add_TimeDependent(GD,A2D,ReceiveMode)
        Implicit none
        Class(Grid3D),intent(inout)  :: GD
        integer,intent(in)  :: ReceiveMode
        Real(8),intent(in)  :: A2D(:,:)
        Real(8)  :: TempA2D(Size(GD%FinalValue,1),Size(GD%FinalValue,2))
        Call AnyAvg(A2D(:,:),TempA2D(:,:))
        select case(ReceiveMode)
            case(STACK_TO_FINAL)
                Associate(ValF=>GD%FinalValue, ValS=>GD%StackValue)
                    ValF(:,:,:,:)=ValF(:,:,:,:)+ValS(:,:,:,:)
                EndAssociate
            case(INPUT_TO_STACK)
                Associate(ValS=>GD%StackValue, NsI=>GD%NsIndex, NtI=>GD%NtInnerIndex, NtO=>GD%NtOuterIndex, Loc=>GD%GVIFinal%ValueLocInt, Weight=>GD%GVIFinal%ValueWeight)
                    ValS(:,:,Loc(NtO,1),NsI)=ValS(:,:,Loc(NtO,1),NsI)+Weight(NtO,1)*TempA2D
                    ValS(:,:,Loc(NtO,2),NsI)=ValS(:,:,Loc(NtO,2),NsI)+Weight(NtO,2)*TempA2D
                    Call GD%StepNs
                    If(NsI==1) Then
                        call GD%StepNtInner
                        if(NtI ==1) then
                            call GD%StepNtOuter
                        end if
                    EndIf
                EndAssociate
            case default
                error stop "=== Error: Add2D report that ReceiveMode does not exist! ==="
        end select
        Return
    EndSubroutine Grid3D_Add_TimeDependent

    Subroutine Grid3D_Add_TimeIndependent(GD,A3D,ReceiveMode)
        Implicit none
        Class(Grid3D),intent(inout)  :: GD
        integer,intent(in)  :: ReceiveMode
        Real(8),intent(in)  :: A3D(:,:,:)
        Real(8)  :: TempA3D(Size(GD%FinalValue,1),Size(GD%FinalValue,2),Size(GD%FinalValue,3))
        Integer(4):: i,j,k
        error stop "=== Error: 3D GRID unfinished! ==="
        ! Call AnyAvg(A2D(:,:),TempA2D(:,:))
        ! select case(ReceiveMode)
        !     case(STACK_TO_FINAL)
        !         Associate(ValF=>GD%FinalValue, ValS=>GD%StackValue)
        !             ValF(:,:,:)=ValF(:,:,:)+ValS(:,:,:)
        !             call GD%ResetStack
        !         EndAssociate
        !     case(INPUT_TO_STACK)
        !         Associate(ValS=>GD%StackValue, NsI=>GD%NsIndex, NtI=>GD%NtInnerIndex, NtO=>GD%NtOuterIndex)
        !             ValS(:,:,NsI)=ValS(:,:,NsI)+TempA2D(:,:)
        !             Call GD%StepNs
        !             If(NsI==1) Then
        !                 Call GD%StepNtInner
        !                 if(NtI ==1) then
        !                     call GD%StepNtOuter
        !                 end if
        !             EndIf
        !         EndAssociate
        !     case default
        !         error stop "=== Error: Add2D report that ReceiveMode does not exist! ==="
        ! end select
        ! Return
        Return
    EndSubroutine Grid3D_Add_TimeIndependent
ENdModule ModuleGrid3D