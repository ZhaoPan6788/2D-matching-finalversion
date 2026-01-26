Module DiagnosticsParticleField
    use mpi
    use ModuleControlFlow
    use ModuleGridControlFlow
    use ModuleParallelDump
    use ModuleParticleBundle
    use ModuleSpecyOne
    use ModuleFieldEM
    use ModuleFieldOne
    use ModuleFieldSource
    use ModulePICCommunication
    use ModuleGrid

    implicit none

    type ParticleFieldDiagOne
        integer(4) :: Nz, Nr
        real(8), allocatable :: RhoOne(:, :)
        real(8), allocatable :: EnergyOne(:, :)
        real(8), allocatable :: TOne(:, :)
        real(8), allocatable :: JzOne(:, :)
        real(8), allocatable :: JrOne(:, :)
        real(8), allocatable :: MPOne(:, :)

    end type ParticleFieldDiagOne

    type ParticleFieldDiag
        integer(4) :: Nz, Nr, Ns
        integer(4) :: Period
        type(ParticleFieldDiagOne), allocatable :: DiagOne(:)

        real(8), allocatable :: Phi(:, :)
        real(8), allocatable :: Chi(:, :)

    contains

        procedure :: init       => initParticleFieldDiag
        procedure :: destroy    => destroyParticleFieldDiag

    end type ParticleFieldDiag

    ! Grid类实例化
    type(GridControlFlow),   save :: particle_field_grid_2d
    type(ParticleFieldDiag), save :: particle_field_diag_2d
    type(ParticleFieldDiag), save :: particle_field_diag_2d_local

    integer(4), parameter :: steady_diag_Nx = 50
    integer(4), parameter :: n_monitor = 3
    type(Grid1DS), save :: steady_diag(n_monitor)
    logical, save :: is_open_steady_diag = .False.

    type(HDF5_PDump), save, private :: hdf5ParticleFieldDiagDump

contains

    ! 初始化过程
    subroutine DiagParticleFieldInitilalization(CF)
        class(ControlFlow), intent(in) :: CF
        integer(4) :: k

        associate (grid => particle_field_grid_2d, &
                   global => particle_field_diag_2d, &
                   local => particle_field_diag_2d_local, &
                   Nz => CF%DM%GlobalShape(1), &
                   Nr => CF%DM%GlobalShape(2), &
                   lz => CF%DM%LocalShape(1), &
                   lr => CF%DM%LocalShape(2), &
                   zstart => CF%DM%CornerIndex(BOUNDARY_LEFT, 1), &
                   zend => CF%DM%CornerIndex(BOUNDARY_RIGHT, 1), &
                   rstart => CF%DM%CornerIndex(BOUNDARY_LEFT, 2), &
                   rend => CF%DM%CornerIndex(BOUNDARY_RIGHT, 2))

            call DiagParticleFieldPeriodRelease()

            call global%init(CF)
            call local%init(CF)

            call ZeroParticleFieldDiag(global)
            call ZeroParticleFieldDiag(local)

            ! 对Grid进行初始化，Nx1和Nt建议在更前置的模块使用Parameter引入以使其在所有诊断中相同
            particle_field_grid_2d%Sim_Mode = SIM_MODE_EVOLUTION
            call particle_field_grid_2d%Init('DiagParticleField2D', ExtensionMode=FILE_EXTENSION_MODE_H5, &
                                ParallelMode=FILE_PARALLEL_MODE_CLOSE, &
                                DynamicIndex=FILE_DYNAMIC_MODE_OPEN, &
                                Nx1=Nz, Nx2=Nr, Ns=6*(CF%Ns + 1) + 1)
            
            if (0 == CF%DM%MyId .and. is_open_steady_diag) then
                do k = 1, n_monitor
                    call steady_diag(k)%Init('steady_diag1d_'//trim(num2str(k, 1)), steady_diag_Nx, 4*(CF%Ns + 1) + 1, CF%Period, 1, diag_type_steady, CF%Timer/CF%Period)
                end do
            end if

        end associate

    end subroutine DiagParticleFieldInitilalization


    ! 诊断过程
    subroutine DiagParticleFieldPeriod(PB, FG, FO, FT, Geom)
        type(ParticleBundle), intent(in) :: PB(0:particle_field_diag_2d%Ns)
        type(FieldEM), intent(in) :: FG
        type(FieldOne), intent(in) :: FO(0:particle_field_diag_2d%Ns)
        type(FieldSource), intent(in) :: FT
        type(Geometry), intent(in) :: Geom
        integer(4) :: i, j, k, size, rank, ierr, m

        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

        call WeightingParticleField(particle_field_diag_2d, &
                                    particle_field_diag_2d_local, &
                                    PB, FG, FO, FT, Geom)

        associate (diag => particle_field_diag_2d, grid => particle_field_grid_2d)
            if (0 == rank) then
                do k = 0, diag%Ns
                    call grid%Update(diag%DiagOne(k)%RhoOne,      "RhoOne"//"-"//trim(SpecyGlobal(k)%Name))
                    call grid%Update(diag%DiagOne(k)%EnergyOne,   "EnergyOne"//"-"//trim(SpecyGlobal(k)%Name))
                    call grid%Update(diag%DiagOne(k)%TOne,        "TOne"//"-"//trim(SpecyGlobal(k)%Name))
                    call grid%Update(diag%DiagOne(k)%JzOne,       "JzOne"//"-"//trim(SpecyGlobal(k)%Name))
                    call grid%Update(diag%DiagOne(k)%JrOne,       "JrOne"//"-"//trim(SpecyGlobal(k)%Name))
                    call grid%Update(diag%DiagOne(k)%MPOne,       "MPOne"//"-"//trim(SpecyGlobal(k)%Name))
                end do

                call grid%Update(diag%Phi, "Phi")

            end if
        end associate

    end subroutine DiagParticleFieldPeriod


    subroutine ParticleFieldOpenSteadyDiag()

        is_open_steady_diag = .True.

    end subroutine ParticleFieldOpenSteadyDiag


    ! 存盘过程
    subroutine DiagParticleFieldPeriodDump()
        integer(4) :: rank, ierr, m

        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

        if (0 == rank) then
            if (is_open_steady_diag) then
                do m = 1, n_monitor
                    call steady_diag(m)%Dump()
                end do
            end if
        end if

    end subroutine DiagParticleFieldPeriodDump


    subroutine DiagParticleFieldPeriodRelease()
        integer(4) :: m

        call particle_field_grid_2d%Destroy()
        call particle_field_diag_2d%destroy()
        call particle_field_diag_2d_local%destroy()

        do m = 1, n_monitor
            call steady_diag(m)%Destroy()
        end do

    end subroutine DiagParticleFieldPeriodRelease


    ! 诊断量计算过程
    subroutine WeightingParticleField(diag_global, diag_local, PB, FG, FO, FT, Geom)
        class(ParticleFieldDiag), intent(inout) :: diag_global
        class(ParticleFieldDiag), intent(inout) :: diag_local
        type(ParticleBundle), intent(in) :: PB(0:diag_local%Ns)
        type(FieldEM), intent(in) :: FG
        type(FieldOne), intent(in) :: FO(0:diag_local%Ns)
        type(FieldSource), intent(in) :: FT
        type(Geometry), intent(in) :: Geom
        type(ParticleOne) :: tmp_one
        integer(4) :: i, j, k
        real(8)    :: EFactor, BFactor, EnergyFactor
        integer(4) :: NzL, NzU, NrL, NrU
        real(8)    :: ZL, ZU, RL, RU
        real(8)    :: Energy
        integer(4) :: zend_on_bound, rend_on_bound
        integer(4) :: ierr

        ! init
        call ZeroParticleFieldDiag(diag_global)
        call ZeroParticleFieldDiag(diag_local)

        ! weight
        do k = 0, diag_local%Ns
            do i = 1, PB(k)%Npar
                NzL = Ceiling(PB(k)%PO(i)%Z)
                NzU = NzL + 1
                ZL  = Dble(NzL) - PB(k)%PO(i)%Z
                ZU  = 1.d0 - ZL

                NrL = Ceiling(PB(k)%PO(i)%R)
                NrU = NrL + 1
                RL  = Dble(NrL) - PB(k)%PO(i)%R
                RU  = 1.d0 - RL

                diag_local%DiagOne(k)%RhoOne(NzL, NrL) = diag_local%DiagOne(k)%RhoOne(NzL, NrL) + ZL * RL * PB(k)%PO(i)%WQ
                diag_local%DiagOne(k)%RhoOne(NzL, NrU) = diag_local%DiagOne(k)%RhoOne(NzL, NrU) + ZL * RU * PB(k)%PO(i)%WQ
                diag_local%DiagOne(k)%RhoOne(NzU, NrL) = diag_local%DiagOne(k)%RhoOne(NzU, NrL) + ZU * RL * PB(k)%PO(i)%WQ
                diag_local%DiagOne(k)%RhoOne(NzU, NrU) = diag_local%DiagOne(k)%RhoOne(NzU, NrU) + ZU * RU * PB(k)%PO(i)%WQ

                Energy = PB(k)%PO(i)%Vr**2 + PB(k)%PO(i)%Vt**2 + PB(k)%PO(i)%Vz**2
                diag_local%DiagOne(k)%EnergyOne(NzL, NrL) = diag_local%DiagOne(k)%EnergyOne(NzL, NrL) + ZL * RL * PB(k)%PO(i)%WQ * Energy
                diag_local%DiagOne(k)%EnergyOne(NzL, NrU) = diag_local%DiagOne(k)%EnergyOne(NzL, NrU) + ZL * RU * PB(k)%PO(i)%WQ * Energy
                diag_local%DiagOne(k)%EnergyOne(NzU, NrL) = diag_local%DiagOne(k)%EnergyOne(NzU, NrL) + ZU * RL * PB(k)%PO(i)%WQ * Energy
                diag_local%DiagOne(k)%EnergyOne(NzU, NrU) = diag_local%DiagOne(k)%EnergyOne(NzU, NrU) + ZU * RU * PB(k)%PO(i)%WQ * Energy

                diag_local%DiagOne(k)%MPOne(NzL, NrL) = diag_local%DiagOne(k)%MPOne(NzL, NrL) + ZL * RL
                diag_local%DiagOne(k)%MPOne(NzL, NrU) = diag_local%DiagOne(k)%MPOne(NzL, NrU) + ZL * RU
                diag_local%DiagOne(k)%MPOne(NzU, NrL) = diag_local%DiagOne(k)%MPOne(NzU, NrL) + ZU * RL
                diag_local%DiagOne(k)%MPOne(NzU, NrU) = diag_local%DiagOne(k)%MPOne(NzU, NrU) + ZU * RU

                if (pic_type_implicit == pic_type) then
                    tmp_one = PB(k)%PO(i)
                    call MoveParticleOneElectrostaticImplicit(PB(k), tmp_one, FG)

                    diag_local%DiagOne(k)%JzOne(NzL, NrL) = diag_local%DiagOne(k)%JzOne(NzL, NrL) + ZL * RL * tmp_one%WQ * tmp_one%Vz
                    diag_local%DiagOne(k)%JzOne(NzL, NrU) = diag_local%DiagOne(k)%JzOne(NzL, NrU) + ZL * RU * tmp_one%WQ * tmp_one%Vz
                    diag_local%DiagOne(k)%JzOne(NzU, NrL) = diag_local%DiagOne(k)%JzOne(NzU, NrL) + ZU * RL * tmp_one%WQ * tmp_one%Vz
                    diag_local%DiagOne(k)%JzOne(NzU, NrU) = diag_local%DiagOne(k)%JzOne(NzU, NrU) + ZU * RU * tmp_one%WQ * tmp_one%Vz

                    diag_local%DiagOne(k)%JrOne(NzL, NrL) = diag_local%DiagOne(k)%JrOne(NzL, NrL) + ZL * RL * tmp_one%WQ * tmp_one%Vr
                    diag_local%DiagOne(k)%JrOne(NzL, NrU) = diag_local%DiagOne(k)%JrOne(NzL, NrU) + ZL * RU * tmp_one%WQ * tmp_one%Vr
                    diag_local%DiagOne(k)%JrOne(NzU, NrL) = diag_local%DiagOne(k)%JrOne(NzU, NrL) + ZU * RL * tmp_one%WQ * tmp_one%Vr
                    diag_local%DiagOne(k)%JrOne(NzU, NrU) = diag_local%DiagOne(k)%JrOne(NzU, NrU) + ZU * RU * tmp_one%WQ * tmp_one%Vr

                else if (pic_type_explicit == pic_type) then
                    diag_local%DiagOne(k)%JzOne(NzL, NrL) = diag_local%DiagOne(k)%JzOne(NzL, NrL) + ZL * RL * PB(k)%PO(i)%WQ * PB(k)%PO(i)%Vz
                    diag_local%DiagOne(k)%JzOne(NzL, NrU) = diag_local%DiagOne(k)%JzOne(NzL, NrU) + ZL * RU * PB(k)%PO(i)%WQ * PB(k)%PO(i)%Vz
                    diag_local%DiagOne(k)%JzOne(NzU, NrL) = diag_local%DiagOne(k)%JzOne(NzU, NrL) + ZU * RL * PB(k)%PO(i)%WQ * PB(k)%PO(i)%Vz
                    diag_local%DiagOne(k)%JzOne(NzU, NrU) = diag_local%DiagOne(k)%JzOne(NzU, NrU) + ZU * RU * PB(k)%PO(i)%WQ * PB(k)%PO(i)%Vz

                    diag_local%DiagOne(k)%JrOne(NzL, NrL) = diag_local%DiagOne(k)%JrOne(NzL, NrL) + ZL * RL * PB(k)%PO(i)%WQ * tmp_one%Vr
                    diag_local%DiagOne(k)%JrOne(NzL, NrU) = diag_local%DiagOne(k)%JrOne(NzL, NrU) + ZL * RU * PB(k)%PO(i)%WQ * tmp_one%Vr
                    diag_local%DiagOne(k)%JrOne(NzU, NrL) = diag_local%DiagOne(k)%JrOne(NzU, NrL) + ZU * RL * PB(k)%PO(i)%WQ * tmp_one%Vr
                    diag_local%DiagOne(k)%JrOne(NzU, NrU) = diag_local%DiagOne(k)%JrOne(NzU, NrU) + ZU * RU * PB(k)%PO(i)%WQ * tmp_one%Vr

                end if

            end do
        end do

        ! communication
        associate(zstart => FT%DM%CornerIndex(BOUNDARY_LEFT, 1), &
                  zend   => FT%DM%CornerIndex(BOUNDARY_RIGHT, 1), &
                  rstart => FT%DM%CornerIndex(BOUNDARY_LEFT, 2), &
                  rend   => FT%DM%CornerIndex(BOUNDARY_RIGHT, 2), &
                  zRightImageType => FT%DM%NeighborType(BOUNDARY_RIGHT, 1), &
                  rRightImageType => FT%DM%NeighborType(BOUNDARY_RIGHT, 2), &
                  image_shape => FT%DM%ImageShape)

            zend_on_bound = zend
            rend_on_bound = rend
            if (image_shape(1) > 1 .and. NEIGHBOR_TYPE_DOMAIN == zRightImageType) zend_on_bound = zend - 1
            if (image_shape(2) > 1 .and. NEIGHBOR_TYPE_DOMAIN == rRightImageType) rend_on_bound = rend - 1

            diag_local%Phi(zstart:zend_on_bound, rstart:rend_on_bound) = FT%Phi(zstart:zend_on_bound, rstart:rend_on_bound)
            diag_local%Chi(zstart:zend_on_bound, rstart:rend_on_bound) = FT%Chi(zstart:zend_on_bound, rstart:rend_on_bound)

            do k = 0, diag_local%Ns
                call MPI_ALLREDUCE(diag_local%DiagOne(k)%RhoOne, diag_global%DiagOne(k)%RhoOne, &
                                   diag_global%Nz*diag_global%Nr, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)

                call MPI_ALLREDUCE(diag_local%DiagOne(k)%EnergyOne, diag_global%DiagOne(k)%EnergyOne, &
                                   diag_global%Nz*diag_global%Nr, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)

                call MPI_ALLREDUCE(diag_local%DiagOne(k)%MPOne, diag_global%DiagOne(k)%MPOne, &
                                   diag_global%Nz*diag_global%Nr, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)

                call MPI_ALLREDUCE(diag_local%DiagOne(k)%JzOne, diag_global%DiagOne(k)%JzOne, &
                                   diag_global%Nz*diag_global%Nr, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)

                call MPI_ALLREDUCE(diag_local%DiagOne(k)%JrOne, diag_global%DiagOne(k)%JrOne, &
                                   diag_global%Nz*diag_global%Nr, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)

            end do

            call MPI_ALLREDUCE(diag_local%Phi, diag_global%Phi, &
                               diag_global%Nz*diag_global%Nr, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)
            call MPI_ALLREDUCE(diag_local%Chi, diag_global%Chi, &
                               diag_global%Nz*diag_global%Nr, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)

        end associate
    
        ! scale
        associate (zstart => 1, &
                   zend   => FG%DM%GlobalShape(1), &
                   rstart => 1, &
                   rend   => FG%DM%GlobalShape(2), &
                   dx     => FG%DM%SpaceStep(1))

            do k = 0, diag_global%Ns
                diag_global%DiagOne(k)%RhoOne = diag_global%DiagOne(k)%RhoOne * Geom%point_inv_volume_diag
                diag_global%DiagOne(k)%EnergyOne = diag_global%DiagOne(k)%EnergyOne * Geom%point_inv_volume_diag

                EnergyFactor = 0.5 * PB(k)%mass * PB(k)%Vfactor * PB(k)%Vfactor
                diag_global%DiagOne(k)%EnergyOne = diag_global%DiagOne(k)%EnergyOne * EnergyFactor

                diag_global%DiagOne(k)%JzOne = diag_global%DiagOne(k)%JzOne * Geom%point_inv_volume_diag * PB(k)%Charge * PB(k)%VFactor
                diag_global%DiagOne(k)%JrOne = diag_global%DiagOne(k)%JrOne * Geom%point_inv_volume_diag * PB(k)%Charge * PB(k)%VFactor

                do i = zstart, zend
                    do j = rstart, rend
                        if (diag_global%DiagOne(k)%RhoOne(i, j) > 0.d0) then
                            diag_global%DiagOne(k)%EnergyOne(i, j) = diag_global%DiagOne(k)%EnergyOne(i, j) / JtoeV / diag_global%DiagOne(k)%RhoOne(i, j)
                        end if
                    end do
                end do

                diag_global%DiagOne(k)%TOne = diag_global%DiagOne(k)%EnergyOne
            end do

        end associate

    end subroutine WeightingParticleField


    subroutine ZeroParticleFieldDiag(diag)
        class(ParticleFieldDiag), intent(inout) :: diag
        integer(4) :: k

        do k = 0, diag%Ns
            diag%DiagOne(k)%RhoOne      = 0.d0
            diag%DiagOne(k)%EnergyOne   = 0.d0
            diag%DiagOne(k)%TOne        = 0.d0
            diag%DiagOne(k)%JzOne       = 0.d0
            diag%DiagOne(k)%JrOne       = 0.d0
            diag%DiagOne(K)%MPOne       = 0.d0
        end do

        diag%Phi = 0.d0
        diag%Chi = 0.d0

    end subroutine ZeroParticleFieldDiag


    subroutine MultParticleFieldDiag(diag, val)
        class(ParticleFieldDiag), intent(inout) :: diag
        real(8), intent(in) :: val
        integer(4) :: k

        do k = 0, diag%Ns
            diag%DiagOne(k)%RhoOne      = val
            diag%DiagOne(k)%EnergyOne   = val
            diag%DiagOne(k)%TOne        = val
            diag%DiagOne(k)%JzOne       = val
            diag%DiagOne(k)%JrOne       = val
            diag%DiagOne(k)%MPOne       = val
        end do

        diag%Phi = val
        diag%Chi = val

    end subroutine MultParticleFieldDiag


    subroutine AddParticleFieldDiag(left, right)
        class(ParticleFieldDiag), intent(inout) :: left
        class(ParticleFieldDiag), intent(inout) :: right
        integer(4) :: k

        do k = 0, left%Ns
            left%DiagOne(k)%RhoOne      = left%DiagOne(k)%RhoOne        + right%DiagOne(k)%RhoOne
            left%DiagOne(k)%EnergyOne   = left%DiagOne(k)%EnergyOne     + right%DiagOne(k)%EnergyOne
            left%DiagOne(k)%TOne        = left%DiagOne(k)%TOne          + right%DiagOne(k)%TOne
            left%DiagOne(k)%JzOne       = left%DiagOne(k)%JzOne         + right%DiagOne(k)%JzOne
            left%DiagOne(k)%JrOne       = left%DiagOne(k)%JrOne         + right%DiagOne(k)%JrOne
            left%DiagOne(k)%MPOne       = left%DiagOne(k)%MPOne         + right%DiagOne(k)%MPOne
        end do

        left%Phi = left%Phi + right%Phi
        left%Chi = left%Chi + right%Chi

    end subroutine AddParticleFieldDiag


    subroutine NormParticleFieldDiag(diag)
        class(ParticleFieldDiag), intent(inout) :: diag
        integer(4) :: k

        do k = 0, diag%Ns
            diag%DiagOne(k)%RhoOne      = diag%DiagOne(k)%RhoOne        / dble(diag%Period)
            diag%DiagOne(k)%EnergyOne   = diag%DiagOne(k)%EnergyOne     / dble(diag%Period)
            diag%DiagOne(k)%TOne        = diag%DiagOne(k)%TOne          / dble(diag%Period)
            diag%DiagOne(k)%JzOne       = diag%DiagOne(k)%JzOne         / dble(diag%Period)
            diag%DiagOne(k)%JrOne       = diag%DiagOne(k)%JrOne         / dble(diag%Period)
            diag%DiagOne(k)%MPOne       = diag%DiagOne(k)%MPOne         / dble(diag%Period)
        end do

        diag%Phi = diag%Phi / dble(diag%Period)
        diag%Chi = diag%Chi / dble(diag%Period)

    end subroutine NormParticleFieldDiag


    subroutine initParticleFieldDiag(diag, CF)
        class(ParticleFieldDiag), intent(inout) :: diag
        type(ControlFlow), intent(in) :: CF
        integer(4) :: k

        diag%Nz = CF%DM%GlobalShape(1)
        diag%Nr = CF%DM%GlobalShape(2)
        diag%Ns = CF%Ns
        diag%Period = CF%Period

        call diag%destroy()

        allocate(diag%DiagOne(0:diag%Ns))
        do k = 0, diag%Ns
            allocate(diag%DiagOne(k)%RhoOne(diag%Nz, diag%Nr))
            allocate(diag%DiagOne(k)%EnergyOne(diag%Nz, diag%Nr))
            allocate(diag%DiagOne(k)%TOne(diag%Nz, diag%Nr))
            allocate(diag%DiagOne(k)%JzOne(diag%Nz, diag%Nr))
            allocate(diag%DiagOne(k)%JrOne(diag%Nz, diag%Nr))
            allocate(diag%DiagOne(k)%MPOne(diag%Nz, diag%Nr))
        end do

        allocate(diag%Phi(diag%Nz, diag%Nr))
        allocate(diag%Chi(diag%Nz, diag%Nr))

    end subroutine initParticleFieldDiag


    subroutine destroyParticleFieldDiag(diag)
        class(ParticleFieldDiag), intent(inout) :: diag
        integer(4) :: k

        if (allocated(diag%DiagOne)) then
            do k = 0, diag%Ns
                if (allocated(diag%DiagOne(k)%RhoOne))      deallocate(diag%DiagOne(k)%RhoOne)
                if (allocated(diag%DiagOne(k)%EnergyOne))   deallocate(diag%DiagOne(k)%EnergyOne)
                if (allocated(diag%DiagOne(k)%TOne))        deallocate(diag%DiagOne(k)%TOne)
                if (allocated(diag%DiagOne(k)%JzOne))       deallocate(diag%DiagOne(k)%JzOne)
                if (allocated(diag%DiagOne(k)%JrOne))       deallocate(diag%DiagOne(k)%JrOne)
                if (allocated(diag%DiagOne(k)%MPOne))       deallocate(diag%DiagOne(k)%MPOne)
            end do

            deallocate(diag%DiagOne)
        end if

        if (allocated(diag%Phi)) deallocate(diag%Phi)
        if (allocated(diag%Chi)) deallocate(diag%Chi)

    end subroutine destroyParticleFieldDiag

end Module DiagnosticsParticleField