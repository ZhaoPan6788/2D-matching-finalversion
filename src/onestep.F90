module ModuleOneStep
    use Constants
    use ModuleControlFlow
    use ModuleFieldEM
    use ModuleFieldOne
    use ModuleFieldSource
    use ModuleFieldSolver
    use ModuleFieldBoundary2DZR
    use ModuleSpecyOne
    use ModuleParticleBundle
    use ModuleParticleBoundary1D
    use ModuleMCCPublic
    use ModuleMCCInitialization
    use ModuleReactionOnePegasus
    use ModuleCircuitVoltageSource
    use ModuleFTime
    use json_module
    use ModuleInit
    use ModuleExtCircuit
    use ModuleParticleMaterial
    use ModulePICCommunication
    use ModuleMaterialsCommunication
    use ModuleRecombination
    use ModuleParticleRezoning

    implicit none

    type(ControlFlow), save                     :: ControlFlowGlobal
    type(FieldEM), save                         :: FieldEMLocal
    type(FieldOne), save, allocatable           :: FieldOneLocal(:)
    type(FieldSource), save                     :: FieldSourceLocal
    type(FieldSolver), save                     :: FieldSolverLocal
    type(FieldBoundary2DZR), save               :: FieldBoundaryGlobal

    type(ParticleBundle), save, allocatable     :: ParticleBundleLocal(:)
    type(Recombination2D), save                 :: RecombinationLocal

    type(Materials), save                       :: MaterialsGlobal
    type(Geometry), save                        :: GeometryGlobal
    type(ExtCircuits), save                     :: ExtCircuitsGlobal

    type(PICCom2D), save                        :: pic_com

    real(8), save                               :: density_init = 0.d0

contains

    subroutine InitializeAllModules()
        integer(4) :: k
        type(json_file) :: json_para
        character(len=99) :: jsonFilename
        logical :: alive
        integer(4) :: size, rank, ierr
        type(SimpleTimer) :: st_init

        call st_init%init()
        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

        ! Necessary Modules
        call random_seed()
        call InitFileName()

        ! read config from json file
        jsonFilename = "./config.json"
        inquire(file=jsonFilename, exist=alive)
        if (alive) then
            call json_para%initialize()
            call json_para%load(jsonFilename)
            if (0 == rank) write(*, *) "Load config.json."

        else
            if (0 == rank) write(*, '(a, a, a)') "The ", jsonFilename, " is not found."
            stop

        end if

        ! Control Flow
        call InitControlFlowJson(json_para, ControlFlowGlobal)

        ! materials and geometry
        call initGeometryAndMaterialsJson(json_para, ControlFlowGlobal, MaterialsGlobal, GeometryGlobal)

        ! Field
        call InitFieldJson(json_para, ControlFlowGlobal, FieldEMLocal, FieldSourceLocal, FieldBoundaryGlobal)

        ! Gas
        call InitGasJson(json_para, ControlFlowGlobal)

        ! particle
        call InitParticleJson(json_para, ControlFlowGlobal, ParticleBundleLocal, FieldOneLocal, GeometryGlobal)

        ! circuit
        call InitCircuitJson(json_para, ControlFlowGlobal, ExtCircuitsGlobal)

        ! field solver
        call InitFieldSolverJson(ControlFlowGlobal, FieldSolverLocal, FieldSourceLocal, ExtCircuitsGlobal, FieldBoundaryGlobal, MaterialsGlobal, GeometryGlobal)

        ! collision
        call MCCBundleInit(ControlFlowGlobal, SpecyGlobal, GasGlobal)

        ! pic com
        call pic_com%init(ControlFlowGlobal%DM%LocalShape(1), &
                          ControlFlowGlobal%DM%LocalShape(2), &
                          ControlFlowGlobal%DM%ImageShape(1), &
                          ControlFlowGlobal%DM%ImageShape(2), com_type=com_type_nonblock)

        call InitRecombinationJson(json_para, ControlFlowGlobal, RecombinationLocal)

        call calculateCapacitance(FieldSolverLocal, FieldSourceLocal, pic_com, MaterialsGlobal, ExtCircuitsGlobal, GeometryGlobal)

        call st_init%time()
        if (0 == rank) then
            write(*, *) ''
            write(*, '(1x, a, f16.2)') 'Init time : ', st_init%getT()
        end if

    end subroutine InitializeAllModules


    subroutine OneStep()
        integer(4) :: k

        associate(zstart => ControlFlowGlobal%DM%CornerIndex(BOUNDARY_LEFT, 1), &
                 zend => ControlFlowGlobal%DM%CornerIndex(BOUNDARY_RIGHT, 1), &
                 rstart => ControlFlowGlobal%DM%CornerIndex(BOUNDARY_LEFT, 2), &
                 rend => ControlFlowGlobal%DM%CornerIndex(BOUNDARY_RIGHT, 2))

            if (load_type_vacuum == load_type) then
                do k = 0, ControlFlowGlobal%Ns
                    ParticleBundleLocal(k)%NPar = 0
                end do
            end if

            ! ! particle velocity limit
            ! do k = 0, ControlFlowGlobal%Ns
            !     call ParticleBundleLocal(k)%VelLimit(0.75d0, 0.25d0)
            ! end do

            ! particle move
            do k = 0, ControlFlowGlobal%Ns
                call ParticleBundleLocal(k)%MoveES(FieldEMLocal)
                call MaterialAborptionParticle(ControlFlowGlobal, GeometryGlobal, MaterialsGlobal, ParticleBundleLocal(k))
            end do

            call CommunicationMetalCharge(MaterialsGlobal)
            call CommunicationDielectricSigma(MaterialsGlobal)

            ! particle boundary and communication
            do k = 0, ControlFlowGlobal%Ns
                call pic_com%comp(ParticleBundleLocal(k), dble(zstart-1), dble(zend-1), dble(rstart-1), dble(rend-1))
            end do

            ! particle weight
            do k = 0, ControlFlowGlobal%Ns
                call FieldOneLocal(k)%P2CES(ParticleBundleLocal(k), GeometryGlobal)
            end do

            ! particle split and merge
            call particle_split_merge_control(ControlFlowGlobal, ParticleBundleLocal, GeometryGlobal, ppg_dif)

            ! density field communication
            call FieldSourceLocal%Sum(FieldOneLocal)
            call pic_com%comf(FieldSourceLocal%Rho, com_field_opt_sum, zstart, zend, rstart, rend)
            call pic_com%comf(FieldSourceLocal%Chi, com_field_opt_sum, zstart, zend, rstart, rend)

            ! field solver
            if (pic_type_implicit == pic_type) then
                call calculateCapacitance(FieldSolverLocal, FieldSourceLocal, pic_com, MaterialsGlobal, ExtCircuitsGlobal, GeometryGlobal)
            end if

            call FieldSolverLocal%SolveV(ControlFlowGlobal, FieldSourceLocal, ExtCircuitsGlobal, FieldBoundaryGlobal, MaterialsGlobal, GeometryGlobal)
            call pic_com%comf(FieldSolverLocal%Solve, com_field_opt_sum, zstart, zend, rstart, rend)
            FieldSourceLocal%Phi(zstart:zend, rstart:rend) = FieldSolverLocal%Solve(zstart:zend, rstart:rend)
            call pic_com%comf(FieldSourceLocal%Phi, com_field_opt_ext, zstart, zend, rstart, rend)
    
            call FieldSourceLocal%SetE(FieldEMLocal)

            ! Monte Carlo (none-collision method)
            call MCC(ControlFlowGlobal%Ns, ControlFlowGlobal%Ng, ParticleBundleLocal, SpecyGlobal, GasGlobal, MCCBundleGlobal)

            ! Recombination
            if (is_open_recombination .and. &
                mod(ControlFlowGlobal%Timer, recombination_time_step_number) == 0) &
            call RecombinationLocal%recb(ControlFlowGlobal%Ns, ParticleBundleLocal, dble(recombination_time_step_number) * ControlFlowGlobal%dt)

            ! call changeWeight(ControlFlowGlobal%Ns, ParticleBundleLocal, FieldOneLocal)
        end associate

    end subroutine OneStep


    subroutine ShowModulesInfo()
        integer(4) :: size, rank, ierr

        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
        call MPI_BARRIER(MPI_COMM_WORLD, ierr)

        if (0 == rank) write(*, *) '-------------------------Simulation Configure-------------------------'
        call ControlFlowGlobal%Show()

        if (0 == rank) then
            write(*, '(a30, es12.4)') 'init density:', init_density
            write(*, '(a30, i)') 'ppg:', ppg

            if (pic_type_explicit == pic_type) write(*, '(a30, a12)') 'pic:', 'explicit'
            if (pic_type_implicit == pic_type) write(*, '(a30, a12)') 'pic:', 'implicit'

            if (load_type_vacuum == load_type) write(*, '(a30, a12)') 'load:', 'vacuum'
            if (load_type_plasma == load_type) write(*, '(a30, a12)') 'load:', 'plasma'

            write(*, *) ''
        end if

        call ControlFlowGlobal%DM%Show()
        call ShowGas(ControlFlowGlobal)

        if (0 == rank .and. is_open_recombination) then
            write(*, *) 'Recombination:'
            call RecombinationLocal%show()

        end if

        call MaterialsGlobal%show()
        call ExtCircuitsGlobal%show()

    end subroutine ShowModulesInfo


    subroutine DumpAllModules()
        integer(4) :: size, rank, ierr, k

        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
        call MPI_BARRIER(MPI_COMM_WORLD, ierr)

        if (0 == rank) write(*, *) ''
        call ControlFlowGlobal%Dump()
        call ControlFlowGlobal%DM%Dump()

        call FieldEMLocal%Dump()

        do k = 0, ControlFlowGlobal%Ns
            call ParticleBundleLocal(k)%Dump()
        end do

        call ExtCircuitsGlobal%Dump()
        call MaterialsGlobal%dump()

    end subroutine DumpAllModules


    subroutine LoadAllModules()
        integer(4) :: size, rank, ierr, k

        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
        call MPI_BARRIER(MPI_COMM_WORLD, ierr)

        if (.not. is_restart) then
            if (0 == rank) write(*, *) ''
            call ControlFlowGlobal%Load()
            call ControlFlowGlobal%DM%Load()

            call FieldEMLocal%Load()

            do k = 0, ControlFlowGlobal%Ns
                call ParticleBundleLocal(k)%Load()
            end do

            call ExtCircuitsGlobal%Load()
            call MaterialsGlobal%load()
            if (0 == rank) write(*, *) ''
        end if

    end subroutine LoadAllModules


    subroutine ReleaseAllModules()
        integer(4) :: k, ierr

        call MPI_BARRIER(MPI_COMM_WORLD, ierr)

        call ControlFlowGlobal%Destroy()

        call FieldEMLocal%Destroy()
        call FieldSourceLocal%Destroy()
        call FieldSolverLocal%Destroy()
        call FieldBoundaryGlobal%Destroy()
        call ExtCircuitsGlobal%destroy()

        do k = 0, ControlFlowGlobal%Ns
            call ParticleBundleLocal(k)%Destroy()
            call FieldOneLocal(k)%Destroy()
        end do

        if (Allocated(ParticleBundleLocal))     Deallocate(ParticleBundleLocal)
        if (Allocated(FieldOneLocal))           Deallocate(FieldOneLocal)

        call MaterialsGlobal%destroy()
        call GeometryGlobal%destroy()
        call DestroyGas()
        call pic_com%destroy()

        call MCCBundleRelease()
        if (is_open_recombination) call RecombinationLocal%destroy()

    end subroutine ReleaseAllModules


    function getParticleNum(Ns, PB)
        integer(4), intent(in) :: Ns
        Type(ParticleBundle), intent(inout) :: PB(0:Ns)
        integer(4) :: getParticleNum(0:Ns)
        integer(4) :: npar_send(0:Ns), npar_recv(0:Ns)
        integer(4) :: k, ierr

        do k = 0, Ns
            npar_send(k) = PB(k)%NPar
        end do

        call MPI_ALLREDUCE(npar_send, npar_recv, Ns+1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)

        do k = 0, Ns
            getParticleNum(k) = npar_recv(k)
        end do

    end


    subroutine calculateCapacitance(FS, FT, com, MT, Ext, Geom)
        type(FieldSolver), intent(inout) :: FS
        type(FieldSource), intent(inout) :: FT
        type(PICCom2D), intent(inout) :: com
        type(Materials), intent(inout) :: MT
        type(ExtCircuits), intent(inout) :: Ext
        type(Geometry), intent(inout) :: Geom
        integer(4) :: i, ierr
        real(8), allocatable :: Ez(:, :), Er(:, :), volume(:, :)
        real(8), allocatable :: cap_energy_send(:), cap_energy_recv(:)

        if (pic_type_explicit == pic_type) then
            call calculateCapacitanceExplicit(FS, com, MT, Ext, Geom)
        
        else if (pic_type_implicit == pic_type) then
            call calculateCapacitanceImplicit(FS, FT, com, MT, Ext, Geom)
        
        end if

    end subroutine calculateCapacitance


    subroutine calculateCapacitanceExplicit(FS, com, MT, Ext, Geom)
        type(FieldSolver), intent(inout) :: FS
        type(PICCom2D), intent(inout) :: com
        type(Materials), intent(inout) :: MT
        type(ExtCircuits), intent(inout) :: Ext
        type(Geometry), intent(inout) :: Geom
        integer(4) :: i, ierr
        real(8), allocatable :: Ez(:, :), Er(:, :), volume(:, :)
        real(8), allocatable :: cap_energy_send(:), cap_energy_recv(:)

        associate(zstart => FS%DM%CornerIndex(BOUNDARY_LEFT, 1), &
                 zend => FS%DM%CornerIndex(BOUNDARY_RIGHT, 1), &
                 rstart => FS%DM%CornerIndex(BOUNDARY_LEFT, 2), &
                 rend => FS%DM%CornerIndex(BOUNDARY_RIGHT, 2), &
                 dz  => FS%DM%SpaceStep(1), dr => FS%DM%SpaceStep(2))

            if (MT%metal_count > 0) then
                allocate(Ez(zstart:zend-1, rstart:rend))
                allocate(Er(zstart:zend, rstart:rend-1))
                allocate(volume(zstart:zend-1, rstart:rend-1))
                allocate(cap_energy_send(MT%metal_count))
                allocate(cap_energy_recv(MT%metal_count))

                cap_energy_send = 0.d0                
                do i = rstart, rend-1
                    volume(:, i) = PI * dr * dr * dz * dble(i*i - (i-1)*(i-1))
                end do

                do i = 1, MT%metal_count
                    if (Ext%circuits_type(i) /= circuit_type_digital_ground) then
                        FS%Solve = FS%laplace(:, :, i)
                        call com%comf(FS%Solve, com_field_opt_sum, zstart, zend, rstart, rend)
                        Ez = (FS%Solve(zstart:zend-1, rstart:rend) - FS%Solve(zstart+1:zend, rstart:rend)) / dz
                        Er = (FS%Solve(zstart:zend, rstart:rend-1) - FS%Solve(zstart:zend, rstart+1:rend)) / dr

                        cap_energy_send(i) = 2.d0 * sum(volume * 0.5d0 * Epsilon * &
                            Geom%cell_epsilon(zstart:zend-1, rstart:rend-1) * &
                            ((0.5d0 * (Ez(zstart:zend-1, rstart:rend-1) + Ez(zstart:zend-1, rstart+1:rend))) ** 2 + &
                            (0.5d0 * (Er(zstart:zend-1, rstart:rend-1) + Er(zstart+1:zend, rstart:rend-1))) ** 2))

                    end if
                end do

                call MPI_ALLREDUCE(cap_energy_send, cap_energy_recv, MT%metal_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)

                do i = 1, MT%metal_count
                    MT%metls(i)%capacitance = cap_energy_recv(i)
                end do

                ! if (0 == FS%DM%MyId) write(*, *) MT%metls%capacitance

                deallocate(Ez)
                deallocate(Er)
                deallocate(volume)
                deallocate(cap_energy_send)
                deallocate(cap_energy_recv)
            end if

        end associate

    end subroutine calculateCapacitanceExplicit


    subroutine calculateCapacitanceImplicit(FS, FT, com, MT, Ext, Geom)
        type(FieldSolver), intent(inout) :: FS
        type(FieldSource), intent(inout) :: FT
        type(PICCom2D), intent(inout) :: com
        type(Materials), intent(inout) :: MT
        type(ExtCircuits), intent(inout) :: Ext
        type(Geometry), intent(inout) :: Geom
        integer(4) :: i, ierr
        real(8), allocatable :: Ez(:, :), Er(:, :), volume(:, :)
        real(8), allocatable :: cap_energy_send(:), cap_energy_recv(:)

        associate(zstart => FS%DM%CornerIndex(BOUNDARY_LEFT, 1), &
                 zend => FS%DM%CornerIndex(BOUNDARY_RIGHT, 1), &
                 rstart => FS%DM%CornerIndex(BOUNDARY_LEFT, 2), &
                 rend => FS%DM%CornerIndex(BOUNDARY_RIGHT, 2), &
                 dz  => FS%DM%SpaceStep(1), dr => FS%DM%SpaceStep(2))

            if (MT%metal_count > 0) then
                allocate(Ez(zstart:zend-1, rstart:rend))
                allocate(Er(zstart:zend, rstart:rend-1))
                allocate(volume(zstart:zend-1, rstart:rend-1))
                allocate(cap_energy_send(MT%metal_count))
                allocate(cap_energy_recv(MT%metal_count))

                cap_energy_send = 0.d0                
                do i = rstart, rend-1
                    volume(:, i) = PI * dr * dr * dz * dble(i*i - (i-1)*(i-1))
                end do

                do i = 1, MT%metal_count
                    if (Ext%circuits_type(i) /= circuit_type_digital_ground) then
                        FS%Solve = FS%laplace(:, :, i)
                        call com%comf(FS%Solve, com_field_opt_sum, zstart, zend, rstart, rend)
                        Ez = (FS%Solve(zstart:zend-1, rstart:rend) - FS%Solve(zstart+1:zend, rstart:rend)) / dz
                        Er = (FS%Solve(zstart:zend, rstart:rend-1) - FS%Solve(zstart:zend, rstart+1:rend)) / dr

                        cap_energy_send(i) = 2.d0 * sum(volume * 0.5d0 * Epsilon * &
                            (1.d0 + 0.25d0 * (FT%Chi(zstart:zend-1, rstart:rend-1) + &
                            FT%Chi(zstart+1:zend, rstart:rend-1) + &
                            FT%Chi(zstart:zend-1, rstart+1:rend) + &
                            FT%Chi(zstart+1:zend, rstart+1:rend))) * &
                            Geom%cell_epsilon(zstart:zend-1, rstart:rend-1) * &
                            ((0.5d0 * (Ez(zstart:zend-1, rstart:rend-1) + Ez(zstart:zend-1, rstart+1:rend))) ** 2 + &
                            (0.5d0 * (Er(zstart:zend-1, rstart:rend-1) + Er(zstart+1:zend, rstart:rend-1))) ** 2))

                    end if
                end do

                call MPI_ALLREDUCE(cap_energy_send, cap_energy_recv, MT%metal_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)

                do i = 1, MT%metal_count
                    MT%metls(i)%capacitance = cap_energy_recv(i)
                end do

                ! if (0 == FS%DM%MyId) write(*, *) MT%metls%capacitance

                deallocate(Ez)
                deallocate(Er)
                deallocate(volume)
                deallocate(cap_energy_send)
                deallocate(cap_energy_recv)
            end if

        end associate

    end subroutine calculateCapacitanceImplicit

end module ModuleOneStep