program fortran_mpi
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

    implicit none

    type(ControlFlow), save                     :: ControlFlowGlobal
    type(FieldEM), save                         :: FieldEMLocal
    type(FieldOne), save, allocatable           :: FieldOneLocal(:)
    type(FieldSource), save                     :: FieldSourceLocal
    type(FieldSolver), save                     :: FieldSolverLocal
    type(FieldBoundary2DZR), save               :: FieldBoundaryGlobal

    type(ParticleBundle), save, allocatable     :: ParticleBundleLocal(:)

    type(Materials), save                       :: MaterialsGlobal
    type(Geometry), save                        :: GeometryGlobal
    type(ExtCircuits), save                     :: ExtCircuitsGlobal

    type(PICCom2D), save                        :: pic_com

    real(8), save                               :: density_init = 0.d0

    integer(4) :: size, rank, ierr, i, j, k
    integer(4) :: par(0:100)
    type(SimpleTimer) :: st
    type(json_file) :: json_para
    character(len=99) :: jsonFilename
    logical :: alive

    ! init mpi env
    call st%init()
    call MPI_INIT(ierr)
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

    ! collision
    call MCCBundleInit(ControlFlowGlobal, SpecyGlobal, GasGlobal)

    ! field solver
    call InitFieldSolverJson(ControlFlowGlobal, FieldSolverLocal, FieldSourceLocal, ExtCircuitsGlobal, FieldBoundaryGlobal, MaterialsGlobal, GeometryGlobal)

    ! pic com
    call pic_com%init(ControlFlowGlobal%DM%LocalShape(1), &
                        ControlFlowGlobal%DM%LocalShape(2), &
                        ControlFlowGlobal%DM%ImageShape(1), &
                        ControlFlowGlobal%DM%ImageShape(2), com_type=com_type_nonblock)

    call st%time()
    if (0 == rank) then
        write(*, *) ''
        write(*, '(1x, a, f16.2)') 'Init time : ', st%getT()
    end if

    ! show all info
    call MPI_BARRIER(MPI_COMM_WORLD, ierr)

    if (0 == rank) write(*, *) '-------------------------Simulation Configure-------------------------'
    call ControlFlowGlobal%Show()

    if (0 == rank) then
        write(*, '(a30, es12.4)') 'init density:', init_density
        write(*, '(a30, i)') 'ppg:', ppg
        write(*, *) ''
    end if

    call ControlFlowGlobal%DM%Show()
    call ShowGas(ControlFlowGlobal)
    call MaterialsGlobal%show()
    call ExtCircuitsGlobal%show()

    ! one step
    associate(zstart => ControlFlowGlobal%DM%CornerIndex(BOUNDARY_LEFT, 1), &
             zend => ControlFlowGlobal%DM%CornerIndex(BOUNDARY_RIGHT, 1), &
             rstart => ControlFlowGlobal%DM%CornerIndex(BOUNDARY_LEFT, 2), &
             rend => ControlFlowGlobal%DM%CornerIndex(BOUNDARY_RIGHT, 2))

        do i = 1, 10
            ControlFlowGlobal%Timer = ControlFlowGlobal%Timer + 1
            ! dump
            do k = 0, ControlFlowGlobal%Ns
                call ParticleBundleLocal(k)%Diag(trim(SpecyGlobal(k)%Name)//'_raw_'//trim(num2str(i, 3)))
            end do

            ! particle move
            do k = 0, ControlFlowGlobal%Ns
                call ParticleBundleLocal(k)%MoveES(FieldEMLocal)
            end do

            ! dump
            do k = 0, ControlFlowGlobal%Ns
                call ParticleBundleLocal(k)%Diag(trim(SpecyGlobal(k)%Name)//'_mov_'//trim(num2str(i, 3)))
            end do

            ! abort
            do k = 0, ControlFlowGlobal%Ns
                call MaterialAborptionParticle(ControlFlowGlobal, GeometryGlobal, MaterialsGlobal, ParticleBundleLocal(k))
            end do

            ! dump
            do k = 0, ControlFlowGlobal%Ns
                call ParticleBundleLocal(k)%Diag(trim(SpecyGlobal(k)%Name)//'_abt_'//trim(num2str(i, 3)))
            end do

            call CommunicationMetalCharge(MaterialsGlobal)
            call CommunicationDielectricSigma(MaterialsGlobal)

            ! particle boundary and communication
            do k = 0, ControlFlowGlobal%Ns
                call pic_com%comp(ParticleBundleLocal(k), dble(zstart-1), dble(zend-1), dble(rstart-1), dble(rend-1))
            end do

            ! dump
            do k = 0, ControlFlowGlobal%Ns
                call ParticleBundleLocal(k)%Diag(trim(SpecyGlobal(k)%Name)//'_com_'//trim(num2str(i, 3)))
            end do

            ! particle weight
            do k = 0, ControlFlowGlobal%Ns
                call FieldOneLocal(k)%P2CES(ParticleBundleLocal(k), GeometryGlobal)
            end do

            call FieldSourceLocal%Sum(FieldOneLocal)
            call pic_com%comf(FieldSourceLocal%Rho, com_field_opt_sum, zstart, zend, rstart, rend)
            call pic_com%comf(FieldSourceLocal%Chi, com_field_opt_sum, zstart, zend, rstart, rend)

            call FieldSourceLocal%Diag('source_'//trim(num2str(i, 3)))

            ! field solver
            MaterialsGlobal%metls%capacitance = 1.d-12
            call FieldSolverLocal%SolveV(ControlFlowGlobal, FieldSourceLocal, ExtCircuitsGlobal, FieldBoundaryGlobal, MaterialsGlobal, GeometryGlobal)
            call FieldSolverLocal%Diag('solver_'//trim(num2str(i, 3)))

            call pic_com%comf(FieldSolverLocal%Solve, com_field_opt_sum, zstart, zend, rstart, rend)

            FieldSourceLocal%Phi(zstart:zend, rstart:rend) = FieldSolverLocal%Solve(zstart:zend, rstart:rend)
            call pic_com%comf(FieldSourceLocal%Phi, com_field_opt_ext, zstart, zend, rstart, rend)

            call FieldSourceLocal%SetE(FieldEMLocal)

            call FieldSolverLocal%Diag('solver2_'//trim(num2str(i, 3)))
            call FieldSourceLocal%Diag('source2_'//trim(num2str(i, 3)))

            ! Monte Carlo (none-collision method)
            Call MCC(ControlFlowGlobal%Ns, ControlFlowGlobal%Ng, ParticleBundleLocal, SpecyGlobal, GasGlobal, MCCBundleGlobal)
        
            if (0 == rank) write(*, *) i
        end do
    end associate

    ! release all modules
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

    call DestroyGas()
    call st%show()

    call MPI_FINALIZE(ierr)

end program fortran_mpi