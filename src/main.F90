program pic2d
    use mpi
    use ModuleOneStep
    use ModuleDiagStep

    implicit none

    integer(4) :: size, rank, ierr, i, j, k
    integer(4) :: par(0:NsMax)
    type(SimpleTimer) :: st

    call st%init()
    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

    call InitializeAllModules()
    call LoadAllModules()
    call DiagInitilalization(ControlFlowGlobal)
    call ShowModulesInfo()

    do i = 1+ControlFlowGlobal%Timer/ControlFlowGlobal%Period, ControlFlowGlobal%NRun
        do j = 1, ControlFlowGlobal%Period
            ControlFlowGlobal%Timer = ControlFlowGlobal%Timer + 1

            call OneStep()
            if (mod (i, 400) == 0) then
                call DiagOneStep(ParticleBundleLocal, FieldEMLocal, FieldOneLocal, FieldSourceLocal, GeometryGlobal)
            end if
        end do

        par(0:ControlFlowGlobal%Ns) = getParticleNum(ControlFlowGlobal%Ns, ParticleBundleLocal)
        call st%time()
        if (0 == rank) write(*, '(i6, 1x, f16.2, *(i12))') i, &
            st%getT(), par(0:ControlFlowGlobal%Ns), ParticleBundleLocal%NPar

        if (mod(i, 20) == 0) call DumpAllModules()
    end do

    call DiagOneStepFinal()
    call DumpAllModules()
    call ReleaseAllModules()
    call DiagReleaseAll()

    call MPI_FINALIZE(ierr)

end program pic2d
