Module DiagnosticsEDF
    use mpi
    use Constants
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

    integer(4), parameter :: NeMax = 500
    integer(4), save :: diag_edf_num = 0
    real(8), allocatable, save :: edf_limit(:, :)
    type(GridControlFlow), save :: edf_grid

    contains

        subroutine DiagEDFInitilalization(CF)
            class(ControlFlow), intent(in) :: CF

            if (diag_edf_num > 0) then
                call DiagEDFRelease()

                edf_grid%Sim_Mode = SIM_MODE_EVOLUTION
                call edf_grid%Init('DiagEDF', ExtensionMode=FILE_EXTENSION_MODE_H5, &
                                    ParallelMode=FILE_PARALLEL_MODE_CLOSE, &
                                    DynamicIndex=FILE_DYNAMIC_MODE_OPEN, &
                                    Nx1=(NeMax*2+1) , Nx2=1, Ns= (4*diag_edf_num+1)*(CF%Ns + 1))
            end if

        end subroutine DiagEDFInitilalization


        subroutine DiagEDFPeriod(CF, PB, bd)
            class(ControlFlow), intent(in) :: CF
            type(ParticleBundle), intent(in) :: PB(0:CF%Ns)
            type(ParticleBundle), intent(inout) :: bd(0:CF%Ns)
            real(8), allocatable :: edf_local(:, :), edf_global(:, :)
            real(8), allocatable :: vtmax_local(:, :), vtmax_global(:, :)
            real(8), allocatable :: vzmax_local(:, :), vzmax_global(:, :)
            real(8), allocatable :: vrmax_local(:, :), vrmax_global(:, :)
            real(8) :: EnergyInterval
            integer(4) :: k, j, ierr

            if (diag_edf_num > 0) then

                allocate(edf_local(0:2*NeMax, diag_edf_num+1))
                allocate(edf_global(0:2*NeMax, diag_edf_num+1))

                allocate(vtmax_local(0:2*NeMax, diag_edf_num))
                allocate(vtmax_global(0:2*NeMax, diag_edf_num))
                allocate(vzmax_local(0:2*NeMax, diag_edf_num))
                allocate(vzmax_global(0:2*NeMax, diag_edf_num))
                allocate(vrmax_local(0:2*NeMax, diag_edf_num))
                allocate(vrmax_global(0:2*NeMax, diag_edf_num))

                do k = 0, CF%Ns

                    edf_local = 0.d0
                    edf_global = 0.d0

                    vtmax_local = 0.d0
                    vtmax_global = 0.d0
                    vzmax_local = 0.d0
                    vzmax_global = 0.d0
                    vrmax_local = 0.d0
                    vrmax_global = 0.d0

                    if (k == 0) then
                        EnergyInterval = 0.1d0
                    else
                        EnergyInterval = 1.d0
                    end if

                    do j = 1, diag_edf_num
                        call WeightingParticleEDF(CF, PB(k), edf_local(:, j), edf_limit(j, :), EnergyInterval, vtmax_local(:, j), vzmax_local(:, j), vrmax_local(:, j))
                    end do

                    bd(k)%Mass = PB(k)%Mass
                    bd(k)%VFactor = PB(k)%VFactor

                    call WeightingParticleEDF2(CF, bd(k), edf_local(:, diag_edf_num+1), EnergyInterval)

                    call MPI_ALLREDUCE(edf_local, edf_global, (2*NeMax+1)*(diag_edf_num+1), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)
                    call MPI_ALLREDUCE(vtmax_local, vtmax_global, (2*NeMax+1)*diag_edf_num, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)
                    call MPI_ALLREDUCE(vzmax_local, vzmax_global, (2*NeMax+1)*diag_edf_num, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)
                    call MPI_ALLREDUCE(vrmax_local, vrmax_global, (2*NeMax+1)*diag_edf_num, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)

                    if (0 == CF%DM%MyId) then

                        do j = 1, diag_edf_num
                            call edf_grid%Update(edf_global(:, j:j), "EDF"//"-"//trim(SpecyGlobal(k)%Name)//'-'//trim(num2str(j, 3)))
                            call edf_grid%Update(vtmax_global(:, j:j), "vtmax"//"-"//trim(SpecyGlobal(k)%Name)//'-'//trim(num2str(j, 3)))
                            call edf_grid%Update(vzmax_global(:, j:j), "vzmax"//"-"//trim(SpecyGlobal(k)%Name)//'-'//trim(num2str(j, 3)))
                            call edf_grid%Update(vrmax_global(:, j:j), "vrmax"//"-"//trim(SpecyGlobal(k)%Name)//'-'//trim(num2str(j, 3)))
                        end do

                        call edf_grid%Update(edf_global(:, diag_edf_num+1:diag_edf_num+1), "EDF"//"-"//trim(SpecyGlobal(k)%Name)//'-bd')
                        
                    end if
                end do

                deallocate(edf_local)
                deallocate(edf_global)
                deallocate(vtmax_local)
                deallocate(vtmax_global)
                deallocate(vzmax_local)
                deallocate(vzmax_global)
                deallocate(vrmax_local)
                deallocate(vrmax_global)
            
            end if

        end subroutine DiagEDFPeriod

        subroutine DiagEDFRelease()

            call edf_grid%Destroy()

        end subroutine DiagEDFRelease

        subroutine WeightingParticleEDF(CF, PB, pedf, particle_limit, EnergyInterval, ptmax, pzmax, prmax)
            class(ControlFlow), intent(in) :: CF
            type(ParticleBundle), intent(in) :: PB
            real(8), intent(inout) :: pedf(0:2*NeMax)
            real(8), intent(in) :: particle_limit(4)
            real(8), intent(in) :: EnergyInterval
            real(8), intent(inout) :: ptmax(0:2*NeMax)
            real(8), intent(inout) :: pzmax(0:2*NeMax)
            real(8), intent(inout) :: prmax(0:2*NeMax)
            real(8) :: Energy, Energyt, Energyvz, Energyvr
            integer(4) :: i, N

            associate (zstart => CF%DM%CornerIndex(BOUNDARY_LEFT, 1), &
                       zend => CF%DM%CornerIndex(BOUNDARY_RIGHT, 1), &
                       rstart => CF%DM%CornerIndex(BOUNDARY_LEFT, 2), &
                       rend => CF%DM%CornerIndex(BOUNDARY_RIGHT, 2))

                if (dble(zend-1)   <= particle_limit(1) .or. &
                    dble(zstart-1) >= particle_limit(2) .or. &
                    dble(rend-1)   <= particle_limit(3) .or. &
                    dble(rstart-1) >= particle_limit(4)) then

                    pedf = 0.d0
                    ptmax = 0.d0
                    pzmax = 0.d0
                    prmax = 0.d0

                else

                    pedf = 0.d0
                    ptmax = 0.d0
                    pzmax = 0.d0
                    prmax = 0.d0

                    do i = 1, PB%Npar
                        if (PB%PO(i)%Z >= particle_limit(1) .and. &
                            PB%PO(i)%Z <= particle_limit(2) .and. &
                            PB%PO(i)%R >= particle_limit(3) .and. &
                            PB%PO(i)%R <= particle_limit(4)) then

                            Energy = PB%PO(i)%Energy(PB%Mass, PB%VFactor) / JtoeV
                            N = ceiling(Energy/EnergyInterval)
                            if (N >= 0 .and. N <= 2*NeMax) then
                                pedf(N) = pedf(N) + PB%PO(i)%WQ
                            end if

                            Energyt =  PB%PO(i)%Energyvt(PB%Mass, PB%VFactor) / JtoeV
                            N = ceiling(Energyt/EnergyInterval) + 500
                            if (N >= 0 .and. N <= 2*NeMax) then
                                ptmax(N) = ptmax(N) + PB%PO(i)%WQ
                            end if

                            Energyvz =  PB%PO(i)%Energyvz(PB%Mass, PB%VFactor) / JtoeV
                            N = ceiling(Energyvz/EnergyInterval) + 500
                            if (N >= 0 .and. N <= 2*NeMax) then
                                pzmax(N) = pzmax(N) + PB%PO(i)%WQ
                            end if

                            Energyvr =  PB%PO(i)%Energyvr(PB%Mass, PB%VFactor) / JtoeV
                            N = ceiling(Energyvr/EnergyInterval) + 500
                            if (N >= 0 .and. N <= 2*NeMax) then
                                prmax(N) = prmax(N) + PB%PO(i)%WQ
                            end if

                        end if
                    end do
                end if
            end associate

        end subroutine WeightingParticleEDF


        subroutine WeightingParticleEDF2(CF, PB, pedf, EnergyInterval)
            class(ControlFlow), intent(in) :: CF
            type(ParticleBundle), intent(in) :: PB
            real(8), intent(inout) :: pedf(NeMax)
            real(8), intent(in) :: EnergyInterval
            real(8) :: Energy
            integer(4) :: i, N

            pedf = 0.d0
            do i = 1, PB%Npar
                Energy = PB%PO(i)%Energy(PB%Mass, PB%VFactor) / JtoeV
                N = ceiling(Energy/EnergyInterval)
                if (N >= 1 .and. N <= NeMax) then
                    pedf(N) = pedf(N) + PB%PO(i)%WQ
                end if
            end do

        end subroutine WeightingParticleEDF2

end Module DiagnosticsEDF