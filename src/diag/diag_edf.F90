Module DiagnosticsEDF
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

    integer(4), parameter :: NeMax = 500
    integer(4), save :: diag_edf_num = 1
    ! real(8), allocatable, save :: edf_limit(:, :)
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
                                    Nx1=NeMax, Nx2=1, Ns=(diag_edf_num+1)*(CF%Ns + 1))
            end if

        end subroutine DiagEDFInitilalization


        subroutine DiagEDFPeriod(CF, PB, bd)
            class(ControlFlow), intent(in) :: CF
            type(ParticleBundle), intent(in) :: PB(0:CF%Ns)
            type(ParticleBundle), intent(inout) :: bd(0:CF%Ns)
            real(8), allocatable :: edf_local(:, :), edf_global(:, :)
            real(8) :: EnergyInterval
            integer(4) :: k, j, ierr

            if (diag_edf_num > 0) then
                allocate(edf_local(NeMax, diag_edf_num+1))
                allocate(edf_global(NeMax, diag_edf_num+1))

                do k = 0, CF%Ns
                    edf_local = 0.d0
                    edf_global = 0.d0
                    if (k == 0) then
                        EnergyInterval = 0.1d0
                    else
                        EnergyInterval = 0.5d0
                    end if

                    do j = 1, diag_edf_num
                        ! call WeightingParticleEDF(CF, PB(k), edf_local(:, j), edf_limit(j, :), EnergyInterval)
                        call WeightingParticleEDF2(CF, PB(k), edf_local(:, j), EnergyInterval)
                    end do

                    call WeightingParticleEDF2(CF, bd(k), edf_local(:, diag_edf_num+1), EnergyInterval)

                    call MPI_ALLREDUCE(edf_local, edf_global, NeMax*(diag_edf_num+1), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)

                    if (0 == CF%DM%MyId) then
                        do j = 1, diag_edf_num
                            call edf_grid%Update(edf_global(:, j:j), "EDF"//"-"//trim(SpecyGlobal(k)%Name)//'-'//trim(num2str(j, 3)))
                        end do

                        call edf_grid%Update(edf_global(:, diag_edf_num+1:diag_edf_num+1), "EDF"//"-"//trim(SpecyGlobal(k)%Name)//'-bd')
                    end if
                end do

                deallocate(edf_local)
                deallocate(edf_global)
            end if

        end subroutine DiagEDFPeriod


        subroutine DiagEDFDump()
        
        end subroutine DiagEDFDump


        subroutine DiagEDFRelease()

            call edf_grid%Destroy()

        end subroutine DiagEDFRelease


        subroutine WeightingParticleEDF(CF, PB, pedf, particle_limit, EnergyInterval)
            class(ControlFlow), intent(in) :: CF
            type(ParticleBundle), intent(in) :: PB
            real(8), intent(inout) :: pedf(NeMax)
            real(8), intent(in) :: particle_limit(4)
            real(8), intent(in) :: EnergyInterval
            real(8) :: Energy
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

                else
                    pedf = 0.d0
                    do i = 1, PB%Npar
                        if (PB%PO(i)%Z >= particle_limit(1) .and. &
                            PB%PO(i)%Z <= particle_limit(2) .and. &
                            PB%PO(i)%R >= particle_limit(3) .and. &
                            PB%PO(i)%R <= particle_limit(4)) then

                            Energy = PB%PO(i)%Energy(PB%Mass, PB%VFactor) / JtoeV
                            N = ceiling(Energy/EnergyInterval)
                            if (N >= 1 .and. N <= NeMax) then
                                pedf(N) = pedf(N) + PB%PO(i)%WQ
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