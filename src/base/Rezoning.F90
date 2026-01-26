Module ModuleParticleRezoning
    use ModuleControlFlow
    use ModuleParticleOne
    use ModuleParticleBundle
    use ModuleGeometry
    implicit none

    logical, save :: is_open_split_merge = .True.

    real(8), save :: paritlc_upper_limit = 1.01d0
    real(8), save :: paritlc_lower_limit = 0.99d0
    real(8), save :: min_mp_to_split_k = 0.1d0
    real(8), save :: merge_limit = 0.34d0

    contains

        subroutine particle_split_merge_control(CF, PB, geom, ppg_ref)
            type(ControlFlow), intent(in) :: CF
            type(ParticleBundle), intent(inout) :: PB(0:CF%Ns)
            type(Geometry), intent(in) :: geom
            integer(4), intent(in) :: ppg_ref(0:CF%Ns)
            integer(4) :: size, rank, ierr
            integer(4) :: Npar_recv
            real(8) :: Npar_eq
            integer(4) :: k

            Associate(zstart => CF%DM%CornerIndex(BOUNDARY_LEFT, 1), &
                      zend   => CF%DM%CornerIndex(BOUNDARY_RIGHT, 1)-1, &
                      rstart => CF%DM%CornerIndex(BOUNDARY_LEFT, 2), &
                      rend   => CF%DM%CornerIndex(BOUNDARY_RIGHT, 2)-1)

                if (is_open_split_merge) then
                    call split_particle_weight(PB(0))
                    call merge_particle_random(PB(0))

                    if (mod(CF%Timer, CF%Period) == 0) then
                        call weighting_update_base_particle_number(CF, PB(0), geom, ppg_ref(0))

                        do k = 1, CF%Ns
                            call weighting_update_base_particle_number(CF, PB(k), geom, ppg_ref(k))

                            call split_particle_weight(PB(k))
                            call merge_particle_random(PB(k))
                        end do
                    end if

                end if
            end Associate

        end subroutine particle_split_merge_control


        subroutine merge_particle_random(PB)
            type(ParticleBundle), intent(inout) :: PB
            integer(4) :: i
            real(8) :: WQ

            do i = PB%Npar, 1, -1
                WQ = PB%CalRW(PB%PO(i)%R, PB%PO(i)%Z)
                if (PB%PO(i)%WQ < merge_limit * WQ) then
                    call random_number(R)
                    if (R < PB%PO(i)%WQ/WQ) then
                        PB%PO(i)%WQ = WQ
                    else
                        call PB%DelOne(i)
                    end if
                end if
            end do

        end subroutine merge_particle_random


        subroutine split_particle_weight(PB)
            type(ParticleBundle), intent(inout) :: PB
            type(ParticleOne) :: TempParticle
            integer(4) :: i, j, Ncr
            real(8) :: WQ

            do i = 1, PB%Npar
                WQ = PB%CalRW(PB%PO(i)%R, PB%PO(i)%Z)
                Ncr = int(PB%PO(i)%WQ / WQ)
                if (Ncr > 1) then
                    do j = 1, Ncr - 1
                        PB%PO(i)%WQ = PB%PO(i)%WQ - WQ
                        TempParticle = PB%PO(i)
                        TempParticle%WQ = WQ
                        call PB%AddOne(TempParticle)
                    end do
                end if
            end do

        end subroutine split_particle_weight


        subroutine weighting_update_base_particle_number(CF, PB, geom, ppg_ref)
            type(ControlFlow), intent(in) :: CF
            type(ParticleBundle), intent(inout) :: PB
            type(Geometry), intent(in) :: geom
            integer(4), intent(in) :: ppg_ref
            integer(4) :: k, ierr
            integer(4) :: npar_max, npar_min
            real(8) :: limit
            integer(4) :: npar_global

            call MPI_ALLREDUCE(PB%NPar, npar_global, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)

            npar_max = int(dble((geom%Nz - 1) * (geom%Nr - 1) * ppg_ref) * paritlc_upper_limit)
            npar_min = int(dble((geom%Nz - 1) * (geom%Nr - 1) * ppg_ref) * paritlc_lower_limit)
            limit = int(dble((geom%Nz - 1) * (geom%Nr - 1) * ppg_ref) * min_mp_to_split_k)

            if (npar_global > npar_max) then
                call PB%UpdW0(PB%Weight * paritlc_upper_limit)
            
            else if (npar_global < npar_min .and. npar_global > limit) then
                call PB%UpdW0(PB%Weight * paritlc_lower_limit)

            end if

        end subroutine weighting_update_base_particle_number

end Module ModuleParticleRezoning