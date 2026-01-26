Module ModuleParticleBoundary1D
    use Constants
    use ModuleParticleBundle

    implicit none

    Type ParticleBoundary
        Type(FileName) :: IOName
        Type(Domain), Pointer :: DM => Null()
        real(8) :: limit(BOUNDARY_PER_DIM, DIM_DOMAIN) = 0.d0

        Type(ParticleBundle) :: PBDump(BOUNDARY_PER_DIM, DIM_DOMAIN)

    contains

        procedure :: Init => InitializationParticleBoundary
        procedure :: Dump => DumpParticleBoundary
        procedure :: Load => LoadParticleBoundary
        procedure :: Destroy => DestroyParticleBoundary

        procedure :: AborpZ => AborptionParticleBoundary1DZ
        procedure :: AborpR => AborptionParticleBoundary1DR

    end Type ParticleBoundary

contains

    subroutine InitializationParticleBoundary(PBound, PB, DM, minLimit, maxLimit, pbbdname)
        Class(ParticleBoundary), intent(inout) :: PBound
        Type(ParticleBundle), intent(in) :: PB
        Type(Domain), intent(in), target :: DM
        real(8), intent(in), optional :: minLimit(DIM_DOMAIN)
        real(8), intent(in), optional :: maxLimit(DIM_DOMAIN)
        character(*), optional, intent(in) :: pbbdname
        Character(len=99) :: bdfilename
        integer(4) :: i, j

        PBound%DM => DM
        if (present(pbbdname)) then
            call PBound%IOName%Init(pbbdname, RESTART_FILE_NAME)
        else
            call PBound%IOName%Init("ParticleBoundary", RESTART_FILE_NAME)
        end if

        PBound%limit = dble(PBound%DM%CornerIndex - 1)
        if (present(minLimit)) PBound%limit(BOUNDARY_LEFT, :) = minLimit
        if (present(maxLimit)) PBound%limit(BOUNDARY_RIGHT, :) = maxLimit

        do i = 1, DIM_DOMAIN
            do j = 1, BOUNDARY_PER_DIM
                call PBound%PBDump(j, i)%IOName%Init(trim(pbbdname)//"_"//trim(num2str(i, 3))//"_"//trim(num2str(j, 3)), RESTART_FILE_NAME)
                call PBound%PBDump(j, i)%Copy(PB)
            end do
        end do

        return
    end subroutine InitializationParticleBoundary


    subroutine DumpParticleBoundary(PBound)
        Class(ParticleBoundary), intent(inout) :: PBound
        integer(4) :: i, j

        do i = 1, DIM_DOMAIN
            do j = 1, BOUNDARY_PER_DIM
                call PBound%PBDump(j, i)%Dump()
            end do
        end do

        return
    end subroutine DumpParticleBoundary


    subroutine LoadParticleBoundary(PBound)
        Class(ParticleBoundary), intent(inout) :: PBound
        integer(4) :: i, j

        do i = 1, DIM_DOMAIN
            do j = 1, BOUNDARY_PER_DIM
                call PBound%PBDump(j, i)%Load()
            end do
        end do

        return
    end subroutine LoadParticleBoundary


    subroutine DestroyParticleBoundary(PBound)
        Class(ParticleBoundary), intent(inout) :: PBound
        integer(4) :: i, j

        do i = 1, DIM_DOMAIN
            do j = 1, BOUNDARY_PER_DIM
                call PBound%PBDump(j, i)%Destroy()
            end do
        end do

        return
    end subroutine DestroyParticleBoundary


    subroutine AborptionParticleBoundary1DZ(PBound, PB)
        Class(ParticleBoundary), intent(inout) :: PBound
        Type(ParticleBundle), intent(inout) :: PB
        integer(4) :: i, dim1d

        dim1d = 1
        PBound%PBDump(BOUNDARY_LEFT, dim1d)%NPar = 0
        PBound%PBDump(BOUNDARY_RIGHT, dim1d)%NPar = 0
        do i = PB%NPar, 1, -1
            if (PB%PO(i)%Z <= PBound%limit(BOUNDARY_LEFT, dim1d)) then
                call PBound%PBDump(BOUNDARY_LEFT, dim1d)%AddOne(PB%PO(i))
                call PB%DelOne(i)

            else if (PB%PO(i)%Z >= PBound%limit(BOUNDARY_RIGHT, dim1d)) then
                call PBound%PBDump(BOUNDARY_RIGHT, dim1d)%AddOne(PB%PO(i))
                call PB%DelOne(i)

            end if
        end do

        return
    end subroutine AborptionParticleBoundary1DZ


    subroutine AborptionParticleBoundary1DR(PBound, PB)
        Class(ParticleBoundary), intent(inout) :: PBound
        Type(ParticleBundle), intent(inout) :: PB
        integer(4) :: i, dim1d

        dim1d = 2
        PBound%PBDump(BOUNDARY_LEFT, dim1d)%NPar = 0
        PBound%PBDump(BOUNDARY_RIGHT, dim1d)%NPar = 0
        do i = PB%NPar, 1, -1
            if (PB%PO(i)%R <= PBound%limit(BOUNDARY_LEFT, dim1d)) then
                call PBound%PBDump(BOUNDARY_LEFT, dim1d)%AddOne(PB%PO(i))
                call PB%DelOne(i)

            else if (PB%PO(i)%R >= PBound%limit(BOUNDARY_RIGHT, dim1d)) then
                call PBound%PBDump(BOUNDARY_RIGHT, dim1d)%AddOne(PB%PO(i))
                call PB%DelOne(i)
            
            end if
        end do

        return
    end subroutine AborptionParticleBoundary1DR

end Module ModuleParticleBoundary1D
