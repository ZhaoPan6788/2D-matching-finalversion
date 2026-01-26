Module ModuleRecombination
    use ModuleParticleBundle

    implicit none

    integer(4), parameter :: recombination_type_const = 0
    integer(4), parameter :: recombination_type_temperature = 0

    logical, save :: is_open_recombination = .False.
    integer(4), save :: recombination_time_step_number = 1

    type Recombination2D
        integer(4) :: region_zstart
        integer(4) :: region_zend
        integer(4) :: region_rstart
        integer(4) :: region_rend

        integer(4) :: specy_num
        integer(4) :: reaction_num
        integer(4), allocatable :: specy_index(:)
        logical, allocatable :: is_cal_temperature(:)
        integer(4), allocatable :: recb_reaction(:, :)
        real(8), allocatable :: recb_rate(:)

        integer(4), allocatable :: specy_to_array_index(:)
        real(8), allocatable :: particle_density(:, :, :)
        real(8), allocatable :: inv_particle_temperature(:, :, :)
        real(8), allocatable :: volume(:, :)
        real(8), allocatable :: inv_volume(:, :)

        contains

            procedure :: init => initRecombination2D
            procedure :: show => showRecombination2D
            procedure :: recb => executingRecombination2D
            procedure :: delt => deletingParticles
            procedure :: destroy => destroyRecombination2D

    end type

    contains

        subroutine initRecombination2D(this, ns, reaction_num, reaction_list, reaction_rate, zstart, zend, rstart, rend, dz, dr)
            class(Recombination2D), intent(inout) :: this
            integer(4), intent(in) :: reaction_num, ns
            integer(4), intent(in) :: reaction_list(reaction_num, 3)
            real(8), intent(in) :: reaction_rate(reaction_num)
            integer(4), intent(in) :: zstart, zend, rstart, rend
            real(8), intent(in) :: dz, dr
            integer(4) :: i, j, index

            call this%destroy()
            if (reaction_num > 0) then
                this%reaction_num = reaction_num
                allocate(this%recb_reaction(this%reaction_num, 3))
                allocate(this%recb_rate(this%reaction_num))
                allocate(this%specy_to_array_index(0:ns))

                this%recb_reaction = reaction_list
                this%recb_rate = reaction_rate
                this%specy_to_array_index = 0

                do i = 1, reaction_num
                    index = this%recb_reaction(i, 1)
                    if (index < 0 .or. index > ns) then
                        write(*, *) "The specified particle index is invalid in recombination reactions."
                    end if

                    index = this%recb_reaction(i, 2)
                    if (index < 0 .or. index > ns) then
                        write(*, *) "The specified particle index is invalid in recombination reactions."
                    end if
                end do

                do i = 1, reaction_num
                    index = this%recb_reaction(i, 1)
                    if (this%specy_to_array_index(index) == 0) this%specy_to_array_index(index) = 1

                    index = this%recb_reaction(i, 2)
                    if (this%specy_to_array_index(index) == 0) this%specy_to_array_index(index) = 1
                end do

                this%specy_num = sum(this%specy_to_array_index)
                allocate(this%specy_index(this%specy_num))
                allocate(this%is_cal_temperature(this%specy_num))

                this%specy_to_array_index = 0
                j = 1
                do i = 1, reaction_num
                    index = this%recb_reaction(i, 1)
                    if (this%specy_to_array_index(index) == 0) then
                        this%specy_to_array_index(index) = 1
                        this%specy_index(j) = index
                        j = j + 1

                    end if

                    index = this%recb_reaction(i, 2)
                    if (this%specy_to_array_index(index) == 0) then
                        this%specy_to_array_index(index) = 1
                        this%specy_index(j) = index
                        j = j + 1

                    end if

                end do

                this%specy_to_array_index = -1
                do i = 1, this%specy_num
                    this%specy_to_array_index(this%specy_index(i)) = i
                end do

                this%is_cal_temperature = .False.
                do i = 1, reaction_num
                    if (this%recb_reaction(i, 3) == 1) then
                        index = this%recb_reaction(i, 1)
                        this%is_cal_temperature(this%specy_to_array_index(index)) = .True.

                        index = this%recb_reaction(i, 2)
                        this%is_cal_temperature(this%specy_to_array_index(index)) = .True.
                    end if
                end do

                this%region_zstart = zstart
                this%region_zend = zend
                this%region_rstart = rstart
                this%region_rend = rend
                allocate(this%particle_density(this%region_zstart:this%region_zend, this%region_rstart:this%region_rend, this%specy_num))
                allocate(this%inv_particle_temperature(this%region_zstart:this%region_zend, this%region_rstart:this%region_rend, this%specy_num))
                allocate(this%volume(this%region_zstart:this%region_zend, this%region_rstart:this%region_rend))
                allocate(this%inv_volume(this%region_zstart:this%region_zend, this%region_rstart:this%region_rend))

                do i = this%region_rstart, this%region_rend
                    this%volume(:, i) = PI * dr * dr * dz * dble(i*i - (i-1)*(i-1))
                    this%inv_volume(:, i) = 1.d0 / (PI * dr * dr * dz * dble(i*i - (i-1)*(i-1)))

                end do

            end if

        end subroutine


        subroutine showRecombination2D(this)
            class(Recombination2D), intent(inout) :: this
            integer(4) :: i

            write(*, '(a30, *(i8))') 'region:', this%region_zstart, this%region_zend, this%region_rstart, this%region_rend
            write(*, '(a30, i8)') 'specy num:', this%specy_num
            write(*, '(a30, i8)') 'reaction num:', this%reaction_num
            write(*, '(a30, *(i4))') 'specy index:', this%specy_index
            write(*, '(a30, *(l4))') 'is temperature:', this%is_cal_temperature
            write(*, '(a30, *(i4))') 'array index:', this%specy_to_array_index

            write(*, '(a30, *(es20.6))') 'reaction rate:', this%recb_rate
            write(*, '(a30)') 'reactions:'
            do i = 1, this%reaction_num
                write(*, '(30x, *(i4))') this%recb_reaction(i, :)
            end do

        end subroutine


        subroutine executingRecombination2D(this, ns, PB, time_step)
            class(Recombination2D), intent(inout) :: this
            integer(4), intent(in) :: ns
            type(ParticleBundle), intent(inout) :: PB(0:ns)
            real(8), intent(in) :: time_step
            integer(4) :: i, j, k, index
            integer(4) :: Nz, Nr
            real(8), allocatable :: recb_neg(:, :), recb_pos(:, :)
            integer(4) :: neg_index, pos_index
            real(8) :: energy

            this%particle_density = 0.d0
            this%inv_particle_temperature = 0.d0

            do k = 1, this%specy_num
                index = this%specy_index(k)

                do i = 1, PB(index)%NPar
                    Nz = ceiling(PB(index)%PO(i)%Z)
                    Nr = ceiling(PB(index)%PO(i)%R)

                    this%particle_density(Nz, Nr, k) = this%particle_density(Nz, Nr, k) + PB(index)%PO(i)%WQ

                    if (this%is_cal_temperature(k)) then
                        energy = PB(index)%PO(i)%Vz**2 + PB(index)%PO(i)%Vr**2 + PB(index)%PO(i)%Vt**2
                        this%inv_particle_temperature(Nz, Nr, k) = this%inv_particle_temperature(Nz, Nr, k) + energy * PB(index)%PO(i)%WQ

                    end if
                end do

                this%particle_density(:, :, k) = this%particle_density(:, :, k) * this%inv_volume

                if (this%is_cal_temperature(k)) then
                    this%inv_particle_temperature(:, :, k) = this%inv_particle_temperature(:, :, k) * this%inv_volume &
                                                        * 0.5 * PB(index)%mass * PB(index)%Vfactor * PB(index)%Vfactor

                    do i = this%region_zstart, this%region_zend
                        do j = this%region_rstart, this%region_rend
                            if (this%particle_density(i, j, k) > 0.d0 .and. this%inv_particle_temperature(i, j, k) > 0.d0) then
                                this%inv_particle_temperature(i, j, k) = 1.d0 / (this%inv_particle_temperature(i, j, k) / JtoeV / this%particle_density(i, j, k))

                            else
                                this%inv_particle_temperature(i, j, k) = 0.d0

                            end if
                        end do
                    end do
                end if
            end do

            allocate(recb_neg(this%region_zstart:this%region_zend, this%region_rstart:this%region_rend))
            allocate(recb_pos(this%region_zstart:this%region_zend, this%region_rstart:this%region_rend))

            do i = 1, this%reaction_num
                neg_index = this%recb_reaction(i, 1)
                pos_index = this%recb_reaction(i, 2)

                if (this%recb_reaction(i, 3) == 0) then
                    recb_neg = this%particle_density(:, :, this%specy_to_array_index(neg_index)) * &
                               this%particle_density(:, :, this%specy_to_array_index(pos_index)) * &
                               this%recb_rate(i) * time_step * this%volume
                    recb_pos = recb_neg

                else if (this%recb_reaction(i, 3) == 1) then
                    recb_neg = this%particle_density(:, :, this%specy_to_array_index(neg_index)) * &
                               this%particle_density(:, :, this%specy_to_array_index(pos_index)) * &
                               time_step * this%volume * this%recb_rate(i) * &
                               sqrt(this%inv_particle_temperature(:, :, this%specy_to_array_index(neg_index))) * &
                               this%inv_particle_temperature(:, :, this%specy_to_array_index(pos_index))
                    recb_pos = recb_neg

                end if

                call this%delt(PB(neg_index), recb_neg)
                call this%delt(PB(pos_index), recb_pos)

            end do

            deallocate(recb_neg)
            deallocate(recb_pos)

        end subroutine


        subroutine deletingParticles(this, PB, recb_particle)
            class(Recombination2D), intent(inout) :: this
            type(ParticleBundle), intent(inout) :: PB
            real(8), intent(inout) :: recb_particle(this%region_zstart:this%region_zend, this%region_rstart:this%region_rend)
            integer(4) :: i, j, k
            integer(4) :: Nz, Nr

            do i = PB%NPar, 1, -1
                Nz = ceiling(PB%PO(i)%Z)
                Nr = ceiling(PB%PO(i)%R)

                if (recb_particle(Nz, Nr) > PB%PO(i)%WQ .and. PB%NPar > 0) then
                    recb_particle(Nz, Nr) = recb_particle(Nz, Nr) - PB%PO(i)%WQ
                    call PB%DelOne(i)

                else if (recb_particle(Nz, Nr) > 0.d0) then
                    call random_number(R)
                    if (R < recb_particle(Nz, Nr)/PB%PO(i)%WQ  .and. PB%NPar > 0) then
                        recb_particle(Nz, Nr) = 0.d0
                        call PB%DelOne(i)
                    end if

                end if
            end do

        end subroutine


        subroutine destroyRecombination2D(this)
            class(Recombination2D), intent(inout) :: this

            this%reaction_num = 0
            this%specy_num = 0

            if (allocated(this%is_cal_temperature)) deallocate(this%is_cal_temperature)
            if (allocated(this%specy_index)) deallocate(this%specy_index)
            if (allocated(this%recb_reaction)) deallocate(this%recb_reaction)
            if (allocated(this%recb_rate)) deallocate(this%recb_rate)
            if (allocated(this%specy_to_array_index)) deallocate(this%specy_to_array_index)
            if (allocated(this%particle_density)) deallocate(this%particle_density)
            if (allocated(this%inv_particle_temperature)) deallocate(this%inv_particle_temperature)
            if (allocated(this%volume)) deallocate(this%volume)
            if (allocated(this%inv_volume)) deallocate(this%inv_volume)

        end subroutine

end Module ModuleRecombination
