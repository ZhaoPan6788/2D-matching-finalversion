Module ModuleParticleBundle
    use ModuleParticleOne
    use ModuleFieldEM
    use ModuleBspline
    use ModuleParallelDump
    use ModuleGeometry

    implicit none

    real(8), save :: particle_number_max_ratio = 0.d0
    real(8), save :: particle_limit_max_velocity = 0.d0
    real(8), save :: particle_number_lower_boundary_ratio = 0.25d0
    real(8), save :: particle_number_decrease_ratio = 0.5d0
    real(8), save :: particle_number_increase_ratio = 1.5d0

    Type :: ParticleBundle
        Type(FileName) :: IOName
        integer(4) :: NPar                              ! restart
        integer(4) :: size                              ! restart
        integer(4) :: npar_min, npar_max
        real(8) :: k_lb, k_dec, k_inc

        real(8) :: XFactor, VFactor                     ! restart
        logical :: LXScaled=.False., LVScaled=.False.   ! restart
        real(8) :: Charge, Mass, Weight                 ! restart
        real(8) :: dx, dt

        real(8) :: alpha = 0.01d0
        real(8) :: grid_z, grid_r

        Type(ParticleOne), Allocatable :: PO(:)         ! restart
        Type(Bspline1N2D), Allocatable :: BS(:)

    contains

        procedure :: AllInit => AllInitializationParticleBundle

        procedure :: Init => initParticleBundle
        procedure :: Adjust => adjustParticleBundle

        procedure :: Destroy => DestroyParticleBundle
        procedure :: Realloc => reallocParticleBundle

        procedure :: AddOne  => AddParticleOneParticleBundle
        procedure :: AddBun  => AddParticleBundleParticleBundle
        procedure :: DelOne  => DelParticleOneParticleBundle

        procedure :: PosRes  => PositionRescaleParticleBundle
        procedure :: VelRes  => VelocityRescaleParticleBundle

        procedure :: CalRW   => CalculateRadialWeighting
        procedure :: UpdW0   => UpdateWeighting
        procedure :: SetW0   => SetWeighting

        procedure :: MoveES  => MoveElectrostaticParticleBundle
        procedure :: MoveESEx  => MoveElectrostaticParticleBundleExplicit
        procedure :: MoveESIm  => MoveElectrostaticParticleBundleImplicit
        procedure :: MoveEM  => MoveElectroMagneticParticleBundle

        procedure :: VelLimit => VelocityLimitParticleBundle

        procedure :: Dump => DumpParticleBundle
        procedure :: Load => LoadParticleBundle
        procedure :: Copy => CopyParticleBundle

        procedure :: Diag => DiagParticleBundle

        procedure, private :: DumpParticleBundleHDF5
        procedure, private :: DumpParticleBundleDat
        procedure, private :: LoadParticleBundleHDF5
        procedure, private :: LoadParticleBundleDat

    end Type ParticleBundle

    Type(HDF5_PDump), private :: hdf5ParticleBundleDump

    contains

        subroutine DumpParticleBundle(this)
            class(ParticleBundle), intent(inout) :: this

            if (FILE_EXTENSION_MODE_H5 == this%IOName%ExtensionMode) then
                call this%DumpParticleBundleHDF5()

            else if (FILE_EXTENSION_MODE_DAT == this%IOName%ExtensionMode) then
                call this%DumpParticleBundleDat()
            
            else
                call this%DumpParticleBundleHDF5()

            end if

        end subroutine DumpParticleBundle


        subroutine LoadParticleBundle(this, Status)
			class(ParticleBundle), intent(inout) :: this
            logical, intent(inout), optional :: Status
            logical :: alive

            Inquire(file=this%IOName%FullName%str, exist=alive)
            if (alive) then
                if (FILE_EXTENSION_MODE_H5 == this%IOName%ExtensionMode) then
                    call this%LoadParticleBundleHDF5()

                else if (FILE_EXTENSION_MODE_DAT == this%IOName%ExtensionMode) then
                    call this%LoadParticleBundleDat()
                
                end if
            end if

            if (present(Status)) Status = alive

		end subroutine LoadParticleBundle


        subroutine DumpParticleBundleHDF5(this)
			class(ParticleBundle), intent(inout) :: this
            integer(4) :: size, rank, ierr

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The ParticleBundle is save as hdf5 file."

            call hdf5ParticleBundleDump%init(filename=this%IOName%FullName%str, mode='write', serial=.True.)
            call hdf5ParticleBundleDump%open()

            call hdf5ParticleBundleDump%writeattr('/', 'NPar', this%NPar)
            call hdf5ParticleBundleDump%writeattr('/', 'size', this%size)
            call hdf5ParticleBundleDump%writeattr('/', 'Charge', this%Charge)
            call hdf5ParticleBundleDump%writeattr('/', 'Mass', this%Mass)
            call hdf5ParticleBundleDump%writeattr('/', 'Weight', this%Weight)
            call hdf5ParticleBundleDump%writeattr('/', 'XFactor', this%XFactor)
            call hdf5ParticleBundleDump%writeattr('/', 'VFactor', this%VFactor)
            call hdf5ParticleBundleDump%writeattr('/', 'LXScaled', this%LXScaled)
            call hdf5ParticleBundleDump%writeattr('/', 'LVScaled', this%LVScaled)

            call hdf5ParticleBundleDump%write('Z', this%PO%Z)
            call hdf5ParticleBundleDump%write('R', this%PO%R)
            call hdf5ParticleBundleDump%write('Vz', this%PO%Vz)
            call hdf5ParticleBundleDump%write('Vr', this%PO%Vr)
            call hdf5ParticleBundleDump%write('Vt', this%PO%Vt)
            call hdf5ParticleBundleDump%write('Az', this%PO%Az)
            call hdf5ParticleBundleDump%write('Ar', this%PO%Ar)
            call hdf5ParticleBundleDump%write('WQ', this%PO%WQ)

            call hdf5ParticleBundleDump%close()

        end subroutine DumpParticleBundleHDF5


        subroutine DumpParticleBundleDat(this)
            class(ParticleBundle), intent(inout) :: this
            integer(4) :: size, rank, ierr

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The ParticleBundle is save as hdf5 file."

            open(10, file=this%IOName%FullName%str)
                write(10, *) this%NPar, this%size
                write(10, *) this%Charge, this%Mass, this%Weight
                write(10, *) this%XFactor, this%VFactor
                write(10, *) this%LXScaled, this%LVScaled
                write(10, *) this%PO%Z
                write(10, *) this%PO%R
                write(10, *) this%PO%Vz
                write(10, *) this%PO%Vr
                write(10, *) this%PO%Vt
                write(10, *) this%PO%Az
                write(10, *) this%PO%Ar
                write(10, *) this%PO%WQ
            close(10)

        end subroutine DumpParticleBundleDat


        subroutine LoadParticleBundleHDF5(this)
            class(ParticleBundle), intent(inout) :: this
            real(8),ALLOCATABLE :: Temp1D(:)
            integer(4) :: size, rank, ierr, npar

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The ParticleBundle is load from hdf5 file."

            call hdf5ParticleBundleDump%init(filename=this%IOName%FullName%str, mode='read')
            call hdf5ParticleBundleDump%open()

            call hdf5ParticleBundleDump%readattr('/', 'NPar', npar)
            call hdf5ParticleBundleDump%readattr('/', 'size', this%size)
            call hdf5ParticleBundleDump%readattr('/', 'Charge', this%Charge)
            call hdf5ParticleBundleDump%readattr('/', 'Mass', this%Mass)
            call hdf5ParticleBundleDump%readattr('/', 'Weight', this%Weight)
            call hdf5ParticleBundleDump%readattr('/', 'XFactor', this%XFactor)
            call hdf5ParticleBundleDump%readattr('/', 'VFactor', this%VFactor)
            call hdf5ParticleBundleDump%readattr('/', 'LXScaled', this%LXScaled)
            call hdf5ParticleBundleDump%readattr('/', 'LVScaled', this%LVScaled)

            call this%Init(this%size, this%npar_min, this%npar_max, &
                           this%k_lb, this%k_dec, this%k_inc)

            this%NPar = npar

            Allocate(Temp1D(this%size))
            call hdf5ParticleBundleDump%read('Z', Temp1D)
            this%PO%Z = Temp1D

            call hdf5ParticleBundleDump%read('R', Temp1D)
            this%PO%R = Temp1D

            call hdf5ParticleBundleDump%read('Vz', Temp1D)
            this%PO%Vz = Temp1D

            call hdf5ParticleBundleDump%read('Vr', Temp1D)
            this%PO%Vr = Temp1D

            call hdf5ParticleBundleDump%read('Vt', Temp1D)
            this%PO%Vt = Temp1D

            call hdf5ParticleBundleDump%read('Az', Temp1D)
            this%PO%Az = Temp1D

            call hdf5ParticleBundleDump%read('Ar', Temp1D)
            this%PO%Ar = Temp1D

            call hdf5ParticleBundleDump%read('WQ', Temp1D)
            this%PO%WQ = Temp1D

            Deallocate(Temp1D)
            call hdf5ParticleBundleDump%close()

        end subroutine LoadParticleBundleHDF5


        subroutine LoadParticleBundleDat(this)
            class(ParticleBundle), intent(inout) :: this
            integer(4) :: size, rank, ierr, npar
            integer(4) :: tmpint1, tmpint2
            real(8) :: tmpreal1, tmpreal2, tmpreal3

            call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
            call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

            if (0 == rank) write(*, '(a)') "The ParticleBundle is load from dat file."

            open(10, file=this%IOName%FullName%str)
                read(10, *) npar, this%size
                read(10, *) this%Charge, this%Mass, this%Weight
                read(10, *) this%XFactor, this%VFactor
                read(10, *) this%LXScaled, this%LVScaled
                call this%Init(this%size, this%npar_min, this%npar_max, &
                               this%k_lb, this%k_dec, this%k_inc)

                this%NPar = npar
                read(10, *) this%PO%Z
                read(10, *) this%PO%R
                read(10, *) this%PO%Vz
                read(10, *) this%PO%Vr
                read(10, *) this%PO%Vt
                read(10, *) this%PO%Az
                read(10, *) this%PO%Ar
                read(10, *) this%PO%WQ
            close(10)

        end subroutine LoadParticleBundleDat


        subroutine DiagParticleBundle(this, name)
			class(ParticleBundle), intent(inout) :: this
            Character(*), intent(in) :: name
            Type(FileName) :: tmpName

            call tmpName%Init(name, PathMode=FILE_PATH_MODE_DIAG, &
                                    ExtensionMode=FILE_EXTENSION_MODE_H5, &
                                    ParallelMode=FILE_PARALLEL_MODE_OPEN, &
                                    DynamicIndex=FILE_DYNAMIC_MODE_CLOSE)

            call hdf5ParticleBundleDump%init(filename=tmpName%FullName%str, mode='write', serial=.True.)
            call hdf5ParticleBundleDump%open()

            call hdf5ParticleBundleDump%writeattr('/', 'NPar', this%NPar)
            call hdf5ParticleBundleDump%writeattr('/', 'size', this%size)
            call hdf5ParticleBundleDump%writeattr('/', 'npar_min', this%npar_min)
            call hdf5ParticleBundleDump%writeattr('/', 'npar_max', this%npar_max)
            call hdf5ParticleBundleDump%writeattr('/', 'k_lb', this%k_lb)
            call hdf5ParticleBundleDump%writeattr('/', 'k_dec', this%k_dec)
            call hdf5ParticleBundleDump%writeattr('/', 'k_inc', this%k_inc)
            call hdf5ParticleBundleDump%writeattr('/', 'Charge', this%Charge)
            call hdf5ParticleBundleDump%writeattr('/', 'Mass', this%Mass)
            call hdf5ParticleBundleDump%writeattr('/', 'Weight', this%Weight)
            call hdf5ParticleBundleDump%writeattr('/', 'XFactor', this%XFactor)
            call hdf5ParticleBundleDump%writeattr('/', 'VFactor', this%VFactor)
            call hdf5ParticleBundleDump%writeattr('/', 'LXScaled', this%LXScaled)
            call hdf5ParticleBundleDump%writeattr('/', 'LVScaled', this%LVScaled)

            call hdf5ParticleBundleDump%write('Z', this%PO%Z)
            call hdf5ParticleBundleDump%write('R', this%PO%R)
            call hdf5ParticleBundleDump%write('Vz', this%PO%Vz)
            call hdf5ParticleBundleDump%write('Vr', this%PO%Vr)
            call hdf5ParticleBundleDump%write('Vt', this%PO%Vt)
            call hdf5ParticleBundleDump%write('Az', this%PO%Az)
            call hdf5ParticleBundleDump%write('Ar', this%PO%Ar)
            call hdf5ParticleBundleDump%write('WQ', this%PO%WQ)

            call hdf5ParticleBundleDump%close()

        end subroutine DiagParticleBundle


        subroutine AllInitializationParticleBundle(this, CF, geom, charge, mass, temperature, density, ppg, pbname)
            class(ParticleBundle), intent(inout) :: this
            class(ControlFlow), intent(in) :: CF
            type(Geometry), intent(in) :: geom
            real(8), intent(in) :: charge, mass, temperature, density
            integer(4), intent(in) :: ppg
            character(*), optional, intent(in) :: pbname
            integer(4) :: i, j, k, NPArMax
            real(8) :: XFactor, VFactor, Wq
            logical ::  Status
            type(ParticleOne) :: tmp_one

            if (present(pbname)) then
                call this%IOName%Init(pbname, RESTART_FILE_NAME)
            else
                call this%IOName%Init("ParticleBundle", RESTART_FILE_NAME)
            end if

            associate (Nz => CF%DM%GlobalShape(1), &
                       Nr => CF%DM%GlobalShape(2), &
                       Lz => CF%DM%LocalShape(1), &
                       Lr => CF%DM%LocalShape(2), &
                       dz => CF%DM%SpaceStep(1), &
                       dr => CF%DM%SpaceStep(2), &
                       zstart => CF%DM%CornerIndex(BOUNDARY_LEFT, 1), &
                       zend => CF%DM%CornerIndex(BOUNDARY_RIGHT, 1), &
                       rstart => CF%DM%CornerIndex(BOUNDARY_LEFT, 2), &
                       rend => CF%DM%CornerIndex(BOUNDARY_RIGHT, 2))

                this%dx = dz
                this%dt = CF%dt
                this%XFactor = this%dx
                this%VFactor = this%dx / this%dt
                this%grid_r = dble(Nr-1)
                this%grid_z = dble(Nz-1)

                this%Charge = charge
                this%Mass   = mass
                this%Weight = density * PI * (dble(Nr - 1) * dr) * (dble(Nr - 1) * dr) * (dble(Nz - 1) * dz) / (dble(Nr - 1) * dble(Nz - 1) * dble(ppg))

                call this%Init(ppg * (Lz-1) * (Lr-1), &
                               10, &
                               Ceiling(particle_number_max_ratio * ppg * (Lz-1) * (Lr-1)), &
                               particle_number_lower_boundary_ratio, &
                               particle_number_decrease_ratio, &
                               particle_number_increase_ratio)

                this%LXScaled = .True.
                this%LVScaled = .True.
                VFactor = 1.d0 / this%VFactor

                do i = zstart, zend-1
                    do j = rstart, rend-1
                        if (geom%cell_load(i, j) == 1) then
                            Wq = density / Dble(ppg) * 2.d0 * PI * this%dx * this%dx * this%dx * dble(j)

                            do k = 1, ppg
                                call tmp_one%PosInit(dble(i-1), dble(i), dble(j-1), dble(j))
                                call RANDOM_NUMBER(R)

                                if (R < tmp_one%R/dble(j)) then
                                    call tmp_one%VelMaxInit(mass, temperature)
                                    call tmp_one%VelRes(VFactor)
                                    call tmp_one%AccInpInit()
                                    tmp_one%Wq = Wq
                                    call this%AddOne(tmp_one)
                                end if
                            end do
                        end if
                    end do
                end do

                call this%Adjust()

            end associate

        end subroutine AllInitializationParticleBundle


        function CalculateRadialWeighting(this, pos_r, pos_z)
            class(ParticleBundle), intent(inout) :: this
            real(8), intent(in) :: pos_r
            real(8), intent(in), optional :: pos_z
            real(8) :: CalculateRadialWeighting
            integer(4) :: NzL, NzU, NrL, NrU
            real(8) :: ZL, ZU, RL, RU

            CalculateRadialWeighting = this%Weight * ((1.d0 - this%alpha) * pos_r / this%grid_r + this%alpha)

        end


        subroutine UpdateWeighting(this, new_weight)
            class(ParticleBundle), intent(inout) :: this
            real(8), intent(in) :: new_weight

            this%Weight = new_weight

        end subroutine UpdateWeighting


        subroutine SetWeighting(this, new_weight, pos_r)
            class(ParticleBundle), intent(inout) :: this
            real(8), intent(in) :: new_weight, pos_r

            this%Weight = new_weight / ((1.d0 - this%alpha) * pos_r / this%grid_r + this%alpha)

        end subroutine SetWeighting


        subroutine initParticleBundle(this, init_size, min_size, max_size, k_lb, k_dec, k_inc)
            class(ParticleBundle), intent(inout) :: this
            integer(4), optional, intent(in) :: init_size, min_size, max_size
            real(8), optional, intent(in) :: k_lb, k_dec, k_inc

            this%npar = 0
            this%size = 1000
            this%npar_min = 10
            this%npar_max = 10000
            this%k_lb = 0.25d0
            this%k_dec = 0.5d0
            this%k_inc = 1.5d0

            if (present(init_size) .and. init_size > 0) this%size = init_size
            if (present(min_size)  .and. min_size > 0)  this%npar_min = min_size
            if (present(max_size)  .and. max_size > 0)  this%npar_max = max_size

            if (present(k_lb)  .and. k_lb > 0.d0 .and. k_lb <= 1.d0) this%k_lb = k_lb
            if (present(k_dec) .and. k_dec > 0.d0 .and. k_dec <= 1.d0) this%k_dec = k_dec
            if (present(k_inc) .and. k_inc >= 1.d0) this%k_inc = k_inc

            call this%Destroy()
            call this%Realloc()

        end subroutine initParticleBundle


        subroutine CopyParticleBundle(this, src)
            class(ParticleBundle), intent(inout) :: this
            Type(ParticleBundle), intent(in) :: src

            this%NPar = src%NPar
            call this%Init(src%size, src%npar_min, src%npar_max, src%k_lb, src%k_dec, src%k_inc)

            this%XFactor = src%XFactor
            this%VFactor = src%VFactor
            this%LXScaled = src%LXScaled
            this%LVScaled = src%LVScaled

            this%Charge = src%Charge
            this%Mass = src%Mass
            this%Weight = src%Weight
            this%dx = src%dx
            this%dt = src%dt

        end subroutine CopyParticleBundle


        subroutine reallocParticleBundle(this, newsize)
            class(ParticleBundle), intent(inout) :: this
            integer(4), optional, intent(in) :: newsize
            integer(4) :: size_tmp
            type(ParticleOne), allocatable :: temp(:)
            
            if (allocated(this%PO)) then
                if (present(newsize)) then
                    size_tmp = newsize

                    if (size_tmp < this%npar_min) size_tmp = this%npar_min
                    if (size_tmp > this%npar_max) size_tmp = this%npar_max

                    allocate(temp(size_tmp))
                    if (size_tmp <= this%size) then
                        temp(1:size_tmp) = this%PO(1:size_tmp)
                        deallocate(this%PO)
                        call move_alloc(temp, this%PO)

                    else if (size_tmp > this%size) then
                        temp(1:this%size) = this%PO(1:this%size)
                        deallocate(this%PO)
                        call move_alloc(temp, this%PO)

                    end if

                    if (allocated(temp)) deallocate(temp)
                    this%size = size_tmp

                else
                    deallocate(this%PO)
                    if (this%size > 0) then
                        allocate(this%PO(this%size))
                    else
                        write(*, *) "The particle bundle size needs to be greater than 0."
                        stop
                    end if
                end if

            else
                if (this%size > 0) then
                    allocate(this%PO(this%size))
                else
                    write(*, *) "The particle bundle size needs to be greater than 0."
                    stop
                end if
            end if

        end subroutine reallocParticleBundle


        subroutine DestroyParticleBundle(this)
            class(ParticleBundle), intent(inout) :: this

            if(allocated(this%PO)) deallocate(this%PO)
            if(allocated(this%BS)) deallocate(this%BS)

            this%NPar = 0

        end subroutine DestroyParticleBundle


        subroutine AddParticleOneParticleBundle(this, PO)
            class(ParticleBundle),intent(inout) ::  this
            class(ParticleOne),intent(in) ::  PO

            if (this%NPar+1 > this%size) call this%adjust(this%NPar+1)
            this%PO(this%NPar+1) = PO
            this%NPar = this%NPar + 1

        end subroutine AddParticleOneParticleBundle


        subroutine AddParticleBundleParticleBundle(this, bun)
            class(ParticleBundle), intent(inout) :: this
            class(ParticleBundle), intent(in) :: bun

            if (this%NPar+bun%NPar > this%size) call this%adjust(this%NPar+bun%NPar)
            this%PO(this%NPar+1:this%NPar+bun%NPar) = bun%PO(1:bun%NPar)
            this%NPar = this%NPar + bun%NPar

        end subroutine AddParticleBundleParticleBundle


        subroutine DelParticleOneParticleBundle(this, NDel)
            class(ParticleBundle),intent(inout) :: this
            integer(4),intent(in) :: NDel

            if (this%NPar > 0) then
                if (NDel >= 1 .and. NDel <= this%NPar) then
                    this%PO(NDel) = this%PO(this%NPar)
                    this%NPar = this%NPar - 1                    
                    call this%adjust()

                else
                    write(*, *) "The NDel is out of the range."
                    stop
                end if

            else
                write(*, *) "No particle in particle bundle can be deleted."
                stop
            end if

        end subroutine DelParticleOneParticleBundle


        subroutine adjustParticleBundle(this, newsize)
            class(ParticleBundle), intent(inout) :: this
            integer(4), optional, intent(in) :: newsize
            integer(4) :: tmp_size

            if (present(newsize) .and. newsize > 0) then
                if (newsize >= this%size) then
                    tmp_size = max(newsize, min(this%npar_max, int(this%k_inc*dble(this%size))))
                    if (tmp_size > this%npar_max) then
                        write(*, *) "The size of particle bundle reaches the maximum."
                        stop
                    end if

                    call this%Realloc(tmp_size)
                else
                    call this%Realloc(newsize)
                end if

            else
                if (this%NPar == 0) then
                    call this%Realloc(this%npar_min)
                else if (this%NPar < int(this%k_lb * dble(this%size))) then
                    tmp_size = int(max(this%k_lb, this%k_dec)*dble(this%size))
                    call this%Realloc(tmp_size)
                end if
            end if

        end subroutine adjustParticleBundle


        subroutine PositionRescaleParticleBundle(this)
            class(ParticleBundle),intent(inout) :: this
            integer(4) :: i
            real(8) :: XFactor

            if(this%LXScaled) then
                XFactor=this%XFactor
                this%LXScaled=.False.
            else
                XFactor=1.d0/this%XFactor
                this%LXScaled=.True.
            end If
            
            do i=1,this%NPar
                call this%PO(i)%PosRes(XFactor)
            end do

        end subroutine PositionRescaleParticleBundle


        subroutine VelocityRescaleParticleBundle(this)
            class(ParticleBundle),intent(inout) :: this
            integer(4) :: i
            real(8) :: VFactor

            if(this%LVScaled) then
                VFactor=this%VFactor
                this%LVScaled=.False.
            else
                VFactor=1.d0/this%VFactor
                this%LVScaled=.True.
            end if

            do i=1,this%NPar
                call this%PO(i)%VelRes(VFactor)
            end do

        end subroutine VelocityRescaleParticleBundle


        subroutine MoveElectrostaticParticleBundle(this, FG)
            class(ParticleBundle),intent(inout) :: this
            Type(FieldEM),intent(inout) :: FG

            if (pic_type_explicit == pic_type) then
                call this%MoveESEx(FG)

            else if (pic_type_implicit == pic_type) then
                call this%MoveESIm(FG)

            end if

        end subroutine MoveElectrostaticParticleBundle


        subroutine MoveElectrostaticParticleBundleExplicit(this, FG)
            Class(ParticleBundle),intent(inout) :: this
            Type(FieldEM),intent(inout) :: FG
  
        end subroutine MoveElectrostaticParticleBundleExplicit


        subroutine MoveElectrostaticParticleBundleImplicit(this, FG)
            class(ParticleBundle),intent(inout) :: this
            Type(FieldEM),intent(inout) :: FG
            Integer(4) :: i
            Integer(4) :: N1, N2, N1H, N2H
            Real(8)    :: Z1, Z2, R1, R2
            Real(8)    :: Z1H, Z2H, R1H, R2H
            Real(8)    :: E1, E2, E3
            Real(8)    :: EFactor
            Real(8)    :: CosA, SinA, X, Y, Vx, Vy, P_R

            EFactor = this%Charge / this%Mass * this%dt / (this%dx / this%dt) * 0.5d0

            associate(NPar => this%NPar, Ez => FG%Ez, Er => FG%Er)
                do i = 1, NPar
                    N1  = Ceiling(this%PO(i)%Z)
                    Z1  = Dble(N1) - this%PO(i)%Z
                    Z2  = 1.d0 - Z1
                    N1H = Int(this%PO(i)%Z + 0.5d0)
                    Z1H = Dble(N1H) + 0.5d0 - this%PO(i)%Z
                    Z2H = 1.d0 - Z1H

                    N2  = Ceiling(this%PO(i)%R)
                    R1  = Dble(N2) - this%PO(i)%R
                    R2  = 1.d0 - R1
                    N2H = Int(this%PO(i)%R + 0.5d0)
                    R1H = Dble(N2H) + 0.5d0 - this%PO(i)%R
                    R2H = 1.d0 - R1H

                    E1 = Er(N1,N2H)*Z1*R1H + Er(N1+1,N2H)*Z2*R1H + Er(N1,N2H+1)*Z1*R2H + Er(N1+1,N2H+1)*Z2*R2H
                    E2 = Ez(N1H,N2)*Z1H*R1 + Ez(N1H+1,N2)*Z2H*R1 + Ez(N1H,N2+1)*Z1H*R2 + Ez(N1H+1,N2+1)*Z2H*R2
                    E3 = 0.d0

                    E1 = E1 * EFactor
                    E2 = E2 * EFactor
                    E3 = E3 * EFactor

                    this%PO(i)%Vz = this%PO(i)%Vz + E2
                    this%PO(i)%Z  = this%PO(i)%Z  + E2
                    this%PO(i)%Az = (this%PO(i)%Az + E2) * 0.5d0

                    this%PO(i)%Vr = this%PO(i)%Vr + E1
                    this%PO(i)%R  = this%PO(i)%R  + E1
                    this%PO(i)%Ar = (this%PO(i)%Ar + E1) * 0.5d0

                    ! pre push
                    this%PO(i)%Vz = this%PO(i)%Vz + this%PO(i)%Az
                    this%PO(i)%Z  = this%PO(i)%Z  + this%PO(i)%Vz

                    P_R  = this%PO(i)%R
                    Vx = this%PO(i)%Vr + this%PO(i)%Ar
                    Vy = this%PO(i)%Vt

                    X = P_R + Vx
                    Y = Vy
                    
                    P_R = dsqrt(X*X + Y*Y)
                    this%PO(i)%R = P_R

                    if (this%PO(i)%R < MinReal) then
                        this%PO(i)%Vr = dsqrt(Vx*Vx+Vy*Vy)
                        this%PO(i)%Vt = MinReal
                        this%PO(i)%Ar = abs(this%PO(i)%Ar)

                    else
                        SinA = Y / P_R
                        CosA = X / P_R

                        this%PO(i)%Vr = Vx * CosA + Vy * SinA
                        this%PO(i)%Vt = -Vx * SinA + Vy * CosA
                        this%PO(i)%Ar = this%PO(i)%Ar * CosA

                    end if

                end do
            end associate

        end subroutine MoveElectrostaticParticleBundleImplicit


        subroutine MoveElectroMagneticParticleBundle(this, FG)
            class(ParticleBundle),intent(inout) :: this
            Type(FieldEM),intent(inout) :: FG

        end subroutine MoveElectroMagneticParticleBundle


        subroutine VelocityLimitParticleBundle(this, limitfactor, scalefactor)
            class(ParticleBundle),intent(inout) :: this
            real(8), intent(in) :: limitfactor, scalefactor
            integer(4) :: i
            real(8) :: VelocityMax, limitfactorNorm, scalefactorNorm

            limitfactorNorm = limitfactor
            if (limitfactorNorm <= 0.d0 .or. limitfactorNorm > 1.d0) limitfactorNorm = 0.75d0

            scalefactorNorm = scalefactor
            if (scalefactorNorm <= 0.d0 .or. scalefactorNorm > 1.d0) scalefactorNorm = 0.25d0

            VelocityMax = limitfactorNorm * particle_limit_max_velocity
            do i = 1, this%NPar
                if (sqrt(this%PO(i)%Vz**2 + this%PO(i)%Vr**2 + this%PO(i)%Vt**2) > VelocityMax) then
                    this%PO(i)%Vz = this%PO(i)%Vz * scalefactorNorm
                    this%PO(i)%Vr = this%PO(i)%Vr * scalefactorNorm
                    this%PO(i)%Vt = this%PO(i)%Vt * scalefactorNorm
                end if
            end do

        end subroutine VelocityLimitParticleBundle


        subroutine MoveParticleOneElectrostaticImplicit(this, PO, FG)
            class(ParticleBundle), intent(in) :: this
            type(ParticleOne), intent(inout) :: PO
            type(FieldEM), intent(in) :: FG
            Integer(4) :: i
            Integer(4) :: N1, N2, N1H, N2H
            Real(8)    :: Z1, Z2, R1, R2
            Real(8)    :: Z1H, Z2H, R1H, R2H
            Real(8)    :: E1, E2, E3
            Real(8)    :: EFactor
            Real(8)    :: CosA, SinA, X, Y, Vx, Vy, P_R

            EFactor = this%Charge / this%Mass * this%dt / (this%dx / this%dt) * 0.5d0

            associate(Ez => FG%Ez, Er => FG%Er)
                N1  = Ceiling(PO%Z)
                Z1  = Dble(N1) - PO%Z
                Z2  = 1.d0 - Z1
                N1H = Int(PO%Z + 0.5d0)
                Z1H = Dble(N1H) + 0.5d0 - PO%Z
                Z2H = 1.d0 - Z1H

                N2  = Ceiling(PO%R)
                R1  = Dble(N2) - PO%R
                R2  = 1.d0 - R1
                N2H = Int(PO%R + 0.5d0)
                R1H = Dble(N2H) + 0.5d0 - PO%R
                R2H = 1.d0 - R1H

                E1 = Er(N1,N2H)*Z1*R1H + Er(N1+1,N2H)*Z2*R1H + Er(N1,N2H+1)*Z1*R2H + Er(N1+1,N2H+1)*Z2*R2H
                E2 = Ez(N1H,N2)*Z1H*R1 + Ez(N1H+1,N2)*Z2H*R1 + Ez(N1H,N2+1)*Z1H*R2 + Ez(N1H+1,N2+1)*Z2H*R2
                E3 = 0.d0

                E1 = E1 * EFactor
                E2 = E2 * EFactor
                E3 = E3 * EFactor

                PO%Vz = PO%Vz + E2
                PO%Z  = PO%Z  + E2
                PO%Az = (PO%Az + E2) * 0.5d0

                PO%Vr = PO%Vr + E1
                PO%R  = PO%R  + E1
                PO%Ar = (PO%Ar + E1) * 0.5d0

                ! pre push
                PO%Vz = PO%Vz + PO%Az
                PO%Z  = PO%Z  + PO%Vz

                P_R  = PO%R
                Vx = PO%Vr + PO%Ar
                Vy = PO%Vt

                X = P_R + Vx
                Y = Vy
                
                P_R = dsqrt(X*X + Y*Y)
                PO%R = P_R

                if (PO%R < MinReal) then
                    PO%Vr = dsqrt(Vx*Vx+Vy*Vy)
                    PO%Vt = MinReal
                    PO%Ar = abs(PO%Ar)

                else
                    SinA = Y / P_R
                    CosA = X / P_R

                    PO%Vr = Vx * CosA + Vy * SinA
                    PO%Vt = -Vx * SinA + Vy * CosA
                    PO%Ar = PO%Ar * CosA

                end if

            end associate

        end subroutine MoveParticleOneElectrostaticImplicit

End Module ModuleParticleBundle
