module ModuleInit
    use Constants
    use ModuleDomain
    use json_module
    use ModuleControlFlow
    use ModuleFieldEM
    use ModuleFieldOne
    use ModuleFieldSource
    use ModuleFieldSolver
    use ModuleFieldBoundary2DZR
    use ModuleParticleBundle
    use ModuleParticleBoundary1D
    use ModuleSpecyOne
    use ModuleReactionOnePegasus
    use ModuleCircuitVoltageSource
    use ModuleMaterials
    use ModuleGeometry
    use ModuleExtCircuit
    use ModuleRecombination
    use ModuleParticleRezoning

    implicit none

#define CHECK_FOUND(found)  if (.not. found) call FoundError()

    ! global config var
    logical,    save :: found

    ! gas
    integer(4), save :: Ng = 0
    real(8), save    :: gas_pressure = 0.d0
    real(8), save    :: gas_temperature = 0.d0
    integer(4), save    :: collision_section_type = CS_MODEL_LXCAT

    real(8), save, Allocatable :: gas_ratio(:)
    character(len=99), save, Allocatable :: gas_name(:)

    ! plasma
    real(8),    save :: init_density = 0.d0
    real(8),    save :: ele_temperature = 0.d0
    integer(4), save :: ppg = 100
    integer(4), save :: ppg_dif(0:NsMax-1) = 100

    integer(4), save :: NzMax = 0
    integer(4), save :: NrMax = 0

contains

    subroutine InitControlFlowJson(JP, CF)
        Class(json_file), intent(inout) :: JP
        Type(ControlFlow), intent(inout) :: CF
        character(:), allocatable :: tmpPath
        logical :: alive
        Type(HDF5_PDump) :: hdf5Geom
        integer(4) :: size, rank, ierr, i

        integer(4) :: deps_z = 1
        integer(4) :: deps_r = 1

        real(8) :: ZLength = 0.d0
        real(8) :: Rlength = 0.d0
        real(8) :: dz = 0.d0
        real(8) :: dr = 0.d0

        real(8), allocatable :: deps_z_read(:), deps_r_read(:)
        Character(len=99) :: numstr
        real(8) :: tmp

        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

        call JP%get('diag.diag_type', diag_type_global, found)
        CHECK_FOUND(found)

        call JP%get('diag.diag_cycles', diag_cycles, found)
        CHECK_FOUND(found)

        call JP%get('diag.diag_preiod', diag_preiod, found)
        CHECK_FOUND(found)

        call JP%get('diag.diag_freq', diag_freq, found)
        CHECK_FOUND(found)

        call JP%get('diag.diag_Nt', diag_Nt, found)
        CHECK_FOUND(found)

        call JP%get('control.nrun', CF%NRun, found)
        CHECK_FOUND(found)

        call JP%get('control.dt', CF%dt, found)
        CHECK_FOUND(found)

        call JP%get('grid.z_length', ZLength, found)
        CHECK_FOUND(found)

        call JP%get('control.is_restart', is_restart, found)
        CHECK_FOUND(found)

        call JP%get('plasma.load_type', load_type, found)
        CHECK_FOUND(found)

        call JP%get('control.pic_type', pic_type, found)
        CHECK_FOUND(found)

        ! geometry
        call JP%get('grid.geometry', tmpPath, found)
        CHECK_FOUND(found)

        do i = 0, size-1
            if (rank == i) then
                inquire(file=tmpPath, exist=alive)
                if (alive) then
                    call hdf5Geom%init(filename=tmpPath, mode='read')
                    call hdf5Geom%open()

                else
                    write(*, '(a, a, a)') "The ", tmpPath, " is not found."
                    stop
                end if

                call hdf5Geom%readattr('/', 'Nz', NzMax)
                call hdf5Geom%readattr('/', 'Nr', NrMax)

                call hdf5Geom%close()

            end if

            call MPI_BARRIER(MPI_COMM_WORLD, ierr)
        end do

        ! dz = dr
        ! call JP%get('Grid.rLength', RLength, found)
        ! CHECK_FOUND(found)
        RLength = ZLength / dble(NzMax - 1) * dble(NrMax - 1)

        dz = ZLength / dble(NzMax - 1)
        ! dr = RLength / dble(NrMax - 1)
        dr = dz

        call JP%get('grid.deps_in_z', deps_z, found)
        CHECK_FOUND(found)

        call JP%get('grid.deps_in_r', deps_r, found)
        CHECK_FOUND(found)

        call JP%get('grid.deps_type', deps_type, found)
        CHECK_FOUND(found)

        call CF%Init()
        if (0 == rank) write(*, *) "Init ControlFlow"

        call CF%DM%Init(NzMax, NrMax, 0.d0, ZLength, 0.d0, RLength, deps_z, deps_r)

        CF%Period = diag_preiod
        if (diag_freq > 0.d0) then
            if (diag_preiod == 0) then
                CF%Period = int(1.d0 / diag_freq / CF%dt)
            
            else
                CF%dt = (1.d0 / diag_freq / dble(CF%Period))

            end if

        end if

        if (diag_freq <= 0.d0 .and. CF%Period == 0) then
            if (0 == rank) write(*, *) "Invalid diag period."
            stop

        end if

        if (deps_type_specify == deps_type) then
            Allocate(deps_z_read(deps_z))
            Allocate(deps_r_read(deps_r))

            do i = 1, deps_z
                write(numstr, *) i

                call JP%get('grid.specified_deps_in_z['//trim(numstr)//']', tmp, found)
                CHECK_FOUND(found)
                deps_z_read(i) = tmp
            end do

            do i = 1, deps_r
                write(numstr, *) i

                call JP%get('grid.specified_deps_in_r['//trim(numstr)//']', tmp, found)
                CHECK_FOUND(found)
                deps_r_read(i) = tmp
            end do

            call CF%DM%Specify(deps_z_read, deps_r_read)
        end if

        ! split & merge
        call JP%get('plasma.is_open_split_merge', is_open_split_merge, found)
        CHECK_FOUND(found)

        call JP%get('plasma.paritlc_upper_limit', paritlc_upper_limit, found)
        CHECK_FOUND(found)

        call JP%get('plasma.paritlc_lower_limit', paritlc_lower_limit, found)
        CHECK_FOUND(found)

        call JP%get('plasma.min_mp_to_split_k', min_mp_to_split_k, found)
        CHECK_FOUND(found)

        call JP%get('plasma.merge_limit', merge_limit, found)
        CHECK_FOUND(found)

        if (0 == rank) write(*, *) "Init Domain"

    end subroutine InitControlFlowJson


    subroutine initGeometryAndMaterialsJson(JP, CF, MT, Geom)
        Class(json_file), intent(inout) :: JP
        Type(ControlFlow), intent(inout) :: CF
        Type(Materials), intent(inout) :: MT
        Type(Geometry), intent(inout) :: Geom
        character(:), allocatable :: tmpPath, tmpname
        character(len=99) :: numstr
        Type(json_file) :: json_materials
        Type(HDF5_PDump) :: hdf5Geom
        logical :: alive
        integer(4) :: geom_nz, geom_nr
        integer(4) :: metals_num, dielectric_num, materials_num, i, j
        integer(4) :: material_type_tmp
        integer(4), allocatable :: tmp2d(:, :)
        real(8), allocatable :: tmp2d_real(:, :)
        real(8) :: tmp_real, dx, volume_factor
        integer(4) :: size, rank, ierr

        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

        ! materials
        call JP%get('grid.materials', tmpPath, found)
        CHECK_FOUND(found)

        inquire(file=tmpPath, exist=alive)
        if (alive) then
            call json_materials%initialize()
            call json_materials%load(tmpPath)
            if (0 == rank) write(*, *) "Load materials info from json file."

        else
            if (0 == rank) write(*, '(a, a, a)') "The ", tmpPath, " is not found."
            stop

        end if

        geom_nz = NzMax
        geom_nr = NrMax
        call Geom%init(geom_nz, geom_nr)

        ! geometry
        do i = 0, size-1
            if (rank == i) then
                call JP%get('grid.geometry', tmpPath, found)
                CHECK_FOUND(found)

                inquire(file=tmpPath, exist=alive)
                if (alive) then
                    call hdf5Geom%init(filename=tmpPath, mode='read')
                    call hdf5Geom%open()

                else
                    if (0 == rank) write(*, '(a, a, a)') "The ", tmpPath, " is not found."
                    stop

                end if

                if (allocated(tmp2d_real)) deallocate(tmp2d_real)
                allocate(tmp2d_real(geom_nz-1, geom_nr-1))
                call hdf5Geom%read('epsilon_r', tmp2d_real)
                Geom%cell_epsilon = tmp2d_real

                if (allocated(tmp2d)) deallocate(tmp2d)
                allocate(tmp2d(geom_nz-1, geom_nr-1))
                call hdf5Geom%read('load', tmp2d)
                Geom%cell_load = tmp2d

                if (allocated(tmp2d)) deallocate(tmp2d)
                allocate(tmp2d(geom_nz, geom_nr))
                call hdf5Geom%read('point_type', tmp2d)
                Geom%point_type = tmp2d

                if (allocated(tmp2d_real)) deallocate(tmp2d_real)
                allocate(tmp2d_real(geom_nz, geom_nr))
                call hdf5Geom%read('volume', tmp2d_real)
                Geom%point_volume_diag = tmp2d_real

                if (allocated(tmp2d)) deallocate(tmp2d)
                allocate(tmp2d(geom_nz, geom_nr-1))
                call hdf5Geom%read('hedge_type', tmp2d)
                Geom%hedge_type = tmp2d

                if (allocated(tmp2d)) deallocate(tmp2d)
                allocate(tmp2d(geom_nz-1, geom_nr))
                call hdf5Geom%read('vedge_type', tmp2d)
                Geom%vedge_type = tmp2d

                if (allocated(tmp2d)) deallocate(tmp2d)
                allocate(tmp2d, mold=Geom%cell_material)

                call hdf5Geom%read('cell_material', tmp2d)

                call hdf5Geom%close()
            end if

            call MPI_BARRIER(MPI_COMM_WORLD, ierr)
        end do

        ! init MT
        call json_materials%get('material_amount', materials_num, found)
        CHECK_FOUND(found)

        dielectric_num = 0
        metals_num = 0

        if (materials_num <= 0) then
            if (0 == rank) write(*, '(a)') "Number of materials less than 0."
            stop
        end if

        do i = 1, materials_num
            write(numstr, *) i

            call json_materials%get('materials['//trim(numstr)//'].type', material_type_tmp, found)
            CHECK_FOUND(found)
            
            if (material_type_tmp == material_type_dielectric) dielectric_num = dielectric_num + 1
            if (material_type_tmp == material_type_metal) metals_num = metals_num + 1

        end do

        call MT%init(geom_nz, geom_nr, dielectric_num, metals_num)

        ! set prop
        dielectric_num = 0
        metals_num = 0
        Geom%cell_material = 0
        do i = 1, materials_num
            write(numstr, *) i

            call json_materials%get('materials['//trim(numstr)//'].type', material_type_tmp, found)
            CHECK_FOUND(found)

            call json_materials%get('materials['//trim(numstr)//'].name', tmpname, found)
            CHECK_FOUND(found)

            if (material_type_tmp == material_type_dielectric) then
                dielectric_num = dielectric_num + 1
                MT%material_type(-dielectric_num) = material_type_dielectric
                MT%material_name(-dielectric_num) = trim(tmpname)

                Geom%cell_material = merge(-dielectric_num, Geom%cell_material, tmp2d == i-1)

                call json_materials%get('materials['//trim(numstr)//'].permittivity', tmp_real, found)
                CHECK_FOUND(found)
                MT%diels(-dielectric_num)%permittivity = tmp_real
                Geom%cell_epsilon = merge(tmp_real, Geom%cell_epsilon, tmp2d == i-1)

                call json_materials%get('materials['//trim(numstr)//'].permeability', tmp_real, found)
                CHECK_FOUND(found)
                MT%diels(-dielectric_num)%permeability = tmp_real

                call json_materials%get('materials['//trim(numstr)//'].conductivity', tmp_real, found)
                CHECK_FOUND(found)
                MT%diels(-dielectric_num)%conductivity = tmp_real

            else if (material_type_tmp == material_type_metal) then
                metals_num = metals_num + 1
                MT%material_type(metals_num) = material_type_metal
                MT%material_name(metals_num) = trim(tmpname)

                Geom%cell_material = merge(metals_num, Geom%cell_material, tmp2d == i-1)

            else if (material_type_tmp == material_type_vacuum) then
                MT%material_type(0) = material_type_vacuum
                MT%material_name(0) = trim(tmpname)

                Geom%cell_material = merge(0, Geom%cell_material, tmp2d == i-1)

            end if

        end do

        ! update point type, volume, and area
        do i = 1, Geom%Nz-1
            do j = 1, Geom%Nr-1
                if (MT%material_type(Geom%cell_material(i, j)) == material_type_metal) then
                    Geom%point_type(i, j) = Geom%cell_material(i, j)
                    Geom%point_type(i+1, j) = Geom%cell_material(i, j)
                    Geom%point_type(i, j+1) = Geom%cell_material(i, j)
                    Geom%point_type(i+1, j+1) = Geom%cell_material(i, j)
                end if
            end do
        end do

        dx = CF%DM%SpaceStep(1)
        Geom%point_area = 0.d0
        Geom%point_inv_volume_diag = 0.d0
        do i = 1, Geom%Nz
            do j = 1, Geom%Nr
                volume_factor = Geom%point_volume_diag(i, j)
                if (j == 1) then
                    Geom%point_volume_diag(i, j) = volume_factor * 1.d0 / 3.d0 * PI * dx * dx * dx
                else if (j == Geom%Nr) then
                    Geom%point_volume_diag(i, j) = volume_factor * 1.d0 / 3.d0 * (dble(j-2) + 2.d0 * dble(j-1)) * PI * dx * dx * dx
                else
                    Geom%point_volume_diag(i, j) = volume_factor * 1.d0 / 3.d0 * (dble(j) * (dble(j-1) + dble(j)) - dble(j-2) * (dble(j-2) + dble(j-1))) * PI * dx * dx * dx
                end if

                Geom%point_area(i, j) = 2.d0 * PI * dble(j-1) * dx * dx
                if (volume_factor > 1.d-6) then
                    Geom%point_inv_volume_diag(i, j) = 1.d0 / Geom%point_volume_diag(i, j)
                end if
            end do
        end do

        Geom%point_volume_weighting = Geom%point_volume_diag
        Geom%point_inv_volume_weighting = Geom%point_inv_volume_diag

        call MT%updataIndex(Geom)
        call Geom%check('geom_check')
        if (allocated(tmp2d)) deallocate(tmp2d)
        if (allocated(tmp2d_real)) deallocate(tmp2d_real)

        if (0 == rank) write(*, *) "Init Geometry and Materials"

    end subroutine initGeometryAndMaterialsJson


    subroutine InitFieldJson(JP, CF, FG, FT, FB)
        Class(json_file), intent(inout) :: JP
        Type(ControlFlow), intent(inout) :: CF
        Type(FieldEM), intent(inout) :: FG
        Type(FieldSource), intent(inout) :: FT
        Type(FieldBoundary2DZR), intent(inout) :: FB
        integer(4) :: k
        integer(4) :: size, rank, ierr

        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

        call FG%Init(CF)
        if (0 == rank) write(*, *) "Init FieldEM"

        call FT%Init(CF)
        if (0 == rank) write(*, *) "Init FieldSource"

        call FB%Init(CF%DM%GlobalShape(1), CF%DM%GlobalShape(2))
        if (0 == rank) write(*, *) "Init FieldBoundary2DZR"

    end subroutine InitFieldJson


    subroutine InitGasJson(JP, CF)
        Class(json_file), intent(inout) :: JP
        Type(ControlFlow), intent(inout) :: CF
        Character(len=99) :: numstr
        character(len=:),allocatable :: nametmp
        real(8) :: ratiotmp
        integer(4) :: size, rank, ierr, i

        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

        call JP%get('gas.ng', Ng, found)
        CHECK_FOUND(found)

        call JP%get('gas.pressure', gas_pressure, found)
        CHECK_FOUND(found)

        call JP%get('gas.temperature', gas_temperature, found)
        CHECK_FOUND(found)

        call JP%get('plasma.init_electron_temperature', ele_temperature, found)
        CHECK_FOUND(found)

        call JP%get('gas.collision_section_type', collision_section_type, found)
        CHECK_FOUND(found)

        if (Ng > 0) then
            Allocate(gas_name(Ng))
            Allocate(gas_ratio(Ng))

            do i = 1, Ng
                write(numstr, *) i

                call JP%get('gas.name['//trim(numstr)//']', nametmp, found)
                CHECK_FOUND(found)
                gas_name(i) = nametmp

                call JP%get('gas.ratio['//trim(numstr)//']', ratiotmp, found)
                CHECK_FOUND(found)
                gas_ratio(i) = ratiotmp
            end do

        else
            if (0 == rank) write(*, *) "The amount of gas must be greater than 0."
            stop

        end if

        CF%Ng = Ng
        call GasInitPegasus(CF, gas_pressure, gas_temperature, ele_temperature, gas_name, gas_ratio, collision_section_type)

        if(Allocated(gas_name)) Deallocate(gas_name)
        if(Allocated(gas_ratio)) Deallocate(gas_ratio)

        if (0 == rank) write(*, *) "Init Gas"

    end subroutine InitGasJson


    subroutine InitParticleJson(JP, CF, PB, FO, Geom)
        Class(json_file), intent(inout) :: JP
        Type(ControlFlow), intent(inout) :: CF
        Type(ParticleBundle), Allocatable, intent(inout) :: PB(:)
        Type(FieldOne), Allocatable, intent(inout) :: FO(:)
        Type(Geometry), intent(in) :: Geom
        Character(len=99) :: bdfilename
        integer(4) :: k
        integer(4) :: size, rank, ierr, i
        integer(4) :: positively, negatively
        real(8) :: al, density, coefficient

        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

        Allocate(PB(0:CF%Ns))
        Allocate(FO(0:CF%Ns))

        call JP%get('plasma.init_density', init_density, found)
        CHECK_FOUND(found)

        call JP%get('plasma.ppg', ppg, found)
        CHECK_FOUND(found)

        call JP%get('plasma.particle_number_max_ratio', particle_number_max_ratio, found)
        CHECK_FOUND(found)

        call JP%get('plasma.particle_limit_max_velocity', particle_limit_max_velocity, found)
        CHECK_FOUND(found)

        call JP%get('gas.positively_charged_particles', positively, found)
        CHECK_FOUND(found)

        call JP%get('gas.negatively_charged_particles', negatively, found)
        CHECK_FOUND(found)

        call JP%get('plasma.alpha', al, found)
        CHECK_FOUND(found)
        PB%alpha = al

        ppg_dif = ppg

        coefficient = dble(negatively)/dble(positively)

        do k = 0, CF%Ns

            write(bdfilename, '(a, a)') 'ParticleBundle_',  trim(SpecyGlobal(k)%Name)

            if (SpecyGlobal(k)%Charge > 0.d0)then

                density = init_density * coefficient

            else 

                density = init_density
                
            end if 

            call PB(k)%AllInit(CF, Geom, SpecyGlobal(k)%Charge, SpecyGlobal(k)%Mass, SpecyGlobal(k)%Temperature, density, ppg_dif(k), bdfilename)

            if (0 == rank) write(*, *) "Init ParticleBundle "//trim(SpecyGlobal(k)%Name)

            call FO(k)%Init(PB(k), CF)
        end do

        if (0 == rank) write(*, *) "Init Particles"

    end subroutine InitParticleJson


    subroutine InitCircuitJson(JP, CF, EC)
        Class(json_file), intent(inout) :: JP
        Type(ControlFlow), intent(inout) :: CF
        Type(ExtCircuits), intent(inout) :: EC
        Character(len=99) :: numstr, numstr2
        integer(4) :: i, j
        real(8) :: f, v, ph, t, tmp_val
        integer(4) :: voltate_source_nfre = 0, seg=0
        integer(4) :: circuits_number, type_tmp, metal_index_tmp
        integer(4) :: each_circuit_counts(circuit_type_number)
        integer(4) :: each_circuit_number(circuit_type_number)
        character(:), allocatable :: name_tmp
        integer(4) :: size, rank, ierr

        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

        ! init ExtCircuits
        call JP%get('circuit.circuits_number', circuits_number, found)
        CHECK_FOUND(found)

        each_circuit_number = 0
        do i = 1, circuits_number
            write(numstr, *) i

            call JP%get('circuit.circuits_list['//trim(numstr)//'].type', type_tmp, found)
            CHECK_FOUND(found)

            if (type_tmp >= 1 .and. type_tmp <= circuit_type_number) &
                each_circuit_number(type_tmp) = each_circuit_number(type_tmp) + 1

        end do

        call EC%init(circuits_number, each_circuit_number)

        ! set parameter
        each_circuit_counts = 0
        do i = 1, circuits_number
            write(numstr, *) i

            call JP%get('circuit.circuits_list['//trim(numstr)//'].type', type_tmp, found)
            CHECK_FOUND(found)

            call JP%get('circuit.circuits_list['//trim(numstr)//'].metal_index', metal_index_tmp, found)
            CHECK_FOUND(found)

            call JP%get('circuit.circuits_list['//trim(numstr)//'].name', name_tmp, found)
            CHECK_FOUND(found)

            select case(type_tmp)
                case(circuit_type_digital_ground)
                    each_circuit_counts(type_tmp) = each_circuit_counts(type_tmp) + 1
                    EC%circuits_type(i) = circuit_type_digital_ground
                    EC%circuits_index(i) = each_circuit_counts(type_tmp)
                    EC%circuits_metal(i) = metal_index_tmp

                    call EC%grouds(each_circuit_counts(type_tmp))%init(name_tmp)

                case(circuit_type_voltage_source)
                    each_circuit_counts(type_tmp) = each_circuit_counts(type_tmp) + 1
                    EC%circuits_type(i) = circuit_type_voltage_source
                    EC%circuits_index(i) = each_circuit_counts(type_tmp)
                    EC%circuits_metal(i) = metal_index_tmp

                    call JP%get('circuit.circuits_list['//trim(numstr)//'].n_freq', voltate_source_nfre, found)
                    CHECK_FOUND(found)

                    call JP%get('circuit.circuits_list['//trim(numstr)//'].Vdc', v, found)
                    CHECK_FOUND(found)
                    EC%vol_sources(each_circuit_counts(type_tmp))%Vdc = v

                    if (voltate_source_nfre > 0) then
                        call EC%vol_sources(each_circuit_counts(type_tmp))%Init(voltate_source_nfre, name_tmp)

                        do j = 1, voltate_source_nfre
                            write(numstr2, *) j

                            call JP%get('circuit.circuits_list['//trim(numstr)//'].frequency['//trim(numstr2)//']', f, found)
                            CHECK_FOUND(found)
                            EC%vol_sources(each_circuit_counts(type_tmp))%Frequency(j) = f

                            call JP%get('circuit.circuits_list['//trim(numstr)//'].amplitude['//trim(numstr2)//']', v, found)
                            CHECK_FOUND(found)
                            EC%vol_sources(each_circuit_counts(type_tmp))%Amplitude(j) = v

                            call JP%get('circuit.circuits_list['//trim(numstr)//'].phase['//trim(numstr2)//']', ph, found)
                            CHECK_FOUND(found)
                            EC%vol_sources(each_circuit_counts(type_tmp))%Phase(j) = ph

                        end do
                    end if

                case(circuit_type_pulse_source)
                    each_circuit_counts(type_tmp) = each_circuit_counts(type_tmp) + 1
                    EC%circuits_type(i) = circuit_type_pulse_source
                    EC%circuits_index(i) = each_circuit_counts(type_tmp)
                    EC%circuits_metal(i) = metal_index_tmp

                    call JP%get('circuit.circuits_list['//trim(numstr)//'].seg', seg, found)
                    CHECK_FOUND(found)

                    if (seg > 1) then
                        call EC%pulse_sources(each_circuit_counts(type_tmp))%Init(seg, name_tmp)

                        do j = 1, seg
                            write(numstr2, *) j

                            call JP%get('circuit.circuits_list['//trim(numstr)//'].time['//trim(numstr2)//']', t, found)
                            CHECK_FOUND(found)
                            EC%pulse_sources(each_circuit_counts(type_tmp))%time(j) = t

                            call JP%get('circuit.circuits_list['//trim(numstr)//'].amplitude['//trim(numstr2)//']', v, found)
                            CHECK_FOUND(found)
                            EC%pulse_sources(each_circuit_counts(type_tmp))%amplitude(j) = v

                        end do
                    end if

                case(circuit_type_rlc_source)
                    each_circuit_counts(type_tmp) = each_circuit_counts(type_tmp) + 1
                    EC%circuits_type(i) = circuit_type_rlc_source
                    EC%circuits_index(i) = each_circuit_counts(type_tmp)
                    EC%circuits_metal(i) = metal_index_tmp

                    call JP%get('circuit.circuits_list['//trim(numstr)//'].n_freq', voltate_source_nfre, found)
                    CHECK_FOUND(found)

                    if (voltate_source_nfre > 0) then
                        call EC%rlc_sources(each_circuit_counts(type_tmp))%Init(voltate_source_nfre, name_tmp)
                        call EC%rlc_sources(each_circuit_counts(type_tmp))%ode_rlc%init(3, CF%dt, c_funloc(RhsFnRLC), CF%dt*dble(CF%Timer))

                        do j = 1, voltate_source_nfre
                            write(numstr2, *) j

                            call JP%get('circuit.circuits_list['//trim(numstr)//'].frequency['//trim(numstr2)//']', f, found)
                            CHECK_FOUND(found)
                            rlc_source_frequency(j) = f

                            call JP%get('circuit.circuits_list['//trim(numstr)//'].amplitude['//trim(numstr2)//']', v, found)
                            CHECK_FOUND(found)
                            rlc_source_amplitude(j) = v

                            call JP%get('circuit.circuits_list['//trim(numstr)//'].phase['//trim(numstr2)//']', ph, found)
                            CHECK_FOUND(found)
                            rlc_source_phase(j) = ph

                        end do

                        call JP%get('circuit.circuits_list['//trim(numstr)//'].Rs', rlc_Rs, found)
                        CHECK_FOUND(found)

                        call JP%get('circuit.circuits_list['//trim(numstr)//'].Cs', rlc_Cs, found)
                        CHECK_FOUND(found)

                        call JP%get('circuit.circuits_list['//trim(numstr)//'].Ls', rlc_Ls, found)
                        CHECK_FOUND(found)

                    end if

                case(circuit_type_imn_source)
                    each_circuit_counts(type_tmp) = each_circuit_counts(type_tmp) + 1
                    EC%circuits_type(i) = circuit_type_imn_source
                    EC%circuits_index(i) = each_circuit_counts(type_tmp)
                    EC%circuits_metal(i) = metal_index_tmp

                    call JP%get('circuit.circuits_list['//trim(numstr)//'].n_freq', voltate_source_nfre, found)
                    CHECK_FOUND(found)

                    if (voltate_source_nfre > 0) then
                        call EC%imn_sources(each_circuit_counts(type_tmp))%Init(voltate_source_nfre, name_tmp)
                        call EC%imn_sources(each_circuit_counts(type_tmp))%ode_imn%init(6, CF%dt, c_funloc(RhsFnIMN), CF%dt*dble(CF%Timer))

                        do j = 1, voltate_source_nfre
                            write(numstr2, *) j

                            call JP%get('circuit.circuits_list['//trim(numstr)//'].frequency['//trim(numstr2)//']', f, found)
                            CHECK_FOUND(found)
                            imn_source_frequency(j) = f

                            call JP%get('circuit.circuits_list['//trim(numstr)//'].amplitude['//trim(numstr2)//']', v, found)
                            CHECK_FOUND(found)
                            imn_source_amplitude(j) = v

                            call JP%get('circuit.circuits_list['//trim(numstr)//'].phase['//trim(numstr2)//']', ph, found)
                            CHECK_FOUND(found)
                            imn_source_phase(j) = ph

                        end do

                        call JP%get('circuit.circuits_list['//trim(numstr)//'].Rs', imn_Rs, found)
                        CHECK_FOUND(found)

                        call JP%get('circuit.circuits_list['//trim(numstr)//'].Cm1', imn_Cm1, found)
                        CHECK_FOUND(found)

                        call JP%get('circuit.circuits_list['//trim(numstr)//'].Cm2', imn_Cm2, found)
                        CHECK_FOUND(found)

                        call JP%get('circuit.circuits_list['//trim(numstr)//'].Lm', imn_Lm, found)
                        CHECK_FOUND(found)
                        
                        call JP%get('circuit.circuits_list['//trim(numstr)//'].Rm', imn_Rm, found)
                        CHECK_FOUND(found)

                        call JP%get('circuit.circuits_list['//trim(numstr)//'].is_open_capacitance', imn_is_open_stray_capacitance, found)
                        CHECK_FOUND(found)
                        
                        call JP%get('circuit.circuits_list['//trim(numstr)//'].C_stray', imn_C_stray, found)
                        CHECK_FOUND(found)

                        call JP%get('circuit.circuits_list['//trim(numstr)//'].L_stray', imn_L_stray, found)
                        CHECK_FOUND(found)

                        call JP%get('circuit.circuits_list['//trim(numstr)//'].R_stray', imn_R_stray, found)
                        CHECK_FOUND(found)

                    end if

            end select

        end do

        if (allocated(name_tmp)) deallocate(name_tmp)
        if (0 == rank) write(*, *) "Init Circuits"

    end subroutine InitCircuitJson


    subroutine InitFieldSolverJson(CF, FS, FT, Ext, boundary, MT, Geom)
        type(ControlFlow), intent(inout) :: CF
        type(FieldSolver), intent(inout) :: FS
        type(FieldSource), intent(inout) :: FT
        type(ExtCircuits), intent(inout) :: Ext
        type(FieldBoundary2DZR), intent(inout) :: boundary
        type(Materials), intent(inout) :: MT
        type(Geometry), intent(inout) :: Geom
        integer(4) :: size, rank, ierr

        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

        call FS%Init(CF, MT%metal_count)
        call FT%Zero()
        call FS%SolveL(CF, FT, Ext, boundary, MT, Geom)

        if (0 == rank) write(*, *) "Init FieldSolver"

    end subroutine InitFieldSolverJson


    subroutine InitRecombinationJson(JP, CF, Re2d)
        class(json_file), intent(inout) :: JP
        type(ControlFlow), intent(inout) :: CF
        type(Recombination2D), intent(inout) :: Re2d
        real(8), allocatable :: rate_tmp(:)
        integer(4), allocatable :: reactions_tmp(:, :)
        integer(4) :: size, rank, ierr, i
        integer(4) :: re_num
        Character(len=99) :: numstr

        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

        call JP%get('gas.is_open_recombination', is_open_recombination, found)
        CHECK_FOUND(found)

        if (is_open_recombination) then
            call JP%get('gas.recb.recb_time_step_num', recombination_time_step_number, found)
            CHECK_FOUND(found)

            call JP%get('gas.recb.recb_reaction_num', re_num, found)
            CHECK_FOUND(found)

            allocate(rate_tmp(re_num))
            allocate(reactions_tmp(re_num, 3))

            do i = 1, re_num
                write(numstr, *) i

                call JP%get('gas.recb.recb_list['//trim(numstr)//'][1]', reactions_tmp(i, 1), found)
                CHECK_FOUND(found)

                call JP%get('gas.recb.recb_list['//trim(numstr)//'][2]', reactions_tmp(i, 2), found)
                CHECK_FOUND(found)

                call JP%get('gas.recb.recb_list['//trim(numstr)//'][3]', reactions_tmp(i, 3), found)
                CHECK_FOUND(found)

                call JP%get('gas.recb.recb_rate['//trim(numstr)//']', rate_tmp(i), found)
                CHECK_FOUND(found)

            end do

            Associate (zstart => CF%DM%CornerIndex(BOUNDARY_LEFT, 1), &
                    zend   => CF%DM%CornerIndex(BOUNDARY_RIGHT, 1), &
                    rstart => CF%DM%CornerIndex(BOUNDARY_LEFT, 2), &
                    rend   => CF%DM%CornerIndex(BOUNDARY_RIGHT, 2), &
                    dz     => CF%DM%SpaceStep(1), &
                    dr     => CF%DM%SpaceStep(2))

                call Re2d%init(CF%Ns, re_num, reactions_tmp, rate_tmp, zstart, zend-1, rstart, rend-1, dz, dr)

            end Associate

            if (allocated(rate_tmp)) deallocate(rate_tmp)
            if (allocated(reactions_tmp)) deallocate(reactions_tmp)
            if (0 == rank) write(*, *) "Init Recombination"
        end if

    end subroutine InitRecombinationJson


    subroutine FoundError()
        integer(4) :: size, rank, ierr

        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

        if (0 == rank) write(*, *) "The key is not exit in josn file."
        stop

        return
    end subroutine FoundError

end module ModuleInit
