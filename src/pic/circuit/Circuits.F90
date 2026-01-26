module ModuleExtCircuit
    use mpi
    use ModuleCircuitDigitalGround
    use ModuleCircuitVoltageSource
    use ModuleCircuitPulseSource
    use ModuleCircuitRLC
    use ModuleMaterials
    use ModuleCircuitIMN

    implicit none

    ! 定义的电路类型
    integer(4), parameter :: circuit_type_digital_ground = 1
    integer(4), parameter :: circuit_type_voltage_source = 2
    integer(4), parameter :: circuit_type_pulse_source = 3
    integer(4), parameter :: circuit_type_rlc_source = 4
    integer(4), parameter :: circuit_type_imn_source = 5

    integer(4), parameter :: circuit_type_number = 5

    type ExtCircuits
        integer(4) :: circuits_number                       ! 电路的总数
        integer(4) :: circuits_size(circuit_type_number)    ! 每种电路的数量

        integer(4), allocatable :: circuits_type(:)         ! 每个电路的类型
        integer(4), allocatable :: circuits_index(:)        ! 每个电路在对应类型电路数组中的索引
        integer(4), allocatable :: circuits_metal(:)        ! 每个电路连接的金属电极

        ! 每种类型电路的数组
        type(CircuitDigitalGroud), allocatable :: grouds(:)
        type(CircuitVoltageSource), allocatable :: vol_sources(:)
        type(CircuitPulseSource), allocatable :: pulse_sources(:)
        type(CircuitRLC), allocatable :: rlc_sources(:)
        type(CircuitIMN), allocatable :: imn_sources(:)

    contains

        procedure :: init   => initExtCircuits          ! 只负责创建数组, 具体参数需要额外设定
        procedure :: update => updateExtCircuits
        procedure :: zero => zeroExtCircuits

        procedure :: show => showExtCircuits

        procedure :: dump => dumpExtCircuits
        procedure :: load => loadExtCircuits

        procedure :: destroy => destroyExtCircuits

    end type ExtCircuits


    contains

    subroutine initExtCircuits(this, circuit_num, each_circuit_num)
        class(ExtCircuits), intent(inout) :: this
        integer(4), intent(in) :: circuit_num
        integer(4), intent(in) :: each_circuit_num(circuit_type_number)
        integer(4) :: i, size, rank, ierr

        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

        if (circuit_num > 0 .and. sum(abs(each_circuit_num)) == circuit_num) then
            call this%destroy()

            this%circuits_number = circuit_num
            this%circuits_size = each_circuit_num

            allocate(this%circuits_type(this%circuits_number))
            allocate(this%circuits_index(this%circuits_number))
            allocate(this%circuits_metal(this%circuits_number))

            do i = 1, circuit_type_number
                select case(i)
                    case(circuit_type_digital_ground)
                        if (this%circuits_size(i) > 0) allocate(this%grouds(this%circuits_size(i)))

                    case(circuit_type_voltage_source)
                        if (this%circuits_size(i) > 0) allocate(this%vol_sources(this%circuits_size(i)))

                    case(circuit_type_pulse_source)
                        if (this%circuits_size(i) > 0) allocate(this%pulse_sources(this%circuits_size(i)))

                    case(circuit_type_rlc_source)
                        if (this%circuits_size(i) > 0) allocate(this%rlc_sources(this%circuits_size(i)))

                    case(circuit_type_imn_source)
                        if (this%circuits_size(i) > 0) allocate(this%imn_sources(this%circuits_size(i)))

                end select
            end do

            call this%zero()

        else
            if (0 == rank) write(*, *) "Invalid input parameters."

        end if

    end subroutine initExtCircuits


    subroutine updateExtCircuits(this, sim_time, MT)
        class(ExtCircuits), intent(inout) :: this
        real(8), intent(in) :: sim_time
        type(Materials), intent(inout) :: MT
        integer(4) :: i, rank, ierr

        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

        do i = 1, this%circuits_number
            select case(this%circuits_type(i))
                case(circuit_type_digital_ground)
                    MT%metls(this%circuits_metal(i))%voltage = 0.d0

                case(circuit_type_voltage_source)
                    call this%vol_sources(this%circuits_index(i))%Update(sim_time)
                    MT%metls(this%circuits_metal(i))%voltage = this%vol_sources(this%circuits_index(i))%Voltage

                case(circuit_type_pulse_source)
                    call this%pulse_sources(this%circuits_index(i))%Update(sim_time)
                    MT%metls(this%circuits_metal(i))%voltage = this%pulse_sources(this%circuits_index(i))%voltage

                case(circuit_type_rlc_source)
                    associate(charge => MT%metls(this%circuits_metal(i))%charge, &
                              q0 => MT%metls(this%circuits_metal(i))%q0, &
                              eqc => MT%metls(this%circuits_metal(i))%capacitance, &
                              dt => this%rlc_sources(this%circuits_index(i))%ode_rlc%dt, &
                              Qconv => MT%metls(this%circuits_metal(i))%charge_one_step)

                        call this%rlc_sources(this%circuits_index(i))%Update(sim_time, MT%metls(this%circuits_metal(i))%voltage, Qconv/dt, eqc, q0)
                        charge = this%rlc_sources(this%circuits_index(i))%ode_rlc%out(3)
                        MT%metls(this%circuits_metal(i))%voltage = (charge - q0) / eqc

                        if (0 == rank) then
                            open(10, file="./EC.txt", position='append')
                                write(10, '(*(es20.12, 1x))') sim_time, MT%metls(this%circuits_metal(i))%voltage, &
                                                                        MT%metls(this%circuits_metal(i))%capacitance, &
                                                                        MT%metls(this%circuits_metal(i))%charge, &
                                                                        this%rlc_sources(this%circuits_index(i))%ode_rlc%out, &
                                                                        rlc_source_voltage, &
                                                                        rlc_electrode_voltage, &
                                                                        rlc_Iconv, &
                                                                        q0

                            close(10)
                        end if
                    end associate

                case(circuit_type_imn_source)
                    associate(charge => MT%metls(this%circuits_metal(i))%charge, &
                              q0 => MT%metls(this%circuits_metal(i))%q0, &
                              eqc => MT%metls(this%circuits_metal(i))%capacitance, &
                              dt => this%imn_sources(this%circuits_index(i))%ode_imn%dt, &
                              Qconv => MT%metls(this%circuits_metal(i))%charge_one_step)

                        call this%imn_sources(this%circuits_index(i))%Update(sim_time, MT%metls(this%circuits_metal(i))%voltage, Qconv/dt, eqc, q0)
                        charge = this%imn_sources(this%circuits_index(i))%ode_imn%out(6)
                        MT%metls(this%circuits_metal(i))%voltage = (charge - q0) / eqc

                        if (0 == rank) then
                            open(10, file="./EC.txt", position='append')
                                write(10, '(*(es20.12, 1x))') sim_time, MT%metls(this%circuits_metal(i))%voltage, &
                                                                        MT%metls(this%circuits_metal(i))%capacitance, &
                                                                        MT%metls(this%circuits_metal(i))%charge, &
                                                                        this%imn_sources(this%circuits_index(i))%ode_imn%out, &
                                                                        imn_source_voltage, &
                                                                        imn_electrode_current, &
                                                                        imn_electrode_voltage, &
                                                                        imn_Iconv, &
                                                                        q0
                            close(10)
                        end if
                    end associate

            end select
        end do

        MT%metls%charge_one_step = 0.d0

    end subroutine updateExtCircuits


    subroutine zeroExtCircuits(this)
        class(ExtCircuits), intent(inout) :: this
        integer(4) :: i

        do i = 1, this%circuits_number
            select case(this%circuits_type(i))
                case(circuit_type_digital_ground)

                case(circuit_type_voltage_source)
                    call this%vol_sources(this%circuits_index(i))%Zero()

                case(circuit_type_pulse_source)
                    call this%pulse_sources(this%circuits_index(i))%Zero()

                case(circuit_type_rlc_source)
                    call this%rlc_sources(this%circuits_index(i))%Zero()

                case(circuit_type_imn_source)
                    call this%imn_sources(this%circuits_index(i))%Zero()

            end select
        end do

    end subroutine zeroExtCircuits


    subroutine showExtCircuits(this)
        class(ExtCircuits), intent(inout) :: this
        integer(4) :: i, size, rank, ierr

        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

        if (0 == rank .and. this%circuits_number > 0) then
            write(*, *) ''
            write(*, '(a)') "ExtCircuits information:"

            do i = 1, this%circuits_number
                select case(this%circuits_type(i))
                    case(circuit_type_digital_ground)
                        write(*, '(i4, 2x, a, 2x, a)') i, 'digital ground', &
                                trim(this%grouds(this%circuits_index(i))%IOName%DataName%str)
                    case(circuit_type_voltage_source)
                        write(*, '(i4, 2x, a, 2x, a, 2x, i, i)') i, 'voltage source', &
                                trim(this%vol_sources(this%circuits_index(i))%IOName%DataName%str), &
                                this%vol_sources(this%circuits_index(i))%NFrequency
                        write(*, '(20x, *(es12.4))') this%vol_sources(this%circuits_index(i))%Frequency
                        write(*, '(20x, *(es12.4))') this%vol_sources(this%circuits_index(i))%Amplitude
                        write(*, '(20x, *(es12.4))') this%vol_sources(this%circuits_index(i))%Phase
                        write(*, '(20x, *(es12.4))') this%vol_sources(this%circuits_index(i))%Voltage, &
                                this%vol_sources(this%circuits_index(i))%Vdc

                    case(circuit_type_pulse_source)
                        write(*, '(i4, 2x, a, 2x, a, 2x, i)') i, 'pulse source', &
                                trim(this%pulse_sources(this%circuits_index(i))%IOName%DataName%str), &
                                this%pulse_sources(this%circuits_index(i))%seg
                        write(*, '(20x, *(es12.4))') this%pulse_sources(this%circuits_index(i))%time
                        write(*, '(20x, *(es12.4))') this%pulse_sources(this%circuits_index(i))%amplitude
                        write(*, '(20x, *(es12.4))') this%pulse_sources(this%circuits_index(i))%voltage

                    case(circuit_type_rlc_source)
                        write(*, '(i4, 2x, a, 2x, a, 2x, i4, *(es12.4))') i, 'rlc source', &
                                trim(this%rlc_sources(this%circuits_index(i))%IOName%DataName%str), &
                                rlc_source_frequency_num, &
                                rlc_Rs, &
                                rlc_Cs, &
                                rlc_Ls
                        write(*, '(20x, *(es12.4))') rlc_source_frequency
                        write(*, '(20x, *(es12.4))') rlc_source_amplitude
                        write(*, '(20x, *(es12.4))') rlc_source_phase

                    case(circuit_type_imn_source)
                        write(*, '(i4, 2x, a, 2x, a, 2x, i4)') i, 'imn source', &
                                trim(this%imn_sources(this%circuits_index(i))%IOName%DataName%str), &
                                imn_source_frequency_num
                        write(*, '(20x, *(es12.4))') imn_source_frequency
                        write(*, '(20x, *(es12.4))') imn_source_amplitude
                        write(*, '(20x, *(es12.4))') imn_source_phase
                        write(*, '(a20, es12.4)') 'Rs ', imn_Rs
                        write(*, '(a20, es12.4)') 'Cm1 ', imn_Cm1
                        write(*, '(a20, es12.4)') 'Cm2 ', imn_Cm2
                        write(*, '(a20, es12.4)') 'Lm ', imn_Lm
                        write(*, '(a20, es12.4)') 'Rm ', imn_Rm
                        if (imn_is_open_stray_capacitance) write(*, '(a32)') 'open stray'
                        if (.not. imn_is_open_stray_capacitance) write(*, '(a32)') 'close stray'
                        write(*, '(a20, es12.4)') 'C_stray ', imn_C_stray
                        write(*, '(a20, es12.4)') 'L_stray ', imn_L_stray
                        write(*, '(a20, es12.4)') 'R_stray ', imn_R_stray

                end select
            end do
        end if

    end subroutine showExtCircuits


    subroutine dumpExtCircuits(this)
        class(ExtCircuits), intent(inout) :: this
        integer(4) :: i

        do i = 1, this%circuits_number
            select case(this%circuits_type(i))
                case(circuit_type_digital_ground)

                case(circuit_type_voltage_source)
                    call this%vol_sources(this%circuits_index(i))%Dump()

                case(circuit_type_pulse_source)
                    call this%pulse_sources(this%circuits_index(i))%Dump()

                case(circuit_type_rlc_source)
                    call this%rlc_sources(this%circuits_index(i))%Dump()

                case(circuit_type_imn_source)
                    call this%imn_sources(this%circuits_index(i))%Dump()

            end select
        end do

    end subroutine dumpExtCircuits


    subroutine loadExtCircuits(this)
        class(ExtCircuits), intent(inout) :: this
        integer(4) :: i

        do i = 1, this%circuits_number
            select case(this%circuits_type(i))
                case(circuit_type_digital_ground)

                case(circuit_type_voltage_source)
                    call this%vol_sources(this%circuits_index(i))%Load()

                case(circuit_type_pulse_source)
                    call this%pulse_sources(this%circuits_index(i))%Load()

                case(circuit_type_rlc_source)
                    call this%rlc_sources(this%circuits_index(i))%Load()

                case(circuit_type_imn_source)
                    call this%imn_sources(this%circuits_index(i))%Load()

            end select
        end do

    end subroutine loadExtCircuits



    subroutine destroyExtCircuits(this)
        class(ExtCircuits), intent(inout) :: this
        integer(4) :: i

        do i = 1, this%circuits_number
            select case(this%circuits_type(i))
                case(circuit_type_digital_ground)

                case(circuit_type_voltage_source)
                    call this%vol_sources(this%circuits_index(i))%Destroy()

                case(circuit_type_pulse_source)
                    call this%pulse_sources(this%circuits_index(i))%Destroy()

                case(circuit_type_rlc_source)
                    call this%rlc_sources(this%circuits_index(i))%Destroy()

                case(circuit_type_imn_source)
                    call this%imn_sources(this%circuits_index(i))%Destroy()

            end select
        end do

        this%circuits_number = 0
        this%circuits_size = 0

        if (allocated(this%circuits_type)) deallocate(this%circuits_type)
        if (allocated(this%circuits_index)) deallocate(this%circuits_index)
        if (allocated(this%circuits_metal)) deallocate(this%circuits_metal)
        if (allocated(this%grouds)) deallocate(this%grouds)
        if (allocated(this%vol_sources)) deallocate(this%vol_sources)
        if (allocated(this%pulse_sources)) deallocate(this%pulse_sources)
        if (allocated(this%rlc_sources)) deallocate(this%rlc_sources)
        if (allocated(this%imn_sources)) deallocate(this%imn_sources)

    end subroutine destroyExtCircuits

end module ModuleExtCircuit
