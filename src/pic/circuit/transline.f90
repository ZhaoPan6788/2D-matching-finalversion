module NonlinearTransmissionLine

    implicit none

    integer(4), parameter :: transline_boundary_type_voltage = 1
    integer(4), parameter :: transline_boundary_type_current = 2

    type TransLine
        integer(4) :: seg
        integer(4) :: Mx
        integer(4) :: left_boundary_type
        integer(4) :: right_boundary_type

        real(8) :: dx
        real(8) :: dt
        logical :: is_init

        real(8), allocatable :: voltage(:)    
        real(8), allocatable :: current(:)
        real(8), allocatable :: voltage_last(:)    
        real(8), allocatable :: current_last(:)

        real(8), allocatable :: R(:)
        real(8), allocatable :: G(:)
        real(8), allocatable :: L(:)
        real(8), allocatable :: C(:)

        contains

            procedure :: init => initTransLine
            procedure :: update => updateTransLine
            procedure :: print => printTransLine
            procedure :: destroy => destroyTransLine

    end type TransLine

    contains

        subroutine initTransLine(this, length, segment, time_step, left_boundary_type, right_boundary_type)
            class(TransLine), intent(inout) :: this
            real(8), intent(in) :: length, time_step
            integer(4), intent(in) :: segment, left_boundary_type, right_boundary_type

            if (length > 0 .and. segment > 0 .and. time_step > 0.d0) then
                this%seg = segment
                this%Mx = segment + 1
                this%dx = length / dble(segment)
                this%dt = time_step

                if (transline_boundary_type_voltage == left_boundary_type .or. &
                    transline_boundary_type_current == left_boundary_type) then
                    this%left_boundary_type = left_boundary_type
                else
                    write(*, *) "The input parameters of the TransLine are invalid."
                    stop
                end if

                if (transline_boundary_type_voltage == right_boundary_type .or. &
                    transline_boundary_type_current == right_boundary_type) then
                    this%right_boundary_type = right_boundary_type
                else
                    write(*, *) "The input parameters of the TransLine are invalid."
                    stop
                end if

                call this%destroy()
                allocate(this%voltage(this%Mx))
                allocate(this%voltage_last(this%Mx))
                allocate(this%current(this%Mx))
                allocate(this%current_last(this%Mx))

                this%voltage = 0.d0
                this%voltage_last = 0.d0
                this%current = 0.d0
                this%current_last = 0.d0

                allocate(this%R(this%seg))
                allocate(this%G(this%seg))
                allocate(this%L(this%seg))
                allocate(this%C(this%seg))

                this%is_init = .True.

            else
                write(*, *) "The input parameters of the TransLine are invalid."
                stop
            end if

        end subroutine initTransLine


        subroutine updateTransLine(this, left_input, left_output, right_input, right_output)
            class(TransLine), intent(inout) :: this
            real(8), intent(in) :: left_input, right_input
            real(8), intent(inout) :: left_output, right_output
            real(8) :: upm, unm, ipm, inm
            real(8) :: R0, G0, L0, C0
            real(8) :: R1, G1, L1, C1
            real(8) :: R01, G01, L01, C01
            integer(4) :: i

            if (this%is_init) then

                this%voltage_last = this%voltage
                this%current_last = this%current

                do i = 1, this%Mx
                    if (1 == i) then
                        if (transline_boundary_type_voltage == this%left_boundary_type) then
                            this%voltage(i) = left_input
                            this%voltage_last(i) = left_input

                            this%current(i) = this%current_last(i) + ((this%voltage_last(i) - this%voltage_last(i+1))/this%dx - this%R(i)*this%current_last(i)) * this%dt/this%L(i)
                            left_output = this%current(i)

                        else if (transline_boundary_type_current == this%left_boundary_type) then
                            this%current(i) = left_input
                            this%current_last(i) = left_input

                            this%voltage(i) = this%voltage_last(i) + ((this%current_last(i) - this%current_last(i+1))/this%dx - this%G(i)*this%voltage_last(i)) * this%dt/this%C(i)
                            left_output = this%voltage(i)

                        else
                            write(*, *) "Invalid boundary type."

                        end if

                    else if (i == this%Mx) then
                        if (transline_boundary_type_voltage == this%right_boundary_type) then
                            this%voltage(i) = right_input
                            this%voltage_last(i) = right_input

                            this%current(i) = this%current_last(i) + ((this%voltage_last(i-1) - this%voltage_last(i))/this%dx - this%R(i-1)*this%current_last(i)) * this%dt/this%L(i-1)
                            right_output = this%current(i)

                        else if (transline_boundary_type_current == this%right_boundary_type) then
                            this%current(i) = right_input
                            this%current_last(i) = right_input

                            this%voltage(i) = this%voltage_last(i) + ((this%current_last(i-1) - this%current_last(i))/this%dx - this%G(i-1)*this%voltage_last(i)) * this%dt/this%C(i-1)
                            right_output = this%voltage(i)

                        else
                            write(*, *) "Invalid boundary type."

                        end if

                    else
                        R0 = this%R(i-1); R1 = this%R(i); R01 = 0.5d0 * (R0 + R1)
                        G0 = this%G(i-1); G1 = this%G(i); G01 = 0.5d0 * (G0 + G1)
                        L0 = this%L(i-1); L1 = this%L(i); L01 = 0.5d0 * (L0 + L1)
                        C0 = this%C(i-1); C1 = this%C(i); C01 = 0.5d0 * (C0 + C1)

                        ! k+0.5
                        upm = (0.5d0 - 0.25d0*this%dt*G1/C1) * (this%voltage_last(i+1) + this%voltage_last(i)) &
                              - 0.5d0*this%dt/this%dx/C1 * (this%current_last(i+1) - this%current_last(i))

                        unm = (0.5d0 - 0.25d0*this%dt*G0/C0) * (this%voltage_last(i) + this%voltage_last(i-1)) &
                              - 0.5d0*this%dt/this%dx/C0 * (this%current_last(i) - this%current_last(i-1))

                        ipm = (0.5d0 - 0.25d0*this%dt*R1/L1) * (this%current_last(i+1) + this%current_last(i)) &
                              - 0.5d0*this%dt/this%dx/L1 * (this%voltage_last(i+1) - this%voltage_last(i))

                        inm = (0.5d0 - 0.25d0*this%dt*R0/L0) * (this%current_last(i) + this%current_last(i-1)) &
                              - 0.5d0*this%dt/this%dx/L0 * (this%voltage_last(i) - this%voltage_last(i-1))

                        ! k+1
                        this%voltage(i) = this%voltage_last(i) - 1.d0*this%dt/this%dx/C01 * (ipm - inm) &
                                          - 0.5d0*this%dt*G01/C01 * (upm + unm);
                        this%current(i) = this%current_last(i) - 1.d0*this%dt/this%dx/L01 * (upm - unm) &
                                          - 0.5d0*this%dt*R01/L01 * (ipm + inm)

                    end if
                end do

            else
                write(*, *) 'TransLine is not initialized.'
            end if
            
        end subroutine updateTransLine


        subroutine printTransLine(this)
            class(TransLine), intent(inout) :: this

            write(*, '(a30)') 'Transmission line configuration'
            write(*, '(a10, es20.4)') 'Length', dble(this%seg) * this%dx
            write(*, '(a10, i20)') 'Segment', this%seg
            write(*, '(a10, es20.4)') 'dx', this%dx
            write(*, '(a10, es20.4)') 'dt', this%dt
            write(*, '(a10, i20.4)') 'in', this%left_boundary_type
            write(*, '(a10, i20.4)') 'out', this%right_boundary_type
            write(*, '(a10, es20.4)') 'R', sum(this%R) / dble(this%seg)
            write(*, '(a10, es20.4)') 'G', sum(this%G) / dble(this%seg)
            write(*, '(a10, es20.4)') 'L', sum(this%L) / dble(this%seg)
            write(*, '(a10, es20.4)') 'C', sum(this%C) / dble(this%seg)

        end subroutine printTransLine


        subroutine destroyTransLine(this)
            class(TransLine), intent(inout) :: this

            if (allocated(this%voltage))      deallocate(this%voltage)
            if (allocated(this%voltage_last)) deallocate(this%voltage_last)
            if (allocated(this%current))      deallocate(this%current)
            if (allocated(this%current_last)) deallocate(this%current_last)

            if (allocated(this%R)) deallocate(this%R)
            if (allocated(this%G)) deallocate(this%G)
            if (allocated(this%L)) deallocate(this%L)
            if (allocated(this%C)) deallocate(this%C)

            this%is_init = .False.

        end subroutine destroyTransLine

end module NonlinearTransmissionLine
