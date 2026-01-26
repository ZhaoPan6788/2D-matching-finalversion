Module ModuleFTime
    implicit none

    real(8),    parameter :: TIME_CLOCK_TICKS = 1000000.d0
    integer(4), parameter :: TIME_UNIT_TYPE_SECOND = 1
    integer(4), parameter :: TIME_UNIT_TYPE_MILLISECOND = 2
    integer(4), parameter :: TIME_UNIT_TYPE_MICROSECOND = 3


    type SimpleTimer
        integer(8) :: time_start = 0_8      ! 开始计时的时刻, 只有调用 init 或者 rset 才会被改变
        integer(8) :: time_last  = 0_8      ! 记录上次调用 sythisem_clock 的时刻
        integer(8) :: time_stop  = 0_8      ! 结束计时的时刻, 只有调用 time 才会改变

        integer(4) :: time_unit_type = TIME_UNIT_TYPE_SECOND
        real(8)    :: time_factor = 1.d0

    contains

        procedure :: init => initSimpleTimer
        procedure :: time => timeSimpleTimer
        procedure :: show => showSimpleTimer
        procedure :: getT => getTime
        procedure :: getD => getDifTime
        procedure :: unit => setSimpleTimerUnit

    end type SimpleTimer


    contains

        subroutine initSimpleTimer(this)
            class(SimpleTimer), intent(inout) :: this

            this%time_start = 0_8
            this%time_last  = 0_8
            this%time_stop  = 0_8

            call system_clock(this%time_start)
            this%time_last = this%time_start
            this%time_stop = this%time_start

        end subroutine initSimpleTimer


        subroutine timeSimpleTimer(this)
            class(SimpleTimer), intent(inout) :: this

            this%time_last = this%time_stop
            call system_clock(this%time_stop)

        end subroutine timeSimpleTimer


        subroutine showSimpleTimer(this)
            class(SimpleTimer), intent(inout) :: this

            call this%time()
            write(*, '(a, f16.4, 2x, a, f16.4)') 'Cost time: ', this%getT(), "Interval time: ", this%getD()

        end subroutine showSimpleTimer


        subroutine setSimpleTimerUnit(this, type)
            class(SimpleTimer), intent(inout) :: this
            integer(4), intent(in) :: type

            if (TIME_UNIT_TYPE_SECOND == type) then
                this%time_unit_type = TIME_UNIT_TYPE_SECOND
                this%time_factor = 1.d0

            else if (TIME_UNIT_TYPE_MILLISECOND == type) then
                this%time_unit_type = TIME_UNIT_TYPE_MILLISECOND
                this%time_factor = 1000.d0

            else if (TIME_UNIT_TYPE_MICROSECOND == type) then
                this%time_unit_type = TIME_UNIT_TYPE_MICROSECOND
                this%time_factor = 1000000.d0

            else
                this%time_unit_type = TIME_UNIT_TYPE_MILLISECOND
                this%time_factor = 1000.d0

            end if

        end subroutine setSimpleTimerUnit


        function getTime(this)
            class(SimpleTimer), intent(inout) :: this
            real(8) :: getTime

            getTime = calTime(this%time_start, this%time_stop, this%time_factor)
        end


        function getDifTime(this)
            class(SimpleTimer), intent(inout) :: this
            real(8) :: getDifTime

            getDifTime = calTime(this%time_last, this%time_stop, this%time_factor)
        end


        function calTime(time_start, time_stop, factor)
            real(8) :: calTime
            integer(8), intent(in) :: time_start, time_stop
            real(8), intent(in) :: factor

            calTime = dble(time_stop - time_start) / TIME_CLOCK_TICKS * factor
        end

end Module ModuleFTime