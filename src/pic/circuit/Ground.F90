module ModuleCircuitDigitalGround
    use ModuleFileName
    use ModuleParallelDump

    implicit none

    type CircuitDigitalGroud
        type(FileName) :: IOName

    contains

        procedure :: init => initCircuitDigitalGroud

    end type CircuitDigitalGroud

    contains

        subroutine initCircuitDigitalGroud(this, name)
            class(CircuitDigitalGroud), intent(inout) :: this
            character(*), optional, intent(in) :: name

            if (present(name)) then
                call this%IOName%Init(name, RESTART_FILE_NAME)
            else
                call this%IOName%Init("DigitalGroud", RESTART_FILE_NAME)
            end if

        end subroutine initCircuitDigitalGroud

end module ModuleCircuitDigitalGround
