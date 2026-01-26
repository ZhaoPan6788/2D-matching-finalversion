Module ModuleGrid
    use ModuleFileName
    implicit none

    integer(4), parameter :: diag_type_average  = 1
    integer(4), parameter :: diag_type_interval = 2
    integer(4), parameter :: diag_type_steady   = 3

    type :: Grid0DS
        type(FileName) :: IOName
        integer(4) :: Ns = 0
        integer(4) :: Nt = 0
        integer(4) :: kt = 1

        real(8), allocatable ::  value(:, :)

        integer(4) :: shift = 0
        integer(4) :: timer = 0
        integer(4) :: g_period = 1
        integer(8) :: g_timer = 0_8

        integer(4) :: g_type = diag_type_average

        contains

            procedure :: Init => InitGrid0DS
            procedure :: Dump => DumpGrid0DS
            procedure :: Update => UpdateGrid0DS
            procedure :: Rescale => RescaleGrid0DS
            procedure :: Reset => ResetGrid0DS
            procedure :: Destroy => DestroyGrid0DS

    end type Grid0DS


    type :: Grid1DS
        type(FileName) :: IOName
        integer(4) :: Nx = 0
        integer(4) :: Ns = 0
        integer(4) :: Nt = 0
        integer(4) :: kt = 1

        real(8), allocatable ::  value(:, :, :)

        integer(4) :: shift = 0
        integer(4) :: timer = 0
        integer(4) :: g_period = 1
        integer(8) :: g_timer = 0_8

        integer(4) :: g_type = diag_type_average

        contains

            procedure :: Init => InitGrid1DS
            procedure :: Dump => DumpGrid1DS
            procedure :: Update => UpdateGrid1DS
            procedure :: Rescale => RescaleGrid1DS
            procedure :: Reset => ResetGrid1DS
            procedure :: Destroy => DestroyGrid1DS

    end type Grid1DS

    contains

        ! 0D
        subroutine InitGrid0DS(this, name, Ns, Nt, kt, dtype, s_timer)
            class(Grid0DS), intent(inout)  :: this
            character(*), intent(in) :: name
            integer(4), intent(in) :: Ns, Nt, kt, dtype, s_timer

            call this%Destroy()
            call this%IOName%Init(name, DIAG_FILE_NAME)
            call this%IOName%SetExte(FILE_EXTENSION_MODE_DAT)

            if (Ns > 0 .and. Nt > 0 .and. kt > 0) then
                this%Ns = Ns
                this%Nt = Nt
                this%kt = kt
                this%g_timer = s_timer

            else
                write(*, *) "Invalid (Ns, Nt, kt)."

            end if

            this%g_type = dtype
            if (diag_type_average == dtype) then
                allocate(this%value(this%Ns, 1))

            else if (diag_type_interval == dtype) then
                allocate(this%value(this%Ns, 1))

            else if (diag_type_steady == dtype) then
                allocate(this%value(this%Ns, this%Nt/this%kt))
                this%g_timer = 0_8

            else
                write(*, *) "Invalid diag type."

            end if

            call this%Reset()

        end subroutine InitGrid0DS


        subroutine DumpGrid0DS(this)
            class(Grid0DS), intent(inout)  :: this
            integer(4) :: i, nn

            open (10, file=this%IOName%FullName%str, position='append')

                call this%Rescale()
                if (diag_type_steady == this%g_type) then
                    nn = this%Nt / this%kt
                    do i = 1, nn
                        write(10, '(*(es22.14e3, 1x))') (dble(i)-0.5d0)/dble(nn), this%value(:, i)

                    end do

                else
                    write(10, '(i22, 1x, *(es22.14e3, 1x))') this%g_timer, this%value(:, 1)

                end if
            close(10)

        end subroutine DumpGrid0DS


        subroutine UpdateGrid0DS(this, A0D)
            class(Grid0DS), intent(inout)  :: this
            real(8), intent(in) :: A0D

            if (diag_type_average == this%g_type) then
                this%shift = this%shift + 1
                this%value(this%shift, 1) = this%value(this%shift, 1) + A0D

                if (this%shift == this%Ns) then
                    this%timer = this%timer + 1
                    this%shift = 0
                end if

                if (this%timer == this%Nt) then
                    this%g_timer = this%g_timer + 1
                    call this%Dump()
                    this%timer = 0
                    this%value = 0.d0
                end if

            else if (diag_type_interval == this%g_type) then
                this%shift = this%shift + 1
                this%value(this%shift, 1) = this%value(this%shift, 1) + A0D

                if (this%shift == this%Ns) then
                    this%g_timer = this%g_timer + 1
                    call this%Dump()
                    this%shift = 0
                    this%value = 0.d0
                end if

            else if (diag_type_steady == this%g_type) then
                this%shift = this%shift + 1
                this%value(this%shift, this%g_period) = this%value(this%shift, this%g_period) + A0D

                if (this%shift == this%Ns) then
                    this%timer = this%timer + 1
                    this%g_period = this%timer / this%kt + 1
                    this%shift = 0
                end if

                if (this%timer == this%Nt) then
                    this%g_timer = this%g_timer + 1
                    this%timer = 0
                    this%g_period = 1

                end if

            end if

        end subroutine UpdateGrid0DS


        subroutine RescaleGrid0DS(this)
            class(Grid0DS), intent(inout)  :: this
            integer(4) :: i, nn

            if (diag_type_average == this%g_type) then
                this%value = this%value / dble(this%Nt)
            
            else if (diag_type_steady == this%g_type) then
                nn = this%Nt / this%kt
                do i = 1, nn
                    this%value(:, i) = this%value(:, i) / dble(this%g_timer) / dble(this%kt)
                end do

            end if

        end subroutine  RescaleGrid0DS


        subroutine ResetGrid0DS(this)
            class(Grid0DS), intent(inout)  :: this

            this%value = 0.d0
            this%shift = 0
            this%timer = 0
            this%g_period = 1

        end subroutine ResetGrid0DS   


        subroutine DestroyGrid0DS(this)
            class(Grid0DS), intent(inout)  :: this

            if(allocated(this%value)) deallocate(this%value)

        end subroutine DestroyGrid0DS   


        ! 1D
        subroutine InitGrid1DS(this, name, Nx, Ns, Nt, kt, dtype, s_timer)
            class(Grid1DS), intent(inout) :: this
            character(*), intent(in) :: name
            integer(4), intent(in) :: Nx, Ns, Nt, kt, dtype, s_timer

            call this%Destroy()
            call this%IOName%Init(name, DIAG_FILE_NAME)
            call this%IOName%SetExte(FILE_EXTENSION_MODE_DAT)

            if (Nx > 0 .and. Ns > 0 .and. Nt > 0 .and. kt > 0) then
                this%Nx = Nx
                this%Ns = Ns
                this%Nt = Nt
                this%kt = kt
                this%g_timer = s_timer

            else
                write(*, *) "Invalid (Nx, Ns, Nt, kt)."

            end if

            this%g_type = dtype
            if (diag_type_average == dtype) then
                allocate(this%value(this%Nx, this%Ns, 1))

            else if (diag_type_interval == dtype) then
                allocate(this%value(this%Nx, this%Ns, 1))

            else if (diag_type_steady == dtype) then
                allocate(this%value(this%Nx, this%Ns, this%Nt/this%kt))
                this%g_timer = 0_8

            else
                write(*, *) "Invalid diag type."

            end if

            call this%Reset()

        end subroutine InitGrid1DS


        subroutine DumpGrid1DS(this)
            class(Grid1DS), intent(inout)  :: this
            integer(4) :: i, j, nn

            open (10, file=this%IOName%FullName%str, position='append')

                call this%Rescale()
                if (diag_type_steady == this%g_type) then
                    nn = this%Nt / this%kt
                    
                    do i = 1, nn
                        do j = 1, this%Nx
                            write(10, '(*(es22.14e3, 1x))') (dble(i)-0.5d0)/dble(nn), dble(j-1)/dble(this%Nx-1), this%value(j, :, i)
                        end do
                    end do

                else
                    do j = 1, this%Nx
                        write(10, '(i22, 1x, *(es22.14e3, 1x))') this%g_timer, dble(j-1)/dble(this%Nx-1), this%value(j, :, 1)
                    end do

                end if
            close(10)

        end subroutine DumpGrid1DS


        subroutine UpdateGrid1DS(this, A1D)
            class(Grid1DS), intent(inout)  :: this
            real(8), intent(in) :: A1D(this%Nx)

            if (diag_type_average == this%g_type) then
                this%shift = this%shift + 1
                this%value(:, this%shift, 1) = this%value(:, this%shift, 1) + A1D

                if (this%shift == this%Ns) then
                    this%timer = this%timer + 1
                    this%shift = 0
                end if

                if (this%timer == this%Nt) then
                    this%g_timer = this%g_timer + 1
                    call this%Dump()
                    this%timer = 0
                    this%value = 0.d0
                end if

            else if (diag_type_interval == this%g_type) then
                this%shift = this%shift + 1
                this%value(:, this%shift, 1) = this%value(:, this%shift, 1) + A1D

                if (this%shift == this%Ns) then
                    this%g_timer = this%g_timer + 1
                    call this%Dump()
                    this%shift = 0
                    this%value = 0.d0
                end if

            else if (diag_type_steady == this%g_type) then
                this%shift = this%shift + 1
                this%value(:, this%shift, this%g_period) = this%value(:, this%shift, this%g_period) + A1D

                if (this%shift == this%Ns) then
                    this%timer = this%timer + 1
                    this%g_period = this%timer / this%kt + 1
                    this%shift = 0
                end if

                if (this%timer == this%Nt) then
                    this%g_timer = this%g_timer + 1
                    this%timer = 0
                    this%g_period = 1

                end if

            end if

        end subroutine UpdateGrid1DS


        subroutine RescaleGrid1DS(this)
            class(Grid1DS), intent(inout)  :: this
            integer(4) :: i, nn

            if (diag_type_average == this%g_type) then
                this%value = this%value / dble(this%Nt)
            
            else if (diag_type_steady == this%g_type) then
                nn = this%Nt / this%kt
                do i = 1, nn
                    this%value(:, :, i) = this%value(:, :, i) / dble(this%g_timer) / dble(this%kt)
                end do

            end if

        end subroutine  RescaleGrid1DS


        subroutine ResetGrid1DS(this)
            class(Grid1DS), intent(inout) :: this

            this%value = 0.d0
            this%shift = 0
            this%timer = 0
            this%g_period = 1

        end subroutine ResetGrid1DS


        subroutine DestroyGrid1DS(this)
            class(Grid1DS), intent(inout) :: this

            if(allocated(this%value)) deallocate(this%value)

        end subroutine DestroyGrid1DS   

end Module ModuleGrid          
