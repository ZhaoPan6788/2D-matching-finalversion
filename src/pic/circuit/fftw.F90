module ModuleFFTW
    use Constants

    implicit none

contains

    subroutine fftw(n, I_in, U_in, amp_I, pha_I, amp_U, pha_U)
        integer(4), intent(in) :: n
        real(8), intent(in) :: I_in(n), U_in(n)
        real(8), intent(out) :: amp_I(n), pha_I(n), amp_U(n), pha_U(n)
        integer(4) :: i

        do i = 1, n
            amp_I(i) = abs(I_in(i))
            pha_I(i) = 0.d0
            amp_U(i) = abs(U_in(i))
            pha_U(i) = 0.d0
        end do

    end subroutine fftw


    subroutine get_fftw(n, I_in, U_in, dt, amp_I, pha_I, amp_U, pha_U, freq)
        integer(4), intent(in) :: n
        real(8), intent(in) :: I_in(n), U_in(n)
        real(8), intent(in) :: dt
        real(8), intent(inout) :: amp_I(n), pha_I(n), amp_U(n), pha_U(n), freq(n)
        integer(4) :: i

        call fftw(n, I_in, U_in, amp_I, pha_I, amp_U, pha_U)

        do i = 1, n
            freq(i) = 1.0d0 / (dble(n) * dt) * dble(i-1)
        end do

    end subroutine get_fftw


    subroutine get_base_freq_info(n, freq, amp, pha, freq_search, freq_out, amp_out, pha_out)
        integer(4), intent(in) :: n
        real(8), intent(in) :: freq(n), amp(n), pha(n)
        real(8), intent(in) :: freq_search
        real(8), intent(out) :: freq_out, amp_out, pha_out

        integer(4) :: i_min, i
        real(8) :: min_diff

        i_min = 1
        min_diff = abs(freq(1) - freq_search)

        do i = 2, n
            if (abs(freq(i) - freq_search) < min_diff) then
                i_min = i
                min_diff = abs(freq(i) - freq_search)
            end if
        end do

        freq_out = freq(i_min)
        amp_out = amp(i_min)
        pha_out = pha(i_min)

    end subroutine get_base_freq_info


    subroutine test_fftw()
        integer, parameter :: n = 1000
        real(8) :: dt, t(n), I_in(n), U_in(n)
        real(8) :: amp_I(n), pha_I(n), amp_U(n), pha_U(n), freq(n)
        real(8) :: freq0, freq_out, amp_out, pha_out
        integer :: i

        dt = 1.0d-3     ! Sampling interval: 1 kHz sampling
        freq0 = 50.0d0  ! Test frequency: 50 Hz

        do i = 1, n
            t(i) = (i - 1) * dt
            I_in(i) = sin(2.0d0 * 3.141592653589793d0 * freq0 * t(i))
            U_in(i) = cos(2.0d0 * 3.141592653589793d0 * freq0 * t(i))
        end do

        call get_fftw(n, I_in, U_in, dt, amp_I, pha_I, amp_U, pha_U, freq)

        call get_base_freq_info(n, freq, amp_U, pha_U, freq0, freq_out, amp_out, pha_out)

        print *, "===== FFTW TEST ====="
        print *, "Target Frequency:   ", freq0, "Hz"
        print *, "Detected Frequency: ", freq_out, "Hz"
        print *, "Current Amplitude:  ", amp_out
        print *, "Current Phase:      ", pha_out

    end subroutine test_fftw

end module ModuleFFTW