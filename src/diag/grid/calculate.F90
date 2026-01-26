module ModuleTempCalculate
    implicit none
    interface anyavg
        procedure AnyAvg1D, AnyAvg2D, AnyAvg3D
    end interface anyavg
contains
    Subroutine AnyAvg1D(rN, rM)
        Real(8), Intent(IN) :: rN(:)
        Real(8), Intent(OUT) :: rM(:)
        integer :: n, m, i, j, e, k
        real :: rs, re, rl, rtmp, add
        n = size(rN)
        m = size(rM)
        if (n < m) Return
        if (n == m) then
            rM = rN
            return
        end if
        If (mod(n, m) == 0) then !// 常规整个数平均
            e = n/m
            Do i = 1, m
                rs = 0.0
                Do j = (i - 1)*e + 1, i*e
                    rs = rs + rN(j)/e
                End Do
                rM(i) = rs
            End Do
        Else !// 小数个数平均
            re = n*1.0/m*1.0
            rl = 0.0
            k = 1
            Do i = 1, m
                rs = 0.0
                add = re
                rtmp = rl - int(rl)
                if (rtmp > 0.0001) then !// 如果上次有剩余
                    rs = rs + rN(k)*(1.0 - rtmp)/re
                    add = add - (1.0 - rtmp)
                    k = k + 1
                end if
                Do while (add >= 1.0)
                    rs = rs + rN(k)/re
                    k = k + 1
                    add = add - 1.0
                End Do
                if (add > 0.0001) then !//如果本次有剩余
                    rs = rs + rN(k)*add/re
                end if
                rM(i) = rs
                rl = rl + re
            End Do
        End If
    End Subroutine AnyAvg1D

    Subroutine AnyAvg2D(rN, rM)
        Real(8), Intent(IN) :: rN(:, :)
        Real(8), Intent(OUT) :: rM(:, :)
        Real(8) :: Temp2D(Size(rM, 1), Size(rN, 2))
        integer :: i
        DO i = 1, Size(rN, 2)
            Call AnyAvg(rN(:, i), Temp2D(:, i))
        ENd DO

        DO i = 1, Size(rM, 1)
            Call AnyAvg(Temp2D(i, :), rM(i, :))
        ENd DO
    End Subroutine AnyAvg2D

    Subroutine AnyAvg3D(rN)
        Real(8), Intent(IN) :: rN(:, :, :)
        ! Real(8) :: rM(:,:)
        ! write (*, *) "=== Error: 3D unfinished ==="
        error stop "=== Error: 3D unfinished ==="
        call execute_command_line(' ')
		stop
    End Subroutine AnyAvg3D
end module ModuleTempCalculate