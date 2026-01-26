program fortran_mpi
    use ModuleParticleBundle
    use ModuleRecombination

    implicit none

    integer(4), parameter :: zstart = 1
    integer(4), parameter :: zend = 11
    integer(4), parameter :: rstart = 1
    integer(4), parameter :: rend = 21
    integer(4), parameter :: ns = 3

    integer(4) :: i, j, k
    type(ParticleBundle) :: pb(0:ns)         ! ele, F-, CF3+, CF3-
    type(ParticleOne) :: one
    type(Recombination2D) :: re2d
    real(8) :: RR, dz, dr
    integer(4) :: reactions(2, 3)
    real(8) :: reaction_rate(2)
    real(8) :: density2d(zstart:zend-1, rstart:rend-1, 0:ns)
    real(8) :: volume(zstart:zend-1, rstart:rend-1)

    reactions(1, 1) = 0
    reactions(1, 2) = 2
    reactions(1, 3) = 0

    reactions(2, 1) = 1
    reactions(2, 2) = 2
    reactions(2, 3) = 0

    reaction_rate(1) = 10000
    reaction_rate(2) = 5000

    dz = 0.1d0
    dr = 0.1d0
    do i = rstart, rend-1
        volume(:, i) = PI * dr * dr * dz * dble(i*i - (i-1)*(i-1))
    end do

    do k = 0, ns
        call pb(k)%init(500000, 10, 10000000, 0.25d0, 0.5d0, 1.5d0)
    end do

    call re2d%init(ns, 2, reactions, reaction_rate, zstart, zend-1, rstart, rend-1, dz, dr)
    call re2d%show()

    ! 生成粒子
    do k = 0, ns
        do i = 1, pb(k)%size
            call random_number(RR)
            one%Z = dble(zstart) - 1.d0 + (dble(zend) - dble(zstart)) * RR

            call random_number(RR)
            one%R = dble(rstart) - 1.d0 + (dble(rend) - dble(rstart)) * RR

            one%WQ = one%R

            call pb(k)%addone(one)
        end do
    end do

    ! dump
    density2d = 0.d0
    do k = 0, ns
        do i = 1, pb(k)%NPar
            density2d(ceiling(pb(k)%PO(i)%Z), ceiling(pb(k)%PO(i)%R), k) = &
                density2d(ceiling(pb(k)%PO(i)%Z), ceiling(pb(k)%PO(i)%R), k) + pb(k)%PO(i)%WQ
        end do
    end do

    open(10, file="raw_par.txt")
        do k = 0, ns
            do i = rstart, rend-1
                write(10, '(*(es20.6, 1x))') density2d(:, i, k)/volume(:, i)
            end do
        end do
    close(10)

    write(*, *) pb%NPar
    call re2d%recb(ns, pb, 1.d-10)
    write(*, *) pb%NPar

    ! dump
    density2d = 0.d0
    do k = 0, ns
        do i = 1, pb(k)%NPar
            density2d(ceiling(pb(k)%PO(i)%Z), ceiling(pb(k)%PO(i)%R), k) = &
                density2d(ceiling(pb(k)%PO(i)%Z), ceiling(pb(k)%PO(i)%R), k) + pb(k)%PO(i)%WQ
        end do
    end do

    open(10, file="par.txt")
        do k = 0, ns
            do i = rstart, rend-1
                write(10, '(*(es20.6, 1x))') density2d(:, i, k)/volume(:, i)
            end do
        end do
    close(10)
    
    do k = 0, ns
        call pb(k)%Destroy()
    end do
    call re2d%destroy()

end program fortran_mpi
