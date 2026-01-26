Module ModuleParticleOne
    use Constants

    implicit none

    Type :: ParticleOne

        real(8) :: Z, R, Vz, Vr, Vt, Az, Ar, WQ

    contains

        procedure :: PosInit    => PositionRandomInitializationParticleOne
        procedure :: VelInpInit => VelocityInputInitializationParticleOne
        procedure :: VelMaxInit => VelocityMaxwellianInitializationParticleOne
        procedure :: VelRanInit => VelocityRandomInitializationParticleOne
        procedure :: VecRanXYZInit => VelocityRandomInitializationParticleOneXYZ
        procedure :: AccInpInit => AccelerationInputInitializationParticleOne
        procedure :: WqInpInit  => WeightInputInitializationParticleOne

        procedure :: toXYZ => CylindricalToRectangular
        procedure :: toZRT => RectangularToCylindrical

        procedure :: PosRes => PositionRescaleParticleOne
        procedure :: VelRes => VelocityRescaleParticleOne
        procedure :: AccRes => AccelerationRescaleParticleOne
        procedure :: Energy => EnergyParticleOne

        procedure :: Copy => CopyParticleOne
        procedure :: Swap => SwapParticleOne

    end Type ParticleOne

    contains

        subroutine PositionRandomInitializationParticleOne(PO,ZL,ZU,RL,RU)
            Class(ParticleOne), intent(inout) :: PO
            real(8), intent(in) :: ZL,ZU,RL,RU
            real(8) :: Theta

            call RANDOM_NUMBER(R)
            PO%R=RL+(RU-RL)*R
            call RANDOM_NUMBER(R)
            PO%Z=ZL+(ZU-ZL)*R

            call RANDOM_NUMBER(R)
            Theta=2.d0*PI*R

            return
        end subroutine

        subroutine VelocityInputInitializationParticleOne(PO, Vz, Vr, Vt)
            Class(ParticleOne), intent(inout) :: PO
            real(8), intent(in), optional :: Vz, Vr, Vt

            if (present(Vz)) then
                PO%Vz=Vz
            else
                PO%Vz=0.d0
            end if

            if (present(Vr)) then
                PO%Vr=Vr
            else
                PO%Vr=0.d0
            end if

            if (present(Vt)) then
                PO%Vt=Vt
            else
                PO%Vt=0.d0
            end if

            return
        end subroutine VelocityInputInitializationParticleOne

        subroutine VelocityMaxwellianInitializationParticleOne(PO,Mass,Temperature)
            Class(ParticleOne), intent(inout) :: PO
            real(8), intent(in) :: Mass,Temperature
            real(8) :: V,Beta,FuncA,FuncB
            real(8) :: Theta,CosTheta,SinTheta,Phi

            Beta=1.d0/(kB*Temperature)
            FuncA=1.d0
            FuncB=0.d0
            do while(FuncA>FuncB)
                call RANDOM_NUMBER(R)
                FuncA=R*R
                call RANDOM_NUMBER(R)
                FuncB=-exp*R*Dlog(R)
            end do
            V=DSqrt(-3.d0*Dlog(R)/Beta/Mass)
            call VelocityRandomInitializationParticleOne(PO,V)

            return
        end subroutine VelocityMaxwellianInitializationParticleOne

        subroutine VelocityRandomInitializationParticleOne(PO,V)
            Class(ParticleOne), intent(inout) :: PO
            real(8), intent(in) ::  V

            call PO%VecRanXYZInit(V)
            call PO%toZRT()

            return
        end subroutine VelocityRandomInitializationParticleOne


        subroutine VelocityRandomInitializationParticleOneXYZ(PO,V)
            Class(ParticleOne), intent(inout) :: PO
            real(8), intent(in) ::  V
            real(8) :: Phi,CosTheta,SinTheta

            call RANDOM_NUMBER(R)
            CosTheta=1.d0-2.d0*R
            SinTheta=Dsqrt(1.d0-cosTheta*cosTheta)
            call RANDOM_NUMBER(R)
            Phi=2.d0*PI*R

            PO%Vz = V * CosTheta                ! vz
            PO%Vr = V * SinTheta * dcos(Phi)    ! vx
            PO%Vt = V * SinTheta * dsin(Phi)    ! vy

        end subroutine VelocityRandomInitializationParticleOneXYZ


        subroutine AccelerationInputInitializationParticleOne(PO, Az, Ar)
            Class(ParticleOne), intent(inout) :: PO
            real(8), intent(in), optional :: Az, Ar

            if(present(Az)) then
                PO%Az=Az
            else
                PO%Az=0.d0
            end if

            if(present(Ar)) then
                PO%Ar=Ar
            else
                PO%Ar=0.d0
            end if

            return
        end subroutine AccelerationInputInitializationParticleOne

        subroutine WeightInputInitializationParticleOne(PO,Wq)
            Class(ParticleOne),intent(inout) :: PO
            real(8),intent(in) :: Wq

            PO%Wq=Wq

            return
        end subroutine WeightInputInitializationParticleOne


        subroutine CylindricalToRectangular(PO)
            Class(ParticleOne), intent(inout) :: PO
            ! real(8) :: Vx, Vy

            ! associate(Vr => PO%Vr, T => PO%T, Vt => PO%Vt)

            !     Vx = Vr*cos(T) - Vt*sin(T)
            !     Vy = Vr*sin(T) + Vt*cos(T)
            !     ! Vt是线速度，故不用乘R
            !     Vr = Vx
            !     Vt = Vy 

            ! end associate
        end subroutine CylindricalToRectangular


        subroutine RectangularToCylindrical(PO)
            Class(ParticleOne), intent(inout) :: PO
            ! real(8) :: Vx, Vy
            ! associate( T => PO%T, Vr => PO%Vr, Vt => PO%Vt)

            !     Vx = Vr
            !     Vy = Vt

            !     ! X  = R*cos(T)
            !     ! Y  = R*sin(T)
            !     ! Vr = (Vx*X+Vy*Y)/dsqrt(X*X + Y*Y)
            !     Vr = Vx*cos(T) + Vy*sin(T)
            !     ! Vt = R*(Vy*X-Vx*Y)/(X*X + Y*Y) Vt是线速度，故用乘R
            !     Vt = Vy*cos(T) - Vx*sin(T)

            ! end associate

        end subroutine RectangularToCylindrical


        subroutine PositionRescaleParticleOne(PO,XFactor)
            Class(ParticleOne), intent(inout) :: PO
            real(8), intent(in) :: XFactor

            PO%Z = PO%Z * XFactor
            PO%R = PO%R * XFactor

            return
        end subroutine PositionRescaleParticleOne

        subroutine VelocityRescaleParticleOne(PO,VFactor)
            Class(ParticleOne), intent(inout) :: PO
            real(8), intent(in) :: VFactor

            PO%Vz = PO%Vz * VFactor
            PO%Vr = PO%Vr * VFactor
            PO%Vt = PO%Vt * VFactor

            return
        end subroutine VelocityRescaleParticleOne

        subroutine AccelerationRescaleParticleOne(PO,AFactor)
            Class(ParticleOne), intent(inout) :: PO
            real(8), intent(in) :: AFactor

            PO%Az = PO%Az * AFactor
            PO%Ar = PO%Ar * AFactor

            return
        end subroutine AccelerationRescaleParticleOne

        subroutine CopyParticleOne(POD,POC)
            Class(ParticleOne), intent(inout) :: POD
            Class(ParticleOne), intent(in) :: POC

            select Type(POD)
                Type is(ParticleOne)
                    select Type(POC)
                        Type is(ParticleOne)
                            POD=POC
                    end select
            end select

            return
        end subroutine CopyParticleOne

        subroutine SwapParticleOne(POD,POC)
            Class(ParticleOne), intent(inout) :: POD
            Class(ParticleOne), intent(inout) :: POC
            Type(ParticleOne) :: POT

            select Type(POD)
                Type is(ParticleOne)
                    select Type(POC)
                        Type is(ParticleOne)
                            POT=POC
                            POC=POD
                            POD=POT
                    end select
            end select

            return
        end subroutine SwapParticleOne

        Elemental function EnergyParticleOne(PO,Mass,VFactor)
            Class(ParticleOne), intent(in) :: PO
            real(8), intent(in) :: Mass
            real(8), intent(in), optional :: VFactor
            real(8) :: EnergyParticleOne

            EnergyParticleOne=0.5d0*Mass*(PO%Vt*PO%Vt+PO%Vr*PO%Vr+PO%Vz*PO%Vz)
            if(present(VFactor)) EnergyParticleOne=EnergyParticleOne*VFactor*VFactor

            return
        end function EnergyParticleOne

end Module ModuleParticleOne
