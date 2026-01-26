Module ModuleDiagStep
    use ModuleControlFlow
    use DiagnosticsParticleField

    implicit none

contains

    subroutine DiagInitilalization(CF)
        Class(ControlFlow), intent(in) :: CF

        ! Grid全局参数设置，NtInner为每周期步数，NtOuter为诊断周期数
        call GridSetup(NtInner=CF%Period, NtOuter=CF%NRun, LoopStartIn=CF%Timer/CF%Period)

        ! init particle_field
        call DiagParticleFieldInitilalization(CF)

    end subroutine DiagInitilalization


    subroutine DiagOneStep(PB, FG, FO, FT, Geom)
        Type(ParticleBundle), intent(inout) :: PB(:)
        Type(FieldEM), intent(inout) :: FG
        Type(FieldOne), intent(inout) :: FO(:)
        Type(FieldSource), intent(inout) :: FT
        Type(Geometry), intent(in) :: Geom

        call DiagParticleFieldPeriod(PB, FG, FO, FT, Geom)
    
    end subroutine DiagOneStep


    subroutine DiagOneStepFinal()

        Call DiagParticleFieldPeriodDump()
    
    end subroutine DiagOneStepFinal

    subroutine DiagReleaseAll()

        call DiagParticleFieldPeriodRelease()

    end subroutine DiagReleaseAll

end Module ModuleDiagStep