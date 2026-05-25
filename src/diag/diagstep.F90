Module ModuleDiagStep
    use ModuleControlFlow
    use DiagnosticsParticleField
    use DiagnosticsEDF

    implicit none

contains

    subroutine DiagInitilalization(CF)
        Class(ControlFlow), intent(in) :: CF

        call GridSetup(NtInner=CF%Period, NtOuter=CF%NRun, LoopStartIn=CF%Timer/CF%Period)

        call DiagParticleFieldInitilalization(CF)

        call DiagEDFInitilalization(CF)

    end subroutine DiagInitilalization


    subroutine DiagOneStep(CF, PB, FG, FO, FT, Geom)
        class(ControlFlow), intent(in) :: CF
        Type(ParticleBundle), intent(inout) :: PB(:)
        Type(FieldEM), intent(inout) :: FG
        Type(FieldOne), intent(inout) :: FO(:)
        Type(FieldSource), intent(inout) :: FT
        Type(Geometry), intent(in) :: Geom

        call DiagParticleFieldPeriod(PB, FG, FO, FT, Geom)
    
        call DiagEDFPeriod(CF, PB, PB)
    
    end subroutine DiagOneStep


    subroutine DiagOneStepFinal()

        Call DiagParticleFieldPeriodDump()
    
    end subroutine DiagOneStepFinal

    subroutine DiagReleaseAll()

        call DiagParticleFieldPeriodRelease()

        call DiagEDFRelease()

    end subroutine DiagReleaseAll

end Module ModuleDiagStep