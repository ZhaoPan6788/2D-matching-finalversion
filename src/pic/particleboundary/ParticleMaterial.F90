module ModuleParticleMaterial
    use ModuleParticleBundle
    use ModuleGeometry
    use ModuleMaterials

    implicit none

contains

    subroutine MaterialAborptionParticle(CF, Geom, MT, PB)
        type(ControlFlow), intent(inout) :: CF
        type(Geometry), intent(inout) :: Geom
        type(Materials), intent(inout) :: MT
        type(ParticleBundle), intent(inout) :: PB
        integer(4) :: i, j
        integer(4) :: NzL, NzU, NrL, NrU
        real(8)    :: ZL, ZU, RL, RU
        integer(4) :: cell_material_index

        do i = PB%NPar, 1, -1
            ! 删除超出边界的粒子
            if (PB%PO(i)%Z <= 0.d0 .or. &
                PB%PO(i)%Z >= dble(CF%DM%GlobalShape(1)-1) .or. &
                PB%PO(i)%R >= dble(CF%DM%GlobalShape(2)-1)) then
                call PB%DelOne(i)

            else if (PB%PO(i)%R <= 0.d0) then
                PB%PO(i)%R = abs(PB%PO(i)%R)
                PB%PO(i)%Vr = abs(PB%PO(i)%Vr)
                PB%PO(i)%Ar = abs(PB%PO(i)%Ar)

            else
                NzL = Ceiling(PB%PO(i)%Z)
                NrL = Ceiling(PB%PO(i)%R)
                cell_material_index = Geom%cell_material(NzL, NrL)

                if (MT%material_type(cell_material_index) == material_type_metal) then
                    ! 粒子被金属电极吸收
                    MT%metls(cell_material_index)%charge_one_step = MT%metls(cell_material_index)%charge_one_step + &
                                                           PB%PO(i)%WQ * PB%Charge
                    call PB%DelOne(i)

                else if (MT%material_type(cell_material_index) == material_type_dielectric) then
                    ! 粒子打到介质表面
                    if (Geom%hedge_type(NzL, NrL) == edge_type_vacuum_to_dielectric .or. &
                        Geom%hedge_type(NzL, NrL) == edge_type_dielectric_to_vacuum) then

                        RL  = Dble(NrL) - PB%PO(i)%R
                        RU  = 1.d0 - RL

                        MT%sigma_one_step(NzL, NrL)   = MT%sigma_one_step(NzL, NrL)   + RL * PB%PO(i)%WQ * PB%Charge
                        MT%sigma_one_step(NzL, NrL+1) = MT%sigma_one_step(NzL, NrL+1) + RU * PB%PO(i)%WQ * PB%Charge

                    else if (Geom%hedge_type(NzL+1, NrL) == edge_type_vacuum_to_dielectric .or. &
                        Geom%hedge_type(NzL+1, NrL) == edge_type_dielectric_to_vacuum) then

                        RL  = Dble(NrL) - PB%PO(i)%R
                        RU  = 1.d0 - RL

                        MT%sigma_one_step(NzL+1, NrL)   = MT%sigma_one_step(NzL+1, NrL)   + RL * PB%PO(i)%WQ * PB%Charge
                        MT%sigma_one_step(NzL+1, NrL+1) = MT%sigma_one_step(NzL+1, NrL+1) + RU * PB%PO(i)%WQ * PB%Charge

                    else if (Geom%vedge_type(NzL, NrL) == edge_type_vacuum_to_dielectric .or. &
                        Geom%vedge_type(NzL, NrL) == edge_type_dielectric_to_vacuum) then
                    
                        ZL  = Dble(NzL) - PB%PO(i)%Z
                        ZU  = 1.d0 - ZL

                        MT%sigma_one_step(NzL, NrL)   = MT%sigma_one_step(NzL, NrL)   + ZL * PB%PO(i)%WQ * PB%Charge
                        MT%sigma_one_step(NzL+1, NrL) = MT%sigma_one_step(NzL+1, NrL) + ZU * PB%PO(i)%WQ * PB%Charge

                    else if (Geom%vedge_type(NzL, NrL+1) == edge_type_vacuum_to_dielectric .or. &
                        Geom%vedge_type(NzL, NrL+1) == edge_type_dielectric_to_vacuum) then

                        ZL  = Dble(NzL) - PB%PO(i)%Z
                        ZU  = 1.d0 - ZL

                        MT%sigma_one_step(NzL, NrL+1)   = MT%sigma_one_step(NzL, NrL+1)   + ZL * PB%PO(i)%WQ * PB%Charge
                        MT%sigma_one_step(NzL+1, NrL+1) = MT%sigma_one_step(NzL+1, NrL+1) + ZU * PB%PO(i)%WQ * PB%Charge

                    end if

                    call PB%DelOne(i)

                end if

            end if
        end do

    end subroutine MaterialAborptionParticle

end module ModuleParticleMaterial
