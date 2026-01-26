Module ModuleBspline
    Implicit none
    Type :: Bspline1N2D
!!
        Integer(4) :: Nx(2),Ny(2)
        Real(8) :: X(2),Y(2)
        Real(8) :: Area(4)
    Contains
        procedure :: Update=>UpdateBspline1N2D
        procedure :: UpdateNxy=>UpdateBspline1N2DNxy
        procedure :: UpdateArea=>UpdateBspline1N2DArea
        procedure :: SetArea=>SetBspline1N2DArea
        procedure :: Mold0D=>Mold0DBspline1N2D

        generic :: Mold=>Mold0D
    ENdType

    Type :: Bspline0N2D
        Integer(4) :: Nx,Ny
    Contains
        procedure :: Update=>UpdateBspline0N2D
    ENdType

Contains
    Subroutine SetBspline1N2DArea(BS)
        Class(Bspline1N2D) :: BS
        BS%Area=0.25d0
        return
    Endsubroutine SetBspline1N2DArea

    Subroutine Mold0DBspline1N2D(BS1,BS0)
        Class(Bspline1N2D) :: BS1
        Type(Bspline0N2D) :: BS0
        BS1%Nx(1)=BS0%Nx
        BS1%Ny(1)=BS0%Ny
        BS1%Nx(2)=BS1%Nx(1)+1
        BS1%Ny(2)=BS1%Ny(1)+1
        return
    Endsubroutine Mold0DBspline1N2D

    Subroutine UpdateBspline1N2D(BS,X,Y)
        Class(Bspline1N2D) :: BS
        Real(8) :: X,Y
        Associate(Nx1=>BS%Nx(1),Nx2=>BS%Nx(2),Ny1=>BS%Ny(1),Ny2=>BS%Ny(2), &
                  X1=>BS%X(1),X2=>BS%X(2),Y1=>BS%Y(1),Y2=>BS%Y(2), &
                  A11=>BS%Area(1),A21=>BS%Area(2),A12=>BS%Area(3),A22=>BS%Area(4))
            Nx1=Ceiling(X)
            Nx2=Nx1+1
            X1=Dble(Nx1)-Y
            X2=1.d0-X1
            Ny1=Ceiling(Y)
            Ny2=Ny1+1
            Y1=Dble(Ny1)-Y
            Y2=1.d0-Y1
            A11=X1*Y1
            A21=X2*Y1
            A12=X1*Y2
            A22=X2*Y2
        EndAssociate
        return
    Endsubroutine UpdateBspline1N2D

    Subroutine UpdateBspline1N2DNxy(BS,X,Y)
        Class(Bspline1N2D) :: BS
        Real(8) :: X,Y
        Associate(Nx1=>BS%Nx(1),Nx2=>BS%Nx(2),Ny1=>BS%Ny(1),Ny2=>BS%Ny(2), &
                  X1=>BS%X(1),X2=>BS%X(2),Y1=>BS%Y(1),Y2=>BS%Y(2), &
                  A11=>BS%Area(1),A21=>BS%Area(2),A12=>BS%Area(3),A22=>BS%Area(4))
            Nx1=Ceiling(X)
            Nx2=Nx1+1
            Ny1=Ceiling(Y)
            Ny2=Ny1+1
        EndAssociate
        return
    Endsubroutine UpdateBspline1N2DNxy

    Subroutine UpdateBspline1N2DArea(BS,X,Y)
        Class(Bspline1N2D) :: BS
        Real(8) :: X,Y
        Associate(Nx1=>BS%Nx(1),Nx2=>BS%Nx(2),Ny1=>BS%Ny(1),Ny2=>BS%Ny(2), &
                  X1=>BS%X(1),X2=>BS%X(2),Y1=>BS%Y(1),Y2=>BS%Y(2), &
                  A11=>BS%Area(1),A21=>BS%Area(2),A12=>BS%Area(3),A22=>BS%Area(4))
            X1=Dble(Nx1)-Y
            X2=1.d0-X1
            Y1=Dble(Ny1)-Y
            Y2=1.d0-Y1
            A11=X1*Y1
            A21=X2*Y1
            A12=X1*Y2
            A22=X2*Y2
        EndAssociate
        return
    Endsubroutine UpdateBspline1N2DArea

    Subroutine UpdateBspline0N2D(BS,X,Y)
        Class(Bspline0N2D) :: BS
        Real(8) :: X,Y
        Associate(Nx=>BS%Nx,Ny=>BS%Ny)
            Nx=Ceiling(X)
            !Nx2=Nx1+1
            Ny=Ceiling(Y)
            !Ny2=Ny1+1
        EndAssociate
        return
    Endsubroutine UpdateBspline0N2D

    !Subroutine SearchingBspline2D(BSbefore,BSafter,PropertiesCell,CellCrossed)
    !       Class(Bspline0N2D),Intent(in) :: BSbefore,BSafter
    !       Real(8),Intent(in) :: PropertiesCell(:,:)
    !       Integer(4) :: i,Direction,
    !       Associate(Nx1=>BSbefore%Nx,Ny1=>BSbefore%Ny(1),BSafter=>BS%Nx,Ny2=>BSafter%Ny(2))
    !
    !            DO i=Nx1,Nx2
    !
    !
    !
    !            Nx2=Nx1+1
    !            Ny1=Ceiling(Y)
    !            Ny2=Ny1+1
    !        End Associate
    !       return
    !End subroutine UpdateBspline0N2D

EndModule ModuleBspline
