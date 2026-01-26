program fortran_mpi
    use mpi
    use ModuleFileName

    implicit none

    ! Variables
    type(FileName) :: name
    integer(4) :: size, rank, ierr, i

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

    ! 从已定义的全局常量初始化
    call InitFileName()                                             ! 初始化全局常量 (后续可以在 Json 中设置)
    call name%Init('MyName', RESTART_FILE_NAME)
    if (0 == rank) write(*, *) name%FullName%str

    ! 从外部参数初始化
    call name%Init('MyName', PathMode=FILE_PATH_MODE_RESTART, &     ! 文件夹, 参数可选
                        ExtensionMode=FILE_EXTENSION_MODE_H5, &     ! 拓展名, 参数可选
                        ParallelMode=FILE_PARALLEL_MODE_CLOSE, &    ! 是否添加 image序号, 参数可选
                        DynamicIndex=FILE_DYNAMIC_MODE_CLOSE)       ! 是否指定动态序号, 参数可选
    if (0 == rank) write(*, *) name%FullName%str

    ! 更改 FileName 的目录, FullName将随之更新
    call name%SetPath(FILE_PATH_MODE_CHECK)
    if (0 == rank) write(*, *) name%FullName%str

    ! 更改 FileName 的拓展名, FullName将随之更新
    call name%SetExte(FILE_EXTENSION_MODE_DAT)
    if (0 == rank) write(*, *) name%FullName%str

    ! 更改 FileName 是否添加 image 序号, FullName将随之更新
    call name%SetParl(FILE_PARALLEL_MODE_OPEN)

    do i = 0, size-1
        if (rank == i) then
            write(*, '(i4, a)') rank, name%FullName%str
            call execute_command_line(" ")
        end if
        call MPI_BARRIER(MPI_COMM_WORLD, ierr)
    end do

    ! 更改 FileName 是否指定动态序号, FullName将随之更新
    call name%SetIndex(FILE_DYNAMIC_MODE_OPEN)
    do i = 0, size-1
        if (rank == i) then
            write(*, '(i4, a)') rank, name%FullName%str
            call execute_command_line(" ")
        end if
        call MPI_BARRIER(MPI_COMM_WORLD, ierr)
    end do

    call name%UpdateIndex()
    do i = 0, size-1
        if (rank == i) then
            write(*, '(i4, a)') rank, name%FullName%str
            call execute_command_line(" ")
        end if
        call MPI_BARRIER(MPI_COMM_WORLD, ierr)
    end do

    call name%UpdateIndex()
    do i = 0, size-1
        if (rank == i) then
            write(*, '(i4, a)') rank, name%FullName%str
            call execute_command_line(" ")
        end if
        call MPI_BARRIER(MPI_COMM_WORLD, ierr)
    end do

    call MPI_FINALIZE(ierr)

end program fortran_mpi
