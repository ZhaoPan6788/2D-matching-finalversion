# Grid模块

## 0. 使用前注意事项
- 本模块适用于：1D/2D的稳态/演化模拟的诊断数据存档
- 本模块仅可用于并行程序，因为用到了并行HDF5模块，串行情况下编译通不过。
- 尚未完成的部分：
    - 3D空间数据的存档
    - 0D数据的存档
    - 文件的append
    - 所有物理量统一存档到一个h5

## 1. HDF5安装
需要预装oneapi，并添加至环境变量

- 下载

    前往官网下载最新版本的源码（.tar.gz格式）：
    <https://portal.hdfgroup.org/display/support/Downloads>
- 解压：
    ```shell
    tar zxvf hdf5-1.12.2.tar.gz
    cd hdf5-1.12.2
    ```
- 设置编译器
    ```shell
    export FC=mpiifort
    export F9X=mpiifort
    export CC=mpiicc
    ```
- 配置选项
    ```shell
    ./configure --prefix=安装路径 --enable-fortran --enable-shared --enable-parallel --with-pic CC=mpiicc FC=mpiifort CXX=mpiicpc CFLAGS="-fPIC -O3 -xHost -ip -align" FFLAGS="-fPIC -O3 -xHost -ip -align" CXXFLAGS="-fPIC -O3 -xHost -ip -align"
    ```
    “安装路径”需要使用绝对路径，示例：
    ```/home/spock/software_install/hdf5```

- 安装
    ```shell
    make -j8
    ```
    需要等待较长的时间
    ```shell
    make check install -j8
    ```
    同样需要等待较长的时间

    安装好后安装目录里应当有以下四个文件夹：
    ```
    hdf5
    ├── bin
    ├── include
    ├── lib
    └── share
    ```
- 移动到系统路径（可选）

    ```shell
    sudo cp -a 初始安装路径 /usr/local/hdf5
    ```
- 添加环境变量

    在```~/.bashrc```配置文件末尾添加 (可以选择不复制到 `/usr/local/hdf5`，下面环境变量设置为hdf5所在目录即可)：
    ```shell
    # HDF5 ENV
    export HDF5_ENV_PATH=/usr/local/hdf5
    ```

    重新加载配置文件：
    ```shell
    source .bashrc --force
    ```
## 2. 快速入门
下面以1D粒子密度、温度等诊断模块DiagMomentem来演示诊断模块的写法。

诊断模块本体（DiagMomentem.F90）：
```F90
Module DiagnosticsMomentum
    ! 引入Grid模块
    use ModuleGridControlFlow
    Implicit none
    ! ---------------
    ! 其他诊断所需语句
    ! ---------------
    
    ! Grid类实例化
    type(GridControlFlow),save :: G1DDiagParticleField
    type(GridControlFlow),save :: G2DDiagParticleField
contains

    ! 初始化过程
    subroutine DiagMomentumInitilalization()
        ! 对Grid进行初始化，Nx1和Nt建议在更前置的模块使用Parameter引入以使其在所有诊断中相同
        call G1DDiagParticleField%Init('DiagField1D',ExtensionMode=FILE_EXTENSION_MODE_H5, &
                            ParallelMode=FILE_PARALLEL_MODE_CLOSE, &
                            DynamicIndex=FILE_DYNAMIC_MODE_CLOSE, &
                            Nx1=65, Ns=13)
        call G2DDiagParticleField%Init('DiagField2D',ExtensionMode=FILE_EXTENSION_MODE_H5, &
                            ParallelMode=FILE_PARALLEL_MODE_CLOSE, &
                            DynamicIndex=FILE_DYNAMIC_MODE_CLOSE, &
                            Nx1=65, Nt=100, Ns=13)
    end subroutine DiagMomentumInitilalization

    ! 诊断过程
    Subroutine DiagParticleFieldPeriod(计算诊断量所需形参)
        Implicit none
        ! 计算语句
        Call WeightingParticleMomentum()
        ! 与粒子种类有关的物理量，调取了PB(i)%SO%Name来动态命名
        Do i = 0, NSpecy
            Call G1DDiagParticleField%Update(TempPMOGlobal(i)%RhoOne, trim(PB(i)%SO%Name)//"Density")
            Call G2DDiagParticleField%Update(TempPMOGlobal(i)%RhoOne, trim(PB(i)%SO%Name)//"Density")
            Call G1DDiagParticleField%Update(TempPMOGlobal(i)%JxOne, trim(PB(i)%SO%Name)//"CurrentDensity")
            Call G2DDiagParticleField%Update(TempPMOGlobal(i)%JxOne, trim(PB(i)%SO%Name)//"CurrentDensity")
        End do
        ! 与粒子种类无关的物理量
        Call G1DDiagParticleField%Update(FG%Phi, "Potential")
        Call G2DDiagParticleField%Update(FG%Phi, "Potential")
        Call G1DDiagParticleField%Update(FG%Rho, "Rho")
        Call G2DDiagParticleField%Update(FG%Rho, "Rho")
        Return
    End Subroutine DiagParticleFieldPeriod

    ! 存盘过程
    Subroutine DiagParticleFieldPeriodDump()
        implicit none
        ! 必须限制在一个镜像运行
        if(this_image()==1) then
            Call G1DDiagParticleField%Dump()
            Call G2DDiagParticleField%Dump()
        end if
    end Subroutine DiagParticleFieldPeriodDump

    ! 诊断量计算过程
    subroutine WeightingParticleMomentum()
        ! ---------------
        ! 计算所需语句
        ! ---------------
    end subroutine WeightingParticleMomentum
End Module DiagnosticsMomentum
```

DiagOnestep.f90：
```F90
Module ModuleDiagOneStep
    use DiagMomentem
    Implicit none
    contains
    
    ! 所有诊断的初始化
    Subroutine DiagInitilalization(CF)
        Implicit none

        ! 用于传递全局模拟参数，也可以采用其他手段
        Class(ControlFlow), intent(in) :: CF

        ! 命名系统初始化，实际使用中一般会放在整个程序开始时
        call InitFileName()

        ! Grid全局参数设置，NtInner为每周期步数，NtOuter为诊断周期数
        call GridSetup(NtInner=CF%Period,NtOuter=CF%NDiagshort)

        ! DiagMomentum的初始化
        call DiagMomentumInitilalization()

        return  
    End Subroutine DiagInitilalization
    
    ! DiagOneStep
    Subroutine  DiagOneStep()
        Implicit none
        Call DiagParticleFieldPeriod(计算诊断量所需实参)
        return  
    End Subroutine DiagOneStep

    ! 存盘过程
    Subroutine DiagOneStepFinal()
    Implicit none  
        Call DiagParticleFieldPeriodDump()
        return  
    End Subroutine DiagOneStepFinal
End Module ModuleDiagOneStep
```
## 支持性
### 数据时空维度
| 数据 | 可用性 |HDF5-覆盖|HDF5-新文件|DAT-覆盖|DAT-新文件
| :----: | :----: | :----: |:----: |:----: |:----: |
| 0X0T  | 施工中    |&#10006; |施工中 |施工中 |施工中 |
| 0X1T  | 施工中    |&#10006; |施工中 |施工中 |施工中 |
| 1X0T  | &#10004; |&#10004; |&#10004; |&#10004; |&#10004; |
| 1X1T  | &#10004; |&#10004; |&#10004; |&#10004; |&#10004; |
| 2X0T  | &#10004; |&#10004; |&#10004; |&#10004; |&#10004; |
| 2X1T  | &#10004; |&#10004; |&#10004; |&#10004; |&#10004; |
| 3X0T  | 施工中|施工中|施工中|施工中|施工中|
| 3X1T  | 施工中|施工中|施工中|施工中|施工中|
### 文件类型
## 可能遇到的特殊情况
- 既需要HDF5存盘，有需要DAT存盘？

## 需要实现的功能
- 
- 