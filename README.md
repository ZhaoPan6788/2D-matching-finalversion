# iPM2D

本页为快速开始，详细教程见[iPM2D github wiki](https://github.com/plasmasimulation/iPM2D/wiki)

## 环境配置

在程序运行前，请依照下面顺序，依次检查相关依赖项，确保环境的完整性。

配置教程见[Linux相关工具的安装和使用](https://github.com/plasmasimulation/Development-Guide/blob/main/Linux/Linux-Guide.md)

- 系统：Linux (Ubuntu)
- 语言：Fortran | C | C++ | Python | CMake
- 编译器: oneAPI
- 系统依赖：`gcc gfortran g++ gdb python3 python3-pip wget curl git make cmake nano`
- 构建工具：CMake
- IDE：VS Code
- 项目依赖：`HDF5 ` | `PETSc`
- 可选依赖：`Sundials`

## 重要库安装

### **必读**

安装第三方库都需要配置环境变量，见[环境变量添加方法](https://github.com/plasmasimulation/Development-Guide/blob/main/Packages/packages.md#%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E6%B7%BB%E5%8A%A0%E6%96%B9%E6%B3%95)

### HDF5

需要预装oneapi，并添加至环境变量。

#### 下载

前往 [HDF5](https://portal.hdfgroup.org/display/support/Downloads) 官网下载最新版本的源码（.tar.gz格式），下面以1.12.2版本为例：

```shell
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.2/src/hdf5-1.12.2.tar.gz
```

#### 解压：
```shell
tar zxvf hdf5-1.12.2.tar.gz
cd hdf5-1.12.2
```

#### 设置编译器
```shell
export FC=mpiifort
export F9X=mpiifort
export CC=mpiicc
```

#### 配置选项

配置路径为本地 `/home/yourname/hdf5`（`yourname`是自己的Linux用户名）：

```Shell
./configure --prefix=/home/修改此处为自己的用户名/hdf5 --enable-fortran --enable-shared --enable-parallel --with-pic CC=mpiicc FC=mpiifort CXX=mpiicpc CFLAGS="-fPIC -O3 -xHost -ip -align" FFLAGS="-fPIC -O3 -xHost -ip -align" CXXFLAGS="-fPIC -O3 -xHost -ip -align"
```

### 安装
```shell
make -j8
```
需要等待较长的时间

```shell
make check install -j8
```
同样需要等待较长的时间，然后复制到 `/usr/local/hdf5`：

```shell
sudo cp -r /home/修改此处为自己的用户名/hdf5 /usr/local/hdf5
```

安装好后输入`ll /usr/local/hdf5`应当显示有以下四个文件夹：
```
hdf5
├── bin
├── include
├── lib
└── share
```

#### 添加环境变量

在```~/.bashrc```配置文件末尾添加 (可以选择不复制到 `/usr/local/hdf5`，下面环境变量设置为hdf5所在目录即可)：
```shell
# HDF5 ENV
export HDF5_ENV_PATH=/usr/local/hdf5
```

### PETSc

需要预装 `oneapi`，并添加路径至环境变量

#### 下载

下载 Release 分支 (建议安装 v3.18.4 版本，太新的版本可能会出现一些错误)
- Gitlab 仓库: https://gitlab.com/petsc/petsc
- 安装指南：https://petsc.org/release/install/

```shell
wget https://gitlab.com/petsc/petsc/-/archive/v3.18.4/petsc-v3.18.4.tar.gz
tar zxvf petsc-v3.18.4.tar.gz
```

#### 配置

`/usr/local/petsc`是默认的安装目录。
```shell
cd petsc-v3.18.4
./configure --prefix=/usr/local/petsc --with-cc=mpiicc --with-cxx=mpiicpc --with-fc=mpiifort --with-mpiexec=mpiexec.hydra --with-debugging=no COPTFLAGS=-g -O FOPTFLAGS=-g -O CXXOPTFLAGS=-g -O --with-precision=double --with-blaslapack-dir=$MKLROOT --with-mkl_pardiso-dir=$MKLROOT --with-mkl_cpardiso-dir=$MKLROOT
```
可能等待一段时间。

#### 编译

```shell
make all check -j8
sudo make install
```

#### 添加环境变量

在`~/.bashrc`配置文件末尾添加：
```shell
# PETSc ENV
export PETSC_ENV_PATH=/usr/local/petsc
```

### Sundials

#### 下载

```shell
wget https://github.com/LLNL/sundials/releases/download/v6.6.1/sundials-6.6.1.tar.gz
```

#### 编译&安装
```shell
tar zxvf sundials-6.6.1.tar.gz
cd sundials-6.6.1
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/sundials -DEXAMPLES_INSTALL_PATH=/usr/local/sundials/examples -DBUILD_FORTRAN_MODULE_INTERFACE=on -DCMAKE_Fortran_COMPILER=mpiifort -DCMAKE_C_COMPILER=mpiicc ../
make -j8
sudo make install
```

添加环境变量至 `~/.bashrc`
```shell
# SUNDIALS ENV
export SUNDIALS_ENV_PATH=/usr/local/sundials
```


### FFTW

#### 下载

源代码下载，见 [**FFTW**](http://fftw.org/download.html)

```shell
wget http://fftw.org/fftw-3.3.10.tar.gz
```

#### 解压
```shell
tar -zxvf fftw-3.3.10.tar.gz
cd fftw-3.3.10
```

#### 配置

`/usr/local/fftw` 是自定义目录

```shell
./configure --prefix=/usr/local/fftw --enable-shared --enable-mpi
```

#### 编译

```shell
make -j4
```

#### 安装

```shell
sudo make install
```

#### 环境变量设置

添加环境变量至 `~/.bashrc`
```shell
# FFTW ENV
export FFTW_ENV_PATH=/usr/local/fftw
```
