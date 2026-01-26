/*-----------------------------------------------------------------------------
petsc soler lib's interface
@time       :   2021/7/25
@author     :   zili.chen
@version    :   v1.0
@log        :   v1.0 first build
                v2.0 更简洁的封装
                v3.0 2d poisson solver
                v4.0 use new-solver, see https://github.com/picmc/petscSolve3d/tree/solver-new

-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>

#include "vectorBase.hpp"
#include "solveBase.hpp"
#include "interface.hpp"
#include "json_read.hpp"

using namespace std;

// User defined solver for 2D possion equation
class Solver2D : public ScalarSolver {
public:
    vector<int> localSizeX;          // 存放domain的大小, 实际求解区域需要去除重复的边界
    vector<int> localSizeY;
    vector<int> localSizeZ;

    int Nz = 0;
    int Nr = 0;
    int cod = 0;
    int swidth = 1;

    double dz = 0.0;
    double dr = 0.0;

    double *coeA = nullptr;         // (i-1, j)
    double *coeB = nullptr;         // (i, j)
    double *coeC = nullptr;         // (i+1, j)
    double *coeD = nullptr;         // (i, j-1)
    double *coeE = nullptr;         // (i, j+1)
    double *source = nullptr;

    virtual void init() override;
    virtual void setA() override;
    virtual void setb() override;
};

// Solver Manager
class Manage {
public:
    // add your solver pointer
    Solver2D *poissonPtr = nullptr;

    // instantiation solver
    void init() {
        this->destroy();
        this->poissonPtr = new Solver2D;
    }

    // destroy solver
    void destroy() {
        if (nullptr != this->poissonPtr) {
            delete this->poissonPtr;
            this->poissonPtr = nullptr;
        }
    }
};

// Global management objects
static Manage solverMng;

/*-----------------------------------------------------------------------------
Function initMPI
Initialize MPI and solver Manage

-----------------------------------------------------------------------------*/
void initAllSolvers() {
    initPetsc();
    solverMng.init();
}

/*-----------------------------------------------------------------------------
Function initMPI
destroy solvers

-----------------------------------------------------------------------------*/
void destroyAllSolvers() {
    solverMng.destroy();
    PetscFinalize();
}

/*-----------------------------------------------------------------------------
Function initPoissonSolver
Initialize Poisson solver
    div         Local size of solution domain in each image
                Note that the boundaries between domains overlap

-----------------------------------------------------------------------------*/
void initPoissonSolver(int Nz, int Nr, double dz, double dr, int zsize, int divz[], int rsize, int divr[]) {
    solverMng.poissonPtr->localSizeX.clear();
    solverMng.poissonPtr->localSizeY.clear();
    solverMng.poissonPtr->localSizeZ.clear();

    if (zsize > 0 && divz != NULL) {
        for (auto i = 0; i < zsize; i++) {
            solverMng.poissonPtr->localSizeZ.push_back(divz[i]);
        }
    }

    if (rsize > 0 && divr != NULL) {
        for (auto i = 0; i < rsize; i++) {
            solverMng.poissonPtr->localSizeX.push_back(divr[i]);
        }
    }

    solverMng.poissonPtr->Nz = Nz;
    solverMng.poissonPtr->Nr = Nr;
    solverMng.poissonPtr->dz = dz;
    solverMng.poissonPtr->dr = dr;
    solverMng.poissonPtr->cod = 1;
    solverMng.poissonPtr->swidth = 1;

    solverMng.poissonPtr->init();

    // if (!solverMng.poissonPtr->dm.rank) cout << "2D rz Poisson Solver init complete." << endl;
}

/*-----------------------------------------------------------------------------
Function poissonSolverOnce
Set coefficient matrix A and source term b, for Ax=b
    coe*        [Ly, Lx], A is [ix-1, iy], B is [ix, iy], C is [ix+1, iy], D is [ix, iy-1], B is [ix, iy+1]
                Lx is the local dimension in the r and x directions
                Ly is the local dimension in the z and y directions
    source      [Ly, Lx], Source term of Poisson equation
    solve       [Ly, Lx], Returned solved potential

Note: arrays in FORTRAN are stored in columns first

-----------------------------------------------------------------------------*/
void poissonSolverOnce(double coeA[], double coeB[], double coeC[], double coeD[], double coeE[], double source[], double solve[]) {
    solverMng.poissonPtr->coeA = coeA;
    solverMng.poissonPtr->coeB = coeB;
    solverMng.poissonPtr->coeC = coeC;
    solverMng.poissonPtr->coeD = coeD;
    solverMng.poissonPtr->coeE = coeE;
    solverMng.poissonPtr->source = source;

    solverMng.poissonPtr->setA();
    solverMng.poissonPtr->setb();
    solverMng.poissonPtr->run();

    // return solve value
    PetscInt            i, j, k;
    PetscScalar         ***xPtr;
    DataManage          &dm = solverMng.poissonPtr->dm;

    PetscInt            Mx = solverMng.poissonPtr->dm.Mx;
    PetscInt            My = solverMng.poissonPtr->dm.My;
    PetscInt            Mz = solverMng.poissonPtr->dm.Mz;

    PetscInt            Lx = dm.xend;
    PetscInt            Lz = dm.zend;

    if (dm.xstart+dm.xend != dm.Mx) Lx++;
    if (dm.zstart+dm.zend != dm.Mz) Lz++;

    DMDAVecGetArray(dm.da, solverMng.poissonPtr->solve.x, &xPtr);

    for (i = dm.xstart; i < dm.xstart + dm.xend; i++) {
        for (j = dm.ystart; j < dm.ystart + dm.yend; j++) {
            for (k = dm.zstart; k < dm.zstart + dm.zend; k++) {
                solve[(i - dm.xstart) * Lz + (k - dm.zstart)] = xPtr[i][j][k];
            }
        }
    }

    DMDAVecRestoreArray(dm.da, solverMng.poissonPtr->solve.x, &xPtr);
}

/*-----------------------------------------------------------------------------
Function Solver2D::init
Initialize Poisson solver

Note: When solving Poisson equation, the overlapping regions between domains are ignored

For example:
    The number of image is 4. The size of the domain of each image is [5, 5, 5, 5]
    However, because the domain boundaries are overlapping, the domain actually solved is:
            image 0: [0, 5)
            image 1: [5, 9)
            image 2: [9, 12)
            image 3: [12, 17]

-----------------------------------------------------------------------------*/
void Solver2D::init() {

    this->dm.Mx = this->Nr;
    this->dm.My = 1;
    this->dm.Mz = this->Nz;
    this->dm.cod = this->cod;
    this->dm.sharedWidth = this->swidth;

    this->dm.x0 = 0.0;
    this->dm.y0 = 0.0;
    this->dm.z0 = 0.0;

    if (this->dm.Mx > 1)    this->dm.dx = this->dr;
    else                    this->dm.dx = 1.0;

    if (this->dm.My > 1)    this->dm.dy = 1.0;
    else                    this->dm.dy = 1.0;

    if (this->dm.Mz > 1)    this->dm.dz = this->dz;
    else                    this->dm.dz = 1.0;

    // 局部尺寸设置
    this->dm.npx = 1;
    this->dm.npy = 1;
    this->dm.npz = 1;

    int *localX = NULL, *localY = NULL, *localZ = NULL;
    if (this->localSizeX.size() > 0) {
        int sizex = this->localSizeX.size();
        localX = new int[sizex];
        this->dm.npx = sizex;

        for (auto i = 0; i < sizex; i++) {
            if (i < sizex-1) localX[i] = this->localSizeX[i] - 1;
            else             localX[i] = this->localSizeX[i];
        }
    }

    if (this->localSizeY.size() > 0) {
        int sizey = this->localSizeY.size();
        localY = new int[sizey];
        this->dm.npy = sizey;

        for (auto i = 0; i < sizey; i++) {
            if (i < sizey-1) localY[i] = this->localSizeY[i] - 1;
            else             localY[i] = this->localSizeY[i];
        }
    }

    if (this->localSizeZ.size() > 0) {
        int sizez = this->localSizeZ.size();
        localZ = new int[sizez];
        this->dm.npz = sizez;

        for (auto i = 0; i < sizez; i++) {
            if (i < sizez-1) localZ[i] = this->localSizeZ[i] - 1;
            else             localZ[i] = this->localSizeZ[i];
        }
    }

    this->dm.init(localZ, localY, localX);
    if (nullptr != localX) delete [] localX;
    if (nullptr != localY) delete [] localY;
    if (nullptr != localZ) delete [] localZ;

    // A与b初始化
    this->A.init(this->dm, NormMatrixType, 5); // 这里的字面量 5 是预置的存储尺寸，表示系数矩阵中每行最多有5个非零量
    this->b.init(this->dm);

    this->isInitVar = true;
}

/*-----------------------------------------------------------------------------
Function Solver2D::setA
Set coefficient matrix A, for XY or RZ
    coe*        [Ly, Lx], A is [ix-1, iy], B is [ix, iy], C is [ix+1, iy], D is [ix, iy-1], B is [ix, iy+1]
                Lx is the local dimension in the r and x directions
                Ly is the local dimension in the z and y directions

Note: arrays in FORTRAN are stored in columns first

-----------------------------------------------------------------------------*/
void Solver2D::setA() {
    PetscInt            i, j, k, frow;
    PetscScalar         v[5];
    MatStencil          row, col[5];

    PetscInt            Mx = dm.Mx;
    PetscInt            My = dm.My;
    PetscInt            Mz = dm.Mz;

    PetscInt            Lx = dm.xend;
    PetscInt            Lz = dm.zend;

    if (dm.xstart+dm.xend != dm.Mx) Lx++;
    if (dm.zstart+dm.zend != dm.Mz) Lz++;

    // z, y, x分别是第一、二、三维度, 所以这里i, j, k是反的
    // 对此处, i表示z, k表示r
    for (k = dm.xstart; k < dm.xstart + dm.xend; k++) {
        for (j = dm.ystart; j < dm.ystart + dm.yend; j++) {
            for (i = dm.zstart; i < dm.zstart + dm.zend; i++) {
                row.i = i; row.j = j; row.k = k;
                frow = (k - dm.xstart) * Lz +  (i - dm.zstart);             // Fortran中数据列优先存储

                if (Mx-1 == k || 0 == i || Mz-1 == i) {                     // 边界
                    v[0] = 1.0;
                    MatSetValuesStencil(A.data, 1, &row, 1, &row, v, INSERT_VALUES);
                }
                else if (0 == k) {                                          // 中轴
                    col[0].i = i;       col[0].j = j;   col[0].k = k;       // coeB
                    col[1].i = i;       col[1].j = j;   col[1].k = k+1;     // coeE
                    col[2].i = i-1;     col[2].j = j;   col[2].k = k;       // coeA
                    col[3].i = i+1;     col[3].j = j;   col[3].k = k;       // coeC

                    v[0] = coeB[frow];
                    v[1] = coeE[frow];
                    v[2] = coeA[frow];
                    v[3] = coeC[frow];

                    MatSetValuesStencil(A.data, 1, &row, 4, col, v, INSERT_VALUES);
                }
                else {
                    col[0].i = i;   col[0].j = j;   col[0].k = k-1;     // coeD
                    col[1].i = i;   col[1].j = j;   col[1].k = k;       // coeB
                    col[2].i = i;   col[2].j = j;   col[2].k = k+1;     // coeE
                    col[3].i = i-1; col[3].j = j;   col[3].k = k;       // coeA
                    col[4].i = i+1; col[4].j = j;   col[4].k = k;       // coeC

                    v[0] = coeD[frow];
                    v[1] = coeB[frow];
                    v[2] = coeE[frow];
                    v[3] = coeA[frow];
                    v[4] = coeC[frow];

                    MatSetValuesStencil(A.data, 1, &row, 5, col, v, INSERT_VALUES);
                }
            }
        }
    }

    MatAssemblyBegin(A.data, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.data, MAT_FINAL_ASSEMBLY);

    this->solve._setA(this->A.data);
}

/*-----------------------------------------------------------------------------
Function Solver2D::setb
Set source term b, for XY
    source      [Ly, Lx], Source term of Poisson equation

Note: arrays in FORTRAN are stored in columns first

-----------------------------------------------------------------------------*/
void Solver2D::setb() {
    PetscInt            i, j, k;
    PetscScalar         ***xPtr;

    PetscInt            Lx = dm.xend;
    PetscInt            Lz = dm.zend;

    if (dm.xstart+dm.xend != dm.Mx) Lx++;
    if (dm.zstart+dm.zend != dm.Mz) Lz++;

    DMDAVecGetArray(b.da, b.data, &xPtr);

    for (i = dm.xstart; i < dm.xstart + dm.xend; i++) {
        for (j = dm.ystart; j < dm.ystart + dm.yend; j++) {
            for (k = dm.zstart; k < dm.zstart + dm.zend; k++) {
                xPtr[i][j][k] = source[(i - dm.xstart) * Lz + (k - dm.zstart)];
                // xPtr[i][j][k] = source[(i - dm.xstart) * Mz + (k - dm.zstart)];
            }
        }
    }

    DMDAVecRestoreArray(b.da, b.data, &xPtr);
}
