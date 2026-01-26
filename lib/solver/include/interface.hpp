#ifndef _INTERFACE_HPP_
#define _INTERFACE_HPP_

#ifdef __cplusplus //__cplusplus是cpp中自定义的一个宏
extern "C"
{ //告诉编译器，这部分代码按C语言的格式进行编译，而不是C++的
#endif

    void initAllSolvers();

    void destroyAllSolvers();

    void initPoissonSolver(int Nz, int Nr, double dz, double dr, int zsize, int divz[], int rsize, int divr[]);

    void poissonSolverOnce(double coeA[], double coeB[],
                           double coeC[], double coeD[],
                           double coeE[], double source[],
                           double solve[]);

#ifdef __cplusplus
}
#endif

#endif
