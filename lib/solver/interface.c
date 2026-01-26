#include <stdio.h>
#include "interface.hpp"

void init_all_solvers_() {
    initAllSolvers();
}

void init_poisson_solver_(int *Nz, int *Nr, double *dz, double *dr, int *zsize, int *divz, int *rsize, int *divr) {
    initPoissonSolver(*Nz, *Nr, *dz, *dr, *zsize, divz, *rsize, divr);
}

void poisson_solve_(double *coeA, double *coeB,
            double *coeC, double *coeD,
            double *coeE, double *source,
            double *solve) {
    poissonSolverOnce(coeA, coeB, coeC, coeD, coeE,source, solve);
}

void destroy_all_solvers_() {
    destroyAllSolvers();
}
