rm -rf build
rm -rf run

export FC=mpiifort
export F9X=mpiifort
export CC=mpiicc
export CXX=mpiicpc

# export FC=ifort
# export F9X=ifort
# export CC=icc
# export CXX=icx

cmake -S . -B build
cmake --build build -j 8

cp script/base/clear.sh run/

# cp config.json build/test/
# cp -r input build/test/
# mkdir build/test/check
# mkdir build/test/diag
# mkdir build/test/restart
# ctest --test-dir build/test
