rm -rf build
mkdir build
cd build
cmake .. -DGinkgo_DIR=/home/inseo764/TC_BELL/docker/ICTC/install/ginkgo/lib/cmake/Ginkgo
make -j