# !/bin/bash
cd build
make clean
cd ..
rm -rf build
mkdir build
cd build
cmake ..
make
./cubicStylization_bin

# cd build
# make clean
# make 
# ./cubicStylization_bin

# rm -rf build
# mkdir build
# cd build
# cmake ..
# make -j
# ./cubicStylization_bin