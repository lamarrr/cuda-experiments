set -xe

nvcc tma.cu -std=c++20 -O3 -arch=native
./a.out