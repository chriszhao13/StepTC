MACRO=-DDEBUG_DF -g 
CXX=g++
CXXFLAGS=-O3 -std=c++14 -fopenmp $(MACRO) -lpthread
NVCC=nvcc
NVCCFLAGS=--gpu-architecture=compute_70 --gpu-code=sm_70  -lcudart  -Xcompiler -O3 -std=c++14 $(MACRO) -lineinfo

all : tricount

tricount : main_gpu.o tricount_cu.o gpu.o
	$(CXX) $(CXXFLAGS) -fstack-protector-all -o tricount main_gpu.o tricount_cu.o gpu.o -L /usr/local/cuda/lib64 -lcudart

tricount_cu.o : tricount.cu
	$(NVCC) $(NVCCFLAGS) -dc tricount.cu -o tricount_cu.o

gpu.o : tricount_cu.o
	$(NVCC) $(NVCCFLAGS) -dlink tricount_cu.o -o gpu.o

main_gpu.o : main_gpu.cpp mapfile.hpp
	$(CXX) $(CXXFLAGS) -fstack-protector-all -c main_gpu.cpp

.PHONY : clean
clean :
	rm tricount *.o

