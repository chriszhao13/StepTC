#include <iostream>
#include <fstream>
#include <cstring>
#include <omp.h>
#include <cstdint>
#include <cstdlib> 
#include <immintrin.h>
#include <algorithm>
#include <fstream>
#include <random>
#include <chrono>
#include <pthread.h>

#include "tricount.h"

using namespace std;
// #define DEBUG_DF


#ifdef DEBUG_DF


TimeInterval all;
TimeInterval preProcess;
TimeInterval tmp;


#endif



int main(int argc, char *argv[]) {
    
    // if (argc != 3 || strcmp(argv[1], "-f") != 0) {
    //     cerr << "Usage: -f [data_file_path]" << endl;
    //     exit(1);
    // }
    std::string filepath_edge = std::string(argv[2]) + ".edge.bin";
    std::string filepath_offset = std::string(argv[2]) + ".offset.bin";
    Vertex_type gpu_id  = atoi(argv[3]);
    Vertex_type gpu_num = atoi(argv[4]);

    // sudo apt-get install vmtouch
    std::cout << "graph name = "<< filepath_edge << " cache file using vmtouch " << std::endl;
    std::string command = "vmtouch -vt " + std::string(filepath_edge) + " && vmtouch -vt " + std::string(filepath_offset); // 要执行的Shell命令
    int result = system(command.c_str()); 

    std::ifstream file(filepath_edge, std::ios::binary);
    file.seekg(0, std::ios::end); size_t fileSize = file.tellg(); file.seekg(0, std::ios::beg);

    Offset_type edge_num = fileSize / sizeof(Vertex_type);
    Vertex_type * adjList = (Vertex_type*)aligned_alloc(4096,sizeof(Vertex_type)*edge_num);
    const int numThreads = 1; // omp_get_num_threads();
    size_t blockSize = fileSize / numThreads;
    #pragma omp parallel num_threads(numThreads)
    {
        int threadID   = omp_get_thread_num();
        int numThreads = omp_get_num_threads();
        size_t startOffset = threadID * blockSize;
        size_t actualBlockSize = blockSize;
        if (threadID == numThreads - 1) actualBlockSize = fileSize - startOffset;
        file.seekg(startOffset, std::ios::beg);
        file.read(reinterpret_cast<char*>(&adjList[startOffset / sizeof(Vertex_type)]), actualBlockSize);
    }


    std::ifstream file_offset(filepath_offset, std::ios::binary);
    file_offset.seekg(0, std::ios::end); size_t fileSize_offset = file_offset.tellg(); file_offset.seekg(0, std::ios::beg);

    Vertex_type N = fileSize_offset / sizeof(Offset_type) - 1;
    Offset_type * offSet = (Offset_type*)aligned_alloc(4096,sizeof(Offset_type)*(N+1));
    const int numThreads_offset = 1; // omp_get_num_threads();
    size_t blockSize_offset = fileSize_offset / numThreads_offset;
    #pragma omp parallel num_threads(numThreads_offset)
    {
        int threadID   = omp_get_thread_num();
        int numThreads = omp_get_num_threads();
        size_t startOffset = threadID * blockSize;
        size_t actualBlockSize = blockSize;
        if (threadID == numThreads - 1) actualBlockSize = fileSize - startOffset;
        file_offset.seekg(startOffset, std::ios::beg);
        file_offset.read(reinterpret_cast<char*>(&offSet[startOffset / sizeof(Offset_type)]), actualBlockSize);
    }
    file_offset.close();

    initGPU(edge_num, N, gpu_id);

    unsigned long long sum = tricount(gpu_num,edge_num, N, adjList, offSet);

    cout << "There are " << sum << " triangles in the input graph.\n";

    return 0;
}

