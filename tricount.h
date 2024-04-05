#ifndef TRICOUNT_H
#define TRICOUNT_H

#include <cstdio>
#include <cstdint>
#include <unistd.h>
#include <iostream>
using namespace std;


#define Offset_type uint32_t
#define Vertex_type uint32_t
#define Edge___type uint32_t


// size must be 8
struct Edge_t {
    Vertex_type u;
    Vertex_type v;
} __attribute__ ((aligned (8)));



void initGPU(Offset_type edge_num, Vertex_type N, int gpu_id);


unsigned long long tricount(Vertex_type gpu_num,Offset_type edge_num, Vertex_type N, Vertex_type * adjList, Offset_type * offSet);


# include <sys/time.h>
class TimeInterval{
public:
    TimeInterval(){
        check();
    }

    void check(){
        gettimeofday(&tp, NULL);
    }

    void print(const char* title){
        struct timeval tp_end, tp_res;
        gettimeofday(&tp_end, NULL);
        timersub(&tp_end, &tp, &tp_res);
        cout << title << ": " <<  ((long long)tp_res.tv_sec) * 1000 + tp_res.tv_usec / 1000 << " ms.\n";
    }
private:
    struct timeval tp;
};


#endif
