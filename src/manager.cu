//
// Created by Jie Ren (jieren9806@gmail.com) on 2021/6/11.
//

#include "manager.h"
#include "curand.h"
#include "cuda_runtime.h"

namespace FastSample {
    __global__ void Fill_Status_(unsigned long long seed, curandStateXORWOW_t* status) {
        __shared__ curandStateXORWOW_t status_shared[nThreads_Status_y][nThreads_Status_x];
        unsigned int idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
        curand_init(seed, (unsigned long long)idx, 0, &status_shared[threadIdx.y][threadIdx.x]);
        status[idx] = status_shared[threadIdx.y][threadIdx.x];
    }

    __host__ void Fill_Status(unsigned long long seed, curandStateXORWOW_t* status) {
        dim3 block(nThreads_Status_x, nThreads_Status_y);
        dim3 grid(nStatus_all / nThreads_Status);

        Fill_Status_ <<< block, grid >>> (seed, status);
    }

    namespace {
        template<typename T>
        void INST() {
            Fill_Status(0, (curandStateXORWOW_t*)nullptr);
        }

        __attribute__((unused)) void RUN() {
            INST<double>();
            INST<float>();
        }
    }
}