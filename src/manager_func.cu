//
// Created by Jie Ren (jieren9806@gmail.com) on 2021/6/11.
//

#include "manager.h"
#include "storage.h"

namespace FastSample {
            template<typename T>
            __global__ void _normal(T* output, curandStateXORWOW_t* status, unsigned int global_loc) {
            // We use the normal_unroll function as the optimal sample function.
            size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
            T* start_loc = output + global_loc * nStatus * nUnroll + idx;
            curandStateXORWOW_t status_loc = status[idx];
            float2 tmp;
#pragma unroll
            for (unsigned int i = 0; i < nUnroll; i += 2) {
                tmp = curand_normal2(&status_loc);
                start_loc[nStatus * i] = tmp.x;
                start_loc[nStatus * (i + 1)] = tmp.y;
            }

            status[idx] = status_loc;
        }

        template<typename T>
        void manager<T>::Fill_Normal(T* d_result) {
            dim3 block_Elm(nThreads_base_sample), grid_Elm(nStatus / nThreads_base_sample);
            unsigned int nGenRep = storage_normal.capacity / nStatus;

            for (unsigned int i = 0; i < nGenRep / nUnroll; i++)
                _normal<T> <<< grid_Elm, block_Elm, 0, storage_normal.stream >>>(d_result, rand_status, i);
        }

        template<typename T>
        __global__ void _uniform(T* output, curandStateXORWOW_t* status, unsigned int global_loc) {
        // We use the normal_unroll function as the optimal sample function.
        size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
        T* start_loc = output + global_loc * nStatus * nUnroll + idx;
        curandStateXORWOW_t status_loc = status[idx + nStatus];
#pragma unroll
        for (unsigned int i = 0; i < nUnroll; i++)
            start_loc[nStatus * i] = curand_uniform(&status_loc);

        status[idx + nStatus] = status_loc;
    }

    template<typename T>
    void manager<T>::Fill_Uniform(T* d_result) {
        dim3 block_Elm(nThreads_base_sample), grid_Elm(nStatus / nThreads_base_sample);
        unsigned int nGenRep = storage_uniform.capacity / nStatus;

        for (unsigned int i = 0; i < nGenRep / nUnroll; i++)
            _uniform<T> <<< grid_Elm, block_Elm, 0, storage_uniform.stream >>>(d_result, rand_status, i);
    }

    namespace {
        template<typename T>
        void INST() {
            manager<T> m(0);
            m.Fill_Normal(nullptr);
            m.Fill_Uniform(nullptr);
        }

        __attribute__((unused)) void RUN() {
            INST<double>();
            INST<float>();
        }
    }
}