//
// Created by Jie Ren (jieren9806@gmail.com) on 2021/6/11.
//

#include "manager_func.cuh"

namespace FastSample {
    namespace func {
        __global__ void _normal(float* output, curandStateXORWOW_t* status, unsigned int global_loc) {
            // We use the normal_unroll function as the optimal sample function.
            size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
            float* start_loc = output + global_loc * nStatus * nUnroll + idx;
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

        void sample_normal_with_stream(size_t flatted_shape, curandStateXORWOW_t* status, float* d_result, cudaStream_t* stream) {
            dim3 block_Elm(nThreads_base_sample), grid_Elm(nStatus / nThreads_base_sample);
            unsigned int nGenRep = flatted_shape / nStatus;

            for (unsigned int i = 0; i < nGenRep / nUnroll; i++)
                _normal <<<grid_Elm, block_Elm, 0, *stream >>> (d_result, status, i);

        }
    }
}