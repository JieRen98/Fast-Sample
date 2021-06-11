//
// Created by Jie Ren (jieren9806@gmail.com) on 2021/6/11.
//

#include "manager.h"
#include "storage.h"

namespace FastSample {
    template<typename T>
    manager<T>::manager(unsigned long long seed)
    : storage_normal(nCapacity), storage_uniform(nCapacity) {
        cudaMalloc(&rand_status, nStatus * sizeof(curandStateXORWOW_t));
        Fill_Status(seed, rand_status);
        Fill_Uniform(storage_uniform.storage_0);
        Fill_Uniform(storage_uniform.storage_1);

        Fill_Normal(storage_normal.storage_0);
        Fill_Normal(storage_normal.storage_1);

        cudaStreamSynchronize(storage_uniform.stream);
        cudaStreamSynchronize(storage_normal.stream);
    }

    template<typename T>
    manager<T>::~manager() {
        cudaFree(rand_status);

    }

    namespace {
        template<typename T>
        void INST() {
            manager<T> m(0);
        }

        __attribute__((unused)) void RUN() {
            INST<double>();
            INST<float>();
        }
    }
}