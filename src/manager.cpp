//
// Created by Jie Ren (jieren9806@gmail.com) on 2021/6/11.
//

#include "manager.h"
#include "storage.h"

namespace FastSample {
    template<typename T>
    manager<T>::manager(unsigned long long seed, int Status_Num, size_t capacity)
    : storage_gaussian(capacity), storage_uniform(capacity) {
        cudaMalloc(&rand_status, Status_Num * sizeof(curandStateXORWOW_t));
        Fill_Status(seed, rand_status);
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