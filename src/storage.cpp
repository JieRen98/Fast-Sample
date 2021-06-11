//
// Created by Jie Ren (jieren9806@gmail.com) on 2021/6/11.
//

#include "common.h"
#include "cuda_runtime.h"
#include "storage.h"

namespace FastSample {
    template<typename T>
    storage<T>::storage(size_t capacity) {
        cudaMalloc(&storage_0, capacity * sizeof(T));
        cudaMalloc(&storage_1, capacity * sizeof(T));
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    }

    template<typename T>
    storage<T>::~storage() {
        cudaStreamDestroy(stream);
        cudaFree(storage_0);
        cudaFree(storage_1);
    }

    namespace {
        template<typename T>
        void INST() {
            storage<T> s(0);
        }

        __attribute__((unused)) void RUN() {
            INST<double>();
            INST<float>();
        }
    }
}