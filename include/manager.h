//
// Created by Jie Ren (jieren9806@gmail.com) on 2021/6/11.
//

#pragma once
#include <curand.h>
#include <curand_kernel.h>
#include "common.h"
#include "cuda_runtime.h"

namespace FastSample {
    template<typename T>
    class manager {
    public:
        manager(unsigned long long seed);
        ~manager();

        void Fill_Normal(T* d_result);
        void Fill_Uniform(T* d_result);

    private:
        curandStateXORWOW_t *rand_status = nullptr;
        storage<T> storage_uniform;
        storage<T> storage_normal;
    };

    void Fill_Status(unsigned long long seed, curandStateXORWOW_t* status);
}
