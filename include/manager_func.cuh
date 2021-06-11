//
// Created by Jie Ren (jieren9806@gmail.com) on 2021/6/11.
//

#pragma once

#include <curand_kernel.h>
#include "common.h"

namespace FastSample {
    namespace func {
        void sample_normal_with_stream(size_t flatted_shape, curandStateXORWOW_t* status, float* d_result, cudaStream_t* stream);
    }
}