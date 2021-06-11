//
// Created by Jie Ren (jieren9806@gmail.com) on 2021/6/11.
//

#pragma once
#include "common.h"

namespace FastSample {
    struct substorage{
        size_t require;
        size_t output_num;
        float* output;
        unsigned int* ui_output;
        float* meta_working;
        float* meta_backup;
        float* special_reserve;
        float* special_reserve_backup;
        distribution_kind kind;
    };
}
