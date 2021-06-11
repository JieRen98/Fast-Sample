//
// Created by Jie Ren (jieren9806@gmail.com) on 2021/6/11.
//

#pragma once
#include <cstddef>
#include <driver_types.h>

namespace FastSample {
    template<typename T>
    class storage {
    public:
        storage(size_t capacity);
        ~storage();

    private:
        size_t loc = 0;
        T* storage_0 = nullptr;
        T* storage_1 = nullptr;
        cudaStream_t stream = nullptr;
    };
}