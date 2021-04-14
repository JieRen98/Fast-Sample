#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <stdio.h>
#include "myhead.h"

//#include <cooperative_groups.h>
//#include <cooperative_groups/memcpy_async.h>
//#include <cooperative_groups/reduce.h>
//using namespace cooperative_groups;
//namespace cg = cooperative_groups;


void update_normal_storage(size_t new_loc, storage* st, global_status* status) {
    float* tmp = st->storage_1;
    st->storage_1 = st->storage_0;
    st->storage_0 = tmp;
    st->loc = new_loc;
    _sample_normal_with_stream(nCapacity, status->rand_status, st->storage_1, st->stream);
}

void update_uniform_storage(size_t new_loc, storage* st, global_status* status) {
    float* tmp = st->storage_1;
    st->storage_1 = st->storage_0;
    st->storage_0 = tmp;
    st->loc = new_loc;
    _sample_uniform_with_stream(nCapacity, status->rand_status, st->storage_1, st->stream);
}

void require_uniform(substorage* sst, global_status* status) {
    storage* st = status->uniform_storage;
    size_t res = nCapacity - st->loc;
    if (res < sst->require)
    {
        cudaStreamSynchronize(*st->stream);
        if (res == 0)
            sst->meta_working = st->storage_1;
        else
        {
            cudaMemcpy(sst->meta_backup, st->storage_0 + st->loc, res * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(sst->meta_backup + res, st->storage_1, (sst->require - res) * sizeof(float), cudaMemcpyDeviceToDevice);
            sst->meta_working = sst->meta_backup;
        }

        update_uniform_storage(sst->require - res, st, status);
    }
    else
    {
        sst->meta_working = st->storage_0 + st->loc;
        st->loc += sst->require;
    }
}

void require_normal(substorage* sst, global_status* status) {
    storage* st = status->normal_storage;
    size_t res = nCapacity - st->loc;
    if (res < sst->require)
    {
        cudaStreamSynchronize(*st->stream);
        if (res == 0)
            sst->meta_working = st->storage_1;
        else
        {
            cudaMemcpy(sst->meta_backup, st->storage_0 + st->loc, res * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(sst->meta_backup + res, st->storage_1, (sst->require - res) * sizeof(float), cudaMemcpyDeviceToDevice);
            sst->meta_working = sst->meta_backup;
        }

        update_normal_storage(sst->require - res, st, status);
    }
    else
    {
        sst->meta_working = st->storage_0 + st->loc;
        st->loc += sst->require;
    }
}

void require_normal_and_uniform(substorage* sst, global_status* status) {
    storage* st_normal = status->normal_storage;
    storage* st_uniform = status->uniform_storage;
    size_t res_normal = nCapacity - st_normal->loc;
    size_t res_uniform = nCapacity - st_uniform->loc;
    if (res_normal < sst->require && res_uniform < sst->require)
    {
        cudaStreamSynchronize(*st_normal->stream);
        cudaStreamSynchronize(*st_uniform->stream);
        if (res_normal == 0 && res_uniform == 0) {
            sst->meta_working = st_normal->storage_1;
            sst->special_reserve = st_uniform->storage_1;
        }
        else if (res_normal == 0)
            sst->meta_working = st_normal->storage_1;
        else if (res_uniform == 0)
            sst->special_reserve = st_uniform->storage_1;
        else
        {
            if (!sst->special_reserve_backup)
                cudaMalloc(&sst->special_reserve_backup, sst->require * sizeof(float));
            cudaMemcpy(sst->meta_backup, st_normal->storage_0 + st_normal->loc, res_normal * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(sst->meta_backup + res_normal, st_normal->storage_1, (sst->require - res_normal) * sizeof(float), cudaMemcpyDeviceToDevice);
            sst->meta_working = sst->meta_backup;

            cudaMemcpy(sst->special_reserve_backup, st_uniform->storage_0 + st_uniform->loc, res_uniform * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(sst->special_reserve_backup + res_uniform, st_uniform->storage_1, (sst->require - res_uniform) * sizeof(float), cudaMemcpyDeviceToDevice);
            sst->special_reserve = sst->special_reserve_backup;
        }

        update_normal_storage(sst->require - res_normal, st_normal, status);
        update_uniform_storage(sst->require - res_uniform, st_uniform, status);
    }
    else if (res_normal < sst->require) {
        cudaStreamSynchronize(*st_normal->stream);
        if (res_normal == 0)
            sst->meta_working = st_normal->storage_1;
        else
        {
            cudaMemcpy(sst->meta_backup, st_normal->storage_0 + st_normal->loc, res_normal * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(sst->meta_backup + res_normal, st_normal->storage_1, (sst->require - res_normal) * sizeof(float), cudaMemcpyDeviceToDevice);
            sst->meta_working = sst->meta_backup;
        }

        sst->special_reserve = st_uniform->storage_0 + st_uniform->loc;
        st_uniform->loc += sst->require;

        update_normal_storage(sst->require - res_normal, st_normal, status);
    }
    else if (res_uniform < sst->require) {
        cudaStreamSynchronize(*st_uniform->stream);
        if (res_uniform == 0)
            sst->special_reserve = st_uniform->storage_1;
        else
        {
            if (!sst->special_reserve_backup)
                cudaMalloc(&sst->special_reserve_backup, sst->require * sizeof(float));
            cudaMemcpy(sst->special_reserve_backup, st_uniform->storage_0 + st_uniform->loc, res_uniform * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(sst->special_reserve_backup + res_uniform, st_uniform->storage_1, (sst->require - res_uniform) * sizeof(float), cudaMemcpyDeviceToDevice);
            sst->special_reserve = sst->special_reserve_backup;
        }

        sst->meta_working = st_normal->storage_0 + st_normal->loc;
        st_normal->loc += sst->require;

        update_uniform_storage(sst->require - res_uniform, st_uniform, status);
    }
    else {
        sst->meta_working = st_normal->storage_0 + st_normal->loc;
        st_normal->loc += sst->require;

        sst->special_reserve = st_uniform->storage_0 + st_uniform->loc;
        st_uniform->loc += sst->require;
    }
}

__global__ void _status(unsigned long long seed, curandStateXORWOW_t* status, size_t offset) {
    // We use the init_status_shared_stream function as the optimal init function.
    extern __shared__ curandStateXORWOW_t status_shared[];
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, (unsigned long long)(idx + offset), (unsigned long long)0, &status_shared[threadIdx.x]);
    status[idx + offset] = status_shared[threadIdx.x];
}

__global__ void _status_shared_only (unsigned long long seed, curandStateXORWOW_t* status) {
    __shared__ curandStateXORWOW_t status_shared[nThreads_Status_y][nThreads_Status_x];
    unsigned int idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    curand_init(seed, (unsigned long long)idx, 0, &status_shared[threadIdx.y][threadIdx.x]);
    status[idx] = status_shared[threadIdx.y][threadIdx.x];
}


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

__global__ void _uniform(float* output, curandStateXORWOW_t* status, unsigned int global_loc) {
    // We use the normal_unroll function as the optimal sample function.
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    float* start_loc = output + global_loc * nStatus * nUnroll + idx;
    curandStateXORWOW_t status_loc = status[idx + nStatus];
#pragma unroll
    for (unsigned int i = 0; i < nUnroll; i++)
        start_loc[nStatus * i] = curand_uniform(&status_loc);

    status[idx + nStatus] = status_loc;
}

__global__ void _poisson(float lambda, float sqrt_lambda, unsigned int* output, float* normal) {
    size_t idx = (threadIdx.x + blockIdx.x * blockDim.x) * nUnroll_poisson;

    for (unsigned int i = 0; i < nUnroll_poisson; i++)
        output[idx + i] = (unsigned int)((sqrt_lambda * normal[idx + i]) + lambda + 0.5);
}

__global__ void _exponential(float inv_lambda, float* uniform_array, float* exp_array) {
    // make sure input inverse of lambda instead of lambda!
    size_t gap = gridDim.x * blockDim.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float* uniform_start = uniform_array + idx;
    float* exp_start = exp_array + idx;
#pragma unroll
    for (int i = 0; i < nUnroll_exp; i++)
        exp_start[gap * i] = -logf(uniform_start[gap * i]) * inv_lambda;
}

void _sample_uniform_with_stream(size_t flatted_shape, curandStateXORWOW_t* status, float* d_result, cudaStream_t* stream) {
    dim3 block_Elm(nThreads_base_sample), grid_Elm(nStatus / nThreads_base_sample);
    unsigned int nGenRep = flatted_shape / nStatus;

    for (unsigned int i = 0; i < nGenRep / nUnroll; i++)
        _uniform << <grid_Elm, block_Elm, 0, *stream >> > (d_result, status, i);

}

void _sample_normal_with_stream(size_t flatted_shape, curandStateXORWOW_t* status, float* d_result, cudaStream_t* stream) {
    dim3 block_Elm(nThreads_base_sample), grid_Elm(nStatus / nThreads_base_sample);
    unsigned int nGenRep = flatted_shape / nStatus;

    for (unsigned int i = 0; i < nGenRep / nUnroll; i++)
        _normal << <grid_Elm, block_Elm, 0, *stream >> > (d_result, status, i);

}

__global__ void _gamma_b1(float scale, float d, float c, float* output, float* normal, float* uniform, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t state_idx = idx % (nStatus * nStatus_res_size);
    bool accept = false;
    float result;

    float n = normal[idx];
    float u = uniform[idx];
    float ocn = 1 + c * n;
    while (ocn <= 0.0) {
        n = curand_normal(&status[state_idx + nStatus * 2]);
        ocn = 1 + c * n;
    }
    float ocn_pow3 = ocn * ocn * ocn;
    if (u < 1.0 - 0.0331 * (n * n) * (n * n) || logf(u) < 0.5 * n * n + d * (1. - ocn_pow3 + logf(ocn_pow3))) {
        result = d * ocn_pow3 * scale;
        accept = true;
    }

    while (!accept) {
        n = curand_normal(&status[state_idx + nStatus * 2]);
        u = curand_uniform(&status[state_idx + nStatus * 2]);
        ocn = 1 + c * n;
        while (ocn <= 0.0) {
            n = n = curand_normal(&status[state_idx + nStatus * 2]);
            ocn = 1 + c * n;
        }
        ocn_pow3 = ocn * ocn * ocn;
        if (u < 1.0 - 0.0331 * (n * n) * (n * n) || logf(u) < 0.5 * n * n + d * (1. - ocn_pow3 + logf(ocn_pow3))) {
            result = d * ocn_pow3 * scale;
            accept = true;
        }
    }
    output[idx] = result;
}

//__global__ void _gamma_b1_unroll(float scale, float d, float c, float* output, float* normal, float* uniform, curandStateXORWOW_t* status) {
//    //useage:
//    //_fit_sst_for_gamma((substorage*)sst);
//    //dim3 grid(((substorage*)sst)->require / nThreads_gamma), block(nThreads_gamma * gamma_AR_worker);
//    //_gamma_s1_unroll << <grid, block >> > (scale, d, c, ((substorage*)sst)->output, ((substorage*)sst)->meta_working, ((substorage*)sst)->special_reserve, ((global_status*)status)->rand_status);
//
//    float result;
//    cg::grid_group grid = cg::this_grid();
//    cg::thread_block tb = cg::this_thread_block();
//    cg::thread_block_tile<gamma_AR_worker> local_block = cg::tiled_partition<gamma_AR_worker>(tb);
//    unsigned long long idx = threadIdx.x + blockDim.x * blockIdx.x;
//    unsigned long long state_idx = idx % nStatus;
//    bool accept = false, any_accepted = false;
//
//    float n = normal[idx], u = uniform[idx];
//    float ocn = 1 + c * n;
//    float ocn_pow3;
//    if (ocn > 0.0) {
//        ocn_pow3 = ocn * ocn * ocn;
//        if (u < 1.0 - 0.0331 * (n * n) * (n * n) || logf(u) < 0.5 * n * n + d * (1. - ocn_pow3 + logf(ocn_pow3))) {
//            result = d * ocn_pow3 * scale;
//            accept = true;
//        }
//    }
//    local_block.sync();
//    any_accepted = cg::reduce(local_block, accept, cg::bit_or<bool>());
//
//    while (!any_accepted) {
//        n = curand_normal(&status[state_idx]);
//        u = curand_uniform(&status[state_idx]);
//        ocn = 1 + c * n;
//        if (ocn > 0.0) {
//            ocn_pow3 = ocn * ocn * ocn;
//            if (u < 1.0 - 0.0331 * (n * n) * (n * n) || logf(u) < 0.5 * n * n + d * (1. - ocn_pow3 + logf(ocn_pow3))) {
//                result = d * ocn_pow3 * scale;
//                accept = true;
//            }
//        }
//        local_block.sync();
//        any_accepted = cg::reduce(local_block, accept, cg::bit_or<bool>());
//    }
//    if (accept)
//        output[idx / gamma_AR_worker] = result;
//}

__global__ void _gamma_s1(float shape, float scale, float* output, float* uniform, float* exponential, curandStateXORWOW_t* status) {
    unsigned long long idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned long long state_idx = idx % nStatus;

    float e = exponential[idx], u = uniform[idx];
    float ret, tmp;
    bool accept=false;

    if (u <= 1.0 - shape) {
        ret = powf(u, 1. / shape);
        if (ret <= e)
            accept = true;
    }
    else {
        tmp = -logf((1 - u) / shape);
        ret = powf(1.0 - shape + shape * tmp, 1. / shape);
        if (ret <= (e + tmp))
            accept = true;
    }

    while (!accept) {
        u = curand_uniform(&status[state_idx + nStatus * 2]);
        e = -logf(curand_uniform(&status[state_idx + nStatus * 3]));
        if (u <= 1.0 - shape) {
            ret = powf(u, 1. / shape);
            if (ret <= e)
                accept = true;
        }
        else {
            tmp = -logf((1 - u) / shape);
            ret = powf(1.0 - shape + shape * tmp, 1. / shape);
            if (ret <= (e + tmp))
                accept = true;
        }
    }
    output[idx] = ret * scale;
}

__global__ void _rand_gamma(float *shape, float *scale, float* output, float* normal, float* uniform,size_t matrix_scale, curandStateXORWOW_t* status, cudaStream_t stream = 0) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t matrix_idx = idx % matrix_scale;
    float sh = shape[matrix_idx];
    float sc = scale[matrix_idx];
    //printf("%f", sh);
    if (sh > 1.0) {
        float d = sh - one_third;
        float c = one_third / sqrt(d);
        size_t state_idx = idx % (nStatus * nStatus_res_size);
        bool accept = false;
        float result;

        float n = normal[idx]; // (sst->meta_working)[idx];  
        float u = uniform[idx]; // (sst->special_reserve)[idx];     
        float ocn = 1 + c * n;
        while (ocn <= 0.0) {
            n = curand_normal(&status[state_idx + nStatus * 2]);
            ocn = 1 + c * n;
        }
        float ocn_pow3 = ocn * ocn * ocn;
        if (u < 1.0 - 0.0331 * (n * n) * (n * n) || logf(u) < 0.5 * n * n + d * (1. - ocn_pow3 + logf(ocn_pow3))) {
            result = d * ocn_pow3 * sc;
            accept = true;
        }

        while (!accept) {
            n = curand_normal(&status[state_idx + nStatus * 2]);
            u = curand_uniform(&status[state_idx + nStatus * 2]);
            ocn = 1 + c * n;
            while (ocn <= 0.0) {
                n = n = curand_normal(&status[state_idx + nStatus * 2]);
                ocn = 1 + c * n;
            }
            ocn_pow3 = ocn * ocn * ocn;
            if (u < 1.0 - 0.0331 * (n * n) * (n * n) || logf(u) < 0.5 * n * n + d * (1. - ocn_pow3 + logf(ocn_pow3))) {
                result = d * ocn_pow3 * sc;
                accept = true;
            }
        }
        output[idx] = result; //(sst->output)[idx] = result;
    }
    else if (sh < 1) {
        size_t state_idx = idx % nStatus;
        float u = *(uniform + idx);
        float e = -logf(u) * 1.0; //exponential
        u = curand_uniform(&status[state_idx]);
        float ret, tmp;
        bool accept = false;

        if (u <= 1.0 - sh) {
            ret = powf(u, 1. / sh);
            if (ret <= e)
                accept = true;
        }
        else {
            tmp = -logf((1 - u) / sh);
            ret = powf(1.0 - sh + sh * tmp, 1. / sh);
            if (ret <= (e + tmp))
                accept = true;
        }

        while (!accept) {
            u = curand_uniform(&status[state_idx + nStatus * 2]);
            e = -logf(curand_uniform(&status[state_idx + nStatus * 3]));
            if (u <= 1.0 - sh) {
                ret = powf(u, 1. / sh);
                if (ret <= e)
                    accept = true;
            }
            else {
                tmp = -logf((1 - u) / sh);
                ret = powf(1.0 - sh + sh * tmp, 1. / sh);
                if (ret <= (e + tmp))
                    accept = true;
            }
        }
        output[idx] = ret * sc;
    }
    else {  // shape=1
        float u = *(uniform + idx);
        (output + idx)[0] = -logf(u) * sc;
    }
}

__global__ void _crt_level0(float alpha, unsigned int num, unsigned int rep, unsigned int* output, float* uniform) {
    __shared__ unsigned int result_local[nThreads_crt * crt_AR_worker_level0];
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idx_local = idx % crt_AR_worker_level0;
    unsigned int result_atom = 0;
    unsigned int idx_local_working = idx_local;
    for (unsigned int i = 0; i < rep; i++) {
        if (idx_local_working < num)
            result_atom += uniform[idx * rep + i] < (alpha / (alpha + (float)idx_local_working)) ? 1 : 0;
        idx_local_working += crt_AR_worker_level0;
    }
    result_local[threadIdx.x] = result_atom;

#pragma unroll
    for (unsigned int i = crt_AR_worker_level0 / 2; i > 0; i /= 2) {
        if (idx_local < i)
            result_local[threadIdx.x] = result_local[threadIdx.x] + result_local[threadIdx.x + i];
        __syncthreads();
    }

    if (idx_local == 0)
        output[idx / crt_AR_worker_level0] = result_local[threadIdx.x];
}

__global__ void _crt_level1(float alpha, unsigned int num, unsigned int rep, unsigned int* output, float* uniform) {
    __shared__ unsigned int result_local[nThreads_crt * crt_AR_worker_level1];
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idx_local = idx % crt_AR_worker_level1;
    unsigned int result_atom = 0;
    unsigned int idx_local_working = idx_local;
    for (unsigned int i = 0; i < rep; i++) {
        if (idx_local_working < num)
            result_atom += uniform[idx * rep + i] < (alpha / (alpha + (float)idx_local_working)) ? 1 : 0;
        idx_local_working += crt_AR_worker_level1;
    }
    result_local[threadIdx.x] = result_atom;

#pragma unroll
    for (unsigned int i = crt_AR_worker_level1 / 2; i > 0; i /= 2) {
        if (idx_local < i)
            result_local[threadIdx.x] = result_local[threadIdx.x] + result_local[threadIdx.x + i];
        __syncthreads();
    }

    if (idx_local == 0)
        output[idx / crt_AR_worker_level1] = result_local[threadIdx.x];
}

__global__ void _crt_level2(float alpha, unsigned int num, unsigned int rep, unsigned int* output, float* uniform) {
    __shared__ unsigned int result_local[nThreads_crt * crt_AR_worker_level2];
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idx_local = idx % crt_AR_worker_level2;
    unsigned int result_atom = 0;
    unsigned int idx_local_working = idx_local;
    for (unsigned int i = 0; i < rep; i++) {
        if (idx_local_working < num)
            result_atom += uniform[idx * rep + i] < (alpha / (alpha + (float)idx_local_working)) ? 1 : 0;
        idx_local_working += crt_AR_worker_level2;
    }
    __syncthreads();
    result_local[threadIdx.x] = result_atom;

#pragma unroll
    for (unsigned int i = crt_AR_worker_level2 / 2; i > 0; i /= 2) {
        if (idx_local < i)
            result_local[threadIdx.x] = result_local[threadIdx.x] + result_local[threadIdx.x + i];
        __syncthreads();
    }

    if (idx_local == 0)
        output[idx / crt_AR_worker_level2] = result_local[threadIdx.x];
}

__constant__ float distribution_fit_symbol[multinomial_class_num];

__global__ void _multinomial_atom_post_shared(unsigned int num, unsigned int inner_rep, unsigned int max_rep, unsigned int copy_rep, unsigned int* output, float* uniform) {
    //extern __shared__ unsigned int result_local[nThreads_multinomial][multinomial_class_num];
    extern __shared__ unsigned int result_local[];
    size_t idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    float u;
    unsigned int selected_class;
    unsigned int idx_local = threadIdx.x;
    unsigned int* result_local_biased = result_local + threadIdx.y * num;

    // AR_worker should larger than multinomial_class_num
    if (threadIdx.x < multinomial_class_num)
        result_local_biased[threadIdx.x] = 0;

    for (unsigned int i = 0; i < inner_rep; i++) {
        if (idx_local < max_rep) {
            u = uniform[idx * inner_rep + i];

            for (selected_class = 0; selected_class < num; selected_class++)
                if (u < distribution_fit_symbol[selected_class])
                    break;

            atomicAdd(&result_local_biased[selected_class], 1);
        }
        idx_local += multinomial_AR_worker_level2;
    }
    __syncthreads();

    idx_local = threadIdx.x;
    for (unsigned int i = 0; i < copy_rep; i++) {
        if (idx_local < num)
            output[(threadIdx.y + blockIdx.x * blockDim.y) * num + idx_local] = result_local_biased[idx_local];
        idx_local += blockDim.x;
    }
}

__global__ void _multinomial(unsigned int num, unsigned int inner_rep, unsigned int max_rep, unsigned int* output, float* uniform) {
    __shared__ unsigned int result_local[nThreads_multinomial][multinomial_AR_worker_level2][multinomial_class_num];
    size_t idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    float u;
    unsigned int selected_class;
    unsigned int idx_local = threadIdx.x;

    for (unsigned int j = 0; j < num; j++)
        result_local[threadIdx.y][threadIdx.x][j] = 0;

    for (unsigned int i = 0; i < inner_rep; i++){
        if (idx_local < max_rep) {
            u = uniform[idx * inner_rep + i];

            for (selected_class = 0; selected_class < num; selected_class++) {
                if (u < distribution_fit_symbol[selected_class])
                    break;
            }
            result_local[threadIdx.y][threadIdx.x][selected_class]++;
        }
        idx_local += multinomial_AR_worker_level2;
    }

    __syncthreads();
#pragma unroll
    for (unsigned int i = multinomial_AR_worker_level2 / 2; i > 0; i /= 2) {
        if (threadIdx.x < i)
            for (unsigned int j = 0; j < num; j++)
                result_local[threadIdx.y][threadIdx.x][j] = result_local[threadIdx.y][threadIdx.x][j] + result_local[threadIdx.y][threadIdx.x + i][j];

        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (unsigned int j = 0; j < num; j++)
            output[(threadIdx.y + blockIdx.x * blockDim.y) * num + j] = result_local[threadIdx.y][threadIdx.x][j];
}

__global__ void _multinomial_atom(unsigned int num, unsigned int inner_rep, unsigned int max_rep, unsigned int* output, float* uniform) {
    __shared__ unsigned int result_local[nThreads_multinomial][multinomial_class_num];
    size_t idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    float u;
    unsigned int selected_class;
    unsigned int idx_local = threadIdx.x;

    // AR_worker should larger than multinomial_class_num
    if (threadIdx.x < multinomial_class_num)
        result_local[threadIdx.y][threadIdx.x] = 0;

    for (unsigned int i = 0; i < inner_rep; i++) {
        if (idx_local < max_rep) {
            u = uniform[idx * inner_rep + i];

            for (selected_class = 0; selected_class < num; selected_class++)
                if (u < distribution_fit_symbol[selected_class])
                    break;

            atomicAdd(&result_local[threadIdx.y][selected_class], 1);
        }
        idx_local += multinomial_AR_worker_level2;
    }
    __syncthreads();

    if (threadIdx.x == 0)
        for (unsigned int j = 0; j < num; j++)
            output[(threadIdx.y + blockIdx.x * blockDim.y) * num + j] = result_local[threadIdx.y][j];
}

__global__ void _normalize_dirichlet(unsigned int class_num, float* output) {
    extern __shared__ float output_smem[];
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    float* output_smem_biased = output_smem + threadIdx.x * class_num;
    float sum = 0, inv_sum;
    for (unsigned int i = 0; i < class_num; i++)
        output_smem_biased[i] =  output[idx + i * blockDim.x * gridDim.x];

    for (unsigned int i = 0; i < class_num; i++)
        sum += output_smem_biased[i];
    inv_sum = 1 / sum;
    for (unsigned int i = 0; i < class_num; i++)
        output_smem_biased[i] *= inv_sum;

    for (unsigned int i = 0; i < class_num; i++)
        output[idx + i * blockDim.x * gridDim.x] = output_smem_biased[i];
}

extern "C" void sample_multinomial(float* distribution, unsigned int num, unsigned int rep, void* sst, void* status) {
    if (num == 0 || rep == 0)
    {
        printf("The number of the categories or the repeat time should larger than 0, no action has been taken.\n");
        return;
    }
    float* distribution_fit = (float*)malloc(num * sizeof(float));
    distribution_fit[0] = distribution[0];
    for (unsigned int i = 0; i < num - 1; i++)
    {
        if (distribution[i] < 0) {
            printf("The distribution parameter should larger than 0, no action has been taken.\n");
            return;
        }
        distribution_fit[i + 1] = distribution_fit[i] + distribution[i + 1];
    }
    if (distribution_fit[num - 1] != 1) {
        printf("The summation of the categorical distribution should equal to 1, no action has been taken.\n");
        return;
    }
    substorage* _sst = (substorage*)sst;
    //float* d_distribution_fit;
    //cudaMalloc(&d_distribution_fit, num * sizeof(float));
    //cudaMemcpy(d_distribution_fit, distribution_fit, num * sizeof(float), cudaMemcpyHostToDevice);
    dim3 grid(_sst->output_num / nThreads_multinomial), block(multinomial_AR_worker_level2, nThreads_multinomial);
    cudaMemcpyToSymbol(distribution_fit_symbol, distribution_fit, num * sizeof(float));
    _fit_multinomial_substorage(num, rep, multinomial_AR_worker_level2, (substorage*)sst);
    require_uniform((substorage*)sst, (global_status*)status);
    //_multinomial << < grid, block >> > (num, (rep + multinomial_AR_worker_level2 - 1) / multinomial_AR_worker_level2, rep, _sst->ui_output, _sst->meta_working);
    //_multinomial_atom << < grid, block >> > (num, (rep + multinomial_AR_worker_level2 - 1) / multinomial_AR_worker_level2, rep, _sst->ui_output, _sst->meta_working);
    _multinomial_atom_post_shared << < grid, block, nThreads_multinomial * num * sizeof(unsigned int) >> > (num, (rep + multinomial_AR_worker_level2 - 1) / multinomial_AR_worker_level2, rep, (num + nThreads_multinomial - 1) / nThreads_multinomial, _sst->ui_output, _sst->meta_working);

    free(distribution_fit);
}

extern "C" void sample_crt(float alpha, unsigned int num, void* sst, void* status) {
    if (num == 0 || alpha <= 0)
    {
        printf("The table number should larger than 0 and $\\alpha$ should bigger than 0 too, no action has benn taken.\n");
        return;
    }
    substorage* _sst = (substorage*)sst;

    if (num >= crt_AR_worker_level2) {
        dim3 grid(_sst->require / nThreads_crt), block(nThreads_crt * crt_AR_worker_level2);
        _fit_crt_substorage(num, crt_AR_worker_level2, (substorage*)sst);
        require_uniform((substorage*)sst, (global_status*)status);
        _crt_level2 << < grid, block >> > (alpha, num, (num + crt_AR_worker_level2 - 1) / crt_AR_worker_level2, _sst->ui_output, _sst->meta_working);
    }
    else if (num >= crt_AR_worker_level1) {
        dim3 grid(_sst->require / nThreads_crt), block(nThreads_crt * crt_AR_worker_level1);
        _fit_crt_substorage(num, crt_AR_worker_level1, (substorage*)sst);
        require_uniform((substorage*)sst, (global_status*)status);
        _crt_level1 << < grid, block >> > (alpha, num, (num + crt_AR_worker_level1 - 1) / crt_AR_worker_level1, _sst->ui_output, _sst->meta_working);
    }
    else {
        dim3 grid(_sst->require / nThreads_crt), block(nThreads_crt * crt_AR_worker_level0);
        _fit_crt_substorage(num, crt_AR_worker_level0, (substorage*)sst);
        require_uniform((substorage*)sst, (global_status*)status);
        _crt_level0 << < grid, block >> > (alpha, num, (num + crt_AR_worker_level0 - 1) / crt_AR_worker_level0, _sst->ui_output, _sst->meta_working);
    }
}

extern "C" void sample_gamma(float shape, float scale, void* sst, void* status, cudaStream_t stream = 0) {
    if (shape < 0.0 || scale < 0.0)
    {
        printf("The shape and scale parameter should larger than 0, no action has been taken.\n");
        return;
    }
    substorage* _sst = (substorage*)sst;

    if (shape == 1.0)
        sample_exponential(scale, sst, status, stream);
    else if (shape > 1.0) {
        float d = shape - one_third;
        float c = one_third / sqrt(d);
        require_normal_and_uniform(_sst, (global_status*)status);
        dim3 grid(_sst->require / nThreads_gamma), block(nThreads_gamma);
        _gamma_b1 << <grid, block, 0, stream >> > (scale, d, c, _sst->output, _sst->meta_working, _sst->special_reserve, ((global_status*)status)->rand_status);
    }
    else {
        sample_exponential(1.0, sst, status, stream);
//        _sst->special_reserve = _sst->meta_working;
//        _sst->special_reserve_backup = _sst->meta_backup;
        require_uniform(_sst, (global_status*)status);
        dim3 grid(_sst->require / nThreads_gamma), block(nThreads_gamma);
//        _gamma_s1 << <grid, block, 0, stream >> > (shape, scale, _sst->output, _sst->meta_working, _sst->special_reserve, ((global_status*)status)->rand_status);
        _gamma_s1 << <grid, block, 0, stream >> > (shape, scale, _sst->output, _sst->meta_working, _sst->output, ((global_status*)status)->rand_status);
    }
}

extern "C" void sample_multi_gamma(float* shape_host, float* scale_host, void* sst, void* status, int repeat, cudaStream_t stream = 0) {
    substorage* _sst = (substorage*)sst;
    size_t matrix_scale = _sst->require / repeat;
    float* shape, * scale;
    int nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&shape, nBytes);
    cudaMalloc((void**)&scale, nBytes);

    cudaMemcpy(shape, shape_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(scale, scale_host, nBytes, cudaMemcpyHostToDevice);

    require_normal_and_uniform(_sst, (global_status*)status);

    dim3 grid(_sst->require / nThreads_gamma), block(nThreads_gamma);
    _rand_gamma << <grid, block, 0, stream >> > (shape, scale, _sst->output, _sst->meta_working,_sst->special_reserve, matrix_scale,((global_status*)status)->rand_status);
 
    cudaFree(shape);
    cudaFree(scale);
}

extern "C" void sample_exponential(float inv_lambda, void* sst, void* status, cudaStream_t stream = 0) {
    // make sure input inverse of lambda instead of lambda!
    if (inv_lambda < 0)
    {
        printf("The inverse lambda parameter should larger than 0, no action has been taken.\n");
        return;
    }
    substorage* _sst = (substorage*)sst;

    require_uniform(_sst, (global_status*)status);
    dim3 grid(_sst->require / nThreads_exp / nUnroll_exp), block(nThreads_exp);
    _exponential << < grid, block, 0, stream >> > (inv_lambda, _sst->meta_working, _sst->output);
}

extern "C" void sample_normal(void* sst, void* status) {
    substorage* _sst = (substorage*)sst;

    require_normal((substorage*)sst, (global_status*)status);
    cudaMemcpy(_sst->output, _sst->meta_working, _sst->require * sizeof(float), cudaMemcpyDeviceToDevice);
}

extern "C" void sample_uniform(void* sst, void* status) {
    substorage* _sst = (substorage*)sst;
    require_uniform(_sst, (global_status*)status);
    cudaMemcpy(_sst->output, _sst->meta_working, _sst->require * sizeof(float), cudaMemcpyDeviceToDevice);
}

extern "C" void sample_poisson(float lambda, void* sst, void* status) {
    substorage* _sst = (substorage*)sst;
    _fit_poisson_substorage(_sst);
    require_uniform(_sst, (global_status*)status);
    dim3 grid(_sst->require / nUnroll_poisson / nThreads_poisson), block(nThreads_poisson);
    _poisson << < grid, block >> > (lambda, sqrt(lambda), _sst->ui_output, _sst->meta_working);
}

extern "C" void sample_dirichlet(float* alpha, unsigned int class_num, void* sst, void* status) {
    substorage* _sst = (substorage*)sst;
    _fit_dirichlet_substorage(class_num, _sst);
    substorage** _ssts = (substorage**)_sst->meta_working;
    cudaStream_t* streams = (cudaStream_t*)malloc(class_num * sizeof(cudaStream_t));
    for (unsigned int i = 0; i < class_num; i++)
        cudaStreamCreate(&streams[i]);
    for (unsigned int i = 0; i < class_num; i++){
        sample_gamma(alpha[i], 1., (void*)_ssts[i], status, streams[i]);
        cudaMemcpyAsync(_sst->output + _sst->require * i, _ssts[i]->output, _sst->require * sizeof(float), cudaMemcpyDeviceToDevice, streams[i]);
    }
    for (unsigned int i = 0; i < class_num; i++)
        cudaStreamDestroy(streams[i]);

    //float* h_result = (float*)malloc(_sst->require * 8 * sizeof(float));
    //cudaMemcpy(h_result, ((substorage*)sst)->output, _sst->require * 8 * sizeof(float), cudaMemcpyDeviceToHost);
    //float mean = 0.;
    //for (size_t i = 0; i < _sst->require * 8; i += _sst->require)
    //    mean += h_result[i + 1];
    //printf("%f", (float)mean);

    dim3 grid(_sst->require / nThreads_dirichlet), block(nThreads_dirichlet);
    _normalize_dirichlet << < grid, block, nThreads_dirichlet * class_num * sizeof(float), 0 >> > (class_num, _sst->output);
    free(streams);
}

//void _init_rand_status(int seed, global_status* status) {
//    size_t nBlocks_Status = nStatus / nThreads_Status;
//    dim3 block_Status(nThreads_Status), grid_Status(nBlocks_Status);
//
//    cudaMalloc((void**)&status->rand_status, nStatus * sizeof(curandStateXORWOW_t));
//
//    cudaStream_t stream_status_init[nStream_Status_Init];
//    for (int i = 0; i < nStream_Status_Init; i++)
//        cudaStreamCreate(&stream_status_init[i]);
//
//    dim3 grid_Status_Stream(grid_Status.x / nStream_Status_Init);
//    size_t nStream = nStatus / nStream_Status_Init;
//    for (int i = 0; i < nStream_Status_Init; i++)
//        _status << <grid_Status_Stream, block_Status, nThreads_Status * sizeof(curandStateXORWOW_t), stream_status_init[i] >> > (seed, status->rand_status, nStream * i);
//
//    for (int i = 0; i < nStream_Status_Init; i++)
//        cudaStreamDestroy(stream_status_init[i]);
//}

void _init_rand_status_shared_only(int seed, global_status* status) {
    dim3 grid(nStatus_all / nThreads_Status), block(nThreads_Status_x, nThreads_Status_y);

    cudaMalloc((void**)&status->rand_status, nStatus_all * sizeof(curandStateXORWOW_t));

    _status_shared_only << <grid, block >> > (seed, status->rand_status);

}

void _init_uniform_storage(global_status* status) {
    storage* st = (storage*)malloc(sizeof(storage));
    st->loc = 0;
    st->stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
    cudaMalloc(&st->storage_0, nCapacity * sizeof(float));
    cudaMalloc(&st->storage_1, nCapacity * sizeof(float));
    cudaStreamCreateWithFlags(st->stream, cudaStreamNonBlocking);
    _sample_uniform_with_stream(nCapacity, status->rand_status, st->storage_0, st->stream);
    _sample_uniform_with_stream(nCapacity, status->rand_status, st->storage_1, st->stream);
    status->uniform_storage = st;
}

void _init_normal_storage(global_status* status) {
    storage* st = (storage*)malloc(sizeof(storage));
    st->loc = 0;
    st->stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
    cudaMalloc(&st->storage_0, nCapacity * sizeof(float));
    cudaMalloc(&st->storage_1, nCapacity * sizeof(float));
    cudaStreamCreateWithFlags(st->stream, cudaStreamNonBlocking);
    _sample_normal_with_stream(nCapacity, status->rand_status, st->storage_0, st->stream);
    _sample_normal_with_stream(nCapacity, status->rand_status, st->storage_1, st->stream);
    status->normal_storage = st;
}

void _destroy_storage(void* st) {
    cudaFree(((storage*)st)->storage_0);
    cudaFree(((storage*)st)->storage_1);
    free(((storage*)st)->stream);
    free(st);
}

void _destroy_rand_status(void* status) {
    cudaFree(status);
}

//void _fit_sst_for_gamma(substorage* sst) {
//    if (sst->gamma == false) {
//        cudaFree(sst->meta_backup);
//        sst->require *= gamma_AR_worker;
//        cudaMalloc(&sst->meta_backup, sst->require * sizeof(float));
//        sst->gamma = true;
//    }
//}

//void _withdraw_sst_for_gamma(substorage* sst) {
//    if (sst->gamma == true) {
//        cudaFree(sst->meta_backup);
//        sst->require /= gamma_AR_worker;
//        cudaMalloc(&sst->meta_backup, sst->require * sizeof(float));
//        sst->gamma = false;
//    }
//}

void _fit_dirichlet_substorage(unsigned int class_num, substorage* sst) {
    if (sst->kind != kind_dirichlet) {
        cudaFree(sst->output);
        cudaMalloc(&sst->output, sst->require * class_num * sizeof(float));

        substorage** ssts = (substorage**)malloc(class_num * sizeof(substorage*));

        for (unsigned int i = 0; i < class_num; i++)
            ssts[i] = (substorage*)init_substorage(sst->require);

        sst->kind = kind_dirichlet;
        sst->ui_output = (unsigned int*)malloc(sizeof(unsigned int));
        sst->ui_output[0] = class_num;
        sst->meta_working = (float*)ssts;
    }
}

void _fit_crt_substorage(unsigned int class_num, unsigned int worker_num, substorage* sst) {
    if (sst->kind != kind_crt) {
        _fit_general_substorage(sst);
        //cudaFree(sst->output);
        cudaMalloc(&sst->ui_output, sst->require * sizeof(unsigned int));

        sst->require *= worker_num * ((class_num + worker_num - 1) / worker_num);
        cudaFree(sst->meta_backup);
        cudaMalloc(&sst->meta_backup, sst->require * sizeof(unsigned int));
        sst->meta_working = NULL;
        sst->special_reserve = NULL;
        sst->special_reserve_backup = NULL;
        sst->kind = kind_crt;
    }
}

void _fit_poisson_substorage(substorage* sst) {
    if (sst->kind != kind_poisson) {
        _fit_general_substorage(sst);
        //cudaFree(sst->output);
        cudaMalloc(&sst->ui_output, sst->require * sizeof(unsigned int));
        sst->kind = kind_poisson;
    }
}

void _fit_multinomial_substorage(unsigned int class_num, unsigned int rep, unsigned int worker_num, substorage* sst) {
    if (sst->kind != kind_multinomial) {
        _fit_general_substorage(sst);
        //cudaFree(sst->output);
        cudaMalloc(&sst->ui_output, sst->require * class_num * sizeof(unsigned int));

        sst->require *= worker_num * ((rep + worker_num - 1) / worker_num);
        //sst->output_num *= class_num;
        cudaFree(sst->meta_backup);
        cudaMalloc(&sst->meta_backup, sst->require * sizeof(unsigned int));
        sst->meta_working = NULL;
        sst->special_reserve = NULL;
        sst->special_reserve_backup = NULL;
        sst->kind = kind_multinomial;
    }
}

void _fit_general_substorage(substorage* sst) {
    sst->require = sst->output_num;
    switch (sst->kind) {
    case kind_crt: {
        cudaFree(sst->ui_output);
        cudaFree(sst->meta_backup);
        cudaMalloc(&sst->meta_backup, sst->require * sizeof(float));
        break;
    }
    case kind_poisson: {
        cudaFree(sst->ui_output);
        break;
    }
    case kind_multinomial: {
        cudaFree(sst->ui_output);
        cudaFree(sst->meta_backup);
        cudaMalloc(&sst->meta_backup, sst->require * sizeof(float));
        break;
    }
    case kind_dirichlet: {
        cudaFree(sst->output + sst->require);
        substorage** ssts = (substorage**)sst->meta_working;
        for (unsigned int i = 0; i < sst->ui_output[0]; i++)
            destroy_substorage((void*)ssts[i]);

        free(sst->meta_working);
        free(sst->ui_output);
        break;
    }
    default:
        break;
    }
    sst->kind = kind_general;
}

extern "C" void destroy_substorage(void* sst) {
    substorage* _sst = (substorage*)sst;

    cudaFree(_sst->output);
    cudaFree(_sst->meta_backup);
    if (_sst->special_reserve_backup)
        cudaFree(_sst->special_reserve_backup);
    free(sst);
}

extern "C" void destroy(void* status) {
    _destroy_storage(((global_status*)status)->normal_storage);
    _destroy_storage(((global_status*)status)->uniform_storage);
    _destroy_rand_status(((global_status*)status)->rand_status);
    free(status);
}

extern "C" void* init(int seed) {
    global_status* status = (global_status*)malloc(sizeof(global_status));
    //_init_rand_status(seed, status);
    _init_rand_status_shared_only(seed, status);
    cudaDeviceSynchronize();
    _init_normal_storage(status);
    _init_uniform_storage(status);
    cudaStreamSynchronize(*status->normal_storage->stream);
    cudaStreamSynchronize(*status->uniform_storage->stream);
    return (void*)status;
}

extern "C" void* init_substorage(size_t require) {
    substorage* sst = (substorage*)malloc(sizeof(substorage));
    sst->require = require;
    sst->output_num = require;
    cudaMalloc(&sst->output, require * sizeof(float));
    cudaMalloc(&sst->meta_backup, require * sizeof(float));
    sst->ui_output = NULL;
    sst->meta_working = NULL;
    sst->special_reserve = NULL;
    sst->special_reserve_backup = NULL;
    sst->kind = kind_general;
    return (void*)sst;
}

extern "C" float* to_cpu(void* sst) {
    size_t nByte;
    substorage* _sst = (substorage*)sst;

    nByte = _sst->require * sizeof(float);
    float* h_array = (float*)malloc(nByte);
    cudaMemcpy(h_array, _sst->output, nByte, cudaMemcpyDeviceToHost);
    destroy_substorage(sst);
    return h_array;
}

extern "C" size_t to_cpu_num(void* sst) {
    return ((substorage*)sst)->require;
}

extern "C" void sync() {
    cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
    size_t nElm = 2048 * 8 * 4;
    unsigned int REP = 1;

    int seed = 5;

    size_t l1 = 0, l3 = 0, l5 = 0, l7 = 0;

    float multinomial_para[16];
    for (size_t i = 0; i < 16; i++)
        multinomial_para[i] = 1. / 16;

    void* sst, * status;
    status = init(seed);
    sst = init_substorage(nElm);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (size_t i = 0; i < REP; i++)
        //sample_gamma(2.0, 2.0, sst, status);
        //sample_uniform(sst, status);
        //sample_crt(0.5, 16, sst, status);
        //sample_multinomial(multinomial_para, 16, 64, sst, status);
        sample_exponential(1., sst, status);
    //sample_dirichlet(multinomial_para, 8, sst, status);
    cudaEventRecord(end);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Elapsed %f ms\n", time / REP);

    //h_result = (unsigned int*)malloc(nElm * 16 * sizeof(unsigned int));
    //cudaMemcpy(h_result, ((substorage*)sst)->ui_output, nElm * 16 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //size_t mean = 0.;
    //for (size_t i = 0; i < nElm * 16; i+=1)
    //    mean += h_result[i];
    //printf("%f", (float)mean / nElm);

    //h_result = (float*)malloc(nElm * 8 * sizeof(float));
    //cudaMemcpy(h_result, ((substorage*)sst)->output, nElm * 8 * sizeof(float), cudaMemcpyDeviceToHost);
    //float mean = 0.;
    //for (size_t i = 0; i < nElm; i += 1)
    //    mean += h_result[i];
    //printf("%f", mean / nElm);

    //h_result = (unsigned int*)malloc(nElm * sizeof(unsigned int));
    //cudaMemcpy(h_result, ((substorage*)sst)->ui_output, nElm * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //size_t mean = 0.;
    //for (size_t i = 0; i < nElm; i++)
    //    mean += h_result[i];
    //printf("%f", (float)mean / nElm);

    float *h_result = (float*)malloc(nElm * sizeof(float));

    h_result = to_cpu(sst);
    float tmp;
    float mean = 0;

    printf("Success:\n");
    for (size_t i = 0; i < nElm; i++) {
        tmp = h_result[i];
        mean += tmp;
        if (tmp > 0.1)
            l1++;
        if (tmp > 0.3)
            l3++;
        if (tmp > 0.5)
            l5++;
        if (tmp > 0.7)
            l7++;
    }
    printf(">0.1: %f >0.3: %f >0.5: %f >0.7: %f \n", (float)l1 / nElm, (float)l3 / nElm, (float)l5 / nElm, (float)l7 / nElm);


    //free(h_result);

    destroy(status);

    return 0;
}
