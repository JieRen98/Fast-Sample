#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <stdio.h>

#define nCapacity 8388608

// status
#define nStatus_all 8192
#define nThreads_Status_x 32
#define nThreads_Status_y 2
#define nThreads_Status (nThreads_Status_x * nThreads_Status_y)
#define nStream_Status_Init 16
#define nStatus_partition 4
#define nStatus_res_size (nStatus_partition - 2)
#define nStatus (nStatus_all / nStatus_partition)


#define nThreads_base_sample_x 32
#define nThreads_base_sample_y 4
#define nThreads_base_sample (nThreads_base_sample_x * nThreads_base_sample_y)

#define nUnroll 64

// const
#define euler_num 2.718281828459045
#define inv_euler_num 0.367879441171442
#define two_third 0.666666666666666
#define one_third 0.333333333333333

// poisson
#define nUnroll_poisson 8
#define nThreads_poisson_x 32
#define nThreads_poisson_y 32
#define nThreads_poisson (nThreads_poisson_x * nThreads_poisson_y)

// exp
#define nUnroll_exp 8
#define nThreads_exp_x 32
#define nThreads_exp_y 32
#define nThreads_exp (nThreads_exp_x * nThreads_exp_y)


// gamma
#define gamma_AR_worker 2
#define nThreads_gamma_x 32
#define nThreads_gamma_y 4
#define nThreads_gamma (nThreads_gamma_x * nThreads_gamma_y)

// crt
#define crt_AR_worker_level2 16
#define crt_AR_worker_level1 8
#define crt_AR_worker_level0 4
#define nThreads_crt_x 32
#define nThreads_crt_y 2
#define nThreads_crt (nThreads_crt_x * nThreads_crt_y)

// multinomial
#define multinomial_AR_worker_level2 32
#define multinomial_class_num 16
#define multinomial_rep_num 128
#define nThreads_multinomial 8

// dirichlet
#define nThreads_dirichlet 64

typedef struct {
    size_t loc;
    float* storage_0;
    float* storage_1;
    cudaStream_t* stream;
} storage;

enum distribution_kind
{
    kind_general, kind_gamma, kind_crt, kind_multinomial, kind_poisson, kind_dirichlet
};

typedef struct {
    size_t require;
    size_t output_num;
    float* output;
    unsigned int* ui_output;
    float* meta_working;
    float* meta_backup;
    float* special_reserve;
    float* special_reserve_backup;
    distribution_kind kind;
} substorage;

typedef struct {
    curandStateXORWOW_t* rand_status;
    storage* uniform_storage;
    storage* normal_storage;
} global_status;

void _sample_normal_with_stream(size_t, curandStateXORWOW_t*, float*, cudaStream_t*);
void _sample_uniform_with_stream(size_t, curandStateXORWOW_t*, float*, cudaStream_t*);
extern "C" void sample_exponential(float, void*, void*, cudaStream_t);
substorage* _init_gamma_substorage(size_t);
//void _fit_sst_for_gamma(substorage*);
//void _withdraw_sst_for_gamma(substorage*);
void _fit_crt_substorage(unsigned int, unsigned int, substorage*);
void _fit_multinomial_substorage(unsigned int, unsigned int, unsigned int, substorage*);
void _fit_poisson_substorage(substorage*);
void _fit_dirichlet_substorage(unsigned int, substorage*);
extern "C" void* init_substorage(size_t);
extern "C" void destroy_substorage(void*);
void _fit_general_substorage(substorage*);
