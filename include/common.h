//
// Created by Jie Ren (jieren9806@gmail.com) on 2021/6/11.
//

#pragma once

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

namespace FastSample {
    enum distribution_kind {
        kind_general,
        kind_gamma,
        kind_crt,
        kind_multinomial,
        kind_poisson,
        kind_dirichlet
    };

    template<typename T>
    class manager;

    template<typename T>
    class storage;
}
