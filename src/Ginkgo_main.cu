


#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif


#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
// #include <cuda_fp16.hpp>
// ---------- Ginkgo ----------
#include <ginkgo/ginkgo.hpp>

// ---------- Kokkos ----------
// #include <Kokkos_Core.hpp>
// #include <KokkosSparse_CrsMatrix.hpp>
// #include <KokkosSparse_spmv.hpp>

// ---------- BSA ----------
#include "bsa_spmm.cuh"
#include "option.h"
#include "logger.h"
#include "utilities.h"
#include "matrices.h"
#include "spmm.h"
#include "reorder.h"
#include "validate.h"

// ---------- etc ----------
#include "spmm_logger.cuh"
#include "nvtx_helper.cuh"

#define WARM_UP 0
#define ITERATIONS 0

using namespace std;

#define cuSPARSE_COL palette::blue
#define GINKGO_COL palette::orange
#define KOKKOS_COL palette::green

float run_cuSparse(const CSR &lhs, const ARR &rhs, ARR &result);
float run_ginkgo(const CSR &lhs, const ARR &rhs, ARR &result);
float run_kokkos(const CSR &lhs, const ARR &rhs, ARR &result);

float validate_results_with_host(const ARR &result, const CSR &lhs, const ARR &rhs);
double gflops(float ms, long long nnz, int n);

int main(int argc, char *argv[])
{
    Option option = Option(argc, argv);

    option.n_cols = 128;

    option.input_filename = "/workspace/TC_BELL/TC-BELL/dataset/DLMC/dlmc/rn50/magnitude_pruning/0.7/bottleneck_1_block_group_projection_block_group4.smtx";
    // option.input_filename = "/workspace/TC_BELL/TC-BELL/dataset/DLMC/dlmc/rn50/random_pruning/0.7/bottleneck_1_block_group_projection_block_group3.smtx";
    // option.input_filename = "/workspace/TC_BELL/TC-BELL/dataset/DLMC/dlmc/rn50/random_pruning/0.7/bottleneck_1_block_group_projection_block_group2.smtx";

    option.output_filename = "/workspace/ICTC/tmp/Ginkgo_results.csv";


    option.compress_rows = false;
    option.zero_padding = true;
    option.pattern_only = false;

    CSR lhs = CSR(option);
    SpMM_LOGGER logger = SpMM_LOGGER(option);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////[]


    random_device rd;
    mt19937 e2(rd());
    uniform_real_distribution<> dist(0, 1);

    lhs.values = new DataT[lhs.total_nonzeros];

    for(int i =0; i< lhs.total_nonzeros; i++)
    {
        lhs.values[i] = dist(e2);
    } 
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////[]


    ARR rhs = ARR(lhs.original_cols, lhs.cols, option.n_cols, true);
    rhs.fill_random(option.zero_padding);
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////[]


    ARR cusparse_result = ARR(lhs.original_rows, lhs.rows, option.n_cols, false);
    ARR Ginkgo_result = ARR(lhs.original_rows, lhs.rows, option.n_cols, false);
    ARR Kokkos_kernel_result = ARR(lhs.original_rows, lhs.rows, option.n_cols, false);
    

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////[]


    // logger.cusparse_time = run_cuSparse(lhs, rhs, cusparse_result);
    // logger.cusparse_error = validate_results_with_host(cusparse_result, lhs, rhs);
    // if(logger.cusparse_error > 0) logger.cusparse_result = RESULTS::FAILURE;
    // else logger.cusparse_result = RESULTS::SUCCESS;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////[]



    logger.ginkgo_time = run_ginkgo(lhs, rhs, Ginkgo_result);
    logger.ginkgo_error = validate_results_with_host(Ginkgo_result, lhs, rhs);
    if(logger.ginkgo_error > 0) logger.ginkgo_result = RESULTS::FAILURE;
    else logger.ginkgo_result = RESULTS::SUCCESS;



    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////[]


    // Kokkos::initialize(argc, argv);
    // {
    //     Kokkos::print_configuration(std::cout);
    //     std::cout << "DefaultExecutionSpace = " << Kokkos::DefaultExecutionSpace::name() << "\n";

    //     Kokkos::Profiling::pushRegion("Kokkos SpMM");
    //     logger.kokkos_time = run_kokkos(lhs, rhs, Kokkos_kernel_result);
    //     Kokkos::Profiling::popRegion();
    //     Kokkos::fence();
    //     cudaDeviceSynchronize();

    //     logger.kokkos_error = validate_results_with_host(Kokkos_kernel_result, lhs, rhs);
    //     if(logger.kokkos_error > 0) logger.kokkos_result = RESULTS::FAILURE;
    //     else logger.kokkos_result = RESULTS::SUCCESS;

    // }
    // Kokkos::finalize();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////[]

    logger.infile = option.input_filename;
    logger.outfile = option.output_filename;
    logger.repetitions = ITERATIONS;
    logger.M = lhs.rows;
    logger.N = rhs.cols;
    logger.K = lhs.cols;
    logger.NNZ = lhs.total_nonzeros;
    logger.density = static_cast<float>(lhs.total_nonzeros) / (lhs.rows * lhs.cols);

    logger.MU = (float)logger.NNZ / logger.M;
    
    int max_nnz = 0;
    for(int i =0;i < lhs.rows; i++)
    {
        if(lhs.rowptr[i+1] - lhs.rowptr[i] > max_nnz)
            max_nnz = lhs.rowptr[i+1] - lhs.rowptr[i];
    }
    logger.MAX = max_nnz;

    float diff = 0.0f;
    for(int i =0; i< lhs.rows; i++) diff += (lhs.rowptr[i+1] - lhs.rowptr[i] - logger.MU) * (lhs.rowptr[i+1] - lhs.rowptr[i] - logger.MU);
    float avg_diff = diff / lhs.rows;
    logger.STD_NNZ = sqrt(avg_diff);

    logger.MAX_MU = logger.MAX - logger.MU;

    int distance = 0;
    for(int i =0; i< lhs.rows; i++)
    {
        distance += abs(lhs.colidx[lhs.rowptr[i+1] - 1] - lhs.colidx[lhs.rowptr[i]]);
    }

    logger.AVE_BW = distance / lhs.rows;
 
    float diff_bw = 0.0f;
    for(int i =0; i< lhs.rows; i++)
    {
        diff_bw += (abs(lhs.colidx[lhs.rowptr[i+1] - 1] - lhs.colidx[lhs.rowptr[i]]) - logger.AVE_BW) * (abs(lhs.colidx[lhs.rowptr[i+1] - 1] - lhs.colidx[lhs.rowptr[i]]) - logger.AVE_BW);
    }
    float avg_diff_bw = diff_bw / lhs.rows;
    logger.STD_BW = sqrt(avg_diff_bw);

    float winner_time = logger.cusparse_time;
    if(logger.ginkgo_time < winner_time)
    {
        winner_time = logger.ginkgo_time;
        logger.winner = METHODS::GINKGO;
    }
    if(logger.kokkos_time < winner_time)
    {
        winner_time = logger.kokkos_time;
        logger.winner = METHODS::KOKKOS;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////[]


    cout << endl << "===========================" << endl;
    cout << "Input File: " << option.input_filename << endl;
    cout << "Output File: " << option.output_filename << endl;
    cout << "Repetitions: " << ITERATIONS << endl;
    cout << "M, N, K: " << lhs.rows << ", " << rhs.cols << ", " << lhs.cols << endl;
    cout << "Density: " << logger.density << endl;
    cout << "NNZ: " << lhs.total_nonzeros << endl;
    cout << "MU: " << logger.MU << endl;
    cout << "MAX: " << logger.MAX << endl;
    cout << "STD_NNZ: " << logger.STD_NNZ << endl;
    cout << "MAX_MU: " << logger.MAX_MU << endl;
    cout << "AVE_BW: " << logger.AVE_BW << endl;
    cout << "STD_BW: " << logger.STD_BW << endl;

    cout << endl << "===========================" << endl;
    cout << "cuSPARSE Time: " << logger.cusparse_time << " seconds" << endl;
    cout << "cuSPARSE Error: " << logger.cusparse_error << endl;
    cout << "cuSPARSE Result: " << (logger.cusparse_result == RESULTS::SUCCESS ? "SUCCESS" : "FAILURE") << endl;
    cout << endl << "===========================" << endl;
    cout << "Ginkgo Time: " << logger.ginkgo_time << " seconds" << endl;
    cout << "Ginkgo Error: " << logger.ginkgo_error << endl;
    cout << "Ginkgo Result: " << (logger.ginkgo_result == RESULTS::SUCCESS ? "SUCCESS" : "FAILURE") << endl;
    cout << endl << "===========================" << endl;
    cout << "Kokkos Time: " << logger.kokkos_time << " seconds" << endl;
    cout << "Kokkos Error: " << logger.kokkos_error << endl;
    cout << "Kokkos Result: " << (logger.kokkos_result == RESULTS::SUCCESS ? "SUCCESS" : "FAILURE") << endl;
    cout << endl << "===========================" << endl;
    cout << "Winner: " << (logger.winner == METHODS::CUSPARSE ? "cuSPARSE" :
                      (logger.winner == METHODS::GINKGO ? "Ginkgo" : "Kokkos")) << endl;
    cout << "Winner Time: " << winner_time << " seconds" << endl;
    cout << "===========================" << endl;  

    if (option.output_filename.length())
        logger.save_logfile();
}


float run_cuSparse(const CSR &lhs, const ARR &rhs, ARR &result)
{   
    int nnz = lhs.total_nonzeros;
    int m = lhs.rows;
    int n = rhs.cols;
    int k = lhs.cols;

    int *d_rowptr=nullptr, *d_colidx=nullptr;
    float *d_vals=nullptr, *d_B=nullptr, *d_C=nullptr;
    float *d_C_dummy=nullptr;

    cudaMalloc(&d_rowptr, (size_t)(m+1)*sizeof(int));
    cudaMalloc(&d_colidx, (size_t)nnz*sizeof(int));
    cudaMalloc(&d_vals,   (size_t)nnz*sizeof(float));
    cudaMalloc(&d_B,      (size_t)k*n*sizeof(float));
    cudaMalloc(&d_C,      (size_t)m*n*sizeof(float));
    cudaMalloc(&d_C_dummy, (size_t)m*n*sizeof(float));

    cudaMemcpy(d_rowptr, lhs.rowptr, (size_t)(m+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colidx, lhs.colidx, (size_t)nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals,   lhs.values, (size_t)nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,      rhs.mat, (size_t)k*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, (size_t)m*n*sizeof(float));
    cudaMemset(d_C_dummy, 0, (size_t)m*n*sizeof(float));

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseSpMatDescr_t A;
    cusparseCreateCsr(  &A, m, k, nnz,
                        d_rowptr, d_colidx, d_vals,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnMatDescr_t B, C, dummy_C;
    cusparseCreateDnMat(&B, k, n, n, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&C, m, n, n, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&dummy_C, m, n, n, d_C_dummy, CUDA_R_32F, CUSPARSE_ORDER_ROW);


    float alpha = 1.0f, beta = 0.0f;
    size_t ws_size=0; void* d_ws=nullptr;
    
    cusparseSpMM_bufferSize(handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, A, B, &beta, C, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &ws_size);
    
    if (ws_size>0) cudaMalloc(&d_ws, ws_size);

    
        cusparseSpMM(   handle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, A, B, &beta, C, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_ws);

    

    cudaDeviceSynchronize();

    cudaMemcpy(result.mat, d_C, (size_t)m*n*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i =0; i< WARM_UP; i++)
    {
        cusparseSpMM(   handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, A, B, &beta, dummy_C, CUDA_R_32F,
                        CUSPARSE_SPMM_ALG_DEFAULT, d_ws);
    }

    cudaDeviceSynchronize();

    float time = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i =0; i< ITERATIONS; i++)
    {
        cusparseSpMM(   handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, A, B, &beta, dummy_C, CUDA_R_32F,
                        CUSPARSE_SPMM_ALG_DEFAULT, d_ws);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_rowptr);
    cudaFree(d_colidx);
    cudaFree(d_vals);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_dummy);
    cusparseDestroySpMat(A);
    cusparseDestroyDnMat(B);
    cusparseDestroyDnMat(C);
    cusparseDestroy(handle);    

    return time / ITERATIONS; // Return average time per iteration

}


float run_ginkgo(const CSR &lhs, const ARR &rhs, ARR &result)
{   
    int nnz = lhs.total_nonzeros;
    int m = lhs.rows;
    int n = rhs.cols;
    int k = lhs.cols;

    auto omp = gko::OmpExecutor::create();
    auto exec = gko::CudaExecutor::create(0, omp);  // GPU 0

    auto A_h = gko::matrix::Csr<float,int>::create(omp, gko::dim<2>(m,k), (size_t)lhs.total_nonzeros);
    std::copy_n(lhs.rowptr,  m+1,         A_h->get_row_ptrs());
    std::copy_n(lhs.colidx,  lhs.total_nonzeros, A_h->get_col_idxs());
    std::copy_n(lhs.values,  lhs.total_nonzeros, A_h->get_values());
    // 2) Device로 복사
    auto A = gko::matrix::Csr<float,int>::create(exec);
    A->copy_from(A_h.get());

    auto B_h = gko::matrix::Dense<float>::create(omp, gko::dim<2>(k,n));
    std::copy_n(rhs.mat, (size_t)k*n, B_h->get_values());
    auto B   = gko::matrix::Dense<float>::create(exec);
    B->copy_from(B_h.get());

    auto C = gko::matrix::Dense<float>::create(exec, gko::dim<2>(m, n));
    C->fill(0.0f);

    auto C_dummy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(m, n));
    C_dummy->fill(0.0f);


    A->apply(B.get(), C.get());

    
    exec->synchronize();

    auto C_h = gko::matrix::Dense<float>::create(omp, gko::dim<2>(m,n));
    C_h->copy_from(C.get());
    std::copy_n(C_h->get_values(), (size_t)m*n, result.mat);
    
    for (int it=0; it< WARM_UP; ++it) A->apply(B.get(), C_dummy.get());
    exec->synchronize();


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int it=0; it< ITERATIONS; ++it) A->apply(B.get(), C_dummy.get());
    exec->synchronize();
    cudaEventRecord(stop);  
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return elapsed_time / ITERATIONS; // Return average time per iteration

}



// float run_kokkos(const CSR &lhs, const ARR &rhs, ARR &result)
// {   
//     int nnz = lhs.total_nonzeros;
//     int m = lhs.rows;
//     int n = rhs.cols;
//     int k = lhs.cols;

//     using Scalar=float; using Ordinal=int; using Offset=int;
//     using Exec = Kokkos::DefaultExecutionSpace;
//     using Dev  = Kokkos::Device<Exec, typename Exec::memory_space>;
//     using Crs  = KokkosSparse::CrsMatrix<Scalar, Ordinal, Dev, void, Offset>;

//     // CSR → device
//     Kokkos::View<Offset*,  Dev> rowptr("rowptr", m+1);
//     Kokkos::View<Ordinal*, Dev> colidx("colidx", (int)lhs.total_nonzeros);   
//     Kokkos::View<Scalar*,  Dev> vals  ("vals",   (int)lhs.total_nonzeros);

//     auto h_r = Kokkos::create_mirror_view(rowptr);
//     auto h_c = Kokkos::create_mirror_view(colidx);
//     auto h_v = Kokkos::create_mirror_view(vals);

//     for (int i=0;i<m+1;i++) h_r(i)=lhs.rowptr[i];
//     for (size_t i=0;i<lhs.total_nonzeros;i++){ h_c(i)=lhs.colidx[i]; h_v(i)=lhs.values[i]; }
//     Kokkos::deep_copy(rowptr, h_r); Kokkos::deep_copy(colidx, h_c); Kokkos::deep_copy(vals, h_v);

//     Crs A("A", m, k, lhs.total_nonzeros, vals, rowptr, colidx);

//     // rank-2 multivector (LayoutLeft 권장)
//     Kokkos::View<Scalar**, Kokkos::LayoutRight, Dev> X("X", k, n);
//     Kokkos::View<Scalar**, Kokkos::LayoutRight, Dev> Y("Y", m, n);
//     Kokkos::View<Scalar**, Kokkos::LayoutRight, Dev> Dummy_Y("Y", m, n);    


//     auto Xh = Kokkos::create_mirror_view(X);
    
//     for (int i=0;i<k;i++)
//         for (int j=0;j<n;j++)
//             Xh(i,j) = rhs.mat[i*n + j];
    
//     Kokkos::deep_copy(X, Xh);

//     Kokkos::deep_copy(Y, 0.0f);
//     Kokkos::deep_copy(Dummy_Y, 0.0f);


//     KokkosSparse::spmv("N", Scalar(1), A, X, Scalar(0), Y);
    

    
//     Kokkos::fence();

//     auto Yh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Y);
    
//     for (int i=0;i<m;i++)
//         for (int j=0;j<n;j++)
//             result.mat[(size_t)i*n + j] = Yh(i,j);

//     Kokkos::fence();

//     for (int it=0; it<WARM_UP; ++it) {
//         KokkosSparse::spmv("N", Scalar(1), A, X, Scalar(0), Dummy_Y);
//     }

//     Kokkos::fence();

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);
    
//     for(int i =0; i< ITERATIONS; i++)
//     {
//         KokkosSparse::spmv("N", Scalar(1), A, X, Scalar(0), Dummy_Y);
//     }
//     Kokkos::fence();
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     float elapsed_time;
//     cudaEventElapsedTime(&elapsed_time, start, stop);
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop); 

//     return elapsed_time / ITERATIONS; // Return average time per iteration


// }



float validate_results_with_host(const ARR &result, const CSR &lhs, const ARR &rhs)
{   
    int nnz = lhs.total_nonzeros;
    int m = lhs.rows;
    int n = rhs.cols;
    int k = lhs.cols;

    float *answer = new float[m*n];

    for(int r =0; r< m; r++)
    {
        for(int c =0; c< n; c++)
        {
            answer[r*n + c] = 0.0f;
            int start = lhs.rowptr[r];
            int end = lhs.rowptr[r+1];

            for(int k = start; k < end; k++)
            {
                int col = lhs.colidx[k];
                answer[r*n + c] += lhs.values[k] * rhs.mat[col*n + c];
            }
        }
    }

    for(int i =0; i< m; i++)
    {
        int nnz_row = lhs.rowptr[i+1] - lhs.rowptr[i];
        double epsilon = (1e-6f * nnz_row);

        for(int j =0; j<n; j++)
        {
            if (fabs(result.mat[i*n + j] - answer[i*n + j]) > epsilon)
            {
                cout << "Validation failed at (" << i << ", " << j << "): "
                     << "Expected: " << answer[i*n + j] 
                     << ", Got: " << result.mat[i*n + j] << endl;



                delete[] answer;
                return fabs(result.mat[i*n + j] - answer[i*n + j]);
            }
        }
    }

    cout << "Validation successful!" << endl;
    delete[] answer;
    return 0.0f;
}

double gflops(float ms, long long nnz, int n)
{
    double fl = 2.0 * (double)nnz * (double)n;
    return fl / (ms*1e-3) / 1e9;
}


