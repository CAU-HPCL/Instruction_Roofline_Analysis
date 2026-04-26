#include "definitions.h"
#include "option.h"
#include <fstream>

using namespace std;
#ifndef LOGGER_H
#define LOGGER_H
class LOGGER
{
public:

    float density_of_A = 0;

    float avg_reordering_time = 0;
    float avg_csr_spmm_time = 0;
    float avg_bellpack_spmm_time = 0;
    float avg_total_spmm_time = 0;
    float avg_density_of_tiles = 0;

    // double avg_reordering_time = 0;
    // double avg_csr_spmm_time = 0;
    // double avg_bellpack_spmm_time = 0;
    // double avg_total_spmm_time = 0;
    // double avg_density_of_tiles = 0;

    float avg_format_conversion_time = 0;

    // float avg_modified_bellpack_spmm_time = 0;
    // float avg_modified_csr_spmm_time = 0;
    // float avg_modified_total_spmm_time = 0;

    // float avg_modified_bellpack_spmm_time_1 = 0;
    // float avg_modified_csr_spmm_time_1 = 0;
    // float avg_modified_total_spmm_time_1 = 0;

    // float avg_modified_bellpack_spmm_time_2 = 0;
    // float avg_modified_csr_spmm_time_2 = 0;
    // float avg_modified_total_spmm_time_2 = 0;

    // float avg_modified_bellpack_spmm_time_3 = 0;
    // float avg_modified_csr_spmm_time_3 = 0;
    // float avg_modified_total_spmm_time_3 = 0;

    // float avg_modified_bellpack_spmm_time_4 = 0;
    // float avg_modified_csr_spmm_time_4 = 0;
    // float avg_modified_total_spmm_time_4 = 0;

    // float avg_modified_bellpack_spmm_time_5 = 0;
    // float avg_modified_csr_spmm_time_5 = 0;
    // float avg_modified_total_spmm_time_5 = 0;

    // float avg_modified_bellpack_spmm_time_6 = 0;
    // float avg_modified_csr_spmm_time_6 = 0;
    // float avg_modified_total_spmm_time_6 = 0;

    // float avg_modified_bellpack_spmm_time_7 = 0;
    // float avg_modified_csr_spmm_time_7 = 0;
    // float avg_modified_total_spmm_time_7 = 0;

    // float avg_modified_bellpack_spmm_time_8 = 0;
    // float avg_modified_csr_spmm_time_8 = 0;
    // float avg_modified_total_spmm_time_8 = 0;

    // float avg_modified_bellpack_spmm_time_9 = 0;
    // float avg_modified_csr_spmm_time_9 = 0;
    // float avg_modified_total_spmm_time_9 = 0;

    // float avg_modified_bellpack_spmm_time_10 = 0;
    // float avg_modified_csr_spmm_time_10 = 0;
    // float avg_modified_total_spmm_time_10 = 0;

    // float avg_modified_bellpack_spmm_time_11 = 0;
    // float avg_modified_csr_spmm_time_11 = 0;
    // float avg_modified_total_spmm_time_11 = 0;

    // float avg_cusparse_spmm_time = 0;
    // float avg_cublas_spmm_time = 0;

    // float bsa_error = 0;
    // float cusparse_error = 0;
    // float cublas_error = 0;

    // float modified_1_error = 0;
    // float modified_2_error = 0;
    // float modified_3_error = 0;
    // float modified_4_error = 0;
    // float modified_5_error = 0;
    // float modified_6_error = 0;
    // float modified_7_error = 0;
    // float modified_8_error = 0;
    // float modified_9_error = 0;
    // float modified_10_error = 0;
    // float modified_11_error = 0;

    float alpha = 0;
    float delta = 0;

    intT num_tiles = 0;
    intT nnz_in_bellpack = 0;
    intT nnz_in_csr = 0;
    intT cluster_cnt = 0;
    intT n_cols = 0;
    intT rows = 0, cols = 0;
    intT block_size = 0;
    intT method = 0;
    intT spmm = 0;

    int repetitions = 0;
    int M = 0, N = 0, K = 0;

    float epsilon = 0;
    bool valid = false;
    float sparsity_of_tiles = 0;
    
    string infile;
    string outfile;

    float BSA_cusparse_CSR = 0;
    float BSA_cusparse_BELL = 0;
    float BSA_cusparse_TOTAL = 0;
    float BSA_cusparse_ERR = 0;

    float BSA_IK_32_A_CSR = 0;
    float BSA_IK_32_A_BELL = 0;
    float BSA_IK_32_A_TOTAL = 0;
    float BSA_IK_32_A_ERR = 0;

    float BSA_IK_64_A_CSR = 0;
    float BSA_IK_64_A_BELL = 0;
    float BSA_IK_64_A_TOTAL = 0;
    float BSA_IK_64_A_ERR = 0;

    float BSA_IK_128_A_CSR = 0;
    float BSA_IK_128_A_BELL = 0;
    float BSA_IK_128_A_TOTAL = 0;
    float BSA_IK_128_A_ERR = 0;

    float BSA_IK_32_A_XOR_CSR = 0;
    float BSA_IK_32_A_XOR_BELL = 0;
    float BSA_IK_32_A_XOR_TOTAL = 0;
    float BSA_IK_32_A_XOR_ERR = 0;

    float BSA_IK_64_A_XOR_CSR = 0;
    float BSA_IK_64_A_XOR_BELL = 0;
    float BSA_IK_64_A_XOR_TOTAL = 0;
    float BSA_IK_64_A_XOR_ERR = 0;

    float BSA_IK_128_A_XOR_CSR = 0; 
    float BSA_IK_128_A_XOR_BELL = 0;
    float BSA_IK_128_A_XOR_TOTAL = 0;
    float BSA_IK_128_A_XOR_ERR = 0;

    float BSA_IK_32_B_CSR = 0;
    float BSA_IK_32_B_BELL = 0;
    float BSA_IK_32_B_TOTAL = 0;
    float BSA_IK_32_B_ERR = 0;

    float BSA_IK_64_B_CSR = 0;
    float BSA_IK_64_B_BELL = 0;
    float BSA_IK_64_B_TOTAL = 0;
    float BSA_IK_64_B_ERR = 0;

    float BSA_IK_128_B_CSR = 0;
    float BSA_IK_128_B_BELL = 0;
    float BSA_IK_128_B_TOTAL = 0;
    float BSA_IK_128_B_ERR = 0;

    float BSA_IK_32_B_XOR_CSR = 0;
    float BSA_IK_32_B_XOR_BELL = 0;
    float BSA_IK_32_B_XOR_TOTAL = 0;
    float BSA_IK_32_B_XOR_ERR = 0;
    
    float BSA_IK_64_B_XOR_CSR = 0;
    float BSA_IK_64_B_XOR_BELL = 0;
    float BSA_IK_64_B_XOR_TOTAL = 0;
    float BSA_IK_64_B_XOR_ERR = 0;

    float BSA_IK_128_B_XOR_CSR = 0;
    float BSA_IK_128_B_XOR_BELL = 0;
    float BSA_IK_128_B_XOR_TOTAL = 0;
    float BSA_IK_128_B_XOR_ERR = 0;



    LOGGER(Option option)
    {
        infile = option.input_filename;
        outfile = option.output_filename;
        n_cols = option.n_cols;
        method = option.method;
        spmm = option.spmm;

        valid = option.valid;

        alpha = option.alpha;
        delta = option.delta;
    }

    void save_logfile()
    {
        std::ofstream fout;
        fout.open(outfile, std::ios_base::app);
        
        fout << infile << ",";
        fout << repetitions << ",";
        fout << epsilon << ",";

        fout << num_tiles << ",";
        fout << nnz_in_bellpack << ",";
        fout << nnz_in_csr << ",";
        fout << block_size << ",";

        fout << alpha << ",";
        fout << delta << ",";
        
        fout << M << ",";
        fout << N << ",";
        fout << K << ",";
        fout << density_of_A << ",";
        fout << sparsity_of_tiles << ",";

        fout << avg_reordering_time << ",";
        fout << avg_format_conversion_time << ",";

        fout << BSA_cusparse_CSR << ",";
        fout << BSA_cusparse_BELL << ",";
        fout << BSA_cusparse_TOTAL << ",";
        
        fout << BSA_IK_32_A_CSR << ",";
        fout << BSA_IK_32_A_BELL << ",";
        fout << BSA_IK_32_A_TOTAL << ",";

        fout << BSA_IK_64_A_CSR << ",";
        fout << BSA_IK_64_A_BELL << ",";
        fout << BSA_IK_64_A_TOTAL << ",";

        fout << BSA_IK_128_A_CSR << ",";
        fout << BSA_IK_128_A_BELL << ",";
        fout << BSA_IK_128_A_TOTAL << ",";

        fout << BSA_IK_32_A_XOR_CSR << ",";
        fout << BSA_IK_32_A_XOR_BELL << ",";
        fout << BSA_IK_32_A_XOR_TOTAL << ",";

        fout << BSA_IK_64_A_XOR_CSR << ",";
        fout << BSA_IK_64_A_XOR_BELL << ",";
        fout << BSA_IK_64_A_XOR_TOTAL << ",";

        fout << BSA_IK_128_A_XOR_CSR << ",";
        fout << BSA_IK_128_A_XOR_BELL << ",";
        fout << BSA_IK_128_A_XOR_TOTAL << ",";

        fout << BSA_IK_32_B_CSR << ",";
        fout << BSA_IK_32_B_BELL << ",";
        fout << BSA_IK_32_B_TOTAL << ",";   

        fout << BSA_IK_64_B_CSR << ",";
        fout << BSA_IK_64_B_BELL << ",";
        fout << BSA_IK_64_B_TOTAL << ",";

        fout << BSA_IK_128_B_CSR << ",";
        fout << BSA_IK_128_B_BELL << ",";
        fout << BSA_IK_128_B_TOTAL << ",";

        fout << BSA_IK_32_B_XOR_CSR << ",";
        fout << BSA_IK_32_B_XOR_BELL << ",";
        fout << BSA_IK_32_B_XOR_TOTAL << ",";

        fout << BSA_IK_64_B_XOR_CSR << ",";
        fout << BSA_IK_64_B_XOR_BELL << ",";
        fout << BSA_IK_64_B_XOR_TOTAL << ",";

        fout << BSA_IK_128_B_XOR_CSR << ",";
        fout << BSA_IK_128_B_XOR_BELL << ",";
        fout << BSA_IK_128_B_XOR_TOTAL << ",";

        fout << BSA_cusparse_ERR << ",";
        fout << BSA_IK_32_A_ERR << ",";
        fout << BSA_IK_64_A_ERR << ",";
        fout << BSA_IK_128_A_ERR << ",";
        fout << BSA_IK_32_A_XOR_ERR << ",";
        fout << BSA_IK_64_A_XOR_ERR << ",";
        fout << BSA_IK_128_A_XOR_ERR << ",";
        fout << BSA_IK_32_B_ERR << ",";
        fout << BSA_IK_64_B_ERR << ",";
        fout << BSA_IK_128_B_ERR << ",";
        fout << BSA_IK_32_B_XOR_ERR << ",";
        fout << BSA_IK_64_B_XOR_ERR << ",";
        fout << BSA_IK_128_B_XOR_ERR << endl;   

    
    }
};

#endif





// #include "definitions.h"
// #include "option.h"
// #include <fstream>

// using namespace std;
// #ifndef LOGGER_H
// #define LOGGER_H
// class LOGGER
// {
// public:
//     float avg_reordering_time = 0;
//     float avg_csr_spmm_time = 0;
//     float avg_bellpack_spmm_time = 0;
//     float avg_total_spmm_time = 0;
//     float avg_density_of_tiles = 0;
//     // double avg_reordering_time = 0;
//     // double avg_csr_spmm_time = 0;
//     // double avg_bellpack_spmm_time = 0;
//     // double avg_total_spmm_time = 0;
//     // double avg_density_of_tiles = 0;


//     float alpha = 0;
//     float delta = 0;

//     intT num_tiles = 0;
//     intT nnz_in_bellpack = 0;
//     intT nnz_in_csr = 0;
//     intT cluster_cnt = 0;
//     intT n_cols = 0;
//     intT rows = 0, cols = 0;
//     intT block_size = 0;
//     intT method = 0;
//     intT spmm = 0;

//     string infile;
//     string outfile;

//     LOGGER(Option option)
//     {
//         infile = option.input_filename;
//         outfile = option.output_filename;
//         n_cols = option.n_cols;
//         method = option.method;
//         spmm = option.spmm;

//         avg_reordering_time = 0;
//         avg_csr_spmm_time = 0;
//         avg_bellpack_spmm_time = 0;
//         avg_total_spmm_time = 0;
//         avg_density_of_tiles = 0;

//         alpha = option.alpha;
//         delta = option.delta;

//         num_tiles = 0;
//         nnz_in_bellpack = 0;
//         nnz_in_csr = 0;
//         cluster_cnt = 0;
//         rows = 0, cols = 0;
//         block_size = 0;
//     }

//     void save_logfile()
//     {
//         std::ofstream fout;
//         fout.open(outfile, std::ios_base::app);
//         // string header = "matrix,avg_reordering_time,avg_csr_spmm_time,avg_bellpack_spmm_time,avg_total_time,avg_density_of_tiles,num_tiles,nnz_in_bellpack,nnz_in_csr,cluster_cnt,n_cols,rows,cols,block_size,method";
//         // fout << header << endl;
//         fout << infile << ",";
//         fout << avg_reordering_time << ",";
//         fout << avg_csr_spmm_time << ",";
//         fout << avg_bellpack_spmm_time << ",";
//         fout << avg_total_spmm_time << ",";
//         fout << avg_density_of_tiles << ",";

//         fout << alpha << ",";
//         fout << delta << ",";

//         fout << num_tiles << ",";
//         fout << nnz_in_bellpack << ",";
//         fout << nnz_in_csr << ",";
//         fout << cluster_cnt << ",";
//         fout << n_cols << ",";
//         fout << rows << ",";
//         fout << cols << ",";
//         fout << block_size << ",";
//         fout << method << ",";
//         fout << spmm << endl;
//     }
// };

// #endif