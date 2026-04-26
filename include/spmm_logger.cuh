




// ###########################BSA_SPMM###################################

#pragma once

#include "option.h"
#include <fstream>

using namespace std;


enum class RESULTS
{
    SUCCESS,
    FAILURE
};

enum class METHODS
{
    CUSPARSE,
    GINKGO,
    KOKKOS,
};

class SpMM_LOGGER
{
public:

    string infile;
    string outfile;

    bool valid = true;;
    int repetitions = 0;


    int M = 0, N = 0, K = 0;
    int NNZ = 0;
    float density = 0;
    float MU = 0;
    int MAX = 0;
    float STD_NNZ =0;
    float MAX_MU = 0;
    float AVE_BW = 0;
    float STD_BW = 0;

    float cusparse_time = 0;
    float ginkgo_time = 0;
    float kokkos_time = 0;
    
    float cusparse_error = 0;
    float ginkgo_error = 0;
    float kokkos_error = 0;

    RESULTS cusparse_result = RESULTS::SUCCESS;
    RESULTS ginkgo_result = RESULTS::SUCCESS;
    RESULTS kokkos_result = RESULTS::SUCCESS;

    METHODS winner = METHODS::CUSPARSE;
    
    SpMM_LOGGER(Option option)
    {
        ;
    }

    void save_logfile()
    {
        std::ofstream fout;
        fout.open(outfile, std::ios_base::app);

        // string header = "matrix,avg_reordering_time,avg_csr_spmm_time,avg_bellpack_spmm_time,avg_total_time,avg_density_of_tiles,num_tiles,nnz_in_bellpack,nnz_in_csr,cluster_cnt,n_cols,rows,cols,block_size,method";
        // fout << header << endl;
        
        float sparsity = 1 - density;

        fout << infile << ",";
        fout << repetitions << ",";

        fout << N << ",";

        fout << M << ",";
        fout << K << ",";

        fout << NNZ << ",";
        fout << density << ",";
        fout << sparsity << ",";
        fout << MU << ",";
        fout << MAX << ",";
        fout << STD_NNZ << ",";
        fout << MAX_MU << ",";
        fout << AVE_BW << ",";
        fout << STD_BW << ",";

        fout << cusparse_time << ",";
        fout << ginkgo_time << ",";
        fout << kokkos_time << ",";
        fout << cusparse_error << ",";
        fout << ginkgo_error << ",";
        fout << kokkos_error << ",";

        if(cusparse_result == RESULTS::SUCCESS)
            fout << "success,";
        else
            fout << "failure,";
        
        if(ginkgo_result == RESULTS::SUCCESS)
            fout << "success,";
        else
            fout << "failure,";

        if(kokkos_result == RESULTS::SUCCESS)
            fout << "success,";
        else
            fout << "failure,";

        switch (winner)
        {
        case METHODS::CUSPARSE:
            fout << "cusparse" << endl;
            break;
        case METHODS::GINKGO:
            fout << "ginkgo" << endl;
            break;
        case METHODS::KOKKOS:
            fout << "kokkos" << endl;
            break;
        default:
            fout << "unknown" << endl;
            break;
        }
    
    }
};
