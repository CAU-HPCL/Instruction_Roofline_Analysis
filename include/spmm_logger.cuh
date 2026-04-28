
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
        // 파일이 없거나 비어있으면 헤더 추가
        std::ifstream check(outfile);
        bool need_header = !check.good() || check.peek() == std::ifstream::traits_type::eof();
        check.close();

        std::ofstream fout;
        fout.open(outfile, std::ios_base::app);

        if (need_header) {
            fout << "infile,repetitions,N,M,K,NNZ,density,sparsity,MU,MAX,STD_NNZ,MAX_MU,AVE_BW,STD_BW,"
                << "cusparse_time,ginkgo_time,kokkos_time,"
                << "cusparse_error,ginkgo_error,kokkos_error,"
                << "cusparse_result,ginkgo_result,kokkos_result,winner\n";
        }

        float sparsity = 1 - density;

        fout << infile << ","
            << repetitions << ","
            << N << ","
            << M << ","
            << K << ","
            << NNZ << ","
            << density << ","
            << sparsity << ","
            << MU << ","
            << MAX << ","
            << STD_NNZ << ","
            << MAX_MU << ","
            << AVE_BW << ","
            << STD_BW << ","
            << cusparse_time << ","
            << ginkgo_time << ","
            << kokkos_time << ","
            << cusparse_error << ","
            << ginkgo_error << ","
            << kokkos_error << ","
            << (cusparse_result == RESULTS::SUCCESS ? "success" : "failure") << ","
            << (ginkgo_result   == RESULTS::SUCCESS ? "success" : "failure") << ","
            << (kokkos_result   == RESULTS::SUCCESS ? "success" : "failure") << ","
            << (winner == METHODS::CUSPARSE ? "cusparse" :
                winner == METHODS::GINKGO   ? "ginkgo"   : "kokkos") << "\n";
    }




};

