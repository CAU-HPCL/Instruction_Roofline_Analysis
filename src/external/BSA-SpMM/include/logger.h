
// Modified from original BSA-SpMM (Euro-PAR 2024)
// Original: https://doi.org/10.5281/zenodo.11579181
// Changes: Modified log fields



#include "definitions.h"
#include "option.h"
#include <fstream>

using namespace std;
#ifndef LOGGER_H
#define LOGGER_H
class LOGGER
{
public:
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

    string infile;
    string outfile;

    LOGGER(Option option)
    {
        infile = option.input_filename;
        outfile = option.output_filename;
        n_cols = option.n_cols;
        method = option.method;
        spmm = option.spmm;

        avg_reordering_time = 0;
        avg_csr_spmm_time = 0;
        avg_bellpack_spmm_time = 0;
        avg_total_spmm_time = 0;
        avg_density_of_tiles = 0;

        alpha = option.alpha;
        delta = option.delta;

        num_tiles = 0;
        nnz_in_bellpack = 0;
        nnz_in_csr = 0;
        cluster_cnt = 0;
        rows = 0, cols = 0;
        block_size = 0;
    }

    void save_logfile()
    {
        std::ofstream fout;
        fout.open(outfile, std::ios_base::app);
        // string header = "matrix,avg_reordering_time,avg_csr_spmm_time,avg_bellpack_spmm_time,avg_total_time,avg_density_of_tiles,num_tiles,nnz_in_bellpack,nnz_in_csr,cluster_cnt,n_cols,rows,cols,block_size,method";
        // fout << header << endl;
        fout << infile << ",";
        fout << avg_reordering_time << ",";
        fout << avg_csr_spmm_time << ",";
        fout << avg_bellpack_spmm_time << ",";
        fout << avg_total_spmm_time << ",";
        fout << avg_density_of_tiles << ",";

        fout << alpha << ",";
        fout << delta << ",";

        fout << num_tiles << ",";
        fout << nnz_in_bellpack << ",";
        fout << nnz_in_csr << ",";
        fout << cluster_cnt << ",";
        fout << n_cols << ",";
        fout << rows << ",";
        fout << cols << ",";
        fout << block_size << ",";
        fout << method << ",";
        fout << spmm << endl;
    }
};

#endif