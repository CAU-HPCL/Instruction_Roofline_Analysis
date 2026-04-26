#pragma once

#include "definitions.h"
#include "option.h"
#include <string.h>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <assert.h>
#include <algorithm>
#include <random>
#include "logger.h"

#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#ifndef MATRICES_H
#define MATRICES_H


    class CSR
    {
    public:

        int rows;
        int cols;
        int original_rows;
        int original_cols;
        int total_nonzeros;

        bool pattern_only;

        intT *rowptr;

        intT *colidx;
        DataT *values;

        void read_from_mtx(std::ifstream &fin, Option option, bool zero_base);
        void read_from_smtx(std::ifstream &fin, Option option, bool zero_base);

        CSR() = default;


        CSR(Option &option)
        {
            FileFormatType format = static_cast<FileFormatType>(option.input_format);
            std::ifstream fin;
            pattern_only = option.pattern_only;
            fin.open(option.input_filename);
            if (format == mtx)
            {
                read_from_mtx(fin, option, true);
                // read_from_mtx(fin, option, false);
            }
            else if (format == smtx)
            {
                read_from_smtx(fin, option, true);
            }
        }

        ~CSR()
        {
            delete[] rowptr;
            delete[] colidx;
            // if (not pattern_only)
            if( values != nullptr)
                delete[] values;
        }
    };

    class CSR_uint64_t
    {
    public:
        uint64_t rows;
        uint64_t cols;
        // for zero paading
        uint64_t original_rows;
        uint64_t original_cols;
        uint64_t total_nonzeros;
        bool pattern_only;

        uint64_t *rowptr;
        // intT *nzcount;
        uint64_t *colidx;
        DataT *values;

        void read_from_mtx_uint64_t(std::ifstream &fin, Option option, bool zero_base);
        // void read_from_smtx(std::ifstream &fin, Option option, bool zero_base);

        CSR_uint64_t() = default;


        CSR_uint64_t(Option &option)
        {
            FileFormatType format = static_cast<FileFormatType>(option.input_format);
            std::ifstream fin;
            pattern_only = option.pattern_only;
            fin.open(option.input_filename);
            if (format == mtx)
            {
                read_from_mtx_uint64_t(fin, option, true);
                // read_from_mtx(fin, option, false);
            }
            // else if (format == smtx)
            // {
            //     read_from_smtx(fin, option, true);
            // }
        }

        ~CSR_uint64_t()
        {
            delete[] rowptr;
            delete[] colidx;
            // if (not pattern_only)
            if( values != nullptr)
                delete[] values;
        }
    };

    class BSA_HYBRID
    {
    public:
        // uint64_t rows;
        // uint64_t cols;
        // // for zero paading
        // uint64_t original_rows;
        // uint64_t original_cols;

        int rows;
        int cols;
        int original_rows;
        int original_cols;


        bool pattern_only;

        // The result of reordering
        vector<uint64_t> row_permutation_uint64;
        vector<int> row_permutation;

        // BELL PACK
        // uint64_t bellpack_total_nonzeros = 0;
        int bellpack_total_nonzeros = 0;
        intT block_size;
        intT *ellColInd;
        int64_t *ellColInd_uint64_t;
        DataT *ellValues;
        DataT_H *h_ellValues;

        // uint64_t ell_rows;
        // uint64_t ell_cols;
        int ell_rows;
        int ell_cols;


        // CSR
        // uint64_t csr_total_nonzeros = 0;
        int csr_total_nonzeros = 0;
        intT *csr_rowptr;
        intT *csr_colidx;
        DataT *csr_values;
        DataT_H *csv_h_values;

        BSA_HYBRID(CSR &csr, LOGGER &logger, intT block_size_, float delta, vector<int> row_permutation_)
        {
            rows = csr.rows;
            cols = csr.cols;
            block_size = block_size_;

            original_rows = csr.original_rows;
            original_cols = csr.original_cols;
            pattern_only = csr.pattern_only;
            row_permutation = row_permutation_;

            assert(rows % block_size == 0);
            assert(cols % block_size == 0);

            ell_rows = rows / block_size;
            ell_cols = 0;
            vector<vector<int>> nnz(ell_rows, vector<intT>(cols / block_size, 0));
            vector<vector<int>> dense_tiles(ell_rows);
            intT offset = 0;

            logger.rows = original_rows;
            logger.cols = original_cols;
            logger.block_size = block_size;

    #pragma omp parallel for num_threads(12)
            for (int i = 0; i < rows; i++)
            {
                int r = row_permutation[i];
                intT start_pos = csr.rowptr[r];
                intT end_pos = csr.rowptr[r + 1];
                for (int nz = start_pos; nz < end_pos; nz++)
                {
                    intT col = csr.colidx[nz];
                    nnz[i / block_size][col / block_size]++;
                }
            }

            vector<vector<intT>> vec_csr_colidx(rows);
            vector<vector<DataT>> vec_csr_values(rows);

            for (int row_panel_id = 0; row_panel_id < nnz.size(); row_panel_id++)
            {
                for (int j = 0; j < nnz[row_panel_id].size(); j++)
                {
                    if (nnz[row_panel_id][j] > block_size * block_size * delta)
                    {
                        // BELL PACK
                        dense_tiles[row_panel_id].push_back(j);
                        logger.num_tiles++;
                        logger.avg_density_of_tiles += (float)nnz[row_panel_id][j] / (block_size * block_size);
                    }
                    else
                    {
                        if (nnz[row_panel_id][j] == 0)
                            continue;
                        // CSR
                        for (intT row = row_panel_id * block_size; row < (row_panel_id + 1) * block_size; row++)
                        {
                            intT row_id = row_permutation[row];
                            if (row_id >= original_rows)
                                continue;
                            intT start_pos = csr.rowptr[row_id];
                            intT end_pos = csr.rowptr[row_id + 1];
                            for (intT nz = start_pos; nz < end_pos; nz++)
                            {
                                if (csr.colidx[nz] / block_size != j)
                                    continue;

                                vec_csr_colidx[row].push_back(csr.colidx[nz]);
                                csr_total_nonzeros++;
                                if (pattern_only)
                                    vec_csr_values[row].push_back(1);
                                else
                                    vec_csr_values[row].push_back(csr.values[nz]);
                            }
                        }
                    }
                }
                ell_cols = max(ell_cols, (int)dense_tiles[row_panel_id].size());
            }

            // printf("ell_cols: %d\n", ell_cols);
            // printf("ell_rows: %d\n", ell_rows);
            // printf("block_size: %d\n", block_size);

            // size_t free_mem, total_mem;
            // cudaMemGetInfo(&free_mem, &total_mem);
            // printf("Needed memory for BELL PACK: %lu MB\n", (size_t)ell_rows * (size_t)ell_cols * (size_t)block_size * (size_t)block_size * sizeof(DataT_H) / (1024 * 1024));
            // printf("Free memory: %lu MB, Total memory: %lu MB\n", free_mem / (1024 * 1024), total_mem / (1024 * 1024));

            // std::fprintf(stdout,
            //     "Needed memory for BELL PACK: %.2f MB | Free: %.2f MB | Total: %.2f MB\n",
            //     (size_t)ell_rows * (size_t)ell_cols * (size_t)block_size * (size_t)block_size * sizeof(DataT_H) / (1024.0*1024.0),
            //     free_mem / (1024.0*1024.0),
            //     total_mem / (1024.0*1024.0));
            // std::fflush(stdout);

            // if( (size_t)ell_rows * (size_t)ell_cols * (size_t)block_size * (size_t)block_size * sizeof(DataT_H) > free_mem)
            // {
            //     std::fprintf(stderr,
            //     "Not enough GPU memory for BELL PACK (need=%.2f MB, free=%.2f MB) — switch to CSR only\n", 
            //     (size_t)ell_rows * (size_t)ell_cols * (size_t)block_size * (size_t)block_size * sizeof(DataT_H) / (1024.0*1024.0),
            //     free_mem / (1024.0*1024.0));
            //     std::fflush(stderr);
            //     std::exit(EXIT_FAILURE);
            // }

            ellColInd = new intT[(size_t)ell_rows * ell_cols];
            ellValues = new DataT[(size_t)ell_rows * (size_t)ell_cols * (size_t)block_size * block_size];
            h_ellValues = new DataT_H[(size_t)ell_rows * ell_cols * (size_t)block_size * block_size];

            memset(ellColInd, -1, (size_t)ell_rows * ell_cols * sizeof(intT));
            memset(ellValues, 0, (size_t)ell_rows * ell_cols * (size_t)block_size * block_size * sizeof(DataT));
            memset(h_ellValues, 0, (size_t)ell_rows * ell_cols * (size_t)block_size * block_size * sizeof(DataT_H));

            for (int i = 0; i < ell_rows; i++)
            {
                for (int j = 0; j < dense_tiles[i].size(); j++)
                {
                    ellColInd[i * ell_cols + j] = dense_tiles[i][j];
                }
            }
            for (int i = 0; i < rows; i++)
            {
                intT r = row_permutation[i];
                intT start_pos = csr.rowptr[r];
                intT end_pos = csr.rowptr[r + 1];
                for (int nz = start_pos; nz < end_pos; nz++)
                {
                    bellpack_total_nonzeros++;
                    int row_panel_id = i / block_size;
                    int original_col = csr.colidx[nz];
                    int inner_row = i % block_size;

                    auto it_ellblock_col = find(dense_tiles[row_panel_id].begin(), dense_tiles[row_panel_id].end(), (original_col / block_size));
                    if (it_ellblock_col == dense_tiles[row_panel_id].end())
                        continue;
                    int ellblock_col = it_ellblock_col - dense_tiles[row_panel_id].begin();
                    int inner_col = original_col % block_size;
                    int idx = (i * ell_cols * block_size) + (ellblock_col * block_size) + inner_col;
                    // uint64_t idx =  (uint64_t)row_panel_id * ell_cols * block_size * block_size +
                                    // (uint64_t)ellblock_col * block_size * block_size +
                                    // (uint64_t)inner_row    * block_size +
                                    // inner_col;

                    if (not pattern_only)
                    {
                        // ellValues[idx] = csr.values[(size_t)offset + nz];
                        // h_ellValues[idx] = __float2half(csr.values[(size_t)offset + nz]);
                        ellValues[idx] = csr.values[offset + nz];
                        h_ellValues[idx] = __float2half(csr.values[offset + nz]);
                    }
                    else
                    {
                        ellValues[idx] = 1;
                        h_ellValues[idx] = __float2half(1);
                    }
                }
            }
            csr_rowptr = new intT[rows + 1];
            csr_colidx = new intT[csr_total_nonzeros];
            csr_values = new DataT[csr_total_nonzeros];
            csr_rowptr[0] = 0;
            offset = 0;
            for (intT r = 0; r < rows; r++)
            {
                std::copy(vec_csr_colidx[r].begin(), vec_csr_colidx[r].end(), csr_colidx + offset);
                std::copy(vec_csr_values[r].begin(), vec_csr_values[r].end(), csr_values + offset);
                csr_rowptr[r + 1] = csr_rowptr[r] + (intT)vec_csr_colidx[r].size();
                offset += vec_csr_colidx[r].size();
            }

            logger.nnz_in_csr = csr_total_nonzeros;
            // logger.nnz_in_bellpack = bellpack_total_nonzeros;
            if (logger.num_tiles)
                logger.avg_density_of_tiles /= logger.num_tiles;

            
            bellpack_total_nonzeros = csr.total_nonzeros - csr_total_nonzeros;
            assert(csr_total_nonzeros + bellpack_total_nonzeros == csr.total_nonzeros);
        }

    //     BSA_HYBRID(CSR &csr, LOGGER &logger, intT block_size_, float delta, vector<intT> row_permutation_)
    //     {
    //         rows = csr.rows;
    //         cols = csr.cols;
    //         block_size = block_size_;

    //         original_rows = csr.original_rows;
    //         original_cols = csr.original_cols;
    //         pattern_only = csr.pattern_only;
    //         row_permutation = row_permutation_;

    //         assert(rows % block_size == 0);
    //         assert(cols % block_size == 0);

    //         ell_rows = rows / block_size;
    //         ell_cols = 0;
    //         vector<vector<intT>> nnz(ell_rows, vector<intT>(cols / block_size, 0));
    //         vector<vector<intT>> dense_tiles(ell_rows);
    //         intT offset = 0;

    //         logger.rows = original_rows;
    //         logger.cols = original_cols;
    //         logger.block_size = block_size;

    // #pragma omp parallel for num_threads(12)
    //         for (intT i = 0; i < rows; i++)
    //         {
    //             intT r = row_permutation[i];
    //             intT start_pos = csr.rowptr[r];
    //             intT end_pos = csr.rowptr[r + 1];
    //             for (intT nz = start_pos; nz < end_pos; nz++)
    //             {
    //                 intT col = csr.colidx[nz];
    //                 nnz[i / block_size][col / block_size]++;
    //             }
    //         }

    //         vector<vector<intT>> vec_csr_colidx(rows);
    //         vector<vector<DataT>> vec_csr_values(rows);

    //         for (intT row_panel_id = 0; row_panel_id < nnz.size(); row_panel_id++)
    //         {
    //             for (intT j = 0; j < nnz[row_panel_id].size(); j++)
    //             {
    //                 if (nnz[row_panel_id][j] > block_size * block_size * delta)
    //                 {
    //                     // BELL PACK
    //                     dense_tiles[row_panel_id].push_back(j);
    //                     logger.num_tiles++;
    //                     logger.avg_density_of_tiles += (float)nnz[row_panel_id][j] / (block_size * block_size);
    //                 }
    //                 else
    //                 {
    //                     if (nnz[row_panel_id][j] == 0)
    //                         continue;
    //                     // CSR
    //                     for (intT row = row_panel_id * block_size; row < (row_panel_id + 1) * block_size; row++)
    //                     {
    //                         intT row_id = row_permutation[row];
    //                         if (row_id >= original_rows)
    //                             continue;
    //                         intT start_pos = csr.rowptr[row_id];
    //                         intT end_pos = csr.rowptr[row_id + 1];
    //                         for (intT nz = start_pos; nz < end_pos; nz++)
    //                         {
    //                             if (csr.colidx[nz] / block_size != j)
    //                                 continue;

    //                             vec_csr_colidx[row].push_back(csr.colidx[nz]);
    //                             csr_total_nonzeros++;
    //                             if (pattern_only)
    //                                 vec_csr_values[row].push_back(1);
    //                             else
    //                                 vec_csr_values[row].push_back(csr.values[nz]);
    //                         }
    //                     }
    //                 }
    //             }
    //             ell_cols = max(ell_cols, (intT)dense_tiles[row_panel_id].size());
    //         }
    //         ellColInd = new intT[ell_rows * ell_cols];
    //         ellValues = new DataT[ell_rows * ell_cols * block_size * block_size];
    //         h_ellValues = new DataT_H[ell_rows * ell_cols * block_size * block_size];

    //         memset(ellColInd, -1, ell_rows * ell_cols * sizeof(intT));
    //         memset(ellValues, 0, ell_rows * ell_cols * block_size * block_size * sizeof(DataT));
    //         memset(h_ellValues, 0, ell_rows * ell_cols * block_size * block_size * sizeof(DataT_H));
    //         for (int i = 0; i < ell_rows; i++)
    //         {
    //             for (int j = 0; j < dense_tiles[i].size(); j++)
    //             {
    //                 ellColInd[i * ell_cols + j] = dense_tiles[i][j];
    //             }
    //         }
    //         for (int i = 0; i < rows; i++)
    //         {
    //             intT r = row_permutation[i];
    //             intT start_pos = csr.rowptr[r];
    //             intT end_pos = csr.rowptr[r + 1];
    //             for (int nz = start_pos; nz < end_pos; nz++)
    //             {
    //                 bellpack_total_nonzeros++;
    //                 intT row_panel_id = i / block_size;
    //                 intT original_col = csr.colidx[nz];

    //                 auto it_ellblock_col = find(dense_tiles[row_panel_id].begin(), dense_tiles[row_panel_id].end(), (original_col / block_size));
    //                 if (it_ellblock_col == dense_tiles[row_panel_id].end())
    //                     continue;
    //                 intT ellblock_col = it_ellblock_col - dense_tiles[row_panel_id].begin();
    //                 intT inner_col = original_col % block_size;
    //                 intT idx = (i * ell_cols * block_size) + (ellblock_col * block_size) + inner_col;
                    
    //                 if (not pattern_only)
    //                 {
    //                     ellValues[idx] = csr.values[offset + nz];
    //                     h_ellValues[idx] = __float2half(csr.values[offset + nz]);
    //                 }
    //                 else
    //                 {
    //                     ellValues[idx] = 1;
    //                     h_ellValues[idx] = __float2half(1);
    //                 }
    //             }
    //         }
    //         csr_rowptr = new intT[rows + 1];
    //         csr_colidx = new intT[csr_total_nonzeros];
    //         csr_values = new DataT[csr_total_nonzeros];
    //         csr_rowptr[0] = 0;
    //         offset = 0;
    //         for (intT r = 0; r < rows; r++)
    //         {
    //             std::copy(vec_csr_colidx[r].begin(), vec_csr_colidx[r].end(), csr_colidx + offset);
    //             std::copy(vec_csr_values[r].begin(), vec_csr_values[r].end(), csr_values + offset);
    //             csr_rowptr[r + 1] = csr_rowptr[r] + (intT)vec_csr_colidx[r].size();
    //             offset += vec_csr_colidx[r].size();
    //         }

    //         logger.nnz_in_csr = csr_total_nonzeros;


    //         bellpack_total_nonzeros = csr.total_nonzeros - csr_total_nonzeros;
    //         logger.nnz_in_bellpack = bellpack_total_nonzeros;

    //         printf("-----------------------------------------\n");
    //         printf("ell_cols: %d\n", ell_cols);
    //         printf("ell_rows: %d\n", ell_rows);
    //         printf("block_size: %d\n", block_size);
    //         printf("nnz in CSR: %d\n", csr_total_nonzeros);
    //         printf("nnz in BELL PACK: %d\n", bellpack_total_nonzeros);
    //         printf("-----------------------------------------\n");
    //         assert(csr_total_nonzeros + bellpack_total_nonzeros == csr.total_nonzeros);


            
    //         if (logger.num_tiles)
    //             logger.avg_density_of_tiles /= logger.num_tiles;
    //     }





    //     BSA_HYBRID(CSR &csr, LOGGER &logger, intT block_size_, float delta, vector<uint64_t> row_permutation_)
    //     {
    //         rows = csr.rows;
    //         cols = csr.cols;
    //         block_size = block_size_;

    //         original_rows = csr.original_rows;
    //         original_cols = csr.original_cols;
    //         pattern_only = csr.pattern_only;
    //         row_permutation_uint64 = row_permutation_;

    //         assert(rows % block_size == 0);
    //         assert(cols % block_size == 0);

    //         ell_rows = rows / block_size;
    //         ell_cols = 0;
    //         vector<vector<int>> nnz(ell_rows, vector<intT>(cols / block_size, 0));
    //         vector<vector<uint64_t>> dense_tiles(ell_rows);
    //         intT offset = 0;

    //         logger.rows = original_rows;
    //         logger.cols = original_cols;
    //         logger.block_size = block_size;

    // #pragma omp parallel for num_threads(12)
    //         for (uint64_t i = 0; i < rows; i++)
    //         {
    //             uint64_t r = row_permutation_uint64[i];
    //             intT start_pos = csr.rowptr[r];
    //             intT end_pos = csr.rowptr[r + 1];
    //             for (uint64_t nz = start_pos; nz < end_pos; nz++)
    //             {
    //                 intT col = csr.colidx[nz];
    //                 nnz[i / block_size][col / block_size]++;
    //             }
    //         }

    //         vector<vector<intT>> vec_csr_colidx(rows);
    //         vector<vector<DataT>> vec_csr_values(rows);

    //         for (uint64_t row_panel_id = 0; row_panel_id < nnz.size(); row_panel_id++)
    //         {
    //             for (uint64_t j = 0; j < nnz[row_panel_id].size(); j++)
    //             {
    //                 if (nnz[row_panel_id][j] > block_size * block_size * delta)
    //                 {
    //                     // BELL PACK
    //                     dense_tiles[row_panel_id].push_back(j);
    //                     logger.num_tiles++;
    //                     logger.avg_density_of_tiles += (float)nnz[row_panel_id][j] / (block_size * block_size);
    //                 }
    //                 else
    //                 {
    //                     if (nnz[row_panel_id][j] == 0)
    //                         continue;
    //                     // CSR
    //                     for (intT row = row_panel_id * block_size; row < (row_panel_id + 1) * block_size; row++)
    //                     {
    //                         intT row_id = row_permutation_uint64[row];
    //                         if (row_id >= original_rows)
    //                             continue;
    //                         intT start_pos = csr.rowptr[row_id];
    //                         intT end_pos = csr.rowptr[row_id + 1];
    //                         for (intT nz = start_pos; nz < end_pos; nz++)
    //                         {
    //                             if (csr.colidx[nz] / block_size != j)
    //                                 continue;

    //                             vec_csr_colidx[row].push_back(csr.colidx[nz]); 
    //                             csr_total_nonzeros++;
    //                             if (pattern_only)
    //                                 vec_csr_values[row].push_back(1);
    //                             else
    //                                 vec_csr_values[row].push_back(csr.values[nz]);
    //                         }
    //                     }
    //                 }
    //             }
    //             ell_cols = max(ell_cols, (int)dense_tiles[row_panel_id].size());
    //         }

    //         printf("ell_cols: %d\n", ell_cols);
    //         printf("ell_rows: %d\n", ell_rows);
    //         printf("block_size: %d\n", block_size);

    //         size_t free_mem, total_mem;
    //         cudaMemGetInfo(&free_mem, &total_mem);

    //         printf("Needed memory for BELL PACK: %lu MB\n", (size_t)ell_rows * (size_t)ell_cols * (size_t)block_size * (size_t)block_size * sizeof(DataT_H) / (1024 * 1024));
    //         printf("Free memory: %lu MB, Total memory: %lu MB\n", free_mem / (1024 * 1024), total_mem / (1024 * 1024));

    //         if( (size_t)ell_rows * (size_t)ell_cols * (size_t)block_size * (size_t)block_size * sizeof(DataT_H) > free_mem)
    //         {
    //             std::fprintf(stderr,
    //             "Not enough GPU memory for BELL PACK (need=%.2f MB, free=%.2f MB) — switch to CSR only\n", 
    //             (size_t)ell_rows * (size_t)ell_cols * (size_t)block_size * (size_t)block_size * sizeof(DataT_H) / (1024.0*1024.0),
    //             free_mem / (1024.0*1024.0));
    //             std::fflush(stderr);
    //             std::exit(EXIT_FAILURE);
    //         }

    //         ellColInd_uint64_t = new int64_t[(size_t)ell_rows * ell_cols];
    //         ellValues = new DataT[(size_t)ell_rows * (size_t)ell_cols * (size_t)block_size * block_size];
    //         h_ellValues = new DataT_H[(size_t)ell_rows * ell_cols * (size_t)block_size * block_size];

    //         memset(ellColInd_uint64_t, -1, (size_t)ell_rows * ell_cols * sizeof(int64_t));
    //         memset(ellValues, 0, (size_t)ell_rows * ell_cols * (size_t)block_size * block_size * sizeof(DataT));
    //         memset(h_ellValues, 0, (size_t)ell_rows * ell_cols * (size_t)block_size * block_size * sizeof(DataT_H));

    //         for (int i = 0; i < ell_rows; i++)
    //         {
    //             for (int j = 0; j < dense_tiles[i].size(); j++)
    //             {
    //                 ellColInd_uint64_t[i * ell_cols + j] = dense_tiles[i][j];
    //             }
    //         }
    //         for (int i = 0; i < rows; i++)
    //         {
    //             intT r = row_permutation_uint64[i];
    //             intT start_pos = csr.rowptr[r];
    //             intT end_pos = csr.rowptr[r + 1];
    //             for (uint64_t nz = start_pos; nz < end_pos; nz++)
    //             {
    //                 bellpack_total_nonzeros++;
    //                 uint64_t row_panel_id = i / block_size;
    //                 uint64_t original_col = csr.colidx[nz];
    //                 uint64_t inner_row = i % block_size;

    //                 auto it_ellblock_col = find(dense_tiles[row_panel_id].begin(), dense_tiles[row_panel_id].end(), (original_col / block_size));
    //                 if (it_ellblock_col == dense_tiles[row_panel_id].end())
    //                     continue;
    //                 uint64_t ellblock_col = it_ellblock_col - dense_tiles[row_panel_id].begin();
    //                 uint64_t inner_col = original_col % block_size;
    //                 uint64_t idx = (uint64_t)(i * ell_cols * block_size) + (uint64_t)(ellblock_col * block_size) + (uint64_t)inner_col;
    //                 // uint64_t idx =  (uint64_t)row_panel_id * ell_cols * block_size * block_size +
    //                 //                 (uint64_t)ellblock_col * block_size * block_size +
    //                 //                 (uint64_t)inner_row    * block_size +
    //                 //                 inner_col;

    //                 if (not pattern_only)
    //                 {
    //                     ellValues[idx] = csr.values[(size_t)offset + nz];
    //                     h_ellValues[idx] = __float2half(csr.values[(size_t)offset + nz]);
    //                 }
    //                 else
    //                 {
    //                     ellValues[idx] = 1;
    //                     h_ellValues[idx] = __float2half(1);
    //                 }
    //             }
    //         }
    //         csr_rowptr = new intT[rows + 1];
    //         csr_colidx = new intT[csr_total_nonzeros];
    //         csr_values = new DataT[csr_total_nonzeros];
    //         csr_rowptr[0] = 0;
    //         offset = 0;
    //         for (intT r = 0; r < rows; r++)
    //         {
    //             std::copy(vec_csr_colidx[r].begin(), vec_csr_colidx[r].end(), csr_colidx + offset);
    //             std::copy(vec_csr_values[r].begin(), vec_csr_values[r].end(), csr_values + offset);
    //             csr_rowptr[r + 1] = csr_rowptr[r] + (intT)vec_csr_colidx[r].size();
    //             offset += vec_csr_colidx[r].size();
    //         }

    //         logger.nnz_in_csr = csr_total_nonzeros;
    //         logger.nnz_in_bellpack = bellpack_total_nonzeros;
    //         if (logger.num_tiles)
    //             logger.avg_density_of_tiles /= logger.num_tiles;
            
    //         bellpack_total_nonzeros = csr.total_nonzeros - csr_total_nonzeros;
    //         assert(csr_total_nonzeros + bellpack_total_nonzeros == csr.total_nonzeros);
    //     }

    //     BSA_HYBRID(CSR_uint64_t &csr, LOGGER &logger, intT block_size_, float delta, vector<uint64_t> row_permutation_)
    //     {
    //         rows = csr.rows;
    //         cols = csr.cols;
    //         block_size = block_size_;

    //         original_rows = csr.original_rows;
    //         original_cols = csr.original_cols;
    //         pattern_only = csr.pattern_only;
    //         row_permutation_uint64 = row_permutation_;

    //         assert(rows % block_size == 0);
    //         assert(cols % block_size == 0);

    //         ell_rows = rows / block_size;
    //         ell_cols = 0;
    //         vector<vector<int>> nnz(ell_rows, vector<intT>(cols / block_size, 0));
    //         vector<vector<uint64_t>> dense_tiles(ell_rows);
    //         uint64_t offset = 0;

    //         logger.rows = original_rows;
    //         logger.cols = original_cols;
    //         logger.block_size = block_size;

    // #pragma omp parallel for num_threads(12)
    //         for (uint64_t i = 0; i < rows; i++)
    //         {
    //             uint64_t r = row_permutation_uint64[i];
    //             uint64_t start_pos = csr.rowptr[r];
    //             uint64_t end_pos = csr.rowptr[r + 1];
    //             for (uint64_t nz = start_pos; nz < end_pos; nz++)
    //             {
    //                 uint64_t col = csr.colidx[nz];
    //                 nnz[i / block_size][col / block_size]++;
    //             }
    //         }

    //         vector<vector<uint64_t>> vec_csr_colidx(rows);
    //         vector<vector<DataT>> vec_csr_values(rows);

    //         for (uint64_t row_panel_id = 0; row_panel_id < nnz.size(); row_panel_id++)
    //         {
    //             for (uint64_t j = 0; j < nnz[row_panel_id].size(); j++)
    //             {
    //                 if (nnz[row_panel_id][j] > block_size * block_size * delta)
    //                 {
    //                     // BELL PACK
    //                     dense_tiles[row_panel_id].push_back(j);
    //                     logger.num_tiles++;
    //                     logger.avg_density_of_tiles += (float)nnz[row_panel_id][j] / (block_size * block_size);
    //                 }
    //                 else
    //                 {
    //                     if (nnz[row_panel_id][j] == 0)
    //                         continue;
    //                     // CSR
    //                     for (uint64_t row = row_panel_id * block_size; row < (row_panel_id + 1) * block_size; row++)
    //                     {
    //                         uint64_t row_id = row_permutation_uint64[row];
    //                         if (row_id >= original_rows)
    //                             continue;
    //                         uint64_t start_pos = csr.rowptr[row_id];
    //                         uint64_t end_pos = csr.rowptr[row_id + 1];
    //                         for (uint64_t nz = start_pos; nz < end_pos; nz++)
    //                         {
    //                             if (csr.colidx[nz] / block_size != j)
    //                                 continue;

    //                             vec_csr_colidx[row].push_back(csr.colidx[nz]); 
    //                             csr_total_nonzeros++;
    //                             if (pattern_only)
    //                                 vec_csr_values[row].push_back(1);
    //                             else
    //                                 vec_csr_values[row].push_back(csr.values[nz]);
    //                         }
    //                     }
    //                 }
    //             }
    //             ell_cols = max(ell_cols, (int)dense_tiles[row_panel_id].size());
    //         }

    //         printf("ell_cols: %d\n", ell_cols);
    //         printf("ell_rows: %d\n", ell_rows);
    //         printf("block_size: %d\n", block_size);

    //         size_t free_mem, total_mem;
    //         cudaMemGetInfo(&free_mem, &total_mem);

    //         printf("Needed memory for BELL PACK: %lu MB\n", (size_t)ell_rows * (size_t)ell_cols * (size_t)block_size * (size_t)block_size * sizeof(DataT_H) / (1024 * 1024));
    //         printf("Free memory: %lu MB, Total memory: %lu MB\n", free_mem / (1024 * 1024), total_mem / (1024 * 1024));

    //         if( (size_t)ell_rows * (size_t)ell_cols * (size_t)block_size * (size_t)block_size * sizeof(DataT_H) > free_mem)
    //         {
    //             std::fprintf(stderr,
    //             "Not enough GPU memory for BELL PACK (need=%.2f MB, free=%.2f MB) — switch to CSR only\n", 
    //             (size_t)ell_rows * (size_t)ell_cols * (size_t)block_size * (size_t)block_size * sizeof(DataT_H) / (1024.0*1024.0),
    //             free_mem / (1024.0*1024.0));
    //             std::fflush(stderr);
    //             std::exit(EXIT_FAILURE);
    //         }

    //         ellColInd_uint64_t = new int64_t[(size_t)ell_rows * ell_cols];
    //         ellValues = new DataT[(size_t)ell_rows * (size_t)ell_cols * (size_t)block_size * block_size];
    //         h_ellValues = new DataT_H[(size_t)ell_rows * ell_cols * (size_t)block_size * block_size];

    //         memset(ellColInd_uint64_t, -1, (size_t)ell_rows * ell_cols * sizeof(int64_t));
    //         memset(ellValues, 0, (size_t)ell_rows * ell_cols * (size_t)block_size * block_size * sizeof(DataT));
    //         memset(h_ellValues, 0, (size_t)ell_rows * ell_cols * (size_t)block_size * block_size * sizeof(DataT_H));

    //         for (uint64_t i = 0; i < ell_rows; i++)
    //         {
    //             for (uint64_t j = 0; j < dense_tiles[i].size(); j++)
    //             {
    //                 ellColInd_uint64_t[i * ell_cols + j] = dense_tiles[i][j];
    //             }
    //         }
    //         for (uint64_t i = 0; i < rows; i++)
    //         {
    //             uint64_t r = row_permutation_uint64[i];
    //             uint64_t start_pos = csr.rowptr[r];
    //             uint64_t end_pos = csr.rowptr[r + 1];
    //             for (uint64_t nz = start_pos; nz < end_pos; nz++)
    //             {
    //                 bellpack_total_nonzeros++;
    //                 uint64_t row_panel_id = i / block_size;
    //                 uint64_t original_col = csr.colidx[nz];
    //                 uint64_t inner_row = i % block_size;

    //                 auto it_ellblock_col = find(dense_tiles[row_panel_id].begin(), dense_tiles[row_panel_id].end(), (original_col / block_size));
    //                 if (it_ellblock_col == dense_tiles[row_panel_id].end())
    //                     continue;
    //                 uint64_t ellblock_col = it_ellblock_col - dense_tiles[row_panel_id].begin();
    //                 uint64_t inner_col = original_col % block_size;
    //                 uint64_t idx = (uint64_t)(i * ell_cols * block_size) + (uint64_t)(ellblock_col * block_size) + (uint64_t)inner_col;
    //                 // uint64_t idx =  (uint64_t)row_panel_id * ell_cols * block_size * block_size +
    //                 //                 (uint64_t)ellblock_col * block_size * block_size +
    //                 //                 (uint64_t)inner_row    * block_size +
    //                 //                 inner_col;

    //                 if (not pattern_only)
    //                 {
    //                     ellValues[idx] = csr.values[(size_t)offset + nz];
    //                     h_ellValues[idx] = __float2half(csr.values[(size_t)offset + nz]);
    //                 }
    //                 else
    //                 {
    //                     ellValues[idx] = 1;
    //                     h_ellValues[idx] = __float2half(1);
    //                 }
    //             }
    //         }
    //         csr_rowptr = new intT[rows + 1];
    //         csr_colidx = new intT[csr_total_nonzeros];
    //         csr_values = new DataT[csr_total_nonzeros];
    //         csr_rowptr[0] = 0;
    //         offset = 0;
    //         for (intT r = 0; r < rows; r++)
    //         {
    //             std::copy(vec_csr_colidx[r].begin(), vec_csr_colidx[r].end(), csr_colidx + offset);
    //             std::copy(vec_csr_values[r].begin(), vec_csr_values[r].end(), csr_values + offset);
    //             csr_rowptr[r + 1] = csr_rowptr[r] + (uint64_t)vec_csr_colidx[r].size();
    //             offset += vec_csr_colidx[r].size();
    //         }

    //         logger.nnz_in_csr = csr_total_nonzeros;
    //         logger.nnz_in_bellpack = bellpack_total_nonzeros;
    //         if (logger.num_tiles)
    //             logger.avg_density_of_tiles /= logger.num_tiles;
            
    //         bellpack_total_nonzeros = csr.total_nonzeros - csr_total_nonzeros;
    //         assert(csr_total_nonzeros + bellpack_total_nonzeros == csr.total_nonzeros);
    //     }


        // ~BSA_HYBRID()
        // {
        //     // delete[] ellColInd;
        //     if(ellColInd != nullptr)
        //         delete[] ellColInd;

        //     if(ellColInd_uint64_t != nullptr)
        //         delete[] ellColInd_uint64_t;

        //     delete[] ellValues;
        //     delete[] h_ellValues;
        //     delete[] csr_rowptr;
        //     delete[] csr_colidx;
        //     delete[] csr_values;
        // }

    };


    class BSA_HYBRID_uint64_t
    {
    public:

        uint64_t rows;
        uint64_t cols;
        // for zero paading
        uint64_t original_rows;
        uint64_t original_cols;

        // int rows;
        // int cols;
        // int original_rows;
        // int original_cols;

        bool pattern_only;

        // The result of reordering
        vector<uint64_t> row_permutation_uint64;
        vector<int> row_permutation;

        // BELL PACK
        uint64_t bellpack_total_nonzeros = 0;
        // int bellpack_total_nonzeros = 0;
        intT block_size;
        intT *ellColInd = nullptr;
        int64_t *ellColInd_uint64_t = nullptr;
        DataT *ellValues= nullptr;
        DataT_H *h_ellValues= nullptr;

        uint64_t ell_rows;
        uint64_t ell_cols;
        // int ell_rows;
        // int ell_cols;


        // CSR
        uint64_t csr_total_nonzeros = 0;
        // int csr_total_nonzeros = 0;
        // intT *csr_rowptr= nullptr;
        // intT *csr_colidx= nullptr;
        uint64_t *csr_rowptr= nullptr;
        uint64_t *csr_colidx= nullptr;
        DataT *csr_values= nullptr;
        DataT_H *csv_h_values= nullptr;


        BSA_HYBRID_uint64_t(CSR_uint64_t &csr, LOGGER &logger, intT block_size_, float delta, vector<uint64_t> row_permutation_)
        {
            rows = csr.rows;
            cols = csr.cols;
            block_size = block_size_;

            original_rows = csr.original_rows;
            original_cols = csr.original_cols;
            pattern_only = csr.pattern_only;
            row_permutation_uint64 = row_permutation_;

            assert(rows % block_size == 0);
            assert(cols % block_size == 0);

            ell_rows = rows / block_size;
            ell_cols = 0;
            vector<vector<int>> nnz(ell_rows, vector<intT>(cols / block_size, 0));
            vector<vector<uint64_t>> dense_tiles(ell_rows);
            uint64_t offset = 0;

            logger.rows = original_rows;
            logger.cols = original_cols;
            logger.block_size = block_size;

    #pragma omp parallel for num_threads(12)
            for (uint64_t i = 0; i < rows; i++)
            {
                uint64_t r = row_permutation_uint64[i];
                uint64_t start_pos = csr.rowptr[r];
                uint64_t end_pos = csr.rowptr[r + 1];
                for (uint64_t nz = start_pos; nz < end_pos; nz++)
                {
                    uint64_t col = csr.colidx[nz];
                    nnz[i / block_size][col / block_size]++;
                }
            }

            vector<vector<uint64_t>> vec_csr_colidx(rows);
            vector<vector<DataT>> vec_csr_values(rows);

            for (uint64_t row_panel_id = 0; row_panel_id < nnz.size(); row_panel_id++)
            {
                for (uint64_t j = 0; j < nnz[row_panel_id].size(); j++)
                {
                    if (nnz[row_panel_id][j] > block_size * block_size * delta)
                    {
                        // BELL PACK
                        dense_tiles[row_panel_id].push_back(j);
                        logger.num_tiles++;
                        logger.avg_density_of_tiles += (float)nnz[row_panel_id][j] / (block_size * block_size);
                    }
                    else
                    {
                        if (nnz[row_panel_id][j] == 0)
                            continue;
                        // CSR
                        for (uint64_t row = row_panel_id * block_size; row < (row_panel_id + 1) * block_size; row++)
                        {
                            uint64_t row_id = row_permutation_uint64[row];
                            if (row_id >= original_rows)
                                continue;
                            uint64_t start_pos = csr.rowptr[row_id];
                            uint64_t end_pos = csr.rowptr[row_id + 1];
                            for (uint64_t nz = start_pos; nz < end_pos; nz++)
                            {
                                if (csr.colidx[nz] / block_size != j)
                                    continue;

                                vec_csr_colidx[row].push_back(csr.colidx[nz]); 
                                csr_total_nonzeros++;
                                if (pattern_only)
                                    vec_csr_values[row].push_back(1);
                                else
                                    vec_csr_values[row].push_back(csr.values[nz]);
                            }
                        }
                    }
                }
                ell_cols = max(ell_cols, (uint64_t)dense_tiles[row_panel_id].size());
            }

            // printf("ell_cols: %d\n", ell_cols);
            // printf("ell_rows: %d\n", ell_rows);
            // printf("block_size: %d\n", block_size);

            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);

            printf("Needed memory for BELL PACK: %lu MB\n", (size_t)ell_rows * (size_t)ell_cols * (size_t)block_size * (size_t)block_size * sizeof(DataT_H) / (1024 * 1024));
            printf("Free memory: %lu MB, Total memory: %lu MB\n", free_mem / (1024 * 1024), total_mem / (1024 * 1024));

            if( (size_t)ell_rows * (size_t)ell_cols * (size_t)block_size * (size_t)block_size * sizeof(DataT_H) > free_mem)
            {
                std::fprintf(stderr,
                "Not enough GPU memory for BELL PACK (need=%.2f MB, free=%.2f MB) — switch to CSR only\n", 
                (size_t)ell_rows * (size_t)ell_cols * (size_t)block_size * (size_t)block_size * sizeof(DataT_H) / (1024.0*1024.0),
                free_mem / (1024.0*1024.0));
                std::fflush(stderr);
                std::exit(EXIT_FAILURE);
            }

            ellColInd_uint64_t = new int64_t[(size_t)ell_rows * ell_cols];
            ellValues = new DataT[(size_t)ell_rows * (size_t)ell_cols * (size_t)block_size * block_size];
            h_ellValues = new DataT_H[(size_t)ell_rows * ell_cols * (size_t)block_size * block_size];

            memset(ellColInd_uint64_t, -1, (size_t)ell_rows * ell_cols * sizeof(int64_t));
            memset(ellValues, 0, (size_t)ell_rows * ell_cols * (size_t)block_size * block_size * sizeof(DataT));
            memset(h_ellValues, 0, (size_t)ell_rows * ell_cols * (size_t)block_size * block_size * sizeof(DataT_H));

            for (uint64_t i = 0; i < ell_rows; i++)
            {
                for (uint64_t j = 0; j < dense_tiles[i].size(); j++)
                {
                    ellColInd_uint64_t[i * ell_cols + j] = dense_tiles[i][j];
                }
            }
            for (uint64_t i = 0; i < rows; i++)
            {
                uint64_t r = row_permutation_uint64[i];
                uint64_t start_pos = csr.rowptr[r];
                uint64_t end_pos = csr.rowptr[r + 1];
                for (uint64_t nz = start_pos; nz < end_pos; nz++)
                {
                    bellpack_total_nonzeros++;
                    uint64_t row_panel_id = i / block_size;
                    uint64_t original_col = csr.colidx[nz];
                    uint64_t inner_row = i % block_size;

                    auto it_ellblock_col = find(dense_tiles[row_panel_id].begin(), dense_tiles[row_panel_id].end(), (original_col / block_size));
                    if (it_ellblock_col == dense_tiles[row_panel_id].end())
                        continue;
                    uint64_t ellblock_col = it_ellblock_col - dense_tiles[row_panel_id].begin();
                    uint64_t inner_col = original_col % block_size;
                    uint64_t idx = (uint64_t)(i * ell_cols * block_size) + (uint64_t)(ellblock_col * block_size) + (uint64_t)inner_col;
                    // uint64_t idx =  (uint64_t)row_panel_id * ell_cols * block_size * block_size +
                    //                 (uint64_t)ellblock_col * block_size * block_size +
                    //                 (uint64_t)inner_row    * block_size +
                    //                 inner_col;

                    if (not pattern_only)
                    {
                        ellValues[idx] = csr.values[(size_t)offset + nz];
                        h_ellValues[idx] = __float2half(csr.values[(size_t)offset + nz]);
                    }
                    else
                    {
                        ellValues[idx] = 1;
                        h_ellValues[idx] = __float2half(1);
                    }
                }
            }
            csr_rowptr = new uint64_t[rows + 1];
            csr_colidx = new uint64_t[csr_total_nonzeros];
            csr_values = new DataT[csr_total_nonzeros];
            csr_rowptr[0] = 0;
            offset = 0;
            for (intT r = 0; r < rows; r++)
            {
                std::copy(vec_csr_colidx[r].begin(), vec_csr_colidx[r].end(), csr_colidx + offset);
                std::copy(vec_csr_values[r].begin(), vec_csr_values[r].end(), csr_values + offset);
                csr_rowptr[r + 1] = csr_rowptr[r] + (uint64_t)vec_csr_colidx[r].size();
                offset += vec_csr_colidx[r].size();
            }

            logger.nnz_in_csr = csr_total_nonzeros;
            logger.nnz_in_bellpack = bellpack_total_nonzeros;
            if (logger.num_tiles)
                logger.avg_density_of_tiles /= logger.num_tiles;
            
            bellpack_total_nonzeros = csr.total_nonzeros - csr_total_nonzeros;
            assert(csr_total_nonzeros + bellpack_total_nonzeros == csr.total_nonzeros);
        }


        // ~BSA_HYBRID()
        // {
        //     // delete[] ellColInd;
        //     if(ellColInd != nullptr)
        //         delete[] ellColInd;

        //     if(ellColInd_uint64_t != nullptr)
        //         delete[] ellColInd_uint64_t;

        //     delete[] ellValues;
        //     delete[] h_ellValues;
        //     delete[] csr_rowptr;
        //     delete[] csr_colidx;
        //     delete[] csr_values;
        // }

        ~BSA_HYBRID_uint64_t()
        {
            // delete[] ellColInd;
            if(ellColInd != nullptr)
                delete[] ellColInd;

            if(ellColInd_uint64_t != nullptr)
                delete[] ellColInd_uint64_t;

            delete[] ellValues;
            delete[] h_ellValues;

            if(csr_rowptr != nullptr) delete[] csr_rowptr;
            if(csr_colidx != nullptr) delete[] csr_colidx;
            if(csr_values != nullptr) delete[] csr_values;

        }



    };




    class ARR
    {
    public:
        intT original_rows;
        intT rows;
        intT cols;
        DataT *mat;
        DataT_H *h_mat;
        bool with_half;

        ARR(intT original_rows_, intT rows_, intT cols_, bool with_half_)
        {
            original_rows = original_rows_;
            rows = rows_;
            cols = cols_;
            with_half = with_half_;

            mat = new DataT[(size_t)rows * (size_t)cols];
            if (with_half)
                h_mat = new DataT_H[(size_t)rows * (size_t)cols];

            memset(mat, 0, (size_t)rows * (size_t)cols * sizeof(DataT));
            if (with_half)
                memset(h_mat, 0, (size_t)rows * (size_t)cols * sizeof(DataT_H));
        }

        void fill_random(bool zero_padding)
        {
            random_device rd;
            mt19937 e2(rd());
            uniform_real_distribution<> dist(0, 1);
            if (zero_padding)
            {

                for (uint64_t n = 0; n < (size_t)rows * (size_t)cols; n++)
                {
                    int r = n / cols;
                    if (r < original_rows)
                    {
                        mat[n] = dist(e2);
                        if (with_half)
                            h_mat[n] = __float2half(mat[n]);
                    }
                    else
                    {
                        mat[n] = 0.0;
                        if (with_half)
                            h_mat[n] = __float2half(0.0);
                    }
                }
            }
            else
            {
                for (int n = 0; n < rows * cols; n++)
                {
                    mat[n] = dist(e2);
                    if (with_half)
                        h_mat[n] = __float2half(mat[n]);
                }
            }
        }

        ~ARR()
        {
            delete[] mat;
            if (with_half)
                delete[] h_mat;
        }
    };

    class ARR_64
    {
    public:
        int64_t original_rows;
        int64_t rows;
        int64_t cols;
        DataT *mat;
        DataT_H *h_mat;
        bool with_half;

        ARR_64(int64_t original_rows_, int64_t rows_, int64_t cols_, bool with_half_)
        {
            original_rows = original_rows_;
            rows = rows_;
            cols = cols_;
            with_half = with_half_;

            mat = new DataT[(size_t)rows * (size_t)cols];
            if (with_half)
                h_mat = new DataT_H[(size_t)rows * (size_t)cols];

            memset(mat, 0, (size_t)rows * (size_t)cols * sizeof(DataT));
            if (with_half)
                memset(h_mat, 0, (size_t)rows * (size_t)cols * sizeof(DataT_H));
        }

        void fill_random(bool zero_padding)
        {
            random_device rd;
            mt19937 e2(rd());
            uniform_real_distribution<> dist(0, 1);
            if (zero_padding)
            {

                for (uint64_t n = 0; n < (size_t)rows * (size_t)cols; n++)
                {
                    int r = n / cols;
                    if (r < original_rows)
                    {
                        mat[n] = dist(e2);
                        if (with_half)
                            h_mat[n] = __float2half(mat[n]);
                    }
                    else
                    {
                        mat[n] = 0.0;
                        if (with_half)
                            h_mat[n] = __float2half(0.0);
                    }
                }
            }
            else
            {
                for (int64_t n = 0; n < rows * cols; n++)
                {
                    mat[n] = dist(e2);
                    if (with_half)
                        h_mat[n] = __float2half(mat[n]);
                }
            }
        }

        ~ARR_64()
        {
            delete[] mat;
            if (with_half)
                delete[] h_mat;
        }
    };






#endif