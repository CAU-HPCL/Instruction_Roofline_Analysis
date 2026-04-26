#include "definitions.h"
#include "matrices.h"
#include "logger.h"
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include "utilities.h"
#include "helper_cuda.h"

#ifndef SPMM_H
#define SPMM_H

void spmm_csr(BSA_HYBRID &bsa_lhs, ARR &rhs, DataT_C *matC, float &elapsed_time);
void spmm_bellpack(BSA_HYBRID &bsa_lhs, ARR &rhs, DataT_C *matC, float &elapsed_time, bool compress_rows);


void bsa_spmm(BSA_HYBRID &bsa_lhs, ARR &rhs, ARR &result_mat, LOGGER &logger, intT n_repetitions, bool compress_rows, bool valid);
void cusparse_spmm(CSR &lhs, ARR &rhs, ARR &result_mat, LOGGER &logger, intT n_repetitions, bool pattern_only);
void cublas_gemm(CSR &lhs, ARR &rhs, ARR &result_mat, LOGGER &logger, intT n_repetitions, bool pattern_only);
#endif