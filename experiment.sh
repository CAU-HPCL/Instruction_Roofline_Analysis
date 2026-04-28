#!/bin/bash

BUILD_DIR="./build"
DATASET_DIR="./dataset"
OUTPUT_DIR="./profile"

mkdir -p $OUTPUT_DIR
mkdir -p ./result

MATRICES=(
    "bottleneck_1_block_group_projection_block_group4.smtx"
    "bottleneck_1_block_group_projection_block_group3.smtx"
    "bottleneck_1_block_group_projection_block_group2.smtx"
)

METRICS="smsp__inst_executed.sum,\
smsp__thread_inst_executed.sum,\
smsp__inst_executed_op_global_ld.sum,\
smsp__inst_executed_op_global_st.sum,\
smsp__inst_executed_op_shared_ld.sum,\
smsp__inst_executed_op_shared_st.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_write.sum,\
dram__sectors_read.sum,\
dram__sectors_write.sum"

# cuSPARSE 프로파일링
echo "===== cuSPARSE NCU Profiling ====="
for MATRIX in "${MATRICES[@]}"; do
    NAME="${MATRIX%.smtx}"
    echo "Profiling: $MATRIX"
    ncu --metrics $METRICS \
        --target-processes all \
        --csv \
        $BUILD_DIR/NCU_spmm_cuSPARSE -f $DATASET_DIR/$MATRIX \
        2>&1 | tee $OUTPUT_DIR/ncu_cusparse_${NAME}.csv
done

# Ginkgo 프로파일링
echo "===== Ginkgo NCU Profiling ====="
for MATRIX in "${MATRICES[@]}"; do
    NAME="${MATRIX%.smtx}"
    echo "Profiling: $MATRIX"
    ncu --metrics $METRICS \
        --target-processes all \
        --csv \
        $BUILD_DIR/NCU_spmm_Ginkgo -f $DATASET_DIR/$MATRIX \
        2>&1 | tee $OUTPUT_DIR/ncu_ginkgo_${NAME}.csv
done

# cuSPARSE 시간 측정
echo "===== cuSPARSE Time Measurement ====="
for MATRIX in "${MATRICES[@]}"; do
    NAME="${MATRIX%.smtx}"
    echo "Timing: $MATRIX"
    $BUILD_DIR/Time_spmm_cuSPARSE -f $DATASET_DIR/$MATRIX \
        2>&1 | tee $OUTPUT_DIR/time_cusparse_${NAME}.txt
done

# Ginkgo 시간 측정
echo "===== Ginkgo Time Measurement ====="
for MATRIX in "${MATRICES[@]}"; do
    NAME="${MATRIX%.smtx}"
    echo "Timing: $MATRIX"
    $BUILD_DIR/Time_spmm_Ginkgo -f $DATASET_DIR/$MATRIX \
        2>&1 | tee $OUTPUT_DIR/time_ginkgo_${NAME}.txt
done

echo "Done!"


