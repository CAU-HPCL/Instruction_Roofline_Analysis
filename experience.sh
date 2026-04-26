#!/bin/bash

WEBHOOK_URL="https://hooks.slack.com/services/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"

ROOT_DIR="/workspace/ICTC"
DATA_ROOT_DIR="/workspace/TC_BELL/TC-BELL/dataset/DLMC/dlmc"
TEST_EXE_1="/workspace/ICTC/build/spmm_all"

TIME_LIMIT=600  # 초 단위 timeout 시간
N_COL=(128)
MODELS=("rn50" "transformer")
SUB_DATASET=("variational_dropout" "magnitude_pruning" "random_pruning")

ALPHAS=(1.0)
DELTA=(0.0)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="/workspace/ICTC/results/ICTC_TEST_RESULT_$TIMESTAMP"

mkdir -p "$RESULT_DIR"

TEMP_ERR="$RESULT_DIR/temp_err.log"
TIMEOUT_CSV="$RESULT_DIR/timeout_cases.csv"
ERROR_CSV="$RESULT_DIR/error_cases.csv"
LOGFILE="$RESULT_DIR/test_run.log"

OUTPUT_FILE_1="$RESULT_DIR/ICTC_spmm_compare_$TIMESTAMP.csv"
SEGFAULT_CSV="$RESULT_DIR/segfault_cases.csv"

if [ -e ${OUTPUT_FILE_1} ]; then
    rm -rf ${OUTPUT_FILE_1}
    touch ${OUTPUT_FILE_1}
fi


if [ ! -f "$SEGFAULT_CSV" ]; then
    echo "file,alpha,delta,n_cols,exit_code,last_stderr" > "$SEGFAULT_CSV"
fi

if [ ! -f "$TIMEOUT_CSV" ]; then
    echo "file,alpha,delta,n_cols,exit_code,last_stderr" > "$TIMEOUT_CSV"
fi

if [ ! -f "$ERROR_CSV" ]; then
    echo "file,alpha,delta,n_cols,exit_code,last_stderr" > "$ERROR_CSV"
fi

### ✅ Slack 알림 함수
function notify_slack {
    local STATUS="$1"
    local MESSAGE="$2"
    curl -s -X POST -H 'Content-type: application/json' \
         --data "{\"text\":\"[$STATUS] $MESSAGE\n시간: $(date)\"}" \
         "$WEBHOOK_URL"
}

### ✅ GPU 상태 감시 함수
function check_gpu_status {
    GPU_INFO=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits)
    TEMP=$(echo $GPU_INFO | cut -d',' -f1 | xargs)
    POWER=$(echo $GPU_INFO | cut -d',' -f2 | xargs)
    UTIL=$(echo $GPU_INFO | cut -d',' -f3 | xargs)

    echo "GPU 상태 - 온도: ${TEMP}°C, 전력: ${POWER}W, 사용률: ${UTIL}%" >> "$LOGFILE"

    if [ "$TEMP" -gt 85 ]; then
        notify_slack " GPU 온도 이상" "GPU 온도 ${TEMP}°C 초과!"
    fi
    if [ "$(echo "$POWER > 300" | bc -l)" -eq 1 ]; then
        notify_slack " GPU 전력 이상" "GPU 전력 ${POWER}W 초과!"
    fi
}


notify_slack "🚀 시작" "1. 테스트가 시작되었습니다(modified_kernel). 결과 폴더: $RESULT_DIR"

echo "infile,repetitions,N,M,K,NNZ,density,sparsity,MU,MAX,STD_NNZ,MAX_MU,AVE_BW,STD_BW,cusparse_time,cginkgo_time,kokkos_time,cusparse_error,ginkgo_error,kokkos_error,cusparse_PF,ginkgo_PF,kokkos_PF,winner" > "$OUTPUT_FILE_1"
###  메인 루프 시작
for MODEL in "${MODELS[@]}"; do
    for SUB in "${SUB_DATASET[@]}"; do
        for SP in "$DATA_ROOT_DIR/$MODEL/$SUB"/*; do
            SPARSITY=$(basename "$SP")

            # if [ "$(echo "$SPARSITY <= 0.9" | bc)" -eq 1 ]; then
                # CNT=0
                # STOP=3
                for FILE in "$SP"/*; do
                    # if [ $CNT -eq $STOP ]; then
                    #     break
                    # fi
                    echo "Testing $FILE"

                        for N in "${N_COL[@]}"; do

                            for A in  "${ALPHAS[@]}"; do
                                
                                for D in "${DELTA[@]}"; do

                                    check_gpu_status  # 전

                                    timeout "$TIME_LIMIT" "$TEST_EXE_1" -f "$FILE" -n "$N" -x "$A" -d"$D" -o "$OUTPUT_FILE_1" -c "0" > /dev/null 2> "$TEMP_ERR"
                                    RESULT=$?

                                    check_gpu_status  # 후

                                    LAST_LINE=$(tail -n 1 "$TEMP_ERR" | tr -d '\n' | sed 's/,/ /g')

                                    if [ $RESULT -eq 124 ]; then
                                        echo "$FILE,$A,$D,$N,124,$LAST_LINE" >> "$TIMEOUT_CSV"
                                        notify_slack "⏰ Timeout" "FILE=$FILE A=$A D=$D N=$N\n마지막 에러: $LAST_LINE"

                                    elif [ $RESULT -eq 139 ]; then
                                        echo "$FILE,$A,$D,$N,139,$LAST_LINE" >> "$SEGFAULT_CSV"
                                        notify_slack "💥 Segfault" "FILE=$FILE A=$A D=$D N=$N (exit=139)\n마지막 에러: $LAST_LINE"

                                    elif [ $RESULT -ne 0 ]; then
                                        echo "$FILE,$A,$D,$N,$RESULT,$LAST_LINE" >> "$ERROR_CSV"
                                        notify_slack "❌ Error" "FILE=$FILE A=$A D=$D N=$N (exit=$RESULT)\n마지막 에러: $LAST_LINE"
                                    fi
                                done
                            done
                        done

                    # CNT=$((CNT+1))
                done
            # fi
        done
    done
done
notify_slack "✅ 완료" "1. 테스트가 모두 끝났습니다(segment2-kernel5). 결과 폴더: $RESULT_DIR"

if [ ! -f "$SEGFAULT_CSV" ]; then
    echo "-----------------------------------------------------" > "$SEGFAULT_CSV"
fi

if [ ! -f "$TIMEOUT_CSV" ]; then
    echo "-----------------------------------------------------" > "$TIMEOUT_CSV"
fi

if [ ! -f "$ERROR_CSV" ]; then
    echo "-----------------------------------------------------" > "$ERROR_CSV"
fi


# echo "▶ Running Python script..."
# python3 "$PYTHON_SCRIPT" -i "$OUTPUT_FILE_1" -o "$RESULT_ANALYSIS_DIR"

# echo "▶ Compressing results..."
# zip -r "$ZIP_NAME" "$RESULT_ANALYSIS_DIR"/*

