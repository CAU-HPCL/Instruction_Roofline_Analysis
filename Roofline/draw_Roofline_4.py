import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

# =========================
# 1) HW 상수/피크 (SI 단위)
# =========================
PEAK_INST_PER_CYCLE = 128 * 4 * 1
SM_HZ               = 2.52e9
PEAK_INST_PER_S = PEAK_INST_PER_CYCLE * SM_HZ
PEAK_GIPS = PEAK_INST_PER_S / 1e9

#RTX 4090 (Ada) 기준, 128 SM
NUM_SM = 128

BW_DRAM = 1008e9  # GB/s
S_DRAM = 32      # Bytes/TXN (32B/사이클 가정)
BW_DRAM_TXN_S = BW_DRAM / S_DRAM

L2_BYTES_PER_CYCLE = 1708
L2_HZ              = SM_HZ
S_L2               = 32
BW_L2_TXN_S = (L2_BYTES_PER_CYCLE * L2_HZ) / S_L2

L1_BYTES_PER_CYCLE = 121.2 * NUM_SM
L1_HZ = SM_HZ
S_L1 = 32
BW_L1_TXN_S = (L1_BYTES_PER_CYCLE * L1_HZ) / S_L1

SMEM_BYTES_PER_CYCLE = 127.9 * NUM_SM
SMEM_HZ = SM_HZ
S_SMEM = 128
BW_SMEM_TXN_S = (SMEM_BYTES_PER_CYCLE * SMEM_HZ) / S_SMEM

# X축 범위(inst/TXN)
AI_MIN, AI_MAX = 1e-2, 1e3
X_OVERRIDE = (AI_MIN, AI_MAX)
Y_OVERRIDE = None  # (ymin, ymax) 수동 지정 원하면 값 넣기 (단위=GIPS)


# =========================
# 2) Achieved points from counters
# =========================

# cuSPARSE start ========================== 
ACH_FROM_COUNTERS_1 = {
    # 시간
    "kernel_execution_time":                0.026432 * 1e-3,  # CUDA event time (초)
    
    # 인스트럭션
    "warp_level_executed_instructions":     4708021,   # smsp__inst_executed.sum
    "thread_level_executed_instructions":   4708021,   # smsp__thread_inst_executed.sum
    
    # 메모리 트랜잭션
    "gld_txn":    0,   # l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
    "gst_txn":    0,   # l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
    "smem_ld_txn": 0,  # l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
    "smem_st_txn": 0,  # l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
    "l2_rd_txn":  0,   # lts__t_sectors_op_read.sum
    "l2_wr_txn":  0,   # lts__t_sectors_op_write.sum
    "dram_rd_txn": 0,  # dram__sectors_read.sum
    "dram_wr_txn": 0,  # dram__sectors_write.sum

    # 그래프 스타일
    "color":        "#f0ba28",
    "marker":       "D",
    "size":         80,
    "label_prefix": "achieved",
}

ACH_FROM_COUNTERS_2 = {
    # 시간
    "kernel_execution_time":                0.026432 * 1e-3,  # CUDA event time (초)
    
    # 인스트럭션
    "warp_level_executed_instructions":     4708021,   # smsp__inst_executed.sum
    "thread_level_executed_instructions":   4708021,   # smsp__thread_inst_executed.sum
    
    # 메모리 트랜잭션
    "gld_txn":    0,   # l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
    "gst_txn":    0,   # l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
    "smem_ld_txn": 0,  # l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
    "smem_st_txn": 0,  # l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
    "l2_rd_txn":  0,   # lts__t_sectors_op_read.sum
    "l2_wr_txn":  0,   # lts__t_sectors_op_write.sum
    "dram_rd_txn": 0,  # dram__sectors_read.sum
    "dram_wr_txn": 0,  # dram__sectors_write.sum

    # 그래프 스타일
    "color":        "#f0ba28",
    "marker":       "D",
    "size":         80,
    "label_prefix": "achieved",
}

ACH_FROM_COUNTERS_3 = {
    # 시간
    "kernel_execution_time":                0.026432 * 1e-3,  # CUDA event time (초)
    
    # 인스트럭션
    "warp_level_executed_instructions":     4708021,   # smsp__inst_executed.sum
    "thread_level_executed_instructions":   4708021,   # smsp__thread_inst_executed.sum
    
    # 메모리 트랜잭션
    "gld_txn":    0,   # l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
    "gst_txn":    0,   # l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
    "smem_ld_txn": 0,  # l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
    "smem_st_txn": 0,  # l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
    "l2_rd_txn":  0,   # lts__t_sectors_op_read.sum
    "l2_wr_txn":  0,   # lts__t_sectors_op_write.sum
    "dram_rd_txn": 0,  # dram__sectors_read.sum
    "dram_wr_txn": 0,  # dram__sectors_write.sum

    # 그래프 스타일
    "color":        "#f0ba28",
    "marker":       "D",
    "size":         80,
    "label_prefix": "achieved",
}
# cuSPARSE end ========================== 

# Ginkgo start ========================== 
ACH_FROM_COUNTERS_4 = {
    # 시간
    "kernel_execution_time":                0.026432 * 1e-3,  # CUDA event time (초)
    
    # 인스트럭션
    "warp_level_executed_instructions":     4708021,   # smsp__inst_executed.sum
    "thread_level_executed_instructions":   4708021,   # smsp__thread_inst_executed.sum
    
    # 메모리 트랜잭션
    "gld_txn":    0,   # l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
    "gst_txn":    0,   # l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
    "smem_ld_txn": 0,  # l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
    "smem_st_txn": 0,  # l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
    "l2_rd_txn":  0,   # lts__t_sectors_op_read.sum
    "l2_wr_txn":  0,   # lts__t_sectors_op_write.sum
    "dram_rd_txn": 0,  # dram__sectors_read.sum
    "dram_wr_txn": 0,  # dram__sectors_write.sum

    # 그래프 스타일
    "color":        "#f0ba28",
    "marker":       "D",
    "size":         80,
    "label_prefix": "achieved",
}

ACH_FROM_COUNTERS_5 = {
    # 시간
    "kernel_execution_time":                0.026432 * 1e-3,  # CUDA event time (초)
    
    # 인스트럭션
    "warp_level_executed_instructions":     4708021,   # smsp__inst_executed.sum
    "thread_level_executed_instructions":   4708021,   # smsp__thread_inst_executed.sum
    
    # 메모리 트랜잭션
    "gld_txn":    0,   # l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
    "gst_txn":    0,   # l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
    "smem_ld_txn": 0,  # l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
    "smem_st_txn": 0,  # l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
    "l2_rd_txn":  0,   # lts__t_sectors_op_read.sum
    "l2_wr_txn":  0,   # lts__t_sectors_op_write.sum
    "dram_rd_txn": 0,  # dram__sectors_read.sum
    "dram_wr_txn": 0,  # dram__sectors_write.sum

    # 그래프 스타일
    "color":        "#f0ba28",
    "marker":       "D",
    "size":         80,
    "label_prefix": "achieved",
}

ACH_FROM_COUNTERS_6 = {
    # 시간
    "kernel_execution_time":                0.026432 * 1e-3,  # CUDA event time (초)
    
    # 인스트럭션
    "warp_level_executed_instructions":     4708021,   # smsp__inst_executed.sum
    "thread_level_executed_instructions":   4708021,   # smsp__thread_inst_executed.sum
    
    # 메모리 트랜잭션
    "gld_txn":    0,   # l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
    "gst_txn":    0,   # l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
    "smem_ld_txn": 0,  # l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
    "smem_st_txn": 0,  # l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
    "l2_rd_txn":  0,   # lts__t_sectors_op_read.sum
    "l2_wr_txn":  0,   # lts__t_sectors_op_write.sum
    "dram_rd_txn": 0,  # dram__sectors_read.sum
    "dram_wr_txn": 0,  # dram__sectors_write.sum

    # 그래프 스타일
    "color":        "#f0ba28",
    "marker":       "D",
    "size":         80,
    "label_prefix": "achieved",
}
# Ginkgo end ========================== 





# =========================
# ceiling & wall color
# =========================

Color_setting = {
        "c1_ceiling": "black",
        "L1_ceiling": "#a6e467",
        "L2_ceiling": "#8767e4",
        "DRAM_ceiling": "#e46767",
        "global_memory_wall": "#e48f67",
        "c2_ceiling": "black",
        "SMEM_ceiling": "#67e4e4",
        "shared_memory_wall": "#67e4e4",
}
# =========================
# gloabal_Memory + Cache (DRAM + L2 + L1) Roofline 계산
# =========================

def roofline_y_txn(bw_txn_per_s, peak_inst_per_s, ai_inst_per_txn):
    return np.minimum(peak_inst_per_s, bw_txn_per_s * ai_inst_per_txn)

def knee_ai_txn(peak_inst_per_s, bw_txn_per_s):
    return peak_inst_per_s / bw_txn_per_s

def wall_y_max(x, bw_txn_per_s):
    return min(PEAK_GIPS, bw_txn_per_s * x / 1e9)

def draw_global_cache_roofline(ax):
    ai = np.logspace(math.log10(AI_MIN), math.log10(AI_MAX), 800)
    y_dram = roofline_y_txn(BW_DRAM_TXN_S, PEAK_INST_PER_S, ai) / 1e9
    y_l2   = roofline_y_txn(BW_L2_TXN_S,   PEAK_INST_PER_S, ai) / 1e9
    y_l1   = roofline_y_txn(BW_L1_TXN_S,   PEAK_INST_PER_S, ai) / 1e9

    knee_dram = knee_ai_txn(PEAK_INST_PER_S, BW_DRAM_TXN_S)
    knee_l2   = knee_ai_txn(PEAK_INST_PER_S, BW_L2_TXN_S)
    knee_l1   = knee_ai_txn(PEAK_INST_PER_S, BW_L1_TXN_S)

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "axes.edgecolor": "#999999",
        "axes.linewidth": 0.8,
    })

    # 루프라인
    ax.loglog(ai, y_dram, color=Color_setting["DRAM_ceiling"], lw=2.6, label="DRAM roof (GTXN/s slope)")
    ax.loglog(ai, y_l2,   color=Color_setting["L2_ceiling"],   lw=2.6, label="L2 roof (GTXN/s slope)")
    ax.loglog(ai, y_l1,   color=Color_setting["L1_ceiling"],   lw=2.6, label="L1 roof (GTXN/s slope)")

    # ax.text(AI_MIN * 1.5, BW_L1_TXN_S * AI_MIN * 1.5 / 1e9,
    #         "L1", fontsize=10, color=Color_setting["L1_ceiling"], va='bottom')
    # ax.text(AI_MIN * 1.5, BW_L2_TXN_S * AI_MIN * 1.5 / 1e9,
    #         "L2", fontsize=10, color=Color_setting["L2_ceiling"], va='bottom')
    # ax.text(AI_MIN * 1.5, BW_DRAM_TXN_S * AI_MIN * 1.5 / 1e9,
    #         "DRAM", fontsize=10, color=Color_setting["DRAM_ceiling"], va='bottom')

    ax.hlines(PEAK_GIPS, xmin=knee_l1, xmax=AI_MAX,
            linestyles=(0, (6, 3)), color=Color_setting["c1_ceiling"], lw=2.4, label="Compute peak [GIPS]")
    # memory access pattern

    memory_walls = [1/32, 1/4, 1.0]  # stride 8, stride 1, stride 0 (float)
    labels = ["Stride 8(float)", "Stride 1(float)", "Stride 0(float)"]

    ymin = ax.get_ylim()[0]
    for x_wall, label in zip(memory_walls, labels):
        y_top = wall_y_max(x_wall, BW_L1_TXN_S)
        ax.plot([x_wall, x_wall], [ymin, y_top], color=Color_setting["global_memory_wall"], lw=2, ls="--")
        ax.text(x_wall * 1.05, y_top * 0.5, label,
            fontsize=9, color=Color_setting["global_memory_wall"],
            rotation=90, va='center', ha='left')

    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    # ax.hlines(  Point_result["warp_level_executed_instrucions"]/ Point_result["kernel_execution_time"] / 1e9, xmin=AI_MIN, xmax=AI_MAX,
    #             linestyles=(0, (3, 3)), color="#ff7f0e", lw=2.4, label="warp-level achieved [GIPS]")

    return fig, ax


def draw_shared_roofline(ax):
    ai = np.logspace(math.log10(AI_MIN), math.log10(AI_MAX), 800)
    y_shared = roofline_y_txn(BW_SMEM_TXN_S, PEAK_INST_PER_S, ai) / 1e9
    knee_shared = knee_ai_txn(PEAK_INST_PER_S, BW_SMEM_TXN_S)

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "axes.edgecolor": "#999999",
        "axes.linewidth": 0.8,
    })

    # 루프라인
    ax.loglog(ai, y_shared, color=Color_setting["SMEM_ceiling"], lw=2.6, label="Shared Memory roof (GTXN/s slope)")
    # ax.text(AI_MIN * 1.5, BW_SMEM_TXN_S * AI_MIN * 1.5 / 1e9,
    #         "Shared memory ceiling", fontsize=10, color=Color_setting["SMEM_ceiling"], va='bottom')
    ax.hlines(PEAK_GIPS, xmin=knee_shared, xmax=AI_MAX,
            linestyles=(0, (6, 3)), color=Color_setting["c2_ceiling"], lw=2.4, label="Compute peak [GIPS]")


    #shared memory wall
    memory_walls = [1/32, 1.0]  # stride 8, stride 1, stride 0 (float)
    labels = ["32-way bank conflict", "No bank conflict"]
    ymin = ax.get_ylim()[0]
    for x_wall, label in zip(memory_walls, labels):
        y_top = wall_y_max(x_wall, BW_SMEM_TXN_S)
        ax.plot([x_wall, x_wall], [ymin, y_top], color=Color_setting["shared_memory_wall"], lw=2, ls="--")
        ax.text(x_wall * 1.05, y_top * 0.5, label,
            fontsize=9, color=Color_setting["shared_memory_wall"],
            rotation=90, va='center', ha='left')

    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    return fig, ax


def plot_from_counters_global(ax, ach):
    t_sec        = float(ach["t_sec"])
    inst_sum     = float(ach["thread_level_executed_instructions"])

    ax.hlines(  ach["warp_level_executed_instructions"]/ ach["kernel_execution_time"] / 1e9, xmin=AI_MIN, xmax=AI_MAX,
                linestyles=(0, (3, 3)), color="black", ls='-.', lw=2.4, label="warp-level achieved [GIPS]")

    gips = (inst_sum / t_sec) / 1e9  # y값

    tx_dram = float(ach["dram_rd_txn"]) + float(ach["dram_wr_txn"])
    tx_l2   = float(ach["l2_rd_txn"])  + float(ach["l2_wr_txn"])
    tx_l1   = float(ach["gld_txn"])    + float(ach["gst_txn"])

    pts = []
    if tx_dram > 0: pts.append(("DRAM", inst_sum / tx_dram, gips))
    if tx_l2   > 0: pts.append(("L2",   inst_sum / tx_l2,   gips))
    if tx_l1   > 0: pts.append(("L1",   inst_sum / tx_l1,   gips))

    for level, ai_pt, gips_pt in pts:
        ax.scatter([ai_pt], [gips_pt],
                   s=ach.get("size", 56),
                   marker=ach.get("marker", "D"),
                   c=ach.get("color", "#444"),
                   edgecolors="white", linewidths=0.9, zorder=7)
        ax.annotate(f'{ach.get("label_prefix","achieved")}@{level}',
                    (ai_pt, gips_pt),
                    fontsize=9, ha='left', va='bottom')

def plot_from_counters_shared(ax, ach):
    t_sec        = float(ach["t_sec"])
    inst_sum     = float(ach["thread_level_executed_instructions"])

    ax.hlines(  ach["warp_level_executed_instructions"]/ ach["kernel_execution_time"] / 1e9, xmin=AI_MIN, xmax=AI_MAX,
                linestyles=(0, (3, 3)), color="black", ls='-.', lw=2.4, label="warp-level achieved [GIPS]")

    gips = (inst_sum / t_sec) / 1e9  # y값

    tx_smem = float(ach["smem_ld_txn"]) + float(ach["smem_st_txn"])
    
    pts = []
    if tx_smem > 0: pts.append(("SMEM", inst_sum / tx_smem, gips))

    for level, ai_pt, gips_pt in pts:
        ax.scatter([ai_pt], [gips_pt],
                   s=ach.get("size", 56),
                   marker=ach.get("marker", "D"),
                   c=ach.get("color", "#444"),
                   edgecolors="white", linewidths=0.9, zorder=7)
        ax.annotate(f'{ach.get("label_prefix","achieved")}@{level}',
                    (ai_pt, gips_pt),
                    fontsize=9, ha='left', va='bottom')


cuSPARSE_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
draw_global_cache_roofline(ax1)
draw_shared_roofline(ax2)

for ach in (ACH_FROM_COUNTERS_1, ACH_FROM_COUNTERS_2, ACH_FROM_COUNTERS_3):
    plot_from_counters_global(ax1, ach)
    plot_from_counters_shared(ax2, ach)

cuSPARSE_fig.savefig("/workspace/ICTC/tmp/cuSPARSE_roofline.png", dpi=300)

Ginkgo_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
draw_global_cache_roofline(ax1)
draw_shared_roofline(ax2)

for ach in (ACH_FROM_COUNTERS_4, ACH_FROM_COUNTERS_5, ACH_FROM_COUNTERS_6):
    plot_from_counters_global(ax1, ach)
    plot_from_counters_shared(ax2, ach)

Ginkgo_fig.savefig("/workspace/ICTC/tmp/Ginkgo_roofline.png", dpi=300)