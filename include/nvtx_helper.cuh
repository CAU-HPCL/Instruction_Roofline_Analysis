#pragma once
#include <cstdint>
#include <nvToolsExt.h>

// palette (ARGB)
namespace palette {
    inline constexpr std::uint32_t blue     = 0xFF2980B9;
    inline constexpr std::uint32_t green    = 0xFF27AE60;
    inline constexpr std::uint32_t purple   = 0xFF8E44AD;
    inline constexpr std::uint32_t orange   = 0xFFF39C12;
    inline constexpr std::uint32_t red      = 0xFFE74C3C;
    inline constexpr std::uint32_t yellow   = 0xFFF1C40F;
    inline constexpr std::uint32_t teal     = 0xFF16A085;
    inline constexpr std::uint32_t gray     = 0xFF95A5A6;
    inline constexpr std::uint32_t darkgray = 0xFF34495E;
}

// push/pop/mark
inline void nvtx_push_color(const char* name, std::uint32_t argb){
    nvtxEventAttributes_t e{}; e.version=NVTX_VERSION; e.size=NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    e.colorType=NVTX_COLOR_ARGB; e.color=argb;
    e.messageType=NVTX_MESSAGE_TYPE_ASCII; e.message.ascii=name;
    nvtxRangePushEx(&e);
}
inline void nvtx_pop(){ nvtxRangePop(); }
inline void nvtx_mark(const char* name, std::uint32_t argb=palette::teal){
    nvtxEventAttributes_t e{}; e.version=NVTX_VERSION; e.size=NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    e.colorType=NVTX_COLOR_ARGB; e.color=argb;
    e.messageType=NVTX_MESSAGE_TYPE_ASCII; e.message.ascii=name;
    nvtxMarkEx(&e);
}
inline void nvtx_push_color_i64(const char* name, std::uint32_t argb, long long payload){
    nvtxEventAttributes_t e{}; e.version=NVTX_VERSION; e.size=NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    e.colorType=NVTX_COLOR_ARGB; e.color=argb;
    e.messageType=NVTX_MESSAGE_TYPE_ASCII; e.message.ascii=name;
    e.payloadType=NVTX_PAYLOAD_TYPE_INT64; e.payload.llValue=payload;
    nvtxRangePushEx(&e);
}
inline void nvtx_push_color_f64(const char* name, std::uint32_t argb, double payload){
    nvtxEventAttributes_t e{}; e.version=NVTX_VERSION; e.size=NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    e.colorType=NVTX_COLOR_ARGB; e.color=argb;
    e.messageType=NVTX_MESSAGE_TYPE_ASCII; e.message.ascii=name;
    e.payloadType=NVTX_PAYLOAD_TYPE_DOUBLE; e.payload.dValue=payload;
    nvtxRangePushEx(&e);
}

// RAII guard
struct NvtxRange {
    NvtxRange(const char* name, std::uint32_t argb){ nvtx_push_color(name,argb); }
    NvtxRange(const char* name, std::uint32_t argb, long long payload){ nvtx_push_color_i64(name,argb,payload); }
    ~NvtxRange(){ nvtx_pop(); }
};

// one-liners
#define NVTX_RANGE(name,color)            NvtxRange __nvtx_scope__(name,color)
#define NVTX_RANGE_I64(name,color,value)  NvtxRange __nvtx_scope__(name,color,value)
