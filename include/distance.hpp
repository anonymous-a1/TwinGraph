#pragma once 

#include <unistd.h>
#include <cstdio>
#include <cstdint>
#if defined(__x86_64__)||(__i386__)
#include <immintrin.h>
#elif defined(__aarch64__)||(__arm__)
#if defined __ARM_NEON
#include <arm_neon.h>
#endif
#if defined __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#endif 
#include <type_traits>
#include <iostream>

#ifndef THROW
#define THROW(ERROR_MESSAGE) do { \
    char buf[2048]; \
    int n = snprintf(buf, sizeof(buf), "\033[31m[ERROR]\033[0m%s", ERROR_MESSAGE); \
    if (n > 0 && buf[n-1] != '\n') { \
        buf[n] = '\n'; \
        buf[n+1] = '\0'; \
        n++; \
    } \
    write(2, buf, n); \
    _exit(-1); \
} while (0)
#endif

namespace Distance{

enum MetricType {ERRORC,L2,IP};

template<typename T> struct dstTypeHelper {};
template<> struct dstTypeHelper<float> { using type = float; };
template<> struct dstTypeHelper<int8_t> { using type = int32_t; };
template<> struct dstTypeHelper<uint8_t> { using type = int32_t; };

template <typename T>
using distance_t = dstTypeHelper<T>::type;

template <typename T>
using DstFuncPtr = distance_t<T> (*) (const T *, const T *);



template<typename T, int Dim>
[[gnu::always_inline]] distance_t<T> basicL2(const T * __restrict x, const T * __restrict y){
    distance_t<T> res = 0;
    for (int i = 0; i < Dim; ++i){
        auto diff = static_cast<distance_t<T>>(*x) - static_cast<distance_t<T>>(*y);
        res += diff * diff;
        x++; 
        y++;
    }
    return res;
}

template<typename T, int Dim>
[[gnu::always_inline]] distance_t<T> basicIP(const T * __restrict x, const T * __restrict y){
    distance_t<T> res = 0;
    for (int i = 0; i < Dim; ++i){
        res += static_cast<distance_t<T>>(*x) * static_cast<distance_t<T>>(*y);
        x++; 
        y++;
    }
    return 1.0 - res;
}

template<typename T, int Dim>
requires std::is_same_v<T, float>
float simdL2(const float * __restrict x, const float * __restrict y) {
#if defined __AVX512F__
    __m512 d2_vec = _mm512_setzero();
    __m512 x_vec, y_vec;
    for(int i = 0; i < Dim; i+=16){
        x_vec = _mm512_load_ps(x + i);
        y_vec = _mm512_load_ps(y + i);
        __m512 d_vec = _mm512_sub_ps(x_vec, y_vec);
        d2_vec = _mm512_fmadd_ps(d_vec, d_vec, d2_vec);
    }
    return _mm512_reduce_add_ps(d2_vec);
#elif defined __ARM_FEATURE_SVE
    svfloat32_t d2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svbool_t pg_vec = svwhilelt_b32(0, Dim);
    for(int i = 0; i < Dim; i += svcntw()){
        svfloat32_t x_vec = svld1_f32(pg_vec, x + i);
        svfloat32_t y_vec = svld1_f32(pg_vec, y + i);
        svfloat32_t d_vec = svsub_f32_x(pg_vec, x_vec, y_vec);
        d2_vec = svmla_f32_x(pg_vec, d2_vec, d_vec, d_vec);
    }
    return (float)svaddv_f32(svptrue_b32(), d2_vec);
#elif defined __AVX2__
    __m256 d2_vec = _mm256_setzero_ps();
    __m256 x_vec, y_vec;
    for (int i = 0; i < Dim; i += 8) {
        x_vec = _mm256_load_ps(x + i);
        y_vec = _mm256_load_ps(y + i);
        __m256 d_vec = _mm256_sub_ps(x_vec, y_vec);
        d2_vec = _mm256_fmadd_ps(d_vec, d_vec, d2_vec);
    }
    __m256 sumh = _mm256_hadd_ps(d2_vec, d2_vec);
    sumh = _mm256_hadd_ps(sumh, sumh);
    auto res = _mm_add_ps(_mm256_castps256_ps128(sumh), _mm256_extractf128_ps(sumh, 1));
    return _mm_cvtss_f32(res);
#elif defined __ARM_NEON
    float32x4_t d2_vec = vdupq_n_f32(0);
    for (int i = 0; i < Dim; i += 4) {
        float32x4_t x_vec = vld1q_f32(x + i);
        float32x4_t y_vec = vld1q_f32(y + i);
        float32x4_t d_vec = vsubq_f32(x_vec, y_vec);
        d2_vec = vfmaq_f32(d2_vec, d_vec, d_vec);
    }
    return (float)vaddvq_f32(d2_vec);
#else
    return basicL2<float, Dim>(x, y);
#endif
}

template<typename T, int Dim>
requires std::is_same_v<T, uint8_t>
int32_t simdL2(const uint8_t * __restrict x, const uint8_t * __restrict y){
#if defined __AVX2__
    __m256i d2_high_vec = _mm256_setzero_si256();
    __m256i d2_low_vec = _mm256_setzero_si256();
    for (int i = 0; i < Dim; i += 32) {
        __m256i x_vec = _mm256_load_si256((__m256i const *)(x + i));
        __m256i y_vec = _mm256_load_si256((__m256i const *)(y + i));

        __m256i x_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(x_vec));
        __m256i x_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(x_vec, 1));
        __m256i y_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_vec));
        __m256i y_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(y_vec, 1));

        __m256i d_low = _mm256_sub_epi16(x_low, y_low);
        __m256i d_high = _mm256_sub_epi16(x_high, y_high);
        __m256i d2_low_part = _mm256_madd_epi16(d_low, d_low);
        __m256i d2_high_part = _mm256_madd_epi16(d_high, d_high);

        d2_low_vec = _mm256_add_epi32(d2_low_vec, d2_low_part);
        d2_high_vec = _mm256_add_epi32(d2_high_vec, d2_high_part);
    }
    __m256i d2_vec = _mm256_add_epi32(d2_low_vec, d2_high_vec);
    __m128i d2_sum = _mm_add_epi32(_mm256_extracti128_si256(d2_vec, 0), _mm256_extracti128_si256(d2_vec, 1));
    d2_sum = _mm_hadd_epi32(d2_sum, d2_sum);
    d2_sum = _mm_hadd_epi32(d2_sum, d2_sum);
    return (int32_t)_mm_extract_epi32(d2_sum, 0);
#elif defined __ARM_NEON
    int32x4_t d2_vec = vdupq_n_s32(0);
    for(int i = 0; i < Dim; i += 16){
        uint8x16_t x_vec = vld1q_u8(x + i);
        uint8x16_t y_vec = vld1q_u8(y + i);
        int16x8_t x_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(x_vec)));
        int16x8_t x_high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(x_vec)));
        int16x8_t y_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_vec)));
        int16x8_t y_high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y_vec)));

        int16x8_t d_low = vsubq_s16(x_low, y_low);
        int16x8_t d_high = vsubq_s16(x_high, y_high);

        int32x4_t d2_low_1 = vmull_s16(vget_low_s16(d_low), vget_low_s16(d_low));
        int32x4_t d2_low_2 = vmull_s16(vget_high_s16(d_low), vget_high_s16(d_low));
        int32x4_t d2_high_1 = vmull_s16(vget_low_s16(d_high), vget_low_s16(d_high));
        int32x4_t d2_high_2 = vmull_s16(vget_high_s16(d_high), vget_high_s16(d_high));

        int32x4_t d2_low = vaddq_s32(d2_low_1, d2_low_2);
        int32x4_t d2_high = vaddq_s32(d2_high_1, d2_high_2);

        d2_vec = vaddq_s32(d2_vec, d2_low);
        d2_vec = vaddq_s32(d2_vec, d2_high);
    }
    return (int32_t)vaddvq_s32(d2_vec);
#else
    return basicL2<uint8_t, Dim>(x, y);
#endif
}

template<typename T, int Dim>
requires std::is_same_v<T, int8_t>
int32_t simdL2(const int8_t * __restrict x, const int8_t * __restrict y){
#if defined __AVX2__
    __m256i d2_high_vec = _mm256_setzero_si256();
    __m256i d2_low_vec = _mm256_setzero_si256();
    for (int i = 0; i < Dim; i += 32) {
        __m256i x_vec = _mm256_load_si256((__m256i const *)(x + i));
        __m256i y_vec = _mm256_load_si256((__m256i const *)(y + i));

        __m256i x_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_vec));
        __m256i x_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_vec, 1));
        __m256i y_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(y_vec));
        __m256i y_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(y_vec, 1));

        __m256i d_low = _mm256_sub_epi16(x_low, y_low);
        __m256i d_high = _mm256_sub_epi16(x_high, y_high);
        __m256i d2_low_part = _mm256_madd_epi16(d_low, d_low);
        __m256i d2_high_part = _mm256_madd_epi16(d_high, d_high);

        d2_low_vec = _mm256_add_epi32(d2_low_vec, d2_low_part);
        d2_high_vec = _mm256_add_epi32(d2_high_vec, d2_high_part);
    }
    __m256i d2_vec = _mm256_add_epi32(d2_low_vec, d2_high_vec);
    __m128i d2_sum = _mm_add_epi32(_mm256_extracti128_si256(d2_vec, 0), _mm256_extracti128_si256(d2_vec, 1));
    d2_sum = _mm_hadd_epi32(d2_sum, d2_sum);
    d2_sum = _mm_hadd_epi32(d2_sum, d2_sum);
    return (int32_t)_mm_extract_epi32(d2_sum, 0);
#elif defined __ARM_NEON
    int32x4_t d2_vec = vdupq_n_s32(0);
    for(int i = 0; i < Dim; i += 16){
        int8x16_t x_vec = vld1q_s8(x + i);
        int8x16_t y_vec = vld1q_s8(y + i);
        int16x8_t x_low = vreinterpretq_s16_u16(vmovl_s8(vget_low_s8(x_vec)));
        int16x8_t x_high = vreinterpretq_s16_u16(vmovl_s8(vget_high_s8(x_vec)));
        int16x8_t y_low = vreinterpretq_s16_u16(vmovl_s8(vget_low_s8(y_vec)));
        int16x8_t y_high = vreinterpretq_s16_u16(vmovl_s8(vget_high_s8(y_vec)));

        int16x8_t d_low = vsubq_s16(x_low, y_low);
        int16x8_t d_high = vsubq_s16(x_high, y_high);

        int32x4_t d2_low_1 = vmull_s16(vget_low_s16(d_low), vget_low_s16(d_low));
        int32x4_t d2_low_2 = vmull_s16(vget_high_s16(d_low), vget_high_s16(d_low));
        int32x4_t d2_high_1 = vmull_s16(vget_low_s16(d_high), vget_low_s16(d_high));
        int32x4_t d2_high_2 = vmull_s16(vget_high_s16(d_high), vget_high_s16(d_high));

        int32x4_t d2_low = vaddq_s32(d2_low_1, d2_low_2);
        int32x4_t d2_high = vaddq_s32(d2_high_1, d2_high_2);

        d2_vec = vaddq_s32(d2_vec, d2_low);
        d2_vec = vaddq_s32(d2_vec, d2_high);
    }
    return (int32_t)vaddvq_s32(d2_vec);
#else
    return basicL2<int8_t, Dim>(x, y);
#endif
}

template<typename T, int Dim>
requires std::is_same_v<T, uint16_t>
int32_t simdL2(const uint16_t * __restrict x, const uint16_t * __restrict y) {
#if defined __AVX2__    
    __m256i d2_vec = _mm256_setzero_si256();
    for (int i = 0; i < Dim; i += 16) {
        __m256i x_vec = _mm256_load_si256((__m256i const *)(x + i));
        __m256i y_vec = _mm256_load_si256((__m256i const *)(y + i));

        __m256i x_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(x_vec));
        __m256i x_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(x_vec, 1));
        __m256i y_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(y_vec));
        __m256i y_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(y_vec, 1));

        __m256i d_low = _mm256_sub_epi32(x_low, y_low);
        __m256i d_high = _mm256_sub_epi32(x_high, y_high);
        __m256i d2_low = _mm256_mullo_epi32(d_low, d_low);  
        __m256i d2_high = _mm256_mullo_epi32(d_high, d_high);  

        d2_vec = _mm256_add_epi32(d2_vec, d2_low);
        d2_vec = _mm256_add_epi32(d2_vec, d2_high);
    }
    __m128i d2_sum = _mm_add_epi32(_mm256_extracti128_si256(d2_vec, 0), _mm256_extracti128_si256(d2_vec, 1));
    d2_sum = _mm_hadd_epi32(d2_sum, d2_sum);
    d2_sum = _mm_hadd_epi32(d2_sum, d2_sum);
    return _mm_extract_epi32(d2_sum, 0);
#elif defined __ARM_NEON
    int32x4_t d2_vec = vdupq_n_s32(0);
    uint16x8_t x_vec, y_vec;
    for (int i = 0; i < Dim; i += 8) {
        x_vec = vld1q_u16(x + i);
        y_vec = vld1q_u16(y + i);
        int16x8_t diff_vec = vreinterpretq_s16_u16(vsubq_u16(x_vec, y_vec));        
        int32x4_t diff_lo = vmull_s16(vget_low_s16(diff_vec), vget_low_s16(diff_vec));  
        int32x4_t diff_hi = vmull_s16(vget_high_s16(diff_vec), vget_high_s16(diff_vec));
        d2_vec = vaddq_s32(d2_vec, diff_lo);
        d2_vec = vaddq_s32(d2_vec, diff_hi);
    }
    return vaddvq_s32(d2_vec);
#else
    return basicL2<uint16_t, Dim>(x, y);
#endif
}

template<typename T, int Dim>
requires std::is_same_v<T, int16_t>
int32_t simdL2(const int16_t * __restrict x, const int16_t * __restrict y) {
#if defined __AVX2__    
    __m256i d2_vec = _mm256_setzero_si256();
    for (int i = 0; i < Dim; i += 16) {
        __m256i x_vec = _mm256_load_si256((__m256i const *)(x + i));
        __m256i y_vec = _mm256_load_si256((__m256i const *)(y + i));

        __m256i x_low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(x_vec));
        __m256i x_high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(x_vec, 1));
        __m256i y_low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(y_vec));
        __m256i y_high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(y_vec, 1));

        __m256i d_low = _mm256_sub_epi32(x_low, y_low);
        __m256i d_high = _mm256_sub_epi32(x_high, y_high);
        __m256i d2_low = _mm256_mullo_epi32(d_low, d_low);  
        __m256i d2_high = _mm256_mullo_epi32(d_high, d_high);  

        d2_vec = _mm256_add_epi32(d2_vec, d2_low);
        d2_vec = _mm256_add_epi32(d2_vec, d2_high);
    }
    __m128i d2_sum = _mm_add_epi32(_mm256_extracti128_si256(d2_vec, 0), _mm256_extracti128_si256(d2_vec, 1));
    d2_sum = _mm_hadd_epi32(d2_sum, d2_sum);
    d2_sum = _mm_hadd_epi32(d2_sum, d2_sum);
    return _mm_extract_epi32(d2_sum, 0);
#elif defined __ARM_NEON
    int32x4_t d2_vec = vdupq_n_s32(0);
    int16x8_t x_vec, y_vec;
    for (int i = 0; i < Dim; i += 8) {
        x_vec = vld1q_s16(x + i);
        y_vec = vld1q_s16(y + i);
        int16x8_t diff_vec = vreinterpretq_s16_s16(vsubq_s16(x_vec, y_vec));        
        int32x4_t diff_lo = vmull_s16(vget_low_s16(diff_vec), vget_low_s16(diff_vec));  
        int32x4_t diff_hi = vmull_s16(vget_high_s16(diff_vec), vget_high_s16(diff_vec));
        d2_vec = vaddq_s32(d2_vec, diff_lo);
        d2_vec = vaddq_s32(d2_vec, diff_hi);
    }
    return vaddvq_s32(d2_vec);
#else
    return basicL2<int16_t, Dim>(x, y);
#endif
}

template<typename T, int Dim>
requires std::is_same_v<T, float>
float simdIP(const float * __restrict x, const float * __restrict y){
#if defined __AVX512F__
    __m512 ip_vec = _mm512_setzero_ps();
    __m512 x_vec, y_vec;
    for(int i = 0; i < Dim; i+=16){
        x_vec = _mm512_load_ps(x + i);
        y_vec = _mm512_load_ps(y + i);
        ip_vec = _mm512_fmadd_ps(x_vec, y_vec, ip_vec);
    }
    return 1.0f - _mm512_reduce_add_ps(ip_vec);    
#elif defined __AVX2__
    __m256 ip_vec = _mm256_setzero_ps();
    __m256 x_vec, y_vec;
    for (int i = 0; i < Dim; i += 8) {
        x_vec = _mm256_load_ps(x + i);
        y_vec = _mm256_load_ps(y + i);
        ip_vec = _mm256_fmadd_ps(x_vec, y_vec, ip_vec);
    }
    __m256 sumh = _mm256_hadd_ps(ip_vec, ip_vec);
    sumh = _mm256_hadd_ps(sumh, sumh);
    auto res = _mm_add_ps(_mm256_castps256_ps128(sumh), _mm256_extractf128_ps(sumh, 1));
    return 1.0f - _mm_cvtss_f32(res);
#else
    return basicIP<float, Dim>(x, y);
#endif
}

template <typename T, int Dim = 1>
auto getL2DstFunc(int d){
    if constexpr (Dim <= MAXDIM){
        if(Dim == d){
            return simdL2<T, Dim>;
        }else{
            return getL2DstFunc<T, Dim + 1>(d);
        }
    }else{
        THROW("getL2DstFunc: dimensional is unsupported");
        return simdL2<T, 0>;
    }
}

template <typename T, int Dim = 1>
auto getIPDstFunc(int d){
    THROW("getIPDstFunc: data type is unsupported for IP");
    return basicIP<T, 0>;
}

template <typename T, int Dim = 1>
requires std::is_same_v<T, float>
auto getIPDstFunc(int d){
    if constexpr (Dim <= MAXDIM){
        if(Dim == d){
            return simdIP<T, Dim>;
        }else{
            return getIPDstFunc<T, Dim + 1>(d);
        }
    }else{
        THROW("getIPDstFunc: dimensional is unsupported");
        return simdIP<T, 0>;    
    }
}


};