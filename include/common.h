#pragma once

#include <unistd.h>
#include <cstdio>
#include <mutex>
#include "boost/smart_ptr/detail/spinlock.hpp"
typedef boost::detail::spinlock Lock;
typedef std::lock_guard<Lock> LockGuard;

typedef int32_t idx_t;

enum DataType {ERRORD,FLOAT,INT8,INT16,UINT8,UINT16,INT32};

template<typename T> inline constexpr DataType datatype_v = DataType::ERRORD;
template<> inline constexpr DataType datatype_v<float> = DataType::FLOAT;
template<> inline constexpr DataType datatype_v<int8_t> = DataType::INT8;
template<> inline constexpr DataType datatype_v<int16_t> = DataType::INT16;
template<> inline constexpr DataType datatype_v<uint8_t> = DataType::UINT8;
template<> inline constexpr DataType datatype_v<uint16_t> = DataType::UINT16;
template<> inline constexpr DataType datatype_v<int32_t> = DataType::INT32;

template<typename T> inline constexpr char * datatype_s = "ERRORTYPE";
template<> inline constexpr char * datatype_s<float> = "float";
template<> inline constexpr char * datatype_s<int8_t> = "int8";
template<> inline constexpr char * datatype_s<int16_t> = "int16";
template<> inline constexpr char * datatype_s<uint8_t> = "uint8";
template<> inline constexpr char * datatype_s<uint16_t> = "uint16";
template<> inline constexpr char * datatype_s<int32_t> = "int32";

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