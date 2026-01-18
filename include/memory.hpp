#pragma once

#include <cstdlib>
#include <iostream>
#include <memory>
#include <cmath>
#include <assert.h>
#ifndef MMALIGN 
#define MMALIGN 64
#endif
constexpr int mmalignp = log(MMALIGN)/log(2);

namespace Memory{

inline void prefetchHelper(char *ptr, const int32_t num_lines) {
    switch (num_lines) {
    default: [[fallthrough]];
    case 16:    __builtin_prefetch(ptr, 0, 3);  ptr += 64; [[fallthrough]];
    case 15:    __builtin_prefetch(ptr, 0, 3);  ptr += 64; [[fallthrough]];
    case 14:    __builtin_prefetch(ptr, 0, 3);  ptr += 64; [[fallthrough]];
    case 13:    __builtin_prefetch(ptr, 0, 3);  ptr += 64; [[fallthrough]];
    case 12:    __builtin_prefetch(ptr, 0, 3);  ptr += 64; [[fallthrough]];
    case 11:    __builtin_prefetch(ptr, 0, 3);  ptr += 64; [[fallthrough]];
    case 10:    __builtin_prefetch(ptr, 0, 3);  ptr += 64; [[fallthrough]];
    case  9:    __builtin_prefetch(ptr, 0, 3);  ptr += 64; [[fallthrough]];
    case  8:    __builtin_prefetch(ptr, 0, 3);  ptr += 64; [[fallthrough]];
    case  7:    __builtin_prefetch(ptr, 0, 3);  ptr += 64; [[fallthrough]];
    case  6:    __builtin_prefetch(ptr, 0, 3);  ptr += 64; [[fallthrough]];
    case  5:    __builtin_prefetch(ptr, 0, 3);  ptr += 64; [[fallthrough]];
    case  4:    __builtin_prefetch(ptr, 0, 3);  ptr += 64; [[fallthrough]];
    case  3:    __builtin_prefetch(ptr, 0, 3);  ptr += 64; [[fallthrough]];
    case  2:    __builtin_prefetch(ptr, 0, 3);  ptr += 64; [[fallthrough]];
    case  1:    __builtin_prefetch(ptr, 0, 3);  break;
    case  0:    break;
    }
}

template <typename T, bool ifListLength = false, size_t alignSz = MMALIGN>
struct AlignedMemoryManager{
    bool aligned_tag{false};
    std::shared_ptr<char> mass{nullptr};
    size_t N{0};
    size_t dim{0};
    size_t sizeRow{0};
    int prefetch_size{0};
    char * data{nullptr};   // for faster access

    AlignedMemoryManager() = default;
    AlignedMemoryManager(size_t N0, size_t dim0, size_t size_per_row0 = 0, void * mass0 = nullptr){
        realloc(N0, dim0, size_per_row0, mass0);
    }
    ~AlignedMemoryManager(){}

    void set_zero(){
        memset(mass.get(), 0, N * sizeRow);
    }
    void prune(){
        mass.reset();
        data = nullptr;
        aligned_tag = false;
        N = 0;
        dim = 0;
        sizeRow = 0;
    }
    void realloc(size_t N0, size_t dim0, size_t size_per_row0 = 0, void * mass0 = nullptr){
        if(mass.get() && !mass0) {
            if(N == N0 && dim == dim0)
            if(sizeRow == (sizeof(T) * dim + ifListLength * sizeof(uint32_t) + alignSz - 1) / alignSz * alignSz){
                return;
            }
        }
        prune();
        N = N0;
        dim = dim0;
    
        if(mass0){
            sizeRow = size_per_row0;
            prefetch_size = (sizeRow + alignSz - 1) / alignSz;
            mass = std::shared_ptr<char>(static_cast<char *>(mass0));
        }
        else{
            sizeRow = (sizeof(T) * dim + ifListLength * sizeof(uint32_t) + alignSz - 1) / alignSz * alignSz; 
            prefetch_size = (sizeRow + alignSz - 1) / alignSz;            
            mass = std::shared_ptr<char>((char *)std::aligned_alloc(alignSz, N * sizeRow)); 
            aligned_tag = true;            
        }
        data = mass.get();
    }

    inline void swap(AlignedMemoryManager & obj){
        std::swap(N, obj.N);
        std::swap(dim, obj.dim);
        std::swap(sizeRow, obj.sizeRow);
        std::swap(prefetch_size, obj.prefetch_size);
        std::swap(aligned_tag, obj.aligned_tag);
        mass.swap(obj.mass);
        std::swap(data, obj.data);
    }
    inline void operator= (const AlignedMemoryManager & obj){
        N = obj.N;
        dim = obj.dim;
        sizeRow = obj.sizeRow;
        prefetch_size = obj.prefetch_size;
        aligned_tag = obj.aligned_tag;
        mass = obj.mass;
        data = mass.get();
    }
    
    inline T * get() const { return reinterpret_cast<T *>(mass.get()); }
    inline T * getrow(size_t idx) const { return reinterpret_cast<T *>(mass.get() + idx * sizeRow); }
    inline T * operator[] (size_t idx) const { return reinterpret_cast<T *>(data + idx * sizeRow + ifListLength * sizeof(uint32_t)); }
    
    inline const uint32_t getLength(size_t idx) const {
        if constexpr (!ifListLength) return (uint32_t)dim;
        return *reinterpret_cast<const uint32_t *>(data + idx * sizeRow);
    }
    inline bool setLength(size_t idx, uint32_t length){
        if constexpr (!ifListLength) return false;
        *reinterpret_cast<uint32_t *>(mass.get() + idx * sizeRow) = length;
        return true;
    }
    void prefetch(size_t idx) const {
        prefetchHelper((char *)(*this)[idx], prefetch_size);
    }
};


};  