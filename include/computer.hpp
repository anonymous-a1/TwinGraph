#pragma once

#include "distance.hpp"
#include "memory.hpp"
#include "common.h"
#include <iostream>


namespace Computer {


template <typename data_type>
struct SearchComputer {
    using dst_t = Distance::distance_t<data_type>;
    
    const data_type * const query;
    const Memory::AlignedMemoryManager<data_type> & data;
    const Distance::DstFuncPtr<data_type> distfunc;
    
    const int prefetch_len;
    const int prefetch_size;
    const int graph_prefetch_len;

    SearchComputer(const data_type * const query0, const Memory::AlignedMemoryManager<data_type> & data0, const Distance::DstFuncPtr<data_type> distfunc0)
    : query(query0), data(data0), distfunc(distfunc0), 
    prefetch_len(std::min(16384UL / data0.sizeRow, 16UL)), 
    prefetch_size(std::min(16384UL / data0.sizeRow, data0.sizeRow / MMALIGN)), 
    graph_prefetch_len(std::min(16384UL / data0.sizeRow, 10UL)){}

    inline dst_t operator() (const idx_t id) const { 
        return distfunc(query, data[id]);
    }

    inline void prefetch(const idx_t id) const {
        Memory::prefetchHelper(reinterpret_cast<char *>(data[id]), prefetch_size);
    }
};

template <typename data_type>
struct CommonComputer{
    using dst_t = Distance::distance_t<data_type>;

    const Memory::AlignedMemoryManager<data_type> & data;
    const Distance::DstFuncPtr<data_type> distfunc;

    CommonComputer(const Memory::AlignedMemoryManager<data_type> & data0, const Distance::DstFuncPtr<data_type> distfunc0)
    : data(data0), distfunc(distfunc0){}

    inline dst_t operator() (const idx_t x, const idx_t y) const { 
        return distfunc(data[x], data[y]);
    }     

    inline void prefetch(const idx_t id) const {
        data.prefetch(id);
    }
};


}; 
