#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <vector>
#include "common.h"
#include "memory.hpp"


namespace nns {  

template <typename dist_t = float> struct Neighbor {
    idx_t idx{0};
    dist_t dst{std::numeric_limits<dist_t>::max()};

    Neighbor() = default;
    Neighbor(idx_t idx0, dist_t dst0) : idx(idx0), dst(dst0) {}

    inline friend bool operator<(const Neighbor &x, const Neighbor &y) {
        return x.dst < y.dst || (x.dst == y.dst && x.idx < y.idx);
    }
    inline friend bool operator>(const Neighbor &x, const Neighbor &y) {
        return !(x < y);
    }
};

template <typename dist_t>
struct indNNList{
    using neighbor_t = Neighbor<dist_t>;
    neighbor_t * nnlist;
    bool * statlist;
    dist_t radius_max;    
    int16_t capasity;
    int16_t valid_range{0};    
    Lock lock;

    indNNList() = default;

    inline void operator = (indNNList other){
        assert(capasity == other.capasity);
        memcpy(nnlist, other.nnlist, sizeof(neighbor_t) * capasity);
        memcpy(statlist, other.statlist, sizeof(bool) * capasity);
        radius_max = other.radius_max;
        valid_range = other.valid_range;
    } 

    int16_t updateknnlist(const neighbor_t newnb){
        int lo = 0, hi = valid_range;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if(nnlist[mid].idx == newnb.idx) return capasity;
            if(nnlist[mid] < newnb){
                lo = mid + 1;
            }else{
                hi = mid;
            }
        }
        if(lo == capasity) return capasity;
        if(nnlist[lo].idx == newnb.idx) return capasity;
        std::memmove(&nnlist[lo + 1], &nnlist[lo], (valid_range - lo - (valid_range == capasity)) * sizeof(neighbor_t));
        std::memmove(&statlist[lo + 1], &statlist[lo], (valid_range - lo - (valid_range == capasity)) * sizeof(bool));
        nnlist[lo] = newnb;
        statlist[lo] = false;
        return lo;
    }
    
    inline bool parallelInsert(neighbor_t newnb){
        if (valid_range == capasity && newnb > nnlist[valid_range - 1]) return false; 
        LockGuard guard(lock);
        auto insert_pos = updateknnlist(newnb);

        if (insert_pos < capasity){
            if (valid_range < capasity){
                ++valid_range;
            }
            else radius_max = nnlist[valid_range - 1].dst;
        }
        return true;
    }

    inline void deleteNeighbor(int pos){
        if(pos >= valid_range) return;
        std::memmove(&nnlist[pos], &nnlist[pos + 1], (valid_range - pos - 1) * sizeof(neighbor_t));
        std::memmove(&statlist[pos], &statlist[pos + 1], (valid_range - pos - 1) * sizeof(bool));
        valid_range--;
        radius_max = std::numeric_limits<dist_t>::max(); 
    }
};

};

