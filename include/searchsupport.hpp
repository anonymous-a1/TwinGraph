#pragma once

#include <cstring>
#include <mutex>
#include <deque>
#include <limits>
#include <vector>
#include <omp.h>
#include <iostream>
#include <memory>
#include "neighbor.hpp"

namespace nns{

using vl_type = uint16_t;

struct VisitedList {
    vl_type curV;
    vl_type * mass;
    size_t size;
    VisitedList(vl_type * mass0, size_t size0) : mass(mass0), size(size0), curV(1) {}

    inline void reset(){
        if(curV == std::numeric_limits<vl_type>::max()){
            memset(mass, 0, size * sizeof(vl_type));
            curV = 1;
        }else{
            curV++;
        }
    }
    ~VisitedList(){}
};


template <typename dist_t>
class CandidateList {

private:
    constexpr static int kMask = 2147483647;
    uint32_t size_{0};
    uint32_t cur_{0};
    uint32_t capacity_;
    std::vector<Neighbor<dist_t>> data_;

    inline idx_t get_idx(idx_t id) const {
        return id & kMask; 
    }
    inline void set_checked(idx_t &id){
        id |= 1 << 31;
    }
    inline bool is_checked(idx_t id) const {
        return id >> 31 & 1;
    }    
    int32_t find_bsearch(Neighbor<dist_t> cand) {
        int32_t lo = 0, hi = size_;
        while (lo < hi) {
            int32_t mid = (lo + hi) / 2;
            if (Neighbor<dist_t>(data_[mid].idx & kMask, data_[mid].dst) < cand) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    } 

public:
    CandidateList(uint32_t capacity) : capacity_(capacity), data_(capacity_ + 1) {}

    bool insert(idx_t u, dist_t dist){
        if (size_ == capacity_ && dist >= data_[size_ - 1].dst) return false;
        int32_t lo = find_bsearch({u, dist});
        std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Neighbor<dist_t>));
        data_[lo] = {u, dist};
        if (size_ < capacity_) size_++;
        if (lo < cur_) cur_ = lo;
        return true;
    }
    idx_t pop(){
        idx_t ret = data_[cur_].idx;
        set_checked(data_[cur_].idx);
        while (cur_ < size_ && is_checked(data_[cur_].idx)) {
            cur_++;
        }
        return ret;
    }
    inline bool has_next() const {
        return cur_ < size_; 
    }
    inline idx_t idx(idx_t i) const {
        return get_idx(data_[i].idx); 
    }
    inline dist_t dst(idx_t i) const {
        return data_[i].dst; 
    }
    inline uint32_t size() const {
        return size_; 
    }
    inline uint32_t capacity() const {
        return capacity_; 
    }
    inline uint32_t cur() const {
        return cur_;
    }
    inline Neighbor<dist_t>& operator[] (int pos){
        return data_[pos];
    }
    void reset_checked(){
        for(int i = 0; i < size_; i++){
            data_[i].idx = get_idx(data_[i].idx);
        }
        cur_ = 0;
    }
    inline void reset(){
        size_ = 0;
    }    
    ~CandidateList(){}
};



class SearchSupporterManager {
 
private:
    size_t stride;
    size_t sizePerList;
    size_t numLists;
    std::vector<VisitedList> lists;
    std::shared_ptr<char> mass{nullptr};

public:
    template <typename dist_t>
    struct SearchSupporter {

        CandidateList<dist_t> candidate_list;
        VisitedList * visited_list;

        SearchSupporter(int efsearch, VisitedList * vlist) : candidate_list(efsearch), visited_list(vlist) {}

        inline bool is_visited(idx_t idx) const {
            return visited_list->mass[idx] == visited_list->curV;
        }
        inline void set_visited(idx_t idx){
            visited_list->mass[idx] = visited_list->curV;
        }
        inline size_t size() const {
            return candidate_list.size();
        }
        inline size_t capacity() const {
            return candidate_list.capacity();
        }
        inline int cur() const {
            return candidate_list.cur();
        }
        inline idx_t idx(int pos) const {
            return candidate_list.idx(pos);
        }
        inline dist_t dst(int pos) const {
            return candidate_list.dst(pos);
        }
        inline void reset_checked(){
            candidate_list.reset_checked();
        }
        inline Neighbor<dist_t>& operator[] (int pos){
            return candidate_list[pos];
        }
        inline bool insert(idx_t idx, dist_t dst){
            return candidate_list.insert(idx, dst);
        }
        inline idx_t pop(){
            return candidate_list.pop();
        }
        inline bool has_next() const {
            return candidate_list.has_next();
        }
        inline void prefetchVL(idx_t idx) const { 
            Memory::prefetchHelper((char *)(visited_list->mass + idx), 1);
        }
        ~SearchSupporter(){
            visited_list->reset();
        }
    };

    SearchSupporterManager() = default;
    SearchSupporterManager(size_t sizePerList0)
    : sizePerList(sizePerList0), numLists(omp_get_max_threads()){
        stride = (sizeof(vl_type) * sizePerList - 1 + 64) / 64 * 64;
        mass = std::shared_ptr<char>((char *)std::aligned_alloc(64, numLists * stride)); 
        memset(mass.get(), 0, numLists * stride);
        lists.reserve(numLists);
        for(int i = 0; i < numLists; i++){
            lists.emplace_back(VisitedList((vl_type *)(mass.get() + i * stride), sizePerList));
        }        
    }

    template <typename dist_t>
    inline SearchSupporter<dist_t> getSupporter(int efsearch){
        return SearchSupporter<dist_t>(efsearch, lists.data() + omp_get_thread_num());
    }

    ~SearchSupporterManager(){}
};


};