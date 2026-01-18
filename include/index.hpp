#pragma once

#include <iostream>
#include <string>
#include <cstring>
#include <optional>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include "distance.hpp"
#include "memory.hpp"
#include "iomanager.hpp"
#include "computer.hpp"
#include "utils.hpp"
#include "searchsupport.hpp"
#include "common.h"
#include "loader.hpp"

namespace nns{

struct build_params{
    int K;
    int MaxIter;
    float alpha;
    int R;
    bool get_knng = false;
};

template <typename T>
class Index {

private:
    uint32_t search_k_{10};
    float efsearch_expand_{1.0f};
    int n_threads_{omp_get_max_threads()};
public:
    size_t n_comps{0};
    double sum{0};

    inline void set_search_k(uint32_t k0) { search_k_ = k0; }
    inline void set_expand(float e0) { efsearch_expand_ = e0; }
    inline void set_num_threads(int nthread) { n_threads_ = nthread; };
    inline int num_threads() { return n_threads_; }
public:
    using data_type = T;
    using dist_type = Distance::distance_t<data_type>;
    using edge_t = Neighbor<dist_type>;

private:
    Memory::AlignedMemoryManager<data_type> data_;
    Memory::AlignedMemoryManager<idx_t, true> graph_index_;
    std::vector<indNNList<dist_type>> knn_graph_;
    Memory::AlignedMemoryManager<edge_t> knn_pool_;
    bool * status_pool_;

    size_t max_vertex_{0};
    size_t cur_vertex_{0};   
    size_t del_vertex_{0};
    float alpha_{1.2f};
    int R_;
    int K_;    
    int beam_width_ = 4;
    float addvertex_expand_{2.0f};

    Distance::MetricType metric_;
    Distance::DstFuncPtr<data_type> distfunc_;
    SearchSupporterManager search_supporter_manager_;
    std::vector<idx_t> startPoints_;
    constexpr static size_t prefetch_size_data_ = 8;

    std::vector<idx_t> external_label_map_;
    std::unordered_map<idx_t, idx_t> internal_idx_map_;
    std::vector<char> available_tags_;

    Lock global_lock_;
    std::vector<Lock> graph_locks_;
    friend struct loader;

private:

    inline void init_distfunc(){
        if(metric_ == Distance::MetricType::L2) {
            distfunc_ = Distance::getL2DstFunc<data_type>(data_.dim);
        }else if(metric_ == Distance::MetricType::IP){
            distfunc_ = Distance::getIPDstFunc<data_type>(data_.dim);
        }else{
            THROW("unsupported metric");
        }
    }

    static inline void refreshIndexList(edge_t * naviglist, int index_n, edge_t * newlist, int new_n, int R, int & alive_n){
        int it_new = new_n - 1;
        int it_index = -1;
        while(it_new >= 0 && newlist[it_new].dst < 0) it_new--;
        if(it_new < 0) return;
        for(int i = 0; i < index_n; i++){
            if(naviglist[i].dst >= 0){
                naviglist[++it_index] = naviglist[i];
            }
        }
        while(alive_n > R){
            if(it_index >= 0 && naviglist[it_index] > newlist[it_new]) it_index--;
            else do it_new--; while(newlist[it_new].dst < 0);
            alive_n--;
        }

        int it_main = alive_n - 1;
        while(it_index >= 0 && it_new >= 0){
            if(naviglist[it_index] > newlist[it_new]){
                naviglist[it_main] = naviglist[it_index];
                it_index--;
            }else{
                naviglist[it_main] = newlist[it_new];
                do it_new--; while(newlist[it_new].dst < 0);
            }
            it_main--;
        }
        while(it_new >= 0){
            naviglist[it_main] = newlist[it_new];
            do it_new--; while(newlist[it_new].dst < 0);
            it_main--;
        }
    }

    void inDescent(auto & gindex, const auto & computer, int maxiter = 100){
        
        int smpN = K_ / 4;
        size_t sampled_num = std::numeric_limits<size_t>::max();
        size_t stop_num = std::sqrt(cur_vertex_ * K_);
        
        Memory::AlignedMemoryManager<edge_t, true> rvsSamplePools(cur_vertex_, K_);
        for (int n = 0; n < cur_vertex_; n++){
            rvsSamplePools.setLength(n, 0);
        }

        #ifdef DEBUG_VERBOSE
            int iter_num = 1;
        #endif

        while(maxiter-- && sampled_num > stop_num){
            #ifdef DEBUG_VERBOSE
                std::cout << "iter: " << iter_num++; 
            #endif
            sampled_num = 0;
            if(smpN + 2 <= K_) smpN += 2; else smpN = K_;
            #pragma omp parallel reduction(+:sampled_num)
            {
                std::vector<edge_t> sampleCache(1 + 2 * smpN);
                #pragma omp for 
                for(idx_t n = 0; n < cur_vertex_; n++){
                    auto & nbhood = knn_graph_[n];
                    int sampled_sz = 0;
                    for(int i = 0; i < nbhood.valid_range && sampled_sz < smpN; i++){
                        auto nb = nbhood.nnlist[i];
                        if(!nbhood.statlist[i]){
                            sampleCache[sampled_sz++] = {nb.idx, nb.dst};
                            nbhood.statlist[i] = true;
                            auto rvs_sz = rvsSamplePools.getLength(nb.idx);
                            if(rvs_sz < smpN && nb.dst > knn_graph_[nb.idx].radius_max){
                                LockGuard guard(graph_locks_[nb.idx]);
                                rvsSamplePools[nb.idx][rvs_sz] = {n, nb.dst};
                                rvsSamplePools.setLength(nb.idx, rvs_sz + 1);
                            }
                        }
                    }

                    {
                        int sampled_sz_bak = sampled_sz;
                        LockGuard guard(graph_locks_[n]); 
                        auto rvs_sz = rvsSamplePools.getLength(n);
                        for(int i = 0; i < rvs_sz; i++){
                            bool repeat = false;
                            for(int j = 0; !repeat && j < sampled_sz_bak; j++){
                                repeat = (rvsSamplePools[n][i].idx == sampleCache[j].idx);
                            }
                            if(!repeat){
                                sampleCache[sampled_sz++] = rvsSamplePools[n][i];
                            }
                        }
                        rvsSamplePools.setLength(n, 0);  
                    };
                    sampled_num += sampled_sz;
                    std::sort(sampleCache.begin(), sampleCache.begin() + sampled_sz);

                    auto threshold = nbhood.radius_max;
                    auto nvglist = gindex[n];
                    auto list_sz = gindex.getLength(n);
                    int alive_sz = list_sz + sampled_sz;

                    for(int i = 0; i < sampled_sz; i++){
                        auto & x = sampleCache[i];
                        for(int j = 0; j < i; j++){
                            auto & y = sampleCache[j];
                            if(x.idx == y.idx){
                                if(x.dst != -1){
                                    x.dst = -1;
                                    alive_sz--;
                                };
                                break;
                            }
                            if(y.dst < 0) continue;
                            auto dst_btw = computer(x.idx, y.idx);
                            knn_graph_[x.idx].parallelInsert({y.idx, dst_btw});
                            knn_graph_[y.idx].parallelInsert({x.idx, dst_btw});
                            if(x.dst < 0) continue;
                            if(alpha_ * dst_btw < x.dst){
                                x.dst = -1;
                                alive_sz--;
                            }
                        }
                        for(int j = 0; j < list_sz; j++){
                            auto & y = nvglist[j];
                            if(x.idx == y.idx){
                                if(x.dst != -1){
                                    x.dst = -1;
                                    alive_sz--;
                                };
                                break;
                            }
                            if(y.dst < 0) continue;
                            auto dst_btw = computer(x.idx, y.idx);
                            knn_graph_[x.idx].parallelInsert({y.idx, dst_btw});
                            knn_graph_[y.idx].parallelInsert({x.idx, dst_btw});
                            if(x.dst < 0) continue;
                            if(x.dst >= y.dst){
                                if(alpha_ * dst_btw < x.dst){
                                    x.dst = -1;
                                    alive_sz--;
                                }
                            }else{
                                if(alpha_ * dst_btw < y.dst){
                                    y.dst = -1;
                                    alive_sz--;
                                }
                            }
                        }
                    }
                    refreshIndexList(nvglist, list_sz, sampleCache.data(), sampled_sz, R_, alive_sz);

                    for(int i = 0; i < alive_sz; i++){
                        if(nvglist[i].dst > threshold){
                            alive_sz = i;
                            break;
                        } 
                    }
                    gindex.setLength(n, alive_sz);

                }
            };
            #ifdef DEBUG_VERBOSE
                std::cout << ", sampled num: " << sampled_num << std::endl;
            #endif    
        }  
    }    

    void bestFirstSearch(auto & support, auto & computer) const {
        while(support.has_next()) { 
            auto cur_id = support.pop();
            auto list_sz = graph_index_.getLength(cur_id);
            const auto nbhood = graph_index_[cur_id];

            #ifndef NOPREFETCH
                graph_index_.prefetch(cur_id);  
                for(int i = 0; i < computer.prefetch_len; i++){
                    computer.prefetch(nbhood[i]);                
                    support.prefetchVL(nbhood[i]);
                }
            #endif
            for(int i = 0; i < list_sz; i++){
                auto candidate = nbhood[i];
                #ifndef NOPREFETCH
                    if(i + computer.prefetch_len < list_sz) {
                        computer.prefetch(nbhood[i + computer.prefetch_len]);
                        support.prefetchVL(nbhood[i + computer.prefetch_len]);
                    }
                #endif
                if(support.is_visited(candidate)) continue;
                support.set_visited(candidate); 
                auto dst = computer(candidate); 
                support.insert(candidate, dst); 
                #ifndef NOPREFETCH
                    if(i >= list_sz - computer.graph_prefetch_len) {
                        graph_index_.prefetch(support.idx(support.cur()));
                    }
                #endif
            }
        }
    }

    idx_t allocInternalIDX(idx_t label){
        LockGuard guard(global_lock_);        
        if(cur_vertex_ >= max_vertex_) {
            THROW("ERROR: The current number of nodes has reached the limit and cannot be increased");
        }
        idx_t interID = cur_vertex_;
        cur_vertex_++;
        external_label_map_[interID] = label;
        if(internal_idx_map_.find(label) != internal_idx_map_.end()){
            std::cerr << "warning: allocInternalIDX: label exist, overwrite\n";
            deleteVertex(internal_idx_map_[label]);
        }
        internal_idx_map_[label] = interID;
        available_tags_[interID] = true;
        return interID;
    }

    void prune_edge(auto & neighbor_list, auto & computer) const {
        std::vector<bool> is_del(neighbor_list.size(), false);
        for(int i = 1; i < neighbor_list.size(); i++){
            auto x = neighbor_list[i].idx;
            auto dst_x = neighbor_list[i].dst;
            for(int j = 0; j < i; j++){
                if(is_del[j]) continue;
                auto y = neighbor_list[j].idx;
                auto dst_y = neighbor_list[j].dst;
                auto dst_btw = computer(x, y);
                
                if(alpha_ * dst_btw < dst_x){
                    is_del[i] = true;
                    break;
                }
            }
        }

        int p_1 = 0, p_2 = 0;
        while(p_2 < neighbor_list.size() && p_1 < R_){
            if(!is_del[p_2]){
                neighbor_list[p_1++] = neighbor_list[p_2];
            }
            p_2++;
        }
        neighbor_list.resize(p_1);
    }

    bool update_nbhood(idx_t host, idx_t candidate, dist_type dst, auto & computer, auto & support){

        LockGuard guard(graph_locks_[host]);
        
        const auto list_sz = graph_index_.getLength(host);
        auto nbhood = graph_index_[host];
        for(int i = 0; i < list_sz; i++){
            if(nbhood[i] == candidate) return false;
        }
        std::vector<bool> is_del(list_sz, false);

        auto radius = knn_graph_[host].radius_max;
        bool is_rvs;
        if(dst < radius) is_rvs = false;
        else is_rvs = true;

        int insert_pos = -1;
        if(is_rvs){
            int pos = list_sz;
            while(pos){
                auto other_nb = nbhood[pos - 1];
                auto dst_other = computer(host, other_nb);
                if(dst_other < dst && insert_pos == -1) insert_pos = pos;
                if(dst_other < radius) break;
                if(dst_other > knn_graph_[other_nb].radius_max) is_del[pos - 1] = true;
                pos--;
            }
            if(insert_pos == -1) insert_pos = pos;
        }else{
            int pos = 0;
            while(pos < list_sz){
                auto dst_other = computer(host, nbhood[pos]);
                if(dst_other > radius) break;
                auto dst_btw = computer(candidate, nbhood[pos]);
                if(dst_other <= dst){
                    if(alpha_ * dst_btw < dst){
                        return false;
                    }
                }else{
                    if(insert_pos == -1) insert_pos = pos;
                    if(alpha_ * dst_btw < dst_other){
                        is_del[pos] = true;
                    }
                }

                pos++;
            }
            if(insert_pos == -1) insert_pos = pos;
            while(pos < list_sz){
                auto dst_other = computer(host, nbhood[pos]);
                if(dst_other > (1.0 + alpha_) * knn_graph_[nbhood[pos]].radius_max) is_del[pos] = true;
                pos++;
            }

        }
        if(insert_pos >= R_) return false;
        std::vector<idx_t> alive_nbhood;
        alive_nbhood.reserve(R_ + 1);
        for(int i = 0; i < list_sz; i++){
            if(i == insert_pos) alive_nbhood.emplace_back(candidate);
            if(is_del[i]) continue;
            alive_nbhood.emplace_back(nbhood[i]);
        }
        if(insert_pos == list_sz) alive_nbhood.emplace_back(candidate);
        if(alive_nbhood.size() > R_) alive_nbhood.resize(R_);
        memcpy(nbhood, alive_nbhood.data(), alive_nbhood.size() * sizeof(idx_t));
        graph_index_.setLength(host, alive_nbhood.size());

        return true;
    }

    void getRandomStartpoints(){
        uint32_t random_size = 128;
        startPoints_.resize(random_size);
        std::mt19937 rng(cur_vertex_ + 2026);
        utils::GenRandom(rng, startPoints_.data(), random_size, cur_vertex_); 
    }

public:

    inline void printAvgEdge() const {
        size_t edge_sum = 0;
        edge_sum = 0;
        uint32_t max = 0, min = cur_vertex_;
        for (int i = 0; i < cur_vertex_; i++){
            auto list_sz = graph_index_.getLength(i);
            edge_sum += list_sz;
            max = std::max(max, list_sz);
            min = std::min(min, list_sz);
        }
        std::cout << "avg edge: " << (double)edge_sum / cur_vertex_ << std::endl;
        std::cout << "min: " << min << ", " << "max: " << max << std::endl;
    }

    inline size_t getCurrNum() const {
        return cur_vertex_;
    }

    inline size_t getMaxNum() const {
        return max_vertex_;
    }

    void test() {
    }

    Index() = default;

    void initData(const std::string data_path, size_t max_vertex = 0, Distance::MetricType metric = Distance::MetricType::L2){
        data_ = loader::loadData<data_type>(data_path);
        cur_vertex_ = data_.N;
        max_vertex_ = max_vertex == 0 ? cur_vertex_ : max_vertex;
        std::cout << "get data: " << data_.N << " x " << data_.dim << std::endl;
        std::vector<Lock>(max_vertex_).swap(graph_locks_);
        metric_ = metric;
        init_distfunc();
    }

    void initSearcher(){
        if(max_vertex_ == 0){
            THROW("ERROR: max_vertex_ = 0, cannot init searcher");
        }
        getRandomStartpoints();
        search_supporter_manager_ = SearchSupporterManager(max_vertex_);
    }

    void saveKNNG(const std::string out_path) const {
        std::vector<std::vector<uint32_t>> out_knng(cur_vertex_);

        for(idx_t n = 0; n < cur_vertex_; n++){
            out_knng.resize(K_);
            for(int i = 0; i < K_; i++){
                out_knng[n][i] = knn_graph_[n].nnlist[i].idx;
            }
        }
        
        if(io::endsWith(out_path, "vecs")){
            io::saveXVECS<uint32_t, true>(out_path, out_knng);
        }else if(io::endsWith(out_path, "bin")){
            io::saveXBIN<uint32_t, true>(out_path, out_knng);
        }else{
            THROW("saveKNNG: unsupported file format");
        }
    }

    void saveIndexDir(const std::string out_dir) const {
        loader::saveIndex(*this, out_dir);
    }

    void loadindexDir(const std::string in_dir){
        loader::loadIndex(*this, in_dir);
        std::vector<Lock>(max_vertex_).swap(graph_locks_);

        knn_graph_.resize(max_vertex_);
        status_pool_ = new bool[max_vertex_ * K_];
        for(idx_t i = 0; i < max_vertex_; i++){
            knn_graph_[i].nnlist = knn_pool_[i];
            knn_graph_[i].statlist = status_pool_ + i * K_;
            knn_graph_[i].capasity = K_;
        }
        for(idx_t i = 0; i < cur_vertex_; i++){
            knn_graph_[i].valid_range = K_; 
            knn_graph_[i].radius_max = knn_graph_[i].nnlist[K_ - 1].dst;
        }

        for(idx_t i = 0; i < cur_vertex_; i++){
            if(!available_tags_[i]) continue;
            internal_idx_map_[external_label_map_[i]] = i;
        }

    }

    std::vector<idx_t> search(void * query, uint32_t efsearch) const {

        auto computer = Computer::SearchComputer(static_cast<data_type *>(query), data_, distfunc_);
        auto supporter = search_supporter_manager_.getSupporter<dist_type>(efsearch * efsearch_expand_);

        for(int i = 0; i < startPoints_.size(); i++){
            auto dst = computer(startPoints_[i]); 
            supporter.insert(startPoints_[i], dst);
            supporter.set_visited(startPoints_[i]);
        }

        bestFirstSearch(supporter, computer);

        std::vector<idx_t> search_res(search_k_);
        int p_1 = 0, p_2 = 0;
        while(p_1 < search_k_ && p_2 < supporter.size()){
            if(!available_tags_.size() || available_tags_[supporter.idx(p_2)]){
                search_res[p_1++] = supporter.idx(p_2);
            }
            p_2++;
        }
        if(external_label_map_.size()) {
            for(int i = 0; i < search_k_; i++) {
                search_res[i] = external_label_map_[search_res[i]];
            }
        }
        return search_res;
    }

    void buildTwinGraph(const build_params bp){

        R_ = bp.R;
        alpha_ = bp.alpha;
        K_ = bp.K;
        
        knn_graph_.resize(cur_vertex_);
        auto computer = Computer::CommonComputer(data_, distfunc_);
       
        knn_pool_.realloc(cur_vertex_, K_, sizeof(edge_t) * K_, new edge_t[cur_vertex_ * K_]);
        status_pool_ = new bool[cur_vertex_ * K_];
        for(size_t i = 0; i < cur_vertex_; i++){
            knn_graph_[i].nnlist = knn_pool_[i];
            knn_graph_[i].statlist = status_pool_ + i * K_;
            knn_graph_[i].capasity = K_;
        }

        auto tmp_s = std::chrono::high_resolution_clock::now();
        #pragma omp parallel
        {
        std::mt19937 rng(omp_get_thread_num());
        std::vector<idx_t> random(K_ + 1);
        #pragma omp for
        for(idx_t n = 0; n < cur_vertex_; ++n){
            auto &nnlist = knn_graph_[n].nnlist;
            utils::GenRandom(rng, random.data(), random.size(), cur_vertex_);
            for (uint32_t l = 0, i = 0; l < K_; l++, i++){
                if (random[i] == n) i++;
                nnlist[l].idx = random[i];
                nnlist[l].dst = computer(nnlist[l].idx, n);
            }
            std::sort(nnlist, nnlist + K_);
            knn_graph_[n].valid_range = K_; 
            knn_graph_[n].radius_max = std::numeric_limits<dist_type>::max();
        }
        };

        size_t stride_resindex = sizeof(uint32_t) + R_ * sizeof(edge_t);
        auto resindex = Memory::AlignedMemoryManager<edge_t, true>(
            cur_vertex_, 
            R_, 
            stride_resindex, 
            new char[cur_vertex_ * stride_resindex]
        );
        for(int n = 0; n < cur_vertex_; n++){
            resindex.setLength(n, 0);
        }

        inDescent(resindex, computer, bp.MaxIter);
        delete [] status_pool_;

        std::vector<std::vector<idx_t>> rvsgraph(cur_vertex_);
        #pragma omp parallel for
        for(int n = 0; n < cur_vertex_; n++){
            int list_sz = resindex.getLength(n);
            for(int i = 0; i < list_sz; i++){
                bool repeated = false;
                int nb = resindex[n][i].idx;

                LockGuard guard(graph_locks_[nb]);
                rvsgraph[nb].emplace_back(n);
            }
        }
        graph_index_.realloc(cur_vertex_, R_, sizeof(idx_t) * R_ + sizeof(uint32_t), new char[cur_vertex_ * (sizeof(idx_t) * R_ + sizeof(uint32_t))]);
        #pragma omp parallel
        {
        std::vector<edge_t> suplist;
        #pragma omp for
        for(int n = 0; n < cur_vertex_; n++){
            auto list_sz = resindex.getLength(n);
            int p_1 = 0, p_2 = 0;
            while(p_2 < rvsgraph[n].size()){
                bool repeat = false;
                for(int i = 0; i < list_sz; i++){
                    if(rvsgraph[n][p_2] == resindex[n][i].idx){
                        repeat = true;
                        break;
                    }
                }
                if(!repeat){
                    rvsgraph[n][p_1++] = rvsgraph[n][p_2];
                }
                p_2++;
            }
            rvsgraph[n].resize(p_1);
            suplist.resize(list_sz);
            memcpy(suplist.data(), resindex[n], list_sz * sizeof(edge_t));
            for(int i = 0; i < rvsgraph[n].size(); i++){
                suplist.emplace_back(edge_t(rvsgraph[n][i], computer(n, rvsgraph[n][i])));
            }
            std::sort(suplist.begin(), suplist.end());
            for(int i = 0; i < suplist.size() && i < R_; i++){
                graph_index_[n][i] = suplist[i].idx;
            }
            graph_index_.setLength(n, std::min((int)suplist.size(), R_));
        }
        };
        

        auto tmp_e = std::chrono::high_resolution_clock::now();
        std::cout << "time: " << std::chrono::duration<double>(tmp_e - tmp_s).count() << std::endl;
        
        external_label_map_.resize(max_vertex_);
        available_tags_.resize(max_vertex_, false);
        for(int i = 0; i < cur_vertex_; i++){
            external_label_map_[i] = i;
            internal_idx_map_[i] = i;
            available_tags_[i] = true;
        }

        printAvgEdge();

    }

    void addVertex(void * query, idx_t label){

        auto idx = allocInternalIDX(label);

        memcpy(data_[idx], query, data_.sizeRow);
        auto computer = Computer::SearchComputer(static_cast<data_type *>(query), data_, distfunc_);
        auto common_computer = Computer::CommonComputer(data_, distfunc_);
        auto supporter = search_supporter_manager_.getSupporter<dist_type>(K_ * addvertex_expand_);
        supporter.set_visited(idx); 
        std::vector<edge_t> candidate_set;
        std::vector<edge_t> rvsneighbors;
        std::queue<idx_t> beam_list;

        for(int i = 0; i < startPoints_.size(); i++){
            auto sp = startPoints_[i];
            auto dst = computer(sp); 
            supporter.insert(sp, dst);
            supporter.set_visited(sp);
            
            if(knn_graph_[sp].parallelInsert({idx, dst}) && update_nbhood(sp, idx, dst, common_computer, supporter)){
                rvsneighbors.emplace_back(sp, dst);
            }
        }

        while(supporter.has_next()){
            while(beam_list.size() < beam_width_ && supporter.has_next()){
                beam_list.push(supporter.pop());
            }
            while(!beam_list.empty()){
                auto cur_id = beam_list.front();
                beam_list.pop();

                auto list_sz = graph_index_.getLength(cur_id);
                const auto nbhood = graph_index_[cur_id];
                #ifndef NOPREFETCH
                    graph_index_.prefetch(cur_id);  
                    for(int i = 0; i < prefetch_size_data_; i++){
                        computer.prefetch(nbhood[i]);                
                        supporter.prefetchVL(nbhood[i]);
                    }
                #endif
                for(int i = 0; i < list_sz; i++){
                    auto candidate = nbhood[i];

                    #ifndef NOPREFETCH
                        if(i + prefetch_size_data_ < list_sz) {
                            computer.prefetch(nbhood[i + prefetch_size_data_]);
                            supporter.prefetchVL(nbhood[i + prefetch_size_data_]);
                        }
                    #endif
                    if(supporter.is_visited(candidate)) continue;
                    supporter.set_visited(candidate); 
                    auto dst = computer(candidate); 
                    supporter.insert(candidate, dst); 

                    if(knn_graph_[candidate].parallelInsert({idx, dst}) && update_nbhood(candidate, idx, dst, common_computer, supporter)){
                        rvsneighbors.emplace_back(candidate, dst);
                    }

                    #ifndef NOPREFETCH
                        if(i >= list_sz - 10) {
                            graph_index_.prefetch(supporter.idx(supporter.cur()));
                        }
                    #endif 
                }
            }
        }

        supporter.reset_checked();
        knn_graph_[idx].radius_max = std::numeric_limits<dist_type>::max();
        for(int i = 0; i < K_; i++){
            candidate_set.emplace_back(supporter[i]);
            knn_graph_[idx].parallelInsert(supporter[i]);
        }

        prune_edge(candidate_set, common_computer);

        for(int i = 0; i < candidate_set.size(); i++){
            update_nbhood(candidate_set[i].idx, idx, candidate_set[i].dst, common_computer, supporter);
        }

        int size_bak = candidate_set.size();
        for(int i = 0; i < rvsneighbors.size(); i++){
            bool repeat = false;
            auto cand = rvsneighbors[i];
            for(int j = 0; j < size_bak; j++){
                if(candidate_set[j].idx == cand.idx) {
                    repeat = true;
                    break;
                }
            }
            if(!repeat) candidate_set.emplace_back(cand);
        }

        std::sort(candidate_set.begin(), candidate_set.end());
        if(candidate_set.size() > R_) candidate_set.resize(R_);

        for(int i = 0; i < candidate_set.size(); i++){
            graph_index_[idx][i] = candidate_set[i].idx;
        }
        graph_index_.setLength(idx, candidate_set.size());

    }

    void deleteVertex(idx_t label){
        if(internal_idx_map_.find(label) == internal_idx_map_.end()){
            std::cerr << "waning: deleteVertex: label do not exist\n";
            return;
        }
        auto idx = internal_idx_map_[label];
        if(idx < 0 || idx >= max_vertex_) return;
        available_tags_[idx] = false;
        internal_idx_map_.erase(idx);
        del_vertex_++;
        if(del_vertex_ >= 0.2 * cur_vertex_){
            rejuvenation();
        }
    }

    void rejuvenation(){

        std::cout << "Start rejuvenation ... current deleted vertex num: " << del_vertex_ << ", current total vertex num: " << cur_vertex_ <<  std::endl;

        if(del_vertex_ == 0) {
            std::cout << "no deleted vertex, skipped" << std::endl;
            return;
        }
        
        std::unordered_map<idx_t, idx_t> reallocated_idx;
        auto computer = Computer::CommonComputer(data_, distfunc_);

        idx_t end_idx = cur_vertex_;
        while(!available_tags_[--end_idx]);

        for(int i = 0; i < end_idx; i++){
            if(available_tags_[i]) continue;

            reallocated_idx[end_idx] = i;
            internal_idx_map_[external_label_map_[end_idx]] = i;
            external_label_map_[i] = external_label_map_[end_idx];
            external_label_map_[end_idx] = -1;
            while(!available_tags_[--end_idx] && end_idx > i);
        }

        size_t stride_resindex = sizeof(uint32_t) + R_ * sizeof(edge_t);
        auto resindex = Memory::AlignedMemoryManager<edge_t, true>(
            cur_vertex_, 
            R_, 
            stride_resindex, 
            new char[cur_vertex_ * stride_resindex]
        );
        std::fill(status_pool_, status_pool_ + cur_vertex_ * K_, false);

        #pragma omp parallel
        {
        std::mt19937 rng(cur_vertex_ + omp_get_thread_num());
        #pragma omp for
        for(int n = 0; n < cur_vertex_; n++){
            if(!available_tags_[n]) continue;
            auto & nbhood = knn_graph_[n];
            bool tag;

            tag = false;
            for(int i = 0; i < nbhood.valid_range; i++){
                if(!available_tags_[nbhood.nnlist[i].idx]){
                    auto deleted_id = nbhood.nnlist[i].idx;
                    nbhood.deleteNeighbor(i);
                    i -= 1;
                    tag = true;
                } 
            }

            int l = 0, r = 0;
            auto naviglist = graph_index_[n];
            auto indexlist_size = graph_index_.getLength(n);
            while(r < indexlist_size){
                if(available_tags_[naviglist[r]]){
                    naviglist[l] = naviglist[r];
                    if(reallocated_idx.find(naviglist[l]) != reallocated_idx.end()){
                        naviglist[l] = reallocated_idx[naviglist[l]];
                    }
                    resindex[n][l] = {naviglist[l], computer(n, naviglist[l])};
                    l++;
                }
                r++;
            }
            graph_index_.setLength(n, l);
            resindex.setLength(n, l);
        }
        };

        #pragma omp parallel for
        for(int n = 0; n < cur_vertex_; n++){
            if(!available_tags_[n]) continue;
            auto & nbhood = knn_graph_[n];

            for(int i = 0; i < nbhood.valid_range; i++){
                if(reallocated_idx.find(nbhood.nnlist[i].idx) == reallocated_idx.end()) continue;
                nbhood.nnlist[i].idx = reallocated_idx[nbhood.nnlist[i].idx];
            }
        }

        #pragma omp parallel for
        for(int i = end_idx + 1; i < cur_vertex_; i++){
            if(!available_tags_[i]) continue;
            if(reallocated_idx.find(i) == reallocated_idx.end()) THROW("rejuvenation: reallocate error");
            auto new_id = reallocated_idx[i];

            knn_graph_[new_id] = knn_graph_[i];
            memcpy(graph_index_.getrow(new_id), graph_index_.getrow(i), graph_index_.sizeRow);
            memcpy(data_.getrow(new_id), data_.getrow(i), data_.sizeRow);
            available_tags_[new_id] = true;
            available_tags_[i] = false; 
        }
        cur_vertex_ -= del_vertex_;
        del_vertex_ = 0;

        inDescent(resindex, computer, 10);

        std::vector<std::vector<idx_t>> rvsgraph(cur_vertex_);
        #pragma omp parallel for
        for(int n = 0; n < cur_vertex_; n++){
            int list_sz = resindex.getLength(n);
            for(int i = 0; i < list_sz; i++){
                bool repeated = false;
                int nb = resindex[n][i].idx;
                int nb_listsz = resindex.getLength(nb);

                LockGuard guard(graph_locks_[nb]);
                rvsgraph[nb].emplace_back(n);
            }
        }
        #pragma omp parallel
        {
        std::vector<edge_t> suplist;
        #pragma omp for
        for(int n = 0; n < cur_vertex_; n++){
            auto list_sz = resindex.getLength(n);
            int p_1 = 0, p_2 = 0;
            while(p_2 < rvsgraph[n].size()){
                bool repeat = false;
                for(int i = 0; i < list_sz; i++){
                    if(rvsgraph[n][p_2] == resindex[n][i].idx){
                        repeat = true;
                        break;
                    }
                }
                if(!repeat){
                    rvsgraph[n][p_1++] = rvsgraph[n][p_2];
                }
                p_2++;
            }
            rvsgraph[n].resize(p_1);
            suplist.resize(list_sz);
            memcpy(suplist.data(), resindex[n], list_sz * sizeof(edge_t));
            for(int i = 0; i < rvsgraph[n].size(); i++){
                suplist.emplace_back(edge_t(rvsgraph[n][i], computer(n, rvsgraph[n][i])));
            }
            std::sort(suplist.begin(), suplist.end());
            for(int i = 0; i < suplist.size() && i < R_; i++){
                graph_index_[n][i] = suplist[i].idx;
            }
            graph_index_.setLength(n, std::min((int)suplist.size(), R_));
        }
        };

        getRandomStartpoints();
    }

};

};  //namespace nns
