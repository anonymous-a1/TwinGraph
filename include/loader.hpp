#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <functional>
#include <filesystem>
#include <assert.h>
#include "version.h"
#include "common.h"
#include "memory.hpp"
#include "computer.hpp"
#include "neighbor.hpp"
#include "iomanager.hpp"


namespace nns{

#define OVERWRITE

struct Paths{

    std::string graph_path  = "";
    std::string index_dir   = "";    
    std::string query_path  = "";
    std::string data_path   = "";
    std::string gt_path     = "";
    std::string out_path    = "";
    std::string out_dir     = "";
    std::string out_knng    = "";

    static std::string getSuffix(std::string const & s) {
        if(s.size() == 0) {
            THROW("Path error");
        }
        auto ret = s;
        auto it = ret.end();
        do it--; while(it!=ret.begin() && *it != '.');
        ret.erase(ret.begin(), it);
        return ret;
    }
};

struct loader{

    static std::string getIndexDataType(std::string index_dir){
        if(index_dir.back() != '/') index_dir += "/";
        std::ifstream config_file(index_dir + "config_list.txt");
        std::string line;
        std::getline(config_file, line);
        std::getline(config_file, line);
        std::getline(config_file, line);
        const std::string prefix = "Data type: ";
        std::string data_type = line.substr(prefix.size());

        return data_type; 
    }

    template <typename data_type>
    static Memory::AlignedMemoryManager<data_type> load_queries(std::string path){
        data_type * data = nullptr;
        size_t size_n, size_d, stride;
        if(io::endsWith(path, "vecs")){
            data = io::loadAlignXVECSPtr<data_type>(path, size_n, size_d, stride);
        }else if(io::endsWith(path, "bin")){
            data = io::loadAlignXBINPtr<data_type>(path, size_n, size_d, stride);
        }else{
            THROW("ERROR: Unsupported data file format");
        }

        return {size_n, size_d, stride, data};
    }

    static std::vector<std::vector<uint32_t>> load_groundtruth(std::string path){
        if(io::endsWith(path, "vecs")){
            return io::loadXVECS<uint32_t>(path);
        }        
        else if(io::endsWith(path, "bin")){
            return io::loadXBIN<uint32_t>(path);
        } else{
            THROW("ERROR: Unsupported ground truth file format");
        }
    }

    template <typename T>
    static Memory::AlignedMemoryManager<T>
    loadData(std::string path){
        if(path.size() == 0) return {};
        T* (*loadfunc)(const std::string, size_t&, size_t&, size_t&);
        if(io::endsWith(path, "vecs")){
            loadfunc = io::loadAlignXVECSPtr;
        }else if(io::endsWith(path, "bin")){
            loadfunc = io::loadAlignXBINPtr;
        }else if(io::endsWith(path, "hnsw"))
            loadfunc = io::loadAlignDataFromHNSW;
        else{
            THROW("ERROR: Unsupported data file format");
        }
        size_t n, d, stride = 0;
        auto data = loadfunc(path, n, d, stride);
        return {n, d, stride, (char *)data}; 
    }


    template <typename Index_t>
    static void 
    loadGraph(Index_t& index, std::string path){
        if(io::endsWith(path, "ivecs")){
            loadIVECSGraph(index, path);
        }else{
            THROW("ERROR: Unsupported graph_index_ file format");
        }
    }

    template <typename Index_t>
    static void 
    saveIndex(const Index_t & index, std::string out_dir){
        std::cout << "Saving index files ... \n";

        if(out_dir.back() != '/') out_dir += '/';
        checkOutputPath(out_dir);

        std::ofstream viewable_config_file(out_dir + "config_list.txt");
        viewable_config_file << "Version: " << __version__ << std::endl << std::endl;
        viewable_config_file << "Data type: " << datatype_s<typename Index_t::data_type> << std::endl;
        viewable_config_file << "Maximum vertex num: " << index.max_vertex_ << std::endl;
        viewable_config_file << "Current vertex num: " << index.cur_vertex_ << std::endl;
        viewable_config_file << "Active vertex num: " << index.cur_vertex_ - index.del_vertex_ << std::endl;
        viewable_config_file << "alpha_: " << index.alpha_ << std::endl;
        viewable_config_file << "R_: " << index.R_ << std::endl;
        viewable_config_file.close();

        std::ofstream data_file(out_dir + "data", std::ios::binary);
        data_file.write((char *)&index.max_vertex_, sizeof(size_t));
        data_file.write((char *)&index.cur_vertex_, sizeof(size_t));
        data_file.write((char *)&index.metric_, sizeof(index.metric_));
        data_file.write((char *)&index.data_.N, sizeof(size_t));
        data_file.write((char *)&index.data_.dim, sizeof(size_t));
        data_file.write((char *)index.data_.get(), index.cur_vertex_ * index.data_.sizeRow);
        data_file.close();

        std::ofstream graph_file(out_dir + "TwinGraph_Index", std::ios::binary);
        graph_file.write((char *)&index.max_vertex_, sizeof(size_t));
        graph_file.write((char *)&index.cur_vertex_, sizeof(size_t));
        graph_file.write((char *)&index.K_, sizeof(int));
        graph_file.write((char *)&index.alpha_, sizeof(float));
        graph_file.write((char *)&index.R_, sizeof(int));
        graph_file.write((char *)&index.graph_index_.aligned_tag, sizeof(bool));
        graph_file.write((char *)&index.graph_index_.N, sizeof(size_t));
        graph_file.write((char *)&index.graph_index_.dim, sizeof(size_t));
        graph_file.write((char *)&index.graph_index_.sizeRow, sizeof(size_t));
        graph_file.write((char *)index.graph_index_.get(), index.cur_vertex_ * index.graph_index_.sizeRow);
        graph_file.write((char *)index.external_label_map_.data(), index.cur_vertex_ * sizeof(idx_t));
        graph_file.write((char *)&index.del_vertex_, sizeof(size_t));
        graph_file.write((char *)index.available_tags_.data(), index.cur_vertex_ * sizeof(bool));
        
        graph_file.write((char *)index.knn_pool_.get(), index.cur_vertex_ * index.knn_pool_.sizeRow);
        graph_file.close();

        std::cout << "Successfully saved index in directory " << out_dir << std::endl;
    }

    template <typename Index_t>
    static void 
    loadIndex(Index_t & index, std::string in_dir){
        std::cout << "Loading index files ... \n";

        if(in_dir.back() != '/') in_dir += '/';

        size_t max_vertex, cur_vertex;

        std::ifstream graph_file(in_dir + "TwinGraph_Index", std::ios::binary);
        graph_file.read((char *)&max_vertex, sizeof(size_t));
        graph_file.read((char *)&cur_vertex, sizeof(size_t));
        assert(cur_vertex <= max_vertex);
        index.max_vertex_ = max_vertex;
        index.cur_vertex_ = cur_vertex;
        graph_file.read((char *)&index.K_, sizeof(int));
        graph_file.read((char *)&index.alpha_, sizeof(float));
        graph_file.read((char *)&index.R_, sizeof(int));
        bool level0_aligned_tag;
        size_t level0_N, level0_dim, level0_sizeRow;
        graph_file.read((char *)&level0_aligned_tag, sizeof(bool));
        graph_file.read((char *)&level0_N, sizeof(size_t));
        graph_file.read((char *)&level0_dim, sizeof(size_t));
        graph_file.read((char *)&level0_sizeRow, sizeof(size_t));
        char * level0_data = new char[level0_N * level0_sizeRow];
        graph_file.read(level0_data, cur_vertex * level0_sizeRow);
        index.graph_index_.realloc(level0_N, level0_dim, level0_sizeRow, level0_data);
        index.graph_index_.aligned_tag = level0_aligned_tag;
        index.external_label_map_.resize(max_vertex);
        index.available_tags_.resize(max_vertex, false);
        graph_file.read((char *)index.external_label_map_.data(), cur_vertex * sizeof(idx_t));
        graph_file.read((char *)&index.del_vertex_, sizeof(size_t));
        graph_file.read((char *)index.available_tags_.data(), cur_vertex * sizeof(bool));
        auto dists_data = new typename Index_t::edge_t[max_vertex * index.K_];
        graph_file.read((char *)dists_data, cur_vertex * index.K_ * sizeof(typename Index_t::edge_t));
        index.knn_pool_.realloc(max_vertex, index.K_, index.K_ * sizeof(typename Index_t::edge_t), dists_data);
        graph_file.close();


        std::ifstream data_file(in_dir + "data", std::ios::binary);
        data_file.read((char *)&max_vertex, sizeof(size_t));
        data_file.read((char *)&cur_vertex, sizeof(size_t));
        assert(max_vertex == index.max_vertex_);
        assert(cur_vertex == index.cur_vertex_);
        data_file.read((char *)&index.metric_, sizeof(index.metric_));
        size_t size_n, size_d;
        data_file.read((char *)&size_n, sizeof(size_t));
        data_file.read((char *)&size_d, sizeof(size_t));
        index.data_.realloc(max_vertex, size_d);
        data_file.read((char *)index.data_.get(), cur_vertex * index.data_.sizeRow);
        data_file.close();

        index.init_distfunc();
        
        std::cout << "Loaded index from directory " << in_dir << std::endl;
    }

private:

    template <typename Index_t>
    static void
    loadIVECSGraph(Index_t& index, std::string path){
        size_t n, d, stride;
        auto data = io::loadAlignXVECSPtr<idx_t, 0, true>(path, n, d, stride);
        index.graph_index_.realloc(n, d, stride, (char *)data);
    }


    static void checkOutputPath(std::string out_dir){
        std::filesystem::path dirPath(out_dir);
        int status = 0;

        if(std::filesystem::exists(dirPath)){
            if(std::filesystem::is_directory(dirPath)){
                #if defined OVERWRITE
                    std::cout << "Warning: Output directory \"" + out_dir + "\" exists, it will be overwritten\n";
                    status += std::filesystem::remove_all(dirPath);
                    status += std::filesystem::create_directory(dirPath);
                    if(!status){
                        THROW(("Output directory \"" + out_dir + "\" exists and can not access").c_str());
                    }
                #else
                    THROW(("Output directory \"" + out_dir + "\" exists, overwritting not allow\n
                        Add \"-DOVERWRITE\" to Compilation Options to allow overwritting").c_str());
                #endif
            }
        }else{
            status += std::filesystem::create_directory(dirPath);
            if(!status){
                THROW(("Cannot create output directory \"" + out_dir + "\"").c_str());
            }
        }
    }
};

};