#pragma once

#include <unistd.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

#ifndef MMALIGN 
#define MMALIGN 64
#endif

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

namespace io{

template <typename d_type, size_t _align = MMALIGN, bool _listlength = false, bool _verbose = false>
d_type * loadAlignXBINPtr(const std::string data_path, size_t & size_n, size_t & size_d, size_t & stride_){

    size_t offset = _listlength ? sizeof(uint32_t) : 0;
    std::ifstream infile(data_path, std::ios::binary);
    if(!infile.is_open()) THROW(("\"" + data_path + "\" can not open for read").c_str());

    uint32_t num, dim;
    infile.read((char *)&num, sizeof(uint32_t));
    infile.read((char *)&dim, sizeof(uint32_t));
    size_n = num;
    size_d = dim;

    if(_align) stride_ = (sizeof(d_type) * size_d + offset + _align - 1) / _align * _align;
    else stride_ = size_d * sizeof(d_type) + offset;

    if(_verbose){
        std::cout << "Loading from \"" << data_path << "\", shape: " << size_n << " X " << size_d << std::endl;
    }

    d_type * data = (d_type *)std::aligned_alloc(_align, size_n * stride_);
    if(_align) memset(data, 0, size_n * stride_);
    char * ptr = (char *)data;

    for(size_t i = 0; i < size_n; i++){
        if(_listlength) *(uint32_t *)ptr = dim;
        infile.read(ptr + offset, size_d * sizeof(d_type));
        ptr += stride_;
    }
    infile.close();
    
    if(_verbose){
        std::cout << "Successfully loaded" << std::endl;
    }
    return data;
}

template <typename d_type, size_t _align = MMALIGN, bool _listlength = false, bool _verbose = false>
d_type * loadAlignXBINHeadPtr(const std::string data_path, const size_t headnum, size_t & size_d, size_t & stride_){

    size_t offset = _listlength ? sizeof(uint32_t) : 0;
    std::ifstream infile(data_path, std::ios::binary);
    if(!infile.is_open()) THROW(("\"" + data_path + "\" can not open for read").c_str());

    uint32_t num, dim;
    infile.read((char *)&num, sizeof(uint32_t));
    infile.read((char *)&dim, sizeof(uint32_t));
    size_d = dim;

    if(num < headnum) THROW("io::loadAlignXBINHeadPtr: num < headnum");

    if(_align) stride_ = (sizeof(d_type) * size_d + offset + _align - 1) / _align * _align;
    else stride_ = size_d * sizeof(d_type) + offset;

    if(_verbose){
        std::cout << "Loading from \"" << data_path << "\", shape: " << num << " X " << size_d << ", head " << headnum << std::endl;
    }

    d_type * data = (d_type *)std::aligned_alloc(_align, headnum * stride_);
    if(_align) memset(data, 0, headnum * stride_);
    char * ptr = (char *)data;
    
    for(size_t i = 0; i < headnum; i++){
        if(_listlength) *(uint32_t *)ptr = dim;
        infile.read(ptr + offset, size_d * sizeof(d_type));
        ptr += stride_;
    }
    infile.close();
    
    if(_verbose){
        std::cout << "Successfully loaded" << std::endl;
    }
    return data;
}


template <typename d_type, size_t _align = MMALIGN, bool _listlength = false, bool _verbose = false>
d_type * loadAlignXVECSPtr(const std::string data_path, size_t & size_n, size_t & size_d, size_t & stride_){

    size_t offset = _listlength ? sizeof(uint32_t) : 0;   
    std::ifstream infile(data_path, std::ios::binary);
    if(!infile.is_open()) THROW(("\"" + data_path + "\" can not open for read").c_str());


    uint32_t num = 0, dim = 0, d; 
    while(infile.read((char*)&d, sizeof(uint32_t))){
        dim = std::max(dim, d);
        num++;
        infile.seekg(d * sizeof(d_type), std::ios::cur);
    }

    size_n = num; 
    size_d = dim; 

    if(_align) stride_ = (sizeof(d_type) * size_d + offset + _align - 1) / _align * _align;
    else stride_ = size_d * sizeof(d_type) + offset;

    if(_verbose){
        std::cout << "Loading from \"" << data_path << "\", shape: " << size_n << " X " << size_d << std::endl;
    }

    d_type * data = (d_type *)std::aligned_alloc(_align, size_n * stride_);
    if(_align) memset(data, 0, size_n * stride_);
    char * ptr = (char *)data;

    infile.clear();
    infile.seekg(0, std::ios::beg);
    for(size_t i = 0; i < size_n; ++i){
        infile.read((char *)&dim, sizeof(uint32_t));
        if(_listlength) *(uint32_t *)ptr = dim;
        infile.read(ptr + offset, dim * sizeof(d_type));
        ptr += stride_;
    }
    infile.close();

    if(_verbose){
        std::cout << "Successfully loaded" << std::endl;
    }
    return data;
}

template <typename d_type, bool _verbose = false>
d_type * loadXBINPtr(const std::string data_path, size_t & size_n, size_t & size_d){

    std::ifstream infile(data_path, std::ios::binary);
    if(!infile.is_open()) THROW(("\"" + data_path + "\" can not open for read").c_str());


    uint32_t num, dim;
    infile.read((char *)&num, sizeof(uint32_t));
    infile.read((char *)&dim, sizeof(uint32_t));
    size_n = num;
    size_d = dim;

    if(_verbose){
        std::cout << "Loading from \"" << data_path << "\", shape: " << size_n << " X " << size_d << std::endl;
    }

    d_type * data = new d_type[size_n * size_d];

    infile.read((char *)data, size_n * size_d * sizeof(d_type));
    infile.close();
    
    if(_verbose){
        std::cout << "Successfully loaded" << std::endl;
    }

    return data;
}

template <typename d_type, bool _verbose = false>
d_type * loadXVECSPtr(const std::string data_path, size_t & size_n, size_t & size_d){
    std::streampos fileSize; 
    std::ifstream infile(data_path, std::ios::binary);
    if(!infile.is_open()) THROW(("\"" + data_path + "\" can not open for read").c_str());


    uint32_t num, dim; 
    infile.read((char *)&dim, sizeof(uint32_t));
    infile.seekg(0, std::ios::end);
    fileSize = infile.tellg();    
    num = (double)fileSize / (sizeof(uint32_t) + dim * sizeof(d_type));
    size_n = num;
    size_d = dim; 

    if(_verbose){
        std::cout << "Loading from \"" << data_path << "\", shape: " << size_n << " X " << size_d << std::endl;
    }
    
    d_type * data = new d_type[size_n * size_d];
    
    infile.seekg(0, std::ios::beg);
    for(size_t i = 0; i < size_n; ++i){
        infile.read((char *)&dim, sizeof(uint32_t));
        infile.read((char *)(data + i * dim), dim * sizeof(d_type));
    }
    infile.close();

    if(_verbose){
        std::cout << "Successfully loaded" << std::endl;
    }

    return data;
}

template <typename d_type, bool _verbose = false>
std::vector<std::vector<d_type>> loadXBIN(const std::string data_path){
    std::ifstream infile(data_path, std::ios::binary);
    if(!infile.is_open()) THROW(("\"" + data_path + "\" can not open for read").c_str());


    uint32_t num, dim;
    infile.read((char *)&num, sizeof(uint32_t));
    infile.read((char *)&dim, sizeof(uint32_t));

    if(_verbose){
        std::cout << "Loading from \"" << data_path << "\", shape: " << num << " X " << dim << std::endl;
    }

    std::vector<std::vector<d_type>> data(num);
    for(size_t i = 0; i < num; i++){
        data[i].resize(dim);
        infile.read((char *)data[i].data(), dim * sizeof(d_type));
    }
    infile.close();

    if(_verbose){
        std::cout << "Successfully loaded" << std::endl;
    }

    return data;
}

template <typename d_type, bool _verbose = false>
std::vector<std::vector<d_type>> loadXVECS(const std::string data_path){
    std::ifstream infile(data_path, std::ios::binary);
    if(!infile.is_open()) THROW(("\"" + data_path + "\" can not open for read").c_str());

    std::vector<std::vector<d_type>> data;

    if(_verbose){
        std::cout << "Loading from \"" << data_path << std::endl         ;
    }

    uint32_t dim;
    while( infile.read((char *)&dim, sizeof(uint32_t)) ){
        std::vector<d_type> row(dim);
        infile.read((char *)row.data(), dim * sizeof(d_type));
        data.emplace_back(row);
    }
    infile.close();

    if(_verbose){
        std::cout << "Successfully loaded, shape: " << data.size() << " X " << dim << std::endl;
    }

    return data;
}

template <typename d_type, bool _verbose = false>
void saveXBIN(const std::string out_path, const void * data, const size_t size_n, const size_t size_d){
    std::ofstream ofile(out_path, std::ios::binary);
    if(!ofile.is_open()) THROW(("\"" + out_path + "\" can not open for write").c_str());


    uint32_t num = size_n, dim = size_d;
    ofile.write((char *)&num, sizeof(uint32_t));
    ofile.write((char *)&dim, sizeof(uint32_t));

    if(_verbose){
        std::cout << "Saving to \"" << out_path << "\", shape: " << size_n << " X " << size_d << std::endl;
    }

    ofile.write((char *)data, size_n * size_d * sizeof(d_type));
    ofile.close();

    if(_verbose){
        std::cout << "Successfully saved" << std::endl;
    }
}

template <typename d_type, bool _verbose = false>
void saveXBIN(const std::string out_path, std::vector<std::vector<d_type>> const & data){
    std::ofstream ofile(out_path, std::ios::binary);
    if(!ofile.is_open()) THROW(("\"" + out_path + "\" can not open for write").c_str());

    
    uint32_t num = data.size(), dim = data[0].size();
    ofile.write((char *)&num, sizeof(uint32_t));
    ofile.write((char *)&dim, sizeof(uint32_t));

    if(_verbose){
        std::cout << "Saving to \"" << out_path << "\", shape: " << num << " X " << dim << std::endl;
    }
    for(size_t i = 0; i < num; i++){
        ofile.write((char *)data[i].data(), dim * sizeof(d_type));
    }
    ofile.close();

    if(_verbose){
        std::cout << "Successfully saved" << std::endl;
    }
}

template <typename d_type, bool _verbose = false>
void saveXVECS(const std::string out_path, const void * data, const size_t size_n, const size_t size_d){
    std::ofstream ofile(out_path, std::ios::binary);
    if(!ofile.is_open()) THROW(("\"" + out_path + "\" can not open for write").c_str());


    if(_verbose){
        std::cout << "Saving to \"" << out_path << "\", shape: " << size_n << " X " << size_d << std::endl;
    }

    uint32_t dim = size_d;
    for(size_t i = 0; i < size_n; i++){
        ofile.write((char *)&dim, sizeof(uint32_t));
        ofile.write((char *)((d_type *)data + i * dim), dim * sizeof(d_type));
    }
    ofile.close();

    if(_verbose){
        std::cout << "Successfully saved" << std::endl;
    }
}

template <typename d_type, bool _verbose = false>
void saveXVECS(const std::string out_path, std::vector<std::vector<d_type>> const & data){
    std::ofstream ofile(out_path, std::ios::binary);
    if(!ofile.is_open()) THROW(("\"" + out_path + "\" can not open for write").c_str());


    if(_verbose){
        std::cout << "Saving to \"" << out_path << std::endl;
    }

    uint32_t dim;
    for(size_t i = 0; i < data.size(); i++){
        dim = data[i].size();
        ofile.write((char *)&dim, sizeof(uint32_t));
        ofile.write((char *)data[i].data(), dim * sizeof(d_type));
    }
    ofile.close();

    if(_verbose){
        std::cout << "Successfully saved, shape: " << data.size() << " X " << dim << std::endl;
    }
}

template <typename d_type, bool _verbose = false>
d_type * loadDataFromHNSW(const std::string data_path, size_t & size_n, size_t & size_d) {
    std::ifstream infile(data_path, std::ios::binary);
    if(!infile.is_open()) THROW(("\"" + data_path + "\" can not open for read").c_str());


    size_t cur_element_count;
    size_t size_data_per_element;
    size_t label_offset;
    size_t offsetData;

    infile.seekg(2 * sizeof(size_t), std::ios::cur);
    infile.read((char *)&cur_element_count, sizeof(size_t));
    infile.read((char *)&size_data_per_element, sizeof(size_t));
    infile.read((char *)&label_offset, sizeof(size_t));
    infile.read((char *)&offsetData, sizeof(size_t));

    size_n = cur_element_count;
    size_d = (label_offset - offsetData) / sizeof(d_type);


    if(_verbose){
        std::cout << "Loading (only data) from \"" << data_path << "\", shape: " << size_n << " X " << size_d << std::endl;
    }

    d_type * data = new d_type[size_n * size_d];
    infile.seekg(96 + offsetData, std::ios::beg);

    for(size_t i = 0; i < size_n; i++){
        infile.read((char *)(data + i * size_d), size_d * sizeof(d_type));
        infile.seekg(offsetData + sizeof(size_t), std::ios::cur);
    }
    infile.clear();
    infile.close();

    if(_verbose){
        std::cout << "Successfully loaded" << std::endl;
    }
    return data;
}

template <typename d_type, size_t _align = MMALIGN, bool _verbose = false>
d_type * loadAlignDataFromHNSW(const std::string data_path, size_t & size_n, size_t & size_d, size_t & stride) {
    std::ifstream infile(data_path, std::ios::binary);
    if(!infile.is_open()) THROW(("\"" + data_path + "\" can not open for read").c_str());


    size_t cur_element_count;
    size_t size_data_per_element;
    size_t label_offset;
    size_t offsetData;

    infile.seekg(2 * sizeof(size_t), std::ios::cur);
    infile.read((char *)&cur_element_count, sizeof(size_t));
    infile.read((char *)&size_data_per_element, sizeof(size_t));
    infile.read((char *)&label_offset, sizeof(size_t));
    infile.read((char *)&offsetData, sizeof(size_t));

    size_n = cur_element_count;
    size_d = (label_offset - offsetData) / sizeof(d_type);

    if(_align) stride = (sizeof(d_type) * size_d + _align - 1) / _align * _align;
    else stride = size_d * sizeof(d_type);

    if(_verbose){
        std::cout << "Loading (only vector data) from \"" << data_path << "\", shape: " << size_n << " X " << size_d << std::endl;
    }

    d_type * data = (d_type *)std::aligned_alloc(_align, size_n * stride);
    if(_align) memset(data, 0, size_n * stride);
    infile.seekg(96 + offsetData, std::ios::beg);

    char * ptr = (char *)data;
    for(size_t i = 0; i < size_n; i++){
        infile.read(ptr, size_d * sizeof(d_type));
        infile.seekg(offsetData + sizeof(size_t), std::ios::cur);
        ptr += stride;
    }
    infile.clear();
    infile.close();

    if(_verbose){
        std::cout << "Successfully loaded" << std::endl;
    }
    return data;
}

inline bool endsWith(std::string const &str, std::string const &suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

void checkDirectory(std::string dir) {

    if(dir.back() == '/') dir.pop_back();

    if(std::filesystem::exists(dir)){
        if(!std::filesystem::is_directory(dir)){
            std::cerr << "\"" << dir << "\" is a file" << std::endl;
        }
        auto perms = std::filesystem::status(dir).permissions();
        if ((perms & std::filesystem::perms::owner_write) == std::filesystem::perms::none &&
            (perms & std::filesystem::perms::group_write) == std::filesystem::perms::none &&
            (perms & std::filesystem::perms::others_write) == std::filesystem::perms::none){
            std::cerr << "directory \"" << dir << "\" cannot open for write" << std::endl;
        }
    }else{
        if(!std::filesystem::create_directories(dir)){
             std::cerr << "directory \"" << dir << "\" failed to create" << std::endl;
        }
        std::cout << "Directory created: \"" << dir << "\"" << std::endl;
    }
}

};

