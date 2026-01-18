#pragma once

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <random>
#include <algorithm>

namespace MemInfo{

inline int parseLine(char *line)
{
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char *p = line;
    while (*p < '0' || *p > '9')
        p++;
    line[i - 3] = '\0';
    i = atoi(p);
    return i;
}
double getMemoryUsage()
{
    FILE *file = fopen("/proc/self/status", "r");
    int highwater_mark = -1;
    int current_memory = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL)
    {
        if (strncmp(line, "VmHWM:", 6) == 0)
        {
            highwater_mark = parseLine(line);
        }

        if (strncmp(line, "VmRSS:", 6) == 0)
        {
            current_memory = parseLine(line);
        }
        if (highwater_mark > 0 && current_memory > 0)
        {
            break;
        }
    }
    fclose(file);
    return (double)1.0 * highwater_mark / 1024;
}
};

namespace utils{

void GenRandom(std::mt19937 & rng, auto * addr, size_t size, size_t N) {
    if (N == size) {
        for (uint32_t i = 0; i < size; ++i) {
            addr[i] = i;
        }
        return;
    }
    for (uint32_t i = 0; i < size; ++i) {
        addr[i] = rng() % (N - size);
    }
    std::sort(addr, addr + size);
    for (uint32_t i = 1; i < size; ++i) {
        if (addr[i] <= addr[i - 1]) {
            addr[i] = addr[i - 1] + 1;
        }
    }
    uint32_t off = rng() % N;
    for (uint32_t i = 0; i < size; ++i) {
        addr[i] = (addr[i] + off) % N;
    }
}

float getRecall(const auto & anng, const auto & knng, size_t checkK) {
    size_t hit = 0;
    size_t checkN = std::min(anng.size(), knng.size());

    for(size_t i = 0; i < checkN; ++i) {
        auto &ann = anng[i];
        auto &knn = knng[i];
        for(size_t j = 0; j < checkK && j < ann.size(); ++j) {
            auto idx = ann[j];
            for(size_t l = 0; l < checkK; ++l) {
                auto nb = knn[l];
                if(idx == nb) { ++hit; break; }
            }
        }
    }
    return 1.0 * hit / (checkK * checkN);
}

};