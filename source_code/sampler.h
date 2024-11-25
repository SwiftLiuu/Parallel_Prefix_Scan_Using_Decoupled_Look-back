////////////////////////////////////////////////////////////////////////////////
// CAUTION: DO NOT MODIFY OR SUBMIT THIS FILE
////////////////////////////////////////////////////////////////////////////////

#ifndef SAMPLER_H
#define SAMPLER_H

#include <vector>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <algorithm>

class Sampler
{
    static constexpr int kbest = 5;
    static constexpr int maxsamples = 20;
    static constexpr double epsilon = 0.01;
    static constexpr int cache_bytes = 1 << 24;
    static constexpr int cache_block = 64;

    volatile int sink = 0;

    int sample_count = 0;
    std::vector<uint32_t> samples;
    std::vector<uint8_t> cache_buf = std::vector<uint8_t>(cache_bytes);

    void addSample(uint32_t sample)
    {
        ++sample_count;
        if (samples.size() <= kbest)
        {
            samples.push_back(sample);
        }
        else
        {
            if (sample < samples.back())
            {
                samples.back() = sample;
            }
        }
        std::sort(samples.begin(), samples.end());
    }

    bool hasConverged()
    {
        return (sample_count >= maxsamples) ||
               ((samples.size() >= kbest) && (samples.back() <= (1 + epsilon) * samples.front()));
    }

    void flushCache()
    {
        int x = sink;
        int *cptr, *cend;
        int incr = cache_block / sizeof(int);
        cptr = (int *)cache_buf.data();
        cend = cptr + cache_bytes / sizeof(int);
        while (cptr < cend)
        {
            x += *cptr;
            cptr += incr;
        }
        sink = x;
    }

public:
    template<typename FuncType, typename... ArgTypes>
    uint32_t sample(FuncType &&func, ArgTypes &&...args)
    {
        samples.clear();
        sample_count = 0;

        do
        {
            auto start = std::chrono::high_resolution_clock::now();
            std::invoke(std::forward<FuncType>(func), std::forward<ArgTypes>(args)...);
            auto end = std::chrono::high_resolution_clock::now();
            uint32_t duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            addSample(duration);
        } while (!hasConverged());

        return samples.front();
    }
};

#endif // SAMPLER_H