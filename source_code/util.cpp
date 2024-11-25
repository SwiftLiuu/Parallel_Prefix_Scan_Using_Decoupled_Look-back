////////////////////////////////////////////////////////////////////////////////
// CAUTION: DO NOT MODIFY OR SUBMIT THIS FILE
////////////////////////////////////////////////////////////////////////////////

#include "util.h"

#include <random>

void generateInput(int32_t *h_input, size_t size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dist(
        std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());

    int32_t sum = 0;
    for (size_t i = 0; i < size; ++i)
    {
        int next_val;
        do {
            next_val = dist(gen);
        } while (__builtin_add_overflow_p(sum, next_val, sum));
        sum += next_val;
        h_input[i] = next_val;
    }
}
