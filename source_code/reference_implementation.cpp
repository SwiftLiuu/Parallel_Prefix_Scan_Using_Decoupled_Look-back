////////////////////////////////////////////////////////////////////////////////
// CAUTION: DO NOT MODIFY OR SUBMIT THIS FILE
////////////////////////////////////////////////////////////////////////////////

#include "reference_implementation.h"

void referenceImplementation(const int32_t *h_input, int32_t *h_output, size_t size) {
    if (size == 0) {
        return;
    }
    h_output[0] = h_input[0];
    for (size_t i = 1; i < size; ++i) {
        h_output[i] = h_output[i - 1] + h_input[i];
    }
}
