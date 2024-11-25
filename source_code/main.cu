////////////////////////////////////////////////////////////////////////////////
// CAUTION: DO NOT MODIFY OR SUBMIT THIS FILE
////////////////////////////////////////////////////////////////////////////////

#include <cuda_device_runtime_api.h>
#include <vector>
#include <iostream>
#include <unistd.h>
#include <algorithm>

#include "util.h"
#include "sampler.h"
#include "implementation.h"
#include "util_gpu_err_check.cuh"
#include "reference_implementation.h"

void printHelp(char *argv[])
{
    std::cout << "Usage: " << argv[0] << " [-g] [-h]\n";
    std::cout << "Options:\n";
    std::cout << "  -g    Set grade mode\n";
    std::cout << "  -h    Display this help message\n";
}





int main(int argc, char *argv[])
{
    bool grade_mode = false;

    int opt;
    while ((opt = getopt(argc, argv, "gh")) != -1)
    {
        switch (opt)
        {
        case 'g':
            grade_mode = true;
            break;
        case 'h':
            printHelp(argv);
            return 0;
        default:
            std::cerr << "Unknown option: " << opt << "\n";
            printHelp(argv);
            return 1;
        }
    }

    printSubmissionInfo();
    printf("*******************************************************************************************************\n");

    /* generate input */
    constexpr size_t input_size = 100000007u;
    std::vector<int32_t> input(input_size);
    std::vector<int32_t> reference_output(input_size);
    std::vector<int32_t> student_output(input_size);
    generateInput(input.data(), input_size);

    int32_t *d_input, *d_output;
    gpu_err_check(cudaMalloc((void **)&d_input, input_size * sizeof(int32_t)));
    gpu_err_check(cudaMalloc((void **)&d_output, input_size * sizeof(int32_t)));
    gpu_err_check(cudaMemcpy(d_input, input.data(), input_size * sizeof(int32_t), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    gpu_err_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if (grade_mode)
    {
        Sampler sampler;
        std::cout << "Performance Results:" << std::endl;

        uint32_t reference_time =
            sampler.sample(referenceImplementation, input.data(), reference_output.data(), input_size);
        std::cout << "\tTime consumed by the sequential implementation: " << reference_time << "us" << std::endl;

        uint32_t student_time = sampler.sample(implementation, d_input, d_output, input_size);
        std::cout << "\tTime consumed by your implementation: " << student_time << "us" << std::endl;

        std::cout << "\tOptimization Speedup Ratio (nearest integer): "
                  << (int)((double)reference_time / std::max(student_time, 1u) + 0.5) << std::endl;
        printf("*******************************************************************************************************"
               "\n");
    }

    /* verify results */
    std::fill(reference_output.begin(), reference_output.end(), 0);
    std::fill(student_output.begin(), student_output.end(), 0);

    referenceImplementation(input.data(), reference_output.data(), input_size);
    implementation(d_input, d_output, input_size);
    gpu_err_check(
        cudaMemcpyAsync(student_output.data(), d_output, input_size * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    gpu_err_check(cudaStreamSynchronize(stream));
    gpu_err_check(cudaFree(d_input));
    gpu_err_check(cudaFree(d_output));
    gpu_err_check(cudaStreamDestroy(stream));





    if (std::equal(reference_output.begin(), reference_output.end(), student_output.begin()))
    {
        std::cout << "Your implementation is correct." << std::endl;
    }
    else
    {
        std::cerr << "Your implementation is incorrect." << std::endl;
        exit(-1);
    }

    return 0;
}