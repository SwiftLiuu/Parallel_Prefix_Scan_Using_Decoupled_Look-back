#include "implementation.h"

#include "stdio.h"

void printSubmissionInfo()
{
    // This will be published in the leaderboard on piazza
    // Please modify this field with something interesting
    char nick_name[] = "Decoupled-Look-Back-Team";

    // Please fill in your information (for marking purposes only)
    char student_first_name[] = "Xiangyu";
    char student_last_name[] = "Liu";
    char student_student_number[] = "1006743179";

    // Printing out team information
    printf("*******************************************************************************************************\n");
    printf("Submission Information:\n");
    printf("\tnick_name: %s\n", nick_name);
    printf("\tstudent_first_name: %s\n", student_first_name);
    printf("\tstudent_last_name: %s\n", student_last_name);
    printf("\tstudent_student_number: %s\n", student_student_number);
}

/**
 * Implement your CUDA inclusive scan here. Feel free to add helper functions, kernels or allocate temporary memory.
 * However, you must not modify other files. CAUTION: make sure you synchronize your kernels properly and free all
 * allocated memory.
 *
 * @param d_input: input array on device
 * @param d_output: output array on device
 * @param size: number of elements in the input array
 */





__global__ void koggeStoneDecoupledLookbackKernel(const int32_t *d_input, int32_t *d_output, volatile uint64_t *d_aggregates, size_t size, size_t block_workload) {
    extern __shared__ int32_t shared_mem[]; 
    int32_t *s_data = shared_mem;           

    int tid = threadIdx.x;                  
    int blockStart = blockIdx.x * block_workload; 
    int end = min(blockStart + block_workload, size); // End index for this block

    // First Offset Calculation Directly Using Global Memory (offset = 1)
    for (int i = blockStart + tid; i < end; i += blockDim.x) {
        if (i > blockStart) { 
            s_data[i - blockStart] = d_input[i] + d_input[i - 1];
        } else {
            // First element has no previous element
            s_data[i - blockStart] = d_input[i];
        }
    }
    __syncthreads();

     // Perform Remaining Offsets Using Shared Memory (offset = 2, 4, ...)
    for (int offset = 2; offset < block_workload; offset *= 2) {
        // Calculate the lower bound for valid iterations in this offset
       int lower_bound = 0 - (blockDim.x - (block_workload - offset) % blockDim.x);

       // Reverse iteration ensures updates are not read by other threads
       for (int i = block_workload - 1 - tid; i - offset >= lower_bound; i -= blockDim.x) {
           int32_t local_temp = 0;
   
           if (i >= 0 && i - offset >= 0) {
               local_temp = s_data[i - offset];
           }
       
   
           // Synchronize threads to ensure all threads have finished reading
           __syncthreads();
   
           if (i >= 0) { 
               s_data[i] += local_temp;
           }
   
       }
   
       // Synchronize threads after completing all updates for the current offset
       __syncthreads();
   }



    // Thread 0 sets the aggregate and flag
    if (tid == 0) {
        int32_t block_aggregate = s_data[block_workload - 1]; 
        uint64_t encoded_value = ((uint64_t)block_aggregate << 2); // Shift aggregate to the left by 2 to spare spaces for the flag

        if (blockIdx.x == 0) {
            d_aggregates[blockIdx.x] = encoded_value | 0b10; // Set flag to 'P' (pre-aggregate available)
        } else {
            d_aggregates[blockIdx.x] = encoded_value | 0b01; // Set flag to 'A' (aggregate available)
        }
        __threadfence();
    }
    



    // Decoupled Lookback for Pre-Aggregate (Only for Non-Block 0)
    uint64_t pre_aggregate = 0;
    if (blockIdx.x > 0 && tid == 0) {
        for (int i = blockIdx.x - 1; i >= 0; i--) {
            uint64_t value;
            do {
                value = d_aggregates[i];
            } while ((value & 0b11) == 0); // Poll until flag is not 'X'

            uint64_t aggregate = value >> 2;
            uint64_t flag = value & 0b11;

            if (flag == 0b01) { // 'A' (aggregate available)
                pre_aggregate += aggregate;
            } else if (flag == 0b10) { // 'P' (pre-aggregate available)
                pre_aggregate += aggregate;
                break; // Stop looking back
            }
        }

        // Update d_aggregates with pre-aggregate + block_aggregate, set flag to 'P'
        uint64_t block_aggregate = s_data[block_workload - 1];
        uint64_t final_aggregate = pre_aggregate + block_aggregate;
        d_aggregates[blockIdx.x] = (final_aggregate << 2) | 0b10; // Set flag to 'P'
        __threadfence();
    }




    // Share the pre-aggregate within the block
    __shared__ uint64_t shared_pre_aggregate;
    if (tid == 0) {
        shared_pre_aggregate = pre_aggregate;
    }
    __syncthreads();
    pre_aggregate = shared_pre_aggregate;


    // Add pre-aggregate to all elements and write results to global memory
    for (int i = tid; i < block_workload; i += blockDim.x) {
        if (blockStart + i < size) {
            d_output[blockStart + i] = s_data[i] + pre_aggregate;
        }
    }
}





void implementation(const int32_t *d_input, int32_t *d_output, size_t size) {
    const size_t threadsPerBlock = 512;     
    const size_t block_workload = 8160;   
    const size_t blocks = (size + block_workload - 1) / block_workload;

    // Allocate memory for the block aggregates (64-bit for encoding)
    uint64_t *d_aggregates;
    cudaMalloc(&d_aggregates, blocks * sizeof(uint64_t));
    cudaMemset(d_aggregates, 0, blocks * sizeof(uint64_t)); // Initialize to 'X' (invalid)

    size_t sharedMemorySize = block_workload * sizeof(int32_t);

    koggeStoneDecoupledLookbackKernel<<<blocks, threadsPerBlock, sharedMemorySize>>>(
        d_input, d_output, d_aggregates, size, block_workload);

    cudaFree(d_aggregates);
}


