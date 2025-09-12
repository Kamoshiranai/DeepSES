// thrust_wrapper.cu
#include "thrust_wrapper.cuh" // Include the header for the function declaration

#include <thrust/device_vector.h> // Although we use device_ptr, including headers is safe
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h> // For device execution policies

// Include necessary CUDA headers if not already included by Thrust/execution_policy
#include <cuda_runtime.h>


// Define the function declared in the header
int perform_thrust_scan_and_reduce(
    int* d_flags,
    int* d_scan_output,
    int num_patches,
    cudaStream_t stream
) {
    // --- Error Checking (Highly Recommended) ---
    // Add checks here to ensure pointers are not null, num_patches > 0 etc.
    // Example:
    if (!d_flags || !d_scan_output || num_patches <= 0) {
        fprintf(stderr, "Error: Invalid arguments to perform_thrust_scan_and_reduce\n");
        return -1; // Indicate error
    }

    try {
        // Wrap raw device pointers with Thrust device_ptr
        thrust::device_ptr<int> d_flags_ptr = thrust::device_pointer_cast(d_flags);
        thrust::device_ptr<int> d_scan_output_ptr = thrust::device_pointer_cast(d_scan_output);

        // Get the CUDA execution policy associated with the stream
        auto policy = thrust::cuda::par.on(stream);

        // --- Perform Exclusive Scan ---
        // Input range: d_flags_ptr to d_flags_ptr + num_patches
        // Output: d_scan_output_ptr
        thrust::exclusive_scan(
            policy,
            d_flags_ptr,                // Start of input range
            d_flags_ptr + num_patches,  // End of input range
            d_scan_output_ptr           // Start of output range
        );

        // --- Perform Reduction (Sum) ---
        // Input range: d_flags_ptr to d_flags_ptr + num_patches
        // Result: sum of elements
        int selected_count = thrust::reduce(
            policy,
            d_flags_ptr,                // Start of input range
            d_flags_ptr + num_patches   // End of input range
        );

        // --- Optional: Synchronize Stream ---
        // Depending on your workflow, you might need to synchronize the stream
        // *after* calling this function in your .cc code, or potentially here
        // if subsequent CPU logic depends immediately on selected_count.
        // Generally, synchronizing later is better for performance.
        // cudaError_t syncErr = cudaStreamSynchronize(stream);
        // if (syncErr != cudaSuccess) {
        //     fprintf(stderr, "CUDA stream sync error after Thrust: %s\n", cudaGetErrorString(syncErr));
        //     return -1; // Indicate error
        // }

        return selected_count;

    } catch (const std::exception& e) {
        fprintf(stderr, "Thrust error: %s\n", e.what());
        return -1; // Indicate error
    } catch (...) {
        fprintf(stderr, "Unknown Thrust error occurred.\n");
        return -1; // Indicate error
    }
}