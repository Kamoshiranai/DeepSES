// thrust_wrapper.cuh
#ifndef THRUST_WRAPPER_CUH
#define THRUST_WRAPPER_CUH

#include <cuda_runtime.h> // For cudaStream_t definition

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Performs Thrust exclusive scan and reduction on GPU flags.
 *
 * @param d_flags Device pointer to the input flags array (int*, size num_patches).
 *               0 indicates patch is not selected, 1 indicates selected.
 * @param d_scan_output Device pointer to the output array for exclusive scan results
 *                      (int*, size num_patches). scan_output[i] will contain the
 *                      number of selected patches before index i.
 * @param num_patches The total number of patches (size of the arrays).
 * @param stream The CUDA stream to execute the operations on.
 * @return The total number of selected patches (sum of flags).
 */
int perform_thrust_scan_and_reduce(
    int* d_flags,
    int* d_scan_output,
    int num_patches,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // THRUST_WRAPPER_CUH