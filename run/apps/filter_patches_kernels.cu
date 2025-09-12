#include <cuda_runtime.h>     // For CUDA memory management
#include <device_launch_parameters.h>
#include <vector_types.h> // For int3, float3, etc.
#include <cmath> // For fabsf
#include <stdio.h> // For printf

struct float4x4 {
    float4 rows[4];
};

struct RaymarchParams {
    // Input Data & Grid Info
    cudaTextureObject_t gridTexture;     // Texture object for distance field
    int3                gridDimsVoxels;        // Total grid dimensions in voxels (e.g., 512x512x512)
    float               gridRes;         // Size of one voxel in world units
    // float3              gridOriginOffset;// Precalculated offset (gridDimsVoxels * gridRes * 0.5f)
  
    // Patch Info
    int3                patchDimsVoxels; // Size of one patch in voxels (e.g., 64x64x64)
    int3                patchesPerDim;   // Number of patches per axis (e.g., 8x8x8)
    int                 totalPatches;    // Total number of patches (patchesPerDim.x * y * z)
  
    // Camera Info
    float3              cameraPos;       // Camera position in world space
    float3              cameraFront;     // Direction the camera is facing
    float4x4            invViewProj;     // Combined Inverse View * Projection Matrix
  
    // Screen Info
    uint2               resolution;      // Screen width/height in pixels
  
    // Raymarching Params
    float               epsilon;
    int                 maxSteps;
  
    // Output Buffer
    int*                visitedPatches;  // Device pointer to output flags buffer
  };

// Kernel to initialize flags to 0 (assume patch is initially outside of range)
__global__ void initializeFlagsToInt(int *flags, int bufferSize, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < bufferSize) {
        flags[idx] = value; // Initialize to 0 (assume outside of range)
    }
}

// Wrapper function that can be called from C/C++.
extern "C" void launchInitializeFlagsToInt(
    int* flags,
    int bufferSize,
    int value,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
) {
    initializeFlagsToInt<<<grid, block, 0, stream>>>(
        flags,
        bufferSize,
        value
    );
}

// Kernel to initialize flags to 0 (assume patch is initially outside of range)
__global__ void initializeFlagsToFloat(float *flags, int bufferSize, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < bufferSize) {
        flags[idx] = value; // Initialize to 0 (assume outside of range)
    }
}

// Wrapper function that can be called from C/C++.
extern "C" void launchInitializeFlagsToFloat(
    float* flags,
    int bufferSize,
    float value,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
) {
    initializeFlagsToFloat<<<grid, block, 0, stream>>>(
        flags,
        bufferSize,
        value
    );
}

//NOTE: drop in replacement for raymarchAndCheckRangeKernel, but only checks range (view independet)

__global__ void checkRangeKernel(
    RaymarchParams p,
    float min_val, float max_val,
    float epsilon               // Tolerance for float comparison 
) {
    // Use block index directly as the patch index (Batch dimension)
    int patch_idx = blockIdx.x;
    if (patch_idx >= p.totalPatches) return;

    // Thread index within the block
    int thread_idx = threadIdx.x;
    int block_size = blockDim.x;

    long long patch_voxels = (long long)p.patchDimsVoxels.x * p.patchDimsVoxels.y * p.patchDimsVoxels.z;
    long long patch_start_index = (long long)patch_idx * patch_voxels;
    long long global_idx; // current voxel

    float value; // value at current voxel

    // If flag is already 0 (set by another thread), no need to check further
    // Optional optimization, adds global read overhead.
    // if (atomicOr(&flags[patch_idx], 0) == 0) return; // Check current val before loop

    for (long long voxel_offset = thread_idx; voxel_offset < patch_voxels; voxel_offset += block_size) { //NOTE: apparently more efficient, as else more threads would need to write to same location in flags via atomic operation
        // Early exit if another thread already marked this patch as invalid
        if (p.visitedPatches[patch_idx] == 1) {
            break;
        }

        // // NOTE: only hotfix for comparison
        // atomicExch(&p.visitedPatches[patch_idx], 1);
        // break;

        global_idx = patch_start_index + voxel_offset;

        // calculate (normalized) 3d coordinates in grid
        int coords_grid_z = global_idx / (p.gridDimsVoxels.x * p.gridDimsVoxels.y);
        int temp_idx = global_idx % (p.gridDimsVoxels.x * p.gridDimsVoxels.y);
        int coords_grid_y = temp_idx / p.gridDimsVoxels.x;
        int coords_grid_x = temp_idx % p.gridDimsVoxels.x;

        float norm_coords_grid_x = float(coords_grid_x) / p.gridDimsVoxels.x;
        float norm_coords_grid_y = float(coords_grid_y) / p.gridDimsVoxels.y;
        float norm_coords_grid_z = float(coords_grid_z) / p.gridDimsVoxels.z;
        value = tex3D<float>(p.gridTexture, norm_coords_grid_x, norm_coords_grid_y, norm_coords_grid_z);

        // Check if value is INSIDE the range [min_val, max_val]
        if (value > (min_val + epsilon) && value < (max_val - epsilon)) {
            atomicExch(&p.visitedPatches[patch_idx], 1); // Mark as relevant (INSIDE range)
            break; // This thread found a relevant value
        }
    }
}

// Wrapper function that can be called from C/C++.
extern "C" void launchCheckRangeKernel(
    RaymarchParams p,
    float min_val, float max_val,
    float epsilon,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
) {
    checkRangeKernel<<<grid, block, 0, stream>>>(
        p,
        min_val, max_val,
        epsilon
    );
}

__global__ void logicalAndKernel(const int* inputA, // Pointer from mapped SSBO
    const int* inputB, // Your other CUDA buffer
    int* output,       // Result buffer
    int N)             // Number of elements (512)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Perform logical AND. Since inputs are 0 or 1, bitwise AND (&) works perfectly.
        // If inputs could be other non-zero values, use (inputA[idx] != 0 && inputB[idx] != 0) ? 1 : 0;
        output[idx] = inputA[idx] & inputB[idx];
        // output[idx] = inputB[idx];
    }
}

// Wrapper function that can be called from C/C++.
extern "C" void launchLogicalAndKernel(
    const int* inputA,
    const int* inputB,
    int* output,
    int N,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
) {
    logicalAndKernel<<<grid, block, 0, stream>>>(
        inputA,
        inputB,
        output,
        N
    );
}

// Kernel to GATHER selected patches from a source Texture into a compact buffer
__global__ void gatherPatchesFromTexture(
    cudaTextureObject_t sourceTex,              // Source 3D texture (e.g., 512x512x512)
    float *__restrict__ compact_data,           // Destination compact buffer (e.g., N_active * 64 * 64 * 64)
    const int *__restrict__ flags,              // Flags indicating selection (0 or 1), size = totalSourcePatches
    const int *__restrict__ scan_results,       // Exclusive scan results (destination index), size = totalSourcePatches
    int patchSize,                              // Dimension of one patch (e.g., 64)
    int patchesPerDim,                          // Patches along one dim in source (e.g., 8 for 512/64)
    int totalSourcePatches                      // Total number of patches in source (e.g., 512 = 8*8*8)
) {
    // blockIdx.x represents the original source patch index (0 to totalSourcePatches-1)
    int source_patch_idx = blockIdx.x;

    // Bounds check for the source patch index
    if (source_patch_idx >= totalSourcePatches) return;

    // --- Selection Check ---
    // Only process patches that were selected (flag == 1)
    if (flags[source_patch_idx] == 0) {
        return; // Entire block exits if this source patch is not selected
    }

    // --- Destination Index ---
    // Get the destination index in the compact buffer from the exclusive scan result
    int dest_patch_idx = scan_results[source_patch_idx];

    // --- Calculate Patch Geometry ---
    long long patch_voxels = (long long)patchSize * patchSize * patchSize;

    // Calculate the base linear index for the start of the destination patch
    long long dest_patch_start_idx = (long long)dest_patch_idx * patch_voxels;

    // Calculate the 3D index (px, py, pz) of the *source* patch
    int src_pz = source_patch_idx / (patchesPerDim * patchesPerDim);
    int temp_idx = source_patch_idx % (patchesPerDim * patchesPerDim);
    int src_py = temp_idx / patchesPerDim;
    int src_px = temp_idx % patchesPerDim;

    // Calculate the starting global coordinates (voxel indices) for the *source* patch
    int src_startX = src_px * patchSize;
    int src_startY = src_py * patchSize;
    int src_startZ = src_pz * patchSize;

    // --- Grid-Stride Loop to Copy Voxels ---
    int thread_idx = threadIdx.x;
    int block_size = blockDim.x; // Number of threads cooperating on this patch copy

    for (long long voxel_offset = thread_idx; voxel_offset < patch_voxels; voxel_offset += block_size) {
        // Calculate the 3D coordinates (rel_x, rel_y, rel_z) *within* the patch
        // from the linear voxel_offset within the patch.
        // Assumes Z-major layout for offset calculation: Z varies slowest, X fastest
        int patch_slice_size = patchSize * patchSize;
        int rel_z = voxel_offset / patch_slice_size;
        long long temp_offset = voxel_offset % patch_slice_size;
        int rel_y = temp_offset / patchSize;
        int rel_x = temp_offset % patchSize;

        // Calculate the global 3D coordinates in the source texture (normalized!)
        float globalX = (src_startX + rel_x) * 1.0f / (patchSize * patchesPerDim);
        float globalY = (src_startY + rel_y) * 1.0f / (patchSize * patchesPerDim);
        float globalZ = (src_startZ + rel_z) * 1.0f / (patchSize * patchesPerDim);

        // --- Read from source texture ---
        // Replace 'float' if your texture stores a different data type!
        float source_value = tex3D<float>(sourceTex, globalX, globalY, globalZ);

        // Calculate the linear destination index in the compact buffer
        long long dest_idx = dest_patch_start_idx + voxel_offset;

        // --- Write to destination buffer ---
        compact_data[dest_idx] = source_value;
    }
}

// Wrapper function that can be called from C/C++.
extern "C" void launchGatherPatchesFromTexture(
    cudaTextureObject_t sourceTex,
    float* compact_data,
    const int* flags,
    const int* scan_results,
    int patchSize, 
    int patchesPerDim,
    int totalSourcePatches,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
) {
    gatherPatchesFromTexture<<<grid, block, 0, stream>>>(
        sourceTex,
        compact_data,
        flags,
        scan_results,
        patchSize,
        patchesPerDim,
        totalSourcePatches
    );
}

// Kernel to SCATTER processed patches into a destination Surface
// and copy unprocessed patches from a source Texture to the destination Surface.
__global__ void scatterPatchesToSurfaceKernel(
    const float *__restrict__ processed_compact_data, // Compact buffer of processed patches (read)
    cudaTextureObject_t originalSourceTex,          // Original source data texture (read for unprocessed)
    cudaSurfaceObject_t finalOutputSurf,            // Final destination surface (write)
    const int *__restrict__ flags,                  // Flags indicating which patches were processed (0 or 1)
    const int *__restrict__ scan_results,           // Exclusive scan results (index into compact data)
    int patchSize,                                  // Dimension of one patch (e.g., 64)
    int patchesPerDim,                              // Patches along one dim (e.g., 8 for 512/64)
    int totalPatches                                // Total number of patches (e.g., 512 = 8*8*8)
) {
    // blockIdx.x represents the destination patch index (0 to totalPatches-1)
    int patch_idx = blockIdx.x;

    // Bounds check for the patch index
    if (patch_idx >= totalPatches) return;

    // //DEBUG:
    // if (patch_idx % 50 == 0 && threadIdx.x == 0) {
    //     printf("Linear patch idx:%d\n", patch_idx);
    // }

    // --- Calculate Patch Geometry ---
    long long patch_voxels = (long long)patchSize * patchSize * patchSize;

    // Calculate the 3D index (px, py, pz) of the current *destination* patch
    int pz = patch_idx / (patchesPerDim * patchesPerDim);
    int temp_idx = patch_idx % (patchesPerDim * patchesPerDim);
    int py = temp_idx / patchesPerDim;
    int px = temp_idx % patchesPerDim;

    // Calculate the starting global coordinates (voxel indices) for this destination patch
    int startX = px * patchSize;
    int startY = py * patchSize;
    int startZ = pz * patchSize;

    // //DEBUG:
    // if (patch_idx % 50 == 0 && threadIdx.x == 0) {
    //     printf("3D patch idx:(%d, %d, %d)\n", px, py, pz);
    // }

    // --- Grid-Stride Loop over Voxels within the Patch ---
    int tid = threadIdx.x;          // Thread index within the block
    int block_size = blockDim.x;    // Number of threads cooperating on this patch

    // Determine the source location (either compact buffer or original texture)
    bool wasProcessed = (flags[patch_idx] == 1);

    // //DEBUG:
    // if (patch_idx % 50 == 0 && threadIdx.x == 0) {
    //     printf("wasProcessed: %d\n", wasProcessed);
    // }

    int compact_patch_idx = -1;
    long long compact_patch_start_idx = (long long) -1;
    if (wasProcessed) {
        compact_patch_idx = scan_results[patch_idx]; // Index in the compact buffer
        compact_patch_start_idx = (long long)compact_patch_idx * patch_voxels;
    }

    for (long long voxel_offset = tid; voxel_offset < patch_voxels; voxel_offset += block_size) {
        // Calculate the 3D coordinates (rel_x, rel_y, rel_z) *within* the patch
        int patch_slice_size = patchSize * patchSize;
        int rel_z = voxel_offset / patch_slice_size;
        long long temp_offset = voxel_offset % patch_slice_size;
        int rel_y = temp_offset / patchSize;
        int rel_x = temp_offset % patchSize;

        // Calculate the global 3D coordinates for writing to the output surface
        // These are also the coordinates for reading from the original texture if needed
        int outputX = startX + rel_x;
        int outputY = startY + rel_y;
        int outputZ = startZ + rel_z;

        //DEBUG:
        // if (patch_idx % 50 == 0 && threadIdx.x == 0) {
        //     printf("Coordinate within patch:(%d, %d, %d), Output Coordinates: (%d, %d, %d)\n", rel_x, rel_y, rel_z, outputX, outputY, outputZ);
        // }

        float value_to_write;

        if (wasProcessed) {
            // --- This patch WAS processed ---
            // Read from the compact processed data buffer
            long long compact_idx = compact_patch_start_idx + voxel_offset;
            value_to_write = processed_compact_data[compact_idx];

            // --- Write the determined value to the destination surface ---
            // surf3Dwrite uses byte offset for X coordinate!
            surf3Dwrite(value_to_write, finalOutputSurf, outputX * sizeof(float), outputY, outputZ);
        }

            // //DEBUG:
            // if (patch_idx % 50 == 0 && threadIdx.x == 0) {
            //     printf("compact_idx: %d, value to write:%.2f\n", compact_idx, value_to_write);
            // }

        // } else {
        //     // --- This patch was NOT processed ---
        //     // Read the original value from the source texture
        //     // normalize coordinates for texture read
        //     float texCoords_X = outputX * 1.0f / 512;
        //     float texCoords_Y = outputY * 1.0f / 512;
        //     float texCoords_Z = outputZ * 1.0f / 512;

        //     value_to_write = tex3D<float>(originalSourceTex, texCoords_X, texCoords_Y, texCoords_Z);
        // }

        // // --- Write the determined value to the destination surface ---
        // // surf3Dwrite uses byte offset for X coordinate!
        // surf3Dwrite(value_to_write, finalOutputSurf, outputX * sizeof(float), outputY, outputZ);
    }
}

// Wrapper function that can be called from C/C++.
extern "C" void launchScatterPatchesToSurfaceKernel(
    const float* processed_compact_data,
    cudaTextureObject_t originalSourceTex,
    cudaSurfaceObject_t finalOutputSurf,
    const int* flags,
    const int* scan_results,
    int patchSize,
    int patchesPerDim,
    int totalPatches,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
) {
    scatterPatchesToSurfaceKernel<<<grid, block, 0, stream>>>(
        processed_compact_data,
        originalSourceTex,
        finalOutputSurf,
        flags,
        scan_results,
        patchSize, patchesPerDim, totalPatches
    );
}