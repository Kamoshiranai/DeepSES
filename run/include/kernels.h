// kernels.h
#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>
#include <vector_types.h> // For int3, float3, etc.

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

#ifdef __cplusplus
extern "C" {
#endif

// Declare the wrapper function for the kernel launch
void launchReshapeVolumeToPatches(const float* d_input, float* d_output, int volumeDim, int gridSize, int patchSize, dim3 grid, dim3 block, cudaStream_t stream);

void launchReshapeVolumeToPatchesPitched(
    const float* d_input,
    size_t inputPitch,
    size_t inputSlicePitch,
    float* d_output,
    size_t outputPitch,
    size_t outputSlicePitch,
    int volumeDim,
    int gridSize,
    int patchSize,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
);

// Declare the wrapper function for the kernel launch
void launchReshapePatchesToVolume(const float* d_input, float* d_output, int volumeDim, int gridSize, int patchSize, dim3 grid, dim3 block, cudaStream_t stream);

void launchReshapePatchesToVolumePitched(
    const float* d_input,
    size_t inputPitch,
    size_t inputSlicePitch,
    float* d_output,
    size_t outputPitch,
    size_t outputSlicePitch,
    int volumeDim,
    int gridSize,
    int patchSize,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
);

void launchCopyBufferToBuffer(
    cudaPitchedPtr input,
    cudaPitchedPtr output,
    int gridSize,             
    int patchSize,        
    dim3 grid,
    dim3 block,
    cudaStream_t stream
);

void launchInitializeFlagsToInt(
    int* flags,
    int num_patches,
    int value,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
);

void launchInitializeFlagsToFloat(
    float* flags,
    int num_patches,
    float value,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
);

void launchLogicalAndKernel(
    const int* inputA,
    const int* inputB,
    int* output,
    int N,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
);

void launchCheckPatchRange(
    const float* data,
    int* flags,
    int B, int D, int W, int H,
    float min_val, float max_val,
    float epsilon,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
);

void launchGatherPatches(
    const float* source_data,
    float* compact_data,
    const int* flags,
    const int* scan_results,
    int B, int D, int W, int H,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
);

void launchGatherPatchesFromTexture(
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
);

void launchScatterProcessedToSource(
    const float* processed_compact_data,
    float* source_data,
    const int* flags,
    const int* scan_results,
    int B, int D, int W, int H,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
);

void launchScatterPatchesKernel(
    const float* processed_compact_data,
    const float* original_data,
    float* final_output_data,
    const int* flags,
    const int* scan_results,
    int B, int D, int W, int H,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
);

void launchScatterPatchesToSurfaceKernel(
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
);

void launchRaymarchKernel(
    RaymarchParams p,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
);

void launchRaymarchAndCheckRangeKernel(
    RaymarchParams p,
    float min_val, float max_val,
    float epsilon,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
);

void launchCheckRangeKernel(
    RaymarchParams p,
    float min_val, float max_val,
    float epsilon,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // KERNELS_H
