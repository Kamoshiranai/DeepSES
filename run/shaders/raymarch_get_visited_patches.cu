#include <cuda_runtime.h>     // For CUDA memory management
#include <device_launch_parameters.h>
#include <vector_types.h> // For int3, float3, etc.
#include <vector_functions.hpp> // For make_int3, floor, etc.
#include "helper_math.h"
#include <stdio.h> // For printf

struct float4x4 {
    float4 rows[4];
};

__device__ inline float4 mat_vec_mul(const float4x4& M, const float4& v) {
    float4 result;
    // Manual expansion or using dot products if available
    result.x = M.rows[0].x * v.x + M.rows[0].y * v.y + M.rows[0].z * v.z + M.rows[0].w * v.w;
    result.y = M.rows[1].x * v.x + M.rows[1].y * v.y + M.rows[1].z * v.z + M.rows[1].w * v.w;
    result.z = M.rows[2].x * v.x + M.rows[2].y * v.y + M.rows[2].z * v.z + M.rows[2].w * v.w;
    result.w = M.rows[3].x * v.x + M.rows[3].y * v.y + M.rows[3].z * v.z + M.rows[3].w * v.w;
    return result;
}

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

// Helper device function for world -> normalized grid coords [0, 1]
// Mirrors glsl pos_in_grid
__device__ inline float3 worldToNormGrid(float3 world_pos, const RaymarchParams& p) {
    float3 pos_relative_to_origin = world_pos + make_float3(p.gridDimsVoxels) * p.gridRes * 0.5; // places the center of the grid in the origin of the world space (has units in Angstrom!)
    float3 voxel_coords_float = pos_relative_to_origin / p.gridRes;
    // Normalize to [0, 1] range for texture sampling
    return voxel_coords_float / make_float3(p.gridDimsVoxels);
}

// Helper device function to transform NDC to World
// Mirrors glsl getWorldPosfromScreenPos logic (partially)
__device__ inline float3 ndcToWorld(float3 ndc, const RaymarchParams& p) {
    float4 clip_pos = make_float4(ndc.x, ndc.y, ndc.z, 1.0f);
    // Apply inverse view-projection matrix
    float4 world_h = mat_vec_mul(p.invViewProj, clip_pos);
    // Perspective divide
    return make_float3(world_h.x / world_h.w, world_h.y / world_h.w, world_h.z / world_h.w);
}

__global__ void raymarchAndCheckRangeKernel(
    RaymarchParams p, 
    float min_val, float max_val,
    float epsilon               // Tolerance for float comparison 
) {
    
    // //DEBUG:
    // if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0) {
    //     printf("Device sizeof(RaymarchParams): %llu\n", (unsigned long long)sizeof(RaymarchParams));
    // }
    
    // --- 1. Calculate Pixel Coordinates ---
    unsigned int x = (blockIdx.x * blockDim.x + threadIdx.x) * 8; //NOTE take only every second/4th pixel or so
    unsigned int y = (blockIdx.y * blockDim.y + threadIdx.y) * 8;

    // Check if pixel is within screen bounds
    if (x >= p.resolution.x || y >= p.resolution.y) {
        return;
    }

    //DEBUG:
    // printf("Kernel launch check: Thread (%u,%u)\n", x, y);

    // --- 2. Calculate Ray Origin and Direction ---
    // Convert pixel coordinates (x, y) to Normalized Device Coordinates (NDC) [-1, 1]
    // Note: OpenGL NDC Y is often inverted compared to image coordinates
    float ndc_x = (2.0f * x / (float)p.resolution.x) - 1.0f;
    float ndc_y = 1.0f - (2.0f * y / (float)p.resolution.y); // Invert Y for typical screen coords
    
    float3 rayOrigin = ndcToWorld(make_float3(ndc_x, ndc_y, 0.0f), p); // Or use z=0/1 depending on projection

    float3 rayDirection = normalize(p.cameraFront);

    // //DEBUG:
    // // --- Ray Info ---
    // // Print information only for the thread processing the center pixel (or close to it)
    // if(x == p.resolution.x / 2 && y == p.resolution.y / 2) {
    //     printf("Center Ray: Origin(%.2f, %.2f, %.2f) Dir(%.3f, %.3f, %.3f)\n",
    //         rayOrigin.x,
    //         rayOrigin.y,
    //         rayOrigin.z,
    //         rayDirection.x,
    //         rayDirection.y,
    //         rayDirection.z);
    // }

    // --- 3. Raymarch ---
    float depth = 0.0f;
    float dist = 2.8f;
    int last_marked_patch_index = -1; // Track last marked patch

    for (int i = 0; i < p.maxSteps; ++i) {
        // Current position in world space
        float3 world_pos = rayOrigin + depth * rayDirection;

        // Convert to normalized grid coordinates for sampling and bounds check
        float3 coords_grid_norm = worldToNormGrid(world_pos, p);

        // Check if the current position is inside the grid bounds [0, 1]
        bool inside_grid = (coords_grid_norm.x >= 0.0f && coords_grid_norm.x <= 1.0f &&
                            coords_grid_norm.y >= 0.0f && coords_grid_norm.y <= 1.0f &&
                            coords_grid_norm.z >= 0.0f && coords_grid_norm.z <= 1.0f);
        
        // //DEBUG:
        // // --- Inside Loop (for center pixel thread, print every N steps) ---
        // // This prevents printing too much data
        // if(x == p.resolution.x / 2 && y == p.resolution.y / 2 && i % 25 == 0) { // Print every 25 steps for the center pixel
        //     printf("Step %d: WorldPos(%.2f, %.2f, %.2f) NormCoord(%.3f, %.3f, %.3f) Inside=%d Depth=%.3f",
        //         i,
        //         world_pos.x,
        //         world_pos.y,
        //         world_pos.z,
        //         coords_grid_norm.x,
        //         coords_grid_norm.y,
        //         coords_grid_norm.z,
        //         inside_grid,
        //         depth);
        // }

        if (inside_grid) {

            // --- Sample Distance Field ---
            // Sample using normalized coordinates
            // float dist = tex3D<float>(p.gridTexture, coords_grid_norm.x, coords_grid_norm.y, coords_grid_norm.z);
            dist = tex3D<float>(p.gridTexture, coords_grid_norm.x, coords_grid_norm.y, coords_grid_norm.z);

            // //DEBUG:
            // // This prevents printing too much data
            // if(x == p.resolution.x / 2 && y == p.resolution.y / 2 && i % 25 == 0) { // Print every 25 steps for the center pixel
            //     printf("Dist=%.3f\n", dist);
            // }
            
            // --- Check if sdf value is in range ---
            if (dist > (min_val + epsilon) && dist < (max_val - epsilon)) {

                // --- Mark Visited Patch ---
                // Calculate voxel coordinates (non-normalized)
                float3 voxel_coord_float = coords_grid_norm * make_float3(p.gridDimsVoxels);

                // Calculate 3D patch index (integer division effectively floors)
                // Clamp coordinates slightly inwards to avoid boundary issues with floor/int conversion
                // Alternatively, use floor directly and clamp integer coords
                float3 safe_voxel_coord = clamp(voxel_coord_float, make_float3(0.0f), make_float3(p.gridDimsVoxels) - make_float3(1.0f));
                int3 patch_coord_3d = make_int3(floorf(safe_voxel_coord / make_float3(p.patchDimsVoxels)));

                // // Integer-based approach:
                // int3 voxel_coord_int = make_int3(floor(voxel_coord_float));
                // // Clamp integer voxel coordinates
                // voxel_coord_int.x = max(0, min(voxel_coord_int.x, p.gridDimsVoxels.x - 1));
                // voxel_coord_int.y = max(0, min(voxel_coord_int.y, p.gridDimsVoxels.y - 1));
                // voxel_coord_int.z = max(0, min(voxel_coord_int.z, p.gridDimsVoxels.z - 1));

                // int3 patch_coord_3d = voxel_coord_int / p.patchDimsVoxels; // Integer division

                // // Clamp patch coordinates (robustness, though shouldn't be needed if voxel clamp is correct)
                // patch_coord_3d.x = max(0, min(patch_coord_3d.x, p.patchesPerDim.x - 1));
                // patch_coord_3d.y = max(0, min(patch_coord_3d.y, p.patchesPerDim.y - 1));
                // patch_coord_3d.z = max(0, min(patch_coord_3d.z, p.patchesPerDim.z - 1));


                // Calculate linear patch index
                int patch_index_linear = patch_coord_3d.x +
                                        patch_coord_3d.y * p.patchesPerDim.x +
                                        patch_coord_3d.z * p.patchesPerDim.x * p.patchesPerDim.y;

                // //DEBUG:
                // if(x == p.resolution.x / 2 && y == p.resolution.y / 2) {
                //     printf("  Inside Grid: VoxelF(%.1f, %.1f, %.1f) Patch3D(%d, %d, %d) Index=%d\n",
                //            voxel_coord_float.x,
                //            voxel_coord_float.y,
                //            voxel_coord_float.z,
                //            patch_coord_3d.x,
                //            patch_coord_3d.y,
                //            patch_coord_3d.z,
                //            patch_index_linear);
                // }

                // Optimization: Only perform atomic if we entered a *new* patch and check bounds
                if (patch_index_linear != last_marked_patch_index &&
                    patch_index_linear >= 0 && patch_index_linear < p.totalPatches) // Sanity bounds check
                {
                    // *** Perform atomic operation on the output buffer ***
                    // //DEBUG:
                    // // --- Right Before Atomic (for center pixel thread) ---
                    // if(x == p.resolution.x / 2 && y == p.resolution.y / 2) {
                    //     printf("    Attempting atomicMax on index %d (Last marked: %d)\n",
                    //         patch_index_linear,
                    //         last_marked_patch_index); // Show last marked for context
                    // }

                    atomicMax(&p.visitedPatches[patch_index_linear], 1);
                    last_marked_patch_index = patch_index_linear; // Update last marked patch
                }
            }

            // --- Advance Ray ---
            depth += p.gridRes; // advance by a voxel

        } else if (last_marked_patch_index != -1) {
            break; // Exit loop if we were inside and just stepped out

        } else {
            // not yet inside the grid
            // --- Advance Ray ---
            depth += 1.4; //NOTE: advance by probe_radius
        }

        // --- 6. Check for Hit ---
        if (dist < p.epsilon) {
            break; // Hit surface
        }
    } // End raymarch loop
}

// Wrapper function that can be called from C/C++.
extern "C" void launchRaymarchAndCheckRangeKernel(
    RaymarchParams p,
    float min_val, float max_val,
    float epsilon,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
) {
    raymarchAndCheckRangeKernel<<<grid, block, 0, stream>>>(
        p,
        min_val, max_val,
        epsilon
    );
}