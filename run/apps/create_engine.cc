#include <NvInfer.h>          // Core TensorRT API
#include <cuda_runtime.h>
#include <NvInferRuntime.h>   // For runtime functionalities
#include <NvOnnxParser.h>     // If parsing ONNX models
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <numeric> // For std::iota
#include <algorithm> // For std::shuffle
#include <stdexcept> // For exceptions
#include <map>
#include <memory> // For std::unique_ptr
#include <hdf5.h>

// #include <dirent.h>   // For opendir, readdir, closedir //NOTE: this package is not availabe under Windows, there is a C++17 version of the file handling below which does not require this...
#include <sys/stat.h> // For stat (optional but recommended for checking file type)
#include <errno.h>    // For error checking (errno)
#include <cstring>    // For strerror

// instantiate logger
class Logger : public nvinfer1::ILogger
{
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        // suppress info-level messages
        if (severity <= nvinfer1::ILogger::Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class VerboseLogger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        // Show all messages including VERBOSE
        if (severity <= nvinfer1::ILogger::Severity::kVERBOSE)
        {
            std::cout << "[TRT] ";
            switch (severity)
            {
                case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: std::cout << "[INTERNAL_ERROR] "; break;
                case nvinfer1::ILogger::Severity::kERROR:          std::cout << "[ERROR] ";          break;
                case nvinfer1::ILogger::Severity::kWARNING:        std::cout << "[WARNING] ";        break;
                case nvinfer1::ILogger::Severity::kINFO:           std::cout << "[INFO] ";           break;
                case nvinfer1::ILogger::Severity::kVERBOSE:        std::cout << "[VERBOSE] ";        break;
                default:                        std::cout << "[UNKNOWN] ";         break;
            }
            std::cout << msg << std::endl;
        }
    }
};

// Helper function to check CUDA calls
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t status = call;                                           \
        if (status != cudaSuccess) {                                         \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status)        \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("CUDA error");                          \
        }                                                                    \
    } while (0)

// Helper function to check HDF5 calls
#define CHECK_HDF5(call)                                                      \
    do {                                                                      \
        herr_t status = call;                                                 \
        if (status < 0) {                                                     \
            std::cerr << "HDF5 Error at " << __FILE__ << ":" << __LINE__      \
                      << std::endl;                                           \
            /* H5Eprint(H5E_DEFAULT, stderr); // Optional: Print detailed HDF5 error stack */ \
            throw std::runtime_error("HDF5 error");                           \
        }                                                                     \
    } while (0)

// NOTE: this can be used for INT8 quantization with tensorRT, but we found better performance with single precision (FP16) on our setup

/**
 * @brief Lists all regular files with a specific extension in a directory (POSIX version).
 * @param directoryPath The path to the directory.
 * @param extension The desired file extension (e.g., ".h5"). Case-sensitive.
 * @return A vector of filenames (not full paths) found in the directory.
 * @throws std::runtime_error on directory access errors.
 */
/* std::vector<std::string> listFilesWithExtensionPosix(const std::string& directoryPath, const std::string& extension) {
    std::vector<std::string> filenames;
    DIR* dirHandle = opendir(directoryPath.c_str());

    if (!dirHandle) {
        throw std::runtime_error("Error opening directory '" + directoryPath + "': " + strerror(errno));
    }

    struct dirent* dirEntry;
    // Reset errno before calling readdir - needed to distinguish end-of-stream from error
    errno = 0;
    while ((dirEntry = readdir(dirHandle)) != nullptr) {
        std::string entryName = dirEntry->d_name;

        // Skip "." and ".." entries
        if (entryName == "." || entryName == "..") {
            continue;
        }

        // Check if it has the desired extension
        if (entryName.length() > extension.length() &&
            entryName.substr(entryName.length() - extension.length()) == extension)
        {

            // OPTION 2: Always use stat (more reliable, slightly slower)
            std::string fullPath = directoryPath;
            if (fullPath.empty() || fullPath.back() != '/') { // Handle trailing slash
                 fullPath += '/';
            }
            fullPath += entryName;

            struct stat statbuf;
            if (stat(fullPath.c_str(), &statbuf) == 0) {
                if (S_ISREG(statbuf.st_mode)) { // Check if it's a regular file
                    filenames.push_back(entryName);
                }
                // else: it's a directory, symlink, etc. with the .h5 extension - ignore it.
            } else {
                // Log error but continue, maybe permissions issue on one file
                std::cerr << "Warning: Could not stat entry " << fullPath << ": " << strerror(errno) << std::endl;
            }
        }
         // Reset errno for the next readdir call
         errno = 0;
    }

    // After the loop, check errno again to see if readdir failed
    if (errno != 0) {
         closedir(dirHandle); // Attempt to close even on error
         throw std::runtime_error("Error reading directory '" + directoryPath + "': " + strerror(errno));
    }


    if (closedir(dirHandle) == -1) {
        // Log error, but we likely have the filenames already
         std::cerr << "Warning: Error closing directory '" << directoryPath << "': " << strerror(errno) << std::endl;
    }

    // Optional: Sort the filenames alphabetically
    std::sort(filenames.begin(), filenames.end());

    return filenames;
}

class HDF5Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    HDF5Calibrator(
        int batchSize,
        int patchSize,
        const std::string& pathVdw, // Only VdW path needed
        const std::vector<std::string>& fileNames,
        int numCalibrationBatches, // How many batches to use for calibration
        const std::string& inputBlobNameVdw, // Name of the VdW input tensor
        const std::string& cacheFileName,
        nvinfer1::ILogger& logger)
        : mBatchSize(batchSize)
        , mPatchSize(patchSize)
        , mPathVdw(pathVdw) // Store VdW path
        , mFileNames(fileNames)
        , mNumCalibrationBatches(numCalibrationBatches)
        , mInputBlobNameVdw(inputBlobNameVdw) // Store VdW input name
        , mCacheFileName(cacheFileName)
        , mLogger(logger)
        , mCurrentBatch(0)
        , mRandEngine(std::random_device{}()) // Seed RNG
    {
        if (mFileNames.empty()) {
            throw std::invalid_argument("File names vector cannot be empty.");
        }
        if (mBatchSize <= 0 || mPatchSize <= 0 || mNumCalibrationBatches <= 0) {
                throw std::invalid_argument("Batch size, patch size, and num calibration batches must be positive.");
        }

        mPatchVolume = static_cast<size_t>(mPatchSize) * mPatchSize * mPatchSize;
        mBatchDataSize = static_cast<size_t>(mBatchSize) * mPatchVolume; // In elements (floats)

        // --- Determine volume dimensions and load VdW data ---
        loadAndPrepareData(); // Now only loads VdW

        // --- Allocate GPU memory for one batch of VdW input ---
        CHECK_CUDA(cudaMalloc(&mDeviceInputVdw, mBatchDataSize * sizeof(float)));

        // Temporary host buffer for staging VdW data before GPU copy
        mHostBatchVdw.resize(mBatchDataSize);

        mLogger.log(nvinfer1::ILogger::Severity::kINFO, ("Calibrator initialized. Found " + std::to_string(mNumFiles)
            + " files. Volume dimensions: " + std::to_string(mVolDimX) + "x" + std::to_string(mVolDimY) + "x" + std::to_string(mVolDimZ)
            + ". Using " + std::to_string(mNumCalibrationBatches) + " batches of size " + std::to_string(mBatchSize) + ".").c_str());
    }

    virtual ~HDF5Calibrator() {
        // Free GPU memory
        if (mDeviceInputVdw) {
            CHECK_CUDA(cudaFree(mDeviceInputVdw));
            mDeviceInputVdw = nullptr;
        }
        // No SES memory to free
    }

    // === IInt8Calibrator mandatory overrides ===

    int getBatchSize() const noexcept override {
        return mBatchSize;
    }

    // This function is called by TensorRT to get calibration batches.
    // It needs to copy data to the GPU pointers provided in 'bindings'.
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
        if (mCurrentBatch >= mNumCalibrationBatches) {
            mLogger.log(nvinfer1::ILogger::Severity::kINFO, "Finished calibration batches.");
            return false; // No more batches
        }

        try {
            // 1. Generate random indices for this batch
            std::uniform_int_distribution<int> fileDist(0, mNumFiles - 1);
            // Ensure patches fit entirely within the volume
            std::uniform_int_distribution<int> patchDistX(0, mVolDimX - mPatchSize);
            std::uniform_int_distribution<int> patchDistY(0, mVolDimY - mPatchSize);
            std::uniform_int_distribution<int> patchDistZ(0, mVolDimZ - mPatchSize);

            // 2. Extract patches from pre-loaded host data and fill VdW host buffer
            for (int i = 0; i < mBatchSize; ++i) {
                int fileIdx = fileDist(mRandEngine);
                int pz = patchDistZ(mRandEngine);
                int py = patchDistY(mRandEngine);
                int px = patchDistX(mRandEngine);

                // Copy VdW patch
                copyPatch(mVdwDataHost[fileIdx], mHostBatchVdw.data() + i * mPatchVolume, pz, py, px);
                // No SES patch needed
            }

            // 3. Copy VdW batch data from host buffer to pre-allocated GPU buffer
            CHECK_CUDA(cudaMemcpy(mDeviceInputVdw, mHostBatchVdw.data(), mBatchDataSize * sizeof(float), cudaMemcpyHostToDevice));
            // No SES copy needed

            // 4. Find the binding index for the VdW input tensor
            //    and assign our GPU buffer to the TensorRT binding.
            bool foundVdw = false;
            for (int i = 0; i < nbBindings; ++i) {
                if (std::string(names[i]) == mInputBlobNameVdw) {
                    bindings[i] = mDeviceInputVdw; // Point TRT to our device buffer
                    foundVdw = true;
                    break; // Found the only input we care about
                }
            }

            if (!foundVdw) {
                    mLogger.log(nvinfer1::ILogger::Severity::kERROR, ("Input tensor '" + mInputBlobNameVdw + "' provided to calibrator not found in network bindings!").c_str());
                    return false; // Indicate failure
            }

            mLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, ("Providing calibration batch " + std::to_string(mCurrentBatch + 1)).c_str());
            mCurrentBatch++;
            return true; // Batch provided successfully

        } catch (const std::exception& e) {
            mLogger.log(nvinfer1::ILogger::Severity::kERROR, ("Error in getBatch: " + std::string(e.what())).c_str());
            return false; // Indicate failure
        } catch (...) {
                mLogger.log(nvinfer1::ILogger::Severity::kERROR, "Unknown error in getBatch.");
            return false; // Indicate failure
        }
    }

    // Called by TensorRT before calibration to read a previously generated cache file.
    const void* readCalibrationCache(size_t& length) noexcept override {
        mCalibrationCache.clear();
        std::ifstream cacheFile(mCacheFileName, std::ios::binary);
        if (cacheFile.good()) {
            cacheFile.seekg(0, std::ios::end);
            length = cacheFile.tellg();
            cacheFile.seekg(0, std::ios::beg);
            mCalibrationCache.resize(length);
            cacheFile.read(mCalibrationCache.data(), length);
            cacheFile.close();
            mLogger.log(nvinfer1::ILogger::Severity::kINFO, ("Read calibration cache: " + mCacheFileName + ", size: " + std::to_string(length) + " bytes").c_str());
            return mCalibrationCache.data();
        } else {
            mLogger.log(nvinfer1::ILogger::Severity::kWARNING, ("Calibration cache file not found or could not be opened: " + mCacheFileName).c_str());
            length = 0;
            return nullptr;
        }
    }

    // Called by TensorRT after calibration to write the generated cache file.
    void writeCalibrationCache(const void* cache, size_t length) noexcept override {
        std::ofstream cacheFile(mCacheFileName, std::ios::binary);
        if (cacheFile.good()) {
            cacheFile.write(static_cast<const char*>(cache), length);
            cacheFile.close();
            mLogger.log(nvinfer1::ILogger::Severity::kINFO, ("Written calibration cache: " + mCacheFileName + ", size: " + std::to_string(length) + " bytes").c_str());
        } else {
            mLogger.log(nvinfer1::ILogger::Severity::kWARNING, ("Could not write calibration cache file: " + mCacheFileName).c_str());
        }
    }

private:
    int mBatchSize;
    int mPatchSize;
    size_t mPatchVolume;     // patchSize^3
    size_t mBatchDataSize;   // batchSize * patchVolume (in elements)
    std::string mPathVdw; // Only VdW path
    std::vector<std::string> mFileNames;
    int mNumCalibrationBatches;
    std::string mInputBlobNameVdw; // Only VdW input name
    std::string mCacheFileName;
    nvinfer1::ILogger& mLogger;

    int mNumFiles;
    int mVolDimX, mVolDimY, mVolDimZ; // Dimensions of the full volumes
    size_t mVolumeSize;              // Total elements in one full volume

    // --- Host Data Storage ---
    // WARNING: This stores ALL VdW data in CPU RAM. Can be very large!
    std::vector<std::vector<float>> mVdwDataHost; // Only VdW data
    // No SES data storage needed

    // --- GPU Data Storage ---
    void* mDeviceInputVdw = nullptr; // Only VdW input buffer
    // No SES buffer needed

    // --- Host buffer for staging batch data ---
    std::vector<float> mHostBatchVdw; // Only VdW host buffer
    // No SES host buffer needed

    // --- Calibration State ---
    int mCurrentBatch;
    std::vector<char> mCalibrationCache; // For reading cache

    // --- Random Number Generation ---
    std::mt19937 mRandEngine;


    // --- Helper Methods ---

    void loadAndPrepareData() {
        mNumFiles = mFileNames.size();
        mVdwDataHost.resize(mNumFiles); // Resize only for VdW
        // No resize for SES needed

        mLogger.log(nvinfer1::ILogger::Severity::kINFO, "Loading VdW HDF5 data into host memory...");

        for (int i = 0; i < mNumFiles; ++i) {
            // Load VdW
            std::string vdwFilePath = mPathVdw + "/" + mFileNames[i];
            loadHDF5File(vdwFilePath, "texture", mVdwDataHost[i], i == 0); // Get dims from first file

            // No SES file loading needed

            if (mVdwDataHost[i].size() != mVolumeSize) { // Check only VdW size
                    throw std::runtime_error("Inconsistent volume sizes detected in VdW HDF5 files.");
            }
                mLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, ("Loaded file " + std::to_string(i+1) + "/" + std::to_string(mNumFiles)).c_str());
        }
        mLogger.log(nvinfer1::ILogger::Severity::kINFO, "Finished loading HDF5 data.");
    }

    // Loads a single HDF5 file's dataset into a float vector (Unchanged)
    void loadHDF5File(const std::string& filename, const std::string& datasetName, std::vector<float>& data, bool getDims) {
        hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0) throw std::runtime_error("Could not open HDF5 file: " + filename);

        hid_t dataset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
        if (dataset_id < 0) { H5Fclose(file_id); throw std::runtime_error("Could not open dataset: " + datasetName); }

        hid_t filespace_id = H5Dget_space(dataset_id);
        if (filespace_id < 0) { H5Dclose(dataset_id); H5Fclose(file_id); throw std::runtime_error("Could not get dataspace"); }

        int ndims = H5Sget_simple_extent_ndims(filespace_id);
        if (ndims != 3) { // Expecting 3D volumes
                H5Sclose(filespace_id); H5Dclose(dataset_id); H5Fclose(file_id);
                throw std::runtime_error("Expected 3D data in HDF5, found " + std::to_string(ndims) + " dimensions.");
        }

        hsize_t dims[3];
        CHECK_HDF5( H5Sget_simple_extent_dims(filespace_id, dims, NULL) );

        if (getDims) {
            mVolDimZ = dims[0];
            mVolDimY = dims[1];
            mVolDimX = dims[2];
            mVolumeSize = static_cast<size_t>(mVolDimZ) * mVolDimY * mVolDimX;

            if (mPatchSize > mVolDimX || mPatchSize > mVolDimY || mPatchSize > mVolDimZ) {
                    H5Sclose(filespace_id); H5Dclose(dataset_id); H5Fclose(file_id);
                    throw std::invalid_argument("Patch size is larger than volume dimensions.");
            }
        } else {
                if (dims[0] != mVolDimZ || dims[1] != mVolDimY || dims[2] != mVolDimX) {
                H5Sclose(filespace_id); H5Dclose(dataset_id); H5Fclose(file_id);
                throw std::runtime_error("Inconsistent dimensions found across HDF5 files.");
                }
        }

        data.resize(mVolumeSize);
        CHECK_HDF5( H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) );

        CHECK_HDF5( H5Sclose(filespace_id) );
        CHECK_HDF5( H5Dclose(dataset_id) );
        CHECK_HDF5( H5Fclose(file_id) );
    }

    // Copies a 3D patch from the flattened full volume source to the destination buffer (Unchanged)
    void copyPatch(const std::vector<float>& sourceVolume, float* destPatch, int startZ, int startY, int startX) {
        size_t srcSliceStride = static_cast<size_t>(mVolDimX) * mVolDimY;
        size_t srcRowStride = mVolDimX;
        size_t patchIdx = 0;

        for (int z = 0; z < mPatchSize; ++z) {
            for (int y = 0; y < mPatchSize; ++y) {
                const float* srcRowStart = sourceVolume.data()
                                            + static_cast<size_t>(startZ + z) * srcSliceStride
                                            + static_cast<size_t>(startY + y) * srcRowStride
                                            + startX;
                memcpy(destPatch + patchIdx, srcRowStart, mPatchSize * sizeof(float));
                patchIdx += mPatchSize;
            }
        }
            if (patchIdx != mPatchVolume) {
                throw std::logic_error("Internal error: Patch copy size mismatch.");
            }
    }

}; */ // End class HDF5Calibrator

int main()
{   
    Logger logger;
    // NOTE this can be used for int8 quantization
    // --- Configuration ---
    // const int PATCH_SIZE = 64;
    // const std::string VDW_PATH = "data/vdw";
    // // Get your list of filenames for calibration (e.g., from a directory listing or a file)
    // std::vector<std::string> calibration_files;
    // try {
    //     // Get all files ending with .h5 from VDW_PATH using POSIX function
    //     calibration_files = listFilesWithExtensionPosix(VDW_PATH, ".h5"); // Or listFilesWithExtension if using C++17
    
    //     if (calibration_files.empty()) {
    //         // Print warning to standard error
    //         std::cerr << "Warning: No .h5 files found for calibration in: " << VDW_PATH << std::endl;
    //         // Original comment's suggestion: Decide how to proceed.
    //         // You might want to throw an error or return here if calibration files are mandatory.
    //         // For example:
    //         // throw std::runtime_error("Calibration failed: No suitable files found.");
    //         // or if in a function:
    //         // return false; // Indicate failure
    //     } else {
    //         // Print info message to standard output
    //         std::cout << "Info: Found " << calibration_files.size() << " calibration files in " << VDW_PATH << std::endl;
    //     }
    
    // } catch (const std::exception& e) {
    //     // Print error message to standard error
    //     std::cerr << "Error: Failed to list calibration files: " << e.what() << std::endl;
    //     // It's usually critical if you can't list files, so you should probably exit or return an error.
    //     // For example, if in a function:
    //     // return false; // Indicate failure
    //     // Or rethrow or exit:
    //     // exit(EXIT_FAILURE);
    // }
    
    // const int NUM_CALIBRATION_BATCHES = 100; // Adjust as needed (more batches -> better calibration, slower build)
    // const std::string INPUT_NAME_VDW = "input"; // *** MUST MATCH your model's input tensor name ***
    // const std::string CACHE_FILENAME = "calibration.cache";

    // Define file paths
    const char* modelFile = "run/data/models/unet_8_ch_1-2-4-4_mults_02_06_25.onnx";  // Path to your ONNX model file
    
    const char* engineFile = "run/data/engines/unet_8_ch_1-2-4-4_mults_02_06_25_FP16.trt";  // Path where the engine will be saved

    // set optimal/max batch size
    const int BATCH_SIZE = 64;

    // Create builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    if (!builder)
    {
        std::cerr << "Failed to create TensorRT builder." << std::endl;
        return -1;
    }

    // Use explicit batch flag for ONNX models
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    if (!network)
    {
        std::cerr << "Failed to create network." << std::endl;
        delete builder;
        return -1;
    }

    // // set Batch size in builder when using implicit batches
    // int BATCH_SIZE = 64;
    // builder->setMaxBatchSize(BATCH_SIZE); 

    // Create ONNX parser
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    if (!parser)
    {
        std::cerr << "Failed to create ONNX parser." << std::endl;
        delete network;
        delete builder;
        return -1;
    }

    // Parse ONNX model from file.
    if (!parser->parseFromFile(modelFile, static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        std::cerr << "Failed to parse ONNX file: " << modelFile << std::endl;
        // Print any parser errors.
        for (int32_t i = 0; i < parser->getNbErrors(); ++i)
        {
            std::cout << parser->getError(i)->desc() << std::endl;
        }
        delete parser;
        delete network;
        delete builder;
        return -1;
    }

    // Create builder config and set workspace size
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    // config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 8ULL * 1024ULL * 1024ULL * 1024ULL);// Example: 8GB workspace (adjust as needed)

    // config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED); //NOTE detailed output of profiling

    config->setFlag(nvinfer1::BuilderFlag::kFP16); //NOTE: this enables mixed precision

    // // --- Enable INT8 Mode ---
    // config->setFlag(nvinfer1::BuilderFlag::kINT8);

    // // --- Create Calibrator ---
    // // Use std::unique_ptr for automatic memory management
    // std::unique_ptr<HDF5Calibrator> calibrator = std::make_unique<HDF5Calibrator>(
    //     BATCH_SIZE, // Use the same batch size for calibration as potentially for inference, or a reasonable size
    //     PATCH_SIZE,
    //     VDW_PATH,
    //     calibration_files, // Pass the list of files to use
    //     NUM_CALIBRATION_BATCHES,
    //     INPUT_NAME_VDW,
    //     CACHE_FILENAME,
    //     logger
    // );

    // // Set the calibrator in the config
    // config->setInt8Calibrator(calibrator.get());

    // Create an optimization profile to fix the input shape.
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    if (!profile) {
        std::cerr << "Error: Failed to create optimization profile." << std::endl;
        delete config;
        delete parser;
        delete network;
        delete builder;
        return -1; // Or handle error appropriately
    }
    

    // Get the network's input tensor.
    nvinfer1::ITensor* inputTensor = network->getInput(0); // adjust index if needed
    if (!inputTensor) {
        std::cerr << "Error: Could not get input tensor at index 0." << std::endl;
        delete config;
        delete parser;
        delete network;
        delete builder;
        return -1; // Or handle error appropriately
    }
    const char* inputName = inputTensor->getName();

    // Get the current dimensions of the input (should be dynamic in the first dimension).
    nvinfer1::Dims dims = inputTensor->getDimensions();
    // For example, dims might be {?, 1, 64, 64, 64}. We might want to fix the batch to 64.
    // dims.d[0] = BATCH_SIZE;  // fixed batch size

    if (dims.d[0] != -1) {
        std::cerr << "Error: Expected input tensor '" << inputName
                  << "' to have a dynamic batch dimension (d[0] == -1) after parsing ONNX,"
                  << " but got " << dims.d[0] << ". Did you use the correct ONNX file?" << std::endl;
        // You might want to clean up the profile pointer before returning
        delete config;
        delete parser;
        delete network;
        delete builder;
        return -1;
    }

    // Define the desired range and an optimal value for the batch dimension
    const int minBatch = 1;
    const int optBatch = BATCH_SIZE; // Choose a typical or common batch size for optimization
    const int maxBatch = BATCH_SIZE;

    // Set the shapes for the dynamic input tensor
    nvinfer1::Dims minDims = dims; // Copy static dimensions
    minDims.d[0] = minBatch;       // Set min value for dynamic batch dim

    nvinfer1::Dims optDims = dims; // Copy static dimensions
    optDims.d[0] = optBatch;       // Set opt value for dynamic batch dim

    nvinfer1::Dims maxDims = dims; // Copy static dimensions
    maxDims.d[0] = maxBatch;       // Set max value for dynamic batch dim

    // Set the same shape for the minimum, optimum, and maximum in the optimization profile.
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, minDims);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, optDims);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, maxDims);

    // Add the profile to the configuration.
    int profileIndex = config->addOptimizationProfile(profile);
    if (profileIndex < 0) {
        std::cerr << "Error: Failed to add optimization profile to builder config." << std::endl;
        // You might want to clean up the profile pointer before returning
        delete config;
        delete parser;
        delete network;
        delete builder;
        return -1; // Or handle error appropriately
    }

    // Build serialized engine
    std::cout << "building engine ..." << std::endl;
    nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
    if (!serializedModel)
    {
        std::cerr << "Failed to build serialized network." << std::endl;
        delete config;
        delete parser;
        delete network;
        delete builder;
        return -1;
    }

    // Save the serialized engine to disk
    std::ofstream engineOut(engineFile, std::ios::binary);
    if (!engineOut)
    {
        std::cerr << "Could not open output file: " << engineFile << std::endl;
        delete serializedModel;
        delete config;
        delete parser;
        delete network;
        delete builder;
        return -1;
    }
    engineOut.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    engineOut.close();
    std::cout << "Engine successfully saved to: " << engineFile << std::endl;

    // Cleanup: destroy TensorRT objects
    delete serializedModel;
    delete config;
    delete parser;
    delete network;
    delete builder;

    return 0;
}