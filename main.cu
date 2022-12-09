#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <mma.h>
#include <random>

#define HOST_VERIFY

// Each CTA has 8 warps
#define WARPS_PER_CTA 8

// Each warp has 32 threads
#define THREADS_PER_WARP 32
#define HALF_THREADS_PER_WARP (THREADS_PER_WARP / 2)

// Each block is divided into a 4x2 warp array
#define BLOCK_WARP_ROWS 4
#define BLOCK_WARP_COLS 2

// Each warp handles a 2x4 16x16 matrices
#define WARP_TILE_ROWS 2
#define WARP_TILE_COLS 4

// Each block has 8x8 16x16 matrices
#define BLOCK_TILE_ROWS (BLOCK_WARP_ROWS * WARP_TILE_ROWS)
#define BLOCK_TILE_COLS (BLOCK_WARP_COLS * WARP_TILE_COLS)

// The tile size (due to WMMA interface)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Matrix size
#define M_GLOBAL 4096
#define N_GLOBAL 4096
#define K_GLOBAL 4096

// The whole computation is divided into tiles
#define M_TILES (M_GLOBAL / WMMA_M)
#define N_TILES (N_GLOBAL / WMMA_N)
#define K_TILES (K_GLOBAL / WMMA_K)

// The whole computation is divided into blocks
#define M_BLOCKS (M_TILES / BLOCK_TILE_ROWS)
#define N_BLOCKS (N_TILES / BLOCK_TILE_COLS)
#define NUM_BLOCKS (M_BLOCKS * N_BLOCKS)

// The stride while iterating over K dimension
#define K_CHUNK_TILES 8

// Padding to avoid bank conflicts
#define SKEW_HALF 16

// Strides
#define A_GLOBAL_STRIDE K_GLOBAL
#define B_GLOBAL_STRIDE K_GLOBAL
#define A_SHARED_STRIDE (K_CHUNK_TILES * WMMA_K + SKEW_HALF)
#define B_SHARED_STRIDE (K_CHUNK_TILES * WMMA_K + SKEW_HALF)

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)


void initHostMatrices(half* hostA, half *hostB, float *hostC) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1, 1);
    for (int i = 0; i < M_GLOBAL * K_GLOBAL; ++ i)
        hostA[i] = static_cast<half>(dist(gen));
    for (int i = 0; i < K_GLOBAL * N_GLOBAL; ++ i)
        hostB[i] = static_cast<half>(dist(gen));
    for (int i = 0; i < M_GLOBAL * N_GLOBAL; ++ i)
        hostC[i] = dist(gen);
}

#ifdef HOST_VERIFY

void verify(const half *hostA, const half *hostB, const float *hostC,
            const float *hostD, const float alpha, const float beta) {
    for (int i = 0; i < M_GLOBAL; ++ i) {
        for (int j = 0; j < N_GLOBAL; ++ j) {
            float d = 0;
            for (int k = 0; k < K_GLOBAL; ++ k)
                d += static_cast<float>(hostA[i * K_GLOBAL + k]) * static_cast<float>(hostB[j * K_GLOBAL + k]);
            d = d * alpha + hostC[i * N_GLOBAL + j] * beta;
            if (fabs(d - hostD[i * N_GLOBAL + j]) > 1e-1) {
                printf("Verification failed, at (%d, %d): %f, %f\n", i, j, d, hostD[i * N_GLOBAL + j]);
                std::exit(EXIT_FAILURE);
            }
        }
    }
}

#endif

__global__ void gemm(half *A, half *B, float *C, float *D, float alpha, float beta) {
    // Get the shared memory for saving A and B
    extern __shared__ half shmem[];
    half *sharedA = shmem, *sharedB = &shmem[BLOCK_TILE_ROWS * WMMA_M * (K_CHUNK_TILES * WMMA_K + SKEW_HALF)];

    // Warp ID and lane ID
    unsigned int warpId = threadIdx.x / THREADS_PER_WARP;
    unsigned int laneId = threadIdx.x % THREADS_PER_WARP;
    unsigned int workerGroupId = laneId < HALF_THREADS_PER_WARP, workerId = laneId % HALF_THREADS_PER_WARP;

    // Warp indices inside the block
    // (0, 0), (0, 1)
    // (1, 0), (1, 1) ...
    unsigned int wi = warpId / BLOCK_WARP_COLS, wj = warpId % BLOCK_WARP_COLS;

    // Iterate over all 128x128 blocks (1024 blocks in total) with 82 SMs
    // SM1: 0, 82, 164, ...
    // SM2: 1, 83, 165, ...
    using namespace nvcuda;
    for (unsigned int b = blockIdx.x; b < NUM_BLOCKS; b += gridDim.x) {
        // Global block indices
        const unsigned int bi = b / N_BLOCKS, bj = b % N_BLOCKS;

        // Global tile (the first one) indices
        const unsigned int ti = bi * BLOCK_TILE_ROWS, tj = bj * BLOCK_TILE_COLS;

        // Global C's tile (the first one) indices
        const unsigned int cti = bi * BLOCK_TILE_ROWS + wi * WARP_TILE_ROWS, ctj = bj * BLOCK_TILE_COLS + wj * WARP_TILE_COLS;

        // Scale C matrix, D = alpha * (dot(A, B) + beta / alpha * C)
        float scale_factor = beta / alpha;

        // A warp has 2x4 16x16 matrices, load C from global memory into fragments and scale
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC[WARP_TILE_ROWS][WARP_TILE_COLS];
        #pragma unroll
        for (unsigned int i = 0; i < WARP_TILE_ROWS; ++ i) {
            #pragma unroll
            for (unsigned int j = 0; j < WARP_TILE_COLS; ++ j) {
                wmma::load_matrix_sync(fragC[i][j], C + (cti + i) * WMMA_M * N_GLOBAL + (ctj + j) * WMMA_N, N_GLOBAL, wmma::mem_row_major);
                #pragma unroll
                for (int k = 0; k < fragC[i][j].num_elements; ++ k)
                    fragC[i][j].x[k] *= scale_factor;
            }
        }
        __syncthreads();

        // Iterate over chunks
        #pragma unroll
        for (unsigned int tk = 0; tk < K_TILES; tk += K_CHUNK_TILES) {
            // Load A and B into shared memory
            #pragma unroll
            for (unsigned int chunk = 0; chunk < K_CHUNK_TILES; ++ chunk) {
                // Load A and B into shared memory using 8 warps
                // Each warp is responsible for a warp row
                assert(BLOCK_TILE_ROWS == WARPS_PER_CTA);
                int4 *int4Src, *int4Dst;
                if (laneId < HALF_THREADS_PER_WARP) {
                    // Copy the 16x16 matrix with 16 threads for A
                    assert(HALF_THREADS_PER_WARP == WMMA_M);
                    int4Src = reinterpret_cast<int4*>(A + ((ti + warpId) * WMMA_M + workerId) * A_GLOBAL_STRIDE + (tk + chunk) * WMMA_K);
                    int4Dst = reinterpret_cast<int4*>(sharedA + (warpId * WMMA_M + workerId) * A_SHARED_STRIDE + chunk * WMMA_K);
                } else {
                    // Copy the 16x16 matrix with 16 threads for B
                    assert(HALF_THREADS_PER_WARP == WMMA_N);
                    int4Src = reinterpret_cast<int4*>(B + ((tj + warpId) * WMMA_N + workerId) * B_GLOBAL_STRIDE + (tk + chunk) * WMMA_K);
                    int4Dst = reinterpret_cast<int4*>(sharedB + (warpId * WMMA_N + workerId) * B_SHARED_STRIDE + chunk * WMMA_K);
                }
                int4Dst[0] = int4Src[0], int4Dst[1] = int4Src[1];
            }
            __syncthreads();

            // Compute and store into C fragments
            #pragma unroll
            for (unsigned int chunk = 0; chunk < K_CHUNK_TILES; ++ chunk) {
                // Load A from shared memory to fragments
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragA[WARP_TILE_ROWS];
                #pragma unroll
                for (int i = 0; i < WARP_TILE_ROWS; ++ i)
                    wmma::load_matrix_sync(fragA[i], sharedA + (wi * WARP_TILE_ROWS + i) * WMMA_M * A_SHARED_STRIDE + chunk * WMMA_K, A_SHARED_STRIDE);

                // Load B from shared memory to fragments
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> fragB[WARP_TILE_COLS];
                #pragma unroll
                for (int j = 0; j < WARP_TILE_COLS; ++ j)
                    wmma::load_matrix_sync(fragB[j], sharedB + (wj * WARP_TILE_COLS + j) * WMMA_N * B_SHARED_STRIDE + chunk * WMMA_K, B_SHARED_STRIDE);

                // Multiply and accumulate into C fragments
                #pragma unroll
                for (int i = 0; i < WARP_TILE_ROWS; ++ i) {
                    #pragma unroll
                    for (int j = 0; j < WARP_TILE_COLS; ++ j)
                        wmma::mma_sync(fragC[i][j], fragA[i], fragB[j], fragC[i][j]);
                }
            }
        }

        // Scale and store into D
        #pragma unroll
        for (unsigned int i = 0; i < WARP_TILE_ROWS; ++ i) {
            #pragma unroll
            for (unsigned int j = 0; j < WARP_TILE_COLS; ++ j) {
                #pragma unroll
                for (int k = 0; k < fragC[i][j].num_elements; ++ k)
                    fragC[i][j].x[k] *= alpha;
                wmma::store_matrix_sync(D + (cti + i) * WMMA_M * N_GLOBAL + (ctj + j) * WMMA_N, fragC[i][j], N_GLOBAL, wmma::mem_row_major);
            }
        }
    }
}

int main(int argc, const char **argv) {
    // Find device
    printf("Initializing ...\n");
    int dev = findCudaDevice(argc, argv);
    cudaDeviceProp deviceProp = {};
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    if (deviceProp.major < 7) {
        printf("GEMM requires SM 7.0 or higher\n");
        std::exit(EXIT_WAIVED);
    }

    // Problem specs
    float alpha = 1.1, beta = 1.2;
    printf("Problem specs:\n");
    printf("M: %d, N: %d, K: %d\n\n", M_GLOBAL, N_GLOBAL, K_GLOBAL);

    // Host memory
    auto *hostA = reinterpret_cast<half*>(std::malloc(sizeof(half) * M_GLOBAL * K_GLOBAL));
    auto *hostB = reinterpret_cast<half*>(std::malloc(sizeof(half) * K_GLOBAL * N_GLOBAL));
    auto *hostC = reinterpret_cast<float*>(std::malloc(sizeof(float) * M_GLOBAL * N_GLOBAL));

    // Device memory
    half *A, *B;
    float *C, *D;
    checkCudaErrors(cudaMalloc(&A, sizeof(half) * M_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc(&B, sizeof(half) * K_GLOBAL * N_GLOBAL));
    checkCudaErrors(cudaMalloc(&C, sizeof(float) * M_GLOBAL * N_GLOBAL));
    checkCudaErrors(cudaMalloc(&D, sizeof(float) * M_GLOBAL * N_GLOBAL));
    assert(reinterpret_cast<size_t>(A) % 128 == 0);
    assert(reinterpret_cast<size_t>(B) % 128 == 0);
    assert(reinterpret_cast<size_t>(C) % 128 == 0);
    assert(reinterpret_cast<size_t>(D) % 128 == 0);

    // Initialize host matrices and copy
    initHostMatrices(hostA, hostB, hostC);
    checkCudaErrors(cudaMemcpy(A, hostA, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B, hostB, sizeof(half) * K_GLOBAL * N_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(C, hostC, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));

    // Each CTA computes one 128x128 tile of the resulting matrix, using 8 warps
    // Each warp computes eight 16x16 sub-tiles, organized in a 2x4 2D array.
    // Optimizations:
    // - The CTA copies the 128x128 tile of C from global to shared memory, so that each warp could load there
    // - The CTA copies a portion of A and B from global to shared to re-use for different warps
    // - An additional padding to reduce bank conflicts
    // - The CTA stores into shared memory
    // Size of A and B, which are to be stored in shared memory
    size_t sharedMemSize = 0;
    sharedMemSize += BLOCK_TILE_ROWS * WMMA_M * (K_CHUNK_TILES * WMMA_K + SKEW_HALF) * sizeof(half);
    sharedMemSize += BLOCK_TILE_ROWS * WMMA_N * (K_CHUNK_TILES * WMMA_K + SKEW_HALF) * sizeof(half);
    printf("Hardware specs:\n");
    printf("Device shared memory per SM: %zu KiB\n", deviceProp.sharedMemPerMultiprocessor / 1024);
    printf("Required shared memory size: %zu KiB\n", sharedMemSize / 1024);
    printf("SM count: %d\n", deviceProp.multiProcessorCount);
    printf("Threads per CTA: %d\n\n", WARPS_PER_CTA * THREADS_PER_WARP);

    // Timer starts
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    // Computation
    printf("Computing ...\n");
    // <<<GridDim, BlockDim, SharedMemSize>>>
    checkCudaErrors(cudaFuncSetAttribute(gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize));
    checkKernelErrors((
        gemm<<<deviceProp.multiProcessorCount, WARPS_PER_CTA * THREADS_PER_WARP, sharedMemSize>>>(A, B, C, D, alpha, beta)
    ));

    // Timer stops
    float milliseconds = 0;
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time: %f ms\n", milliseconds);
    printf("TFLOPS: %.2f\n\n", 2.0f * static_cast<double>(M_GLOBAL) * static_cast<double>(N_GLOBAL) * static_cast<double>(K_GLOBAL) / (milliseconds / 1000.) / 1e12);

#ifdef HOST_VERIFY
    // Verify
    auto *hostD = reinterpret_cast<float*>(std::malloc(sizeof(float) * M_GLOBAL * N_GLOBAL));
    checkCudaErrors(cudaMemcpy(hostD, D, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));
    verify(hostA, hostB, hostC, hostD, alpha, beta);
    printf("Verification succeed, exit\n");
    std::free(hostD);
#endif

    // Free resources
    std::free(hostA);
    std::free(hostB);
    std::free(hostC);
    checkCudaErrors(cudaFree(A));
    checkCudaErrors(cudaFree(B));
    checkCudaErrors(cudaFree(C));
    checkCudaErrors(cudaFree(D));

    return EXIT_SUCCESS;
}