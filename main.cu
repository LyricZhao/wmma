#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <mma.h>
#include <random>

// #define HOST_VERIFY

// Each CTA has 8 warps
#define WARPS_PER_CTA 8
#define HALF_WARPS_PER_CTA (WARPS_PER_CTA / 2)
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

// Strides
#define A_GLOBAL_STRIDE K_GLOBAL
#define B_GLOBAL_STRIDE K_GLOBAL
#define A_SHARED_STRIDE (K_CHUNK_TILES * WMMA_K)
#define B_SHARED_STRIDE (K_CHUNK_TILES * WMMA_K)


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
                printf("Verification failed, at (%d, %d)\n", i, j);
                std::exit(EXIT_FAILURE);
            }
        }
    }
}

#endif

__global__ void gemm(half *A, half *B, float *C, float *D,
                     const float &alpha, const float& beta) {
    // Get the shared memory for saving A and B
    extern __shared__ half shmem[];
    half *sharedA = shmem, *sharedB = shmem + K_CHUNK_TILES * WARP_TILE_ROWS * WMMA_M * WMMA_K;

    // Warp ID and lane ID
    const unsigned int warpId = threadIdx.x / THREADS_PER_WARP;
    const unsigned int laneId = threadIdx.x % THREADS_PER_WARP;

    // Warp indices inside the block
    // (0, 0), (0, 1)
    // (1, 0), (1, 1) ...
    const unsigned int wi = warpId / BLOCK_WARP_COLS, wj = warpId % BLOCK_WARP_COLS;

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

        // Global element (the first one) indices
        const unsigned int cei = cti * WMMA_M, cej = ctj * WMMA_N;

        // Scale C matrix, D = alpha * (dot(A, B) + beta / alpha * C)
        float scale_factor = beta / alpha;

        // A warp has 2x4 16x16 matrices, load C from global memory into fragments and scale
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC[WARP_TILE_ROWS][WARP_TILE_COLS];
        #pragma unroll
        for (unsigned int i = 0; i < WARP_TILE_ROWS; ++ i) {
            #pragma unroll
            for (unsigned int j = 0; j < WARP_TILE_COLS; ++ j) {
                wmma::load_matrix_sync(fragC[i][j], C + cei * N_GLOBAL + cej, N_GLOBAL, wmma::mem_row_major);
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
                // Load A (BLOCK_TILE_ROWS x 1) into shared memory using 8 warps
                // Each warp is responsible for a warp row
                assert(BLOCK_TILE_ROWS == WARPS_PER_CTA);
                unsigned int ei = 0, ej = 0, eiShared = 0, ejShared = 0;
                unsigned int stride = 0, strideShared = 0;
                half *dst = nullptr, *src = nullptr;
                if (laneId < HALF_THREADS_PER_WARP) {
                    // Copy the 16x16 matrix with 16 threads for A
                    assert(HALF_THREADS_PER_WARP == WMMA_M);
                    ei = (ti + warpId) * WMMA_M, ej = (tk + chunk) * WMMA_K;
                    eiShared = warpId * WMMA_M, ejShared = chunk * WMMA_K;
                    stride = A_GLOBAL_STRIDE, strideShared = A_SHARED_STRIDE;
                    dst = sharedA, src = A;
                } else {
                    // Copy the 16x16 matrix with 16 threads for B
                    assert(HALF_THREADS_PER_WARP == WMMA_N);
                    ei = (tj + warpId) * WMMA_N, ej = (tk + chunk) * WMMA_K;
                    eiShared = warpId * WMMA_N, ejShared = chunk * WMMA_K;
                    stride = B_GLOBAL_STRIDE, strideShared = B_SHARED_STRIDE;
                    dst = sharedB, src = B;
                }
                #pragma unroll
                for (unsigned int copy = 0; copy < WMMA_K; ++ copy)
                    dst[(eiShared + laneId) * strideShared + ejShared + copy] = src[(ei + laneId) * stride + ej + copy];
            }
            __syncthreads();

            // Compute and store into C fragments
            #pragma unroll
            for (unsigned int chunk = 0; chunk < K_CHUNK_TILES; ++ chunk) {
                // Load A from shared memory to fragments
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragA[WARP_TILE_ROWS];
                #pragma unroll
                for (int i = 0; i < WARP_TILE_ROWS; ++ i)
                    wmma::load_matrix_sync(fragA[i], sharedA + wi * WARP_TILE_ROWS * WMMA_M * A_SHARED_STRIDE + (tk + chunk) * WMMA_K, A_SHARED_STRIDE);

                // Load B from shared memory to fragments
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> fragB[WARP_TILE_COLS];
                #pragma unroll
                for (int j = 0; j < WARP_TILE_COLS; ++ j)
                    wmma::load_matrix_sync(fragB[j], sharedB + wj * WARP_TILE_COLS * WMMA_N * B_SHARED_STRIDE + (tk + chunk) * WMMA_K, B_SHARED_STRIDE);

                // Multiply and accumulate into C fragments
                #pragma unroll
                for (int i = 0; i < WARP_TILE_ROWS; ++ i) {
                    #pragma unroll
                    for (int j = 0; j < WARP_TILE_COLS; ++ j)
                        wmma::mma_sync(fragC[i][j], fragA[i], fragB[j], fragC[i][j]);
                }
            }
            __syncthreads();
        }

        // TODO: scale and store into D
        __syncthreads();
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

    // Timer starts
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    // Computation
    // Each CTA computes one 128x128 tile of the resulting matrix, using 8 warps
    // Each warp computes eight 16x16 sub-tiles, organized in a 2x4 2D array.
    // Optimizations:
    // - The CTA copies the 128x128 tile of C from global to shared memory, so that each warp could load there
    // - The CTA copies a portion of A and B from global to shared to re-use for different warps
    // - An additional padding to reduce bank conflicts
    // - The CTA stores into shared memory
    // Size of A and B, which are to be stored in shared memory
    size_t sharedMemSize = K_CHUNK_TILES * (BLOCK_TILE_ROWS * WMMA_M * WMMA_K + BLOCK_TILE_COLS * WMMA_K * WMMA_N) * sizeof(half);
    printf("Required shared memory size: %zu KiB\n", sharedMemSize / 1024);
    printf("Computing ...\n");
    // <<<GridDim, BlockDim, SharedMemSize>>>
    gemm<<<deviceProp.multiProcessorCount, WARPS_PER_CTA * THREADS_PER_WARP, sharedMemSize>>>(A, B, C, D, alpha, beta);

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