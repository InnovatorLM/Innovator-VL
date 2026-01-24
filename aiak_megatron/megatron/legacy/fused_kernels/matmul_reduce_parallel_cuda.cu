#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

#define DIVUP(x, y) (((x) + (y)-1) / (y))

#define CUBLASCHECK(cmd)                                 \
    do {                                                 \
        cublasStatus_t e = cmd;                          \
        if (e != CUBLAS_STATUS_SUCCESS) {                \
            printf("Failed: CUBLAS error %s: %d '%d'\n", \
                   __FILE__,                             \
                   __LINE__,                             \
                   cmd);                                 \
            assert(false);                               \
        }                                                \
    } while (0)

#define CUDACHECK(cmd)                                \
    do {                                              \
        cudaError_t e = cmd;                          \
        if (e != cudaSuccess) {                       \
            printf("Failed: Cuda error %s:%d '%s'\n", \
                   __FILE__,                          \
                   __LINE__,                          \
                   cudaGetErrorString(e));            \
            exit(EXIT_FAILURE);                       \
        }                                             \
    } while (0)

#define NCCLCHECK(cmd)                                \
    do {                                              \
        ncclResult_t r = cmd;                         \
        if (r != ncclSuccess) {                       \
            printf("Failed, NCCL error %s:%d '%s'\n", \
                   __FILE__,                          \
                   __LINE__,                          \
                   ncclGetErrorString(r));            \
            exit(EXIT_FAILURE);                       \
        }                                             \
    } while (0)

void cublasProcessMatMulChunk(cublasHandle_t handle,
                              cudaStream_t stream,
                              size_t chunkStart,
                              size_t chunkEnd,
                              const half* alpha,
                              const half* beta,
                              const at::Half* input,
                              const at::Half* weight,
                              at::Half* output,
                              int M,
                              int N,
                              int K,
                              bool column_parallel) {
    int rows = DIVUP(chunkEnd - chunkStart, N);
    int startRow = DIVUP(chunkStart, N);
    if (chunkStart == 0) {
        rows = min(rows, M);
    } else {
        if (M < startRow + rows) {
            rows = M - startRow;
        }
    }

    if (rows <= 0 || startRow >= M || rows > M || startRow + rows > M) {
        return;
    }

    if (column_parallel) {
        CUBLASCHECK(cublasGemmEx(handle,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 N,
                                 rows,
                                 K,
                                 alpha,
                                 weight,
                                 CUDA_R_16F,
                                 N,
                                 input + startRow * K,
                                 CUDA_R_16F,
                                 K,
                                 beta,
                                 output + startRow * N,
                                 CUDA_R_16F,
                                 N,
                                 CUDA_R_16F,
                                 CUBLAS_GEMM_DFALT_TENSOR_OP));
    } else {
        CUBLASCHECK(cublasGemmEx(handle,
                                 CUBLAS_OP_T,
                                 CUBLAS_OP_N,
                                 N,
                                 rows,
                                 K,
                                 alpha,
                                 weight,
                                 CUDA_R_16F,
                                 K,
                                 input + startRow * K,
                                 CUDA_R_16F,
                                 K,
                                 beta,
                                 output + startRow * N,
                                 CUDA_R_16F,
                                 N,
                                 CUDA_R_16F,
                                 CUBLAS_GEMM_DFALT_TENSOR_OP));
    }
}

void cublasProcessMatMulChunk(cublasHandle_t handle,
                                cudaStream_t stream,
                                size_t chunkStart,
                                size_t chunkEnd,
                                const float* alpha,
                                const float* beta,
                                const float* input,
                                const float* weight,
                                float* output,
                                int M,
                                int N,
                                int K,
                                bool column_parallel) {
    int rows = DIVUP(chunkEnd - chunkStart, N);
    int startRow = DIVUP(chunkStart, N);
    if (chunkStart == 0) {
        rows = min(rows, M);
    } else {
        if (M < startRow + rows) {
            rows = M - startRow;
        }
    }

    if (rows <= 0 || startRow >= M || rows > M || startRow + rows > M) {
        return;
    }

    if (column_parallel) {
        CUBLASCHECK(cublasGemmEx(handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        N,
                        rows,
                        K,
                        alpha,
                        weight,
                        CUDA_R_32F,
                        N,
                        input + startRow * K,
                        CUDA_R_32F,
                        K,
                        beta,
                        output + startRow * N,
                        CUDA_R_32F,
                        N,
                        CUDA_R_32F,
                        CUBLAS_GEMM_DFALT_TENSOR_OP));
    } else {
        CUBLASCHECK(cublasGemmEx(handle,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        N,
                        rows,
                        K,
                        alpha,
                        weight,
                        CUDA_R_32F,
                        K,
                        input + startRow * K,
                        CUDA_R_32F,
                        K,
                        beta,
                        output + startRow * N,
                        CUDA_R_32F,
                        N,
                        CUDA_R_32F,
                        CUBLAS_GEMM_DFALT_TENSOR_OP));
    }
}


int pipe_overlapped_with_split_kernel(cublasHandle_t handle,
                                      const at::Half* input,
                                      const at::Half* weight,
                                      at::Half* output,
                                      int slit_n,
                                      int M,
                                      int N,
                                      int K,
                                      cudaStream_t cublasStream,
                                      ncclComm_t comm,
                                      cudaStream_t ncclStream,
                                      float alpha,
                                      float beta,
                                      bool column_parallel) {
    const half hAlpha = __float2half_rn(alpha);
    const half hBeta = __float2half_rn(beta);

    size_t M_N = static_cast<size_t>(M) * static_cast<size_t>(N);

    if (slit_n >= 0 && M_N % slit_n == 0) {
        
        size_t chunkSize = M_N / slit_n;
        cudaEvent_t gemm_event;
        cudaEventCreate(&gemm_event);
        cudaEvent_t nccl_event;
        cudaEventCreate(&nccl_event);
        cudaEventCreateWithFlags(&gemm_event, cudaEventBlockingSync);
        cudaEventCreateWithFlags(&nccl_event, cudaEventBlockingSync);

        for (size_t chunkStart = 0; chunkStart < M_N; chunkStart += chunkSize) {
            cublasProcessMatMulChunk(handle,
                                    cublasStream,
                                    chunkStart,
                                    chunkStart + chunkSize,
                                    &hAlpha,
                                    &hBeta,
                                    input,
                                    weight,
                                    output,
                                    M,
                                    N,
                                    K,
                                    column_parallel);
            cudaEventRecord(gemm_event, cublasStream);
            cudaStreamWaitEvent(ncclStream, gemm_event, 0);
            cudaEventRecord(nccl_event, ncclStream);
            NCCLCHECK(ncclAllReduce(output + chunkStart,
                                    output + chunkStart,
                                    chunkSize,
                                    ncclFloat16,
                                    ncclSum,
                                    comm,
                                    ncclStream));
            cudaStreamWaitEvent(cublasStream, nccl_event, 0);
        }

        cudaEventRecord(nccl_event, ncclStream);
        if (!column_parallel) {
            cudaStreamWaitEvent(ncclStream, nccl_event, 0);
            cudaStreamWaitEvent(cublasStream, nccl_event, 0);
        }
        cudaEventDestroy(nccl_event);
        cudaEventDestroy(gemm_event);
        return 0;
    } else {
        printf("Failed: kernel error %s:%d '%s'\n", __FILE__, __LINE__, "The setting value " \
                "(tp_atten_parallel > 0 or tp_mlp_parallel > 0) " \
                "needs to be divisible by batch*seq_len*hidden");            
        exit(EXIT_FAILURE); 
    }
}

int pipe_overlapped_with_split_kernel(cublasHandle_t handle,
                                      const float* input,
                                      const float* weight,
                                      float* output,
                                      int slit_n,
                                      int M,
                                      int N,
                                      int K,
                                      cudaStream_t cublasStream,
                                      ncclComm_t comm,
                                      cudaStream_t ncclStream,
                                      float alpha,
                                      float beta,
                                      bool column_parallel) {
    size_t M_N = static_cast<size_t>(M) * static_cast<size_t>(N);

    if (slit_n >= 0 && M_N % slit_n == 0) {
        
        size_t chunkSize = M_N / slit_n;
        cudaEvent_t gemm_event;
        cudaEventCreate(&gemm_event);
        cudaEvent_t nccl_event;
        cudaEventCreate(&nccl_event);
        cudaEventCreateWithFlags(&gemm_event, cudaEventBlockingSync);
        cudaEventCreateWithFlags(&nccl_event, cudaEventBlockingSync);

        for (size_t chunkStart = 0; chunkStart < M_N; chunkStart += chunkSize) {
            cublasProcessMatMulChunk(handle,
                                    cublasStream,
                                    chunkStart,
                                    chunkStart + chunkSize,
                                    &alpha,
                                    &beta,
                                    input,
                                    weight,
                                    output,
                                    M,
                                    N,
                                    K,
                                    column_parallel);
            cudaEventRecord(gemm_event, cublasStream);
            cudaStreamWaitEvent(ncclStream, gemm_event, 0);
            cudaEventRecord(nccl_event, ncclStream);
            NCCLCHECK(ncclAllReduce(output + chunkStart,
                                    output + chunkStart,
                                    chunkSize,
                                    ncclFloat32,
                                    ncclSum,
                                    comm,
                                    ncclStream));
            cudaStreamWaitEvent(cublasStream, nccl_event, 0);
        }

        cudaEventRecord(nccl_event, ncclStream);
        if (!column_parallel) {
            cudaStreamWaitEvent(ncclStream, nccl_event, 0);
            cudaStreamWaitEvent(cublasStream, nccl_event, 0);
        }
        cudaEventDestroy(nccl_event);
        cudaEventDestroy(gemm_event);
        return 0;
    } else {
        printf("Failed: kernel error %s:%d '%s'\n", __FILE__, __LINE__, "The setting value " \
                "(tp_atten_parallel > 0 or tp_mlp_parallel > 0) " \
                "needs to be divisible by batch*seq_len*hidden");            
        exit(EXIT_FAILURE); 
    }
}

int pipe_overlapped_with_split_kernel(cublasHandle_t handle,
                                      const double* input,
                                      const double* weight,
                                      double* output,
                                      int slit_n,
                                      int M,
                                      int N,
                                      int K,
                                      cudaStream_t cublasStream,
                                      ncclComm_t comm,
                                      cudaStream_t ncclStream,
                                      float alpha,
                                      float beta,
                                      bool column_parallel) {
    // TODO(lixiao31): support double date type

    return 0;
}

template <typename T>
int matmul_reduce_parallel_forward_cuda(at::Tensor input,
                                        T* weight,
                                        int in_features,
                                        int batch_size,
                                        int out_features,
                                        T* output,
                                        void* lt_workspace,
                                        ncclComm_t comm,
                                        cudaStream_t nccl_stream,
                                        int opt_num,
                                        float alpha,
                                        float beta,
                                        bool column_parallel) {
    int status = 1;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t cublas_stream;
    cublasGetStream(handle, &cublas_stream);

    status = pipe_overlapped_with_split_kernel(handle,
                                               input.data_ptr<T>(),
                                               weight,
                                               output,
                                               opt_num,
                                               batch_size,
                                               out_features,
                                               in_features,
                                               cublas_stream,
                                               comm,
                                               nccl_stream,
                                               alpha,
                                               beta,
                                               column_parallel);
    return status;
}

template int matmul_reduce_parallel_forward_cuda<at::Half>(
    at::Tensor input,
    at::Half* weight,
    int in_features,
    int batch_size,
    int out_features,
    at::Half* output,
    void* lt_workspace,
    ncclComm_t comm,
    cudaStream_t nccl_stream,
    int opt_num,
    float alpha,
    float beta,
    bool column_parallel);

template int matmul_reduce_parallel_forward_cuda<float>(
    at::Tensor input,
    float* weight,
    int in_features,
    int batch_size,
    int out_features,
    float* output,
    void* lt_workspace,
    ncclComm_t comm,
    cudaStream_t nccl_stream,
    int opt_num,
    float alpha,
    float beta,
    bool column_parallel);

template int matmul_reduce_parallel_forward_cuda<double>(
    at::Tensor input,
    double* weight,
    int in_features,
    int batch_size,
    int out_features,
    double* output,
    void* lt_workspace,
    ncclComm_t comm,
    cudaStream_t nccl_stream,
    int opt_num,
    float alpha,
    float beta,
    bool column_parallel);
