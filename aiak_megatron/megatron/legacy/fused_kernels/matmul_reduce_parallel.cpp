#include <nccl.h>
#include <pybind11/chrono.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <torch/torch.h>
#include <vector>

#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

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

// Return CUDA device with ordinal given by input rank.
at::Device get_device_for_rank(int rank) {
    TORCH_CHECK(rank >= 0, "Invalid rank ", rank);
    auto numGPUs = at::cuda::getNumGPUs();
    int16_t deviceIdx = static_cast<int16_t>(rank % numGPUs);
    return at::Device(at::DeviceType::CUDA, deviceIdx);
}

// Get the deviceList String from the list of devices
std::string get_key_from_devices(const std::vector<at::Device>& devices) {
    std::string deviceList;
    for (auto& device : devices) {
        if (deviceList.empty()) {
            deviceList = std::to_string(device.index());
        } else {
            deviceList += "," + std::to_string(device.index());
        }
    }
    return deviceList;
}

class NCCLCommExtForParallel : public c10d::NCCLComm {
public:
    ncclComm_t get_nccl_comm_ext() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (aborted_) {
            auto commFailureMsg =
                commFailureReason_ != c10::nullopt
                    ? c10::str(" Original reason for failure was: ",
                               *commFailureReason_)
                    : "";
            TORCH_CHECK(false,
                        c10::str("NCCL communicator was aborted on rank ",
                                 rank_,
                                 ". ",
                                 commFailureMsg));
        }
        return ncclComm_;
    }
};

// if want to use nccl_name, maybe only support pytorch >= 1.12
class ProcessGroupNCCLEXT : public c10d::ProcessGroupNCCL {
public:
    ncclComm_t get_comm_rank(const std::string& devicesKey,
                             const std::vector<at::Device>& devices,
                             c10d::OpType opType) {
        auto nccl_comm_vec =
            getNCCLComm(devicesKey, devices, c10d::OpType::ALLREDUCE);
        auto nccl_comm_ext =
            reinterpret_cast<NCCLCommExtForParallel*>((&(*(nccl_comm_vec[0]))));
        auto comm = nccl_comm_ext->get_nccl_comm_ext();
        return comm;
    }

    cudaStream_t get_nccl_cuda_stream(const std::string& devicesKey) {
        return ncclStreams_[devicesKey][0].stream();
    }
};

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
                                        bool column_parallel);

at::Tensor matmul_reduce_parallel_forward(at::Tensor input,
                                          at::Tensor weight,
                                          bool column_parallel,
                                          int opt_num,
                                          int rank_gpu,
                                          const c10d::ProcessGroup& pg) {
    auto tmp_pg = const_cast<c10d::ProcessGroup*>(&pg);
    auto backend = tmp_pg->getBackend(c10::DeviceType::CUDA);
    auto h_ = reinterpret_cast<ProcessGroupNCCLEXT*>(backend.get());

    std::vector<at::Device> rankDevice = {get_device_for_rank(rank_gpu)};
    const auto key = get_key_from_devices(rankDevice);
    auto nccl_comms =
        h_->get_comm_rank(key, rankDevice, c10d::OpType::ALLREDUCE);
    auto nccl_stream = h_->get_nccl_cuda_stream(key);

    // allocate fixed 4MB workspace for cublaslt for now, and this gets at least
    // 4 MB
    auto lt_workspace = at::empty({1 << 22}, input.type());

    float h_alpha = 1.0f;
    float h_beta = 0.0f;

    std::vector<int64_t> ret_shape;

    unsigned batch_size = 1;
    auto input_sizes = input.sizes();
    for (unsigned i = 0; i < input_sizes.size() - 1; i++) {
        batch_size = batch_size * input_sizes[i];
        ret_shape.push_back(input_sizes[i]);
    }

    auto in_features = input_sizes.back();
    int out_features = weight.size(0);

    if (column_parallel) {
        out_features = weight.size(1);
    }
    ret_shape.push_back(out_features);

    auto out = at::empty(ret_shape, input.type());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.type(), "matmul_reduce_parallel_forward", [&] {
            scalar_t* w_ptr = weight.data_ptr<scalar_t>();
            scalar_t* out_ptr = out.data_ptr<scalar_t>();
            auto result = matmul_reduce_parallel_forward_cuda<scalar_t>(
                input,
                w_ptr,
                in_features,
                batch_size,
                out_features,
                out_ptr,
                reinterpret_cast<void*>(lt_workspace.data_ptr<scalar_t>()),
                nccl_comms,
                nccl_stream,
                opt_num,
                h_alpha,
                h_beta,
                column_parallel);
        });
    return {out};
}

at::Tensor matmul_reduce_parallel_forward_nccl(at::Tensor input,
                                          at::Tensor weight,
                                          bool column_parallel,
                                          int opt_num,
                                          int rank_gpu,
                                          const c10d::ProcessGroupNCCL& pg) {
    auto tmp_p = const_cast<c10d::ProcessGroupNCCL*>(&pg);
    auto h_ = reinterpret_cast<ProcessGroupNCCLEXT*>(tmp_p);

    std::vector<at::Device> rankDevice = {get_device_for_rank(rank_gpu)};
    const auto key = get_key_from_devices(rankDevice);
    auto nccl_comms =
        h_->get_comm_rank(key, rankDevice, c10d::OpType::ALLREDUCE);
    auto nccl_stream = h_->get_nccl_cuda_stream(key);

    // allocate fixed 4MB workspace for cublaslt for now, and this gets at least
    // 4 MB
    auto lt_workspace = at::empty({1 << 22}, input.type());

    float h_alpha = 1.0f;
    float h_beta = 0.0f;

    std::vector<int64_t> ret_shape;

    unsigned batch_size = 1;
    auto input_sizes = input.sizes();
    for (unsigned i = 0; i < input_sizes.size() - 1; i++) {
        batch_size = batch_size * input_sizes[i];
        ret_shape.push_back(input_sizes[i]);
    }

    auto in_features = input_sizes.back();
    int out_features = weight.size(0);

    if (column_parallel) {
        out_features = weight.size(1);
    }
    ret_shape.push_back(out_features);

    auto out = at::empty(ret_shape, input.type());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.type(), "matmul_reduce_parallel_forward", [&] {
            scalar_t* w_ptr = weight.data_ptr<scalar_t>();
            scalar_t* out_ptr = out.data_ptr<scalar_t>();
            auto result = matmul_reduce_parallel_forward_cuda<scalar_t>(
                input,
                w_ptr,
                in_features,
                batch_size,
                out_features,
                out_ptr,
                reinterpret_cast<void*>(lt_workspace.data_ptr<scalar_t>()),
                nccl_comms,
                nccl_stream,
                opt_num,
                h_alpha,
                h_beta,
                column_parallel);
        });
    return {out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_reduce_parallel",
          &matmul_reduce_parallel_forward_nccl,
          "matmul_reduce_parallel forward",
          py::arg("input"),
          py::arg("weight"),
          py::arg("column_parallel"),
          py::arg("opt_num"),
          py::arg("rank_gpu"),
          py::arg("nccl_comm"));
    m.def("matmul_reduce_parallel",
          &matmul_reduce_parallel_forward,
          "matmul_reduce_parallel forward",
          py::arg("input"),
          py::arg("weight"),
          py::arg("column_parallel"),
          py::arg("opt_num"),
          py::arg("rank_gpu"),
          py::arg("nccl_comm"));
}
