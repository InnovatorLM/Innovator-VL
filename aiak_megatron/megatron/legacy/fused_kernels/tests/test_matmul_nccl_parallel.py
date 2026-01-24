# run cmd
# export PYTHONPATH="../../../../Megatron/"
# GPUS_PER_NODE=4
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# MASTER_PORT=${MASTER_PORT:-9999}
# NNODES=${1:-1}
# NODE_RANK=${2:-0}

# python -m torch.distributed.launch --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT test_matmul_nccl_parallel.py


import argparse
import torch
import os
import torch.distributed as dist
import torch.distributed.distributed_c10d
from megatron.fused_kernels import load
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()
args.rank = 0
args.tp_gemm_nccl_parallel = True
args.masked_softmax_fusion = False
args.apply_rotary_positional_embedding_kernel = False
load(args)

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")

#  test half
m1 = torch.randn((4, 4), dtype = torch.half, device = device)
n1 = torch.randn((5, 4), dtype = torch.half, device = device)

output2 = torch.matmul(m1, n1.t())
dist.all_reduce(output2)
# print("torch_matmul=reduce=: \n ", output2)

for x in range(1, 10): 
    import matmul_reduce_parallel_cuda
    output1 = matmul_reduce_parallel_cuda.matmul_reduce_parallel(m1, 
                            n1, False, 4, local_rank, dist.distributed_c10d._get_default_group())
# print("====overlap=reduce=========: \n", output1)

np.testing.assert_allclose(output1.cpu().numpy(),
                                    output2.cpu().numpy(),
                                    rtol=1,
                                    atol=0.03)

#  test float32
m1 = torch.randn((64, 64), device = device)
n1 = torch.randn((128, 64), device = device)

output2 = torch.matmul(m1, n1.t())
dist.all_reduce(output2)
# print("torch_matmul=reduce=: \n ", output2)

for x in range(1, 10): 
    import matmul_reduce_parallel_cuda
    output1 = matmul_reduce_parallel_cuda.matmul_reduce_parallel(m1, 
                            n1, False, 4, local_rank, dist.distributed_c10d._get_default_group())
# print("====overlap=reduce=========: \n", output1)
np.testing.assert_allclose(output1.cpu().numpy(),
                                    output2.cpu().numpy(),
                                    rtol=1,
                                    atol=0.03)