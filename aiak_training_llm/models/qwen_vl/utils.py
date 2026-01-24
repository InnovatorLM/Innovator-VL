""" Utils """

import torch
from megatron.core import  mpu
try:
    import transformer_engine_torch as tex
except ImportError:
    tex = None


class _Select(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, val):
        """Forward function."""
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            return val

        cp_rank = mpu.get_context_parallel_rank()
        index = get_select_ids(cp_rank, cp_size)

        val = val.view(
            2 * cp_size,
            val.shape[0] // (2 * cp_size),
            *val.shape[1:]
        )
        val = val.index_select(0, index)
        val = val.view(-1, *val.shape[2: ])
        return val

    @staticmethod
    def backward(ctx, val):
        """Backward function."""
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            return val

        output = torch.zeros(
            2 * cp_size,
            val.shape[0] // 2,
            *val.shape[1:],
            dtype=val.dtype,
            device=val.device
        )
        cp_rank = mpu.get_context_parallel_rank()
        index = get_select_ids(cp_rank, cp_size)
        output[index] = val.view(2, -1, *val.shape[1:])
        output = output.view(-1, *output.shape[2: ])
        return output


def get_select_ids(cp_rank, cp_size):
    """ Get select ids for each gpu."""
    return torch.tensor(
        [cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True
    ).cuda(non_blocking=True)


def get_inputs_on_this_cp_rank(val):
    """ Slice input along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
    """
    return _Select.apply(val)