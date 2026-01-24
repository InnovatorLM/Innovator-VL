# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""tranformer layer."""

import warnings
import math
from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List

import json
import torch
import torch.distributed
from torch.autograd.variable import Variable

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.transformer.cuda_graphs import CudaGraphManager
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor

# from megatron.core.tensor_parallel import RecomputeManager


def _get_layer_offset_for_custom_pp(custom_pipeline_layers: List[int], vpp_scheduler):
    """Calculates the offset from the start of a transformer layer based on
    the provided custom pipeline layers configuration."""
    pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
    pipeline_size = parallel_state.get_pipeline_model_parallel_world_size()

    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        # calculate the number of layers per virtual chunk per pipeline stage
        # for example, custom_pipeline_layers=[8, 7, 6, 5], vp_size=4
        # layers_per_virtual_chunk_per_pp = [[2, 2, 2, 2], [1, 2, 2, 2], [1, 1, 2, 2], [1, 1, 1, 2]]
        layers_per_virtual_chunk_per_pipeline_rank = [
            [q] * (vp_size - r) + [q + 1] * r
            for q, r in (divmod(num, vp_size) for num in custom_pipeline_layers)
        ]

        if vpp_scheduler == "dualpipev":
            assert (
                vp_size == 2
            ), "When using dual pipe, virtual pipeline parallelism size must be 2"

            if vp_rank == 0:
                offset = 0
                for i in range(pipeline_rank):
                    offset += layers_per_virtual_chunk_per_pipeline_rank[i][0]
            else:
                offset = sum(
                    [
                        sublist[0]
                        for sublist in layers_per_virtual_chunk_per_pipeline_rank
                    ]
                )
                for i in range(pipeline_size - pipeline_rank - 1):
                    offset += layers_per_virtual_chunk_per_pipeline_rank[-i][1]

        else:
            offset = 0
            # calculate the total number of virtual chunks across all pp stages before the current vp rank
            for i in range(vp_rank):
                offset += sum(
                    [
                        sublist[i]
                        for sublist in layers_per_virtual_chunk_per_pipeline_rank
                    ]
                )
            # calculate the number of virtual chunks before the current pp stage
            for i in range(pipeline_rank):
                offset += layers_per_virtual_chunk_per_pipeline_rank[i][vp_rank]
    else:
        # 1F1B
        offset = sum(custom_pipeline_layers[:pipeline_rank])

    return offset


def get_transformer_layer_offset(config: TransformerConfig):
    """Get the index offset of current pipeline stage, given the level of pipelining."""
    pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
    if not parallel_state.is_inside_encoder():
        pp_decoder_start = parallel_state.get_pipeline_model_parallel_decoder_start()
        if pp_decoder_start is not None:
            pipeline_rank = pipeline_rank - pp_decoder_start

    if config.pipeline_model_parallel_size > 1:

        if (
            config.num_layers_in_first_pipeline_stage is not None
            or config.num_layers_in_last_pipeline_stage is not None
        ):
            # Calculate number of pipeline stages to distribute the remaining Transformer
            # layers after deducting the Transformer layers in the first or the last stages
            middle_pipeline_stages = config.pipeline_model_parallel_size
            middle_pipeline_stages -= sum(
                [
                    1 if x is not None else 0
                    for x in (
                        config.num_layers_in_first_pipeline_stage,
                        config.num_layers_in_last_pipeline_stage,
                    )
                ]
            )

            # Calculate layers to distribute in each pipeline stage. If the
            # num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage
            # are not set, we will not enable uneven pipeline. All layers will be treated
            # as middle layers.
            num_layers_in_first_pipeline_stage = (
                0
                if config.num_layers_in_first_pipeline_stage is None
                else config.num_layers_in_first_pipeline_stage
            )
            num_layers_in_last_pipeline_stage = (
                0
                if config.num_layers_in_last_pipeline_stage is None
                else config.num_layers_in_last_pipeline_stage
            )

            middle_num_layers = (
                config.num_layers
                - num_layers_in_first_pipeline_stage
                - num_layers_in_last_pipeline_stage
            )

            if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
                vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

                # Calculate number of layers in each virtual model chunk
                # If the num_layers_in_first_pipeline_stage and
                # num_layers_in_last_pipeline_stage are not set, all pipeline stages
                # will be treated as middle pipeline stages in the calculation
                num_layers_virtual_in_first_stage = []
                if config.num_layers_in_first_pipeline_stage:
                    q, r = divmod(config.num_layers_in_first_pipeline_stage, vp_size)
                    num_layers_virtual_in_first_stage = [q] * (vp_size - r) + [q + 1] * r

                num_layers_virtual_in_last_stage = []
                if config.num_layers_in_last_pipeline_stage:
                    q, r = divmod(config.num_layers_in_last_pipeline_stage, vp_size)
                    num_layers_virtual_in_last_stage = [q] * (vp_size - r) + [q + 1] * r

                num_layers_per_middle_stage = middle_num_layers // middle_pipeline_stages
                q, r = divmod(num_layers_per_middle_stage, vp_size)
                num_layers_virtual_in_middle_stage = [q] * (vp_size - r) + [q + 1] * r

                # calculate the number of layers per virtual chunk per pipeline stage
                # for example, pipeline_layers=[8, 7, 6, 5], vp_size=4
                # layers_per_virtual_chunk_per_pp = [[2, 2, 2, 2], [1, 2, 2, 2], [1, 1, 2, 2], [1, 1, 1, 2]]
                layers_per_virtual_chunk_per_pipeline_rank = (
                    (
                        [num_layers_virtual_in_first_stage]
                        if config.num_layers_in_first_pipeline_stage
                        else []
                    )
                    + ([num_layers_virtual_in_middle_stage for _ in range(middle_pipeline_stages)])
                    + (
                        [num_layers_virtual_in_last_stage]
                        if config.num_layers_in_last_pipeline_stage
                        else []
                    )
                )

                # Calculate the layer offset with interleaved uneven pipeline parallelism
                if config.vpp_scheduler == "dualpipev":
                    assert (
                        vp_size == 2
                    ), "When using dual pipe, virtual pipeline parallelism size must be 2"
                    pipeline_size = parallel_state.get_pipeline_model_parallel_world_size()

                    if vp_rank == 0:
                        offset = 0
                        for i in range(pipeline_rank):
                            offset += layers_per_virtual_chunk_per_pipeline_rank[i][0]
                    else:
                        offset = sum(
                            [sublist[0] for sublist in layers_per_virtual_chunk_per_pipeline_rank]
                        )
                        for i in range(pipeline_size - pipeline_rank - 1):
                            offset += layers_per_virtual_chunk_per_pipeline_rank[-i][1]

                else:
                    offset = 0
                    for i in range(vp_rank):
                        offset += sum(
                            [sublist[i] for sublist in layers_per_virtual_chunk_per_pipeline_rank]
                        )
                    for i in range(pipeline_rank):
                        offset += layers_per_virtual_chunk_per_pipeline_rank[i][vp_rank]

            else:
                if middle_pipeline_stages > 0:
                    num_layers_per_pipeline_rank = middle_num_layers // middle_pipeline_stages
                else:
                    num_layers_per_pipeline_rank = 0

                middle_pipeline_rank = (
                    pipeline_rank
                    if config.num_layers_in_first_pipeline_stage is None
                    else pipeline_rank - 1
                )

                if pipeline_rank == 0:
                    offset = 0
                else:
                    offset = (
                        middle_pipeline_rank * num_layers_per_pipeline_rank
                    ) + num_layers_in_first_pipeline_stage

        elif config.custom_pipeline_layers is not None:
            offset = _get_layer_offset_for_custom_pp(
                config.custom_pipeline_layers, config.vpp_scheduler
            )

        else:
            num_layers = config.num_layers

            # Increase the number of layers by one if we include the embedding (loss)
            # layer into pipeline parallelism partition and placement
            if config.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if config.account_for_loss_in_pipeline_split:
                num_layers += 1

            num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

            if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
                vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
                pipeline_size = parallel_state.get_pipeline_model_parallel_world_size()

                q, r = divmod(num_layers_per_pipeline_rank, vp_size)
                layers_per_virtual_chunk_per_pipeline_rank = [q] * (vp_size - r) + [q + 1] * r

                if config.vpp_scheduler == "dualpipev":
                    assert (
                        vp_size == 2
                    ), "When using dual pipe, virtual pipeline parallelism size must be 2"

                    if vp_rank == 0:
                        offset = 0
                        for i in range(pipeline_rank):
                            offset += layers_per_virtual_chunk_per_pipeline_rank[0]
                    else:
                        offset = sum(
                            [sublist[0] for sublist in layers_per_virtual_chunk_per_pipeline_rank]
                        )
                        for i in range(pipeline_size - pipeline_rank - 1):
                            offset += layers_per_virtual_chunk_per_pipeline_rank[1]
                else:
                    offset = 0
                    for i in range(vp_rank):
                        offset += layers_per_virtual_chunk_per_pipeline_rank[i] * pipeline_size
                    for i in range(pipeline_rank):
                        offset += layers_per_virtual_chunk_per_pipeline_rank[vp_rank]

                # Reduce the offset of embedding layer from the total layer number
                if (
                    config.account_for_embedding_in_pipeline_split
                    and not parallel_state.is_pipeline_first_stage()
                ):
                    offset -= 1
            else:
                offset = pipeline_rank * num_layers_per_pipeline_rank

                # Reduce the offset of embedding layer from the total layer number
                if (
                    config.account_for_embedding_in_pipeline_split
                    and not parallel_state.is_pipeline_first_stage()
                ):
                    offset -= 1

    else:
        offset = 0

    return offset


@dataclass
class TransformerLayerSubmodules:
    """
    Configuration class for specifying the submodules of a transformer layer.

    This class defines the structure and default implementations for various
    components of a transformer layer, allowing for flexible customization
    of the layer's architecture.

    Args:
        input_layernorm (Union[ModuleSpec, type]): Specification for the input layer normalization.
        self_attention (Union[ModuleSpec, type]): Specification for the self-attention mechanism.
        self_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after self-attention.
        pre_cross_attn_layernorm (Union[ModuleSpec, type]): Specification for the layer
            normalization before cross-attention.
        cross_attention (Union[ModuleSpec, type]): Specification for the cross-attention mechanism.
        cross_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after cross-attention.
        pre_mlp_layernorm (Union[ModuleSpec, type]): Specification for the layer normalization
            before the MLP.
        mlp (Union[ModuleSpec, type]): Specification for the MLP in Dense layer.
        mlp_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after the MLP.
        sharded_state_dict_keys_map (Dict[str, str]): Mapping for sharded tensor keys to be applied
            in the `sharded_state_dict` method.
    """

    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class BaseTransformerLayer(ABC):
    """A common parent class for `TransformerLayer` like implementations.

    A dummy class that is subclassed by similar `TransformerLayer`s e.g. the
    `TransformerLayer` in this file and possibly other `TransformerLayer`
    implementations that aim to use `TransformerBlock` as the base module.
    The main purpose is to check if any layer (or module) provided in the spec
    is a subclass of this class to allow fanning-out of that spec for all the
    layers in the `TransformerBlock`. See `_get_block_submodules` method
    implementation in `transformer_block.py` file for more details.
    """

    def __init__(self):
        pass


class TransformerLayer(MegatronModule, BaseTransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        **kwargs,
    ):
        super().__init__(config=config)

        if config.enable_cuda_graph:
            if not self.training:
                # Cudagraphs for inference are only enabled with the flash decoding kernel
                assert self.config.flash_decode, "--flash-decode is required to use CUDA graphs during inference"
            self.cudagraph_manager = CudaGraphManager(config)

        self.submodules_config = submodules
        self.layer_number = layer_number + get_transformer_layer_offset(self.config)
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        # [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        attention_optional_kwargs = {}
        if config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[self.layer_number]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type

        # [Module 2: SelfAttention]
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )

        # [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 5: CrossAttention]
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )

        # [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config)

        # [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        # [Module 8: MLP block]
        self.mlp = build_module(submodules.mlp, config=self.config)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        # [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

    @staticmethod
    def _get_layer_offset(config: TransformerConfig):
        """
        Get the layer offset for the current pipeline stage.

        Deprecated: please use `get_transformer_layer_offset` instead.
        """

        warnings.warn(
            "TransformerLayer._get_layer_offset is deprecated."
            "Please use get_transformer_layer_offset instead."
        )
        return get_transformer_layer_offset(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        attn_mask_type=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        **kwargs,
    ):
        # hidden_states: [s, b, h]
        # Residual connection.
        residual = hidden_states
        # Optional Input Layer norm
        if self.config.layernorm_recompute:
            self.input_layernorm_recompute_manager = tensor_parallel.RecomputeManager()
            input_layernorm_output = self.input_layernorm_recompute_manager.checkpoint(
                self.input_layernorm, False, hidden_states
            )
        else:
            input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        if self.config.layernorm_recompute:
            self.input_layernorm_recompute_manager.discard_output()
            if attention_output_with_bias[0].requires_grad:
                attention_output_with_bias[0].register_hook(self.input_layernorm_recompute_manager.recompute)


        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            attn_mask_type=attn_mask_type,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        if self.config.layernorm_recompute:
            self.pre_mlp_layernorm_recompute_manager = tensor_parallel.RecomputeManager()
            pre_mlp_layernorm_output = self.pre_mlp_layernorm_recompute_manager.checkpoint(
                                                            self.pre_mlp_layernorm, False, hidden_states)
        else:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        if self.config.layernorm_recompute:
            self.pre_mlp_layernorm_recompute_manager.discard_output()
            if mlp_output_with_bias[0].requires_grad:
                mlp_output_with_bias[0].register_hook(self.pre_mlp_layernorm_recompute_manager.recompute)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        
        # CUDA graph requires returned values to be Tensors
        if self.config.external_cuda_graph and self.training:
            return output

        return output, context

    def is_deepep_dispatcher(self):
        """ is the dispatcher a DeepEP dispatcher? """
        from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher
        return isinstance(self.mlp.token_dispatcher, MoEFlexTokenDispatcher)

    def _submodule_attention_forward(
            self,
            hidden_states,
            attention_mask=None,
            attn_mask_type=None,
            inference_params=None,
            rotary_pos_emb=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            attention_bias=None,
            packed_seq_params=None,
            sequence_len_offset=None,
    ):
        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            # sequence_len_offset=sequence_len_offset,
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )
        return hidden_states

    def _submodule_attention_router_compound_forward(
            self,
            hidden_states,
            attention_mask=None,
            attn_mask_type=None,
            inference_params=None,
            rotary_pos_emb=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            attention_bias=None,
            packed_seq_params=None,
            sequence_len_offset=None,
    ):
        """
        Performs a combined forward pass that includes self-attention and MLP routing logic.
        """
        hidden_states = self._submodule_attention_forward(
            hidden_states,
            attention_mask,
            attn_mask_type,
            inference_params,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            attention_bias,
            packed_seq_params,
            sequence_len_offset)

        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
        probs, routing_map = self.mlp.router(pre_mlp_layernorm_output)

        if self.is_deepep_dispatcher():
            deepep_hidden_states = self.mlp.token_dispatcher.dispatch_preprocess(pre_mlp_layernorm_output, routing_map,
                                                                                 probs)
            return (hidden_states, pre_mlp_layernorm_output, None, deepep_hidden_states,
                    self.mlp.token_dispatcher.get_token_probs())
        else:
            tokens_per_expert = self.mlp.token_dispatcher.meta_prepare(pre_mlp_layernorm_output, probs, routing_map)
            permutated_local_input_tokens = self.mlp.token_dispatcher.dispatch_preprocess(pre_mlp_layernorm_output)

            return hidden_states, pre_mlp_layernorm_output, tokens_per_expert, permutated_local_input_tokens, probs

    def _submodule_dispatch_forward(self, tokens, probs=None):
        """
        Dispatches tokens to the appropriate experts based on the router output.
        """
        if self.is_deepep_dispatcher():
            self.mlp.token_dispatcher.set_token_probs(probs)
            output_tokens = self.mlp.token_dispatcher._comm_manager.dispatch(tokens, True, True)
            probs = self.mlp.token_dispatcher.get_dispatched_probs()
            return output_tokens, probs
        else:
            assert probs is None, "For AlltoAll dispatcher, probs should be None."
            output_tokens = self.mlp.token_dispatcher.dispatch_all_to_all(tokens)
            return output_tokens

    def _submodule_dense_forward(self, hidden_states):
        residual = hidden_states
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output

    def _submodule_moe_forward(self, dispatched_input, hidden_states, probs, tokens_per_expert=None,
                               use_recompute=False):
        """
        Performs a forward pass for the MLP submodule, including both expert-based
        and optional shared-expert computations.
        """
        shared_expert_output = None
        if self.is_deepep_dispatcher():
            assert tokens_per_expert is None, "For DeepEP dispatcher, tokens_per_expert should be None."
            self.mlp.token_dispatcher.set_dispatched_probs(probs)
            probs = None
            dispatched_input, tokens_per_expert = self.mlp.token_dispatcher.dispatch_postprocess(dispatched_input)
        else:
            self.mlp.token_dispatcher.probs = probs
            dispatched_input = self.mlp.token_dispatcher.dispatch_postprocess(dispatched_input)
        if use_recompute:
            expert_output, mlp_bias = tensor_parallel.checkpoint(
                self.mlp.experts, False, dispatched_input, tokens_per_expert)
        else:
            expert_output, mlp_bias = self.mlp.experts(dispatched_input, tokens_per_expert)
        if self.is_deepep_dispatcher():
            expert_output = self.mlp.token_dispatcher._comm_manager.get_restored_hidden_states_by_experts(expert_output)
        else:
            expert_output = self.mlp.token_dispatcher.combine_preprocess(expert_output)
        if self.mlp.use_shared_expert and not self.mlp.shared_expert_overlap:
            if use_recompute:
                shared_expert_output = tensor_parallel.checkpoint(self.mlp.shared_experts, False, hidden_states)
            else:
                shared_expert_output = self.mlp.shared_experts(hidden_states)
        return expert_output, shared_expert_output, probs, mlp_bias

    def _submodule_combine_forward(self, output, shared_expert_output, mlp_bias, probs, residual):
        if self.is_deepep_dispatcher():
            output = self.mlp.token_dispatcher._comm_manager.combine(output, True, True).view(
                self.mlp.token_dispatcher.hidden_shape)
        else:
            self.mlp.token_dispatcher.probs = probs
            output = self.mlp.token_dispatcher.combine_all_to_all(output)
            output = self.mlp.token_dispatcher.combine_postprocess(output)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        mlp_output_with_bias = (output, mlp_bias)
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output

    def _submodule_post_combine_forward(self, output, shared_expert_output, mlp_bias, probs, residual):
        """
        Re-combines the expert outputs (and optional shared_expert_output) into the same order
        as the original input tokens, applying any required bias.
        """
        if not self.is_deepep_dispatcher():
            self.mlp.token_dispatcher.probs = probs
            output = self.mlp.token_dispatcher.combine_postprocess(output)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        mlp_output_with_bias = (output, mlp_bias)
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output

    def _submodule_custom_backward(self, stream, outputs, grads, inputs):
        outputs_new = []
        grads_new = []
        for output, grad in zip(outputs, grads):
            if isinstance(output, torch.Tensor) and output.requires_grad:
                outputs_new.append(output)
                grads_new.append(grad)

        Variable._execution_engine.run_backward(
            tensors=tuple(outputs_new),
            grad_tensors=tuple(grads_new),
            keep_graph=False,
            create_graph=False,
            inputs=tuple(),
            allow_unreachable=True,
            accumulate_grad=True,
        )
        for g in inputs:
            if isinstance(g, torch.Tensor) and g.requires_grad:
                g.record_stream(stream)
        return tuple([t.grad if t is not None else None for t in inputs])

    def _submodule_attention_router_compound_dw(self):
        #raise NotImplementedError("Not implemented")
        self.self_attention.backward_dw()

    def _submodule_mlp_dw(self):
        #raise NotImplementedError("Not implemented")
        self.mlp.backward_dw()

    def _submodule_identity_forward(self, *args):
        return args

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the transformer layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the transformer layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict

    def __call__(self, *args, **kwargs):
        # Training and validation mode CUDA graphs
        if hasattr(self, 'cudagraph_manager') and kwargs.get('inference_params') is None:
            return self.cudagraph_manager(self, args, kwargs)
        # Inference mode. CUDA graphs are used in the decode phase only, when attn mask is None
        elif (
            not self.training
            and hasattr(self, 'cudagraph_manager')
            and kwargs.get('inference_params') is not None
            and kwargs['inference_params'].decode_mode
        ):
            assert (
                kwargs.get('attention_mask') is None
            ), f"Attention mask must not be set when using CUDA graphs for decode"
            return self.cudagraph_manager(self, args, kwargs)
        return super(MegatronModule, self).__call__(*args, **kwargs)
