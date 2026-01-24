"""Tests for rope_utils.py"""

import os
import torch
import pytest
from unittest.mock import patch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
    fused_mla_apply_rotary_pos_emb,
)

def gradient_check_loss(output):
    return output.pow(2).mean()

@pytest.mark.parametrize("multi_latent_attention", ['mla','normal'])
@pytest.mark.parametrize("rope_type", ['yarn', 'rope'])
@pytest.mark.parametrize("mscale_factor", ['mscale_0.5', 'mscale_1', 'mscale_2'])
@pytest.mark.parametrize("qkv_format", ['thd','bshd'])
class TestMLAApplyRotaryPosEmb:
    
    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self, multi_latent_attention, rope_type, mscale_factor, qkv_format):
        self.qkv_format = qkv_format
        self.apply_rope = apply_rotary_pos_emb
        self.fused_rope = fused_mla_apply_rotary_pos_emb if multi_latent_attention == 'mla' else apply_rotary_pos_emb

        if multi_latent_attention == 'mla':
            from megatron.core.transformer.multi_latent_attention import MLATransformerConfig
            self.transformer_config = MLATransformerConfig(
                num_layers=2,
                hidden_size=12,
                num_attention_heads=4,
                use_cpu_initialization=True,
                q_lora_rank=32,
                kv_lora_rank=32,
                qk_head_dim=128,
                v_head_dim=128,
                qk_pos_emb_head_dim=64,
                rope_type=rope_type,
                rotary_base=10000,
                max_position_embeddings=32,
            )
            if rope_type == 'rope' and (mscale_factor != 'mscale_1'):
                pytest.skip("Only yarn support mscale and mscale_all_dim.")

            if mscale_factor == 'mscale_0.5':
                self.transformer_config.mscale_all_dim = self.transformer_config.mscale_all_dim * 2
            if mscale_factor == 'mscale_2':
                self.transformer_config.mscale = self.transformer_config.mscale * 2

            if self.transformer_config.rope_type == "rope":
                self.rotary_pos_emb = RotaryEmbedding(
                    self.transformer_config.qk_pos_emb_head_dim,
                    rotary_percent=self.transformer_config.rotary_percent,
                    rotary_base=self.transformer_config.rotary_base,
                )
            elif self.transformer_config.rope_type == "yarn":
                self.rotary_pos_emb = YarnRotaryEmbedding(
                    self.transformer_config.qk_pos_emb_head_dim,
                    rotary_base=self.transformer_config.rotary_base,
                    scaling_factor=self.transformer_config.rotary_scaling_factor,
                    original_max_position_embeddings=self.transformer_config.max_position_embeddings,
                    beta_fast=self.transformer_config.beta_fast,
                    beta_slow=self.transformer_config.beta_slow,
                    mscale=self.transformer_config.mscale,
                    mscale_all_dim=self.transformer_config.mscale_all_dim,
                )
        else:
            if rope_type == 'yarn':
                pytest.skip("Only MLA supports yarn type.")
            if rope_type == 'rope' and (mscale_factor != 'mscale_1'):
                pytest.skip("Only MLA supports yarn type with mscale.")

            from megatron.core.transformer.transformer_config import TransformerConfig
            self.transformer_config = TransformerConfig(
                num_layers=2,
                hidden_size=12,
                num_attention_heads=4,
                use_cpu_initialization=True
            )

            self.rotary_pos_emb = RotaryEmbedding(
                64,
                rotary_percent=1,
                rotary_base=10000,
            )
    
    @patch('megatron.core.parallel_state.get_context_parallel_world_size')
    @patch('megatron.core.parallel_state.get_context_parallel_rank')
    def test_apply_rotary_pos_emb(self, mock_rank, mock_size):
        mock_size.return_value = 1
        mock_rank.return_value = 0

        # s = sequence length, b = batch size, h = hidden size, n = num attention heads
        # Attention heads [s, b, n*h]
        sequence_length = 1024
        batch_size = 2

        # q_pos_emb: [s, b, n, 64], k_pos_emb:[s, b, 1, 64]
        q_pos_emb = torch.randn(
            (sequence_length, batch_size, self.transformer_config.hidden_size, 64),
            device='cuda',
            requires_grad=True
        )
        k_pos_emb = torch.randn(
            (sequence_length, batch_size, 1, 64),
            device='cuda',
            requires_grad=True
        )

        if self.qkv_format == 'thd':
            cu_seqlens = torch.tensor(
                [0, 400, 542, 711, 727, 752, 1270, 1426, 1450, 1954, 2044, 2048],
                dtype=torch.int32,
                device='cuda',
            )
            packed_seq_params = PackedSeqParams(
                qkv_format = 'thd',
                cu_seqlens_q = cu_seqlens,
                cu_seqlens_kv = cu_seqlens,
                cu_seqlens_q_padded = None,
                cu_seqlens_kv_padded = None,
                max_seqlen_q = 2048,
                max_seqlen_kv = 2048,
            )

            # convert to thd
            q_pos_emb = q_pos_emb.reshape(-1, q_pos_emb.shape[2], q_pos_emb.shape[3])
            k_pos_emb = k_pos_emb.reshape(-1, k_pos_emb.shape[2], k_pos_emb.shape[3])
        else:
            packed_seq_params = None

        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, q_pos_emb, self.transformer_config, packed_seq_params
        )

        # rotary_pos_emb:[s, 1, 1, 64]
        mscale = 1.0
        if self.transformer_config.multi_latent_attention:
            if self.transformer_config.rope_type == "rope":
                packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
                rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
            else:
                rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len)
        else:
            packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        
        if packed_seq_params is not None:
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        self.transformer_config.apply_rope_fusion = False
        q_pos_emb_ref = self.apply_rope(
            q_pos_emb.clone(), rotary_pos_emb, self.transformer_config, cu_seqlens=cu_seqlens_q, mscale=mscale
        )
        k_pos_emb_ref = self.apply_rope(
            k_pos_emb.clone(), rotary_pos_emb, self.transformer_config, cu_seqlens=cu_seqlens_kv, mscale=mscale
        )

        self.transformer_config.apply_rope_fusion = True
        q_pos_emb_fused = self.fused_rope(
            q_pos_emb.clone(), rotary_pos_emb, self.transformer_config, cu_seqlens=cu_seqlens_q, mscale=mscale
        )
        k_pos_emb_fused = self.fused_rope(
            k_pos_emb.clone(), rotary_pos_emb, self.transformer_config, cu_seqlens=cu_seqlens_kv, mscale=mscale
        )
        
        assert q_pos_emb_ref.shape == q_pos_emb.shape
        assert k_pos_emb_ref.shape == k_pos_emb.shape
        assert q_pos_emb_fused.shape == q_pos_emb.shape
        assert k_pos_emb_fused.shape == k_pos_emb.shape

        # check forward accuracy
        torch.testing.assert_close(q_pos_emb_ref, q_pos_emb_fused)
        torch.testing.assert_close(k_pos_emb_ref, k_pos_emb_fused)

        grads = {}
        def save_grad(name):
            def hook(grad):
                grads[name] = grad.clone()
            return hook
        q_pos_emb_ref.register_hook(save_grad('q_pos_emb_ref'))
        k_pos_emb_ref.register_hook(save_grad('k_pos_emb_ref'))
        q_pos_emb_fused.register_hook(save_grad('q_pos_emb_fused'))
        k_pos_emb_fused.register_hook(save_grad('k_pos_emb_fused'))

        loss_ref = gradient_check_loss(q_pos_emb_ref) + gradient_check_loss(k_pos_emb_ref)
        loss_fused = gradient_check_loss(q_pos_emb_fused) + gradient_check_loss(k_pos_emb_fused)
        loss_ref.backward()
        loss_fused.backward()

        q_pos_emb_grad_ref = grads['q_pos_emb_ref'].clone()
        k_pos_emb_grad_ref = grads['k_pos_emb_ref'].clone()
        q_pos_emb_grad_fused = grads['q_pos_emb_fused'].clone()
        k_pos_emb_grad_fused = grads['k_pos_emb_fused'].clone()

        # check backward accuracy
        torch.testing.assert_close(q_pos_emb_grad_ref, q_pos_emb_grad_fused)
        torch.testing.assert_close(k_pos_emb_grad_ref, k_pos_emb_grad_fused)
