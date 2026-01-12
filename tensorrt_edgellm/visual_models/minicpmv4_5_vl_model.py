import math
from typing import Any, List, Tuple

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
from modelopt.torch.quantization.nn import TensorQuantizer

from ..onnx_export.onnx_utils import export_onnx

import numpy as np
from .modeling_navit_siglip import SiglipVisionTransformer
from .resampler import Resampler
import torch
import torch.nn as nn

DEVICE = 'cuda'
MAX_SIZE = (70, 70)
DTYPE_TORCH=torch.float32

def get_position_ids(
    patch_attention_mask,
    pixel_values_shape,
    patch_size,
    num_patches_per_side,
    tgt_sizes,
):
    batch_size, _, max_im_h, max_im_w  = pixel_values_shape
    max_nb_patches_h, max_nb_patches_w = max_im_h // patch_size, max_im_w // patch_size
    boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)

    position_ids = torch.full(
        size=(
            batch_size, # 1
            max_nb_patches_h * max_nb_patches_w, # 1 * 1024 = 1024
        ),
        fill_value=0,
    ) # (1, 1024) all 0

    for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
        if tgt_sizes is not None:
            nb_patches_h = tgt_sizes[batch_idx][0]
            nb_patches_w = tgt_sizes[batch_idx][1]
        else:
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

        fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
        fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

        pos_ids = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()
        position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

    return position_ids

def get_position_embedding_resampler(
    embed_dim,
    tgt_sizes
):
    bs = tgt_sizes.shape[0]
    patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

    max_h = torch.max(tgt_sizes[:, 0])
    max_w = torch.max(tgt_sizes[:, 1])

    max_size = MAX_SIZE
    if max_h > MAX_SIZE[0] or max_w > MAX_SIZE[1]:
        max_size = [max(max_h, MAX_SIZE[0]), max(max_w, MAX_SIZE[1])]
    candidate_pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, max_size)).float()
    print(f'pos_emb_lut_resampler.shape: {candidate_pos_embed.reshape(-1, embed_dim).shape}')
    #candidate_pos_embed.reshape(-1, embed_dim).numpy().tofile(f"{OUTPUT_FILE_POS_EMB_LUT_RESAMPLER}")
    #print(f'pos_emb_lut_resampler saved to {OUTPUT_FILE_POS_EMB_LUT_RESAMPLER}')
    max_patch_len = torch.max(patch_len)
    key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=DEVICE)

    pos_embed = []
    for i in range(bs):
        tgt_h, tgt_w = tgt_sizes[i]
        pos_embed.append(candidate_pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(DTYPE_TORCH))  # patches * D
        key_padding_mask[i, patch_len[i]:] = True

    pos_embed = torch.nn.utils.rnn.pad_sequence(
        pos_embed,
        batch_first=True,
        padding_value=0.0
    ).permute(1, 0, 2)  # BLD => L * B * D

    return pos_embed

def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    out = np.einsum('hw,d->hwd', pos, omega)  # (H, W, D/2), outer product

    emb_sin = np.sin(out)  # (H, W, D/2)
    emb_cos = np.cos(out)  # (H, W, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])  # (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])  # (H, W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

class EmbVpmResampler(nn.Module):
    def __init__(self, vpm, resampler, config):
        super().__init__()
        self.vpm = vpm
        self.resampler = resampler
        self.config = config 

    def forward(
        self,
        pixel_values: torch.Tensor,
        position_embedding_vpm: torch.Tensor = None,
        position_embedding_resampler: torch.Tensor = None
    ):
        hidden_states = self.vpm(
            pixel_values=pixel_values,
            position_embedding=position_embedding_vpm
            #attention_mask=attention_mask,
        ) # 1, 1024, 1152
        # print('hidden_states.shape after vpm:',hidden_states.shape)

        hidden_states = self.resampler(
            hidden_states,
            position_embedding_resampler
            #attention_mask=attention_mask,
        ) # 1, 64, 1536
        # print('hidden_states.shape after resampler:',hidden_states.shape)

        return hidden_states


def export_minicpmv4_5_visual(
    model: EmbVpmResampler,
    output_dir: str,
    torch_dtype: torch.dtype,
) -> None:
    """
    Export MiniCPMV4_5 visual model to ONNX format.

    This function takes a patched MiniCPMV4_5 visual model, prepares dummy inputs 
    for ONNX export, and saves the model in ONNX format.
    
    Args:
        model: Patched MiniCPMV4_5 vision transformer model
        output_dir: Directory to save the exported ONNX model
        torch_dtype: PyTorch data type for the model
    """
    # create input tensors
    vpm = model.vpm
    resampler = model.resampler
    vision_config = model.config.vision_config
    PATCH_SIZE = vision_config.patch_size
    # TGT_HEIGHT, TGT_WIDTH = 32, 32
    TGT_HEIGHT, TGT_WIDTH = 1, 1024
    PIXEL_VALUES_SHAPE = (1, 3, PATCH_SIZE * TGT_HEIGHT, PATCH_SIZE * TGT_WIDTH) # cloud, torch.float32 from -1 to 1 (1, 3, 448, 448)
    PATCH_MASK_SHAPE = (1, 1, TGT_HEIGHT * TGT_WIDTH)
    TGT_SIZES = torch.tensor([[32, 32]], device=DEVICE, dtype=torch.int32)
    NUM_PATCHES_PER_SIDE = 70
    position_embeddings_vpm = vpm.embeddings.position_embedding # (4900, 1152)
    pixel_values = torch.rand(size=PIXEL_VALUES_SHAPE, dtype=model.vpm.dtype, device=DEVICE) # (1, 3, 14, 14336)
    pixel_values = (pixel_values - 0.5) / 0.5
    patch_attention_mask = torch.ones(size=PATCH_MASK_SHAPE, dtype=torch.bool, device=DEVICE)

    position_ids = get_position_ids(
        patch_attention_mask=patch_attention_mask,
        pixel_values_shape=pixel_values.shape,
        patch_size=PATCH_SIZE,
        num_patches_per_side=NUM_PATCHES_PER_SIDE,
        tgt_sizes=TGT_SIZES,
    ) # (1, 1024)
    position_ids = position_ids.to(DEVICE)
    position_embedding_vpm = position_embeddings_vpm(position_ids) # (1, 1024, 1152)

    position_embedding_resampler = get_position_embedding_resampler(
        embed_dim=model.config.hidden_size,
        tgt_sizes=TGT_SIZES
    )
    position_embedding_resampler = position_embedding_resampler.to(DEVICE).to(model.vpm.dtype) # (1024, 1, 1536)

    inputs = (
        pixel_values, # (1, 3, 14, 14336)
        position_embedding_vpm, # (1, 1024, 1152)
        position_embedding_resampler # (1024, 1, 1536)
    ) 
    input_names = [
        "pixel_values",
        "position_embedding_vpm",
        "position_embedding_resampler"
    ]
    output_names = ["last_hidden_state"]

    dynamic_axes = {}
    export_onnx(model, inputs, output_dir, input_names, output_names, dynamic_axes=dynamic_axes)