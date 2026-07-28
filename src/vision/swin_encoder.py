from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .drop_path import DropPath
from .outputs import VisionEncoderOutput
from .patch_embedding import VisionPatchEmbedding
from .utils import (
    build_vision_norm,
    initialize_vision_module,
    to_2tuple,
    validate_token_grid,
)
from .vision_mlp import VisionMLP


def _partition_windows(
    x: torch.Tensor, window_size: int
) -> tuple[torch.Tensor, tuple[int, int, int, int, int]]:
    batch, height, width, dim = x.shape
    pad_h = (-height) % window_size
    pad_w = (-width) % window_size
    x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    padded_h, padded_w = height + pad_h, width + pad_w
    windows = x.reshape(
        batch,
        padded_h // window_size,
        window_size,
        padded_w // window_size,
        window_size,
        dim,
    )
    windows = windows.permute(0, 1, 3, 2, 4, 5).reshape(
        -1, window_size * window_size, dim
    )
    return windows, (batch, height, width, padded_h, padded_w)


def _reverse_windows(
    windows: torch.Tensor,
    window_size: int,
    metadata: tuple[int, int, int, int, int],
) -> torch.Tensor:
    batch, height, width, padded_h, padded_w = metadata
    dim = windows.shape[-1]
    x = windows.reshape(
        batch,
        padded_h // window_size,
        padded_w // window_size,
        window_size,
        window_size,
        dim,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch, padded_h, padded_w, dim)
    return x[:, :height, :width]


class WindowSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        *,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        use_relative_position_bias: bool = True,
    ):
        super().__init__()
        if dim <= 0 or num_heads <= 0 or window_size <= 0:
            raise ValueError("dim, num_heads and window_size must be > 0")
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.projection = nn.Linear(dim, dim, bias=proj_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection_dropout = nn.Dropout(projection_dropout)
        if use_relative_position_bias:
            side = 2 * window_size - 1
            self.relative_position_bias = nn.Parameter(
                torch.zeros(side * side, num_heads)
            )
            coords = torch.stack(
                torch.meshgrid(
                    torch.arange(window_size),
                    torch.arange(window_size),
                    indexing="ij",
                )
            ).flatten(1)
            relative = coords[:, :, None] - coords[:, None, :]
            relative = relative.permute(1, 2, 0).contiguous()
            relative += window_size - 1
            relative[..., 0] *= side
            self.register_buffer(
                "relative_position_index", relative.sum(-1), persistent=False
            )
            nn.init.trunc_normal_(self.relative_position_bias, std=0.02)
        else:
            self.relative_position_bias = None
            self.register_buffer("relative_position_index", None, persistent=False)

    def forward(
        self,
        windows: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if windows.ndim != 3 or windows.shape[-1] != self.dim:
            raise ValueError("windows have an invalid shape")
        batch_windows, count, _ = windows.shape
        expected = self.window_size**2
        if count != expected:
            raise ValueError(f"each window must contain {expected} tokens")
        qkv = self.qkv(windows).reshape(
            batch_windows, count, 3, self.num_heads, self.head_dim
        )
        q, k, value = qkv.permute(2, 0, 3, 1, 4)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.relative_position_bias is not None:
            bias = self.relative_position_bias[
                self.relative_position_index.reshape(-1)
            ]
            bias = bias.reshape(count, count, self.num_heads).permute(2, 0, 1)
            scores = scores + bias.unsqueeze(0)
        if attention_mask is not None:
            if attention_mask.shape != (batch_windows, count, count):
                raise ValueError("attention_mask does not match window batches")
            scores = scores.masked_fill(
                ~attention_mask[:, None], torch.finfo(scores.dtype).min
            )
        probabilities = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        if attention_mask is not None:
            probabilities = probabilities * attention_mask[:, None].to(
                probabilities.dtype
            )
            probabilities = probabilities / probabilities.sum(
                dim=-1, keepdim=True
            ).clamp_min(torch.finfo(probabilities.dtype).tiny)
        output = torch.matmul(self.attention_dropout(probabilities), value)
        output = output.transpose(1, 2).reshape(batch_windows, count, self.dim)
        output = self.projection_dropout(self.projection(output))
        return output, probabilities if output_attentions else None


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        *,
        mlp_ratio: float,
        norm_type: str,
        norm_eps: float,
        qkv_bias: bool,
        proj_bias: bool,
        mlp_bias: bool,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
        use_relative_position_bias: bool,
    ):
        super().__init__()
        if not 0 <= shift_size < window_size:
            raise ValueError("shift_size must be in [0, window_size)")
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = build_vision_norm(norm_type, dim, norm_eps)
        self.attention = WindowSelfAttention(
            dim,
            num_heads,
            window_size,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
            use_relative_position_bias=use_relative_position_bias,
        )
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = build_vision_norm(norm_type, dim, norm_eps)
        self.mlp = VisionMLP(
            dim, int(dim * mlp_ratio), dropout=dropout, bias=mlp_bias
        )
        self.drop_path2 = DropPath(drop_path)

    def _attention_mask(
        self,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        valid = torch.ones(batch, height, width, dtype=torch.bool, device=device)
        if padding_mask is not None:
            valid = padding_mask.reshape(batch, height, width)
        region = torch.zeros(1, height, width, dtype=torch.long, device=device)
        if self.shift_size:
            # Rolled tokens may only attend tokens from the same pre-shift window.
            row_ids = torch.arange(height, device=device) // self.window_size
            col_ids = torch.arange(width, device=device) // self.window_size
            region = row_ids[:, None] * (width + 1) + col_ids[None, :]
            region = region.unsqueeze(0)
            valid = torch.roll(
                valid, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            region = torch.roll(
                region, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        valid_windows, _ = _partition_windows(valid.unsqueeze(-1), self.window_size)
        valid_windows = valid_windows.squeeze(-1)
        region_windows, _ = _partition_windows(
            region.unsqueeze(-1), self.window_size
        )
        region_windows = region_windows.squeeze(-1)
        if region_windows.shape[0] != valid_windows.shape[0]:
            region_windows = region_windows.repeat(batch, 1)
        return (
            valid_windows[:, :, None]
            & valid_windows[:, None, :]
            & region_windows[:, :, None].eq(region_windows[:, None, :])
        )

    def forward(
        self,
        tokens: torch.Tensor,
        grid_size: tuple[int, int],
        padding_mask: torch.Tensor | None = None,
        *,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, height, width = validate_token_grid(tokens, grid_size, self.dim)
        shortcut = tokens
        x = self.norm1(tokens).reshape(batch, height, width, self.dim)
        mask = self._attention_mask(
            batch, height, width, x.device, padding_mask
        )
        if self.shift_size:
            x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        windows, metadata = _partition_windows(x, self.window_size)
        windows, weights = self.attention(
            windows, mask, output_attentions=output_attentions
        )
        x = _reverse_windows(windows, self.window_size, metadata)
        if self.shift_size:
            x = torch.roll(
                x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        x = shortcut + self.drop_path1(x.reshape(batch, -1, self.dim))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x, weights


class SwinPatchMerging(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        norm_type: str,
        norm_eps: float,
        bias: bool,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.norm = build_vision_norm(norm_type, 4 * input_dim, norm_eps)
        self.reduction = nn.Linear(4 * input_dim, output_dim, bias=bias)

    def forward(
        self,
        tokens: torch.Tensor,
        grid_size: tuple[int, int],
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int], torch.Tensor | None]:
        batch, height, width = validate_token_grid(
            tokens, grid_size, self.input_dim
        )
        x = tokens.reshape(batch, height, width, self.input_dim)
        pad_h, pad_w = height % 2, width % 2
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        x = torch.cat(
            (x[:, 0::2, 0::2], x[:, 0::2, 1::2],
             x[:, 1::2, 0::2], x[:, 1::2, 1::2]),
            dim=-1,
        )
        new_grid = (x.shape[1], x.shape[2])
        x = self.reduction(self.norm(x)).reshape(batch, -1, self.reduction.out_features)
        new_mask = None
        if padding_mask is not None:
            mask = padding_mask.reshape(batch, height, width)
            mask = F.pad(mask, (0, pad_w, 0, pad_h), value=False)
            new_mask = (
                mask[:, 0::2, 0::2] | mask[:, 0::2, 1::2]
                | mask[:, 1::2, 0::2] | mask[:, 1::2, 1::2]
            ).flatten(1)
        return x, new_grid, new_mask


@dataclass(frozen=True)
class SwinVisionConfig:
    image_size: int | tuple[int, int] = 224
    patch_size: int | tuple[int, int] = 14
    in_channels: int = 3
    embed_dims: tuple[int, ...] = (96, 192, 384)
    depths: tuple[int, ...] = (2, 2, 4)
    num_heads: tuple[int, ...] = (6, 12, 12)
    window_size: int = 4
    mlp_ratio: float = 4.0
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6
    patch_bias: bool = False
    qkv_bias: bool = False
    proj_bias: bool = False
    mlp_bias: bool = False
    merge_bias: bool = False
    dropout: float = 0.0
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.0
    use_relative_position_bias: bool = True
    initializer_std: float = 0.02

    def __post_init__(self):
        image, patch = to_2tuple(self.image_size, "image_size"), to_2tuple(
            self.patch_size, "patch_size"
        )
        if image[0] % patch[0] or image[1] % patch[1]:
            raise ValueError("configured image_size must be divisible by patch_size")
        stages = len(self.embed_dims)
        if stages == 0 or len(self.depths) != stages or len(self.num_heads) != stages:
            raise ValueError("stage sequences must have equal nonzero length")
        if self.in_channels <= 0 or self.window_size <= 0:
            raise ValueError("in_channels and window_size must be > 0")
        if any(value <= 0 for value in self.embed_dims + self.depths + self.num_heads):
            raise ValueError("all stage values must be > 0")
        if any(dim % heads for dim, heads in zip(self.embed_dims, self.num_heads)):
            raise ValueError("every embed_dim must be divisible by num_heads")
        for name in ("dropout", "attention_dropout", "drop_path_rate"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")


class SwinMoonViTEncoder(nn.Module):
    def __init__(self, config: SwinVisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedding = VisionPatchEmbedding(
            config.in_channels,
            config.embed_dims[0],
            config.patch_size,
            bias=config.patch_bias,
        )
        rates = torch.linspace(
            0, config.drop_path_rate, sum(config.depths)
        ).tolist()
        cursor = 0
        self.stages = nn.ModuleList()
        for dim, depth, heads in zip(
            config.embed_dims, config.depths, config.num_heads
        ):
            blocks = nn.ModuleList(
                SwinTransformerBlock(
                    dim,
                    heads,
                    config.window_size,
                    0 if index % 2 == 0 else config.window_size // 2,
                    mlp_ratio=config.mlp_ratio,
                    norm_type=config.norm_type,
                    norm_eps=config.norm_eps,
                    qkv_bias=config.qkv_bias,
                    proj_bias=config.proj_bias,
                    mlp_bias=config.mlp_bias,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    drop_path=rates[cursor + index],
                    use_relative_position_bias=config.use_relative_position_bias,
                )
                for index in range(depth)
            )
            cursor += depth
            self.stages.append(blocks)
        self.mergers = nn.ModuleList(
            SwinPatchMerging(
                config.embed_dims[index],
                config.embed_dims[index + 1],
                norm_type=config.norm_type,
                norm_eps=config.norm_eps,
                bias=config.merge_bias,
            )
            for index in range(len(config.embed_dims) - 1)
        )
        self.final_norm = build_vision_norm(
            config.norm_type, config.embed_dims[-1], config.norm_eps
        )
        self.apply(
            lambda module: initialize_vision_module(module, config.initializer_std)
        )

    def forward(
        self,
        images: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> VisionEncoderOutput:
        x, grid = self.patch_embedding(images)
        if padding_mask is not None:
            if padding_mask.shape != x.shape[:2] or padding_mask.dtype != torch.bool:
                raise ValueError("padding_mask must be boolean and match patch tokens")
        hidden_states = [x] if output_hidden_states else None
        attentions = [] if output_attentions else None
        for stage_index, blocks in enumerate(self.stages):
            for block in blocks:
                x, weights = block(
                    x,
                    grid,
                    padding_mask,
                    output_attentions=output_attentions,
                )
                if hidden_states is not None:
                    hidden_states.append(x)
                if attentions is not None:
                    attentions.append(weights)
            if stage_index < len(self.mergers):
                x, grid, padding_mask = self.mergers[stage_index](
                    x, grid, padding_mask
                )
                if hidden_states is not None:
                    hidden_states.append(x)
        x = self.final_norm(x)
        if hidden_states is not None:
            hidden_states[-1] = x
        return VisionEncoderOutput(
            last_hidden_state=x,
            grid_size=grid,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
            attentions=tuple(attentions) if attentions is not None else None,
        )
