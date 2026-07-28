from __future__ import annotations

import torch
import torch.nn as nn

from src.vision import (
    HierarchicalMoonViTEncoder,
    HierarchicalVisionConfig,
    MoonViTEncoder,
    SwinMoonViTEncoder,
    SwinVisionConfig,
    VisionEncoderConfig,
    VisionEncoderOutput,
)

from .config import KimiK3Config, VisionConfig
from .multimodal_composer import VisualPlaceholderComposer
from .outputs import (
    KimiK3VisionOutput,
    MultimodalMetadata,
)


def make_vision_encoder(config: VisionConfig) -> nn.Module:
    if isinstance(config, VisionEncoderConfig):
        return MoonViTEncoder(config)
    if isinstance(config, HierarchicalVisionConfig):
        return HierarchicalMoonViTEncoder(config)
    if isinstance(config, SwinVisionConfig):
        return SwinMoonViTEncoder(config)
    raise TypeError("unsupported vision configuration")


def pack_visual_mask(
    mask: torch.Tensor,
    grid: tuple[int, int],
) -> torch.Tensor:
    batch, tokens = mask.shape
    height, width = grid
    if tokens != height * width or height % 2 or width % 2:
        raise ValueError("visual mask is incompatible with pixel shuffle")
    spatial = mask.reshape(batch, height, width)
    return (
        spatial[:, 0::2, 0::2]
        & spatial[:, 0::2, 1::2]
        & spatial[:, 1::2, 0::2]
        & spatial[:, 1::2, 1::2]
    ).flatten(1)


def encode_visual_items(
    *,
    values: torch.Tensor,
    vision_mask: torch.Tensor | None,
    is_video: bool,
    vision_encoder: nn.Module,
    vision_token_packer: nn.Module | None,
    vision_projector: nn.Module,
    output_hidden_states: bool,
    output_attentions: bool,
) -> tuple[torch.Tensor, torch.Tensor, VisionEncoderOutput, int]:
    if is_video:
        if values.ndim != 5:
            raise ValueError("video_values must have shape [M,F,C,H,W]")
        item_count, frames = values.shape[:2]
        encoder_values = values.flatten(0, 1)
        encoder_mask = (
            None if vision_mask is None else vision_mask.flatten(0, 1)
        )
    else:
        if values.ndim != 4:
            raise ValueError("pixel_values must have shape [M,C,H,W]")
        item_count, frames = values.shape[0], 1
        encoder_values = values
        encoder_mask = vision_mask
    encoded = vision_encoder(
        encoder_values,
        encoder_mask,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
    )
    visual_tokens = encoded.last_hidden_state
    visual_mask = (
        torch.ones(
            visual_tokens.shape[:2],
            dtype=torch.bool,
            device=visual_tokens.device,
        )
        if encoder_mask is None
        else encoder_mask
    )
    if vision_token_packer is not None:
        visual_mask = pack_visual_mask(visual_mask, encoded.grid_size)
        visual_tokens = vision_token_packer(
            visual_tokens, encoded.grid_size
        ).last_hidden_state
    projected = vision_projector(visual_tokens)
    if is_video:
        projected = projected.reshape(
            item_count,
            frames * projected.shape[1],
            projected.shape[2],
        )
        visual_mask = visual_mask.reshape(item_count, -1)
    return projected, visual_mask, encoded, item_count


def prepare_multimodal_embeddings(
    *,
    config: KimiK3Config,
    input_ids: torch.Tensor,
    text_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor | None,
    video_values: torch.Tensor | None,
    vision_attention_mask: torch.Tensor | None,
    image_counts: torch.Tensor | None,
    video_counts: torch.Tensor | None,
    vision_encoder: nn.Module | None,
    vision_token_packer: nn.Module | None,
    vision_projector: nn.Module | None,
    composer: VisualPlaceholderComposer | None,
    output_hidden_states: bool,
    output_attentions: bool,
) -> tuple[
    torch.Tensor,
    KimiK3VisionOutput | None,
    MultimodalMetadata | None,
]:
    has_visual_input = pixel_values is not None or video_values is not None
    has_image_placeholder = (
        config.image_token_id is not None
        and torch.any(input_ids.eq(config.image_token_id))
    )
    has_video_placeholder = (
        config.video_token_id is not None
        and torch.any(input_ids.eq(config.video_token_id))
    )
    if not has_visual_input and not has_image_placeholder and not has_video_placeholder:
        return text_embeddings, None, None
    if not config.enable_vision:
        raise ValueError("visual inputs cannot be used when vision is disabled")
    if (
        pixel_values is not None
        and video_values is not None
        and vision_attention_mask is not None
    ):
        raise ValueError(
            "vision_attention_mask is ambiguous for mixed image/video input"
        )

    image_embeddings = image_mask = image_output = None
    video_embeddings = video_mask = video_output = None
    if pixel_values is not None:
        (
            image_embeddings,
            image_mask,
            image_output,
            image_items,
        ) = encode_visual_items(
            values=pixel_values,
            vision_mask=vision_attention_mask,
            is_video=False,
            vision_encoder=vision_encoder,
            vision_token_packer=vision_token_packer,
            vision_projector=vision_projector,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        image_counts = composer.resolve_counts(
            "image_counts",
            image_counts,
            input_ids.shape[0],
            image_items,
            input_ids.device,
        )
    if video_values is not None:
        (
            video_embeddings,
            video_mask,
            video_output,
            video_items,
        ) = encode_visual_items(
            values=video_values,
            vision_mask=vision_attention_mask,
            is_video=True,
            vision_encoder=vision_encoder,
            vision_token_packer=vision_token_packer,
            vision_projector=vision_projector,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        video_counts = composer.resolve_counts(
            "video_counts",
            video_counts,
            input_ids.shape[0],
            video_items,
            input_ids.device,
        )
    composed, metadata = composer(
        text_embeddings,
        input_ids,
        attention_mask,
        image_embeddings=image_embeddings,
        image_mask=image_mask,
        image_counts=image_counts,
        video_embeddings=video_embeddings,
        video_mask=video_mask,
        video_counts=video_counts,
    )
    return (
        composed,
        KimiK3VisionOutput(image_output, video_output),
        metadata,
    )
