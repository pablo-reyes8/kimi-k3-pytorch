"""Configuration and multimodal integration helpers used by the KimiK3 orchestrator."""

from __future__ import annotations

import torch
import torch.nn as nn

from .outputs import MultimodalMetadata


class VisualPlaceholderComposer(nn.Module):
    """Replace reserved token embeddings with projected visual tokens."""

    def __init__(
        self,
        d_model: int,
        image_token_id: int,
        video_token_id: int | None,
    ):
        super().__init__()
        self.d_model = d_model
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

    @staticmethod
    def resolve_counts(
        name: str,
        counts: torch.Tensor | None,
        batch_size: int,
        item_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if counts is None:
            if item_count != batch_size:
                raise ValueError(
                    f"{name} is required for {item_count} visual items and "
                    f"text batch size {batch_size}"
                )
            return torch.ones(batch_size, dtype=torch.long, device=device)
        if counts.shape != (batch_size,):
            raise ValueError(f"{name} must have shape {(batch_size,)}")
        if counts.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must use an integer dtype")
        if counts.device != device:
            raise ValueError(f"{name} must share device with the text input")
        if torch.any(counts < 0) or int(counts.sum().item()) != item_count:
            raise ValueError(
                f"{name} must be non-negative and sum to {item_count}"
            )
        return counts

    def _replace(
        self,
        composed: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_id: int,
        projected: torch.Tensor | None,
        projected_mask: torch.Tensor | None,
        counts: torch.Tensor | None,
        modality: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        placeholders = input_ids.eq(token_id)
        if torch.any(placeholders & ~attention_mask):
            raise ValueError(
                f"{modality} placeholder cannot occur in masked padding"
            )
        if projected is None:
            if torch.any(placeholders):
                raise ValueError(
                    f"{modality} placeholders require matching visual inputs"
                )
            return None, None
        if projected.ndim != 3 or projected.shape[-1] != self.d_model:
            raise ValueError(
                f"{modality} projection must have shape [M,N,{self.d_model}]"
            )
        if projected_mask.shape != projected.shape[:2]:
            raise ValueError(f"{modality} projected mask has an invalid shape")
        if counts is None:
            raise ValueError(f"{modality}_counts must be resolved before composition")
        item_cursor = 0
        positions = []
        tokens_per_sample = []
        for batch_index, num_items in enumerate(counts.tolist()):
            items = []
            for _ in range(num_items):
                items.append(projected[item_cursor][projected_mask[item_cursor]])
                item_cursor += 1
            visual_tokens = (
                torch.cat(items, dim=0)
                if items
                else projected.new_empty(0, self.d_model)
            )
            sample_positions = torch.nonzero(
                placeholders[batch_index], as_tuple=False
            ).flatten()
            if sample_positions.numel() != visual_tokens.shape[0]:
                raise ValueError(
                    f"sample {batch_index} has {sample_positions.numel()} "
                    f"{modality} placeholders but {visual_tokens.shape[0]} "
                    "projected valid visual tokens"
                )
            if sample_positions.numel():
                composed[batch_index, sample_positions] = visual_tokens
                positions.append(
                    torch.stack(
                        (
                            torch.full_like(sample_positions, batch_index),
                            sample_positions,
                        ),
                        dim=-1,
                    )
                )
            tokens_per_sample.append(visual_tokens.shape[0])
        all_positions = (
            torch.cat(positions)
            if positions
            else torch.empty(
                0, 2, dtype=torch.long, device=input_ids.device
            )
        )
        return (
            all_positions,
            torch.tensor(
                tokens_per_sample,
                dtype=torch.long,
                device=input_ids.device,
            ),
        )

    def forward(
        self,
        text_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        image_embeddings: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        image_counts: torch.Tensor | None = None,
        video_embeddings: torch.Tensor | None = None,
        video_mask: torch.Tensor | None = None,
        video_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MultimodalMetadata]:
        composed = text_embeddings.clone()
        image_positions, image_token_counts = self._replace(
            composed,
            input_ids,
            attention_mask,
            self.image_token_id,
            image_embeddings,
            image_mask,
            image_counts,
            "image",
        )
        video_positions = None
        video_token_counts = None
        if self.video_token_id is not None:
            video_positions, video_token_counts = self._replace(
                composed,
                input_ids,
                attention_mask,
                self.video_token_id,
                video_embeddings,
                video_mask,
                video_counts,
                "video",
            )
        elif video_embeddings is not None:
            raise ValueError("video_token_id is required for video inputs")
        return composed, MultimodalMetadata(
            image_counts=image_counts,
            video_counts=video_counts,
            image_token_counts=image_token_counts,
            video_token_counts=video_token_counts,
            image_positions=image_positions,
            video_positions=video_positions,
        )
