"""Multi-token prediction components used as an optional KimiK3 output head."""

from __future__ import annotations

import weakref

import torch
import torch.nn as nn

from src.hybrid_backbone import HybridBackboneCache
from src.loss import MultiTokenPredictionLoss

from .alignment import (
    MTPTrainingView,
    build_mtp_feature_mask,
    build_mtp_training_view,
)
from .block import KimiMTPBlock
from .config import KimiMTPConfig
from .fusion import KimiMTPFusion
from .outputs import (
    KimiMTPOutput,
    MTPDiagnostics,
    MTPDraftOutput,
)


class KimiMTPHead(nn.Module):
    """Teacher-forced x[t+2] head sharing the model embedding and LM head."""

    def __init__(
        self,
        config: KimiMTPConfig,
        input_embeddings: nn.Embedding,
        lm_head: nn.Linear,
        *,
        register_shared_modules: bool = True,
    ):
        super().__init__()
        self.config = config
        if (
            input_embeddings.embedding_dim != config.d_model
            or input_embeddings.num_embeddings != config.vocab_size
        ):
            raise ValueError("input_embeddings dimensions do not match config")
        if (
            lm_head.in_features != config.d_model
            or lm_head.out_features != config.vocab_size
        ):
            raise ValueError("lm_head dimensions do not match config")
        self._register_shared_modules = register_shared_modules
        if register_shared_modules:
            self._registered_input_embeddings = input_embeddings
            self._registered_lm_head = lm_head
        else:
            object.__setattr__(
                self,
                "_input_embeddings_ref",
                weakref.ref(input_embeddings),
            )
            object.__setattr__(self, "_lm_head_ref", weakref.ref(lm_head))
        self.fusion = (
            KimiMTPFusion(
                config.d_model,
                eps=config.rms_norm_eps,
                init_std=config.init_std,
            )
            if config.enabled
            else None
        )
        self.block = KimiMTPBlock(config) if config.enabled else None
        self.loss_fn = MultiTokenPredictionLoss(
            ignore_index=config.ignore_index,
            zero_valid_policy="connected_zero",
        )

    @property
    def input_embeddings(self) -> nn.Embedding:
        if self._register_shared_modules:
            return self._registered_input_embeddings
        module = self._input_embeddings_ref()
        if module is None:
            raise RuntimeError("shared input embeddings are no longer alive")
        return module

    @property
    def lm_head(self) -> nn.Linear:
        if self._register_shared_modules:
            return self._registered_lm_head
        module = self._lm_head_ref()
        if module is None:
            raise RuntimeError("shared LM head is no longer alive")
        return module

    def set_shared_modules(
        self,
        input_embeddings: nn.Embedding,
        lm_head: nn.Linear,
    ) -> None:
        if self._register_shared_modules:
            self._registered_input_embeddings = input_embeddings
            self._registered_lm_head = lm_head
        else:
            object.__setattr__(
                self,
                "_input_embeddings_ref",
                weakref.ref(input_embeddings),
            )
            object.__setattr__(self, "_lm_head_ref", weakref.ref(lm_head))

    def get_input_embeddings(self) -> nn.Embedding:
        return self.input_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def _validate_no_packed_segments(
        self,
        segment_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> None:
        if segment_ids is None:
            return
        mask = (
            torch.ones_like(segment_ids, dtype=torch.bool)
            if attention_mask is None
            else attention_mask
        )
        for row_segments, row_mask in zip(segment_ids, mask):
            if torch.unique(row_segments[row_mask]).numel() > 1:
                raise ValueError(
                    "packed segment execution is not supported by the current "
                    "KDA/MLA cache; refusing cross-segment attention leakage"
                )

    def _run_compacted(
        self,
        fused: torch.Tensor,
        feature_mask: torch.Tensor,
        *,
        output_hidden_states: bool,
        output_diagnostics: bool,
        update_routing_bias: bool,
    ) -> tuple[torch.Tensor, dict[str, object] | None]:
        batch, tokens, width = fused.shape
        result = fused.new_zeros(batch, tokens, width)
        lengths = feature_mask.sum(dim=1)
        active_rows = torch.nonzero(lengths > 0, as_tuple=False).flatten()
        if active_rows.numel() == 0:
            return result, None
        max_tokens = int(lengths[active_rows].max().item())
        compact = fused.new_zeros(active_rows.numel(), max_tokens, width)
        compact_mask = torch.zeros(
            active_rows.numel(),
            max_tokens,
            dtype=torch.bool,
            device=fused.device,
        )
        positions: list[torch.Tensor] = []
        for compact_row, source_row in enumerate(active_rows.tolist()):
            index = torch.nonzero(
                feature_mask[source_row], as_tuple=False
            ).flatten()
            positions.append(index)
            length = index.numel()
            compact[compact_row, :length] = fused[source_row, index]
            compact_mask[compact_row, :length] = True
        block_output = self.block(
            compact,
            attention_mask=compact_mask,
            output_hidden_states=output_hidden_states,
            output_diagnostics=output_diagnostics,
            update_routing_bias=update_routing_bias,
        )
        for compact_row, source_row in enumerate(active_rows.tolist()):
            index = positions[compact_row]
            result[source_row, index] = block_output.last_hidden_state[
                compact_row, : index.numel()
            ]
        return result, block_output.diagnostics

    @staticmethod
    def _diagnostics(
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        view: MTPTrainingView,
        block_diagnostics: dict[str, object] | None,
    ) -> MTPDiagnostics:
        valid = view.valid_mask
        count = valid.sum().detach()
        if torch.any(valid):
            selected_logits = logits[valid].float()
            probabilities = selected_logits.softmax(dim=-1)
            entropy = -(
                probabilities
                * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
            ).sum(dim=-1).mean()
            accuracy = (
                selected_logits.argmax(dim=-1)
                == view.target_ids[valid]
            ).float().mean()
            mean_norm = hidden_states.float().norm(dim=-1)[valid].mean()
        else:
            zero = logits.detach().new_zeros((), dtype=torch.float32)
            entropy = zero
            accuracy = zero
            mean_norm = zero
        return MTPDiagnostics(
            valid_token_count=count,
            token_accuracy=accuracy.detach(),
            mean_logit_entropy=entropy.detach(),
            mean_hidden_norm=mean_norm.detach(),
            block=block_diagnostics,
        )

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        compute_loss: bool = True,
        segment_ids: torch.Tensor | None = None,
        return_logits: bool | None = None,
        return_hidden_states: bool = False,
        return_diagnostics: bool = False,
        update_routing_bias: bool = False,
    ) -> KimiMTPOutput:
        view = build_mtp_training_view(
            last_hidden_state,
            input_ids,
            attention_mask,
            labels,
            segment_ids,
            ignore_index=self.config.ignore_index,
        )
        if not self.config.enabled:
            return KimiMTPOutput(None, None, None, view, None)
        self._validate_no_packed_segments(segment_ids, attention_mask)
        source_hidden = view.source_hidden
        if self.config.detach_backbone_hidden:
            source_hidden = source_hidden.detach()
        future_embeddings = self.input_embeddings(view.future_input_ids)
        fused = self.fusion(source_hidden, future_embeddings)
        feature_mask = build_mtp_feature_mask(attention_mask, input_ids)
        hidden_states, block_diagnostics = self._run_compacted(
            fused,
            feature_mask,
            output_hidden_states=return_hidden_states,
            output_diagnostics=return_diagnostics,
            update_routing_bias=update_routing_bias,
        )
        logits = self.lm_head(hidden_states)
        loss = (
            self.loss_fn(
                logits,
                view.target_ids,
                mtp_loss_mask=view.valid_mask,
                future_offsets=(self.config.future_offset,),
            ).loss
            if labels is not None and compute_loss
            else None
        )
        diagnostics = (
            self._diagnostics(
                logits, hidden_states, view, block_diagnostics
            )
            if return_diagnostics
            else None
        )
        keep_logits = (
            self.config.return_logits_by_default
            if return_logits is None
            else return_logits
        )
        return KimiMTPOutput(
            logits=logits if keep_logits else None,
            loss=loss,
            hidden_states=hidden_states if return_hidden_states else None,
            training_view=view,
            diagnostics=diagnostics,
        )

    def draft_one_step(
        self,
        source_hidden_t: torch.Tensor,
        intermediate_token_id: torch.Tensor,
        cache: HybridBackboneCache | None = None,
    ) -> MTPDraftOutput:
        if not self.config.enabled:
            raise RuntimeError("MTP is disabled")
        if self.training:
            raise RuntimeError("draft_one_step is an eval-only API")
        if (
            intermediate_token_id.ndim != 2
            or intermediate_token_id.shape[1] != 1
        ):
            raise ValueError("intermediate_token_id must have shape [B,1]")
        if intermediate_token_id.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("intermediate_token_id must use an integer dtype")
        if source_hidden_t.shape != (
            intermediate_token_id.shape[0],
            1,
            self.config.d_model,
        ):
            raise ValueError("source_hidden_t must have shape [B,1,D]")
        if source_hidden_t.device != intermediate_token_id.device:
            raise ValueError("draft inputs must share device")
        future_embedding = self.input_embeddings(intermediate_token_id)
        fused = self.fusion(source_hidden_t, future_embedding)
        output = self.block(
            fused,
            cache=cache,
            use_cache=True,
            mode="prefill" if cache is None else "decode",
        )
        hidden_states = output.last_hidden_state
        return MTPDraftOutput(
            logits=self.lm_head(hidden_states),
            cache=output.cache,
            hidden_states=hidden_states,
        )
