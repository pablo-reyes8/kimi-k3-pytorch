"""The single public master class for the Kimi K3 architecture.

Configuration, typed outputs, visual composition and validation live in
``src.kimi_k3``. This file contains only the high-level ``KimiK3`` module so
the complete forward can be audited in one place.

Phase 10 closes the architecture and returns vocabulary logits. Training
objectives and losses are intentionally deferred to the next phase.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .hybrid_backbone import HybridBackboneCache
from .kimi_block import KimiBlock
from .kimi_k3.config import KimiK3Config
from .kimi_k3.diagnostics import build_parameter_report
from .kimi_k3.multimodal_composer import VisualPlaceholderComposer
from .kimi_k3.outputs import KimiK3Output, ParameterReport
from .kimi_k3.validation import (
    resolve_execution_mode,
    validate_and_build_attention_mask,
)
from .kimi_k3.vision_integration import (
    make_vision_encoder,
    prepare_multimodal_embeddings,
)
from .mtp import KimiMTPHead
from .vision import SpatialTokenPixelShuffle, VisionProjector


class KimiK3(nn.Module):
    """Orchestrate the complete research-scale Kimi K3 forward.

    The class owns every trainable high-level path exactly once:

    ``text embedding``
        Token identity only. KDA and Gated MLA are NoPE.

    ``MoonViT -> pixel packing -> projector``
        Optional images/videos are converted into ``d_model`` visual tokens
        and replace reserved placeholders without changing sequence length.

    ``KimiBlock``
        The shared sequence contains 3 KDA + 1 Gated MLA per hybrid group,
        one final global Gated MLA, a Stable LatentMoE after every attention,
        Attention Residuals across depth and one final RMSNorm.

    ``LM head``
        Projects the already-normalized final state to vocabulary scores.

    ``MTP``
        Optional, full-sequence-only auxiliary logits. It never feeds back
        into the main logits and is never run during cached generation.

    No internal attention, routing, depth-mixing or vision mathematics is
    reimplemented here; this class only validates, orders and connects the
    tested components.
    """

    config_class = KimiK3Config

    def __init__(self, config: KimiK3Config):
        super().__init__()
        self.config = config

        # Shared text/multimodal path.
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.backbone = KimiBlock(config.backbone)
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=config.use_bias_in_lm_head,
        )

        # Native visual path. Disabled capabilities instantiate no modules.
        if config.enable_vision:
            self.vision_encoder = make_vision_encoder(config.vision)
            self.vision_token_packer = (
                SpatialTokenPixelShuffle()
                if config.vision_use_pixel_shuffle
                else None
            )
            projector = config.vision_projector
            self.vision_projector = VisionProjector(
                projector.input_dim,
                projector.hidden_dim,
                projector.output_dim,
                activation=projector.activation,
                bias=projector.bias,
            )
            self.multimodal_composer = VisualPlaceholderComposer(
                config.d_model,
                config.image_token_id,
                config.video_token_id,
            )
        else:
            self.vision_encoder = None
            self.vision_token_packer = None
            self.vision_projector = None
            self.multimodal_composer = None

        # Only generic top-level parameters are initialized here. Specialized
        # KDA, AttnRes, MoE, MoonViT and MTP initializers remain untouched.
        nn.init.normal_(
            self.embed_tokens.weight,
            std=config.initializer_range,
        )
        if config.pad_token_id is not None:
            with torch.no_grad():
                self.embed_tokens.weight[config.pad_token_id].zero_()
        nn.init.normal_(self.lm_head.weight, std=config.initializer_range)
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)
        self.tie_weights()

        # MTP owns its auxiliary block/fusion, but references the main
        # embedding and LM head without registering either module twice.
        self.mtp = (
            KimiMTPHead(
                config.mtp,
                self.embed_tokens,
                self.lm_head,
                register_shared_modules=False,
            )
            if config.enable_mtp
            else None
        )
        if config.freeze_vision_encoder and self.vision_encoder is not None:
            self.vision_encoder.requires_grad_(False)

    # ------------------------------------------------------------------
    # Stable public module access and weight-tying helpers
    # ------------------------------------------------------------------
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        if (
            value.num_embeddings != self.config.vocab_size
            or value.embedding_dim != self.config.d_model
        ):
            raise ValueError("replacement input embeddings have invalid shape")
        self.embed_tokens = value
        if self.mtp is not None:
            self.mtp.set_shared_modules(self.embed_tokens, self.lm_head)

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        if (
            value.in_features != self.config.d_model
            or value.out_features != self.config.vocab_size
        ):
            raise ValueError("replacement LM head has invalid shape")
        self.lm_head = value
        if self.mtp is not None:
            self.mtp.set_shared_modules(self.embed_tokens, self.lm_head)

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        if hasattr(self, "mtp") and self.mtp is not None:
            self.mtp.set_shared_modules(self.embed_tokens, self.lm_head)

    def get_vision_encoder(self) -> nn.Module | None:
        return self.vision_encoder

    def get_backbone(self) -> KimiBlock:
        return self.backbone

    def get_mtp_head(self) -> KimiMTPHead | None:
        return self.mtp

    def num_parameters(self, only_trainable: bool = False) -> int:
        unique = {
            id(parameter): parameter for parameter in self.parameters()
        }.values()
        return sum(
            parameter.numel()
            for parameter in unique
            if not only_trainable or parameter.requires_grad
        )

    def parameter_report(self) -> ParameterReport:
        return build_parameter_report(self)

    # ------------------------------------------------------------------
    # Master forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        video_values: torch.Tensor | None = None,
        vision_attention_mask: torch.Tensor | None = None,
        image_counts: torch.Tensor | None = None,
        video_counts: torch.Tensor | None = None,
        cache: HybridBackboneCache | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        use_mtp: bool = False,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        output_router_diagnostics: bool | None = None,
        output_attnres_diagnostics: bool | None = None,
        return_dict: bool | None = None,
    ) -> KimiK3Output | tuple:
        """Run raw text/vision inputs through every Kimi K3 component.

        Execution is intentionally linear and visible:

        1. validate text inputs and select full/prefill/decode mode;
        2. create token embeddings;
        3. optionally encode/project vision and replace placeholders;
        4. run the shared KDA/MLA + AttnRes + Stable LatentMoE backbone;
        5. project the final normalized hidden state to vocabulary logits;
        6. optionally run the independent full-sequence MTP branch.

        This phase returns scores and state only. It computes no loss.
        """
        if position_ids is not None:
            raise ValueError(
                "position_ids are unsupported because Kimi K3 uses NoPE"
            )
        if cache_position is not None:
            raise ValueError("cache_position is owned by HybridBackboneCache")
        _, tokens, attention_mask = validate_and_build_attention_mask(
            input_ids,
            inputs_embeds,
            attention_mask,
            d_model=self.config.d_model,
            pad_token_id=self.config.pad_token_id,
        )
        mode = resolve_execution_mode(cache, use_cache, tokens)
        if mode == "decode" and (
            pixel_values is not None or video_values is not None
        ):
            raise ValueError(
                "visual inputs are consumed during prefill, not decode"
            )
        if inputs_embeds is not None and (
            pixel_values is not None or video_values is not None
        ):
            raise ValueError(
                "inputs_embeds cannot be combined with visual inputs without "
                "explicit placeholder positions"
            )
        if use_mtp and self.mtp is None:
            raise ValueError("MTP was requested but is disabled")
        if use_mtp and mode != "full":
            raise ValueError(
                "MTP is a full-sequence branch and cannot use cache"
            )
        if use_mtp and input_ids is None:
            raise ValueError("MTP requires input_ids")

        hidden_flag = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        attention_flag = (
            self.config.output_attentions
            if output_attentions is None
            else output_attentions
        )
        router_flag = (
            self.config.output_router_diagnostics
            if output_router_diagnostics is None
            else output_router_diagnostics
        )
        attnres_flag = (
            self.config.output_attnres_diagnostics
            if output_attnres_diagnostics is None
            else output_attnres_diagnostics
        )

        # 1) Text identity enters the shared embedding space.
        shared_embeddings = (
            self.embed_tokens(input_ids)
            if input_ids is not None
            else inputs_embeds
        )

        # 2) Visual inputs are encoded once and replace reserved embeddings.
        vision_outputs = None
        multimodal_metadata = None
        if input_ids is not None:
            (
                shared_embeddings,
                vision_outputs,
                multimodal_metadata,
            ) = prepare_multimodal_embeddings(
                config=self.config,
                input_ids=input_ids,
                text_embeddings=shared_embeddings,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                video_values=video_values,
                vision_attention_mask=vision_attention_mask,
                image_counts=image_counts,
                video_counts=video_counts,
                vision_encoder=self.vision_encoder,
                vision_token_packer=self.vision_token_packer,
                vision_projector=self.vision_projector,
                composer=self.multimodal_composer,
                output_hidden_states=hidden_flag,
                output_attentions=attention_flag,
            )

        # 3) KimiBlock owns all sequence/channel/depth mixing and final norm.
        backbone_output = self.backbone(
            shared_embeddings,
            attention_mask=attention_mask,
            cache=cache,
            use_cache=use_cache,
            mode=mode,
            output_hidden_states=hidden_flag,
            output_depth_weights=attnres_flag,
            output_diagnostics=router_flag or attnres_flag,
        )
        final_hidden_state = backbone_output.last_hidden_state

        # 4) The sole main vocabulary projection consumes the normalized state.
        logits = self.lm_head(final_hidden_state)

        # 5) MTP is auxiliary: it can produce logits but cannot alter `logits`.
        mtp_output = (
            self.mtp(
                final_hidden_state,
                input_ids,
                attention_mask=attention_mask,
                return_logits=True,
                return_diagnostics=router_flag or attnres_flag,
            )
            if use_mtp
            else None
        )

        output = KimiK3Output(
            logits=logits,
            last_hidden_state=final_hidden_state,
            cache=backbone_output.cache,
            mtp_logits=None if mtp_output is None else mtp_output.logits,
            hidden_states=backbone_output.hidden_states,
            backbone_diagnostics=backbone_output.diagnostics,
            attnres_diagnostics=(
                backbone_output.depth_outputs if attnres_flag else None
            ),
            mtp_diagnostics=(
                None if mtp_output is None else mtp_output.diagnostics
            ),
            vision_outputs=vision_outputs,
            multimodal_metadata=multimodal_metadata,
        )
        use_dict = (
            self.config.return_dict if return_dict is None else return_dict
        )
        return output if use_dict else output.to_tuple()

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> KimiK3Output | tuple:
        """Use the same forward to encode a prompt and create its cache."""

        return self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **kwargs,
        )

    @torch.no_grad()
    def decode_step(
        self,
        input_ids: torch.Tensor,
        cache: HybridBackboneCache,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> KimiK3Output | tuple:
        """Use the same forward for one cached token; vision is not re-run."""

        if cache is None:
            raise ValueError("decode_step requires a non-empty cache")
        return self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache=cache,
            use_cache=True,
            **kwargs,
        )


__all__ = ["KimiK3"]
