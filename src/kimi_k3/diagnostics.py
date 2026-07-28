from __future__ import annotations

from .outputs import ParameterReport


def build_parameter_report(model) -> ParameterReport:
    def unique(*modules):
        result = {}
        for module in modules:
            if module is not None:
                for parameter in module.parameters():
                    result[id(parameter)] = parameter
        return result

    all_parameters = unique(model)
    embeddings = unique(model.embed_tokens)
    vision = unique(model.vision_encoder, model.vision_projector)
    backbone = unique(model.backbone)
    lm_head = unique(model.lm_head)
    mtp = unique(model.mtp)
    lm_unique = set(lm_head) - set(embeddings)
    return ParameterReport(
        total=sum(p.numel() for p in all_parameters.values()),
        trainable=sum(
            p.numel() for p in all_parameters.values() if p.requires_grad
        ),
        embeddings=sum(p.numel() for p in embeddings.values()),
        vision=sum(p.numel() for p in vision.values()),
        backbone=sum(p.numel() for p in backbone.values()),
        lm_head_unique=sum(lm_head[key].numel() for key in lm_unique),
        mtp=sum(p.numel() for p in mtp.values()),
    )
