import os
import re

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


def _unwrap_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "net", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _strip_prefixes_once(state_dict, prefixes):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        cleaned[new_key] = value
    return cleaned


def _add_prefix_if_missing(state_dict, prefix):
    remapped = {}
    for key, value in state_dict.items():
        new_key = key if key.startswith(prefix) else prefix + key
        remapped[new_key] = value
    return remapped


def _remap_feature_list_stage_keys(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        new_key = re.sub(r"^stages\.(\d+)\.", r"stages_\1.", key)
        remapped[new_key] = value
    return remapped


def _score_state_dict(candidate_state_dict, model_state_dict):
    matched = 0
    shape_matched = 0
    for key, value in candidate_state_dict.items():
        model_value = model_state_dict.get(key)
        if model_value is None:
            continue
        matched += 1
        if tuple(model_value.shape) == tuple(value.shape):
            shape_matched += 1
    return shape_matched, matched


def _maybe_filter_pvt_checkpoint(state_dict, target_model, model_name):
    if not str(model_name).startswith("pvt_v2"):
        return state_dict
    try:
        from timm.models.pvt_v2 import checkpoint_filter_fn

        filtered = checkpoint_filter_fn(state_dict, target_model)
        if isinstance(filtered, dict):
            return filtered
    except Exception:
        pass
    return state_dict


def _make_state_dict_candidates(raw_state_dict, target_model, model_name):
    base_candidates = {
        "raw": raw_state_dict,
        "strip_common": _strip_prefixes_once(raw_state_dict, ("module.", "backbone.", "encoder.")),
        "strip_common_and_model": _strip_prefixes_once(raw_state_dict, ("module.", "backbone.", "encoder.", "model.")),
    }

    candidates = {}
    for name, state_dict in base_candidates.items():
        candidates[name] = state_dict
        candidates[f"{name}+model_prefix"] = _add_prefix_if_missing(state_dict, "model.")
        candidates[f"{name}+feature_list_stage"] = _remap_feature_list_stage_keys(state_dict)
        filtered_state_dict = _maybe_filter_pvt_checkpoint(state_dict, target_model, model_name)
        candidates[f"{name}+filter"] = filtered_state_dict
        candidates[f"{name}+filter+model_prefix"] = _add_prefix_if_missing(filtered_state_dict, "model.")
        candidates[f"{name}+filter+feature_list_stage"] = _remap_feature_list_stage_keys(filtered_state_dict)

    return candidates


def _prepare_best_pretrained_state_dict(backbone, weight_path, model_name):
    if not weight_path:
        raise FileNotFoundError("No local pretrained path was provided for the encoder.")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Local pretrained encoder weights not found: {weight_path}")

    checkpoint = torch.load(weight_path, map_location="cpu")
    raw_state_dict = _unwrap_checkpoint_state_dict(checkpoint)
    if not isinstance(raw_state_dict, dict):
        raise TypeError(f"Unsupported pretrained checkpoint format: {type(raw_state_dict)}")

    target_model = getattr(backbone, "model", backbone)
    model_state_dict = target_model.state_dict()
    candidates = _make_state_dict_candidates(raw_state_dict, target_model, model_name)

    best_name = None
    best_candidate = None
    best_score = (-1, -1)
    for name, candidate_state_dict in candidates.items():
        score = _score_state_dict(candidate_state_dict, model_state_dict)
        if score > best_score:
            best_score = score
            best_name = name
            best_candidate = candidate_state_dict

    filtered_state_dict = {
        key: value
        for key, value in best_candidate.items()
        if key in model_state_dict and tuple(model_state_dict[key].shape) == tuple(value.shape)
    }

    return {
        "target_model": target_model,
        "candidate_name": best_name,
        "shape_matched": int(best_score[0]),
        "name_matched": int(best_score[1]),
        "filtered_state_dict": filtered_state_dict,
        "model_key_count": len(model_state_dict),
        "raw_key_count": len(raw_state_dict),
    }


def _load_local_pretrained_weights(backbone, weight_path, model_name):
    load_info = _prepare_best_pretrained_state_dict(backbone, weight_path, model_name)
    target_model = load_info["target_model"]
    filtered_state_dict = load_info["filtered_state_dict"]
    model_key_count = load_info["model_key_count"]
    matched_count = len(filtered_state_dict)

    if matched_count == 0:
        raise RuntimeError(
            "Failed to match any pretrained encoder weights. "
            f"candidate={load_info['candidate_name']} raw_keys={load_info['raw_key_count']} model_keys={model_key_count}"
        )

    if matched_count < int(0.8 * model_key_count):
        raise RuntimeError(
            "Too few pretrained encoder weights matched the current backbone. "
            f"candidate={load_info['candidate_name']} matched={matched_count}/{model_key_count} "
            f"raw_keys={load_info['raw_key_count']}"
        )

    incompatible = target_model.load_state_dict(filtered_state_dict, strict=False)
    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))

    print(
        "[encoder] local pretrained load finished: "
        f"candidate={load_info['candidate_name']} matched={matched_count}/{model_key_count} "
        f"missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
    )


class PVTEncoder(nn.Module):
    def __init__(self, name, pretrained, pretrained_path=None):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required to use PVTv2 encoders. Please install it with `pip install timm`.")

        self.backbone = timm.create_model(name, features_only=True, pretrained=False, out_indices=(0, 1, 2, 3))
        if pretrained:
            _load_local_pretrained_weights(self.backbone, pretrained_path, name)
        self.out_channels = list(self.backbone.feature_info.channels())

    def forward(self, x):
        return list(self.backbone(x))


class FPNDecoder(nn.Module):
    def __init__(self, in_channels, decoder_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(ch, decoder_channels, kernel_size=1, bias=False) for ch in in_channels])
        self.output_convs = nn.ModuleList(
            [ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3, padding=1) for _ in in_channels])
        self.fuse = nn.Sequential(
            ConvBNReLU(decoder_channels * 4, decoder_channels, kernel_size=3, padding=1),
            ConvBNReLU(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.1),
        )
        self.head = nn.Conv2d(decoder_channels // 2, 2, kernel_size=1)

    def forward(self, features, output_size):
        c2, c3, c4, c5 = features
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.lateral_convs[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral_convs[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)

        p2 = self.output_convs[0](p2)
        p3 = self.output_convs[1](p3)
        p4 = self.output_convs[2](p4)
        p5 = self.output_convs[3](p5)

        fused = torch.cat(
            [
                p2,
                F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(p4, size=p2.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(p5, size=p2.shape[-2:], mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        fused = self.fuse(fused)
        logits = self.head(fused)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        evidence = F.softplus(logits)
        alpha = evidence[:, 0:1, :, :] + 1.0
        beta = evidence[:, 1:2, :, :] + 1.0
        S = alpha + beta
        pred = alpha / S
        uncertainty = 2.0 / S
        return {
            "probs": pred,
            "uncertainty": uncertainty,
            "alpha": alpha,
            "beta": beta,
        }


class MeanTeacherSODNet(nn.Module):
    def __init__(self, encoder_name="pvt_v2_b2", encoder_pretrained=True, encoder_pretrained_path=None,
                 decoder_channels=256):
        super().__init__()
        self.encoder = build_encoder(encoder_name, encoder_pretrained, encoder_pretrained_path)
        self.decoder = FPNDecoder(self.encoder.out_channels, decoder_channels)

    def forward(self, x, return_features=False):
        features = self.encoder(x)
        outputs = self.decoder(features, output_size=(x.shape[-2], x.shape[-1]))
        if return_features:
            outputs["encoder_features"] = features
        return outputs


def build_encoder(name, pretrained, pretrained_path=None):
    if name.startswith("pvt_v2"):
        return PVTEncoder(name, pretrained, pretrained_path)
    raise ValueError(f"Unsupported encoder_name: {name}")


def build_sod_model(cfg):
    return MeanTeacherSODNet(
        encoder_name=str(getattr(cfg, "encoder_name", "pvt_v2_b2")),
        encoder_pretrained=bool(getattr(cfg, "encoder_pretrained", True)),
        encoder_pretrained_path=getattr(cfg, "encoder_pretrained_path", None),
        decoder_channels=int(getattr(cfg, "decoder_channels", 256)),
    )


def build_generator(cfg):
    return build_sod_model(cfg)


def get_saliency_logits(output):
    if isinstance(output, dict):
        if "logits" in output:
            return output["logits"]
        return torch.logit(output["probs"].clamp(min=1e-6, max=1.0 - 1e-6))
    if isinstance(output, (tuple, list)):
        return torch.logit(output[0].clamp(min=1e-6, max=1.0 - 1e-6))
    return output


def get_saliency_probs(output):
    if isinstance(output, dict):
        if "probs" in output:
            return output["probs"]
        return torch.sigmoid(output["logits"])
    if isinstance(output, (tuple, list)):
        return output[0]
    return torch.sigmoid(output)


def get_uncertainty_map(output):
    if isinstance(output, dict):
        return output.get("uncertainty", output.get("sigma"))
    if isinstance(output, (tuple, list)) and len(output) > 1:
        return output[1]
    return None


def get_beta_params(output):
    if isinstance(output, dict):
        return output["alpha"], output["beta"]
    if isinstance(output, (tuple, list)) and len(output) > 3:
        return output[2], output[3]
    raise KeyError("The model output does not contain Beta parameters alpha and beta.")
