from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torchvision import models, transforms

try:
    import timm
except Exception:  # pragma: no cover - optional dependency
    timm = None


SUPPORTED_MODEL_ALIASES: dict[str, str] = {
    "mobilenet_v3_small": "mobilenet_v3_small",
    "mobilenet_v3_large": "mobilenet_v3_large",
    "mobilenet_v3_small_quantized": "mobilenet_v3_small_quantized",
    "mobilenet_v3_large_quantized": "mobilenet_v3_large_quantized",
    "quantized_mobilenet_v3_small": "mobilenet_v3_small_quantized",
    "quantized_mobilenet_v3_large": "mobilenet_v3_large_quantized",
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet-b0": "efficientnet_b0",
    "vit_base_patch16_224": "vit_base_patch16_224",
    "vit_b_16": "vit_base_patch16_224",
    "vit-b-16": "vit_base_patch16_224",
    "convnext_tiny": "convnext_tiny",
    "convnext-tiny": "convnext_tiny",
}


def supported_model_names() -> list[str]:
    return sorted(set(SUPPORTED_MODEL_ALIASES.values()))


def resolve_model_name(name: str) -> str:
    key = str(name or "").strip().lower()
    if key not in SUPPORTED_MODEL_ALIASES:
        raise ValueError(
            f"Unsupported model_name: {name}. Supported names: {', '.join(supported_model_names())}"
        )
    return SUPPORTED_MODEL_ALIASES[key]


def build_preprocess(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@dataclass(frozen=True)
class ModelInfo:
    canonical_name: str
    supports_grad_cam: bool
    cam_kind: str | None
    requires_cpu: bool
    num_classes: int = 1000


class BaseEmbedder(torch.nn.Module):
    supports_grad_cam: bool = True
    cam_kind: str | None = "conv"
    requires_cpu: bool = False
    canonical_name: str = ""

    def forward_with_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding, logits, _ = self.forward_with_activations(x)
        return embedding, logits


def _to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.dequantize() if getattr(x, "is_quantized", False) else x


def _flatten_embedding(x: torch.Tensor) -> torch.Tensor:
    x = _to_float_tensor(x)
    if x.ndim == 4:
        if x.shape[1] < x.shape[-1] and x.shape[-1] >= 16:
            x = x.permute(0, 3, 1, 2)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)
    if x.ndim == 3:
        if x.shape[1] > 1:
            return x[:, 0]
        return x.squeeze(1)
    return x.reshape(x.shape[0], -1)


class TorchvisionMobileNetEmbedder(BaseEmbedder):
    def __init__(self, base: torch.nn.Module, canonical_name: str, supports_grad_cam: bool = True, requires_cpu: bool = False):
        super().__init__()
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = base.classifier
        self.canonical_name = canonical_name
        self.supports_grad_cam = supports_grad_cam
        self.cam_kind = "conv" if supports_grad_cam else None
        self.requires_cpu = requires_cpu

    def forward_with_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_maps = self.features(x)
        pooled = self.avgpool(feat_maps)
        embedding = torch.flatten(pooled, 1)
        logits = self.classifier(embedding)
        return _to_float_tensor(embedding), _to_float_tensor(logits), _to_float_tensor(feat_maps)


class TorchvisionResNetEmbedder(BaseEmbedder):
    def __init__(self, base: torch.nn.Module, canonical_name: str):
        super().__init__()
        self.base = base
        self.canonical_name = canonical_name
        self.supports_grad_cam = True
        self.cam_kind = "conv"
        self.requires_cpu = False

    def forward_with_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        feat_maps = self.base.layer4(x)
        pooled = self.base.avgpool(feat_maps)
        embedding = torch.flatten(pooled, 1)
        logits = self.base.fc(embedding)
        return embedding, logits, feat_maps


class TimmEmbedder(BaseEmbedder):
    def __init__(self, canonical_name: str, cam_kind: str):
        super().__init__()
        if timm is None:
            raise ImportError(
                "timm is required for model_name="
                f"{canonical_name}. Install project dependencies including timm."
            )
        self.model = timm.create_model(canonical_name, pretrained=True, num_classes=1000)
        self.canonical_name = canonical_name
        self.supports_grad_cam = True
        self.cam_kind = cam_kind
        self.requires_cpu = False

    def _unwrap(self, x: Any) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            return self._unwrap(x[-1])
        if isinstance(x, dict):
            for key in ("x_norm_clstoken", "x_norm_patchtokens", "x", "feat", "features"):
                if key in x:
                    return self._unwrap(x[key])
            return self._unwrap(next(iter(x.values())))
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Unsupported timm feature type: {type(x)!r}")
        return x

    def forward_with_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self._unwrap(self.model.forward_features(x))
        if hasattr(self.model, "forward_head"):
            logits = self._unwrap(self.model.forward_head(feat, pre_logits=False))
            embedding = self._unwrap(self.model.forward_head(feat, pre_logits=True))
        else:
            logits = self._unwrap(self.model(x))
            embedding = feat
        return _flatten_embedding(embedding), _flatten_embedding(logits), self._unwrap(feat)


def get_model_info(name: str) -> ModelInfo:
    canonical = resolve_model_name(name)
    if canonical in {"mobilenet_v3_small", "mobilenet_v3_large", "resnet50", "efficientnet_b0", "convnext_tiny"}:
        return ModelInfo(canonical, supports_grad_cam=True, cam_kind="conv", requires_cpu=False)
    if canonical == "vit_base_patch16_224":
        return ModelInfo(canonical, supports_grad_cam=True, cam_kind="tokens", requires_cpu=False)
    if canonical in {"mobilenet_v3_small_quantized", "mobilenet_v3_large_quantized"}:
        return ModelInfo(canonical, supports_grad_cam=False, cam_kind=None, requires_cpu=True)
    raise ValueError(f"Unsupported model_name: {name}")


def build_embedder(name: str) -> BaseEmbedder:
    canonical = resolve_model_name(name)

    if canonical == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        base = models.mobilenet_v3_small(weights=weights)
        return TorchvisionMobileNetEmbedder(base, canonical_name=canonical)

    if canonical == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        base = models.mobilenet_v3_large(weights=weights)
        return TorchvisionMobileNetEmbedder(base, canonical_name=canonical)

    if canonical == "mobilenet_v3_small_quantized":
        from torchvision.models import quantization as qmodels

        weights = qmodels.MobileNet_V3_Small_QuantizedWeights.DEFAULT
        base = qmodels.mobilenet_v3_small(weights=weights, quantize=True)
        return TorchvisionMobileNetEmbedder(
            base,
            canonical_name=canonical,
            supports_grad_cam=False,
            requires_cpu=True,
        )

    if canonical == "mobilenet_v3_large_quantized":
        from torchvision.models import quantization as qmodels

        weights = qmodels.MobileNet_V3_Large_QuantizedWeights.DEFAULT
        base = qmodels.mobilenet_v3_large(weights=weights, quantize=True)
        return TorchvisionMobileNetEmbedder(
            base,
            canonical_name=canonical,
            supports_grad_cam=False,
            requires_cpu=True,
        )

    if canonical == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        base = models.resnet50(weights=weights)
        return TorchvisionResNetEmbedder(base, canonical_name=canonical)

    if canonical == "efficientnet_b0":
        return TimmEmbedder(canonical, cam_kind="conv")

    if canonical == "vit_base_patch16_224":
        return TimmEmbedder(canonical, cam_kind="tokens")

    if canonical == "convnext_tiny":
        return TimmEmbedder(canonical, cam_kind="conv")

    raise ValueError(f"Unsupported model_name: {name}")


def resolve_runtime_device(requested_device: str, model_info: ModelInfo) -> str:
    requested = str(requested_device or "auto").strip().lower()
    if model_info.requires_cpu:
        return "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def build_cam_from_activations(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    cam_kind: str,
) -> torch.Tensor:
    if cam_kind == "conv":
        if activations.ndim != 4 or gradients.ndim != 4:
            raise ValueError("Conv CAM expects [B, C, H, W] or [B, H, W, C] feature maps.")
        if activations.shape[1] < activations.shape[-1] and activations.shape[-1] >= 16:
            activations = activations.permute(0, 3, 1, 2)
            gradients = gradients.permute(0, 3, 1, 2)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1))[0]
        return cam

    if cam_kind == "tokens":
        act = activations
        grad = gradients
        if act.ndim != 3 or grad.ndim != 3:
            raise ValueError("Token CAM expects [B, N, C] activations and gradients.")
        if act.shape[1] > 1:
            act = act[:, 1:, :]
            grad = grad[:, 1:, :]
        token_cam = torch.relu((grad * act).sum(dim=-1))[0]
        side = int(round(math.sqrt(token_cam.numel())))
        if side * side != token_cam.numel():
            raise ValueError(
                f"Could not reshape token activations of length {token_cam.numel()} into a square map."
            )
        return token_cam.reshape(side, side)

    raise ValueError(f"Unsupported cam_kind: {cam_kind}")
