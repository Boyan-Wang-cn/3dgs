"""Strict PSNR, SSIM, and LPIPS evaluation for matched 3DGS render views.

The same API can evaluate a fixed camera subset at intermediate checkpoints or
the complete terminal test-view directories. All four input directories must
contain exactly the same relative image paths. Fixed-subset and terminal
full-view results describe different evaluation scopes and must not be used to
compute cross-scope stepwise deltas.

Final experiment tables should prefer terminal full test views. Fixed subsets
are intended for training diagnostics, checkpoint quality curves, and reducing
the cost of intermediate evaluation. This module only evaluates artifacts; it
does not feed PSNR, SSIM, or LPIPS into reinforcement-learning rewards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Protocol
import argparse
import csv
import json
import math
import os
import re
import tempfile

import numpy as np
from PIL import Image, UnidentifiedImageError


_SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_JSON_SCHEMA = "fixed_view_quality_metrics_v1"
_CSV_FIELDS = (
    "relative_path",
    "width",
    "height",
    "original_psnr_db",
    "compressed_psnr_db",
    "psnr_drop_db",
    "original_ssim",
    "compressed_ssim",
    "ssim_drop",
    "original_lpips",
    "compressed_lpips",
    "lpips_increase",
)


@dataclass(frozen=True)
class PerViewQualityMetrics:
    """Quality metrics for one exactly matched camera view."""

    relative_path: str
    width: int
    height: int
    original_psnr_db: float
    compressed_psnr_db: float
    psnr_drop_db: float
    original_ssim: float
    compressed_ssim: float
    ssim_drop: float
    original_lpips: float
    compressed_lpips: float
    lpips_increase: float


@dataclass(frozen=True)
class MetricSummary:
    """Unweighted descriptive statistics for one per-view metric."""

    count: int
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class FixedViewQualityResult:
    """Complete per-view and aggregate fixed-view evaluation result."""

    view_count: int
    relative_paths: tuple[str, ...]
    per_view: tuple[PerViewQualityMetrics, ...]
    original: dict[str, Any]
    compressed: dict[str, Any]
    degradation: dict[str, Any]
    worst_views: dict[str, dict[str, Any]]
    metric_configuration: dict[str, Any]


class LPIPSScorerProtocol(Protocol):
    """Minimal scorer interface accepted by the evaluation functions."""

    def score(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Return one finite perceptual distance for two CHW RGB images."""


def _collect_image_paths(root: Path, label: str) -> dict[str, Path]:
    """Collect supported images and reject case-insensitive ambiguities."""
    if not root.exists() or not root.is_dir():
        raise ValueError(f"{label} must be an existing directory: {root}")
    collected: dict[str, Path] = {}
    casefold_paths: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in _SUPPORTED_IMAGE_EXTENSIONS:
            continue
        relative_path = path.relative_to(root).as_posix()
        folded = relative_path.casefold()
        previous = casefold_paths.get(folded)
        if previous is not None and previous != relative_path:
            raise ValueError(
                f"{label} contains casefold-conflicting images: "
                f"{previous!r} and {relative_path!r}"
            )
        casefold_paths[folded] = relative_path
        collected[relative_path] = path
    return collected


def collect_exact_fixed_view_paths(
    original_render_dir: str | Path,
    original_gt_dir: str | Path,
    compressed_render_dir: str | Path,
    compressed_gt_dir: str | Path,
) -> tuple[str, ...]:
    """Return deterministic relative paths only when all four sets are equal."""
    roots = {
        "original_render": Path(original_render_dir),
        "original_gt": Path(original_gt_dir),
        "compressed_render": Path(compressed_render_dir),
        "compressed_gt": Path(compressed_gt_dir),
    }
    mappings = {
        label: _collect_image_paths(root, label) for label, root in roots.items()
    }
    path_sets = {label: set(mapping) for label, mapping in mappings.items()}
    union = set().union(*path_sets.values())
    if not union:
        raise ValueError("the four fixed-view directories contain no supported images")
    common = set.intersection(*path_sets.values())
    if any(paths != union for paths in path_sets.values()):
        details = []
        for label, paths in path_sets.items():
            missing = sorted(union - paths)
            extra = sorted(paths - common)
            details.append(f"{label}: missing={missing}, extra={extra}")
        raise ValueError(
            "fixed-view relative path sets must be exactly equal; " + "; ".join(details)
        )
    return tuple(sorted(union))


def load_metric_rgb_image(path: str | Path) -> np.ndarray:
    """Decode one still image as float32 RGB CHW sRGB values in [0, 1]."""
    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            if int(getattr(image, "n_frames", 1)) != 1:
                raise ValueError(f"animated or multi-frame image is not allowed: {image_path}")
            image.load()
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)
    except ValueError:
        raise
    except (OSError, UnidentifiedImageError, SyntaxError) as exc:
        raise ValueError(f"unable to decode image {image_path}: {exc}") from exc
    if array.ndim != 3 or array.shape[2] != 3 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"image must decode to a nonempty RGB array: {image_path}")
    chw = np.transpose(array, (2, 0, 1)).astype(np.float32) / np.float32(255.0)
    if not np.all(np.isfinite(chw)) or np.any(chw < 0.0) or np.any(chw > 1.0):
        raise ValueError(f"decoded image values are outside finite [0, 1]: {image_path}")
    return np.ascontiguousarray(chw, dtype=np.float32)


def _validated_rgb_pair(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    minimum_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate equal finite CHW RGB inputs and return float64 arrays."""
    try:
        prediction_array = np.asarray(prediction, dtype=np.float64)
        target_array = np.asarray(target, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("metric inputs must be numeric CHW RGB arrays") from exc
    if (
        prediction_array.ndim != 3
        or target_array.ndim != 3
        or prediction_array.shape[0] != 3
        or target_array.shape[0] != 3
        or prediction_array.shape != target_array.shape
    ):
        raise ValueError(
            "prediction and target must have identical shape [3, H, W]"
        )
    height, width = prediction_array.shape[1:]
    if height <= 0 or width <= 0:
        raise ValueError("metric images must have positive height and width")
    if minimum_size is not None and (height < minimum_size or width < minimum_size):
        raise ValueError(
            f"metric images must be at least {minimum_size}x{minimum_size}"
        )
    if not np.all(np.isfinite(prediction_array)) or not np.all(np.isfinite(target_array)):
        raise ValueError("metric inputs must contain only finite values")
    if (
        np.any(prediction_array < 0.0)
        or np.any(prediction_array > 1.0)
        or np.any(target_array < 0.0)
        or np.any(target_array > 1.0)
    ):
        raise ValueError("metric input values must be in [0, 1]")
    return prediction_array, target_array


def compute_psnr_rgb(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute RGB PSNR from one global MSE with data range fixed to 1.0."""
    prediction_array, target_array = _validated_rgb_pair(prediction, target)
    mse = float(np.mean(np.square(prediction_array - target_array), dtype=np.float64))
    if mse == 0.0:
        return math.inf
    return float(-10.0 * math.log10(mse))


def compute_ssim_rgb(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute classic 3DGS-style valid 11x11 Gaussian-window RGB SSIM."""
    prediction_array, target_array = _validated_rgb_pair(
        prediction, target, minimum_size=11
    )
    try:
        import torch
        import torch.nn.functional as torch_functional
    except ImportError as exc:
        raise ImportError(
            "compute_ssim_rgb requires PyTorch; no approximate fallback is used"
        ) from exc

    coordinates = torch.arange(11, dtype=torch.float64) - 5.0
    gaussian_1d = torch.exp(-(coordinates**2) / (2.0 * 1.5**2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    window_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
    window = window_2d.reshape(1, 1, 11, 11).repeat(3, 1, 1, 1)
    x = torch.from_numpy(np.ascontiguousarray(prediction_array)).unsqueeze(0)
    y = torch.from_numpy(np.ascontiguousarray(target_array)).unsqueeze(0)
    mu_x = torch_functional.conv2d(x, window, groups=3)
    mu_y = torch_functional.conv2d(y, window, groups=3)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x = torch_functional.conv2d(x * x, window, groups=3) - mu_x_sq
    sigma_y = torch_functional.conv2d(y * y, window, groups=3) - mu_y_sq
    sigma_xy = torch_functional.conv2d(x * y, window, groups=3) - mu_xy
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    value = float((numerator / denominator).mean().item())
    if not math.isfinite(value):
        raise RuntimeError("SSIM produced a non-finite value")
    if value < -1e-6 or value > 1.0 + 1e-6:
        raise RuntimeError(f"SSIM result is outside its valid tolerance: {value}")
    return float(min(1.0, max(0.0, value)))


class OfficialLPIPSScorer:
    """Strict wrapper around the official ``lpips.LPIPS`` implementation."""

    _ALLOWED_NETS = {"alex", "vgg", "squeeze"}
    _DEVICE_PATTERN = re.compile(r"^(auto|cpu|cuda(?::[0-9]+)?)$")

    def __init__(self, net: str = "alex", device: str = "auto") -> None:
        if net not in self._ALLOWED_NETS:
            raise ValueError("LPIPS net must be one of: alex, vgg, squeeze")
        if not isinstance(device, str) or self._DEVICE_PATTERN.fullmatch(device) is None:
            raise ValueError("LPIPS device must be auto, cpu, cuda, or cuda:N")
        try:
            import torch
        except ImportError as exc:
            raise ImportError("OfficialLPIPSScorer requires PyTorch") from exc
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "OfficialLPIPSScorer requires the official 'lpips' Python package; "
                "no surrogate metric is used"
            ) from exc
        resolved = "cuda" if device == "auto" and torch.cuda.is_available() else (
            "cpu" if device == "auto" else device
        )
        if resolved.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"requested LPIPS device is unavailable: {resolved}")
        try:
            resolved_device = torch.device(resolved)
            model = lpips.LPIPS(net=net, version="0.1")
            model.eval()
            model.to(resolved_device)
        except Exception:
            raise
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        try:
            package_version = importlib_metadata.version("lpips")
        except Exception:
            package_version = "unknown"
        self._torch = torch
        self._model = model
        self._net = net
        self._resolved_device = str(resolved_device)
        self._package_version = package_version

    @property
    def backend(self) -> str:
        return "official_lpips"

    @property
    def net(self) -> str:
        return self._net

    @property
    def version(self) -> str:
        return "0.1"

    @property
    def resolved_device(self) -> str:
        return self._resolved_device

    @property
    def package_version(self) -> str:
        return self._package_version

    def score(self, prediction: np.ndarray, target: np.ndarray) -> float:
        prediction_array, target_array = _validated_rgb_pair(prediction, target)
        prediction_tensor = self._torch.from_numpy(
            np.ascontiguousarray(prediction_array, dtype=np.float32)
        ).unsqueeze(0)
        target_tensor = self._torch.from_numpy(
            np.ascontiguousarray(target_array, dtype=np.float32)
        ).unsqueeze(0)
        prediction_tensor = prediction_tensor.to(self._resolved_device) * 2.0 - 1.0
        target_tensor = target_tensor.to(self._resolved_device) * 2.0 - 1.0
        with self._torch.no_grad():
            output = self._model(prediction_tensor, target_tensor)
        if not self._torch.is_tensor(output) or output.numel() == 0:
            raise RuntimeError("official LPIPS returned an invalid output")
        value = float(output.detach().mean().cpu().item())
        if not math.isfinite(value):
            raise RuntimeError("official LPIPS returned a non-finite value")
        return value


def _score_lpips(
    scorer: LPIPSScorerProtocol,
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """Require an injected scorer to return exactly one finite scalar."""
    value = scorer.score(prediction, target)
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("LPIPS scorer must return one scalar value")
    try:
        result = float(array.item())
    except (TypeError, ValueError) as exc:
        raise ValueError("LPIPS scorer must return a numeric scalar") from exc
    if not math.isfinite(result):
        raise ValueError("LPIPS scorer returned a non-finite value")
    return result


def _extended_difference(left: float, right: float) -> float:
    """Subtract extended values without producing NaN for equal infinities."""
    if math.isinf(left) and left == right:
        return 0.0
    return float(left - right)


def evaluate_single_fixed_view(
    original_render: np.ndarray,
    original_gt: np.ndarray,
    compressed_render: np.ndarray,
    compressed_gt: np.ndarray,
    lpips_scorer: LPIPSScorerProtocol,
    relative_path: str,
) -> PerViewQualityMetrics:
    """Evaluate one view, using the verified original GT for both renders."""
    original_render_array, original_gt_array = _validated_rgb_pair(
        original_render, original_gt
    )
    compressed_render_array, compressed_gt_array = _validated_rgb_pair(
        compressed_render, compressed_gt
    )
    if original_render_array.shape != compressed_render_array.shape:
        raise ValueError("all four images for one view must have identical dimensions")
    if not np.array_equal(original_gt_array, compressed_gt_array):
        raise ValueError("original_gt and compressed_gt must be pixel-identical")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("relative_path must be a nonempty string")
    original_psnr = compute_psnr_rgb(original_render_array, original_gt_array)
    compressed_psnr = compute_psnr_rgb(compressed_render_array, original_gt_array)
    original_ssim = compute_ssim_rgb(original_render_array, original_gt_array)
    compressed_ssim = compute_ssim_rgb(compressed_render_array, original_gt_array)
    original_lpips = _score_lpips(
        lpips_scorer, original_render_array, original_gt_array
    )
    compressed_lpips = _score_lpips(
        lpips_scorer, compressed_render_array, original_gt_array
    )
    height, width = original_gt_array.shape[1:]
    return PerViewQualityMetrics(
        relative_path=relative_path,
        width=int(width),
        height=int(height),
        original_psnr_db=original_psnr,
        compressed_psnr_db=compressed_psnr,
        psnr_drop_db=_extended_difference(original_psnr, compressed_psnr),
        original_ssim=original_ssim,
        compressed_ssim=compressed_ssim,
        ssim_drop=float(original_ssim - compressed_ssim),
        original_lpips=original_lpips,
        compressed_lpips=compressed_lpips,
        lpips_increase=float(compressed_lpips - original_lpips),
    )


def _metric_summary(values: list[float], name: str) -> MetricSummary:
    """Summarize finite or infinite values without emitting NaN statistics."""
    if not values or any(math.isnan(value) for value in values):
        raise ValueError(f"{name} requires nonempty values without NaN")
    sorted_values = sorted(float(value) for value in values)
    count = len(sorted_values)
    middle = count // 2
    median = (
        sorted_values[middle]
        if count % 2
        else _extended_midpoint(sorted_values[middle - 1], sorted_values[middle])
    )
    has_positive_inf = any(value == math.inf for value in sorted_values)
    has_negative_inf = any(value == -math.inf for value in sorted_values)
    if has_positive_inf and has_negative_inf:
        raise ValueError(f"{name} contains both positive and negative infinity")
    if has_positive_inf:
        mean = math.inf
    elif has_negative_inf:
        mean = -math.inf
    else:
        mean = float(math.fsum(sorted_values) / count)
    if has_positive_inf or has_negative_inf:
        std = 0.0 if all(value == sorted_values[0] for value in sorted_values) else math.inf
    else:
        std = float(
            math.sqrt(math.fsum((value - mean) ** 2 for value in sorted_values) / count)
        )
    return MetricSummary(
        count=count,
        mean=mean,
        median=median,
        std=std,
        minimum=sorted_values[0],
        maximum=sorted_values[-1],
    )


def _extended_midpoint(left: float, right: float) -> float:
    """Return a stable median midpoint for equal extended infinities."""
    if math.isinf(left) and left == right:
        return left
    result = (left + right) / 2.0
    if math.isnan(result):
        raise ValueError("median is undefined for opposite infinities")
    return float(result)


def _pooled_psnr(total_squared_error: float, total_values: int) -> float:
    """Compute PSNR after pooling pixel-channel squared errors across views."""
    if total_values <= 0 or total_squared_error < 0.0:
        raise ValueError("pooled PSNR requires positive sample count and valid SSE")
    mse = total_squared_error / total_values
    return math.inf if mse == 0.0 else float(-10.0 * math.log10(mse))


def _scorer_configuration(scorer: LPIPSScorerProtocol) -> dict[str, str]:
    """Record scorer provenance without retaining the scorer or model."""
    return {
        "lpips_backend": str(getattr(scorer, "backend", "injected_non_official")),
        "lpips_net": str(getattr(scorer, "net", "unknown")),
        "lpips_version": str(getattr(scorer, "version", "unknown")),
        "lpips_device": str(getattr(scorer, "resolved_device", "unknown")),
        "lpips_package_version": str(
            getattr(scorer, "package_version", "unknown")
        ),
        "psnr_data_range": "1.0",
        "ssim_window": "11x11_gaussian_sigma_1.5_valid",
        "evaluation_scope": "exact_relative_path_set",
    }


def evaluate_fixed_view_quality(
    original_render_dir: str | Path,
    original_gt_dir: str | Path,
    compressed_render_dir: str | Path,
    compressed_gt_dir: str | Path,
    *,
    lpips_scorer: LPIPSScorerProtocol | None = None,
    lpips_net: str = "alex",
    device: str = "auto",
) -> FixedViewQualityResult:
    """Evaluate all exactly matched views with equal per-view weighting."""
    relative_paths = collect_exact_fixed_view_paths(
        original_render_dir,
        original_gt_dir,
        compressed_render_dir,
        compressed_gt_dir,
    )
    scorer = lpips_scorer or OfficialLPIPSScorer(net=lpips_net, device=device)
    roots = (
        Path(original_render_dir),
        Path(original_gt_dir),
        Path(compressed_render_dir),
        Path(compressed_gt_dir),
    )
    per_view: list[PerViewQualityMetrics] = []
    original_sse = 0.0
    compressed_sse = 0.0
    total_values = 0
    for relative_path in relative_paths:
        path_parts = relative_path.split("/")
        images = [load_metric_rgb_image(root.joinpath(*path_parts)) for root in roots]
        if any(image.shape != images[0].shape for image in images[1:]):
            raise ValueError(
                f"all four images must have identical H and W for {relative_path!r}"
            )
        original_render, original_gt, compressed_render, compressed_gt = images
        if not np.array_equal(original_gt, compressed_gt):
            raise ValueError(
                f"original and compressed GT differ for {relative_path!r}"
            )
        per_view.append(
            evaluate_single_fixed_view(
                original_render,
                original_gt,
                compressed_render,
                compressed_gt,
                scorer,
                relative_path,
            )
        )
        original_delta = original_render.astype(np.float64) - original_gt.astype(
            np.float64
        )
        compressed_delta = compressed_render.astype(np.float64) - original_gt.astype(
            np.float64
        )
        original_sse += float(np.sum(original_delta * original_delta, dtype=np.float64))
        compressed_sse += float(
            np.sum(compressed_delta * compressed_delta, dtype=np.float64)
        )
        total_values += int(original_gt.size)

    original_psnr = _metric_summary(
        [view.original_psnr_db for view in per_view], "original PSNR"
    )
    compressed_psnr = _metric_summary(
        [view.compressed_psnr_db for view in per_view], "compressed PSNR"
    )
    original_ssim = _metric_summary(
        [view.original_ssim for view in per_view], "original SSIM"
    )
    compressed_ssim = _metric_summary(
        [view.compressed_ssim for view in per_view], "compressed SSIM"
    )
    original_lpips = _metric_summary(
        [view.original_lpips for view in per_view], "original LPIPS"
    )
    compressed_lpips = _metric_summary(
        [view.compressed_lpips for view in per_view], "compressed LPIPS"
    )
    psnr_drop = _metric_summary(
        [view.psnr_drop_db for view in per_view], "PSNR drop"
    )
    ssim_drop = _metric_summary(
        [view.ssim_drop for view in per_view], "SSIM drop"
    )
    lpips_increase = _metric_summary(
        [view.lpips_increase for view in per_view], "LPIPS increase"
    )
    original = {
        "psnr_db": original_psnr,
        "ssim": original_ssim,
        "lpips": original_lpips,
        "mean_psnr_db": original_psnr.mean,
        "mean_per_view_psnr_db": original_psnr.mean,
        "pooled_psnr_db": _pooled_psnr(original_sse, total_values),
        "mean_ssim": original_ssim.mean,
        "mean_lpips": original_lpips.mean,
    }
    compressed = {
        "psnr_db": compressed_psnr,
        "ssim": compressed_ssim,
        "lpips": compressed_lpips,
        "mean_psnr_db": compressed_psnr.mean,
        "mean_per_view_psnr_db": compressed_psnr.mean,
        "pooled_psnr_db": _pooled_psnr(compressed_sse, total_values),
        "mean_ssim": compressed_ssim.mean,
        "mean_lpips": compressed_lpips.mean,
    }
    degradation = {
        "psnr_drop_db": psnr_drop,
        "ssim_drop": ssim_drop,
        "lpips_increase": lpips_increase,
        "mean_psnr_drop_db": psnr_drop.mean,
        "median_psnr_drop_db": psnr_drop.median,
        "mean_ssim_drop": ssim_drop.mean,
        "median_ssim_drop": ssim_drop.median,
        "mean_lpips_increase": lpips_increase.mean,
        "median_lpips_increase": lpips_increase.median,
    }
    worst_specifications = {
        "largest_psnr_drop": "psnr_drop_db",
        "largest_ssim_drop": "ssim_drop",
        "largest_lpips_increase": "lpips_increase",
    }
    worst_views: dict[str, dict[str, Any]] = {}
    for output_name, attribute in worst_specifications.items():
        worst = max(per_view, key=lambda view, field=attribute: getattr(view, field))
        worst_views[output_name] = {
            "relative_path": worst.relative_path,
            "metric_value": float(getattr(worst, attribute)),
        }
    return FixedViewQualityResult(
        view_count=len(per_view),
        relative_paths=relative_paths,
        per_view=tuple(per_view),
        original=original,
        compressed=compressed,
        degradation=degradation,
        worst_views=worst_views,
        metric_configuration=_scorer_configuration(scorer),
    )


def _json_safe(value: Any) -> Any:
    """Convert dataclasses and extended floats to strict JSON-safe values."""
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            raise ValueError("NaN cannot be serialized")
        if number == math.inf:
            return "inf"
        if number == -math.inf:
            return "-inf"
        return number
    return value


def _atomic_text_path(output_path: str | Path) -> tuple[Path, Path]:
    """Create a same-directory temporary path for atomic output replacement."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return destination, Path(temporary_name)


def save_fixed_view_quality_json(
    result: FixedViewQualityResult,
    output_path: str | Path,
) -> Path:
    """Atomically write a strict UTF-8 JSON report."""
    if not isinstance(result, FixedViewQualityResult):
        raise ValueError("result must be FixedViewQualityResult")
    payload = {
        "schema": _JSON_SCHEMA,
        "summary": {
            "view_count": result.view_count,
            "relative_paths": result.relative_paths,
            "original": result.original,
            "compressed": result.compressed,
            "degradation": result.degradation,
            "worst_views": result.worst_views,
        },
        "per_view": result.per_view,
        "metric_configuration": result.metric_configuration,
    }
    destination, temporary = _atomic_text_path(output_path)
    try:
        temporary.write_text(
            json.dumps(
                _json_safe(payload),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _csv_value(value: Any) -> Any:
    """Render extended floats without non-standard CSV numeric tokens."""
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            raise ValueError("NaN cannot be serialized")
        if number == math.inf:
            return "inf"
        if number == -math.inf:
            return "-inf"
    return value


def save_fixed_view_quality_csv(
    result: FixedViewQualityResult,
    output_path: str | Path,
) -> Path:
    """Atomically write deterministic per-view rows without a summary row."""
    if not isinstance(result, FixedViewQualityResult):
        raise ValueError("result must be FixedViewQualityResult")
    destination, temporary = _atomic_text_path(output_path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for view in result.per_view:
                row = asdict(view)
                writer.writerow({field: _csv_value(row[field]) for field in _CSV_FIELDS})
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def validate_fixed_view_quality_metrics() -> bool:
    """Validate metrics and strict artifact handling without loading real LPIPS."""
    import shutil

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    def require_error(
        exception_type: type[BaseException], callback: Any, message: str
    ) -> None:
        try:
            callback()
        except exception_type:
            return
        raise AssertionError(message)

    class FakeLPIPSScorer:
        backend = "fake_lpips_validation"
        net = "fake_mae"
        version = "validation_only"
        resolved_device = "cpu"
        package_version = "not_applicable"

        def score(self, prediction: np.ndarray, target: np.ndarray) -> float:
            return float(np.mean(np.abs(prediction - target), dtype=np.float64))

    class NaNLPIPSScorer(FakeLPIPSScorer):
        def score(self, prediction: np.ndarray, target: np.ndarray) -> float:
            _ = prediction, target
            return math.nan

    class ArrayLPIPSScorer(FakeLPIPSScorer):
        def score(self, prediction: np.ndarray, target: np.ndarray) -> Any:
            _ = prediction, target
            return np.asarray([0.0, 1.0])

    relative_paths = (
        "camera_00.png",
        "indoor/camera_01.png",
        "indoor/camera_02.png",
        "outdoor/camera_03.png",
        "outdoor/deep/camera_04.png",
    )

    def save_rgb(root: Path, relative_path: str, array: np.ndarray) -> None:
        path = root.joinpath(*relative_path.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(path)

    def clone_case(source_roots: tuple[Path, ...], root: Path) -> tuple[Path, ...]:
        cloned = []
        for index, source in enumerate(source_roots):
            destination = root / f"dir_{index}"
            shutil.copytree(source, destination)
            cloned.append(destination)
        return tuple(cloned)

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        roots = tuple(temporary_root / name for name in ("or", "og", "cr", "cg"))
        for root in roots:
            root.mkdir(parents=True)
        rng = np.random.default_rng(1234)
        creation_order = (4, 1, 3, 0, 2)
        for index in creation_order:
            base = rng.integers(32, 224, size=(32, 32, 3), dtype=np.uint8)
            checker = ((np.indices((32, 32)).sum(axis=0) + index) % 2) * 2 - 1
            original = np.clip(
                base.astype(np.int16) + checker[:, :, None] * 2, 0, 255
            ).astype(np.uint8)
            compressed_amplitude = (index + 2) * 4
            compressed = np.clip(
                base.astype(np.int16)
                + checker[:, :, None] * compressed_amplitude,
                0,
                255,
            ).astype(np.uint8)
            save_rgb(roots[0], relative_paths[index], original)
            save_rgb(roots[1], relative_paths[index], base)
            save_rgb(roots[2], relative_paths[index], compressed)
            save_rgb(roots[3], relative_paths[index], base)

        scorer = FakeLPIPSScorer()
        collected = collect_exact_fixed_view_paths(*roots)
        require(collected == relative_paths, "relative paths must sort deterministically")
        reordered_roots = tuple(
            temporary_root / f"reordered_{index}" for index in range(4)
        )
        for root in reordered_roots:
            root.mkdir()
        for relative_path in reversed(relative_paths):
            for source_root, destination_root in zip(roots, reordered_roots):
                source = source_root.joinpath(*relative_path.split("/"))
                with Image.open(source) as source_image:
                    save_rgb(
                        destination_root,
                        relative_path,
                        np.asarray(source_image.convert("RGB")),
                    )
        require(
            collect_exact_fixed_view_paths(*reordered_roots) == relative_paths,
            "file creation order must not affect deterministic path order",
        )
        result = evaluate_fixed_view_quality(*roots, lpips_scorer=scorer)
        repeated = evaluate_fixed_view_quality(*roots, lpips_scorer=scorer)
        require(result == repeated, "repeated evaluation must be deterministic")
        require(result.view_count == 5 and len(result.per_view) == 5, "view count wrong")
        require(
            result.original["mean_psnr_db"] > result.compressed["mean_psnr_db"]
            and result.original["mean_ssim"] > result.compressed["mean_ssim"]
            and result.original["mean_lpips"] < result.compressed["mean_lpips"],
            "original metrics must outperform compressed metrics",
        )
        require(
            result.degradation["mean_psnr_drop_db"] > 0.0
            and result.degradation["mean_ssim_drop"] > 0.0
            and result.degradation["mean_lpips_increase"] > 0.0,
            "mean degradation directions are incorrect",
        )
        require(
            all(
                entry["relative_path"] == "outdoor/deep/camera_04.png"
                for entry in result.worst_views.values()
            ),
            "worst-view paths are incorrect",
        )
        require(
            result.metric_configuration["lpips_backend"] == "fake_lpips_validation",
            "fake scorer must never be labeled official",
        )

        json_path = save_fixed_view_quality_json(result, temporary_root / "metrics.json")
        csv_path = save_fixed_view_quality_csv(result, temporary_root / "metrics.csv")
        with json_path.open("r", encoding="utf-8") as file_handle:
            parsed = json.load(
                file_handle,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"nonstandard JSON constant {value}")
                ),
            )
        require(parsed["schema"] == _JSON_SCHEMA, "JSON schema is incorrect")
        json_text = json_path.read_text(encoding="utf-8")
        require(
            "NaN" not in json_text and "Infinity" not in json_text,
            "JSON contains non-standard numeric constants",
        )
        require(
            len(csv_path.read_text(encoding="utf-8").splitlines()) == 6,
            "CSV must contain one header plus five rows",
        )

        require_error(
            ValueError,
            lambda: collect_exact_fixed_view_paths(
                temporary_root / "missing", roots[1], roots[2], roots[3]
            ),
            "missing directory must fail",
        )
        missing_roots = clone_case(roots, temporary_root / "missing_case")
        missing_roots[0].joinpath(*relative_paths[0].split("/")).unlink()
        require_error(
            ValueError,
            lambda: collect_exact_fixed_view_paths(*missing_roots),
            "missing view must fail",
        )
        extra_roots = clone_case(roots, temporary_root / "extra_case")
        save_rgb(extra_roots[2], "extra.png", np.zeros((32, 32, 3), dtype=np.uint8))
        require_error(
            ValueError,
            lambda: collect_exact_fixed_view_paths(*extra_roots),
            "extra view must fail",
        )
        renamed_roots = clone_case(roots, temporary_root / "renamed_case")
        renamed_source = renamed_roots[3].joinpath(*relative_paths[1].split("/"))
        renamed_source.rename(renamed_source.with_name("different.png"))
        require_error(
            ValueError,
            lambda: collect_exact_fixed_view_paths(*renamed_roots),
            "relative path mismatch must fail",
        )
        conflict_roots = clone_case(roots, temporary_root / "conflict_case")
        save_rgb(
            conflict_roots[0],
            "CAMERA_00.PNG",
            np.zeros((32, 32, 3), dtype=np.uint8),
        )
        require_error(
            ValueError,
            lambda: collect_exact_fixed_view_paths(*conflict_roots),
            "casefold conflict must fail",
        )
        empty_roots = tuple(temporary_root / f"empty_{index}" for index in range(4))
        for root in empty_roots:
            root.mkdir()
        require_error(
            ValueError,
            lambda: collect_exact_fixed_view_paths(*empty_roots),
            "empty directories must fail",
        )
        corrupt_roots = clone_case(roots, temporary_root / "corrupt_case")
        corrupt_roots[0].joinpath(*relative_paths[0].split("/")).write_bytes(b"bad")
        require_error(
            ValueError,
            lambda: evaluate_fixed_view_quality(*corrupt_roots, lpips_scorer=scorer),
            "corrupt image must fail",
        )
        size_roots = clone_case(roots, temporary_root / "size_case")
        save_rgb(
            size_roots[2], relative_paths[0], np.zeros((31, 32, 3), dtype=np.uint8)
        )
        require_error(
            ValueError,
            lambda: evaluate_fixed_view_quality(*size_roots, lpips_scorer=scorer),
            "dimension mismatch must fail",
        )
        gt_roots = clone_case(roots, temporary_root / "gt_case")
        with Image.open(
            gt_roots[3].joinpath(*relative_paths[0].split("/"))
        ) as gt_image:
            altered_gt = np.asarray(gt_image.convert("RGB")).copy()
        altered_gt[0, 0, 0] ^= np.uint8(1)
        save_rgb(gt_roots[3], relative_paths[0], altered_gt)
        require_error(
            ValueError,
            lambda: evaluate_fixed_view_quality(*gt_roots, lpips_scorer=scorer),
            "different GT pixels must fail",
        )

        small = np.zeros((3, 10, 10), dtype=np.float32)
        require_error(
            ValueError,
            lambda: compute_ssim_rgb(small, small),
            "SSIM images smaller than 11x11 must fail",
        )
        sample = load_metric_rgb_image(roots[1].joinpath(*relative_paths[0].split("/")))
        require_error(
            ValueError,
            lambda: evaluate_single_fixed_view(
                sample, sample, sample, sample, NaNLPIPSScorer(), "nan.png"
            ),
            "NaN LPIPS must fail",
        )
        require_error(
            ValueError,
            lambda: evaluate_single_fixed_view(
                sample, sample, sample, sample, ArrayLPIPSScorer(), "array.png"
            ),
            "non-scalar LPIPS must fail",
        )

        zero = np.zeros((3, 16, 16), dtype=np.float32)
        half = np.full((3, 16, 16), 0.5, dtype=np.float32)
        require(
            math.isclose(compute_psnr_rgb(zero, half), -10.0 * math.log10(0.25)),
            "PSNR formula does not match known MSE",
        )
        require(math.isinf(compute_psnr_rgb(zero, zero)), "identical PSNR must be inf")
        identical_ssim = compute_ssim_rgb(half, half)
        error_ssim = compute_ssim_rgb(zero, half)
        require(
            math.isclose(identical_ssim, 1.0, abs_tol=1e-10)
            and error_ssim < identical_ssim,
            "SSIM identity/error behavior is incorrect",
        )
        mild = np.clip(sample + 0.01, 0.0, 1.0)
        severe = np.clip(sample + 0.08, 0.0, 1.0)
        recovery = evaluate_single_fixed_view(
            severe, sample, mild, sample, scorer, "recovery.png"
        )
        require(
            recovery.psnr_drop_db < 0.0
            and recovery.ssim_drop < 0.0
            and recovery.lpips_increase < 0.0,
            "quality recovery must retain negative degradation values",
        )

        original_resize = Image.Image.resize

        def forbidden_resize(*args: Any, **kwargs: Any) -> Any:
            _ = args, kwargs
            raise AssertionError("evaluation must never call resize")

        Image.Image.resize = forbidden_resize
        try:
            no_resize_result = evaluate_fixed_view_quality(*roots, lpips_scorer=scorer)
            require(no_resize_result.view_count == 5, "no-resize evaluation failed")
        finally:
            Image.Image.resize = original_resize
    return True


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the explicit fixed-view quality CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-render-dir", required=True)
    parser.add_argument("--original-gt-dir", required=True)
    parser.add_argument("--compressed-render-dir", required=True)
    parser.add_argument("--compressed-gt-dir", required=True)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", default="fixed_view_quality.json")
    parser.add_argument("--output-csv", default="fixed_view_quality.csv")
    return parser


def main() -> None:
    """Run official fixed-view evaluation from explicit CLI paths."""
    args = _build_cli_parser().parse_args()
    result = evaluate_fixed_view_quality(
        args.original_render_dir,
        args.original_gt_dir,
        args.compressed_render_dir,
        args.compressed_gt_dir,
        lpips_net=args.lpips_net,
        device=args.device,
    )
    json_path = save_fixed_view_quality_json(result, args.output_json)
    csv_path = save_fixed_view_quality_csv(result, args.output_csv)
    print(f"view_count={result.view_count}")
    print(
        "original mean: "
        f"PSNR={result.original['mean_psnr_db']:.6f} dB, "
        f"SSIM={result.original['mean_ssim']:.6f}, "
        f"LPIPS={result.original['mean_lpips']:.6f}"
    )
    print(
        "compressed mean: "
        f"PSNR={result.compressed['mean_psnr_db']:.6f} dB, "
        f"SSIM={result.compressed['mean_ssim']:.6f}, "
        f"LPIPS={result.compressed['mean_lpips']:.6f}"
    )
    print(
        "mean degradation: "
        f"PSNR drop={result.degradation['mean_psnr_drop_db']:.6f} dB, "
        f"SSIM drop={result.degradation['mean_ssim_drop']:.6f}, "
        f"LPIPS increase={result.degradation['mean_lpips_increase']:.6f}"
    )
    print(f"JSON={json_path}")
    print(f"CSV={csv_path}")


if __name__ == "__main__":
    main()
