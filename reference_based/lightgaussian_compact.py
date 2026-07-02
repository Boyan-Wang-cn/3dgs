"""LightGaussian-style compact package writer for RL 3DGS compression.

This module adapts the compact-storage pattern from VITA-Group/LightGaussian:
``extreme_saving`` directory, ``np.savez_compressed`` files, packed bit masks,
and a final zip file whose size is used as the real compact representation.

It does not replace rendering.  The renderer still consumes the decoded float
PLY.  The compact zip is used for model-size reward and reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import zipfile

import numpy as np

try:
    from .compression_ops import (
        decode_action,
        geo_fields,
        active_sh_fields,
        PRUNING_RATES,
        PRECISION_PROFILES,
    )
except ImportError:
    from compression_ops import (
        decode_action,
        geo_fields,
        active_sh_fields,
        PRUNING_RATES,
        PRECISION_PROFILES,
    )


def _pack_integer_array(values: np.ndarray, bits: int) -> bytes:
    """Pack non-negative integer values into a compact byte stream.

    This mirrors the LightGaussian pattern of converting integer indices into
    binary codes and calling ``np.packbits`` before saving.
    """
    values = np.asarray(values, dtype=np.uint32).reshape(-1)
    bits = int(bits)
    if values.size == 0:
        return b""
    if bits <= 0:
        raise ValueError(f"bits must be positive, got {bits}")
    if bits == 8:
        return values.astype(np.uint8).tobytes()
    if bits == 16:
        return values.astype(np.uint16).tobytes()
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint32)
    bit_matrix = ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    return np.packbits(bit_matrix.reshape(-1)).tobytes()


def _quantize(values: np.ndarray, bits: int) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(0, dtype=np.uint32), 0.0, 1.0
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return np.zeros(values.shape[0], dtype=np.uint32), vmin, 1.0
    levels = float((1 << int(bits)) - 1)
    q = np.rint((values - vmin) / (vmax - vmin) * levels)
    q = np.clip(q, 0, levels).astype(np.uint32)
    return q, vmin, vmax


def _save_groupwise_fields(
    out_file: Path,
    vertices: np.ndarray,
    group_ids: np.ndarray,
    action_ids: np.ndarray,
    field_kind: str,
) -> dict[str, Any]:
    payload_parts: list[bytes] = []
    chunks: list[dict[str, Any]] = []
    unique_groups = np.unique(group_ids.astype(np.int32)) if group_ids.size else np.zeros(0, dtype=np.int32)

    for group_id in unique_groups.tolist():
        mask = group_ids == group_id
        if not np.any(mask):
            continue
        group_action_ids = np.unique(action_ids[mask])
        action_id = int(group_action_ids[0]) if len(group_action_ids) else 0
        action = decode_action(action_id)
        if field_kind == "geo":
            fields = geo_fields(vertices)
            bits = action.geo_bit
        elif field_kind == "sh":
            fields = active_sh_fields(vertices, action.sh_degree)
            bits = action.sh_bit
        elif field_kind == "xyz":
            fields = [f for f in ["x", "y", "z"] if f in (vertices.dtype.names or [])]
            bits = action.geo_bit
        else:
            raise ValueError(f"unknown field_kind={field_kind}")

        row_indices = np.nonzero(mask)[0]
        for field in fields:
            q, vmin, vmax = _quantize(vertices[field][row_indices], bits)
            packed = _pack_integer_array(q, bits)
            offset = sum(len(part) for part in payload_parts)
            payload_parts.append(packed)
            chunks.append(
                {
                    "group_id": int(group_id),
                    "action_id": action_id,
                    "field": field,
                    "bits": int(bits),
                    "count": int(len(row_indices)),
                    "offset": int(offset),
                    "nbytes": int(len(packed)),
                    "min": float(vmin),
                    "max": float(vmax),
                }
            )

    payload = b"".join(payload_parts)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        payload=np.frombuffer(payload, dtype=np.uint8),
        chunks_json=np.array(json.dumps(chunks, ensure_ascii=False)),
        field_kind=np.array(field_kind),
    )
    return {"file": str(out_file), "payload_bytes": int(len(payload)), "num_chunks": int(len(chunks))}


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(src_dir.parent)))


def save_lightgaussian_compact_package(
    vertices: np.ndarray,
    group_ids: np.ndarray,
    action_ids: np.ndarray,
    compact_root: str | Path,
    zip_path: str | Path,
    original_vertex_count: int,
    original_size_bytes: int | None = None,
) -> dict[str, Any]:
    """Save a LightGaussian-style compact package and return size info."""
    compact_root = Path(compact_root)
    extreme_dir = compact_root / "extreme_saving"
    zip_path = Path(zip_path)
    if extreme_dir.exists():
        import shutil
        shutil.rmtree(extreme_dir)
    extreme_dir.mkdir(parents=True, exist_ok=True)

    group_ids = np.asarray(group_ids, dtype=np.int32)
    action_ids = np.asarray(action_ids, dtype=np.int16)
    if len(vertices) != len(group_ids) or len(vertices) != len(action_ids):
        raise ValueError(
            f"compact metadata length mismatch: vertices={len(vertices)} "
            f"group_ids={len(group_ids)} action_ids={len(action_ids)}"
        )

    metadata = {
        "format": "rl_lightgaussian_compact_v1",
        "adapted_from": "VITA-Group/LightGaussian/vectree",
        "input_pc_num": int(len(vertices)),
        "original_pc_num": int(original_vertex_count),
        "input_pc_dim": int(len(vertices.dtype.names or [])),
        "original_size_bytes": int(original_size_bytes or 0),
        "pruning_rates": PRUNING_RATES,
        "precision_profiles": PRECISION_PROFILES,
        "notes": (
            "Decoded float PLY is used for rendering; this compact zip is used "
            "for real compressed-size reward/reporting."
        ),
    }
    np.savez_compressed(extreme_dir / "metadata.npz", metadata=metadata)

    # LightGaussian uses packed masks and index files.  We save equivalent RL
    # action/group mappings with compact dtypes.
    np.savez_compressed(extreme_dir / "group_id.npz", group_ids.astype(np.uint16))
    np.savez_compressed(extreme_dir / "action_id.npz", action_ids.astype(np.uint8))
    np.savez_compressed(extreme_dir / "non_vq_mask.npz", np.packbits(np.ones(len(vertices), dtype=bool)))
    np.savez_compressed(extreme_dir / "vq_indexs.npz", np.zeros(0, dtype=np.uint8))
    np.savez_compressed(extreme_dir / "codebook.npz", np.zeros((0, 0), dtype=np.float16))

    xyz_info = _save_groupwise_fields(extreme_dir / "xyz.npz", vertices, group_ids, action_ids, "xyz")
    other_info = _save_groupwise_fields(extreme_dir / "other_attribute.npz", vertices, group_ids, action_ids, "geo")
    sh_info = _save_groupwise_fields(extreme_dir / "non_vq_feats.npz", vertices, group_ids, action_ids, "sh")

    summary = {
        "xyz": xyz_info,
        "other_attribute": other_info,
        "non_vq_feats": sh_info,
    }
    (extreme_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    _zip_dir(extreme_dir, zip_path)
    compact_size = int(zip_path.stat().st_size)
    size_info = {
        "compact_dir": str(compact_root),
        "compact_extreme_saving_dir": str(extreme_dir),
        "compact_package_path": str(zip_path),
        "compact_size_bytes": compact_size,
        "compact_size_mb": float(compact_size / 1024.0 / 1024.0),
        "compact_format": "rl_lightgaussian_compact_v1",
        **summary,
    }
    return size_info
