"""LightGaussian-style compact package writer for RL 3DGS compression.

This module adapts the compact-storage pattern from VITA-Group/LightGaussian:
``extreme_saving`` directory, ``np.savez_compressed`` files, packed bit masks,
and a final zip file whose size is used as the real compact representation.

It does not replace rendering.  The renderer still consumes the decoded float
PLY.  The compact zip is used for model-size reward and reporting.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
import json
import zipfile
import zlib

import numpy as np


_FACTORIZED_COMPACT_FORMAT = "rl_factorized_3dgs_compact_v2"
_FACTORIZED_REQUIRED_MEMBERS = (
    "extreme_saving/metadata.json",
    "extreme_saving/summary.json",
    "extreme_saving/group_ids.npz",
    "extreme_saving/xyz.npz",
    "extreme_saving/other_attribute.npz",
    "extreme_saving/sh_attribute.npz",
)
_FACTORIZED_REQUIRED_FILES = tuple(
    PurePosixPath(member).name for member in _FACTORIZED_REQUIRED_MEMBERS
)
_FACTORIZED_V1_PLACEHOLDERS = {
    "action_id.npz",
    "non_vq_mask.npz",
    "vq_indexs.npz",
    "codebook.npz",
}
_LIGHTGAUSSIAN_DECODER_REFERENCE = {
    "repository": "VITA-Group/LightGaussian",
    "writer": "vectree/vectree.py::dec2bin",
    "decoder": "vectree/utils.py::load_vqgaussian",
    "bin2dec": "vectree/utils.py::bin2dec",
}

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

try:
    from .compression_ops import (
        PRECISION_PROFILES_V2,
        decode_factorized_action,
        encode_factorized_action,
    )
except ImportError:
    from compression_ops import (
        PRECISION_PROFILES_V2,
        decode_factorized_action,
        encode_factorized_action,
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


def _save_groupwise_fields_v2(
    out_file: Path,
    vertices: np.ndarray,
    group_ids: np.ndarray,
    pruning_levels: np.ndarray,
    precision_levels: np.ndarray,
    field_kind: str,
) -> dict[str, Any]:
    """Write one V2 attribute payload using direct factorized configurations.

    Values below 32 bits use the legacy integer quantizer and bit packer.
    Values at 32 bits are stored directly as float32 bytes so the identity
    precision profile is never mapped through an integer quantizer.
    """
    payload_parts: list[bytes] = []
    chunks: list[dict[str, Any]] = []
    payload_offset = 0
    unique_groups = (
        np.unique(group_ids.astype(np.int64))
        if group_ids.size
        else np.zeros(0, dtype=np.int64)
    )

    for group_id in unique_groups.tolist():
        mask = group_ids == group_id
        group_pruning_levels = np.unique(pruning_levels[mask])
        group_precision_levels = np.unique(precision_levels[mask])
        if len(group_pruning_levels) != 1 or len(group_precision_levels) != 1:
            raise ValueError(
                f"group {group_id} must have one pruning and precision level"
            )
        pruning_level = int(group_pruning_levels[0])
        precision_level = int(group_precision_levels[0])
        action = decode_factorized_action((pruning_level, precision_level))
        storage_id = encode_factorized_action(
            action.pruning_level, action.precision_level
        )

        if field_kind == "xyz":
            fields = [
                field
                for field in ("x", "y", "z")
                if field in (vertices.dtype.names or ())
            ]
            bits = action.geo_bit
        elif field_kind == "other_geo":
            fields = [
                field
                for field in geo_fields(vertices)
                if field not in {"x", "y", "z"}
            ]
            bits = action.geo_bit
        elif field_kind == "sh":
            fields = active_sh_fields(vertices, action.sh_degree)
            bits = action.sh_bit
        else:
            raise ValueError(f"unknown V2 field_kind={field_kind}")

        row_indices = np.nonzero(mask)[0]
        for field in fields:
            values = vertices[field][row_indices]
            if bits >= 32:
                payload_part = np.asarray(values, dtype=np.float32).tobytes()
                encoding = "raw_float32"
                stored_bits = 32
                value_min = None
                value_max = None
            else:
                quantized, value_min, value_max = _quantize(values, bits)
                payload_part = _pack_integer_array(quantized, bits)
                encoding = "quantized_uint"
                stored_bits = int(bits)

            nbytes = len(payload_part)
            payload_parts.append(payload_part)
            chunks.append(
                {
                    "group_id": int(group_id),
                    "field": field,
                    "pruning_level": action.pruning_level,
                    "precision_level": action.precision_level,
                    "storage_id": int(storage_id),
                    "encoding": encoding,
                    "bits": stored_bits,
                    "count": int(len(row_indices)),
                    "offset": int(payload_offset),
                    "nbytes": int(nbytes),
                    "min": None if value_min is None else float(value_min),
                    "max": None if value_max is None else float(value_max),
                }
            )
            payload_offset += nbytes

    payload = b"".join(payload_parts)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        payload=np.frombuffer(payload, dtype=np.uint8),
        chunks_json=np.array(json.dumps(chunks, ensure_ascii=False)),
        field_kind=np.array(field_kind),
    )
    return {
        "file": str(out_file),
        "payload_bytes": int(len(payload)),
        "num_chunks": int(len(chunks)),
    }


def _validated_integer_vector_v2(
    values: np.ndarray,
    name: str,
    expected_length: int,
) -> np.ndarray:
    """Validate and copy a one-dimensional integer metadata vector."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if len(array) != expected_length:
        raise ValueError(
            f"{name} length must equal vertices length {expected_length}, "
            f"got {len(array)}"
        )
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise ValueError(f"{name} must contain integers")
    return array.astype(np.int64, copy=True)


def _validate_nonnegative_integer_v2(value: int, name: str) -> int:
    """Validate a strict nonnegative integer package parameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a nonnegative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return normalized


def _validate_factorized_compact_inputs(
    vertices: np.ndarray,
    group_ids: np.ndarray,
    pruning_levels: np.ndarray,
    precision_levels: np.ndarray,
    storage_ids: np.ndarray,
    original_vertex_count: int,
    original_size_bytes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Validate all V2 writer inputs and their per-group consistency."""
    if not isinstance(vertices, np.ndarray) or vertices.dtype.names is None:
        raise ValueError("vertices must be a structured NumPy array")
    vertex_count = int(len(vertices))
    normalized_group_ids = _validated_integer_vector_v2(
        group_ids, "group_ids", vertex_count
    )
    normalized_pruning = _validated_integer_vector_v2(
        pruning_levels, "pruning_levels", vertex_count
    )
    normalized_precision = _validated_integer_vector_v2(
        precision_levels, "precision_levels", vertex_count
    )
    normalized_storage = _validated_integer_vector_v2(
        storage_ids, "storage_ids", vertex_count
    )

    if np.any(normalized_group_ids < -1):
        raise ValueError("group_ids must be -1 or nonnegative")
    if np.any((normalized_pruning < 0) | (normalized_pruning > 4)):
        raise ValueError("pruning_levels must be in [0, 4]")
    if np.any((normalized_precision < 0) | (normalized_precision > 5)):
        raise ValueError("precision_levels must be in [0, 5]")
    if np.any((normalized_storage < 0) | (normalized_storage > 29)):
        raise ValueError("storage_ids must be in [0, 29]")

    expected_storage = np.asarray(
        [
            encode_factorized_action(pruning_level, precision_level)
            for pruning_level, precision_level in zip(
                normalized_pruning, normalized_precision
            )
        ],
        dtype=np.int64,
    )
    if not np.array_equal(normalized_storage, expected_storage):
        raise ValueError(
            "storage_ids are inconsistent with pruning_levels and "
            "precision_levels"
        )

    ungrouped_mask = normalized_group_ids == -1
    if np.any(
        ungrouped_mask
        & (
            (normalized_pruning != 0)
            | (normalized_precision != 0)
            | (normalized_storage != 0)
        )
    ):
        raise ValueError("group_id=-1 must use identity action (0, 0)")

    for group_id in np.unique(normalized_group_ids).tolist():
        mask = normalized_group_ids == group_id
        if (
            len(np.unique(normalized_pruning[mask])) != 1
            or len(np.unique(normalized_precision[mask])) != 1
        ):
            raise ValueError(
                f"group {group_id} must use one pruning and precision level"
            )

    normalized_original_count = _validate_nonnegative_integer_v2(
        original_vertex_count, "original_vertex_count"
    )
    normalized_original_size = _validate_nonnegative_integer_v2(
        original_size_bytes, "original_size_bytes"
    )
    if normalized_original_size <= 0:
        raise ValueError("original_size_bytes must be a positive integer")
    return (
        normalized_group_ids,
        normalized_pruning,
        normalized_precision,
        normalized_storage,
        normalized_original_count,
        normalized_original_size,
    )


def _factorized_group_table(
    group_ids: np.ndarray,
    pruning_levels: np.ndarray,
    precision_levels: np.ndarray,
) -> list[dict[str, Any]]:
    """Build decoder-facing group configuration metadata."""
    table: list[dict[str, Any]] = []
    for group_id in np.unique(group_ids).tolist():
        mask = group_ids == group_id
        pruning_level = int(pruning_levels[mask][0])
        precision_level = int(precision_levels[mask][0])
        action = decode_factorized_action((pruning_level, precision_level))
        table.append(
            {
                "group_id": int(group_id),
                "vertex_count": int(np.count_nonzero(mask)),
                "pruning_level": action.pruning_level,
                "precision_level": action.precision_level,
                "storage_id": encode_factorized_action(
                    action.pruning_level, action.precision_level
                ),
                "pruning_rate": action.pruning_rate,
                "sh_degree": action.sh_degree,
                "sh_bit": action.sh_bit,
                "geo_bit": action.geo_bit,
            }
        )
    return table


def save_factorized_lightgaussian_compact_package(
    vertices: np.ndarray,
    group_ids: np.ndarray,
    pruning_levels: np.ndarray,
    precision_levels: np.ndarray,
    storage_ids: np.ndarray,
    compact_root: str | Path,
    zip_path: str | Path,
    original_vertex_count: int,
    original_size_bytes: int,
) -> dict[str, Any]:
    """Write a directly factorized, independently decodable V2 compact zip."""
    (
        normalized_group_ids,
        normalized_pruning,
        normalized_precision,
        _normalized_storage,
        normalized_original_count,
        normalized_original_size,
    ) = _validate_factorized_compact_inputs(
        vertices,
        group_ids,
        pruning_levels,
        precision_levels,
        storage_ids,
        original_vertex_count,
        original_size_bytes,
    )

    compact_root_path = Path(compact_root)
    extreme_dir = compact_root_path / "extreme_saving"
    zip_path_object = Path(zip_path)
    if extreme_dir.exists():
        import shutil

        shutil.rmtree(extreme_dir)
    extreme_dir.mkdir(parents=True, exist_ok=True)

    xyz_info = _save_groupwise_fields_v2(
        extreme_dir / "xyz.npz",
        vertices,
        normalized_group_ids,
        normalized_pruning,
        normalized_precision,
        "xyz",
    )
    other_info = _save_groupwise_fields_v2(
        extreme_dir / "other_attribute.npz",
        vertices,
        normalized_group_ids,
        normalized_pruning,
        normalized_precision,
        "other_geo",
    )
    sh_info = _save_groupwise_fields_v2(
        extreme_dir / "sh_attribute.npz",
        vertices,
        normalized_group_ids,
        normalized_pruning,
        normalized_precision,
        "sh",
    )

    np.savez_compressed(
        extreme_dir / "group_ids.npz",
        group_ids=normalized_group_ids.astype(np.int32),
    )
    group_table = _factorized_group_table(
        normalized_group_ids, normalized_pruning, normalized_precision
    )
    field_order = list(vertices.dtype.names or ())
    vertex_dtype = [
        [field, vertices.dtype.fields[field][0].str] for field in field_order
    ]
    attribute_files = {
        "xyz": "xyz.npz",
        "other_attribute": "other_attribute.npz",
        "sh_attribute": "sh_attribute.npz",
        "group_ids": "group_ids.npz",
    }
    metadata = {
        "format": "rl_factorized_3dgs_compact_v2",
        "original_vertex_count": normalized_original_count,
        "compressed_vertex_count": int(len(vertices)),
        "original_size_bytes": normalized_original_size,
        "vertex_field_order": field_order,
        "vertex_dtype": vertex_dtype,
        "pruning_rates": PRUNING_RATES,
        "precision_profiles_v2": PRECISION_PROFILES_V2,
        "group_table": group_table,
        "attribute_files": attribute_files,
        "notes": [
            "storage_id is categorical metadata only; compression settings "
            "come directly from pruning_level and precision_level.",
            "Fields above each group's active SH degree are omitted and a "
            "decoder must restore them as zeros using the original schema.",
            "32-bit chunks contain raw float32 bytes; lower-bit chunks use "
            "per-chunk uniform integer quantization.",
        ],
    }
    (extreme_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    raw_payload_bytes = int(
        xyz_info["payload_bytes"]
        + other_info["payload_bytes"]
        + sh_info["payload_bytes"]
    )
    summary = {
        "xyz": xyz_info,
        "other_attribute": other_info,
        "sh_attribute": sh_info,
        "raw_attribute_payload_bytes": raw_payload_bytes,
    }
    (extreme_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _zip_dir(extreme_dir, zip_path_object)
    compact_size = int(zip_path_object.stat().st_size)
    attribute_members = {"xyz.npz", "other_attribute.npz", "sh_attribute.npz"}
    with zipfile.ZipFile(zip_path_object, "r") as zip_file:
        zip_infos = zip_file.infolist()
        member_uncompressed_bytes = int(
            sum(info.file_size for info in zip_infos)
        )
        member_compressed_bytes = int(
            sum(info.compress_size for info in zip_infos)
        )
        attribute_member_compressed_bytes = int(
            sum(
                info.compress_size
                for info in zip_infos
                if Path(info.filename).name in attribute_members
            )
        )
    metadata_aux_member_compressed_bytes = int(
        member_compressed_bytes - attribute_member_compressed_bytes
    )
    zip_container_overhead_bytes = int(
        compact_size - member_compressed_bytes
    )
    if (
        metadata_aux_member_compressed_bytes < 0
        or zip_container_overhead_bytes < 0
    ):
        raise RuntimeError("computed compact package overhead cannot be negative")
    format_overhead_ratio = float(
        (
            metadata_aux_member_compressed_bytes
            + zip_container_overhead_bytes
        )
        / max(compact_size, 1)
    )

    return {
        "compact_dir": str(compact_root_path),
        "compact_extreme_saving_dir": str(extreme_dir),
        "compact_package_path": str(zip_path_object),
        "compact_size_bytes": compact_size,
        "compact_size_mb": float(compact_size / 1024.0 / 1024.0),
        "compact_format": "rl_factorized_3dgs_compact_v2",
        "raw_attribute_payload_bytes": raw_payload_bytes,
        "compressed_vertex_count": int(len(vertices)),
        "original_vertex_count": normalized_original_count,
        "original_size_bytes": normalized_original_size,
        "compact_size_ratio": float(compact_size / normalized_original_size),
        "zip_member_uncompressed_bytes": member_uncompressed_bytes,
        "zip_member_compressed_bytes": member_compressed_bytes,
        "attribute_member_compressed_bytes": attribute_member_compressed_bytes,
        "metadata_aux_member_compressed_bytes": (
            metadata_aux_member_compressed_bytes
        ),
        "zip_container_overhead_bytes": zip_container_overhead_bytes,
        "format_overhead_ratio": format_overhead_ratio,
        **summary,
    }


def _strict_json_loads(payload: bytes | str, name: str) -> Any:
    """Parse strict UTF-8 JSON while rejecting duplicate keys and NaN values."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains non-standard numeric constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"{name} is not valid strict UTF-8 JSON") from exc


def _strict_integer(value: Any, name: str, minimum: int | None = None) -> int:
    """Validate a strict integer without accepting booleans or floats."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a strict integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _strict_finite_number(value: Any, name: str) -> float:
    """Validate a finite real JSON number, excluding booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_zip_member_name(name: str) -> None:
    """Reject unsafe or ambiguous ZIP member paths."""
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"unsafe ZIP member name {name!r}")
    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
        or any(part in {"", "."} for part in posix_path.parts)
    ):
        raise ValueError(f"unsafe ZIP member path {name!r}")


def _read_factorized_package_files(
    package_path: str | Path,
) -> tuple[dict[str, bytes], str, Path]:
    """Read the exact six V2 members from a ZIP or safe local directory."""
    path = Path(package_path)
    if path.is_file():
        try:
            with zipfile.ZipFile(path, "r") as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                for name in names:
                    _validate_zip_member_name(name)
                if len(names) != len(set(names)):
                    raise ValueError("V2 ZIP contains duplicate member names")
                actual = set(names)
                required = set(_FACTORIZED_REQUIRED_MEMBERS)
                forbidden = {
                    name
                    for name in actual
                    if PurePosixPath(name).name in _FACTORIZED_V1_PLACEHOLDERS
                }
                if forbidden:
                    raise ValueError("V2 ZIP contains legacy V1 placeholder files")
                missing = required - actual
                unknown = actual - required
                if missing:
                    raise ValueError(f"V2 ZIP is missing members: {sorted(missing)}")
                if unknown:
                    raise ValueError(f"V2 ZIP contains unknown members: {sorted(unknown)}")
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise ValueError(f"V2 ZIP CRC failed for {bad_member!r}")
                files = {
                    PurePosixPath(name).name: archive.read(name)
                    for name in _FACTORIZED_REQUIRED_MEMBERS
                }
        except ValueError:
            raise
        except (OSError, EOFError, RuntimeError, zipfile.BadZipFile, zlib.error) as exc:
            raise ValueError(f"invalid or truncated V2 ZIP: {exc}") from exc
        return files, "zip", path

    if not path.is_dir():
        raise ValueError("package_path must be a V2 ZIP or package directory")
    if (path / "metadata.json").exists():
        extreme_dir = path
    elif (path / "extreme_saving").is_dir():
        extreme_dir = path / "extreme_saving"
    else:
        raise ValueError(
            "directory must contain metadata.json or an extreme_saving directory"
        )
    if extreme_dir.is_symlink():
        raise ValueError("extreme_saving directory must not be a symbolic link")
    entries = list(extreme_dir.iterdir())
    actual_names = {entry.name for entry in entries}
    required_names = set(_FACTORIZED_REQUIRED_FILES)
    if actual_names.intersection(_FACTORIZED_V1_PLACEHOLDERS):
        raise ValueError("V2 directory contains legacy V1 placeholder files")
    missing = required_names - actual_names
    unknown = actual_names - required_names
    if missing:
        raise ValueError(f"V2 directory is missing files: {sorted(missing)}")
    if unknown:
        raise ValueError(f"V2 directory contains unknown files: {sorted(unknown)}")
    base = extreme_dir.resolve(strict=True)
    files: dict[str, bytes] = {}
    for filename in _FACTORIZED_REQUIRED_FILES:
        member = extreme_dir / filename
        if member.is_symlink() or not member.is_file():
            raise ValueError(f"V2 member {filename!r} must be a regular file")
        try:
            resolved = member.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"unable to resolve V2 member {filename!r}") from exc
        if resolved.parent != base:
            raise ValueError(f"V2 member {filename!r} escapes the package directory")
        try:
            files[filename] = member.read_bytes()
        except OSError as exc:
            raise ValueError(f"unable to read V2 member {filename!r}") from exc
    return files, "directory", path


def _validate_factorized_metadata(
    metadata: Any,
) -> tuple[np.dtype, int, int, int, dict[int, dict[str, Any]]]:
    """Validate the complete decoder-relevant V2 metadata schema."""
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain a JSON object")
    if metadata.get("format") != _FACTORIZED_COMPACT_FORMAT:
        raise ValueError("metadata format is not rl_factorized_3dgs_compact_v2")
    compressed_count = _strict_integer(
        metadata.get("compressed_vertex_count"),
        "compressed_vertex_count",
        0,
    )
    original_count = _strict_integer(
        metadata.get("original_vertex_count"), "original_vertex_count", 1
    )
    original_size = _strict_integer(
        metadata.get("original_size_bytes"), "original_size_bytes", 1
    )
    if original_count < compressed_count:
        raise ValueError("original_vertex_count cannot be smaller than compressed count")

    field_order = metadata.get("vertex_field_order")
    dtype_entries = metadata.get("vertex_dtype")
    if not isinstance(field_order, list) or not field_order:
        raise ValueError("vertex_field_order must be a nonempty list")
    if not isinstance(dtype_entries, list) or len(dtype_entries) != len(field_order):
        raise ValueError("vertex_dtype must align exactly with vertex_field_order")
    if len(set(field_order)) != len(field_order):
        raise ValueError("vertex_field_order contains duplicate field names")
    safe_fields: list[tuple[str, np.dtype]] = []
    for index, field_name in enumerate(field_order):
        if (
            not isinstance(field_name, str)
            or not field_name
            or any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in field_name)
        ):
            raise ValueError("vertex_field_order contains an unsafe field name")
        entry = dtype_entries[index]
        if not isinstance(entry, list) or len(entry) != 2 or entry[0] != field_name:
            raise ValueError("vertex_dtype order or field name does not match metadata")
        if not isinstance(entry[1], str) or not entry[1]:
            raise ValueError(f"dtype for field {field_name!r} must be a string")
        try:
            scalar_dtype = np.dtype(entry[1])
        except TypeError as exc:
            raise ValueError(f"invalid dtype for field {field_name!r}") from exc
        if (
            scalar_dtype.fields is not None
            or scalar_dtype.subdtype is not None
            or scalar_dtype.kind not in {"f", "i", "u"}
            or scalar_dtype.itemsize <= 0
            or scalar_dtype.itemsize > 16
        ):
            raise ValueError(f"unsafe scalar numeric dtype for field {field_name!r}")
        safe_fields.append((field_name, scalar_dtype))
    reconstructed_dtype = np.dtype(safe_fields)
    if tuple(reconstructed_dtype.names or ()) != tuple(field_order):
        raise ValueError("reconstructed dtype field order does not match metadata")

    expected_attribute_files = {
        "xyz": "xyz.npz",
        "other_attribute": "other_attribute.npz",
        "sh_attribute": "sh_attribute.npz",
        "group_ids": "group_ids.npz",
    }
    attribute_files = metadata.get("attribute_files")
    if not isinstance(attribute_files, dict) or set(attribute_files) != set(
        expected_attribute_files
    ):
        raise ValueError("metadata attribute_files has an invalid key set")
    for key, filename in attribute_files.items():
        if (
            not isinstance(filename, str)
            or not filename
            or filename != PurePosixPath(filename).name
            or filename != PureWindowsPath(filename).name
            or "/" in filename
            or "\\" in filename
            or filename in {".", ".."}
            or filename != expected_attribute_files[key]
        ):
            raise ValueError(f"unsafe metadata attribute_files entry for {key!r}")

    group_table = metadata.get("group_table")
    if not isinstance(group_table, list):
        raise ValueError("group_table must be a list")
    groups: dict[int, dict[str, Any]] = {}
    expected_group_keys = {
        "group_id",
        "vertex_count",
        "pruning_level",
        "precision_level",
        "storage_id",
        "pruning_rate",
        "sh_degree",
        "sh_bit",
        "geo_bit",
    }
    for index, entry in enumerate(group_table):
        if not isinstance(entry, dict) or set(entry) != expected_group_keys:
            raise ValueError(f"group_table[{index}] has an invalid schema")
        group_id = _strict_integer(entry["group_id"], f"group_table[{index}].group_id")
        if group_id < -1 or group_id in groups:
            raise ValueError("group_table group IDs must be unique and >= -1")
        vertex_count = _strict_integer(
            entry["vertex_count"], f"group_table[{index}].vertex_count", 1
        )
        pruning_level = _strict_integer(
            entry["pruning_level"], f"group_table[{index}].pruning_level"
        )
        precision_level = _strict_integer(
            entry["precision_level"], f"group_table[{index}].precision_level"
        )
        storage_id = _strict_integer(
            entry["storage_id"], f"group_table[{index}].storage_id"
        )
        if not 0 <= pruning_level <= 4 or not 0 <= precision_level <= 5:
            raise ValueError("group_table action levels are out of range")
        if not 0 <= storage_id <= 29:
            raise ValueError("group_table storage_id is out of range")
        action = decode_factorized_action((pruning_level, precision_level))
        expected_storage = encode_factorized_action(pruning_level, precision_level)
        if storage_id != expected_storage:
            raise ValueError("group_table storage_id is inconsistent with action levels")
        if group_id == -1 and (pruning_level, precision_level, storage_id) != (0, 0, 0):
            raise ValueError("group_id=-1 must use the identity action")
        if (
            _strict_finite_number(entry["pruning_rate"], "pruning_rate")
            != action.pruning_rate
            or _strict_integer(entry["sh_degree"], "sh_degree") != action.sh_degree
            or _strict_integer(entry["sh_bit"], "sh_bit") != action.sh_bit
            or _strict_integer(entry["geo_bit"], "geo_bit") != action.geo_bit
        ):
            raise ValueError("group_table compression profile is inconsistent")
        groups[group_id] = {
            **entry,
            "vertex_count": vertex_count,
            "pruning_level": pruning_level,
            "precision_level": precision_level,
            "storage_id": storage_id,
        }
    if compressed_count == 0 and groups:
        raise ValueError("empty vertices cannot have group_table entries")
    if compressed_count > 0 and not groups:
        raise ValueError("nonempty vertices require group_table entries")
    return reconstructed_dtype, compressed_count, original_count, original_size, groups


def _load_npz_arrays(payload: bytes, name: str) -> dict[str, np.ndarray]:
    """Load an NPZ member without pickle and copy arrays out of its buffer."""
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]).copy() for key in archive.files}
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{name} is not a safe valid NPZ file") from exc


def _read_factorized_group_ids(
    payload: bytes,
    compressed_count: int,
    groups: dict[int, dict[str, Any]],
) -> np.ndarray:
    """Restore signed group IDs and cross-check their complete group table."""
    arrays = _load_npz_arrays(payload, "group_ids.npz")
    if set(arrays) != {"group_ids"}:
        raise ValueError("group_ids.npz must contain only the group_ids key")
    group_ids = arrays["group_ids"]
    if group_ids.ndim != 1 or not np.issubdtype(group_ids.dtype, np.signedinteger):
        raise ValueError("group_ids must be a one-dimensional signed integer array")
    if len(group_ids) != compressed_count or np.any(group_ids < -1):
        raise ValueError("group_ids length or values are invalid")
    normalized = group_ids.astype(np.int64, copy=True)
    actual_groups = set(int(value) for value in np.unique(normalized))
    if actual_groups != set(groups):
        raise ValueError("group_ids unique values do not match group_table")
    for group_id, entry in groups.items():
        if int(np.count_nonzero(normalized == group_id)) != entry["vertex_count"]:
            raise ValueError(f"group {group_id} vertex_count does not match group_ids")
    return normalized


def _unpack_integer_array(payload: bytes, bits: int, count: int) -> np.ndarray:
    """Strictly invert the V2 writer's MSB-first integer packing."""
    bit_count = _strict_integer(bits, "bits", 1)
    value_count = _strict_integer(count, "count", 0)
    if bit_count > 31:
        raise ValueError("bits must be in [1, 31]")
    expected_nbytes = (
        value_count
        if bit_count == 8
        else value_count * 2
        if bit_count == 16
        else (value_count * bit_count + 7) // 8
    )
    if len(payload) != expected_nbytes:
        raise ValueError(
            f"packed integer payload length must be {expected_nbytes}, got {len(payload)}"
        )
    if value_count == 0:
        return np.zeros(0, dtype=np.uint32)
    if bit_count == 8:
        values = np.frombuffer(payload, dtype=np.uint8).astype(np.uint32)
    elif bit_count == 16:
        values = np.frombuffer(payload, dtype=np.uint16).astype(np.uint32)
    else:
        unpacked = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        used = value_count * bit_count
        if np.any(unpacked[used:] != 0):
            raise ValueError("packed integer payload has nonzero padding bits")
        matrix = unpacked[:used].reshape(value_count, bit_count).astype(np.uint32)
        shifts = np.arange(bit_count - 1, -1, -1, dtype=np.uint32)
        values = np.sum(matrix << shifts[None, :], axis=1, dtype=np.uint32)
    if np.any(values > (1 << bit_count) - 1):
        raise ValueError("packed integer exceeds its declared bit width")
    return values.copy()


def _lightgaussian_reference_bin2dec_numpy(
    payload: bytes,
    bits: int,
    count: int,
) -> np.ndarray:
    """Independently reproduce LightGaussian's MSB-first ``bin2dec`` idea.

    This validation-only function intentionally does not call the formal V2
    unpacker and does not validate padding bits. The V2 16-bit direct-byte path
    is a historical writer convention rather than LightGaussian's packed-index
    format, so its byte-order cross-check is handled separately by
    :func:`validate_lightgaussian_bitstream_compatibility`.
    """
    if isinstance(bits, (bool, np.bool_)) or not isinstance(
        bits, (int, np.integer)
    ):
        raise ValueError("bits must be a strict integer in [1, 31]")
    bit_width = int(bits)
    if not 1 <= bit_width <= 31:
        raise ValueError("bits must be a strict integer in [1, 31]")
    if isinstance(count, (bool, np.bool_)) or not isinstance(
        count, (int, np.integer)
    ):
        raise ValueError("count must be a nonnegative strict integer")
    value_count = int(count)
    if value_count < 0:
        raise ValueError("count must be a nonnegative strict integer")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("payload must be a bytes-like object")
    required_bits = value_count * bit_width
    required_bytes = (required_bits + 7) // 8
    if len(payload) < required_bytes:
        raise ValueError("payload is too short for count * bits")
    if value_count == 0:
        return np.zeros(0, dtype=np.uint64)
    unpacked = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    used_bits = unpacked[:required_bits]
    bit_matrix = used_bits.reshape(value_count, bit_width)
    weights = 2 ** np.arange(
        bit_width - 1,
        -1,
        -1,
        dtype=np.uint64,
    )
    decoded = np.sum(
        bit_matrix.astype(np.uint64) * weights[None, :],
        axis=1,
        dtype=np.uint64,
    )
    return decoded.reshape(-1)


def validate_lightgaussian_bitstream_compatibility() -> bool:
    """Cross-check V2 integer streams against an independent LG-style decoder.

    Bits other than 8 and 16 use the same MSB-first packed-bit mathematical
    convention as LightGaussian's ``dec2bin``/``bin2dec`` flow. Eight-bit V2
    values are direct bytes, which remain equivalent to one MSB-first byte per
    value. Sixteen-bit V2 values use the current writer's native ``uint16``
    direct-byte convention; that is not LightGaussian's packbits index format,
    so restored uint16 values are independently expanded into an MSB bit matrix
    before the reference weighted sum.
    """

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    def require_value_error(callback: Any, message: str) -> None:
        try:
            callback()
        except ValueError:
            return
        raise AssertionError(message)

    rng = np.random.default_rng(2026)
    tested_bits = (1, 2, 4, 6, 8, 10, 12, 16, 24, 31)
    for bits in tested_bits:
        maximum = (1 << bits) - 1
        data_sets = [
            np.zeros(0, dtype=np.uint32),
            np.asarray([0], dtype=np.uint32),
            np.asarray([maximum], dtype=np.uint32),
            np.asarray([0, 1, maximum // 2, maximum], dtype=np.uint32),
            rng.integers(0, maximum + 1, size=257, dtype=np.uint32),
        ]
        if bits == 8:
            data_sets.append(
                np.asarray([0, 1, 127, 128, 254, 255], dtype=np.uint32)
            )
        elif bits == 16:
            data_sets.append(
                np.asarray(
                    [0, 1, 255, 256, 32767, 32768, 65534, 65535],
                    dtype=np.uint32,
                )
            )

        for original in data_sets:
            packed = _pack_integer_array(original, bits)
            formal = _unpack_integer_array(packed, bits, len(original))
            if bits == 16:
                require(
                    packed == original.astype(np.uint16).tobytes(),
                    "16-bit V2 stream must use direct native uint16 bytes",
                )
                restored_uint16 = np.frombuffer(packed, dtype=np.uint16).astype(
                    np.uint32
                )
                shifts = np.arange(15, -1, -1, dtype=np.uint32)
                bit_matrix = (
                    restored_uint16[:, None] >> shifts[None, :]
                ) & np.uint32(1)
                weights = 2 ** shifts.astype(np.uint64)
                reference = np.sum(
                    bit_matrix.astype(np.uint64) * weights[None, :],
                    axis=1,
                    dtype=np.uint64,
                )
            else:
                if bits == 8:
                    require(
                        packed == original.astype(np.uint8).tobytes(),
                        "8-bit V2 stream must use direct uint8 bytes",
                    )
                reference = _lightgaussian_reference_bin2dec_numpy(
                    packed, bits, len(original)
                )
            require(
                np.array_equal(original, formal)
                and np.array_equal(original.astype(np.uint64), reference),
                f"formal and LightGaussian-style decoders disagree for {bits} bits",
            )

            if bits not in {8, 16}:
                expected_length = (len(original) * bits + 7) // 8
                require(
                    len(packed) == expected_length,
                    f"packed length is incorrect for {bits} bits",
                )
                used_bits = len(original) * bits
                if packed and used_bits % 8:
                    unpacked = np.unpackbits(
                        np.frombuffer(packed, dtype=np.uint8)
                    )
                    require(
                        np.all(unpacked[used_bits:] == 0),
                        "writer padding bits must be zero",
                    )
                    tampered = bytearray(packed)
                    tampered[-1] |= 1
                    require_value_error(
                        lambda data=bytes(tampered), width=bits, size=len(original): (
                            _unpack_integer_array(data, width, size)
                        ),
                        "formal decoder must reject nonzero padding bits",
                    )
                    reference_with_bad_padding = (
                        _lightgaussian_reference_bin2dec_numpy(
                            bytes(tampered), bits, len(original)
                        )
                    )
                    require(
                        np.array_equal(
                            original.astype(np.uint64), reference_with_bad_padding
                        ),
                        "reference decoder must ignore padding validation",
                    )

    require_value_error(
        lambda: _lightgaussian_reference_bin2dec_numpy(b"", 0, 0),
        "reference decoder must reject bits=0",
    )
    require_value_error(
        lambda: _lightgaussian_reference_bin2dec_numpy(b"", 32, 0),
        "reference decoder must reject bits=32",
    )
    require_value_error(
        lambda: _lightgaussian_reference_bin2dec_numpy(b"", 1, -1),
        "reference decoder must reject negative count",
    )
    require_value_error(
        lambda: _lightgaussian_reference_bin2dec_numpy(b"", 12, 1),
        "reference decoder must reject a short payload",
    )
    return True


def _decode_factorized_attribute_file(
    payload: bytes,
    filename: str,
    expected_kind: str,
    vertices: np.ndarray,
    group_ids: np.ndarray,
    groups: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Decode one strict attribute NPZ directly into reconstructed vertices."""
    arrays = _load_npz_arrays(payload, filename)
    if set(arrays) != {"payload", "chunks_json", "field_kind"}:
        raise ValueError(f"{filename} has an invalid NPZ key set")
    byte_array = arrays["payload"]
    if byte_array.ndim != 1 or byte_array.dtype != np.dtype(np.uint8):
        raise ValueError(f"{filename} payload must be one-dimensional uint8")
    raw_payload = byte_array.tobytes()
    chunks_scalar = arrays["chunks_json"]
    kind_scalar = arrays["field_kind"]
    if chunks_scalar.shape != () or chunks_scalar.dtype.kind != "U":
        raise ValueError(f"{filename} chunks_json must be a scalar string")
    if kind_scalar.shape != () or kind_scalar.dtype.kind != "U":
        raise ValueError(f"{filename} field_kind must be a scalar string")
    if str(kind_scalar.item()) != expected_kind:
        raise ValueError(f"{filename} field_kind must be {expected_kind!r}")
    chunks = _strict_json_loads(str(chunks_scalar.item()), f"{filename}.chunks_json")
    if not isinstance(chunks, list):
        raise ValueError(f"{filename} chunks_json must contain a list")

    expected_chunk_keys = {
        "group_id",
        "field",
        "pruning_level",
        "precision_level",
        "storage_id",
        "encoding",
        "bits",
        "count",
        "offset",
        "nbytes",
        "min",
        "max",
    }
    expected_offset = 0
    seen: set[tuple[int, str]] = set()
    actual_fields: set[tuple[int, str]] = set()
    schema_fields = set(vertices.dtype.names or ())
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or set(chunk) != expected_chunk_keys:
            raise ValueError(f"{filename} chunk {index} has an invalid schema")
        group_id = _strict_integer(chunk["group_id"], f"{filename}.group_id")
        if group_id not in groups:
            raise ValueError(f"{filename} chunk references unknown group {group_id}")
        field = chunk["field"]
        if not isinstance(field, str) or field not in schema_fields:
            raise ValueError(f"{filename} chunk references unknown field {field!r}")
        key = (group_id, field)
        if key in seen:
            raise ValueError(f"{filename} duplicates group-field chunk {key!r}")
        seen.add(key)
        actual_fields.add(key)
        entry = groups[group_id]
        pruning_level = _strict_integer(chunk["pruning_level"], "chunk pruning_level")
        precision_level = _strict_integer(chunk["precision_level"], "chunk precision_level")
        storage_id = _strict_integer(chunk["storage_id"], "chunk storage_id")
        if (
            pruning_level != entry["pruning_level"]
            or precision_level != entry["precision_level"]
            or storage_id != entry["storage_id"]
        ):
            raise ValueError(f"{filename} chunk action metadata is inconsistent")
        count = _strict_integer(chunk["count"], "chunk count", 0)
        offset = _strict_integer(chunk["offset"], "chunk offset", 0)
        nbytes = _strict_integer(chunk["nbytes"], "chunk nbytes", 0)
        bits = _strict_integer(chunk["bits"], "chunk bits", 1)
        if count != entry["vertex_count"]:
            raise ValueError(f"{filename} chunk count does not match group vertex_count")
        if offset != expected_offset or offset + nbytes > len(raw_payload):
            raise ValueError(f"{filename} chunk offsets are not strictly contiguous")
        payload_slice = raw_payload[offset : offset + nbytes]
        expected_offset += nbytes
        encoding = chunk["encoding"]
        if encoding == "raw_float32":
            if bits != 32 or chunk["min"] is not None or chunk["max"] is not None:
                raise ValueError("raw_float32 chunk metadata is invalid")
            if nbytes != count * 4:
                raise ValueError("raw_float32 chunk byte count is invalid")
            values = np.frombuffer(payload_slice, dtype=np.float32).copy()
            if len(values) != count or not np.all(np.isfinite(values)):
                raise ValueError("raw_float32 chunk contains invalid values")
        elif encoding == "quantized_uint":
            if not 1 <= bits <= 31:
                raise ValueError("quantized_uint bits must be in [1, 31]")
            quantized = _unpack_integer_array(payload_slice, bits, count)
            value_min = _strict_finite_number(chunk["min"], "chunk min")
            value_max = _strict_finite_number(chunk["max"], "chunk max")
            legacy_constant = (
                value_max == 1.0
                and value_min > value_max
                and np.all(quantized == 0)
            )
            if value_max < value_min and not legacy_constant:
                raise ValueError("quantized chunk max must be >= min")
            levels = float((1 << bits) - 1)
            if legacy_constant:
                values = np.full(count, value_min, dtype=np.float64)
            else:
                values = value_min + quantized.astype(np.float64) / levels * (
                    value_max - value_min
                )
            if not np.all(np.isfinite(values)):
                raise ValueError("dequantized chunk contains non-finite values")
        else:
            raise ValueError(f"unsupported chunk encoding {encoding!r}")
        rows = np.nonzero(group_ids == group_id)[0]
        target_dtype = vertices.dtype.fields[field][0]
        try:
            converted = np.asarray(values).astype(target_dtype, casting="unsafe")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"cannot convert decoded field {field!r}") from exc
        if np.issubdtype(target_dtype, np.floating) and not np.all(np.isfinite(converted)):
            raise ValueError(f"converted field {field!r} is non-finite")
        vertices[field][rows] = converted
    if expected_offset != len(raw_payload):
        raise ValueError(f"{filename} chunk metadata does not cover its payload")

    empty_vertices = np.zeros(len(vertices), dtype=vertices.dtype)
    expected_fields: set[tuple[int, str]] = set()
    for group_id, entry in groups.items():
        action = decode_factorized_action(
            (entry["pruning_level"], entry["precision_level"])
        )
        if expected_kind == "xyz":
            fields = [name for name in ("x", "y", "z") if name in schema_fields]
        elif expected_kind == "other_geo":
            fields = [
                name
                for name in geo_fields(empty_vertices)
                if name not in {"x", "y", "z"}
            ]
        elif expected_kind == "sh":
            fields = active_sh_fields(empty_vertices, action.sh_degree)
        else:
            raise ValueError(f"unknown expected attribute kind {expected_kind!r}")
        expected_fields.update((group_id, field) for field in fields)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        extra = sorted(actual_fields - expected_fields)
        raise ValueError(
            f"{filename} group-field set mismatch; missing={missing}, extra={extra}"
        )
    return {
        "payload_bytes": len(raw_payload),
        "num_chunks": len(chunks),
        "chunks": chunks,
    }


def _validate_factorized_summary(
    summary: Any, attribute_info: dict[str, dict[str, Any]]
) -> None:
    """Cross-check summary payload and chunk counts without using file paths."""
    if not isinstance(summary, dict):
        raise ValueError("summary.json must contain a JSON object")
    required = {
        "xyz",
        "other_attribute",
        "sh_attribute",
        "raw_attribute_payload_bytes",
    }
    if not required.issubset(summary):
        raise ValueError("summary.json is missing required statistics")
    total = 0
    for key in ("xyz", "other_attribute", "sh_attribute"):
        entry = summary[key]
        if not isinstance(entry, dict):
            raise ValueError(f"summary {key!r} must be an object")
        payload_bytes = _strict_integer(
            entry.get("payload_bytes"), f"summary.{key}.payload_bytes", 0
        )
        num_chunks = _strict_integer(
            entry.get("num_chunks"), f"summary.{key}.num_chunks", 0
        )
        if (
            payload_bytes != attribute_info[key]["payload_bytes"]
            or num_chunks != attribute_info[key]["num_chunks"]
        ):
            raise ValueError(f"summary statistics do not match {key}")
        total += payload_bytes
    if _strict_integer(
        summary["raw_attribute_payload_bytes"],
        "summary.raw_attribute_payload_bytes",
        0,
    ) != total:
        raise ValueError("summary raw_attribute_payload_bytes is inconsistent")


def load_factorized_lightgaussian_compact_package(
    package_path: str | Path,
) -> dict[str, Any]:
    """Safely decode a V2 compact ZIP or directory into compressed vertices.

    The bitstream-unpacking and complete-attribute reconstruction concepts are
    independently adapted from VITA-Group/LightGaussian, specifically
    ``vectree/utils.py::load_vqgaussian`` and ``vectree/utils.py::bin2dec``.
    This is not a loader for the LightGaussian disk format: this V2 format uses
    factorized group-wise scalar quantization, independent pruning/precision
    settings per group, and no VQ codebook.

    In addition to the referenced reconstruction idea, this implementation
    performs in-memory ZIP reads, path-escape and duplicate-member rejection,
    CRC checks, ``allow_pickle=False`` NPZ loading, metadata schema validation,
    chunk-offset/payload-length validation, and padding-bit validation.

    The returned vertices represent the already-pruned and quantized Gaussian
    set. This function never attempts to recreate Gaussians removed by pruning.
    """
    files, source_kind, normalized_path = _read_factorized_package_files(package_path)
    metadata = _strict_json_loads(files["metadata.json"], "metadata.json")
    summary = _strict_json_loads(files["summary.json"], "summary.json")
    (
        vertex_dtype,
        compressed_count,
        original_count,
        original_size,
        groups,
    ) = _validate_factorized_metadata(metadata)
    group_ids = _read_factorized_group_ids(
        files["group_ids.npz"], compressed_count, groups
    )
    vertices = np.zeros(compressed_count, dtype=vertex_dtype)
    attribute_info = {
        "xyz": _decode_factorized_attribute_file(
            files["xyz.npz"], "xyz.npz", "xyz", vertices, group_ids, groups
        ),
        "other_attribute": _decode_factorized_attribute_file(
            files["other_attribute.npz"],
            "other_attribute.npz",
            "other_geo",
            vertices,
            group_ids,
            groups,
        ),
        "sh_attribute": _decode_factorized_attribute_file(
            files["sh_attribute.npz"],
            "sh_attribute.npz",
            "sh",
            vertices,
            group_ids,
            groups,
        ),
    }
    _validate_factorized_summary(summary, attribute_info)
    if len(vertices) != compressed_count or tuple(vertices.dtype.names or ()) != tuple(
        metadata["vertex_field_order"]
    ):
        raise ValueError("reconstructed vertices do not match metadata")
    for field in vertices.dtype.names or ():
        if not np.all(np.isfinite(vertices[field])):
            raise ValueError(f"reconstructed field {field!r} contains non-finite values")
    return {
        "vertices": vertices,
        "group_ids": group_ids,
        "metadata": metadata,
        "summary": summary,
        "compact_format": _FACTORIZED_COMPACT_FORMAT,
        "compressed_vertex_count": compressed_count,
        "original_vertex_count": original_count,
        "original_size_bytes": original_size,
        "source_kind": source_kind,
        "package_path": str(normalized_path),
    }


def validate_factorized_compact_writer() -> bool:
    """Run lightweight invariants for the explicit factorized V2 writer."""
    import tempfile

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    def require_value_error(callable_object: Any, message: str) -> None:
        try:
            callable_object()
        except ValueError:
            return
        raise AssertionError(message)

    def read_attribute_file(
        path: Path,
    ) -> tuple[bytes, list[dict[str, Any]]]:
        with np.load(path, allow_pickle=False) as attribute_data:
            payload = attribute_data["payload"].astype(np.uint8).tobytes()
            chunks = json.loads(str(attribute_data["chunks_json"].item()))
        return payload, chunks

    field_names = [
        "x",
        "y",
        "z",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        *[f"f_rest_{index}" for index in range(45)],
    ]
    vertex_dtype = np.dtype([(name, np.float32) for name in field_names])
    vertices = np.zeros(6, dtype=vertex_dtype)
    base_values = np.linspace(0.07, 1.31, num=6, dtype=np.float32)
    for field_index, field_name in enumerate(field_names):
        vertices[field_name] = base_values + np.float32(field_index * 0.017)

    group_ids = np.asarray([0, 0, 0, 1, 1, -1], dtype=np.int32)
    pruning_levels = np.asarray([0, 0, 0, 4, 4, 0], dtype=np.uint8)
    precision_levels = np.asarray([0, 0, 0, 5, 5, 0], dtype=np.uint8)
    storage_ids = np.asarray([0, 0, 0, 29, 29, 0], dtype=np.uint8)
    original_size_bytes = 1_000_000

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        compact_root = temporary_root / "factorized_compact"
        zip_path = temporary_root / "factorized_compact.zip"
        result = save_factorized_lightgaussian_compact_package(
            vertices=vertices,
            group_ids=group_ids,
            pruning_levels=pruning_levels,
            precision_levels=precision_levels,
            storage_ids=storage_ids,
            compact_root=compact_root,
            zip_path=zip_path,
            original_vertex_count=8,
            original_size_bytes=original_size_bytes,
        )
        require(zip_path.exists(), "V2 compact zip must be created")
        require(result["compact_size_bytes"] > 0, "V2 compact zip must be nonempty")

        extreme_dir = compact_root / "extreme_saving"
        metadata = json.loads(
            (extreme_dir / "metadata.json").read_text(encoding="utf-8")
        )
        require(
            metadata["format"] == "rl_factorized_3dgs_compact_v2",
            "metadata must identify the V2 compact format",
        )
        require(
            metadata["original_size_bytes"] == original_size_bytes,
            "metadata must retain the real original size",
        )
        require(
            np.isclose(
                result["compact_size_ratio"],
                result["compact_size_bytes"] / original_size_bytes,
            ),
            "compact_size_ratio must use the real original size",
        )

        xyz_payload, xyz_chunks = read_attribute_file(extreme_dir / "xyz.npz")
        other_payload, other_chunks = read_attribute_file(
            extreme_dir / "other_attribute.npz"
        )
        sh_payload, sh_chunks = read_attribute_file(
            extreme_dir / "sh_attribute.npz"
        )
        require(
            {chunk["field"] for chunk in xyz_chunks} <= {"x", "y", "z"},
            "xyz chunks may contain only x, y, and z",
        )
        require(
            not any(
                chunk["field"] in {"x", "y", "z"}
                for chunk in other_chunks
            ),
            "other_attribute must not duplicate xyz fields",
        )

        all_chunks = xyz_chunks + other_chunks + sh_chunks
        identity_chunks = [
            chunk for chunk in all_chunks if chunk["group_id"] in {-1, 0}
        ]
        require(
            bool(identity_chunks)
            and all(
                chunk["encoding"] == "raw_float32" and chunk["bits"] == 32
                for chunk in identity_chunks
            ),
            "identity groups must store all active fields as raw float32",
        )
        identity_x_chunk = next(
            chunk
            for chunk in xyz_chunks
            if chunk["group_id"] == 0 and chunk["field"] == "x"
        )
        identity_x_bytes = xyz_payload[
            identity_x_chunk["offset"] :
            identity_x_chunk["offset"] + identity_x_chunk["nbytes"]
        ]
        require(
            identity_x_bytes
            == np.asarray(vertices["x"][group_ids == 0], dtype=np.float32).tobytes(),
            "raw identity payload bytes must preserve float32 values exactly",
        )

        aggressive_other = [
            chunk for chunk in other_chunks if chunk["group_id"] == 1
        ]
        aggressive_sh = [chunk for chunk in sh_chunks if chunk["group_id"] == 1]
        require(
            all(
                chunk["bits"] == 8
                and chunk["encoding"] == "quantized_uint"
                for chunk in aggressive_other
            ),
            "precision level 5 other geometry must use 8 bits",
        )
        require(
            all(
                chunk["bits"] == 6
                and chunk["encoding"] == "quantized_uint"
                for chunk in aggressive_sh
            ),
            "precision level 5 SH must use 6 bits",
        )
        require(
            {chunk["field"] for chunk in aggressive_sh}
            == {"f_dc_0", "f_dc_1", "f_dc_2"},
            "precision level 5 SH payload must contain only DC fields",
        )

        for payload, chunks in (
            (xyz_payload, xyz_chunks),
            (other_payload, other_chunks),
            (sh_payload, sh_chunks),
        ):
            expected_offset = 0
            group_fields: set[tuple[int, str]] = set()
            for chunk in chunks:
                require(
                    chunk["offset"] == expected_offset,
                    "chunk offsets must be contiguous",
                )
                expected_offset += chunk["nbytes"]
                group_field = (chunk["group_id"], chunk["field"])
                require(
                    group_field not in group_fields,
                    "each group-field pair must be written once",
                )
                group_fields.add(group_field)
            require(
                expected_offset == len(payload),
                "chunk metadata must cover the payload without gaps",
            )

        group_table = {
            entry["group_id"]: entry for entry in metadata["group_table"]
        }
        require(
            group_table[0]["pruning_level"] == 0
            and group_table[0]["precision_level"] == 0
            and group_table[1]["pruning_level"] == 4
            and group_table[1]["precision_level"] == 5,
            "group table must store both independent action levels",
        )
        require(
            group_table[1]["storage_id"]
            == encode_factorized_action(4, 5),
            "group table storage ID must match its action components",
        )

        with np.load(extreme_dir / "group_ids.npz", allow_pickle=False) as data:
            stored_group_ids = data["group_ids"]
        require(
            np.issubdtype(stored_group_ids.dtype, np.signedinteger)
            and np.array_equal(stored_group_ids, group_ids),
            "group_ids must remain signed and aligned with vertices",
        )

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            member_names = {Path(name).name for name in zip_file.namelist()}
        forbidden_members = {
            "non_vq_mask.npz",
            "vq_indexs.npz",
            "codebook.npz",
            "action_id.npz",
        }
        require(
            not member_names.intersection(forbidden_members),
            "V2 zip must not contain legacy placeholder or action-ID files",
        )
        require(
            result["raw_attribute_payload_bytes"]
            == result["xyz"]["payload_bytes"]
            + result["other_attribute"]["payload_bytes"]
            + result["sh_attribute"]["payload_bytes"],
            "raw payload total must equal the three attribute payloads",
        )
        require(
            np.isfinite(result["format_overhead_ratio"])
            and result["format_overhead_ratio"] >= 0.0,
            "format_overhead_ratio must be finite and nonnegative",
        )

        tampered_storage = storage_ids.copy()
        tampered_storage[0] = 1
        require_value_error(
            lambda: save_factorized_lightgaussian_compact_package(
                vertices,
                group_ids,
                pruning_levels,
                precision_levels,
                tampered_storage,
                temporary_root / "bad_storage",
                temporary_root / "bad_storage.zip",
                8,
                original_size_bytes,
            ),
            "tampered storage IDs must raise ValueError",
        )

        mixed_precision = precision_levels.copy()
        mixed_storage = storage_ids.copy()
        mixed_precision[1] = 1
        mixed_storage[1] = encode_factorized_action(0, 1)
        require_value_error(
            lambda: save_factorized_lightgaussian_compact_package(
                vertices,
                group_ids,
                pruning_levels,
                mixed_precision,
                mixed_storage,
                temporary_root / "mixed_group",
                temporary_root / "mixed_group.zip",
                8,
                original_size_bytes,
            ),
            "mixed precision within one group must raise ValueError",
        )

        bad_ungrouped_precision = precision_levels.copy()
        bad_ungrouped_storage = storage_ids.copy()
        bad_ungrouped_precision[-1] = 1
        bad_ungrouped_storage[-1] = encode_factorized_action(0, 1)
        require_value_error(
            lambda: save_factorized_lightgaussian_compact_package(
                vertices,
                group_ids,
                pruning_levels,
                bad_ungrouped_precision,
                bad_ungrouped_storage,
                temporary_root / "bad_ungrouped",
                temporary_root / "bad_ungrouped.zip",
                8,
                original_size_bytes,
            ),
            "group_id=-1 with non-identity action must raise ValueError",
        )
        require_value_error(
            lambda: save_factorized_lightgaussian_compact_package(
                vertices,
                group_ids[:-1],
                pruning_levels,
                precision_levels,
                storage_ids,
                temporary_root / "bad_length",
                temporary_root / "bad_length.zip",
                8,
                original_size_bytes,
            ),
            "metadata length mismatch must raise ValueError",
        )
        require_value_error(
            lambda: save_factorized_lightgaussian_compact_package(
                vertices,
                group_ids,
                pruning_levels,
                precision_levels,
                storage_ids,
                temporary_root / "bad_size",
                temporary_root / "bad_size.zip",
                8,
                0,
            ),
            "original_size_bytes=0 must raise ValueError",
        )

        legacy_root = temporary_root / "legacy_compact"
        legacy_zip = temporary_root / "legacy_compact.zip"
        legacy_result = save_lightgaussian_compact_package(
            vertices=vertices,
            group_ids=np.zeros(len(vertices), dtype=np.int32),
            action_ids=np.zeros(len(vertices), dtype=np.int16),
            compact_root=legacy_root,
            zip_path=legacy_zip,
            original_vertex_count=8,
            original_size_bytes=original_size_bytes,
        )
        require(
            legacy_zip.exists()
            and legacy_result["compact_size_bytes"] > 0
            and legacy_result["compact_format"] == "rl_lightgaussian_compact_v1",
            "legacy compact writer must remain operational",
        )
    return True


def validate_factorized_compact_roundtrip() -> bool:
    """Exercise V2 write/decode round-trips and strict corruption rejection."""
    import ast
    import copy
    import hashlib
    import inspect
    import tempfile
    import textwrap
    import warnings

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    def require_value_error(callback: Any, message: str) -> None:
        try:
            callback()
        except ValueError:
            return
        raise AssertionError(message)

    def npz_bytes(**arrays: np.ndarray) -> bytes:
        buffer = BytesIO()
        np.savez_compressed(buffer, **arrays)
        return buffer.getvalue()

    def replace_json(
        files: dict[str, bytes], filename: str, value: Any, *, allow_nan: bool = False
    ) -> dict[str, bytes]:
        changed = dict(files)
        changed[filename] = json.dumps(
            value, ensure_ascii=False, allow_nan=allow_nan
        ).encode("utf-8")
        return changed

    def attribute_parts(
        files: dict[str, bytes], filename: str
    ) -> tuple[np.ndarray, list[dict[str, Any]], str]:
        arrays = _load_npz_arrays(files[filename], filename)
        return (
            arrays["payload"].astype(np.uint8, copy=True),
            _strict_json_loads(str(arrays["chunks_json"].item()), filename),
            str(arrays["field_kind"].item()),
        )

    def replace_attribute(
        files: dict[str, bytes],
        filename: str,
        payload: np.ndarray,
        chunks: list[dict[str, Any]],
        kind: str,
    ) -> dict[str, bytes]:
        changed = dict(files)
        changed[filename] = npz_bytes(
            payload=np.asarray(payload, dtype=np.uint8),
            chunks_json=np.array(json.dumps(chunks, ensure_ascii=False)),
            field_kind=np.array(kind),
        )
        return changed

    def write_zip(
        path: Path,
        files: dict[str, bytes],
        extras: list[tuple[str, bytes]] | None = None,
        duplicate_metadata: bool = False,
    ) -> None:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member in _FACTORIZED_REQUIRED_MEMBERS:
                filename = PurePosixPath(member).name
                if filename in files:
                    archive.writestr(member, files[filename])
            if duplicate_metadata:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    archive.writestr(
                        "extreme_saving/metadata.json", files["metadata.json"]
                    )
            for member, payload in extras or []:
                archive.writestr(member, payload)

    def reject_files(
        case_root: Path,
        case_name: str,
        files: dict[str, bytes],
        extras: list[tuple[str, bytes]] | None = None,
        duplicate_metadata: bool = False,
    ) -> None:
        path = case_root / f"{case_name}.zip"
        write_zip(path, files, extras, duplicate_metadata)
        require_value_error(
            lambda: load_factorized_lightgaussian_compact_package(path),
            f"corrupt package {case_name!r} must raise ValueError",
        )

    field_names = [
        "x",
        "y",
        "z",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        *[f"f_rest_{index}" for index in range(45)],
    ]
    vertex_dtype = np.dtype([(name, np.float32) for name in field_names])
    vertices = np.zeros(6, dtype=vertex_dtype)
    row_values = np.asarray([0.11, 0.27, 0.43, 0.79, 1.07, 1.41], dtype=np.float32)
    for field_index, field_name in enumerate(field_names):
        vertices[field_name] = row_values + np.float32(field_index * 0.019)
    group_ids = np.asarray([-1, 0, 1, -1, 0, 1], dtype=np.int32)
    group_actions = {-1: (0, 0), 0: (2, 2), 1: (4, 5)}
    pruning_levels = np.asarray(
        [group_actions[int(group_id)][0] for group_id in group_ids], dtype=np.int16
    )
    precision_levels = np.asarray(
        [group_actions[int(group_id)][1] for group_id in group_ids], dtype=np.int16
    )
    storage_ids = np.asarray(
        [
            encode_factorized_action(pruning_level, precision_level)
            for pruning_level, precision_level in zip(
                pruning_levels, precision_levels
            )
        ],
        dtype=np.int16,
    )
    original_count = 9
    original_size = 2_000_000

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        compact_root = root / "compact"
        zip_path = root / "compact.zip"
        result = save_factorized_lightgaussian_compact_package(
            vertices,
            group_ids,
            pruning_levels,
            precision_levels,
            storage_ids,
            compact_root,
            zip_path,
            original_count,
            original_size,
        )
        extreme_dir = compact_root / "extreme_saving"
        require(zip_path.is_file() and zip_path.stat().st_size > 0, "V2 ZIP missing")
        decoded_extreme = load_factorized_lightgaussian_compact_package(extreme_dir)
        decoded_root = load_factorized_lightgaussian_compact_package(compact_root)
        decoded_zip = load_factorized_lightgaussian_compact_package(zip_path)
        decoded_results = (decoded_extreme, decoded_root, decoded_zip)
        require(
            decoded_extreme["source_kind"] == "directory"
            and decoded_root["source_kind"] == "directory"
            and decoded_zip["source_kind"] == "zip",
            "decoder source_kind is incorrect",
        )
        core_metadata_keys = (
            "format",
            "compressed_vertex_count",
            "original_vertex_count",
            "original_size_bytes",
            "vertex_field_order",
            "vertex_dtype",
            "group_table",
        )
        for decoded in decoded_results[1:]:
            require(
                decoded["vertices"].dtype == decoded_extreme["vertices"].dtype
                and np.array_equal(decoded["vertices"], decoded_extreme["vertices"])
                and np.array_equal(decoded["group_ids"], group_ids)
                and all(
                    decoded["metadata"][key] == decoded_extreme["metadata"][key]
                    for key in core_metadata_keys
                ),
                "directory and ZIP decode results must be identical",
            )
        reconstructed = decoded_zip["vertices"]
        identity_rows = group_ids == -1
        for field_name in field_names:
            require(
                np.array_equal(
                    reconstructed[field_name][identity_rows].view(np.uint32),
                    vertices[field_name][identity_rows].view(np.uint32),
                ),
                f"identity field {field_name!r} was not restored bit-exactly",
            )

        empty_vertices = np.zeros(len(vertices), dtype=vertex_dtype)
        for group_id in (0, 1):
            action = decode_factorized_action(group_actions[group_id])
            active_fields = set(name for name in ("x", "y", "z") if name in field_names)
            active_fields.update(
                name
                for name in geo_fields(empty_vertices)
                if name not in {"x", "y", "z"}
            )
            active_fields.update(active_sh_fields(empty_vertices, action.sh_degree))
            mask = group_ids == group_id
            for field_name in active_fields:
                bits = action.sh_bit if field_name.startswith("f_") else action.geo_bit
                original = vertices[field_name][mask].astype(np.float64)
                decoded = reconstructed[field_name][mask].astype(np.float64)
                value_range = float(np.max(original) - np.min(original))
                tolerance = value_range / (2.0 * ((1 << bits) - 1)) + 2e-6
                require(
                    np.all(np.abs(decoded - original) <= tolerance),
                    f"quantized field {field_name!r} exceeds its error bound",
                )
        aggressive_rows = group_ids == 1
        for field_name in (f"f_rest_{index}" for index in range(45)):
            require(
                np.all(reconstructed[field_name][aggressive_rows] == 0),
                "SH0 inactive f_rest fields must remain zero",
            )
        for field_name in ("f_dc_0", "f_dc_1", "f_dc_2"):
            require(
                np.any(reconstructed[field_name][aggressive_rows] != 0),
                "SH0 DC fields must be restored",
            )
        require(
            np.array_equal(decoded_zip["group_ids"], group_ids)
            and decoded_zip["group_ids"].dtype.kind == "i",
            "signed group IDs and row order must round-trip",
        )
        require(
            decoded_zip["summary"]["raw_attribute_payload_bytes"]
            == result["raw_attribute_payload_bytes"],
            "summary payload statistics must match the writer",
        )

        table_by_group = {
            entry["group_id"]: entry for entry in decoded_zip["metadata"]["group_table"]
        }
        rewrite_pruning = np.asarray(
            [table_by_group[int(group_id)]["pruning_level"] for group_id in group_ids]
        )
        rewrite_precision = np.asarray(
            [table_by_group[int(group_id)]["precision_level"] for group_id in group_ids]
        )
        rewrite_storage = np.asarray(
            [table_by_group[int(group_id)]["storage_id"] for group_id in group_ids]
        )
        rewritten_root = root / "rewritten"
        rewritten_zip = root / "rewritten.zip"
        save_factorized_lightgaussian_compact_package(
            reconstructed,
            decoded_zip["group_ids"],
            rewrite_pruning,
            rewrite_precision,
            rewrite_storage,
            rewritten_root,
            rewritten_zip,
            original_count,
            original_size,
        )
        decoded_again = load_factorized_lightgaussian_compact_package(rewritten_zip)
        require(
            len(decoded_again["vertices"]) == len(reconstructed)
            and decoded_again["metadata"]["group_table"]
            == decoded_zip["metadata"]["group_table"],
            "second-generation package must retain group configuration and count",
        )
        for field_name in field_names:
            require(
                np.array_equal(
                    decoded_again["vertices"][field_name][identity_rows],
                    reconstructed[field_name][identity_rows],
                ),
                "identity values changed during the second write",
            )

        pristine_files = {
            filename: (extreme_dir / filename).read_bytes()
            for filename in _FACTORIZED_REQUIRED_FILES
        }
        metadata = _strict_json_loads(pristine_files["metadata.json"], "metadata")
        summary = _strict_json_loads(pristine_files["summary.json"], "summary")

        bad_format = copy.deepcopy(metadata)
        bad_format["format"] = "rl_lightgaussian_compact_v1"
        reject_files(root, "bad_format", replace_json(pristine_files, "metadata.json", bad_format))
        reject_files(root, "duplicate_member", pristine_files, duplicate_metadata=True)
        reject_files(root, "path_escape", pristine_files, [("../escape.json", b"{}")])

        bad_storage = copy.deepcopy(metadata)
        bad_storage["group_table"][0]["storage_id"] = (
            bad_storage["group_table"][0]["storage_id"] + 1
        ) % 30
        reject_files(root, "bad_storage", replace_json(pristine_files, "metadata.json", bad_storage))

        bad_group_count = copy.deepcopy(metadata)
        bad_group_count["group_table"][0]["vertex_count"] += 1
        reject_files(root, "bad_group_count", replace_json(pristine_files, "metadata.json", bad_group_count))

        xyz_payload, xyz_chunks, xyz_kind = attribute_parts(pristine_files, "xyz.npz")
        bad_offsets = copy.deepcopy(xyz_chunks)
        bad_offsets[0]["offset"] = 1
        reject_files(
            root,
            "bad_offset",
            replace_attribute(pristine_files, "xyz.npz", xyz_payload, bad_offsets, xyz_kind),
        )
        reject_files(
            root,
            "truncated_payload",
            replace_attribute(
                pristine_files, "xyz.npz", xyz_payload[:-1], xyz_chunks, xyz_kind
            ),
        )

        sh_payload, sh_chunks, sh_kind = attribute_parts(pristine_files, "sh_attribute.npz")
        padding_chunk = next(
            chunk
            for chunk in sh_chunks
            if chunk["group_id"] == 1 and chunk["bits"] == 6
        )
        bad_padding_payload = sh_payload.copy()
        bad_padding_payload[
            padding_chunk["offset"] + padding_chunk["nbytes"] - 1
        ] |= np.uint8(1)
        reject_files(
            root,
            "nonzero_padding",
            replace_attribute(
                pristine_files,
                "sh_attribute.npz",
                bad_padding_payload,
                sh_chunks,
                sh_kind,
            ),
        )

        duplicate_chunks = copy.deepcopy(xyz_chunks)
        duplicate_chunks.append(copy.deepcopy(duplicate_chunks[0]))
        reject_files(
            root,
            "duplicate_chunk",
            replace_attribute(
                pristine_files, "xyz.npz", xyz_payload, duplicate_chunks, xyz_kind
            ),
        )
        unknown_field_chunks = copy.deepcopy(xyz_chunks)
        unknown_field_chunks[0]["field"] = "unknown_field"
        reject_files(
            root,
            "unknown_field",
            replace_attribute(
                pristine_files,
                "xyz.npz",
                xyz_payload,
                unknown_field_chunks,
                xyz_kind,
            ),
        )
        inactive_chunks = copy.deepcopy(sh_chunks)
        inactive_chunk = next(
            chunk for chunk in inactive_chunks if chunk["group_id"] == 1
        )
        inactive_chunk["field"] = "f_rest_0"
        reject_files(
            root,
            "inactive_sh",
            replace_attribute(
                pristine_files, "sh_attribute.npz", sh_payload, inactive_chunks, sh_kind
            ),
        )

        object_dtype = copy.deepcopy(metadata)
        object_dtype["vertex_dtype"][0][1] = "|O"
        reject_files(root, "object_dtype", replace_json(pristine_files, "metadata.json", object_dtype))
        nan_metadata = copy.deepcopy(metadata)
        nan_metadata["unexpected_nan"] = float("nan")
        reject_files(
            root,
            "metadata_nan",
            replace_json(
                pristine_files, "metadata.json", nan_metadata, allow_nan=True
            ),
        )

        group_arrays = _load_npz_arrays(pristine_files["group_ids.npz"], "group_ids")
        unsigned_files = dict(pristine_files)
        unsigned_files["group_ids.npz"] = npz_bytes(
            group_ids=group_arrays["group_ids"].astype(np.uint32)
        )
        reject_files(root, "unsigned_group_ids", unsigned_files)

        bad_summary = copy.deepcopy(summary)
        bad_summary["xyz"]["payload_bytes"] += 1
        reject_files(root, "bad_summary", replace_json(pristine_files, "summary.json", bad_summary))
        missing_member = dict(pristine_files)
        del missing_member["sh_attribute.npz"]
        reject_files(root, "missing_member", missing_member)
        reject_files(
            root,
            "v1_placeholder",
            pristine_files,
            [("extreme_saving/action_id.npz", b"placeholder")],
        )

        legacy_root = root / "legacy"
        legacy_zip = root / "legacy.zip"
        legacy_result = save_lightgaussian_compact_package(
            vertices,
            np.zeros(len(vertices), dtype=np.int32),
            np.zeros(len(vertices), dtype=np.int16),
            legacy_root,
            legacy_zip,
            original_count,
            original_size,
        )
        expected_v1_members = {
            "extreme_saving/metadata.npz",
            "extreme_saving/group_id.npz",
            "extreme_saving/action_id.npz",
            "extreme_saving/non_vq_mask.npz",
            "extreme_saving/vq_indexs.npz",
            "extreme_saving/codebook.npz",
            "extreme_saving/xyz.npz",
            "extreme_saving/other_attribute.npz",
            "extreme_saving/non_vq_feats.npz",
            "extreme_saving/summary.json",
        }
        with zipfile.ZipFile(legacy_zip, "r") as archive:
            actual_v1_members = set(archive.namelist())
        require(
            legacy_result["compact_format"] == "rl_lightgaussian_compact_v1"
            and legacy_zip.stat().st_size > 0
            and actual_v1_members == expected_v1_members,
            "legacy V1 writer format or file set changed",
        )

    legacy_hashes = {
        "_pack_integer_array": "d9769603c772c9b2f5a8bf22b914ec898389aace03e33e253720389977b6e33e",
        "_quantize": "e29953012aa9d0a2e05558fafa97270604cfdda16189e7f1a3967eb132b65190",
        "_save_groupwise_fields": "e8735e069deb638b9864fdb969d599930077b38c7114234c47e4b76a1ab21e65",
        "_zip_dir": "00d6240de8fae937158c7bd5fd1b94c42340442502b87e3dfbf50e3330db6708",
        "save_lightgaussian_compact_package": "90ff308a0b46e0674559db030a6556ad1e30316be51877c55a24c5e99c88f88f",
        "save_factorized_lightgaussian_compact_package": "d4863757a4c865743b6bb5554c7df8bea67e37c22a457c346cb3a0e1e0108638",
        "_unpack_integer_array": "a22c9327b848280ba2934daa910ec0c8574c35dcc9a0ef1fd849b3e462ad7a2a",
    }
    for function_name, expected_hash in legacy_hashes.items():
        source = inspect.getsource(globals()[function_name])
        require(
            hashlib.sha256(source.encode("utf-8")).hexdigest() == expected_hash,
            f"legacy writer function {function_name} changed",
        )
    loader_tree = ast.parse(
        textwrap.dedent(
            inspect.getsource(load_factorized_lightgaussian_compact_package)
        )
    )
    loader_function = loader_tree.body[0]
    if not isinstance(loader_function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise AssertionError("unable to inspect factorized loader body")
    if (
        loader_function.body
        and isinstance(loader_function.body[0], ast.Expr)
        and isinstance(loader_function.body[0].value, ast.Constant)
        and isinstance(loader_function.body[0].value.value, str)
    ):
        loader_function.body = loader_function.body[1:]
    require(
        hashlib.sha256(
            ast.dump(loader_tree, include_attributes=False).encode("utf-8")
        ).hexdigest()
        == "11808edfa4c81c9c6452e055da325096ef6542d22cc0ee4367df7d9fa371bcd9",
        "factorized loader algorithm changed beyond its docstring",
    )
    require(
        validate_factorized_compact_writer(),
        "existing factorized writer validation must remain green",
    )
    require(
        validate_lightgaussian_bitstream_compatibility(),
        "LightGaussian bitstream compatibility validation must remain green",
    )
    return True
