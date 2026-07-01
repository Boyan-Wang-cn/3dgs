from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Iterable

import numpy as np


PLY_TO_NUMPY = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

COMMON_3DGS_FIELDS = {
    "xyz": ["x", "y", "z"],
    "opacity": ["opacity"],
    "scale": ["scale_0", "scale_1", "scale_2"],
    "rotation": ["rot_0", "rot_1", "rot_2", "rot_3"],
    "f_dc": ["f_dc_0", "f_dc_1", "f_dc_2"],
}


@dataclass(frozen=True)
class PLYProperty:
    name: str
    ply_type: str
    numpy_type: str


@dataclass(frozen=True)
class GaussianPLY:
    path: Path
    fmt: str
    header_lines: list[str]
    vertex_count: int
    vertex_properties: list[PLYProperty]
    vertex_data: np.ndarray
    tail_bytes: bytes = b""

    @property
    def property_names(self) -> list[str]:
        return [prop.name for prop in self.vertex_properties]

    @property
    def dtype(self) -> np.dtype:
        return self.vertex_data.dtype

    def with_vertex_data(self, vertex_data: np.ndarray) -> "GaussianPLY":
        return replace(self, vertex_count=len(vertex_data), vertex_data=vertex_data)


def _read_header(handle) -> tuple[list[str], int]:
    lines: list[str] = []
    header_size = 0
    while True:
        raw = handle.readline()
        if not raw:
            raise ValueError("Unexpected end of file before PLY end_header.")
        header_size += len(raw)
        line = raw.decode("ascii", errors="strict").rstrip("\r\n")
        lines.append(line)
        if line == "end_header":
            return lines, header_size


def _parse_vertex_header(header_lines: list[str]) -> tuple[str, int, list[PLYProperty]]:
    if not header_lines or header_lines[0] != "ply":
        raise ValueError("Not a PLY file: first header line must be 'ply'.")

    fmt = ""
    vertex_count: int | None = None
    vertex_properties: list[PLYProperty] = []
    current_element: str | None = None

    for line in header_lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            if len(parts) < 3:
                raise ValueError(f"Invalid PLY format line: {line}")
            fmt = parts[1]
        elif parts[0] == "element":
            if len(parts) != 3:
                raise ValueError(f"Invalid PLY element line: {line}")
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) == 5 and parts[1] == "list":
                raise ValueError("List properties in vertex elements are not supported.")
            if len(parts) != 3:
                raise ValueError(f"Invalid PLY property line: {line}")
            ply_type, name = parts[1], parts[2]
            if ply_type not in PLY_TO_NUMPY:
                raise ValueError(f"Unsupported PLY property type '{ply_type}' for '{name}'.")
            vertex_properties.append(
                PLYProperty(name=name, ply_type=ply_type, numpy_type=PLY_TO_NUMPY[ply_type])
            )

    if fmt not in {"binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Only binary PLY is supported in this baseline, got '{fmt}'.")
    if vertex_count is None:
        raise ValueError("PLY header does not contain an element vertex line.")
    if not vertex_properties:
        raise ValueError("PLY vertex element has no scalar properties.")
    return fmt, vertex_count, vertex_properties


def _dtype_for_properties(fmt: str, properties: Iterable[PLYProperty]) -> np.dtype:
    endian = "<" if fmt == "binary_little_endian" else ">"
    fields = [(prop.name, np.dtype(endian + prop.numpy_type)) for prop in properties]
    return np.dtype(fields)


def read_ply(path: str | Path) -> GaussianPLY:
    """Read a binary 3DGS PLY while preserving header and property order."""
    ply_path = Path(path)
    with ply_path.open("rb") as handle:
        header_lines, _ = _read_header(handle)
        fmt, vertex_count, properties = _parse_vertex_header(header_lines)
        dtype = _dtype_for_properties(fmt, properties)
        vertex_data = np.fromfile(handle, dtype=dtype, count=vertex_count)
        if len(vertex_data) != vertex_count:
            raise ValueError(
                f"Expected {vertex_count} vertices, but only read {len(vertex_data)}."
            )
        tail_bytes = handle.read()

    return GaussianPLY(
        path=ply_path,
        fmt=fmt,
        header_lines=header_lines,
        vertex_count=vertex_count,
        vertex_properties=properties,
        vertex_data=vertex_data,
        tail_bytes=tail_bytes,
    )


def _updated_header_lines(header_lines: list[str], vertex_count: int) -> list[str]:
    updated: list[str] = []
    vertex_line_replaced = False
    for line in header_lines:
        if line.startswith("element vertex "):
            updated.append(f"element vertex {vertex_count}")
            vertex_line_replaced = True
        else:
            updated.append(line)
    if not vertex_line_replaced:
        raise ValueError("Cannot save PLY because the original header has no vertex element.")
    return updated


def write_ply(
    ply: GaussianPLY,
    output_path: str | Path,
    vertex_data: np.ndarray | None = None,
) -> Path:
    """Write a modified PLY without changing property order or scalar dtypes."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = ply.vertex_data if vertex_data is None else vertex_data
    if data.dtype != ply.vertex_data.dtype:
        data = data.astype(ply.vertex_data.dtype, copy=False)

    header_lines = _updated_header_lines(ply.header_lines, len(data))
    with out_path.open("wb") as handle:
        handle.write(("\n".join(header_lines) + "\n").encode("ascii"))
        data.tofile(handle)
        handle.write(ply.tail_bytes)

    return out_path


def numbered_fields(property_names: Iterable[str], prefix: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    matches: list[tuple[int, str]] = []
    for name in property_names:
        match = pattern.match(name)
        if match:
            matches.append((int(match.group(1)), name))
    return [name for _, name in sorted(matches)]


def get_xyz(vertex_data: np.ndarray) -> np.ndarray:
    missing = [name for name in ("x", "y", "z") if name not in vertex_data.dtype.names]
    if missing:
        raise KeyError(f"PLY vertex data is missing xyz fields: {missing}")
    return np.column_stack(
        [vertex_data["x"], vertex_data["y"], vertex_data["z"]]
    ).astype(np.float32, copy=False)


def inspect_ply(ply: GaussianPLY) -> dict:
    names = ply.property_names
    recognized = {
        key: [field for field in fields if field in names]
        for key, fields in COMMON_3DGS_FIELDS.items()
    }
    recognized["f_rest"] = numbered_fields(names, "f_rest")

    missing = {
        key: [field for field in fields if field not in names]
        for key, fields in COMMON_3DGS_FIELDS.items()
    }
    return {
        "path": str(ply.path),
        "format": ply.fmt,
        "vertex_count": ply.vertex_count,
        "property_count": len(names),
        "properties": names,
        "recognized": recognized,
        "missing_common_fields": missing,
    }


def print_ply_summary(path: str | Path) -> GaussianPLY:
    ply = read_ply(path)
    info = inspect_ply(ply)

    print(f"PLY path: {info['path']}")
    print(f"Format: {info['format']}")
    print(f"Vertex count: {info['vertex_count']}")
    print(f"Property count: {info['property_count']}")
    print("Properties:")
    for idx, name in enumerate(info["properties"]):
        print(f"  [{idx:02d}] {name}")

    print("Recognized 3DGS fields:")
    for group, fields in info["recognized"].items():
        preview = ", ".join(fields[:8])
        suffix = " ..." if len(fields) > 8 else ""
        print(f"  {group}: {len(fields)} field(s) {preview}{suffix}")

    missing = {k: v for k, v in info["missing_common_fields"].items() if v}
    if missing:
        print("Missing common 3DGS fields:")
        for group, fields in missing.items():
            print(f"  {group}: {', '.join(fields)}")
    return ply
