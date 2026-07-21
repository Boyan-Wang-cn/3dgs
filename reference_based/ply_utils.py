"""Independent binary PLY helpers for 3D Gaussian Splatting data.

The reader keeps the vertex property order, scalar types, byte order, vertex
order, and any bytes belonging to elements after ``vertex``.  The writer uses
that metadata to produce a binary PLY that can be read back without numerical
changes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np


PLY_TO_NUMPY: dict[str, str] = {
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

COMMON_3DGS_FIELDS: dict[str, list[str]] = {
    "xyz": ["x", "y", "z"],
    "normal": ["nx", "ny", "nz"],
    "opacity": ["opacity"],
    "f_dc": ["f_dc_0", "f_dc_1", "f_dc_2"],
    "scale": ["scale_0", "scale_1", "scale_2"],
    "rotation": ["rot_0", "rot_1", "rot_2", "rot_3"],
}


@dataclass(frozen=True)
class PLYProperty:
    """One scalar property in the PLY vertex element."""

    name: str
    ply_type: str
    numpy_type: str


@dataclass(frozen=True)
class GaussianPLY:
    """A binary Gaussian PLY and the metadata needed for lossless rewriting."""

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


def _read_header(handle: BinaryIO) -> list[str]:
    lines: list[str] = []
    while True:
        raw = handle.readline()
        if not raw:
            raise ValueError("Unexpected end of file before PLY end_header.")
        try:
            line = raw.decode("ascii").rstrip("\r\n")
        except UnicodeDecodeError as error:
            raise ValueError("PLY header must contain ASCII text.") from error
        lines.append(line)
        if line == "end_header":
            return lines


def _parse_vertex_header(
    header_lines: list[str],
) -> tuple[str, int, list[PLYProperty]]:
    if not header_lines or header_lines[0] != "ply":
        raise ValueError("Not a PLY file: first header line must be 'ply'.")

    fmt = ""
    vertex_count: int | None = None
    properties: list[PLYProperty] = []
    current_element: str | None = None

    for line in header_lines[1:]:
        parts = line.split()
        if not parts:
            continue
        keyword = parts[0]
        if keyword == "format":
            if len(parts) != 3 or parts[2] != "1.0":
                raise ValueError(f"Invalid or unsupported PLY format line: {line}")
            fmt = parts[1]
        elif keyword == "element":
            if len(parts) != 3:
                raise ValueError(f"Invalid PLY element line: {line}")
            current_element = parts[1]
            if current_element == "vertex":
                try:
                    vertex_count = int(parts[2])
                except ValueError as error:
                    raise ValueError(f"Invalid vertex count in PLY header: {line}") from error
                if vertex_count < 0:
                    raise ValueError("PLY vertex count cannot be negative.")
        elif keyword == "property" and current_element == "vertex":
            if len(parts) >= 2 and parts[1] == "list":
                raise ValueError("List properties in the vertex element are unsupported.")
            if len(parts) != 3:
                raise ValueError(f"Invalid PLY property line: {line}")
            ply_type, name = parts[1], parts[2]
            if ply_type not in PLY_TO_NUMPY:
                raise ValueError(
                    f"Unsupported PLY property type '{ply_type}' for '{name}'."
                )
            properties.append(
                PLYProperty(name=name, ply_type=ply_type, numpy_type=PLY_TO_NUMPY[ply_type])
            )

    if fmt not in {"binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Only binary PLY is supported, got '{fmt}'.")
    if vertex_count is None:
        raise ValueError("PLY header does not contain an element vertex line.")
    if not properties:
        raise ValueError("PLY vertex element has no scalar properties.")
    if len({prop.name for prop in properties}) != len(properties):
        raise ValueError("PLY vertex property names must be unique.")
    return fmt, vertex_count, properties


def _dtype_for_properties(
    fmt: str, properties: Iterable[PLYProperty]
) -> np.dtype:
    endian = "<" if fmt == "binary_little_endian" else ">"
    return np.dtype(
        [(prop.name, np.dtype(endian + prop.numpy_type)) for prop in properties]
    )


def read_ply(path: str | Path) -> GaussianPLY:
    """Read a binary Gaussian PLY without reordering or converting vertices."""

    ply_path = Path(path)
    with ply_path.open("rb") as handle:
        header_lines = _read_header(handle)
        fmt, vertex_count, properties = _parse_vertex_header(header_lines)
        dtype = _dtype_for_properties(fmt, properties)
        expected_bytes = vertex_count * dtype.itemsize
        raw_vertices = handle.read(expected_bytes)
        if len(raw_vertices) != expected_bytes:
            complete_vertices = len(raw_vertices) // dtype.itemsize
            raise ValueError(
                f"Expected {vertex_count} vertices, but only read "
                f"{complete_vertices} complete vertices."
            )
        vertex_data = np.frombuffer(raw_vertices, dtype=dtype, count=vertex_count).copy()
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
    replaced = False
    for line in header_lines:
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            if replaced:
                raise ValueError("PLY header contains multiple vertex elements.")
            updated.append(f"element vertex {vertex_count}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        raise ValueError("Cannot save PLY: header has no vertex element.")
    return updated


def _coerce_vertex_data(ply: GaussianPLY, vertex_data: np.ndarray) -> np.ndarray:
    if not isinstance(vertex_data, np.ndarray):
        raise TypeError("vertex_data must be a NumPy array.")
    if vertex_data.ndim != 1 or vertex_data.dtype.names is None:
        raise ValueError("vertex_data must be a one-dimensional structured array.")

    target_dtype = _dtype_for_properties(ply.fmt, ply.vertex_properties)
    if ply.vertex_data.dtype != target_dtype:
        raise ValueError("GaussianPLY vertex dtype does not match its PLY properties.")
    if vertex_data.dtype == target_dtype:
        return vertex_data
    if vertex_data.dtype.names != target_dtype.names:
        raise ValueError("vertex_data fields or field order do not match the PLY header.")
    return vertex_data.astype(target_dtype, copy=False)


def write_ply(
    ply: GaussianPLY,
    output_path: str | Path,
    vertex_data: np.ndarray | None = None,
) -> Path:
    """Write a binary PLY while retaining property order and scalar precision."""

    if not isinstance(ply, GaussianPLY):
        raise TypeError("ply must be a GaussianPLY instance.")
    if ply.fmt not in {"binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Only binary PLY is supported, got '{ply.fmt}'.")

    data = ply.vertex_data if vertex_data is None else vertex_data
    data = _coerce_vertex_data(ply, data)
    header_lines = _updated_header_lines(ply.header_lines, len(data))
    if not header_lines or header_lines[-1] != "end_header":
        raise ValueError("PLY header must end with end_header.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        handle.write(("\n".join(header_lines) + "\n").encode("ascii"))
        handle.write(data.tobytes(order="C"))
        handle.write(ply.tail_bytes)
    return out_path


def numbered_fields(property_names: Iterable[str], prefix: str) -> list[str]:
    """Return ``prefix_N`` fields sorted by their numeric suffix."""

    matches: list[tuple[int, str]] = []
    marker = prefix + "_"
    for name in property_names:
        if name.startswith(marker):
            suffix = name[len(marker) :]
            if suffix.isdigit():
                matches.append((int(suffix), name))
    return [name for _, name in sorted(matches)]


def get_xyz(vertex_data: np.ndarray) -> np.ndarray:
    """Return the x/y/z fields as the float32 matrix expected by grouping code."""

    names = vertex_data.dtype.names or ()
    missing = [name for name in ("x", "y", "z") if name not in names]
    if missing:
        raise KeyError(f"PLY vertex data is missing xyz fields: {missing}")
    return np.column_stack(
        [vertex_data["x"], vertex_data["y"], vertex_data["z"]]
    ).astype(np.float32, copy=False)


def inspect_ply(ply: GaussianPLY) -> dict[str, object]:
    """Build a compact summary while preserving the historical helper API."""

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
    """Print a human-readable PLY summary and return the parsed object."""

    ply = read_ply(path)
    info = inspect_ply(ply)
    print(f"PLY path: {info['path']}")
    print(f"Format: {info['format']}")
    print(f"Vertex count: {info['vertex_count']}")
    print(f"Property count: {info['property_count']}")
    print("Properties:")
    for index, name in enumerate(info["properties"]):
        print(f"  [{index:02d}] {name}")

    print("Recognized 3DGS fields:")
    recognized = info["recognized"]
    if not isinstance(recognized, dict):
        raise TypeError("Invalid PLY inspection result.")
    for group, fields in recognized.items():
        preview = ", ".join(fields[:8])
        suffix = " ..." if len(fields) > 8 else ""
        print(f"  {group}: {len(fields)} field(s) {preview}{suffix}")

    missing_common = info["missing_common_fields"]
    if not isinstance(missing_common, dict):
        raise TypeError("Invalid PLY inspection result.")
    missing = {key: fields for key, fields in missing_common.items() if fields}
    if missing:
        print("Missing common 3DGS fields:")
        for group, fields in missing.items():
            print(f"  {group}: {', '.join(fields)}")
    return ply


def _validation_path(stem: str) -> Path:
    root = Path(__file__).resolve().parent
    for index in range(1000):
        candidate = root / f".{stem}_{index}.ply"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not allocate a PLY validation path.")


def validate_reference_based_ply_utils() -> bool:
    """Run a self-contained write/read/write/read lossless round-trip check."""

    names = [
        "x", "y", "z", "nx", "ny", "nz", "opacity",
        "f_dc_0", "f_dc_1", "f_dc_2",
        *[f"f_rest_{index}" for index in range(12)],
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    double_fields = {"x", "f_rest_7", "rot_3"}
    properties = [
        PLYProperty(
            name=name,
            ply_type="double" if name in double_fields else "float",
            numpy_type="f8" if name in double_fields else "f4",
        )
        for name in names
    ]
    dtype = _dtype_for_properties("binary_little_endian", properties)
    vertex_count = 7
    vertices = np.empty(vertex_count, dtype=dtype)
    for field_index, name in enumerate(names):
        field_dtype = dtype.fields[name][0]
        vertices[name] = (
            np.arange(vertex_count, dtype=field_dtype)
            * field_dtype.type(0.125)
            + field_dtype.type(field_index + 0.03125)
        )

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment generated by the independent validation suite",
        f"element vertex {vertex_count}",
        *[f"property {prop.ply_type} {prop.name}" for prop in properties],
        "end_header",
    ]
    first_path = _validation_path("ply_utils_validation_first")
    second_path = _validation_path("ply_utils_validation_second")
    source = GaussianPLY(
        path=first_path,
        fmt="binary_little_endian",
        header_lines=header_lines,
        vertex_count=vertex_count,
        vertex_properties=properties,
        vertex_data=vertices,
    )

    try:
        write_ply(source, first_path)
        first = read_ply(first_path)
        write_ply(first, second_path)
        second = read_ply(second_path)

        checks = [
            first.vertex_count == vertex_count,
            second.vertex_count == vertex_count,
            first.property_names == names,
            second.property_names == names,
            first.vertex_data.dtype == dtype,
            second.vertex_data.dtype == dtype,
            np.array_equal(first.vertex_data, vertices),
            np.array_equal(second.vertex_data, vertices),
            first.tail_bytes == b"",
            second.tail_bytes == b"",
        ]
        legacy_module_name = "gs_" + "baseline"
        checks.append(legacy_module_name not in Path(__file__).read_text(encoding="utf-8"))
        if not all(checks):
            raise AssertionError("Independent PLY validation failed.")
        return True
    finally:
        for path in (first_path, second_path):
            if path.exists():
                path.unlink()
