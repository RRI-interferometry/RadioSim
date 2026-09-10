"""Unwired finite transport identity primitives; no operation-chain acceptance."""

import hashlib
import json
import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_PROFILE = "pyradiosky_1_1_0_theta_phi_v1"
_DOMAIN = b"RADIOSIM_PYRADIOSKY_HEALPIX_PAYLOAD_V1\n"


@dataclass(frozen=True, slots=True)
class SerializedNativePayload:
    """Untrusted actual endpoint observations; callers exclude alias mutation."""

    nside: int
    frequencies: NDArray[np.float64]
    pixel_ids: NDArray[np.int64]
    stokes: NDArray[np.float64]
    profile: str
    coordinate_frame: str
    ordering: str
    component_type: str
    spectral_type: str
    frequency_unit: str
    stokes_unit: str
    brightness_conversion: str


@dataclass(frozen=True, slots=True)
class SerializedPayloadBinding:
    """Computed word identity; neither producer trust nor a completed transfer."""

    metadata_json: bytes
    payload_sha256: str
    preimage_byte_count: int


def _json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def bind_serialized_native(value: SerializedNativePayload) -> SerializedPayloadBinding:
    """Hash finite actual sorted theta/phi arrays without a tensor-sized copy.

    This is identity only. No parent, profile truth or transform is authenticated.
    Exact ndarray owners are readonly and C-contiguous; caller excludes mutation
    or rebinding through all aliases for this entire call. No snapshot is acquired.
    """
    _require(type(value) is SerializedNativePayload, "expected exact endpoint type")
    for actual, expected in (
        (value.profile, _PROFILE),
        (value.coordinate_frame, "icrs"),
        (value.ordering, "ring"),
        (value.component_type, "healpix"),
        (value.spectral_type, "full"),
        (value.frequency_unit, "Hz"),
        (value.stokes_unit, "K"),
        (value.brightness_conversion, "rayleigh-jeans"),
    ):
        _require(
            type(actual) is str and actual == expected, "unsupported endpoint metadata"
        )
    ns = value.nside
    _require(
        type(ns) is int and 1 <= ns <= 2**29 and ns & (ns - 1) == 0, "invalid nside"
    )
    arrays = (value.frequencies, value.pixel_ids, value.stokes)
    for array, dtype in zip(arrays, ("<f8", "<i8", "<f8"), strict=True):
        _require(type(array) is np.ndarray, "expected exact ndarray")
        _require(array.dtype.str == dtype, "unsupported endpoint dtype")
        _require(
            not array.flags.writeable and array.flags.c_contiguous,
            "readonly C owner required",
        )
    frequency, ids, stokes = arrays
    _require(frequency.ndim == ids.ndim == 1, "one-dimensional axes required")
    f, n = frequency.size, ids.size
    _require(1 <= f <= 1024 and 1 <= n <= 65536, "endpoint dimension limit")
    _require(stokes.shape == (4, f, n), "stokes shape differs")
    _require(stokes.nbytes <= 2**31, "tensor exceeds 2 GiB")
    names = ("frequencies", "pixel_ids", "stokes")
    metadata = _json(
        {
            "schema_version": "radiosim.pyradiosky-healpix-payload-metadata.v1",
            "profile": value.profile,
            "component_type": value.component_type,
            "spectral_type": value.spectral_type,
            "coordinate_frame": value.coordinate_frame,
            "ordering": value.ordering,
            "nside": ns,
            "units": {
                "frequencies": value.frequency_unit,
                "pixel_ids": "1",
                "stokes": value.stokes_unit,
            },
            "brightness_conversion": value.brightness_conversion,
            "stokes_axis_order": ["I", "Q", "U", "V"],
            "arrays": [
                {
                    "name": name,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "byte_count": array.nbytes,
                }
                for name, array in zip(names, arrays, strict=True)
            ],
        }
    )
    sizes = (len(metadata), *(array.nbytes for array in arrays))
    total = len(_DOMAIN) + sum(8 + size for size in sizes)
    _require(
        all(0 <= size < 2**64 for size in (*sizes, total)), "preimage exceeds uint64"
    )
    _require(
        bool(np.all(np.isfinite(frequency))) and bool(np.all(frequency > 0)),
        "invalid frequencies",
    )
    _require(
        bool(np.all(frequency[1:] > frequency[:-1])), "frequencies not strictly sorted"
    )
    _require(
        bool(np.all(ids >= 0)) and bool(np.all(ids < 12 * ns**2)),
        "pixel ID outside grid",
    )
    sorted_ids = np.sort(ids, kind="heapsort")
    _require(bool(np.all(sorted_ids[1:] != sorted_ids[:-1])), "duplicate pixel ID")
    del sorted_ids
    digest = hashlib.sha256(_DOMAIN + struct.pack("<Q", len(metadata)) + metadata)
    for index, array in enumerate(arrays):
        digest.update(struct.pack("<Q", array.nbytes))
        flat = array.reshape(-1)
        for start in range(0, flat.size, 65536):
            block = flat[start : start + 65536]
            if index != 1:
                _require(bool(np.all(np.isfinite(block))), "nonfinite endpoint words")
            digest.update(block.tobytes())
    _require(
        all(not array.flags.writeable for array in arrays), "owner became writeable"
    )
    return SerializedPayloadBinding(metadata, digest.hexdigest(), total)


def import_declaration_bytes(
    *, transfer_id: str, parent_materialization_id: str, serialized_payload_sha256: str
) -> bytes:
    """Encode the fixed lane declaration from already verified graph digests.

    This primitive does not verify a graph. Its future consumer must pass actual
    recomputed transfer/P1/E IDs after checking profile/frame/producer joins,
    and compare supplied declaration bytes to this result before acceptance.
    """
    for value in (transfer_id, parent_materialization_id, serialized_payload_sha256):
        _require(
            type(value) is str
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value),
            "expected lowercase SHA256",
        )
    return _json(
        {
            "schema_version": "radiosim.native-import-declaration.v1",
            "source_attribute": "radiosim_polarization_materialization",
            "source_profile": _PROFILE,
            "output_profile": "radiosim_ne_iau_v1",
            "coordinate_frame": "icrs",
            "producer": {
                "library": "pyradiosky",
                "version": "1.1.0",
                "writer_contract": "radiosim-native-skyh5-v1",
            },
            "transfer_id": transfer_id,
            "parent_materialization_id": parent_materialization_id,
            "serialized_payload_sha256": serialized_payload_sha256,
        }
    )
