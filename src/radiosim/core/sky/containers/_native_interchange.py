"""Unwired finite transport primitives, sorted-child consumer, and frequency-sorted copy; no export release."""

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import TypeGuard

import numpy as np
from numpy.typing import NDArray

from ._polarization_materialization import require_native_identity
from ._polarization_payload import bind_healpix_payload
from .constants import BrightnessConversion
from .healpix import HealpixData
from .point import TangentPolarizationFrame
from .polarization_materialization import (
    PolarizationMaterialization,
    PolarizationMaterializationEvidence,
    PolarizationOperation,
)

_PROFILE = "pyradiosky_1_1_0_theta_phi_v1"
_CANONICAL = "radiosim_ne_iau_v1"
_DOMAIN = b"RADIOSIM_PYRADIOSKY_HEALPIX_PAYLOAD_V1\n"
_TRANSFER_DOMAIN = b"RADIOSIM_NATIVE_PYRADIOSKY_TRANSFER_V1\n"
_MATERIALIZATION_DOMAIN = b"RADIOSIM_POLARIZATION_MATERIALIZATION_V1\n"


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


def _is_exact_tuple(value: object) -> TypeGuard[tuple[object, ...]]:
    return type(value) is tuple


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return type(value) is dict


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return type(value) is list


def _is_healpix(value: object) -> TypeGuard[HealpixData]:
    return isinstance(value, HealpixData)


def _is_identity_parent(
    value: object,
) -> TypeGuard[PolarizationMaterializationEvidence]:
    return type(value) is PolarizationMaterializationEvidence


def _is_tangent_frame(value: object) -> TypeGuard[TangentPolarizationFrame]:
    return type(value) is TangentPolarizationFrame


def _is_brightness_conversion(value: object) -> TypeGuard[BrightnessConversion]:
    return type(value) is BrightnessConversion


def _is_rayleigh_jeans(value: object) -> TypeGuard[BrightnessConversion]:
    return (
        type(value) is BrightnessConversion
        and value is BrightnessConversion.RAYLEIGH_JEANS
    )


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


def export_declaration_bytes(*, parent_materialization_id: object) -> bytes:
    """Encode the fixed sorted-child export declaration from a parent ID.

    This primitive does not verify a graph. Its future consumer must pass the
    actual recomputed P0 materialization ID after checking identity replay,
    and compare supplied declaration bytes to this result before acceptance.
    """
    if type(parent_materialization_id) is not str:
        raise ValueError("expected lowercase SHA256")
    _require(
        len(parent_materialization_id) == 64
        and all(c in "0123456789abcdef" for c in parent_materialization_id),
        "expected lowercase SHA256",
    )
    return _json(
        {
            "schema_version": "radiosim.native-export-declaration.v1",
            "parent_materialization_id": parent_materialization_id,
            "source_profile": "radiosim_ne_iau_v1",
            "output_profile": _PROFILE,
            "coordinate_frame": "icrs",
        }
    )


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


def transfer_record_bytes(
    *,
    parent_materialization_id: str,
    input_payload_sha256: str,
    output_payload_sha256: str,
    parameters_sha256: str,
) -> bytes:
    """Encode the ten-field transfer record from already verified digest labels.

    This primitive does not verify a graph. Its future consumer must pass actual
    recomputed P1/E IDs and the basis-parameter digest after checking profile
    joins, and compare supplied record bytes to this result before acceptance.
    Operation endpoints are bound to the supplied payload digests.
    """
    labels = (
        parent_materialization_id,
        input_payload_sha256,
        output_payload_sha256,
        parameters_sha256,
    )
    for value in labels:
        _require(
            type(value) is str
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value),
            "expected lowercase SHA256",
        )
    parent, incoming, outgoing, parameters = labels
    record = {
        "schema_version": "radiosim.native-pyradiosky-transfer.v1",
        "component_kind": "healpix",
        "input_profile": "radiosim_ne_iau_v1",
        "output_profile": _PROFILE,
        "coordinate_frame": "icrs",
        "parent_materialization_id": parent,
        "input_payload_sha256": incoming,
        "output_payload_sha256": outgoing,
        "operation": {
            "kind": "basis_profile_conversion",
            "input_sha256": incoming,
            "output_sha256": outgoing,
            "parameters_sha256": parameters,
        },
    }
    encoded = _json(record)
    record["transfer_id"] = hashlib.sha256(
        _TRANSFER_DOMAIN + struct.pack("<Q", len(encoded)) + encoded
    ).hexdigest()
    return _json(record)


def frequency_permutation_bytes(
    *, source_indices: object, input_frequency_words: object
) -> bytes:
    """Encode the closed frequency-permutation parameters from actual labels.

    This primitive does not permute arrays or authenticate a parent. The
    consumer must pass the actual stable argsort and little-endian word hex
    of the unsorted frequencies, then compare supplied parameter bytes to
    this result before acceptance.
    """
    if not _is_exact_tuple(source_indices):
        raise ValueError("expected exact index tuple")
    if not _is_exact_tuple(input_frequency_words):
        raise ValueError("expected exact word tuple")
    count = len(source_indices)
    _require(count == len(input_frequency_words), "permutation length differs")
    _require(1 <= count <= 1024, "endpoint dimension limit")
    indices: list[int] = []
    for index in source_indices:
        if type(index) is not int:
            raise ValueError("invalid source index")
        _require(0 <= index < count, "invalid source index")
        indices.append(index)
    checked = tuple(indices)
    _require(len(set(checked)) == count, "source indices are not a bijection")
    values: list[float] = []
    words: list[str] = []
    for word in input_frequency_words:
        if type(word) is not str:
            raise ValueError("expected lowercase <f8 word hex")
        _require(
            len(word) == 16
            and all(character in "0123456789abcdef" for character in word),
            "expected lowercase <f8 word hex",
        )
        value = struct.unpack("<d", bytes.fromhex(word))[0]
        _require(math.isfinite(value) and value > 0, "invalid frequencies")
        values.append(value)
        words.append(word)
    expected = tuple(sorted(range(count), key=values.__getitem__))
    _require(checked == expected, "source indices are not the stable sort")
    output_words = [words[index] for index in checked]
    output_values = [values[index] for index in checked]
    _require(
        all(
            left < right
            for left, right in zip(output_values, output_values[1:], strict=False)
        ),
        "frequencies not strictly sorted",
    )
    return _json(
        {
            "schema_version": "radiosim.native-frequency-permutation.v1",
            "algorithm": "stable_frequency_sort_v1",
            "axis": "frequency",
            "source_indices": list(checked),
            "input_frequency_words": list(words),
            "output_frequency_words": output_words,
            "pixel_order": "preserve_physical_id_sequence",
            "arithmetic": "none",
        }
    )


def basis_profile_conversion_bytes(*, direction: object) -> bytes:
    """Encode closed basis-profile-conversion parameters from a direction.

    This primitive does not adapt arrays or authenticate a parent. The
    consumer must pass the actual export or import direction after checking
    profile joins, then compare supplied parameter bytes to this result
    before acceptance.
    """
    if type(direction) is not str:
        raise ValueError("expected exact direction")
    _require(direction in ("export", "import"), "unsupported conversion direction")
    canonical = "radiosim_ne_iau_v1"
    theta_phi = _PROFILE
    if direction == "export":
        incoming, outgoing = canonical, theta_phi
    else:
        incoming, outgoing = theta_phi, canonical
    return _json(
        {
            "schema_version": "radiosim.native-basis-profile-conversion.v1",
            "algorithm": "pyradiosky_1_1_0_ne_theta_phi_v1",
            "direction": direction,
            "input_profile": incoming,
            "output_profile": outgoing,
            "coordinate_frame": "icrs",
            "signs": [1, 1, -1, 1],
            "stokes_axis_order": ["I", "Q", "U", "V"],
            "frequency_action": "preserve",
            "pixel_action": "preserve",
            "units_action": "preserve_K_RJ",
            "storage_action": "preserve_f64le",
            "tensor_layout": "stokes_frequency_pixel",
        }
    )


@dataclass(frozen=True, slots=True)
class NativeChainEvidence:
    """Untrusted depth-1 sorted-child carrier; bind_sorted_canonical_child validates."""

    record: PolarizationMaterialization
    tangent_frame: TangentPolarizationFrame | None
    declaration_json: bytes
    operation_parameters_json: tuple[bytes, ...]
    input_payload_metadata_json: bytes
    output_payload_metadata_json: bytes
    brightness_conversion: BrightnessConversion
    parent_evidence: PolarizationMaterializationEvidence
    transfer_evidence: object


def _is_chain_evidence(value: object) -> TypeGuard[NativeChainEvidence]:
    return type(value) is NativeChainEvidence


def bind_sorted_canonical_child(
    owner: object,
    *,
    brightness_conversion: object,
    source_profile: object,
    tangent_frame: object,
    parent_evidence: object,
    source_indices: object,
    output_payload_sha256: object,
    output_payload_metadata_json: object,
) -> NativeChainEvidence:
    """Bind a depth-1 sorted child after replaying identity on the parent owner.

    This factory does not permute arrays, attach evidence, or dispatch owner
    validation. The caller still excludes alias mutation for the full call.
    """
    if not _is_healpix(owner):
        raise ValueError("expected HealpixData")
    if not _is_identity_parent(parent_evidence):
        raise ValueError("expected identity parent evidence")
    if not _is_rayleigh_jeans(brightness_conversion):
        raise ValueError("sorted child requires rayleigh-jeans")
    if type(source_profile) is not str or source_profile != _CANONICAL:
        raise ValueError("identity requires the explicit canonical source profile")
    if not _is_tangent_frame(tangent_frame):
        raise ValueError("expected a canonical tangent frame")
    if type(output_payload_metadata_json) is not bytes:
        raise ValueError("expected output payload metadata bytes")
    _require(
        output_payload_metadata_json != b"", "expected output payload metadata bytes"
    )
    if type(output_payload_sha256) is not str:
        raise ValueError("expected lowercase SHA256")
    _require(
        len(output_payload_sha256) == 64
        and all(c in "0123456789abcdef" for c in output_payload_sha256),
        "expected lowercase SHA256",
    )
    require_native_identity(
        owner,
        brightness_conversion=brightness_conversion,
        source_profile=_CANONICAL,
        tangent_frame=tangent_frame,
        expected=parent_evidence,
    )
    _require(owner.coordinate_frame == "icrs", "expected icrs owner")
    frequencies = owner.frequencies
    _require(type(frequencies) is np.ndarray, "expected exact ndarray")
    _require(frequencies.dtype.str == "<f8", "unsupported endpoint dtype")
    _require(
        not frequencies.flags.writeable and frequencies.flags.c_contiguous,
        "readonly C owner required",
    )
    blob = bytes(frequencies.tobytes())
    words = tuple(blob[index : index + 8].hex() for index in range(0, len(blob), 8))
    permutation = frequency_permutation_bytes(
        source_indices=source_indices, input_frequency_words=words
    )
    parent_id = parent_evidence.record.materialization_id
    incoming = parent_evidence.record.output_payload_sha256
    declaration = export_declaration_bytes(parent_materialization_id=parent_id)
    operation = PolarizationOperation(
        "frequency_selection",
        incoming,
        output_payload_sha256,
        hashlib.sha256(permutation).hexdigest(),
    )
    declaration_digest = hashlib.sha256(declaration).hexdigest()
    body = {
        "schema_version": "radiosim.polarization-materialization.v1",
        "component_kind": "healpix",
        "source_profile": _CANONICAL,
        "declaration_origin": "inherited_normalized",
        "declaration_digest": declaration_digest,
        "source_frame": "icrs",
        "output_frame": "icrs",
        "input_payload_sha256": incoming,
        "output_payload_sha256": output_payload_sha256,
        "operations": [operation.as_mapping()],
        "parent_materialization_ids": [parent_id],
    }
    encoded = _json(body)
    _require(len(encoded) < 2**64, "materialization preimage length exceeds uint64")
    child_id = hashlib.sha256(
        _MATERIALIZATION_DOMAIN + struct.pack("<Q", len(encoded)) + encoded
    ).hexdigest()
    record = PolarizationMaterialization(
        "radiosim.polarization-materialization.v1",
        "healpix",
        _CANONICAL,
        "inherited_normalized",
        declaration_digest,
        "icrs",
        "icrs",
        incoming,
        output_payload_sha256,
        (operation,),
        (parent_id,),
        child_id,
    )
    return NativeChainEvidence(
        record,
        tangent_frame,
        declaration,
        (permutation,),
        parent_evidence.payload_metadata_json,
        output_payload_metadata_json,
        BrightnessConversion.RAYLEIGH_JEANS,
        parent_evidence,
        None,
    )


def _take_frequency_axis(array: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    return np.ascontiguousarray(array[np.asarray(indices, dtype=np.int64)])


def copy_frequency_sorted_healpix(owner: object) -> HealpixData:
    """Copy Stokes and frequency axes into stable frequency order.

    Parent arrays are not mutated. The copy is unbound: no tangent or
    materialization is attached. This does not pack a theta/phi tensor,
    wrap pyradiosky, or publish export.
    """
    if not _is_healpix(owner):
        raise ValueError("expected HealpixData")
    if owner.coordinate_frame != "icrs":
        raise ValueError("expected icrs owner")
    if owner.ordering != "ring":
        raise ValueError("expected ring owner")
    if (
        type(owner.i_unit) is not str
        or owner.i_unit != "K"
        or type(owner.q_unit) is not str
        or owner.q_unit != "K"
        or type(owner.u_unit) is not str
        or owner.u_unit != "K"
        or type(owner.v_unit) is not str
        or owner.v_unit != "K"
    ):
        raise ValueError("stored units must be K, including absent components")
    if (
        owner.i_brightness_conversion != "rayleigh-jeans"
        or owner.q_brightness_conversion != "rayleigh-jeans"
        or owner.u_brightness_conversion != "rayleigh-jeans"
        or owner.v_brightness_conversion != "rayleigh-jeans"
    ):
        raise ValueError("sorted child requires rayleigh-jeans")
    maps = owner.maps
    q_maps = owner.q_maps
    u_maps = owner.u_maps
    v_maps = owner.v_maps
    frequencies = owner.frequencies
    pixel_ids = owner.hpx_inds
    if q_maps is None or u_maps is None or v_maps is None:
        raise ValueError("sorted child requires all Stokes components")
    if pixel_ids is None:
        raise ValueError("sorted child requires explicit pixel IDs")
    arrays = (maps, q_maps, u_maps, v_maps, frequencies)
    for array in arrays:
        _require(type(array) is np.ndarray, "expected exact ndarray")
        _require(array.dtype.str == "<f8", "unsupported endpoint dtype")
        _require(
            not array.flags.writeable and array.flags.c_contiguous,
            "readonly C owner required",
        )
    _require(type(pixel_ids) is np.ndarray, "expected exact ndarray")
    _require(pixel_ids.dtype.str == "<i8", "unsupported endpoint dtype")
    _require(
        not pixel_ids.flags.writeable and pixel_ids.flags.c_contiguous,
        "readonly C owner required",
    )
    blob = bytes(frequencies.tobytes())
    words = tuple(blob[index : index + 8].hex() for index in range(0, len(blob), 8))
    values = [struct.unpack("<d", bytes.fromhex(word))[0] for word in words]
    indices = tuple(sorted(range(len(values)), key=values.__getitem__))
    _ = frequency_permutation_bytes(source_indices=indices, input_frequency_words=words)
    parent_words = (
        maps.tobytes(),
        q_maps.tobytes(),
        u_maps.tobytes(),
        v_maps.tobytes(),
        frequencies.tobytes(),
        pixel_ids.tobytes(),
    )
    copied = HealpixData(
        maps=_take_frequency_axis(maps, indices),
        q_maps=_take_frequency_axis(q_maps, indices),
        u_maps=_take_frequency_axis(u_maps, indices),
        v_maps=_take_frequency_axis(v_maps, indices),
        frequencies=_take_frequency_axis(frequencies, indices),
        nside=owner.nside,
        hpx_inds=pixel_ids,
        coordinate_frame=owner.coordinate_frame,
        ordering=owner.ordering,
        i_unit=owner.i_unit,
        q_unit=owner.q_unit,
        u_unit=owner.u_unit,
        v_unit=owner.v_unit,
        i_brightness_conversion=owner.i_brightness_conversion,
        q_brightness_conversion=owner.q_brightness_conversion,
        u_brightness_conversion=owner.u_brightness_conversion,
        v_brightness_conversion=owner.v_brightness_conversion,
    )
    _require(
        (
            owner.maps.tobytes(),
            q_maps.tobytes(),
            u_maps.tobytes(),
            v_maps.tobytes(),
            owner.frequencies.tobytes(),
            pixel_ids.tobytes(),
        )
        == parent_words,
        "parent mutated",
    )
    _require(copied.polarization_materialization is None, "copy must stay unbound")
    _require(copied.tangent_polarization_frame is None, "copy must stay unbound")
    copied_q = copied.q_maps
    copied_u = copied.u_maps
    copied_v = copied.v_maps
    if copied_q is None or copied_u is None or copied_v is None:
        raise ValueError("sorted child requires all Stokes components")
    _require(
        not np.shares_memory(copied.maps, maps)
        and not np.shares_memory(copied_q, q_maps)
        and not np.shares_memory(copied_u, u_maps)
        and not np.shares_memory(copied_v, v_maps)
        and not np.shares_memory(copied.frequencies, frequencies),
        "copy shares parent storage",
    )
    return copied


def _permutation_source_indices(permutation: bytes) -> tuple[int, ...]:
    decoded: object = json.loads(permutation.decode("utf-8"))
    if not _is_object_dict(decoded):
        raise ValueError("expected frequency permutation object")
    raw: object = decoded["source_indices"] if "source_indices" in decoded else None
    if not _is_object_list(raw):
        raise ValueError("invalid source index")
    checked: list[int] = []
    for item in raw:
        if type(item) is not int:
            raise ValueError("invalid source index")
        checked.append(item)
    return tuple(checked)


def _restore_frequency_axis(array: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    restored = np.empty_like(array)
    restored[np.asarray(indices, dtype=np.int64)] = array
    return np.ascontiguousarray(restored)


def _identity_owner_from_sorted(
    owner: HealpixData, indices: tuple[int, ...]
) -> HealpixData:
    maps = owner.maps
    q_maps = owner.q_maps
    u_maps = owner.u_maps
    v_maps = owner.v_maps
    if q_maps is None or u_maps is None or v_maps is None:
        raise ValueError("sorted child requires all Stokes components")
    _require(maps.shape[0] == len(indices), "permutation length differs")
    return HealpixData(
        maps=_restore_frequency_axis(maps, indices),
        q_maps=_restore_frequency_axis(q_maps, indices),
        u_maps=_restore_frequency_axis(u_maps, indices),
        v_maps=_restore_frequency_axis(v_maps, indices),
        frequencies=_restore_frequency_axis(owner.frequencies, indices),
        nside=owner.nside,
        hpx_inds=owner.hpx_inds,
        coordinate_frame=owner.coordinate_frame,
        ordering=owner.ordering,
        i_unit=owner.i_unit,
        q_unit=owner.q_unit,
        u_unit=owner.u_unit,
        v_unit=owner.v_unit,
        i_brightness_conversion=owner.i_brightness_conversion,
        q_brightness_conversion=owner.q_brightness_conversion,
        u_brightness_conversion=owner.u_brightness_conversion,
        v_brightness_conversion=owner.v_brightness_conversion,
    )


def _require_sorted_child(
    owner: HealpixData,
    *,
    brightness_conversion: BrightnessConversion,
    expected: NativeChainEvidence,
) -> None:
    transfer: object = expected.transfer_evidence
    if transfer is not None:
        raise ValueError("sorted child has no transfer")
    if not _is_identity_parent(expected.parent_evidence):
        raise ValueError("expected identity parent evidence")
    if not _is_tangent_frame(expected.tangent_frame):
        raise ValueError("expected a canonical tangent frame")
    if expected.brightness_conversion is not brightness_conversion:
        raise ValueError("native identity materialization mismatch")
    if owner.tangent_polarization_frame != expected.tangent_frame:
        raise ValueError("native identity materialization mismatch")
    parameters = expected.operation_parameters_json
    if not _is_exact_tuple(parameters) or len(parameters) != 1:
        raise ValueError("sorted child requires one frequency permutation")
    permutation = parameters[0]
    if type(permutation) is not bytes:
        raise ValueError("sorted child requires one frequency permutation")
    indices = _permutation_source_indices(permutation)
    parent_owner = _identity_owner_from_sorted(owner, indices)
    require_native_identity(
        parent_owner,
        brightness_conversion=brightness_conversion,
        source_profile=_CANONICAL,
        tangent_frame=expected.tangent_frame,
        expected=expected.parent_evidence,
    )
    payload = bind_healpix_payload(owner, brightness_conversion=brightness_conversion)
    rebuilt = bind_sorted_canonical_child(
        parent_owner,
        brightness_conversion=brightness_conversion,
        source_profile=_CANONICAL,
        tangent_frame=expected.tangent_frame,
        parent_evidence=expected.parent_evidence,
        source_indices=indices,
        output_payload_sha256=payload.payload_sha256,
        output_payload_metadata_json=payload.metadata_json,
    )
    if expected != rebuilt:
        raise ValueError("native chain materialization mismatch")


def require_native_materialization(
    owner: object,
    *,
    brightness_conversion: object,
    expected: object,
) -> None:
    """Dispatch identity or depth-1 sorted-child evidence by exact type.

    require_native_identity stays identity-only. This consumer does not export,
    permute a published owner, or accept a bag of operations without replay.
    """
    if not _is_healpix(owner):
        raise ValueError("expected HealpixData")
    if _is_identity_parent(expected):
        if not _is_brightness_conversion(brightness_conversion):
            raise ValueError("native identity materialization mismatch")
        require_native_identity(
            owner,
            brightness_conversion=brightness_conversion,
            source_profile=_CANONICAL,
            tangent_frame=owner.tangent_polarization_frame,
            expected=expected,
        )
        return
    if not _is_chain_evidence(expected):
        raise ValueError("native materialization requires typed evidence")
    if not _is_rayleigh_jeans(brightness_conversion):
        raise ValueError("sorted child requires rayleigh-jeans")
    _require_sorted_child(
        owner,
        brightness_conversion=brightness_conversion,
        expected=expected,
    )
