"""Independent transport words/declaration controls; no completed export claim."""

import hashlib
import json
import struct
from dataclasses import replace

import numpy as np
import pytest

from radiosim.core.sky.containers._native_interchange import (
    SerializedNativePayload,
    basis_profile_conversion_bytes,
    bind_serialized_native,
    export_declaration_bytes,
    frequency_permutation_bytes,
    import_declaration_bytes,
    transfer_record_bytes,
)

_WORDS = (
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    11.0,
    12.0,
    -13.0,
    -14.0,
    -15.0,
    -16.0,
    -17.0,
    -0.0,
    19.0,
    20.0,
    21.0,
    22.0,
    23.0,
    24.0,
)


def _endpoint() -> SerializedNativePayload:
    arrays = (
        np.array([80e6, 120e6], dtype="<f8"),
        np.array([4, 0, 9], dtype="<i8"),
        np.array(_WORDS, dtype="<f8").reshape(4, 2, 3),
    )
    for array in arrays:
        array.flags.writeable = False
    return SerializedNativePayload(
        1,
        *arrays,
        "pyradiosky_1_1_0_theta_phi_v1",
        "icrs",
        "ring",
        "healpix",
        "full",
        "Hz",
        "K",
        "rayleigh-jeans",
    )


def test_serialized_endpoint_independent_complete_preimage() -> None:
    owner = _endpoint()
    metadata = {
        "schema_version": "radiosim.pyradiosky-healpix-payload-metadata.v1",
        "profile": "pyradiosky_1_1_0_theta_phi_v1",
        "component_type": "healpix",
        "spectral_type": "full",
        "coordinate_frame": "icrs",
        "ordering": "ring",
        "nside": 1,
        "units": {"frequencies": "Hz", "pixel_ids": "1", "stokes": "K"},
        "brightness_conversion": "rayleigh-jeans",
        "stokes_axis_order": ["I", "Q", "U", "V"],
        "arrays": [
            {"name": "frequencies", "dtype": "<f8", "shape": [2], "byte_count": 16},
            {"name": "pixel_ids", "dtype": "<i8", "shape": [3], "byte_count": 24},
            {"name": "stokes", "dtype": "<f8", "shape": [4, 2, 3], "byte_count": 192},
        ],
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    preimage = b"RADIOSIM_PYRADIOSKY_HEALPIX_PAYLOAD_V1\n"
    for value in (
        encoded,
        struct.pack("<2d", 80e6, 120e6),
        struct.pack("<3q", 4, 0, 9),
        struct.pack("<24d", *_WORDS),
    ):
        preimage += struct.pack("<Q", len(value)) + value
    arrays = (owner.frequencies, owner.pixel_ids, owner.stokes)
    before = [(id(a), a.dtype.str, a.shape, a.strides, a.tobytes()) for a in arrays]
    result = bind_serialized_native(owner)
    assert result.metadata_json == encoded
    assert result.preimage_byte_count == len(preimage)
    assert result.payload_sha256 == hashlib.sha256(preimage).hexdigest()
    assert before == [
        (id(a), a.dtype.str, a.shape, a.strides, a.tobytes()) for a in arrays
    ]
    assert all(not a.flags.writeable for a in arrays)
    # Zero's sign is a payload word, even when numeric array equality holds.
    changed = owner.stokes.copy()
    changed[2, 1, 2] = 0.0
    changed.flags.writeable = False
    assert np.array_equal(changed, owner.stokes)
    assert bind_serialized_native(replace(owner, stokes=changed)) != result


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_id",
        "id_range",
        "frequency_order",
        "frequency_nan",
        "stokes_nan",
        "dtype",
        "writeable",
        "shape",
        "profile",
        "context",
        "nside",
    ],
)
def test_serialized_endpoint_refuses_invalid_actual_input(mutation: str) -> None:
    owner = _endpoint()
    if mutation in ("profile", "context", "nside"):
        key, value = {
            "profile": ("profile", "radiosim_ne_iau_v1"),
            "context": ("brightness_conversion", "planck"),
            "nside": ("nside", 3),
        }[mutation]
        owner = replace(owner, **{key: value})
    else:
        name = (
            "pixel_ids"
            if mutation.startswith("id_") or mutation == "duplicate_id"
            else "frequencies"
            if mutation.startswith("frequency")
            else "stokes"
        )
        array = getattr(owner, name).copy()
        if mutation == "duplicate_id":
            array[1] = 4
        elif mutation == "id_range":
            array[1] = 12
        elif mutation == "frequency_order":
            array[:] = [120e6, 80e6]
        elif mutation in ("frequency_nan", "stokes_nan"):
            array.flat[0] = np.nan
        elif mutation == "dtype":
            array = array.astype("<f4")
        elif mutation == "shape":
            array = array[:3]
        if mutation != "writeable":
            array.flags.writeable = False
        owner = replace(owner, **{name: array})
    with pytest.raises(ValueError):
        _ = bind_serialized_native(owner)


def test_import_declaration_matches_independent_literal_bytes() -> None:
    expected = {
        "schema_version": "radiosim.native-import-declaration.v1",
        "source_attribute": "radiosim_polarization_materialization",
        "source_profile": "pyradiosky_1_1_0_theta_phi_v1",
        "output_profile": "radiosim_ne_iau_v1",
        "coordinate_frame": "icrs",
        "producer": {
            "library": "pyradiosky",
            "version": "1.1.0",
            "writer_contract": "radiosim-native-skyh5-v1",
        },
        "transfer_id": "11" * 32,
        "parent_materialization_id": "33" * 32,
        "serialized_payload_sha256": "22" * 32,
    }
    actual = import_declaration_bytes(
        transfer_id="11" * 32,
        parent_materialization_id="33" * 32,
        serialized_payload_sha256="22" * 32,
    )
    assert (
        actual == json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    )
    assert (
        hashlib.sha256(actual).hexdigest()
        == "e86d3ac0474dc5253f82d4c5911824b038c0c99af40663ad798314dac6c9852c"
    )
    # Symbolic digest-label oracle only; neither call authenticates a graph.
    assert (
        import_declaration_bytes(
            transfer_id="11" * 32,
            parent_materialization_id="44" * 32,
            serialized_payload_sha256="22" * 32,
        )
        != actual
    )


@pytest.mark.parametrize("value", ["A" * 64, "0" * 63, "g" * 64])
def test_import_declaration_refuses_noncanonical_digest(value: str) -> None:
    with pytest.raises(ValueError, match="lowercase SHA256"):
        _ = import_declaration_bytes(
            transfer_id=value,
            parent_materialization_id="33" * 32,
            serialized_payload_sha256="22" * 32,
        )


def test_transfer_record_matches_independent_literal_bytes() -> None:
    parent = "11" * 32
    incoming = "22" * 32
    outgoing = "33" * 32
    parameters = "44" * 32
    nine = {
        "schema_version": "radiosim.native-pyradiosky-transfer.v1",
        "component_kind": "healpix",
        "input_profile": "radiosim_ne_iau_v1",
        "output_profile": "pyradiosky_1_1_0_theta_phi_v1",
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
    encoded = json.dumps(
        nine,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    preimage = (
        b"RADIOSIM_NATIVE_PYRADIOSKY_TRANSFER_V1\n"
        + struct.pack("<Q", len(encoded))
        + encoded
    )
    transfer_id = hashlib.sha256(preimage).hexdigest()
    expected = dict(nine)
    expected["transfer_id"] = transfer_id
    actual = transfer_record_bytes(
        parent_materialization_id=parent,
        input_payload_sha256=incoming,
        output_payload_sha256=outgoing,
        parameters_sha256=parameters,
    )
    assert actual == json.dumps(
        expected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert json.loads(actual)["operation"]["input_sha256"] == incoming
    assert json.loads(actual)["operation"]["output_sha256"] == outgoing
    # Symbolic digest-label oracle only; neither call authenticates a graph.
    assert (
        transfer_record_bytes(
            parent_materialization_id="55" * 32,
            input_payload_sha256=incoming,
            output_payload_sha256=outgoing,
            parameters_sha256=parameters,
        )
        != actual
    )


@pytest.mark.parametrize(
    "field",
    [
        "parent_materialization_id",
        "input_payload_sha256",
        "output_payload_sha256",
        "parameters_sha256",
    ],
)
@pytest.mark.parametrize("value", ["A" * 64, "0" * 63, "g" * 64])
def test_transfer_record_refuses_noncanonical_digest(field: str, value: str) -> None:
    kwargs = {
        "parent_materialization_id": "11" * 32,
        "input_payload_sha256": "22" * 32,
        "output_payload_sha256": "33" * 32,
        "parameters_sha256": "44" * 32,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="lowercase SHA256"):
        _ = transfer_record_bytes(**kwargs)


def test_frequency_permutation_matches_independent_literal_bytes() -> None:
    frequencies = (120e6, 80e6, 100e6)
    words = tuple(struct.pack("<d", value).hex() for value in frequencies)
    indices = (1, 2, 0)
    expected = {
        "schema_version": "radiosim.native-frequency-permutation.v1",
        "algorithm": "stable_frequency_sort_v1",
        "axis": "frequency",
        "source_indices": [1, 2, 0],
        "input_frequency_words": list(words),
        "output_frequency_words": [words[1], words[2], words[0]],
        "pixel_order": "preserve_physical_id_sequence",
        "arithmetic": "none",
    }
    actual = frequency_permutation_bytes(
        source_indices=indices, input_frequency_words=words
    )
    assert actual == json.dumps(
        expected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    identity = tuple(struct.pack("<d", value).hex() for value in (80e6, 100e6, 120e6))
    assert (
        frequency_permutation_bytes(
            source_indices=(0, 1, 2), input_frequency_words=identity
        )
        != actual
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "bool_index",
        "duplicate_frequency",
        "wrong_permutation",
        "uppercase_hex",
        "short_hex",
        "frequency_nan",
        "frequency_nonpositive",
        "empty",
        "length_mismatch",
    ],
)
def test_frequency_permutation_refuses_invalid_actual_input(mutation: str) -> None:
    words = tuple(struct.pack("<d", value).hex() for value in (120e6, 80e6, 100e6))
    indices: object = (1, 2, 0)
    payload: object = words
    if mutation == "bool_index":
        indices = (True, 2, 0)
    elif mutation == "duplicate_frequency":
        payload = (words[0], words[0], words[2])
        indices = (0, 1, 2)
    elif mutation == "wrong_permutation":
        indices = (2, 1, 0)
    elif mutation == "uppercase_hex":
        payload = (words[0].upper(), words[1], words[2])
    elif mutation == "short_hex":
        payload = (words[0][:15], words[1], words[2])
    elif mutation == "frequency_nan":
        payload = (struct.pack("<d", float("nan")).hex(), words[1], words[2])
    elif mutation == "frequency_nonpositive":
        payload = (struct.pack("<d", 0.0).hex(), words[1], words[2])
    elif mutation == "empty":
        indices = ()
        payload = ()
    else:
        payload = words[:2]
    with pytest.raises(ValueError):
        _ = frequency_permutation_bytes(
            source_indices=indices, input_frequency_words=payload
        )


def test_basis_profile_conversion_matches_independent_literal_bytes() -> None:
    export_expected = {
        "schema_version": "radiosim.native-basis-profile-conversion.v1",
        "algorithm": "pyradiosky_1_1_0_ne_theta_phi_v1",
        "direction": "export",
        "input_profile": "radiosim_ne_iau_v1",
        "output_profile": "pyradiosky_1_1_0_theta_phi_v1",
        "coordinate_frame": "icrs",
        "signs": [1, 1, -1, 1],
        "stokes_axis_order": ["I", "Q", "U", "V"],
        "frequency_action": "preserve",
        "pixel_action": "preserve",
        "units_action": "preserve_K_RJ",
        "storage_action": "preserve_f64le",
        "tensor_layout": "stokes_frequency_pixel",
    }
    import_expected = dict(export_expected)
    import_expected["direction"] = "import"
    import_expected["input_profile"] = "pyradiosky_1_1_0_theta_phi_v1"
    import_expected["output_profile"] = "radiosim_ne_iau_v1"
    export_actual = basis_profile_conversion_bytes(direction="export")
    import_actual = basis_profile_conversion_bytes(direction="import")
    assert export_actual == json.dumps(
        export_expected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert import_actual == json.dumps(
        import_expected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert export_actual != import_actual
    assert json.loads(export_actual)["signs"] == [1, 1, -1, 1]
    assert json.loads(export_actual)["stokes_axis_order"] == ["I", "Q", "U", "V"]


@pytest.mark.parametrize(
    "mutation",
    [
        "bool_direction",
        "uppercase",
        "empty",
        "unknown",
        "bytes_direction",
        "whitespace",
    ],
)
def test_basis_profile_conversion_refuses_invalid_actual_input(mutation: str) -> None:
    payload: object = "export"
    if mutation == "bool_direction":
        payload = True
    elif mutation == "uppercase":
        payload = "Export"
    elif mutation == "empty":
        payload = ""
    elif mutation == "unknown":
        payload = "convert"
    elif mutation == "bytes_direction":
        payload = b"export"
    else:
        payload = "export "
    with pytest.raises(ValueError):
        _ = basis_profile_conversion_bytes(direction=payload)


def test_export_declaration_matches_independent_literal_bytes() -> None:
    expected = {
        "schema_version": "radiosim.native-export-declaration.v1",
        "parent_materialization_id": "33" * 32,
        "source_profile": "radiosim_ne_iau_v1",
        "output_profile": "pyradiosky_1_1_0_theta_phi_v1",
        "coordinate_frame": "icrs",
    }
    actual = export_declaration_bytes(parent_materialization_id="33" * 32)
    assert actual == json.dumps(
        expected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert (
        hashlib.sha256(actual).hexdigest()
        == "30f90656df07d32a1bac2f10a849d2d02df34c60db68c7731bcc8305dc840cbb"
    )
    # Symbolic digest-label oracle only; neither call authenticates a graph.
    assert export_declaration_bytes(parent_materialization_id="11" * 32) != actual


@pytest.mark.parametrize(
    "mutation",
    [
        "bool_parent",
        "uppercase",
        "short",
        "invalid_hex",
        "bytes_parent",
        "empty",
        "whitespace",
    ],
)
def test_export_declaration_refuses_invalid_actual_input(mutation: str) -> None:
    payload: object = "33" * 32
    if mutation == "bool_parent":
        payload = True
    elif mutation == "uppercase":
        payload = "A" * 64
    elif mutation == "short":
        payload = "0" * 63
    elif mutation == "invalid_hex":
        payload = "g" * 64
    elif mutation == "bytes_parent":
        payload = b"33" * 32
    elif mutation == "empty":
        payload = ""
    else:
        payload = "33" * 32 + " "
    with pytest.raises(ValueError):
        _ = export_declaration_bytes(parent_materialization_id=payload)
