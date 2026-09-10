"""Independent transport words/declaration controls; no completed export claim."""

import hashlib
import json
import struct
from dataclasses import replace

import numpy as np

from radiosim.core.sky.containers._native_interchange import (
    SerializedNativePayload,
    bind_serialized_native,
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
