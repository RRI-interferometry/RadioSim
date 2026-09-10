"""Public export containment; no successful child export qualification."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.containers._polarization_materialization import (
    complete_model_native_identity,
)
from radiosim.core.sky.containers.constants import BrightnessConversion as BC
from radiosim.core.sky.containers.healpix import HealpixData
from radiosim.core.sky.containers.model import SkyModel
from radiosim.core.sky.containers.point import PointSourceData, TangentPolarizationFrame
from radiosim.core.sky.io import save_skyh5, to_pyradiosky


def _model(
    *, attached: bool = True, hybrid: bool = False
) -> tuple[SkyModel, NDArray[np.float64]]:
    backing = np.array([[2.0, 3.0], [4.0, 5.0]])
    raw = HealpixData(
        maps=backing.view(),
        q_maps=np.ones((2, 2)),
        u_maps=np.array([[0.5, 0.75], [1.0, 1.25]]),
        v_maps=np.full((2, 2), -0.25),
        frequencies=np.array([200e6, 100e6]),
        nside=1,
        hpx_inds=np.array([4, 9]),
    )
    point = (
        PointSourceData(
            ra_rad=np.array([0.1]),
            dec_rad=np.array([0.2]),
            flux=np.array([7.0]),
            spectral_index=np.array([-0.7]),
            stokes_q=np.zeros(1),
            stokes_u=np.zeros(1),
            stokes_v=np.zeros(1),
            ref_freq=np.array([100e6]),
        )
        if hybrid
        else None
    )
    sky = SkyModel(healpix=raw, point=point, precision=PrecisionConfig.precise())
    if attached:
        sky = complete_model_native_identity(
            sky,
            source_profile="radiosim_ne_iau_v1",
            tangent_frame=TangentPolarizationFrame.canonical("icrs"),
        )
    assert sky.healpix is not None
    assert backing.flags.writeable and np.shares_memory(backing, sky.healpix.maps)
    return sky, backing


def _snapshot(sky: SkyModel) -> tuple[object, ...]:
    owner = sky.healpix
    assert owner is not None
    arrays = (
        owner.maps,
        owner.q_maps,
        owner.u_maps,
        owner.v_maps,
        owner.frequencies,
        owner.hpx_inds,
    )
    states = tuple(
        None
        if a is None
        else (id(a), a.dtype.str, a.shape, a.strides, str(a.flags), a.tobytes())
        for a in arrays
    )
    frame = owner.tangent_polarization_frame
    evidence = owner.polarization_materialization
    evidence_frame = None if evidence is None else evidence.tangent_frame
    return (
        states,
        owner.nside,
        owner.ordering,
        owner.coordinate_frame,
        (owner.i_unit, owner.q_unit, owner.u_unit, owner.v_unit),
        owner.i_brightness_conversion,
        owner.q_brightness_conversion,
        owner.u_brightness_conversion,
        owner.v_brightness_conversion,
        sky.brightness_conversion,
        None if frame is None else (id(frame), frame.as_mapping()),
        None
        if evidence is None
        else (
            id(evidence),
            evidence.record.as_mapping(),
            evidence.declaration_json,
            evidence.identity_parameters_json,
            evidence.payload_metadata_json,
            evidence.brightness_conversion,
            None
            if evidence_frame is None
            else (id(evidence_frame), evidence_frame.as_mapping()),
        ),
    )


def _forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError("upstream constructor or writer reached")


def test_bound_export_refuses_before_constructor_and_preserves_owner() -> None:
    sky, _ = _model()
    before = _snapshot(sky)
    with patch("pyradiosky.SkyModel", side_effect=_forbidden):
        with pytest.raises(ValueError, match="Native export requires child"):
            _ = to_pyradiosky(sky)
    assert _snapshot(sky) == before


@pytest.mark.parametrize("stale", ["alias", "context"])
def test_stale_selected_native_checks_actual_owner_first(stale: str) -> None:
    sky, backing = _model()
    if stale == "alias":
        backing[0, 0] = 6.0
    else:
        # Explicit adversarial bypass of constructor validation.
        object.__setattr__(sky, "brightness_conversion", BC.RAYLEIGH_JEANS)
    before = _snapshot(sky)
    with patch("pyradiosky.SkyModel", side_effect=_forbidden):
        with pytest.raises(
            ValueError, match="native identity materialization mismatch"
        ):
            _ = to_pyradiosky(sky)
    assert _snapshot(sky) == before


@pytest.mark.parametrize("existing", [False, True])
def test_bound_save_refuses_without_creating_or_clobbering(
    tmp_path: Path,
    existing: bool,
) -> None:
    sky, _ = _model()
    destination = tmp_path / "native.skyh5"
    sentinel = b"existing destination must survive"
    if existing:
        _ = destination.write_bytes(sentinel)
    before = _snapshot(sky)
    with patch("pyradiosky.SkyModel.write_skyh5", side_effect=_forbidden):
        with pytest.raises(ValueError, match="Native export requires child"):
            save_skyh5(sky, str(destination), clobber=True)
    assert destination.exists() is existing
    if existing:
        assert destination.read_bytes() == sentinel
    assert _snapshot(sky) == before


def test_raw_sparse_unsorted_export_retains_literal_compatibility() -> None:
    sky, _ = _model(attached=False)
    before = _snapshot(sky)
    exported = to_pyradiosky(sky)
    np.testing.assert_array_equal(exported.hpx_inds, [4, 9])
    np.testing.assert_array_equal(exported.freq_array.to_value("Hz"), [100e6, 200e6])
    # This intentionally asserts existing raw U/V behavior, not profile repair.
    expected: NDArray[np.float64] = np.array(
        [
            [[4, 5], [2, 3]],
            [[1, 1], [1, 1]],
            [[1, 1.25], [0.5, 0.75]],
            [[-0.25, -0.25], [-0.25, -0.25]],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(exported.stokes.to_value("K"), expected)
    assert _snapshot(sky) == before


def test_hybrid_selection_remains_component_local() -> None:
    sky, backing = _model(hybrid=True)
    before = _snapshot(sky)
    with pytest.raises(ValueError, match="both point and HEALPix"):
        _ = to_pyradiosky(sky)
    with pytest.raises(ValueError, match="Native export requires child"):
        _ = to_pyradiosky(sky, representation="healpix_map")
    exported = to_pyradiosky(sky, representation="point_sources")
    assert exported.component_type == "point"
    np.testing.assert_array_equal(exported.stokes.to_value("Jy")[:, 0, 0], [7, 0, 0, 0])
    assert _snapshot(sky) == before
    backing[0, 0] = 6.0
    stale = _snapshot(sky)
    _ = to_pyradiosky(sky, representation="point_sources")
    assert _snapshot(sky) == stale


@pytest.mark.parametrize("mutation", ["frame", "operation", "sidecar"])
def test_snapshot_detaches_frame_and_nested_evidence(mutation: str) -> None:
    sky, _ = _model()
    owner = sky.healpix
    assert owner is not None
    evidence = owner.polarization_materialization
    assert evidence is not None
    before = _snapshot(sky)
    if mutation == "frame":
        frame = TangentPolarizationFrame.canonical("icrs")
        assert frame == owner.tangent_polarization_frame
        assert frame is not owner.tangent_polarization_frame
        object.__setattr__(owner, "tangent_polarization_frame", frame)
    elif mutation == "operation":
        object.__setattr__(evidence.record.operations[0], "kind", "not-identity")
    else:
        object.__setattr__(evidence, "declaration_json", b"changed sidecar")
    assert _snapshot(sky) != before
