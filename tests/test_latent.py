import math
import pytest
from motor_owa.latent import (phi, phi_inv, octile_orness, octile_z,
                              classify_z, classify_orness,
                              latent_from_dimensions)
from motor_owa.config import DIMENSIONS


def test_phi_roundtrip():
    for p in [0.01, 0.0625, 0.5, 0.9375, 0.99]:
        assert phi(phi_inv(p)) == pytest.approx(p, abs=1e-9)


def test_octile_orness_values():
    expected = [1/16, 3/16, 5/16, 7/16, 9/16, 11/16, 13/16, 15/16]
    got = [octile_orness(k) for k in range(1, 9)]
    assert got == pytest.approx(expected)


def test_octile_symmetry():
    # simetria: orness_k + orness_(9-k) = 1; z_k = -z_(9-k)
    for k in range(1, 9):
        assert octile_orness(k) + octile_orness(9 - k) == pytest.approx(1.0)
        assert octile_z(k) == pytest.approx(-octile_z(9 - k), abs=1e-9)


def test_equispaced():
    d = [octile_orness(k + 1) - octile_orness(k) for k in range(1, 8)]
    assert all(x == pytest.approx(1/8) for x in d)


def test_classify_z_matches_own_anchor():
    for k in range(1, 9):
        assert classify_z(octile_z(k)) == k


def test_classify_borders():
    assert classify_z(-10) == 1
    assert classify_z(10) == 8
    assert classify_orness(0.0) == 1
    assert classify_orness(1.0) == 8


def test_latent_neutral_investor():
    dirs = DIMENSIONS
    neutral = {k: 3.0 for k in dirs}       # punto medio Likert 1-5
    z = latent_from_dimensions(neutral, dirs)
    assert z == pytest.approx(0.0, abs=1e-9)
    assert classify_z(z) in (4, 5)


def test_latent_extremes_land_in_extreme_octiles():
    dirs = DIMENSIONS
    aggressive = {k: (5.0 if d > 0 else 1.0) for k, d in dirs.items()}
    conservative = {k: (1.0 if d > 0 else 5.0) for k, d in dirs.items()}
    assert classify_z(latent_from_dimensions(aggressive, dirs)) == 8
    assert classify_z(latent_from_dimensions(conservative, dirs)) == 1
