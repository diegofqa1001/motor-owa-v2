import math
import numpy as np
import pytest
from motor_owa.config import Anchors
from motor_owa.profiles import all_profiles, get_profile, PROFILE_NAMES
from motor_owa.owa import orness


def test_eight_profiles_ordered():
    for anchors in (Anchors.OCTILES, Anchors.ARTICULO2):
        profs = all_profiles(anchors)
        assert len(profs) == 8
        alphas = [p.alpha for p in profs]
        assert alphas == sorted(alphas)


def test_octile_anchor_values():
    profs = all_profiles(Anchors.OCTILES)
    assert profs[0].alpha == pytest.approx(0.0625)
    assert profs[7].alpha == pytest.approx(0.9375)
    assert profs[0].name == "Guardian" and profs[7].name == "Visionary"


def test_articulo2_anchor_values():
    profs = all_profiles(Anchors.ARTICULO2)
    assert profs[0].alpha == pytest.approx(0.158)
    assert profs[7].alpha == pytest.approx(0.865)


def test_weights_materialize_orness_n4_n7():
    for p in all_profiles(Anchors.OCTILES):
        for n in (4, 7):
            assert orness(p.weights(n)) == pytest.approx(p.alpha, abs=1e-8)


def test_z_consistent_with_alpha():
    from motor_owa.latent import phi
    for p in all_profiles(Anchors.OCTILES):
        assert phi(p.z) == pytest.approx(p.alpha, abs=1e-9)


def test_get_profile_by_name_and_index():
    assert get_profile("Visionary").k == 8
    assert get_profile(1).name == "Guardian"
    with pytest.raises(KeyError):
        get_profile("Nope")
