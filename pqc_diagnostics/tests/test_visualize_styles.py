import pytest

from pqc_diagnostics.visualize.styles import StyleRegistry


def test_register_and_get_round_trip():
    reg = StyleRegistry()
    reg.register("feature_map_family", "my_fm", color="#123456", marker="X")
    style = reg.get("feature_map_family", "my_fm")
    assert style.color == "#123456"
    assert style.marker == "X"
    assert style.linestyle is None


def test_registering_a_duplicate_without_overwrite_raises():
    reg = StyleRegistry()
    reg.register("ansatz_family", "dup", color="#000000")
    with pytest.raises(ValueError):
        reg.register("ansatz_family", "dup", color="#ffffff")


def test_overwrite_replaces_existing_style():
    reg = StyleRegistry()
    reg.register("ansatz_family", "replace_me", color="#000000")
    reg.register("ansatz_family", "replace_me", color="#ffffff", overwrite=True)
    assert reg.get("ansatz_family", "replace_me").color == "#ffffff"


def test_unregistered_name_falls_back_to_none_style_with_a_warning():
    reg = StyleRegistry()
    with pytest.warns(UserWarning):
        style = reg.get("feature_map_family", "never_registered")
    assert style.color is None
    assert style.marker is None
    assert style.linestyle is None
