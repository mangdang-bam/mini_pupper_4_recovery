"""The sequence evaluation builds the get-up's thigh-to-base contact as its training does (off when the measured limits
are on)."""
from mp4_rl.getup.eval.sequence import effective_thigh_base


def test_limits_on_drops_thigh_base():
    assert effective_thigh_base({"thigh_base": True, "limits": True}) is False


def test_limits_off_keeps_option():
    assert effective_thigh_base({"thigh_base": True, "limits": False}) is True
    assert effective_thigh_base({"thigh_base": False, "limits": False}) is False
    assert effective_thigh_base({}) is True
