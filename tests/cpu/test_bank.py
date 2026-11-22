"""The bank: every policy folder under mp4_rl/skills is registered, and every registered one is self-contained."""
import pytest

from mp4_rl import paths, skills

PARTS = ("README.md", "__init__.py", "train.py", "eval", "recipes", "policies", "checkpoints")


def test_every_policy_folder_is_registered():
    on_disk = sorted(p.name for p in paths.SKILLS.iterdir() if (p / "__init__.py").is_file())
    assert on_disk == sorted(skills.REGISTRY)


@pytest.mark.parametrize("name", sorted(skills.REGISTRY))
def test_a_registered_policy_holds_its_parts(name):
    folder = skills.folder(name)
    assert [p for p in PARTS if not (folder / p).exists()] == []
    for robot in skills.ROBOTS:
        assert (folder / "policies" / robot / "policy.npz").is_file()
        assert (folder / "recipes" / robot).is_dir() and (folder / "checkpoints" / robot).is_dir()
        assert (folder / "eval" / f"{robot}.sh").is_file()


def test_the_runner_defaults_are_registered_folders():
    assert paths.WALKER_POLICIES / "sku2" == skills.policy_folder("walker", "sku2")
    assert paths.GETUP_POLICIES / "sku2" == skills.policy_folder("getup", "sku2")
    with pytest.raises(KeyError):
        skills.folder("dance")
