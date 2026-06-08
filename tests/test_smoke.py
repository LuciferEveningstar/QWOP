"""Smoke tests — confirm the package is importable and basic invariants hold."""

import qwop_rl


def test_package_has_version() -> None:
    assert hasattr(qwop_rl, "__version__")
    assert isinstance(qwop_rl.__version__, str)
    assert qwop_rl.__version__ != ""


def test_subpackages_importable() -> None:
    """All declared subpackages must import without side effects."""
    import qwop_rl.agents
    import qwop_rl.envs
    import qwop_rl.utils  # noqa: F401
