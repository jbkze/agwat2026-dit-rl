"""Minimal checks of the logic that silently breaks results if it is wrong: the budget in the
oracle decoding, the action mask and the budget cut in the environment, and the Holm correction.

    .venv/Scripts/python tests/test_core.py      (or: pytest tests)
"""

import numpy as np

from analysis.stats import holm
from ditrl.config import agent_config, load_sites
from ditrl.env import EnvSpec, IrrigationEnv
from ditrl.oracle import decode


def test_decode_respects_budget():
    x = np.array([10.0, 10.4, 50.0, 40.0, 30.0, 60.0])  # days 10, 10, 50; depths 40, 30, 60
    daily = decode(x, n_events=3, window=134, budget=100.0)
    assert np.isclose(daily.sum(), 100.0)
    assert np.isclose(daily[10] / daily[50], 70 / 60)  # same-day events add, scaling is uniform
    assert decode(x, 3, 134, budget=0.0).sum() == 0.0
    assert np.isclose(decode(x, 3, 134, budget=500.0).sum(), 130.0)  # below budget: unchanged


def test_mask_and_budget_cut():
    site = load_sites()["sites"]["tunis"]
    cfg = agent_config("drl1", "tunis")
    env = IrrigationEnv(EnvSpec(site=site, budget=40.0, years=[1980], observation="DRL1",
                                depths=cfg["depths"], interval=5, constraint="mask"))
    env.reset(options={"year": 1980})
    assert env.action_masks().tolist() == [True] * 9  # 34 mm fits into 40 mm
    env.step(cfg["depths"].index(34))
    assert env.action_masks().tolist() == [True] * 5 + [False] * 4  # 6 mm left: up to 5 mm
    env.step_mm(21)  # a heuristic asking for more than is left gets the rest only
    assert np.isclose(env.used, 40.0) and env.remaining == 0.0


def test_reject_applies_nothing():
    site = load_sites()["sites"]["tunis"]
    cfg = agent_config("drl1_reject", "tunis")
    env = IrrigationEnv(EnvSpec(site=site, budget=40.0, years=[1980], observation="DRL1",
                                depths=cfg["depths"], interval=5, constraint=cfg["constraint"]))
    env.reset(options={"year": 1980})
    env.step(cfg["depths"].index(34))  # fits: applied in full
    env.step(cfg["depths"].index(8))   # 6 mm left: rejected, not cut to 6 mm
    assert np.isclose(env.used, 34.0)
    assert env.decisions[-1]["proposed_mm"] == 8.0 and env.decisions[-1]["applied_mm"] == 0.0


def test_holm():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
