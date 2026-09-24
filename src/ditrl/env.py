"""One gymnasium environment for all DRL variants.

DRL1, DRL2 and every ablation cell differ only in the `EnvSpec` built from the config:
which features the agent observes, which depths it may choose, how often it decides, how the
seasonal budget is enforced, and how it is rewarded.

A decision step works like this: the chosen depth is applied on the first day, then the model
runs `interval - 1` more days without irrigation. The episode ends when AquaCrop reports the
season as finished (maturity, crop death or the configured season end).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from aquacrop import IrrigationManagement
from gymnasium import spaces

from ditrl import observations
from ditrl.sites import Site, new_model, season_result

UNLIMITED = 1e9  # budget used for full irrigation


@dataclass
class EnvSpec:
    site: Site
    budget: float | None                 # seasonal limit in mm, None = unlimited
    years: list[int]                     # a season draws one of these years
    observation: list[str]               # feature names, see observations.FEATURES
    depths: list[float] | None = None    # discrete action set in mm
    max_depth: float | None = None       # continuous action, 0..max_depth mm (if depths is None)
    interval: int = 5                    # days between decisions
    constraint: str = "mask"             # "mask": infeasible depths cannot be chosen
                                         # "clip": depth is cut to the remaining budget
    reward: str = "dense"                # "dense" | "terminal" | "profit"
    prices: dict = field(default_factory=dict)
    noise: dict = field(default_factory=dict)  # feature -> relative sd of observation noise
    scale: dict | None = None            # {"mean": [...], "std": [...]}: fixed normalization
    reward_scale: float = 1.0            # every reward is multiplied by this factor


class IrrigationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, spec: EnvSpec):
        if spec.constraint == "mask" and spec.depths is None:
            raise ValueError("an action mask needs a discrete action set (depths)")
        self.spec_ = spec
        self.budget = UNLIMITED if spec.budget is None else float(spec.budget)
        self.features = observations.resolve(spec.observation)
        n_obs = observations.width(self.features)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)
        if spec.depths is not None:
            self.action_space = spaces.Discrete(len(spec.depths))
        else:
            # Scaled action in [-1, 1], mapped linearly to 0..max_depth. This matches RLlib's
            # default `normalize_actions` used by Kelly et al.: an untrained policy (mean 0)
            # proposes half the maximum depth instead of 0 mm.
            self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.model = None
        self.decisions: list[dict] = []  # one record per decision, read by the evaluation

    # --- state shortcuts used by the observation features and heuristics ---------------------
    @property
    def state(self):
        return self.model._init_cond

    @property
    def remaining(self) -> float:
        return self.budget - self.used

    @property
    def done(self) -> bool:
        return bool(self.model._clock_struct.model_is_finished)

    # --- gymnasium API -------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        year = (options or {}).get("year") or int(self.np_random.choice(self.spec_.years))
        self.year = year
        # AquaCrop applies whatever depth we set on the next simulated day (method 5).
        irrigation = IrrigationManagement(irrigation_method=5, MaxIrr=1000)
        self.model = new_model(self.spec_.site, year, irrigation)
        self.model.run_model(num_steps=1, initialize_model=True)
        self.used = 0.0
        self.decisions = []
        return self._observe(), {}

    def action_masks(self) -> np.ndarray:
        """True for every depth that fits into the remaining budget. 0 mm always fits.
        MaskablePPO sets the probability of the other depths to zero before sampling."""
        return np.array([d <= self.remaining + 1e-9 for d in self.spec_.depths])

    def step(self, action):
        proposed = self._depth(action)
        self.step_mm(proposed)
        reward = self._reward() * self.spec_.reward_scale
        self.decisions[-1].update(proposed_mm=proposed, step_reward=reward)
        info = season_result(self.model) if self.done else {}
        return self._observe(), reward, self.done, False, info

    def step_mm(self, depth: float) -> float:
        """Apply `depth` mm (cut to the remaining budget) and simulate one decision interval.
        Heuristics call this directly; returns the depth actually applied."""
        applied = float(np.clip(depth, 0.0, max(self.remaining, 0.0)))
        self.decisions.append({"dap": int(self.state.dap), "remaining_mm": self.remaining,
                               "applied_mm": applied})
        self.used += applied
        self.model._param_struct.IrrMngt.depth = applied
        self.model.run_model(num_steps=1, initialize_model=False)
        self.model._param_struct.IrrMngt.depth = 0.0
        if not self.done and self.spec_.interval > 1:
            self.model.run_model(num_steps=self.spec_.interval - 1, initialize_model=False)
        return applied

    # --- helpers -------------------------------------------------------------------------------
    def _depth(self, action) -> float:
        if self.spec_.depths is not None:
            return float(self.spec_.depths[int(action)])
        a = float(np.clip(np.asarray(action).ravel()[0], -1.0, 1.0))
        return (a + 1.0) / 2.0 * self.spec_.max_depth

    def _reward(self) -> float:
        if self.done:
            result = season_result(self.model)
            if self.spec_.reward == "profit":
                return profit(result, self.spec_.prices)
            return result["yield_t_ha"]
        if self.spec_.reward == "dense":
            # relative evapotranspiration of the last simulated day
            et = observations.flux(self, "Tr") + observations.flux(self, "Es")
            et_pot = observations.flux(self, "TrPot") + observations.flux(self, "EsPot")
            return et / et_pot if et_pot > 0 else 0.0
        return 0.0

    def _observe(self) -> np.ndarray:
        if self.done:  # the final observation is never used for a decision
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        obs = observations.observe(self, self.features, self.spec_.noise, self.np_random)
        if self.spec_.scale is not None:
            obs = (obs - np.asarray(self.spec_.scale["mean"])) / np.asarray(self.spec_.scale["std"])
        return obs.astype(np.float32)


def profit(result: dict, prices: dict) -> float:
    """Net profit in USD per hectare at fixed prices (Kelly et al. 2024)."""
    return (prices["crop_usd_per_t"] * result["yield_t_ha"]
            - prices["irrigation_usd_per_mm"] * result["irrigation_mm"]
            - prices["fixed_usd_per_ha"])
