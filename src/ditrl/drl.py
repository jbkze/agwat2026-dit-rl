"""Training and evaluation of PPO agents.

With `constraint: mask` the agent is a MaskablePPO from sb3-contrib; otherwise it is a plain
PPO from stable-baselines3. Hyperparameters are the library defaults unless the agent config
sets them under `ppo`. The values actually used are written to `hyperparameters.json`.

During training the policy is evaluated on the validation years every `eval_every` steps. The
checkpoint with the highest mean validation yield is kept and later evaluated on the test years.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from ditrl.env import EnvSpec, IrrigationEnv

log = logging.getLogger(__name__)

HYPERPARAMETERS = ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
                   "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "policy_kwargs", "seed")


def run_policy(model, env: IrrigationEnv, year: int) -> dict:
    """One deterministic season. Returns yield, water use and the decision log."""
    obs, _ = env.reset(options={"year": year})
    masked = isinstance(model, MaskablePPO)
    done, info = False, {}
    while not done:
        kwargs = {"action_masks": env.action_masks()} if masked else {}
        action, _ = model.predict(obs, deterministic=True, **kwargs)
        obs, _, done, _, info = env.step(action)
    return {"year": year, **info, "decisions": env.decisions}


def evaluate(model, spec: EnvSpec, years: list[int], seed: int | None = None) -> list[dict]:
    """Deterministic seasons for all `years`. `seed` fixes the observation noise, if any."""
    env = IrrigationEnv(spec)
    env.reset(seed=seed)
    return [run_policy(model, env, y) for y in years]


class ValidationCallback(BaseCallback):
    """Evaluates on the validation years, logs the curve and saves the best checkpoint."""

    def __init__(self, spec: EnvSpec, years: list[int], every: int, out: Path):
        super().__init__()
        self.spec, self.years, self.every, self.out = spec, years, every, out
        self.best = -np.inf
        self.curve: list[dict] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.every == 0:
            self._check()
        return True

    def _on_training_end(self) -> None:
        self._check()

    def _check(self) -> None:
        yields = [r["yield_t_ha"] for r in evaluate(self.model, self.spec, self.years)]
        mean = float(np.mean(yields))
        self.curve.append({"step": self.num_timesteps, "val_mean_yield": mean,
                           "val_min_yield": float(np.min(yields))})
        if mean > self.best:
            self.best = mean
            self.model.save(self.out / "best_model.zip")
        log.info("step %d: mean validation yield %.3f t/ha (best %.3f)",
                 self.num_timesteps, mean, self.best)


def observation_scale(spec: EnvSpec, seasons: int, seed: int) -> dict:
    """Mean and standard deviation of each observation value under random decisions on the
    training years (the normalization of Kelly et al.). Values are then fixed for the run."""
    env = IrrigationEnv(spec)
    env.action_space.seed(seed)
    rows = []
    for i in range(seasons):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            rows.append(obs)
            obs, _, done, _, _ = env.step(env.action_space.sample())
    rows = np.vstack(rows)
    std = rows.std(axis=0)
    return {"mean": rows.mean(axis=0).tolist(), "std": np.where(std > 0, std, 1.0).tolist()}


def train(train_spec: EnvSpec, val_spec: EnvSpec, agent: dict, seed: int, out: Path):
    """Train one agent and return the best checkpoint (by validation yield)."""
    set_random_seed(seed)
    env = Monitor(IrrigationEnv(train_spec), filename=str(out / "monitor"))
    env.reset(seed=seed)
    algo = MaskablePPO if train_spec.constraint == "mask" else PPO
    model = algo("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **agent.get("ppo", {}))
    model.set_logger(configure(str(out), ["csv"]))  # training statistics -> progress.csv
    params = {k: getattr(model, k, None) for k in HYPERPARAMETERS}
    params["clip_range"] = model.clip_range(1.0)  # stored as a schedule; the value is constant
    params["algorithm"] = algo.__name__
    params["net_arch"] = str(model.policy.net_arch)
    (out / "hyperparameters.json").write_text(json.dumps(params, indent=2, default=str))

    callback = ValidationCallback(val_spec, val_spec.years, agent["eval_every"], out)
    model.learn(total_timesteps=agent["timesteps"], callback=callback)
    with open(out / "validation_curve.json", "w") as f:
        json.dump(callback.curve, f, indent=1)
    return algo.load(out / "best_model.zip", device="cpu")
