"""Training and evaluation of PPO agents.

With `constraint: mask` the agent is a MaskablePPO from sb3-contrib, which applies the action mask
both when sampling and in the policy update; otherwise it is a plain PPO from stable-baselines3.
Hyperparameters are the library defaults unless the method sets them under `ppo`; the values
actually used are written to `hyperparameters.json`.

Training runs until it converges, judged on the training seasons only: the mean yield of the
seasons played in each block of `block_steps` steps is compared with the best earlier block, and
training stops after `patience` blocks in a row without a gain of more than `tolerance` t/ha
(never before `min_steps`, at the latest at `max_steps`). The policy after the last step is the
one that is tested; nothing is selected on evaluation data. Every `eval_every` steps the policy is
also run on the validation years, only to record the learning curve.

A run can continue a model trained in an earlier round (weights, optimizer state and step count);
the stopping rule then counts its blocks from the point where training resumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from ditrl.env import EnvSpec, IrrigationEnv, profit

HYPERPARAMETERS = ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
                   "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "policy_kwargs", "seed")
DEVICE = "cpu"   # the crop model is sequential Python; a GPU does not help


def algorithm(spec: EnvSpec):
    return MaskablePPO if spec.constraint == "mask" else PPO


def run_policy(model, env: IrrigationEnv, year: int) -> dict:
    """One season with deterministic actions (the most probable feasible depth)."""
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
    return [run_policy(model, env, year) for year in years]


class LearningCurve(BaseCallback):
    """Mean and lowest validation yield (and mean profit) every `every` training steps."""

    def __init__(self, spec: EnvSpec, every: int):
        super().__init__()
        self.spec, self.every = spec, every
        self.points: list[dict] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.every == 0:
            self._record()
        return True

    def _on_training_end(self) -> None:
        self._record()

    def _record(self) -> None:
        if self.points and self.points[-1]["step"] == self.num_timesteps:
            return   # the end of training fell on a regular evaluation
        seasons = evaluate(self.model, self.spec, self.spec.years)
        yields = [s["yield_t_ha"] for s in seasons]
        profits = [profit(s, self.spec.prices) for s in seasons]
        self.points.append({"step": self.num_timesteps, "val_mean_yield": float(np.mean(yields)),
                            "val_min_yield": float(np.min(yields)),
                            "val_mean_profit": float(np.mean(profits))})


class Convergence(BaseCallback):
    """Stops training once the mean yield of the training seasons has stopped rising (see the
    module docstring). Records one row per block and why training ended."""

    def __init__(self, block_steps: int, tolerance: float, patience: int, min_steps: int):
        super().__init__()
        self.block_steps, self.tolerance = block_steps, tolerance
        self.patience, self.min_steps = patience, min_steps
        self.yields: list[float] = []      # seasons finished in the current block
        self.blocks: list[dict] = []
        self.best = -np.inf
        self.stale = 0
        self.stopped_by = "max_steps"
        self.start = 0

    def _on_training_start(self) -> None:
        self.start = self.num_timesteps     # above 0 when an earlier model is continued

    def _on_step(self) -> bool:
        self.yields += [info["yield_t_ha"] for info in self.locals["infos"] if "yield_t_ha" in info]
        next_block = self.start + (len(self.blocks) + 1) * self.block_steps
        if self.num_timesteps < next_block or not self.yields:
            return True
        mean = float(np.mean(self.yields))
        gained = mean > self.best + self.tolerance
        self.stale = 0 if gained else self.stale + 1
        self.best = max(self.best, mean)
        self.blocks.append({"step": self.num_timesteps, "seasons": len(self.yields),
                            "mean_training_yield": mean, "best_so_far": self.best})
        self.yields = []
        if self.stale >= self.patience and self.num_timesteps >= self.min_steps:
            self.stopped_by = "converged"
            return False
        return True


def observation_scale(spec: EnvSpec, seasons: int, seed: int) -> dict:
    """Mean and standard deviation of every observation value under random decisions on the
    training years (the normalization of Kelly et al.), fixed for the whole run."""
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


def train(train_spec: EnvSpec, val_spec: EnvSpec, cfg: dict, seed: int, out: Path,
          resume: Path | None = None):
    """Train one agent until it converges, or continue the model in `resume`; returns the policy
    after the last step, its learning curve on the validation years, its convergence record per
    block, and how training ended."""
    set_random_seed(seed)
    env = Monitor(IrrigationEnv(train_spec), filename=str(out / "training"))
    algo = algorithm(train_spec)
    if algo is MaskablePPO:
        # The mask is read from the environment itself rather than through the Monitor wrapper.
        env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
    if resume is None:
        env.reset(seed=seed)
        model = algo("MlpPolicy", env, seed=seed, device=DEVICE, verbose=0, **cfg.get("ppo", {}))
    else:
        model = algo.load(resume / "model.zip", env=env, device=DEVICE)
        # New seasons from here on: the seed is shifted by the steps already trained.
        env.reset(seed=seed + model.num_timesteps)
    model.set_logger(configure(str(out), ["csv"]))   # training statistics -> progress.csv
    params = {k: getattr(model, k, None) for k in HYPERPARAMETERS}
    params["clip_range"] = model.clip_range(1.0)     # stored as a schedule; the value is constant
    params["algorithm"] = algo.__name__
    params["net_arch"] = str(model.policy.net_arch)
    (out / "hyperparameters.json").write_text(json.dumps(params, indent=1, default=str))

    rule = cfg["training"]
    curve = LearningCurve(val_spec, cfg["eval_every"])
    stop = Convergence(rule["block_steps"], rule["tolerance"], rule["patience"], rule["min_steps"])
    model.learn(total_timesteps=rule["max_steps"] - model.num_timesteps, callback=[curve, stop],
                reset_num_timesteps=resume is None)
    model.save(out / "model.zip")
    training = {"steps_trained": int(model.num_timesteps), "stopped_by": stop.stopped_by,
                "resumed_at": stop.start}
    return model, curve.points, stop.blocks, training


def load(spec: EnvSpec, path: Path):
    return algorithm(spec).load(path, device=DEVICE)
