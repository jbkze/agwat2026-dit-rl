| Setting | DRL1 | DRL1, 18 variables | DRL1, reject | DRL1, yield at the end |
|---|---|---|---|---|
| Observation | 5 variables | 18 variables | 5 variables | 5 variables |
| Budget handling | mask | mask | reject | mask |
| Reward | stepwise + yield | stepwise + yield | stepwise + yield | yield at the end |
| Action set (mm) | 0, 1, 2, 3, 5, 8, 13, 21, 34 | 0, 1, 2, 3, 5, 8, 13, 21, 34 | 0, 1, 2, 3, 5, 8, 13, 21, 34 | 0, 1, 2, 3, 5, 8, 13, 21, 34 |
| Decision interval (days) | 5 | 5 | 5 | 5 |
| PPO algorithm | MaskablePPO | MaskablePPO | PPO | MaskablePPO |
| PPO learning_rate | 0.0003 | 0.0003 | 0.0003 | 0.0003 |
| PPO n_steps | 2048 | 2048 | 2048 | 2048 |
| PPO batch_size | 64 | 64 | 64 | 64 |
| PPO n_epochs | 10 | 10 | 10 | 10 |
| PPO gamma | 0.99 | 0.99 | 0.99 | 0.99 |
| PPO gae_lambda | 0.95 | 0.95 | 0.95 | 0.95 |
| PPO clip_range | 0.2 | 0.2 | 0.2 | 0.2 |
| PPO ent_coef | 0.0 | 0.0 | 0.0 | 0.0 |
| PPO net_arch | {'pi': [64, 64], 'vf': [64, 64]} | {'pi': [64, 64], 'vf': [64, 64]} | {'pi': [64, 64], 'vf': [64, 64]} | {'pi': [64, 64], 'vf': [64, 64]} |
| Training | until the mean yield of the training seasons rises by no more than 0.1 t/ha in 3 blocks of 50,000 steps, 300,000 to 3,000,000 steps | until the mean yield of the training seasons rises by no more than 0.1 t/ha in 3 blocks of 50,000 steps, 300,000 to 3,000,000 steps | until the mean yield of the training seasons rises by no more than 0.1 t/ha in 3 blocks of 50,000 steps, 300,000 to 3,000,000 steps | until the mean yield of the training seasons rises by no more than 0.1 t/ha in 3 blocks of 50,000 steps, 300,000 to 3,000,000 steps |
| Evaluation | final policy, deterministic actions, test years | final policy, deterministic actions, test years | final policy, deterministic actions, test years | final policy, deterministic actions, test years |
