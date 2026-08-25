# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

import gymnasium as gym  # noqa: F401

gym.register(
    id="Isaac-Pace-Actuator-300Nm",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace.act_env_cfg:PaceEnvCfg_300Nm"
    },
)

gym.register(
    id="Isaac-Pace-Actuator-100Nm",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace.act_env_cfg:PaceEnvCfg_100Nm"
    },
)

gym.register(
    id="Isaac-Pace-Actuator-20Nm",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace.act_env_cfg:PaceEnvCfg_20Nm"
    },
)
