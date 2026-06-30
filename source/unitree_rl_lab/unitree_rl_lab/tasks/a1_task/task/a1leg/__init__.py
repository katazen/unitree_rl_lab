import gymnasium as gym

gym.register(
    id="a1",
    entry_point=f"{__name__}.a1_env:A1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.a1_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.a1_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.a1_task.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
