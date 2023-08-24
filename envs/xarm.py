from envs.exceptions import UnknownTaskError


def make_env(cfg):
    """
    Make simulated xArm environment.
    """
    if not cfg.task.startswith('xarm-'):
        raise UnknownTaskError(cfg.task)
    task = cfg.task.split('-', 1)[1]
    import simxarm
    if task not in simxarm.TASKS:
        raise UnknownTaskError(cfg.task)
    env = simxarm.make(
        task,
        obs_mode=cfg.obs_mode,
        image_size=224,
        action_repeat=2,
        seed=cfg.seed,
    )
    env.max_episode_steps = simxarm.TASKS[task]['episode_length']//2
    return env
