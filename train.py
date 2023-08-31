import hydra


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['small'])
  config = config.update({
	  'logdir': '~/logdir/run1',
	  'run.train_ratio': 8,
	  'run.log_every': 120,  # Seconds
	  'batch_size': 4,
	  'jax.prealloc': False,
	  'encoder.mlp_keys': 'vector',
	  'decoder.mlp_keys': 'vector',
	  'encoder.cnn_keys': '$^',
	  'decoder.cnn_keys': '$^',
	  # 'jax.platform': 'cpu',
	  # NH:
	  'replay_size': 10000,
  })
  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
	  embodied.logger.TerminalOutput(),
	  embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
	  embodied.logger.TensorBoardOutput(logdir),
	  # embodied.logger.WandBOutput(logdir.name, config),
	  # embodied.logger.MLFlowOutput(logdir.name),
  ])

  # import crafter
  from embodied.envs import from_gym
  # crafter_env = crafter.Env()  # Replace this with your Gym env.

  from env import make_env
  env = make_env(cfg)

  env = from_gym.FromGym(env, obs_key='vector')  # Or obs_key='vector'.
  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)

  breakpoint()

  replay = embodied.replay.Uniform(
	  config.batch_length, config.replay_size, logdir / 'replay')
  args = embodied.Config(
	  **config.run, logdir=config.logdir,
	  batch_steps=config.batch_size * config.batch_length)
  embodied.run.train(agent, env, replay, logger, args)
  # embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  train()
