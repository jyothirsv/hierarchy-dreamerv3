import collections
import re
import sys
import warnings
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

import numpy as np
import dreamerv3
from dreamerv3 import embodied
import hydra
import wandb
from omegaconf import OmegaConf

from env import make_env


class WandBOutput:
	def __init__(self, pattern, cfg):
		self._pattern = re.compile(pattern)
		wandb.init(
				project="dreamerv3",
				entity='nicklashansen',
				name=str(cfg.seed),
				group=self.cfg_to_group(cfg),
				tags=self.cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
				config=OmegaConf.to_container(cfg, resolve=True)
		)
		self._wandb = wandb
	
	def cfg_to_group(self, cfg, return_list=False):
		"""Return a wandb-safe group name for logging. Optionally returns group name as list."""
		lst = [cfg.task, cfg.obs_mode, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
		return lst if return_list else "-".join(lst)

	def __call__(self, summaries):
		bystep = collections.defaultdict(dict)
		for step, name, value in summaries:
			try:
				bystep[step][name] = float(value)
			except:
				continue
		for step, metrics in bystep.items():
			self._wandb.log(metrics, step=step)


def rand_str(length=6):
	chars = 'abcdefghijklmnopqrstuvwxyz'
	return ''.join(np.random.choice(list(chars)) for _ in range(length))


def train(cfg):
	# See configs.yaml for all options.
	config = embodied.Config(dreamerv3.configs['defaults'])
	config = config.update(dreamerv3.configs['small'])
	config = config.update({
		'logdir': f'~/logdir/{cfg.task}-{cfg.exp_name}-{cfg.seed}-{rand_str()}',
		'run.train_ratio': 512,
		'run.log_every': 120,  # Seconds
		'run.steps': cfg.steps+1000,
		'run.eval_every': cfg.eval_freq,
		'run.eval_eps': cfg.eval_episodes,
		'envs.amount': 1,
		'batch_size': 16,
		'jax.prealloc': False,
		'encoder.mlp_keys': 'vector',
		'decoder.mlp_keys': 'vector',
		'encoder.cnn_keys': '$^',
		'decoder.cnn_keys': '$^',
		'replay_size': 1_000_000,
	})
	config = embodied.Flags(config).parse()

	logdir = embodied.Path(config.logdir)
	step = embodied.Counter()
	logger = embodied.Logger(step, [
		embodied.logger.TerminalOutput(),
		WandBOutput(logdir.name, cfg),
	])

	env = make_env(cfg)

	from embodied.envs import from_gym
	env = from_gym.FromGym(env, obs_key='vector')
	env = dreamerv3.wrap_env(env, config)
	env = embodied.BatchEnv([env], parallel=False)

	agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)

	replay = embodied.replay.Uniform(
		config.batch_length, config.replay_size, logdir / 'replay')
	args = embodied.Config(
		**config.run, logdir=config.logdir,
		batch_steps=config.batch_size * config.batch_length)
	embodied.run.train(agent, env, replay, logger, args)
	# embodied.run.eval_only(agent, env, logger, args)


@hydra.main(config_name='config', config_path='.')
def launch(cfg: dict):
	sys.argv = sys.argv[:1]
	train(cfg)


if __name__ == '__main__':
	launch()
