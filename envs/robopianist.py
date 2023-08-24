from collections import defaultdict, OrderedDict
from IPython.display import HTML
from base64 import b64encode

import gym
from gym.spaces import Box, Dict
from envs.time_limit import TimeLimit
import numpy as np

from envs.exceptions import UnknownTaskError


class RoboPianistWrapper:
	def __init__(self, env, cfg, episode_length):
		obs_shp = []
		for v in env.observation_spec().values():
			try:
				shp = np.prod(v.shape)
			except:
				shp = 1
			obs_shp.append(shp)
		obs_shp = (int(np.sum(obs_shp)),)
		if cfg.obs_mode == 'state':
			self.observation_space = Box(
				low=np.full(
					obs_shp,
					-np.inf,
					dtype=np.float32),
				high=np.full(
					obs_shp,
					np.inf,
					dtype=np.float32),
				shape=obs_shp,
				dtype=np.float32,
			)
		elif cfg.obs_mode == 'rgb':
			rgb_interval, state_interval = (0, 255), (-float('inf'), float('inf'))
			self.observation_space = Dict(
				state=Box(*state_interval, shape=(obs_shp[0],)),
				rgb=Box(*rgb_interval, shape=(3, 224, 224)),
			)
		else:
			raise ValueError('Unknown observation mode: {}'.format(cfg.obs_mode))
		act_shp = env.action_spec().shape
		self.min_ctrl = env.action_spec().minimum
		self.max_ctrl = env.action_spec().maximum
		self.action_space = Box(
			low=np.full(act_shp, -1),
			high=np.full(act_shp, 1),
			shape=act_shp,
			dtype=np.float32)
		self.env = env
		self.cfg = cfg
		self.max_episode_steps = episode_length
		self.t = 0

	@property
	def unwrapped(self):
		return self.env

	@property
	def reward_range(self):
		return None

	@property
	def metadata(self):
		return None

	def _obs_to_array(self, obs):
		return np.concatenate([v.flatten() for v in obs.values()]).astype(np.float32)
	
	def _preprocess(self, obs):
		obs = self._obs_to_array(obs)
		if self.cfg.obs_mode == 'state':
			return obs
		return OrderedDict(state=obs, rgb=self.render(width=224, height=224).transpose(2, 0, 1).copy())

	def reset(self):
		self.t = 0
		return self._preprocess(self.env.reset().observation)

	def step(self, action):
		self.t += 1
		action = self.min_ctrl + (self.max_ctrl - self.min_ctrl) * (action + 1) / 2
		time_step = self.env.step(action)
		info = defaultdict(float)
		success_metrics = self.env.get_musical_metrics()
		info.update(success_metrics)
		info['success'] = success_metrics['f1']
		return self._preprocess(time_step.observation), time_step.reward, time_step.last() or self.t == self.max_episode_steps, info

	def render(self, mode='rgb_array', width=384, height=384, camera_id='piano/back'):
		return self.env._physics.render(height, width, camera_id)


def make_env(cfg):
	"""
	Make simulated RoboPianist environment.
	"""
	if not cfg.task.startswith('piano-'):
		raise UnknownTaskError(cfg.task)
	try:
		task_id = int(cfg.task.split('-', 1)[1])
	except ValueError:
		raise UnknownTaskError(cfg.task)
	from robopianist import suite
	from robopianist.wrappers import MidiEvaluationWrapper
	ROBOPIANIST_TASKS = suite.ETUDE_12
	if task_id < 0 or task_id >= len(ROBOPIANIST_TASKS):
		raise UnknownTaskError(cfg.task)
	task = ROBOPIANIST_TASKS[task_id]
	print('Loading RoboPianist task:', task)
	env = suite.load(
		task,
		seed=cfg.seed,
		task_kwargs=dict(
			n_steps_lookahead=10,
		),
	)
	env = MidiEvaluationWrapper(env)
	episode_length = -1
	time_step = env.reset()
	while not time_step.last():
		episode_length += 1
		time_step = env.step(env.action_spec().generate_value())
	env = RoboPianistWrapper(env, cfg, episode_length)
	return env
