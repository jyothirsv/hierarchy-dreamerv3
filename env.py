from copy import deepcopy
import os
from collections import defaultdict, OrderedDict
import warnings

import gym
import numpy as np

from envs.dmcontrol import make_env as make_dm_control_env
from envs.maniskill import make_env as make_maniskill_env
from envs.metaworld import make_env as make_metaworld_env
from envs.myosuite import make_env as make_myosuite_env
# from envs.xarm import make_env as make_xarm_env
# from envs.robopianist import make_env as make_robopianist_env
from envs.exceptions import UnknownTaskError

warnings.filterwarnings('ignore', category=DeprecationWarning)

__RGB_ENCODER__ = None
__RGB_TRANSFORM__ = None


class TensorWrapper(gym.Wrapper):
	def __init__(self, env):
		super().__init__(env)
	
	def rand_act(self):
		return self.action_space.sample().astype(np.float32)

	def _try_f32_tensor(self, x):
		if x.dtype == np.float64:
			x = x.astype(np.float32)
		return x

	def _obs_to_tensor(self, obs):
		if isinstance(obs, dict):
			for k in obs.keys():
				obs[k] = self._try_f32_tensor(obs[k])
		else:
			obs = self._try_f32_tensor(obs)
		return obs

	def reset(self):
		return self._obs_to_tensor(self.env.reset())

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		info = defaultdict(float, info)
		info['success'] = float(info['success'])
		return self._obs_to_tensor(obs), np.array(reward, dtype=np.float32), done, info
	

class FailSafeWrapper(gym.Wrapper):
	def __init__(self, env):
		super().__init__(env)
		self._prev_transition = None

	def step(self, action):
		try:
			self._prev_transition = self.env.step(action)
			return self._prev_transition
		except Exception as e:
			print(f"Exception in step: {e}")
			prev_obs, prev_reward, _, prev_info = self._prev_transition
			return prev_obs, prev_reward, True, prev_info
	

def make_env(cfg, eval=False):
	"""
	Make environment.
	"""
	gym.logger.set_level(40)
	env = None
	from copy import deepcopy
	_cfg = deepcopy(cfg)
	_cfg.seed = cfg.seed + (42 if eval else 0)
	for fn in [make_dm_control_env, make_metaworld_env, make_myosuite_env, make_maniskill_env]: #, make_xarm_env, make_robopianist_env]:
		try:
			env = fn(_cfg)
		except UnknownTaskError:
			pass
	if env is None:
		raise UnknownTaskError(cfg.task)
	env = TensorWrapper(env)
	env = FailSafeWrapper(env)
	if eval:
		for _ in range(np.random.randint(21)):
			env.reset()
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.obs_mode: env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	cfg.seed_steps = max(1000, 5*cfg.episode_length)
	return env
