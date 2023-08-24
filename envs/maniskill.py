from collections import OrderedDict

import gym
import numpy as np
from envs.time_limit import TimeLimit
from envs.exceptions import UnknownTaskError
from gym.spaces import Box, Dict
import torch
import kornia.augmentation as K

import mani_skill2.envs
from mani_skill2.utils.common import flatten_state_dict, flatten_dict_space_keys, flatten_dict_keys


MANISKILL_TASKS = {
	'lift-cube': dict(
		env='LiftCube-v0',
		control_mode='pd_ee_delta_pos',
		base_camera=dict(
			p=(0.4, 0.4, 0.8),
			q=(0.3647052, 0.27984813, 0.11591689, -0.88047624),
			fov=1.,
		),
	),
	'pick-cube': dict(
		env='PickCube-v0',
		control_mode='pd_ee_delta_pos',
		base_camera=dict(
			p=(0.4, 0.4, 0.8),
			q=(0.3647052, 0.27984813, 0.11591689, -0.88047624),
			fov=1.,
		),
	),
	'stack-cube': dict(
		env='StackCube-v0',
		control_mode='pd_ee_delta_pos',
		base_camera=dict(
			p=(0.4, 0.4, 0.8),
			q=(0.3647052, 0.27984813, 0.11591689, -0.88047624),
			fov=1.,
		),
	),
	'pick-ycb': dict(
		env='PickSingleYCB-v0',
		control_mode='pd_ee_delta_pose',
		base_camera=dict(
			p=(0.4, 0.4, 0.8),
			q=(0.3647052, 0.27984813, 0.11591689, -0.88047624),
			fov=1.,
		),
	),
	'turn-faucet': dict(
		env='TurnFaucet-v0',
		control_mode='pd_ee_delta_pose',
		base_camera=dict(
			p=(0.5, 0.5, 1.),
			q=(0.3647052, 0.27984813, 0.11591689, -0.88047624),
			fov=1.,
		),
	),
	'peg-insert': dict(
		env='PegInsertionSide-v0',
		control_mode='pd_ee_delta_pose',
		base_camera=dict(
			p=(0.3, -0.35, 0.4),
			q=(0.34021395, -0.27015355, 0.1027122 , 0.8948305),
			fov=1.,
		),
	),
	'cabinet-door': dict(
		env='OpenCabinetDoor-v1',
		control_mode='base_pd_joint_vel_arm_pd_ee_delta_pose',
		overhead_camera_0=dict(
			p=(-1.5, 0, 1.5),
			q=(0.9238795, 0, 0.3826834, 0),
			fov=1.,
			actor_uid=None,
		),
		overhead_camera_1=None,
		overhead_camera_2=dict(
			fov=1.,
		),
	),
	'cabinet-drawer': dict(
		env='OpenCabinetDrawer-v1',
		control_mode='base_pd_joint_vel_arm_pd_ee_delta_pose',
		overhead_camera_0=dict(
			p=(-1.5, 0, 1.5),
			q=(0.9238795, 0, 0.3826834, 0),
			fov=1.,
			actor_uid=None,
		),
		overhead_camera_1=None,
		overhead_camera_2=dict(
			fov=1.,
		),
	),
	'push-chair': dict(
		env='PushChair-v1',
		control_mode='base_pd_joint_vel_arm_pd_ee_delta_pose',
		overhead_camera_0=dict(
			fov=1.,
		),
		overhead_camera_1=dict(
			p=(0, 0, 4),
			q=(0.70710678, 0.0, 0.70710678, 0.0),
			fov=1.,
			actor_uid=None,
		),
		overhead_camera_2=None,
	),
	'move-bucket': dict(
		env='MoveBucket-v1',
		control_mode='base_pd_joint_vel_arm_pd_ee_delta_pose',
		overhead_camera_0=dict(
			fov=1.,
		),
		overhead_camera_1=dict(
			p=(0, 0, 4),
			q=(0.70710678, 0.0, 0.70710678, 0.0),
			fov=1.,
			actor_uid=None,
		),
		overhead_camera_2=None,
	),
}


class ManiSkillWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		if cfg.obs_mode == 'state':
			self.observation_space = self.env.observation_space
		else:
			rgb_interval, state_interval = (0, 255), (-float('inf'), float('inf'))
			state_dim = sum([sum(
					[x.shape[0] for x in flatten_dict_space_keys(self.env.observation_space.spaces[s]).spaces.values()]
				) for s in ['agent', 'extra']
			])
			self.observation_space = Dict(state=Box(*state_interval, shape=(state_dim,)))
			for k, v in flatten_dict_keys(self.env.reset()).items():
				if k.endswith('rgb'):
					self.observation_space.spaces[k.split('/', 1)[1].replace('/', '_')] = Box(*rgb_interval, shape=reversed(v.shape))
		self.action_space = gym.spaces.Box(
			low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
			high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
			dtype=self.env.action_space.dtype,
		)

	def _preprocess(self, obs):
		if self.cfg.obs_mode == 'state':
			return obs
		obs = flatten_dict_keys(obs)
		state_obs, rgb_obs = dict(), dict()
		for k, v in obs.items():
			if k.endswith('rgb'):
				rgb_obs[k] = v
			elif k.startswith('agent') or k.startswith('extra'):
				state_obs[k] = v
		rgb_obs = {k.split('/', 1)[1].replace('/', '_'): v.transpose(2, 0, 1) for k, v in rgb_obs.items()}
		return OrderedDict({'state': flatten_state_dict(state_obs), **rgb_obs})

	def reset(self):
		return self._preprocess(self.env.reset())
	
	def step(self, action):
		reward = 0
		for _ in range(2):
			obs, r, _, info = self.env.step(action)
			reward += r
		return self._preprocess(obs), reward, False, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, args, **kwargs):
		return self.env.render(mode='cameras')


def make_env(cfg):
	"""
	Make ManiSkill2 environment.
	"""
	if cfg.task not in MANISKILL_TASKS:
		raise UnknownTaskError(cfg.task)
	task_cfg = MANISKILL_TASKS[cfg.task]
	env = gym.make(
		task_cfg['env'],
		obs_mode=cfg.obs_mode.replace('rgb', 'rgbd'),
		control_mode=task_cfg['control_mode'],
		camera_cfgs=dict(width=224, height=224),
		render_camera_cfgs=dict(width=448, height=448),
	)
	for cfg_k, cfg_v in list(task_cfg.items())[2:]:
		if cfg_k in env.unwrapped._camera_cfgs:
			if cfg_v is None:
				del env.unwrapped._camera_cfgs[cfg_k]
				print('Deleting camera cfg:', cfg_k)
				continue
			for k, v in cfg_v.items():
				env.unwrapped._camera_cfgs[cfg_k].__dict__[k] = v
				print('Setting camera cfg:', cfg_k, k, v)
	env = ManiSkillWrapper(env, cfg)
	env = TimeLimit(env, max_episode_steps=100)
	env.max_episode_steps = env._max_episode_steps
	return env
