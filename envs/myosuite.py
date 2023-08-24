import numpy as np
import gym
from envs.time_limit import TimeLimit
from envs.exceptions import UnknownTaskError


MYOSUITE_TASKS = {
	'myo-relocate': 'myoChallengeRelocateP1-v0',
	'myo-finger-reach': 'myoFingerReachFixed-v0',
	'myo-finger-reach-random': 'myoFingerReachRandom-v0',
	'myo-finger-pose': 'myoFingerPoseFixed-v0',
	'myo-finger-pose-random': 'myoFingerPoseRandom-v0',
	'myo-hand-reach': 'myoHandReachFixed-v0',
	'myo-hand-reach-random': 'myoHandReachRandom-v0',
	'myo-hand-pose': 'myoHandPoseFixed-v0',
	'myo-hand-pose-random': 'myoHandPoseRandom-v0',
	'myo-hand-obj-hold': 'myoHandObjHoldFixed-v0',
	'myo-hand-obj-hold-random': 'myoHandObjHoldRandom-v0',
	'myo-hand-key-turn': 'myoHandKeyTurnFixed-v0',
	'myo-hand-key-turn-random': 'myoHandKeyTurnRandom-v0',
	'myo-hand-pen-twirl': 'myoHandPenTwirlFixed-v0',
	'myo-hand-pen-twirl-random': 'myoHandPenTwirlRandom-v0',
}


class MyoSuiteWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		# Configure camera
		if 'myo-finger' in cfg.task:
			self.camera_id = 'front_view'
		elif 'myo-hand' in cfg.task:
			self.camera_id = 'hand_side_inter'
		elif 'myo-relocate' in cfg.task:
			self.camera_id = 'hand_side_inter'
		else:
			cameras = [self.env.sim.model.camera(i).name for i in range(self.env.sim.model.ncam)]
			print('No camera selected for task:', cfg.task)
			print('Available cameras:', cameras)
			raise NotImplementedError

	def step(self, action):
		obs, reward, _, info = self.env.step(action.copy())
		obs = obs.astype(np.float32)
		info['success'] = info['solved']
		return obs, reward, False, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		return self.env.sim.renderer.render_offscreen(
			width=400, height=400, camera_id=self.camera_id
		).copy()


def make_env(cfg):
	"""
	Make Meta-World environment.
	"""
	if not cfg.task in MYOSUITE_TASKS:
		raise UnknownTaskError(cfg.task)
	import myosuite
	env = gym.make(MYOSUITE_TASKS[cfg.task])
	env = MyoSuiteWrapper(env, cfg)
	env = TimeLimit(env, max_episode_steps=100)
	env.max_episode_steps = env._max_episode_steps
	return env
