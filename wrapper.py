import numpy as np
import gym
from typing import Optional
import torch
def flatten_dict(d):
	return np.concatenate([d[k] for k in sorted(d.keys())], axis=-1)
from collections import OrderedDict, defaultdict
CMU_HUMANOID_OBSERVABLES = (
	'walker/actuator_activation',
	'walker/appendages_pos',
	'walker/body_height',
	'walker/joints_pos',
	'walker/joints_vel',
	'walker/sensors_accelerometer',
	'walker/sensors_gyro',
	'walker/sensors_torque',
	'walker/sensors_touch',
	'walker/sensors_velocimeter',
	'walker/world_zaxis'
)
REFERENCE_OBSERVABLES = (
	'walker/reference_appendages_pos',
)
TRACKING_OBSERVABLES = CMU_HUMANOID_OBSERVABLES + REFERENCE_OBSERVABLES # + ('walker/time_in_clip',)

class HumanoidWrapper(gym.Wrapper):
	def __init__(self, env, cfg, max_episode_steps=100_000):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self._max_episode_steps = max_episode_steps
		self._t = 0
		obs = self._preprocess_obs(env.reset())
		if self.cfg.obs == 'state':
			obs_shape = obs.shape
			self.observation_space = gym.spaces.Box(
				low=np.full(obs_shape, -np.inf),
				high=np.full(obs_shape, np.inf),
				dtype=np.float32,
			)
		elif self.cfg.obs == 'rgb':
			print("wrapper",obs['rgb'].shape)	
			self.observation_space = gym.spaces.Dict(
				state=gym.spaces.Box(-np.inf, np.inf, shape=(obs['state'].shape), dtype=np.float32),
				rgb=gym.spaces.Box(0, 255, shape=obs['rgb'].shape, dtype=np.uint8)
			)

	def _preprocess_obs(self, obs):
		if 'walker/egocentric_camera' in obs:
			rgb = obs.pop('walker/egocentric_camera')
		state = flatten_dict(obs)
		if self.cfg.obs == 'state':
			return state
		elif self.cfg.obs == 'rgb':
			return dict(state= state, rgb = rgb)

	def reset(self, **kwargs):
		self._t = 0
		obs = self.env.reset(**kwargs)
		return self._preprocess_obs(obs)

	def step(self, action):
		self._t += 1
		obs, reward, done, info = self.env.step(action)
		if 'time_in_clip' in info:
			info['success'] = info['time_in_clip'] >= info['last_time_in_clip']
		else:
			info['success'] = False
		info['truncated'] = self._t == self._max_episode_steps or info['success']
		info['terminated'] = done and not info['truncated']
		done = info['truncated'] or info['terminated']
		info['done'] = done
		obs = self._preprocess_obs(obs)
		return obs, reward, done, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, mode='rgb_array', width=384, height=384):
		return self.env.physics.render(height, width, camera_id=3)



class TensorWrapper(gym.Wrapper):
	"""
	Wrapper for converting numpy arrays to torch tensors.
	"""

	def __init__(self, env):
		super().__init__(env)
	
	def rand_act(self):
		return torch.from_numpy(self.action_space.sample().astype(np.float32))

	def _try_f32_tensor(self, x):
		x = torch.from_numpy(x)
		if x.dtype == torch.float64:
			x = x.float()
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
		info['terminated'] = torch.tensor(float(info['terminated']))
		return self._obs_to_tensor(obs), torch.tensor(reward, dtype=torch.float32), done, info


class TimeLimit(gym.Wrapper):
    """This wrapper will issue a `done` signal if a maximum number of timesteps is exceeded.

    Oftentimes, it is **very** important to distinguish `done` signals that were produced by the
    :class:`TimeLimit` wrapper (truncations) and those that originate from the underlying environment (terminations).
    This can be done by looking at the ``info`` that is returned when `done`-signal was issued.
    The done-signal originates from the time limit (i.e. it signifies a *truncation*) if and only if
    the key `"TimeLimit.truncated"` exists in ``info`` and the corresponding value is ``True``.

    Example:
       >>> from gym.envs.classic_control import CartPoleEnv
       >>> from gym.wrappers import TimeLimit
       >>> env = CartPoleEnv()
       >>> env = TimeLimit(env, max_episode_steps=1000)
    """

    def __init__(self, env: gym.Env, max_episode_steps: Optional[int] = None):
        """Initializes the :class:`TimeLimit` wrapper with an environment and the number of steps after which truncation will occur.

        Args:
            env: The environment to apply the wrapper
            max_episode_steps: An optional max episode steps (if ``Ǹone``, ``env.spec.max_episode_steps`` is used)
        """
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        """Steps through the environment and if the number of steps elapsed exceeds ``max_episode_steps`` then truncate.

        Args:
            action: The environment step action

        Returns:
            The environment step ``(observation, reward, done, info)`` with "TimeLimit.truncated"=True
            when truncated (the number of steps elapsed >= max episode steps) or
            "TimeLimit.truncated"=False if the environment terminated
        """
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            # TimeLimit.truncated key may have been already set by the environment
            # do not overwrite it
            episode_truncated = not done or info.get("TimeLimit.truncated", False)
            info["TimeLimit.truncated"] = episode_truncated
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        """Resets the environment with :param:`**kwargs` and sets the number of steps elapsed to zero.

        Args:
            **kwargs: The kwargs to reset the environment with

        Returns:
            The reset environment
        """
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class FailSafeWrapper(gym.Wrapper):
	def __init__(self, env):
		super().__init__(env)
		self._prev_transition = None

	def step(self, action):
		# make sure action is between -1 and 1 and not nan
		eps = 1e-4
		action = np.nan_to_num(action, nan=0, posinf=1-eps, neginf=-1+eps)
		action = np.clip(action, -1+eps, 1-eps)
		try:
			self._prev_transition = self.env.step(action)
			return self._prev_transition
		except Exception as e:
			print(f"Exception in step: {e}")
			prev_obs, prev_reward, _, prev_info = self._prev_transition
			return prev_obs, prev_reward, True, prev_info