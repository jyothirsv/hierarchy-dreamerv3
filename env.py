from wrapper import TensorWrapper, TimeLimit, HumanoidWrapper, FailSafeWrapper

from envs.tasks.go_to_target import GoToTarget
from envs.tasks.pick_box import PickBox
from envs.tasks.walk import Stand, Walk, Run
from envs.tasks.spin import Spin
from envs.tasks.poles import Poles
from envs.tasks.slides import Slides
from envs.tasks.crawl import Crawl

from dm_control.locomotion.tasks.escape import Escape
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.tasks.go_to_target import GoToTarget
from dm_control.locomotion.tasks.random_goal_maze import ManyGoalsMaze
from dm_control.locomotion.tasks.corridors import RunThroughCorridor
from envs import dm_control_wrapper
from collections import OrderedDict
from copy import deepcopy
import numpy as np
TASKS = {
	'reach': {
		'constructor': GoToTarget,
		'task_kwargs': {},
	},
    'maze': {
		'constructor': ManyGoalsMaze,
		'task_kwargs': {
			'arena_type': 'maze',
			'enable_rgb': True,
		},
	},
	'corridor': {
		'constructor': RunThroughCorridor,
		'task_kwargs': {
			'arena_type': 'corridor',
			'target_velocity': 6.0,
		},
	},
	'gaps-corridor': {
		'constructor': RunThroughCorridor,
		'task_kwargs': {
			'arena_type': 'gaps-corridor',
			'target_velocity': 6.0,
		},
	},
	'walls-corridor': {
		'constructor': RunThroughCorridor,
		'task_kwargs': {
			'arena_type': 'walls-corridor',
			'target_velocity': 6.0,
		},
	},
	'stairs-corridor': {
		'constructor': RunThroughCorridor,
		'task_kwargs': {
			'arena_type': 'stairs-corridor',
			'target_velocity': 6.0,
		},
	},
	'hurdles-corridor': {
		'constructor': RunThroughCorridor,
		'task_kwargs': {
			'arena_type': 'hurdles-corridor',
			'target_velocity': 6.0,
		},
	},
	'poles-corridor': {
        'constructor': RunThroughCorridor,
        'task_kwargs': {
			'arena_type': 'poles-corridor',
			'target_velocity': 6.0,
		},
	},
	'slides-corridor': {
        'constructor': RunThroughCorridor,
        'task_kwargs': {
			'arena_type': 'slides-corridor',
			'target_velocity': 6.0,
		},
	},
	'stand': {
		'constructor': Stand,
		'task_kwargs': {},
	},
	'walk': {
		'constructor': Walk,
		'task_kwargs': {},
	},
	'run': {
		'constructor': Run,
		'task_kwargs': {},
	},
	'spin': {
		'constructor': Spin,
		'task_kwargs': {},
	},
	'pick-box': {
		'constructor': PickBox,
		'task_kwargs': {},
	},
	'escape': {
		'constructor': Escape,
		'task_kwargs': {},
	},
	'multiclip': {
		'constructor': Stand,
		'task_kwargs': {},
	},
	'poles': {
        'constructor': Poles,
        'task_kwargs': {'arena_type': 'wide-corridor',
						'target_velocity': 6.0,
						'end_on_contact': False,},
	},
	'slides': {
		'constructor': Slides,
		'task_kwargs': {'arena_type': 'corridor',
				  'target_velocity': 6.0,},
	},
	'crawl': {
		'constructor': Crawl,
		'task_kwargs': {'arena_type': 'corridor'},
	},
}


def make_env(cfg, eval=False):
	"""
	Make CMU Humanoid environment for transfer tasks.
	"""
	if cfg.task not in TASKS:
		raise ValueError('Unknown task:', cfg.task)
	seed = cfg.seed + (42 if eval else 0)
	task_kwargs = dict(
		physics_timestep=tracking.DEFAULT_PHYSICS_TIMESTEP,
		control_timestep=0.03,
	)
	task_kwargs.update(TASKS[cfg.task]['task_kwargs'])
	
	env = dm_control_wrapper.DmControlWrapper.make_env_constructor(
		TASKS[cfg.task]['constructor'])(task_kwargs=task_kwargs)
	max_episode_steps = int(500)
	print(f'max_episode_steps: {max_episode_steps}')
	env = HumanoidWrapper(env, cfg, max_episode_steps=max_episode_steps)
	env = TimeLimit(env, max_episode_steps=max_episode_steps)
	cfg.episode_length = max_episode_steps
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
	cfg.seed_steps = max(1000, 5*cfg.episode_length)

	return env