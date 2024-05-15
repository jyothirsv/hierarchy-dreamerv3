"""
Wraps the dm_control environment and task into a Gym env. The task assumes
the presence of a CMU position-controlled humanoid.

Adapted from:
https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
"""
import dis
import functools

import os.path as osp
import numpy as np
import tree
import mujoco

from typing import Any, Callable, Dict, Optional, Text, Tuple
from dm_env import TimeStep
from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.mocap import loader
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.tasks.reference_pose import utils
from dm_control.locomotion.walkers import initializers
from dm_control.suite.wrappers import action_noise
from dm_control.locomotion.arenas import labmaze_textures
from dm_control.locomotion.arenas import mazes
from dm_control.locomotion.arenas import bowl
from dm_control.locomotion.props import target_sphere
from gym import core
from gym import spaces

from envs.tasks.arenas import EmptyCorridor, WallsCorridor, GapsCorridor, StairsCorridor, HurdlesCorridor, Floor, PolesCorridor, SlidesCorridor
from envs.walkers import cmu_humanoid


class DmControlWrapper(core.Env):
    """
    Wraps the dm_control environment and task into a Gym env. The task assumes
    the presence of a CMU position-controlled humanoid.

    Adapted from:
    https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
    """

    metadata = {"render.modes": ["rgb_array"], "videos.frames_per_second": 30}

    def __init__(
        self,
        task_type: Callable[..., composer.Task],
        task_kwargs: Optional[Dict[str, Any]] = None,
        environment_kwargs: Optional[Dict[str, Any]] = None,
        act_noise: float = 0.,

        # for rendering
        width: int = 640,
        height: int = 480,
        camera_id: int = 3
    ):
        """
        task_kwargs: passed to the task constructor
        environment_kwargs: passed to composer.Environment constructor
        """
        task_kwargs = task_kwargs or dict()
        environment_kwargs = environment_kwargs or dict()

        # create task
        self._env = self._create_env(
            task_type,
            task_kwargs,
            environment_kwargs,
            act_noise=act_noise,
        )
        self._original_rng_state = self._env.random_state.get_state()

        # Set observation and actions spaces
        self._observation_space = self._create_observation_space()
        action_spec = self._env.action_spec()
        dtype = np.float32
        self._action_space = spaces.Box(
            low=action_spec.minimum.astype(dtype),
            high=action_spec.maximum.astype(dtype),
            shape=action_spec.shape,
            dtype=dtype
        )

        # set seed
        self.seed()

        self._height = height
        self._width = width
        self._camera_id = camera_id

    @staticmethod
    def make_env_constructor(task_type: Callable[..., composer.Task]):
        return lambda *args, **kwargs: DmControlWrapper(task_type, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def dm_env(self) -> composer.Environment:
        return self._env

    @property
    def observation_space(self) -> spaces.Dict:
        return self._observation_space
    
    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def np_random(self):
        return self._env.random_state

    def seed(self, seed: Optional[int] = None):
        if seed:
            srng = np.random.RandomState(seed=seed)
            self._env.random_state.set_state(srng.get_state())
        else:
            self._env.random_state.set_state(self._original_rng_state)
        return self._env.random_state.get_state()[1]

    def _create_env(
        self,
        task_type,
        task_kwargs,
        environment_kwargs,
        act_noise=0.,
    ) -> composer.Environment:
        try:
            walker = self._get_walker(
                enable_rgb=task_kwargs.get('enable_rgb', False))
        except:
            walker = self._get_walker(enable_rgb=task_kwargs.get('enable_rgb', True))
        arena = self._get_arena(
            arena_type=task_kwargs.get('arena_type', 'floor'),
            arena_size=task_kwargs.get('arena_size', 12.))
        if 'maze' in str(task_type):
            # create target sphere
            target_reward_scale=50.
            physics_timestep=0.005
            control_timestep=0.03
            task_kwargs.update(target_builder=functools.partial(
                    target_sphere.TargetSphere,
                    radius=0.4,
                    rgb1=(0, 0, 0.4),
                    rgb2=(0, 0, 0.7)),
                    target_reward_scale=target_reward_scale,
                    physics_timestep=physics_timestep,
                    control_timestep=control_timestep)
        for key in ['enable_rgb', 'arena_type', 'arena_size']:
            if key in task_kwargs:
                del task_kwargs[key]
        task = task_type(
            walker,
            arena,
            **task_kwargs
        )
        env = composer.Environment(
            task=task,
            **environment_kwargs
        )
        task.random = env.random_state # for action noise
        if act_noise > 0.:
            env = action_noise.Wrapper(env, scale=act_noise/2)
        return env

    def _get_walker(self, enable_rgb):
        directory = osp.dirname(osp.abspath(__file__))
        initializer = initializers.UprightInitializer()
        return cmu_humanoid.CMUHumanoidPositionControlledV2020(
            initializer=initializer,
            observable_options={'egocentric_camera': dict(enabled=enable_rgb)})

    def _get_arena(self, arena_type, arena_size):
        if arena_type == 'floor':
            return Floor((arena_size, arena_size,))
        elif arena_type == 'maze':
            return mazes.RandomMazeWithTargets(
                x_cells=11,
                y_cells=11,
                xy_scale=3,
                max_rooms=4,
                room_min_size=4,
                room_max_size=5,
                spawns_per_room=1,
                targets_per_room=3,
                skybox_texture=labmaze_textures.SkyBox(style='sky_03'),
                wall_textures=labmaze_textures.WallTextures(style='style_01'),
                floor_textures=labmaze_textures.FloorTextures(style='style_01'),
            )
        elif arena_type == 'corridor':
            return EmptyCorridor(
                corridor_width=6,
                corridor_length=40,
                visible_side_planes=True,
            )
        elif arena_type == 'wide-corridor':
            return EmptyCorridor(
                corridor_width=12,
                corridor_length=40,
                visible_side_planes=True,
            )
        elif arena_type == 'gaps-corridor':
            return GapsCorridor(
                platform_length=distributions.Uniform(1.75, 2.),
                gap_length=distributions.Uniform(.2, .3),
                corridor_width=5,
                corridor_length=40,
                visible_side_planes=True,
            )
        elif arena_type == 'walls-corridor':
            return WallsCorridor(
                wall_gap=distributions.Uniform(4.0, 4.5),
                wall_width=distributions.Uniform(2.5, 3.0),
                wall_height=2.,
                wall_rgba=(.7, .3, .5, 1.),
                corridor_width=5,
                corridor_length=40,
                visible_side_planes=True,
            )
        elif arena_type == 'stairs-corridor':
            return StairsCorridor(
                stair_length=distributions.Uniform(1.0, 1.2),
                stair_height=distributions.Uniform(.08, .10),
                corridor_width=5,
                corridor_length=40,
                visible_side_planes=True,
            )
        elif arena_type == 'hurdles-corridor':
            return HurdlesCorridor(
                hurdle_length=0.05,
                hurdle_height=distributions.Uniform(0.15, 0.20),
                hurdle_spacing=distributions.Uniform(3.5, 4.0),
                corridor_width=5,
                corridor_length=40,
                visible_side_planes=True,
            )
        elif arena_type == 'poles-corridor':
            return PolesCorridor(
                pole_radius=0.05,
                pole_height=distributions.Uniform(1.75, 2.0),
                pole_spacing=distributions.Uniform(1, 1.5),
                corridor_width=5,
                corridor_length=40,
                visible_side_planes=True,
                pole_position=distributions.Uniform(-2.5, 2.5)
            )
        elif arena_type == 'slides-corridor':
            return SlidesCorridor(
                slide_length=distributions.Uniform(3, 7),
                slide_height=distributions.Uniform(0.0, 0.06),
            )
        elif arena_type == 'escape':
            return bowl.Bowl(size=(20., 20.))
        else:
            raise ValueError(f"Unknown arena type: {arena_type}")

    def _create_observation_space(self) -> spaces.Dict:
        obs_spaces = dict()

        for k, v in self._env.observation_spec().items():
            if v.dtype == np.float64 and np.prod(v.shape) > 0:
                if np.prod(v.shape) > 0:
                    if v.shape == ():
                        obs_spaces[k] = spaces.Box(
                            -np.infty,
                            np.infty,
                            shape=(1,),
                            dtype=np.float32)
                        continue
                    obs_spaces[k] = spaces.Box(
                        -np.infty,
                        np.infty,
                        shape=(np.prod(v.shape),),
                        dtype=np.float32
                    )
            elif v.dtype == np.uint8:
                tmp = v.generate_value()
                obs_spaces[k] = spaces.Box(
                    v.minimum.item(),
                    v.maximum.item(),
                    shape=tmp.shape,
                    dtype=np.uint8
                )
        return spaces.Dict(obs_spaces)

    def get_observation(self, time_step: TimeStep) -> Dict[str, np.ndarray]:
        dm_obs = time_step.observation
        obs = dict()
        for k in self.observation_space.spaces:
            if self.observation_space[k].dtype == np.uint8: # image
                obs[k] = dm_obs[k].squeeze()
            else:
                obs[k] = dm_obs[k].ravel().astype(self.observation_space[k].dtype)
        return obs

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        assert self.action_space.contains(action)

        time_step = self._env.step(action)
        reward = time_step.reward or 0.
        done = time_step.last()
        obs = self.get_observation(time_step)
        info = dict(
            internal_state=self._env.physics.get_state().copy(),
            discount=time_step.discount
        )
        return obs, reward, done, info

    def reset(self) -> Dict[str, np.ndarray]:
        time_step = self._env.reset()
        return self.get_observation(time_step)

    def render(
        self,
        mode: Text = 'rgb_array',
        height: Optional[int] = None,
        width: Optional[int] = None,
        camera_id: Optional[int] = None
    ) -> np.ndarray:
        assert mode == 'rgb_array', "This wrapper only supports rgb_array mode, given %s" % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)
