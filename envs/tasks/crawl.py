from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions
from dm_control.utils import rewards
import numpy as np
from dm_control import mjcf
from dm_control import mujoco
import os

from envs.tasks.base import _STAND_HEIGHT



class Crawl(composer.Task):
  """A task that requires a walker to pick up a box from a table."""

  def __init__(self,
               walker,
               arena,
               physics_timestep=0.005,
               control_timestep=0.025):
    """Initializes this task.

    Args:
      walker: an instance of `locomotion.walkers.base.Walker`.
      arena: an instance of `locomotion.arenas.floors.Floor`.
      physics_timestep: a number specifying the timestep (in seconds) of the
        physics simulation.
      control_timestep: a number specifying the timestep (in seconds) at which
        the agent applies its control inputs (in seconds).
    """

    self._arena = arena
    self._walker = walker
    self._move_speed = 1.0
    self._walker.create_root_joints(self._arena.attach(self._walker))
    self.root_entity.mjcf_model.worldbody.add("camera", name="global", mode="fixed", pos=[0, 0, 15], zaxis=[0, 0, 1]
                                              , fovy=90)

    self._walker_spawn_position = distributions.Uniform(
        low=np.array(-1), high=np.array(-0.5))
      # low=np.array(5.0), high=np.array(5.5))
        # <body name="tunnel" pos="2.2 0 0">
        #     <geom type="box" name="left_wall" pos="8 -1.35 0.775" size="8 0.2 0.775" rgba="0.5 0.5 0.5 0.3" class="visual"/>
        #     <geom type="box" name="right_wall" pos="8 1.35 0.775" size="8 0.2 0.775" rgba="0.5 0.5 0.5 0.3" class="visual"/>
        #     <geom type="box" name="ceiling" pos="8 0 1.35" size="8 1.15 0.2" rgba="0.5 0.5 0.5 0.3" class="visual"/>

        #     <geom type="box" pos="8 -1.35 0.775" size="8 0.2 0.775" class="collision"/>
        #     <geom type="box" pos="8 1.35 0.775" size="8 0.2 0.775" class="collision"/>
        #     <geom type="box" pos="8 0 1.35" size="8 1.15 0.2" class="collision"/>
        # </body>
    self.root_entity.mjcf_model.worldbody.add('geom', type='box', name='left_wall', pos=[8, -1.35, 0.775], size=[8, 0.2, 0.775], rgba=[0.5, 0.5, 0.5, 0.3])
    self.root_entity.mjcf_model.worldbody.add('geom', type='box', name='right_wall', pos=[8, 1.35, 0.775], size=[8, 0.2, 0.775], rgba=[0.5, 0.5, 0.5, 0.3])
    self.root_entity.mjcf_model.worldbody.add('geom', type='box', name='ceiling', pos=[8, 0, 1.35], size=[8, 1.15, 0.2], rgba=[0.5, 0.5, 0.5, 0.3])
    enabled_observables = []
    enabled_observables += self._walker.observables.proprioception
    enabled_observables += self._walker.observables.kinematic_sensors
    enabled_observables += self._walker.observables.dynamic_sensors
    enabled_observables.append(self._walker.observables.sensors_touch)
    enabled_observables.append(self._walker.observables.position)
    for obs in enabled_observables:
      obs.enabled = True
    self.set_timesteps(
            physics_timestep=physics_timestep, control_timestep=control_timestep)

  @property
  def root_entity(self):
    return self._arena
  
  def head_height(self, physics):
    return self._walker._observables.head_height(physics)

  def hand_position(self, physics):
    return physics.bind(self._walker.marker_geoms).xpos


  def initialize_episode_mjcf(self, random_state):
    self._arena.regenerate(random_state=random_state)


  def initialize_episode(self, physics, random_state):
    self._walker.reinitialize_pose(physics, random_state)
    walker_x = variation.evaluate(
        self._walker_spawn_position, random_state=random_state)
    self._walker.shift_pose(
        physics,
        position=[walker_x, 0, 0],
        quaternion=None,
        rotate_velocity=True)

    self._failure_termination = False
    walker_foot_geoms = set(self._walker.ground_contact_geoms)
    walker_nonfoot_geoms = [
        geom for geom in self._walker.mjcf_model.find_all('geom')
        if geom not in walker_foot_geoms]
    self._walker_nonfoot_geomids = set(
        physics.bind(walker_nonfoot_geoms).element_id)
    self._ground_geomids = set(
        physics.bind(self._arena.ground_geoms).element_id)

  def _is_disallowed_contact(self, contact):
    set1, set2 = self._walker_nonfoot_geomids, self._ground_geomids
    return ((contact.geom1 in set1 and contact.geom2 in set2) or
            (contact.geom1 in set2 and contact.geom2 in set1))

  def should_terminate_episode(self, physics):
    return self._failure_termination

  def get_discount(self, physics):
    if self._failure_termination:
      return 0.
    else:
      return 1.

  def get_reward(self, physics):
    return 0.
  
  def before_step(self, physics, action, random_state):
    self._walker.apply_action(physics, action, random_state)

  def after_step(self, physics, random_state):
    self._failure_termination = False
    for contact in physics.data.contact:
      if self._is_disallowed_contact(contact):
        self._failure_termination = True
        break

