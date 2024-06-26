from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.variation import distributions
import numpy as np


_STAND_HEIGHT = 1.6


class Base(composer.Task):
  """A base class for custom CMU Humanoid transfer tasks."""

  def __init__(self,
               walker,
               arena,
               walker_spawn_position=None,
               walker_spawn_rotation=None,
               physics_timestep=0.005,
               control_timestep=0.03):
    """Initializes this task.

    Args:
      walker: an instance of `locomotion.walkers.base.Walker`.
      arena: an instance of `locomotion.arenas.floors.Floor`.
      walker_spawn_position: a sequence of 2 numbers, or a `composer.Variation`
        instance that generates such sequences, specifying the position at
        which the walker is spawned at the beginning of an episode.
        If None, the entire arena is used to generate random spawn positions.
      walker_spawn_rotation: a number, or a `composer.Variation` instance that
        generates a number, specifying the yaw angle offset (in radians) that is
        applied to the walker at the beginning of an episode.
      physics_timestep: a number specifying the timestep (in seconds) of the
        physics simulation.
      control_timestep: a number specifying the timestep (in seconds) at which
        the agent applies its control inputs (in seconds).
    """
    self._arena = arena
    self._walker = walker
    self._walker.create_root_joints(self._arena.attach(self._walker))
    arena_position = distributions.Uniform(
        low=-np.array(arena.size) / 2, high=np.array(arena.size) / 2)

    if walker_spawn_position is not None:
      self._walker_spawn_position = walker_spawn_position
    else:
      self._walker_spawn_position = arena_position

    self._walker_spawn_rotation = walker_spawn_rotation

    self._reward_step_counter = 0

    enabled_observables = []
    enabled_observables += self._walker.observables.proprioception
    enabled_observables += self._walker.observables.kinematic_sensors
    enabled_observables += self._walker.observables.dynamic_sensors
    enabled_observables.append(self._walker.observables.sensors_touch)
    for obs in enabled_observables:
      obs.enabled = True

    self.set_timesteps(
        physics_timestep=physics_timestep, control_timestep=control_timestep)

  @property
  def root_entity(self):
    return self._arena

  def initialize_episode_mjcf(self, random_state):
    self._arena.regenerate(random_state=random_state)

  def initialize_episode(self, physics, random_state):
    self._walker.reinitialize_pose(physics, random_state)
    if self._walker_spawn_rotation:
      rotation = variation.evaluate(
          self._walker_spawn_rotation, random_state=random_state)
      quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]
    else:
      quat = None
    walker_x, walker_y = variation.evaluate(
        self._walker_spawn_position, random_state=random_state)
    self._walker.shift_pose(
        physics,
        position=[walker_x, walker_y, 0.],
        quaternion=quat,
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
    self._init_hand_pos = self.hand_position(physics)

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

  def head_height(self, physics):
    return self._walker._observables.head_height(physics)

  def hand_position(self, physics):
    appendages_pos = self._walker._observables.appendages_pos(physics)
    hand_pos = appendages_pos[:6]
    return hand_pos

  def hand_position_diff(self, physics):
    hand_pos = self.hand_position(physics)
    return np.linalg.norm(hand_pos - self._init_hand_pos)

  def angular_velocity(self, physics):
    # backward_roll = self._walker._observables.gyro_backward_roll(physics)
    # rightward_roll = self._walker._observables.gyro_rightward_roll(physics)
    anticlockwise_spin = self._walker._observables.gyro_anticlockwise_spin(physics)
    return anticlockwise_spin

  def get_reward(self, physics):
    raise NotImplementedError

  def before_step(self, physics, action, random_state):
    self._walker.apply_action(physics, action, random_state)

  def after_step(self, physics, random_state):
    self._failure_termination = False
    for contact in physics.data.contact:
      if self._is_disallowed_contact(contact):
        self._failure_termination = True
        break
