from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions
from dm_control.utils import rewards
import numpy as np

from envs.tasks.base import _STAND_HEIGHT


DEFAULT_DISTANCE_TOLERANCE_TO_TARGET = 0.5


class GoToTarget(composer.Task):
  """A task that requires a walker to move towards a target."""

  def __init__(self,
               walker,
               arena,
               distance_tolerance=DEFAULT_DISTANCE_TOLERANCE_TO_TARGET,
               physics_timestep=0.005,
               control_timestep=0.025):
    """Initializes this task.

    Args:
      walker: an instance of `locomotion.walkers.base.Walker`.
      arena: an instance of `locomotion.arenas.floors.Floor`.
      distance_tolerance: Accepted to distance to the target position before
        providing reward.
      physics_timestep: a number specifying the timestep (in seconds) of the
        physics simulation.
      control_timestep: a number specifying the timestep (in seconds) at which
        the agent applies its control inputs (in seconds).
    """

    self._arena = arena
    self._walker = walker
    self._walker.create_root_joints(self._arena.attach(self._walker))

    arena_position = distributions.Uniform(
        low=-np.array(arena.size) / 3, high=np.array(arena.size) / 3)
    self._target_spawn_position = arena_position
    self._walker_spawn_position = arena_position

    self._distance_tolerance = distance_tolerance
    
    self._target = self.root_entity.mjcf_model.worldbody.add(
        'site',
        name='target',
        type='sphere',
        pos=(0., 0., 0.4),
        size=(0.2,),
        rgba=(0.7, 0.9, 0.5, 0.6))

    enabled_observables = []
    enabled_observables += self._walker.observables.proprioception
    enabled_observables += self._walker.observables.kinematic_sensors
    enabled_observables += self._walker.observables.dynamic_sensors
    enabled_observables.append(self._walker.observables.sensors_touch)
    for obs in enabled_observables:
      obs.enabled = True

    walker.observables.add_egocentric_vector(
        'target',
        observable.MJCFFeature('pos', self._target),
        origin_callable=lambda physics: physics.bind(walker.root_body).xpos)

    self.set_timesteps(
        physics_timestep=physics_timestep, control_timestep=control_timestep)

  @property
  def root_entity(self):
    return self._arena
  
  def head_height(self, physics):
    return self._walker._observables.head_height(physics)

  def target_position(self, physics):
    return np.array(physics.bind(self._target).pos)

  def initialize_episode_mjcf(self, random_state):
    self._arena.regenerate(random_state=random_state)

    target_x, target_y = variation.evaluate(
        self._target_spawn_position, random_state=random_state)
    self._target.pos = [target_x, target_y, 0.4]

  def initialize_episode(self, physics, random_state):
    self._walker.reinitialize_pose(physics, random_state)
    walker_x, walker_y = variation.evaluate(
        self._walker_spawn_position, random_state=random_state)
    self._walker.shift_pose(
        physics,
        position=[walker_x, walker_y, 0.],
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
    self._ground_geomids.add(physics.bind(self._target).element_id)

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
    standing = rewards.tolerance(self.head_height(physics),
									 bounds=(_STAND_HEIGHT, float('inf')),
									 margin=_STAND_HEIGHT/4)
    small_control = rewards.tolerance(physics.control(), margin=1,
                   value_at_margin=0,
                   sigmoid='quadratic').mean()
    small_control = (4 + small_control) / 5
    distance = np.linalg.norm(
        physics.bind(self._target).pos[:2] -
        physics.bind(self._walker.root_body).xpos[:2])
    max_distance = np.linalg.norm(np.array(self._arena.size))
    distance_reward = np.clip(1 - distance / max_distance, 0, 1)
    distance_reward = (5*distance_reward + 1) / 6
    within_distance = float(distance < self._distance_tolerance)
    within_distance = (3*within_distance + 1) / 4
    return standing * small_control * distance_reward * within_distance

  def before_step(self, physics, action, random_state):
    self._walker.apply_action(physics, action, random_state)

  def after_step(self, physics, random_state):
    self._failure_termination = False
    for contact in physics.data.contact:
      if self._is_disallowed_contact(contact):
        self._failure_termination = True
        break
