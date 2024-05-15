from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions
from dm_control.utils import rewards
import numpy as np
from envs.tasks.base import _STAND_HEIGHT

class PickBox(composer.Task):
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
    self._walker.create_root_joints(self._arena.attach(self._walker))

    self._table_spawn_position = distributions.Uniform(
        low=np.array(2.50), high=np.array(2.75))
    self._walker_spawn_position = distributions.Uniform(
        low=np.array(-0.1), high=np.array(0.1))
    
    self._table_size = [1, 2, 0.45]
    self._box_size = [0.15, 0.15, 0.15]

    # Add table
    self._table = self.root_entity.mjcf_model.worldbody.add(
        'geom',
        type='box',
        name='table',
        size=self._table_size,
        rgba=[.7, .7, .7, 1])

    # Add box
    self._box = self.root_entity.mjcf_model.worldbody.add(
        'geom',
        type='box',
        name='box',
        size=self._box_size,
        rgba=[.7, .3, .5, 1])
    self._box.set_attributes(
        condim=4,
        mass=0.045,
        friction=[1., 1., 1.],
        solimp=[1., 1., 1.],
        solref=[0.02, 1.0])

    enabled_observables = []
    enabled_observables += self._walker.observables.proprioception
    enabled_observables += self._walker.observables.kinematic_sensors
    enabled_observables += self._walker.observables.dynamic_sensors
    enabled_observables.append(self._walker.observables.sensors_touch)
    for obs in enabled_observables:
      obs.enabled = True

    walker.observables.add_egocentric_vector(
        'table',
        observable.MJCFFeature('pos', self._table),
        origin_callable=lambda physics: physics.bind(walker.root_body).xpos)
    walker.observables.add_egocentric_vector(
        'box',
        observable.MJCFFeature('pos', self._box),
        origin_callable=lambda physics: physics.bind(walker.root_body).xpos)

    self.set_timesteps(
        physics_timestep=physics_timestep, control_timestep=control_timestep)

  @property
  def root_entity(self):
    return self._arena
  
  def head_height(self, physics):
    return self._walker._observables.head_height(physics)

  def hand_position(self, physics):
    return physics.bind(self._walker.marker_geoms).xpos

  def table_position(self, physics):
    return np.array(physics.bind(self._table).pos)

  def box_position(self, physics):
    return np.array(physics.bind(self._box).pos)

  def initialize_episode_mjcf(self, random_state):
    self._arena.regenerate(random_state=random_state)

    table_x = variation.evaluate(
        self._table_spawn_position, random_state=random_state)
    self._table.pos = [table_x, 0, self._table_size[2]]
    self._box.pos = [table_x - self._table_size[0]/3, 0, self._table_size[2]*2 + self._box_size[2]]


  def initialize_episode(self, physics, random_state):
    self._walker.reinitialize_pose(physics, random_state)
    walker_x = variation.evaluate(
        self._walker_spawn_position, random_state=random_state)
    self._walker.shift_pose(
        physics,
        position=[walker_x, 0, 0.],
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
    # self._ground_geomids.add(physics.bind(self._table).element_id)
    # self._ground_geomids.add(physics.bind(self._box).element_id)

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
    box_pos = physics.bind(self._box).xpos
    grasp_pos = np.concatenate([
      box_pos + np.array([0, -self._box_size[1]/2, 0]),
      box_pos + np.array([0, self._box_size[1]/2, 0])
    ])
    hand_pos = self.hand_position(physics).flatten()
    distance = np.linalg.norm(grasp_pos - hand_pos)
    max_distance = np.linalg.norm(np.array(self._arena.size)/2)
    distance_reward = np.clip(1 - distance / max_distance, 0, 1)
    distance_reward = (5*distance_reward + 1) / 6
    box_height = physics.bind(self._box).pos[2]
    box_lift_reward = np.clip(box_height - self._table_size[2] - self._box_size[2], -self._box_size[2]*2, self._box_size[2]*2)
    box_lift_reward = (8*box_lift_reward + 2) / 10
    return standing * small_control * distance_reward * box_lift_reward

  def before_step(self, physics, action, random_state):
    self._walker.apply_action(physics, action, random_state)

  def after_step(self, physics, random_state):
    self._failure_termination = False
    for contact in physics.data.contact:
      if self._is_disallowed_contact(contact):
        self._failure_termination = True
        break

