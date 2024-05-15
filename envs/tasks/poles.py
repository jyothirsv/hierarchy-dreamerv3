from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions
from dm_control.utils import rewards
import numpy as np
from dm_control import mjcf
from dm_control import mujoco

from envs.tasks.base import _STAND_HEIGHT



class Poles(composer.Task):
  """A task that requires a walker to pick up a box from a table."""

  def __init__(self,
               walker,
               arena,
               target_velocity,
               end_on_contact,
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
    self._end_on_contact = end_on_contact
    self._move_speed = target_velocity
    self._walker.create_root_joints(self._arena.attach(self._walker))
    self.root_entity.mjcf_model.worldbody.add("camera", name="global", mode="fixed", pos=[0, 0, 15], zaxis=[0, 0, 1], fovy=90)
    self._spawn_pole_spacing = distributions.Uniform(1.75, 2.0)
    self._walker_spawn_position = distributions.Uniform(
        low=np.array(-1), high=np.array(0.))

    # Initialize poles without setting specific positions
    rows = 8
    self.poles = {}
    for i in range(1, rows + 1):
        self.poles[i] = []
        num_poles = 4 if i % 2 == 0 else 3
        for j in range(num_poles):
            pole_name = f'pole_r{i}_c{j+1}'
            pole = self.root_entity.mjcf_model.worldbody.add('geom', type='cylinder', name=pole_name, size=[0.05, 1], rgba=[1., 0.6, 0.6, 1])
            self.poles[i].append(pole)
    enabled_observables = []
    enabled_observables += self._walker.observables.proprioception
    enabled_observables += self._walker.observables.kinematic_sensors
    enabled_observables += self._walker.observables.dynamic_sensors
    enabled_observables.append(self._walker.observables.position)
    enabled_observables.append(self._walker.observables.sensors_touch)
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
    gap_between_rows = 1
    spacing = self._spawn_pole_spacing(random_state=random_state)
    for i, poles in self.poles.items():
        if i % 2 == 0:
            positions = [(i * gap_between_rows, y * spacing, 1) for y in [2.25, 0.75, -0.75, -2.25]]
        else:
            positions = [(i * gap_between_rows, y * spacing, 1) for y in [1.5, 0, -1.5]]
        
        for pole, pos in zip(poles, positions):
            pole.pos = pos

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
    pole_geoms = []
    for i in self.root_entity.mjcf_model.find_all('geom'):
        if i.name:
          if 'pole' in i.name:
              pole_geoms.append(i)
    self._pole_geomids = set(
        physics.bind(pole_geoms).element_id)
    self._walker_nonfoot_geomids = set(
        physics.bind(walker_nonfoot_geoms).element_id)
    self._ground_geomids = set(
        physics.bind(self._arena.ground_geoms).element_id)
    if self._end_on_contact:
        self._ground_geomids.update(self._pole_geomids)


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
    small_control = rewards.tolerance(physics.control(),
                                        margin=1, value_at_margin=0,
                                        sigmoid='quadratic').mean()
    small_control = (4 + small_control) / 5
    walker_vel = physics.bind(self._walker.root_body).subtree_linvel[0]
    move_reward = rewards.tolerance(walker_vel, bounds=(self._move_speed, self._move_speed),
                             margin=self._move_speed, value_at_margin=0,
                             sigmoid='linear')
    move_reward = (5*move_reward + 1) / 6
    return standing * small_control * move_reward


    

  def before_step(self, physics, action, random_state):
    self._walker.apply_action(physics, action, random_state)

  def after_step(self, physics, random_state):
    self._failure_termination = False
    for contact in physics.data.contact:
      if self._is_disallowed_contact(contact):
        self._failure_termination = True
        break

