import numpy as np
from dm_control.utils import rewards

from envs.tasks.base import Base, _STAND_HEIGHT


_SPIN_SPEED = 6.0


class Spin(Base):
	"""A custom CMU Humanoid spinning task."""

	def __init__(self, *args, spin_speed=0, **kwargs):
		super().__init__(*args, **kwargs)
		self._spin_speed = spin_speed

	def get_reward(self, physics):
		standing = rewards.tolerance(self.head_height(physics),
									 bounds=(_STAND_HEIGHT, float('inf')),
									 margin=_STAND_HEIGHT/4)
		small_control = rewards.tolerance(physics.control(),
										  margin=1, value_at_margin=0,
                                    	  sigmoid='quadratic').mean()
		small_control = (4 + small_control) / 5
		hand_prior = rewards.tolerance(self.hand_position_diff(physics),
									   margin=1, value_at_margin=0,
									   sigmoid='quadratic')
		hand_prior = (3 + hand_prior) / 4
		angular_speed = self.angular_velocity(physics)
		spin_reward = rewards.tolerance(angular_speed,
										bounds=(self._spin_speed, float('inf')),
										margin=self._spin_speed, value_at_margin=0,
										sigmoid='linear')
		spin_reward = (5*spin_reward + 1) / 6
		return standing * small_control * hand_prior * spin_reward
