import numpy as np
from dm_control.utils import rewards

from envs.tasks.base import Base, _STAND_HEIGHT


_WALK_SPEED = 1.0
_RUN_SPEED = 6.0


class BaseWalk(Base):
	"""A base class for custom CMU Humanoid walking transfer tasks."""

	def __init__(self, *args, move_speed=0, **kwargs):
		super().__init__(*args, **kwargs)
		self._move_speed = move_speed

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
		horizontal_speed = np.linalg.norm(physics.bind(self._walker.root_body).subtree_linvel[:2])
		if self._move_speed == 0:
			dont_move = rewards.tolerance(horizontal_speed, margin=2)
			return standing * small_control * hand_prior * dont_move
		else:
			move_reward = rewards.tolerance(horizontal_speed,
								   			bounds=(self._move_speed, float('inf')),
											margin=self._move_speed, value_at_margin=0,
											sigmoid='linear')
			move_reward = (5*move_reward + 1) / 6
			return standing * small_control * hand_prior * move_reward


class Stand(BaseWalk):
	"""A task to stand still."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, move_speed=0, **kwargs)


class Walk(BaseWalk):
	"""A task to walk forward."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, move_speed=_WALK_SPEED, **kwargs)


class Run(BaseWalk):
	"""A task to run forward."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, move_speed=_RUN_SPEED, **kwargs)
