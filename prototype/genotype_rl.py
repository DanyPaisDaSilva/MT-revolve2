"""Genotype class."""

from __future__ import annotations

from dataclasses import dataclass

import multineat
import numpy as np

from revolve2.modular_robot import ModularRobot, ModularRobotControlInterface
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain import BrainInstance, Brain
from revolve2.modular_robot.sensor_state import ModularRobotSensorState
from revolve2.standards.genotypes.cppnwin.modular_robot import BrainGenotypeCpg
from revolve2.standards.genotypes.cppnwin.modular_robot.v2 import BodyGenotypeV2

import gym
from gym import spaces

@dataclass
class GenotypeRL(BodyGenotypeV2, BrainGenotypeCpg):
    """A genotype for a body and brain using CPPN."""

    @classmethod
    def random(
        cls,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
        rng: np.random.Generator,
    ) -> GenotypeRL:
        """
        Create a random genotype.

        :param innov_db_body: Multineat innovation database for the body. See Multineat library.
        :param innov_db_brain: Multineat innovation database for the brain. See Multineat library.
        :param rng: Random number generator.
        :returns: The created genotype.
        """
        body = cls.random_body(innov_db_body, rng)
        brain = cls.random_brain(innov_db_brain, rng)

        return GenotypeRL(body=body.body, brain=brain.brain)

    def mutate(
        self,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
        rng: np.random.Generator,
    ) -> GenotypeRL:
        """
        Mutate this genotype.

        This genotype will not be changed; a mutated copy will be returned.

        :param innov_db_body: Multineat innovation database for the body. See Multineat library.
        :param innov_db_brain: Multineat innovation database for the brain. See Multineat library.
        :param rng: Random number generator.
        :returns: A mutated copy of the provided genotype.
        """
        body = self.mutate_body(innov_db_body, rng)
        # TODO: new brain instance (=empty brain)
        brain = self.mutate_brain(innov_db_brain, rng)

        return GenotypeRL(body=body.body, brain=brain.brain)

    @classmethod
    def crossover(
        cls,
        parent1: GenotypeRL,
        parent2: GenotypeRL,
        rng: np.random.Generator,
    ) -> GenotypeRL:
        """
        Perform crossover between two genotypes.

        :param parent1: The first genotype.
        :param parent2: The second genotype.
        :param rng: Random number generator.
        :returns: A newly created genotype.
        """
        body = cls.crossover_body(parent1, parent2, rng)

        # TODO: new instance
        brain = cls.crossover_brain(parent1, parent2, rng)

        return GenotypeRL(body=body.body, brain=brain.brain)

    def develop(self) -> ModularRobot:
        """
        Develop the genotype into a modular robot.

        :returns: The created robot.
        """
        body = self.develop_body()
        brain = self.develop_brain(body=body)
        return ModularRobot(body=body, brain=brain)

class CustomBrainInstance(BrainInstance):

    active_hinges: list[ActiveHinge]
    def __init__(self, body):
        self.active_hinges = body.find_modules_of_type(ActiveHinge)

    def control(self, dt: float, sensor_state: ModularRobotSensorState,
                control_interface: ModularRobotControlInterface) -> None:
        # TODO: reward needs to be calculated here
        pass

class CustomBrain(Brain):

    def __init__(self):
        super().__init__()

    def make_instance(self) -> BrainInstance:
        """
        Create an instance of this brain.

        :returns: The created instance.
        """
        return CustomBrainInstance()

class CustomContinuousEnv(gym.Env):

    # each environment needs a custom environment

    def __init__(self, CustomBrain):
        super(CustomContinuousEnv, self).__init__()
        # TODO: get action space and observation space from active hinges
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1111,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1111, high=1111, shape=(1111,), dtype=np.float32)
        self.state = None

    def reset(self):
        # get state from active hinges
        self.state = np.random.random(size=(1111,))
        return self.state

    def step(self, action):
        # get reward from evaluator ?
        reward = -np.sum(action**2)  # example reward
        self.state = np.random.random(size=(1111,))
        done = False
        info = {}
        return self.state, reward, done, info