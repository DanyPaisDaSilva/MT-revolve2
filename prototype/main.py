import logging
import pickle
from tqdm import tqdm

import config
import multineat
from evaluator import Evaluator
from genotype import Genotype
from individual import Individual

from revolve2.experimentation.evolution import ModularRobotEvolution
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed
from ea import ParentSelector, SurvivorSelector, CrossoverReproducer
from tools import find_best_robot


def run() -> None:
    # Set up logging.
    setup_logging(file_name="log.txt")

    rng = make_rng_time_seed()

    # initialize CPPN innovation databases (brain & body)
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    """
    - evaluator: Allows us to evaluate a population of modular robots.
    - parent_selector: Allows us to select parents from a population of modular robots.
    - survivor_selector: Allows us to select survivors from a population.
    - crossover_reproducer: Allows us to generate offspring from parents.
    - modular_robot_evolution: The evolutionary process as a object that can be iterated.
    """
    # initialize evolutionary process components
    evaluator = Evaluator(headless=True, num_simulators=config.NUM_SIMULATORS)
    parent_selector = ParentSelector(offspring_size=config.OFFSPRING_SIZE, rng=rng)
    survivor_selector = SurvivorSelector(rng=rng)
    crossover_reproducer = CrossoverReproducer(
        rng=rng, innov_db_body=innov_db_body, innov_db_brain=innov_db_brain
    )

    modular_robot_evolution = ModularRobotEvolution(
        parent_selection=parent_selector,
        survivor_selection=survivor_selector,
        evaluator=evaluator,
        reproducer=crossover_reproducer,
    )

    # Create an initial population as we cant start from nothing.
    logging.info("Generating initial population.")
    initial_genotypes = [
        Genotype.random(
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            rng=rng,
        )
        for _ in range(config.POPULATION_SIZE)
    ]

    # Evaluate the initial population.
    logging.info("Evaluating initial population.")
    initial_fitnesses = evaluator.evaluate(initial_genotypes)

    # Create a population of individuals, combining genotype with fitness.
    population = [
        Individual(genotype, fitness)
        for genotype, fitness in zip(initial_genotypes, initial_fitnesses, strict=True)
    ]

    # Save the best robot
    best_robot = find_best_robot(None, population)

    # Set the current generation to 0.
    generation_index = 0

    # Start the actual optimization process.
    logging.info("Start optimization process.")
    while generation_index < config.NUM_GENERATIONS:
        logging.info(f"Generation {generation_index + 1} / {config.NUM_GENERATIONS}.")

        """
        In contrast to the previous example we do not explicitly stat the order of operations here, but let the ModularRobotEvolution object do the scheduling.
        This does not give a performance boost, but is more readable and less prone to errors due to mixing up the order.

        Not that you are not restricted to the classical ModularRobotEvolution object, since you can adjust the step function as you want.
        """
        population = modular_robot_evolution.step(
            population
        )  # Step the evolution forward.

        # Find the new best robot
        best_robot = find_best_robot(best_robot, population)

        logging.info(f"Best robot until now: {best_robot.fitness}")
        logging.info(f"Genotype pickle: {pickle.dumps(best_robot)!r}")

        # Increase the generation index counter.
        generation_index += 1

if __name__ == "__main__":
    run()


