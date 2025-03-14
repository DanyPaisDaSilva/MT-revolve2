import logging
import pickle

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

        #########
        # SETUP #
        #########

    # initialize logging and rng
    setup_logging(file_name="log.txt")
    rng = make_rng_time_seed()

    # initialize CPPN innovation databases (brain & body)
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    # initialize evolutionary process components
    evaluator = Evaluator(headless=True, num_simulators=config.NUM_SIMULATORS)
    parent_selector = ParentSelector(offspring_size=config.OFFSPRING_SIZE, rng=rng)
    survivor_selector = SurvivorSelector(rng=rng)
    crossover_reproducer = CrossoverReproducer(
        rng=rng, innov_db_body=innov_db_body, innov_db_brain=innov_db_brain
    )

    # initialize ea
    evolver = ModularRobotEvolution(
        parent_selection=parent_selector,
        survivor_selection=survivor_selector,
        evaluator=evaluator,
        reproducer=crossover_reproducer,
    )

    # generate initial genotypes
    logging.info("Generating initial population.")
    initial_genotypes = [
        Genotype.random(
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            rng=rng,
        )
        for _ in range(config.POPULATION_SIZE)
    ]

    # evaluate the initial genotypes
    logging.info("Evaluating initial population.")
    initial_fitnesses = evaluator.evaluate(initial_genotypes)

    # create population of individuals with fitness.
    population = [
        Individual(genotype, fitness)
        for genotype, fitness in zip(initial_genotypes, initial_fitnesses, strict=True)
    ]

    # save best robot
    best_robot = find_best_robot(None, population)

    # set the current generation to 0
    generation_index = 0


        ###########################
        # START EVOLUTIONARY LOOP #
        ###########################

    # start optimization process.
    logging.info("Start optimization process.")
    while generation_index < config.NUM_GENERATIONS:
        logging.info(f"Generation {generation_index + 1} / {config.NUM_GENERATIONS}.")

        # step the evolution forward
        population = evolver.step(population)

        # find the new best robot
        best_robot = find_best_robot(best_robot, population)

        logging.info(f"Best robot until now: {best_robot.fitness}")
        logging.info(f"Genotype pickle: {pickle.dumps(best_robot)!r}")

        # Increase the generation index counter.
        generation_index += 1

if __name__ == "__main__":
    run()


