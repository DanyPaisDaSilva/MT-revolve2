from individual import Individual

#contains functionality

def find_best_robot(
    current_best: Individual | None, population: list[Individual]
) -> Individual:
    """
    Return the best robot between the population and the current best individual.

    :param current_best: The current best individual.
    :param population: The population.
    :returns: The best individual.
    """
    return max(
        population if current_best is None else [current_best] + population,
        key=lambda x: x.fitness,
    )