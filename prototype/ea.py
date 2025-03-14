from typing import Any
import multineat
import numpy as np
import numpy.typing as npt
from genotype import Genotype
from individual import Individual
from revolve2.experimentation.evolution.abstract_elements import Reproducer, Selector
from revolve2.experimentation.optimization.ea import population_management, selection


class ParentSelector(Selector):
    """Selector class for parent selection."""

    rng: np.random.Generator
    offspring_size: int

    def __init__(self, offspring_size: int, rng: np.random.Generator) -> None:
        """
        Initialize the parent selector.

        :param offspring_size: The offspring size.
        :param rng: The rng generator.
        """
        self.offspring_size = offspring_size
        self.rng = rng

    def select(
        self, population: list[Individual], **kwargs: Any
    ) -> tuple[npt.NDArray[np.int_], dict[str, list[Individual]]]:
        """
        Select the parents.

        :param population: The population of robots.
        :param kwargs: Other parameters.
        :return: The parent pairs.
        """
        return np.array(
            [
                selection.multiple_unique(
                    selection_size=2,
                    population=[individual.genotype for individual in population],
                    fitnesses=[individual.fitness for individual in population],
                    selection_function=lambda _, fitnesses: selection.tournament(
                        rng=self.rng, fitnesses=fitnesses, k=1
                    ),
                )
                for _ in range(self.offspring_size)
            ],
        ), {"parent_population": population}


class SurvivorSelector(Selector):
    """Selector class for survivor selection."""

    rng: np.random.Generator

    def __init__(self, rng: np.random.Generator) -> None:
        """
        Initialize the parent selector.

        :param rng: The rng generator.
        """
        self.rng = rng

    def select(
        self, population: list[Individual], **kwargs: Any
    ) -> tuple[list[Individual], dict[str, Any]]:
        """
        Select survivors using a tournament.

        :param population: The population the parents come from.
        :param kwargs: The offspring, with key 'offspring_population'.
        :returns: A newly created population.
        :raises ValueError: If the population is empty.
        """
        offspring = kwargs.get("children")
        offspring_fitness = kwargs.get("child_task_performance")
        if offspring is None or offspring_fitness is None:
            raise ValueError(
                "No offspring was passed with positional argument 'children' and / or 'child_task_performance'."
            )

        original_survivors, offspring_survivors = population_management.steady_state(
            old_genotypes=[i.genotype for i in population],
            old_fitnesses=[i.fitness for i in population],
            new_genotypes=offspring,
            new_fitnesses=offspring_fitness,
            selection_function=lambda n, genotypes, fitnesses: selection.multiple_unique(
                selection_size=n,
                population=genotypes,
                fitnesses=fitnesses,
                selection_function=lambda _, fitnesses: selection.tournament(
                    rng=self.rng, fitnesses=fitnesses, k=2
                ),
            ),
        )

        return [
            Individual(
                population[i].genotype,
                population[i].fitness,
            )
            for i in original_survivors
        ] + [
            Individual(
                offspring[i],
                offspring_fitness[i],
            )
            for i in offspring_survivors
        ], {}


class CrossoverReproducer(Reproducer):
    """A simple crossover reproducer using multineat."""

    rng: np.random.Generator
    innov_db_body: multineat.InnovationDatabase
    innov_db_brain: multineat.InnovationDatabase

    def __init__(
        self,
        rng: np.random.Generator,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
    ):
        """
        Initialize the reproducer.

        :param rng: The ranfom generator.
        :param innov_db_body: The innovation database for the body.
        :param innov_db_brain: The innovation database for the brain.
        """
        self.rng = rng
        self.innov_db_body = innov_db_body
        self.innov_db_brain = innov_db_brain

    def reproduce(
        self, population: npt.NDArray[np.int_], **kwargs: Any
    ) -> list[Genotype]:
        """
        Reproduce the population by crossover.

        :param population: The parent pairs.
        :param kwargs: Additional keyword arguments.
        :return: The genotypes of the children.
        :raises ValueError: If the parent population is not passed as a kwarg `parent_population`.
        """
        parent_population: list[Individual] | None = kwargs.get("parent_population")
        if parent_population is None:
            raise ValueError("No parent population given.")

        offspring_genotypes = [
            Genotype.crossover(
                parent_population[parent1_i].genotype,
                parent_population[parent2_i].genotype,
                self.rng,
            ).mutate(self.innov_db_body, self.innov_db_brain, self.rng)
            for parent1_i, parent2_i in population
        ]
        return offspring_genotypes