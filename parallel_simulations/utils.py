import sys
import csv
import random

import apache_beam as beam


def choose_unique_random_seeds(number_seeds):
    seeds = set()
    while len(seeds) < number_seeds:
        seeds.add(random.randint(1, sys.maxsize))
    return list(seeds)


class WriteMCTrajectoriesToCsvDoFn(beam.DoFn):
    def process(self, element):
        simulations, output_path = element

        with open(output_path, "w") as file:
            writer = csv.writer(file)

            for simulation in simulations:
                writer.writerow(simulation)
