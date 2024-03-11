import os
import sys
import logging

import numpy as np
import apache_beam as beam

from pydantic import validator, ValidationError
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from parallel_simulations.models import BatteryConfigs, SimulationConfigs, OutputPath
from parallel_simulations.utils import choose_unique_random_seeds, WriteMCTrajectoriesToCsvDoFn


class ParallelMCBattery:
    """
    Helper class to orchestrate in parallel Monte Carlo simulations
    for an arbitrary number of models,
    with low-level parameter granularity.
    """

    def __init__(self, battery_configs):
        battery_configs = ParallelMCBattery.handle_validation_battery(
            battery_configs=battery_configs
        )

        ParallelMCBattery.rng_generator = battery_configs.rng
        ParallelMCBattery.pipeline_options = battery_configs.pipeline_options

    def simulate(self, models, simulation_configs, output_paths=None):
        simulation_configs = ParallelMCBattery.handle_validation_simulation(
            simulation_configs
        )

        orchestration_dimension = len(models)

        ParallelMCBattery.output_paths = output_paths

        output_paths = ParallelMCBattery.handle_validation_output(
            output_paths, orchestration_dimension
        )

        ParallelMCBattery.output_paths = output_paths

        index_seeds = choose_unique_random_seeds(orchestration_dimension)

        input_collection = list(
            zip(
                models,
                simulation_configs,
                output_paths,
                index_seeds,
            )
        )

        class SimulateDoFn(beam.DoFn):
            def process(self, element):
                (
                    model,
                    simulation_configs,
                    output_path,
                    index_seed,
                ) = element

                rng_instance = ParallelMCBattery.rng_generator(seed=index_seed)
                rng = np.random.default_rng(rng_instance)

                number_points = simulation_configs["number_points"]
                number_simulations = simulation_configs["number_simulations"]
                starting_point = simulation_configs["starting_point"]
                parameters = simulation_configs["parameters"]
                monte_carlo_traces = []

                for _ in range(number_simulations):
                    monte_carlo_trace = []

                    if starting_point is None and parameters is None:
                        monte_carlo_trace = model(
                            number_points,
                            rng,
                        )

                    if starting_point is None and parameters is not None:
                        monte_carlo_trace = model(number_points, rng, parameters)

                    if starting_point is not None and parameters is not None:
                        monte_carlo_trace = model(
                            number_points,
                            rng,
                            parameters,
                            starting_point,
                        )

                    monte_carlo_traces.append(monte_carlo_trace)
                yield (monte_carlo_traces, output_path)

        with beam.Pipeline(options=ParallelMCBattery.pipeline_options) as pipeline:
            (
                pipeline
                | "Initialize models workbench..." >> beam.Create(input_collection)
                | "Generate Monte Carlo simulations" >> beam.ParDo(SimulateDoFn())
                | "Write simulations to output files"
                >> beam.ParDo(WriteMCTrajectoriesToCsvDoFn())
            )

        logging.info("The Monte Carlo simulations completed succesfully.")

    @classmethod
    def handle_validation_battery(self, battery_configs: dict[Any]) -> dict[Any]:
        try:
            battery_configs = battery_configs or ParallelMCBattery.battery_configs
            rng = battery_configs["rng"]
            pipeline_options = battery_configs["pipeline_options"]

            battery_configs = BatteryConfigs(rng=rng, pipeline_options=pipeline_options)
            ParallelMCBattery.pipeline_options = battery_configs.pipeline_options

            rng_mapping = {
                "PCG64": np.random.PCG64,
                "Philox": np.random.Philox,
                "SFC64": np.random.SFC64,
                "MT19937": np.random.MT19937,
            }

            rng_generator = rng_mapping[battery_configs.rng]

            battery_configs.rng = rng_generator
        except KeyError:
            logging.exception("Missing battery configurations")
        except ValidationError:
            logging.exception("Validation of battery configuration failed")
            raise

        return battery_configs

    @classmethod
    def handle_validation_simulation(self, simulation_configs: dict[Any]) -> dict[Any]:
        for simulation_config in simulation_configs:
            if "parameters" not in simulation_config:
                simulation_config["parameters"] = None
            if "starting_point" not in simulation_config:
                simulation_config["starting_point"] = None

        for simulation_config in simulation_configs:
            try:
                parameters = simulation_config["parameters"]
                starting_point = simulation_config["starting_point"]
                number_simulations = simulation_config["number_simulations"]
                number_points = simulation_config["number_points"]
                simulation_config = SimulationConfigs(
                    number_simulations=number_simulations,
                    number_points=number_points,
                    parameters=parameters,
                    starting_point=starting_point,
                )
            except KeyError:
                logging.exception(
                    f"Missing parameters from simulation configuration {str(simulation_config)}"
                )
                raise
            except ValidationError:
                logging.exception(
                    f"Validation of simulation configurations failed at {str(simulation_config)}"
                )
                raise

        return simulation_configs

    @classmethod
    def handle_validation_output(
        self, output_paths: list[str], orchestration_dimension
    ) -> list[str]:
        if output_paths is None:
            output_paths = ParallelMCBattery.output_paths or [
                os.path.join(".", f"{i}.txt") for i in range(orchestration_dimension)
            ]
        else:
            try:
                for output_path in output_paths:
                    output_path = OutputPath(output_path=output_path)
            except PermissionError:
                logging.exception(
                    f"The file for the path {output_path} cannot be accessed"
                )
                raise

        return output_paths
