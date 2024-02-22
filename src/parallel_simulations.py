import os
import csv
import logging
from typing import Optional, Union, List

import numpy as np
import apache_beam as beam
from apache_beam.options.pipeline_options import (
    PipelineOptions,
)

from pydantic import BaseModel, validator, ValidationError


class BatteryConfigs(BaseModel):
    rng: Optional[str] = None
    pipeline_options: PipelineOptions

    @validator("rng")
    def validate_rng(cls, rng):
        allowed_rngs = [
            "PCG64",
            "Philox",
            "SFC64",
            "MT19937",
        ]

        if rng is None:
            rng = "PCG64"

        if rng not in allowed_rngs:
            raise ValueError(
                f"Unsupported RNG choice. Allowed options: {' ,'.join(allowed_rngs)}"
            )

        return rng

    class Config:
        arbitrary_types_allowed = True


class SimulationConfigs(BaseModel):
    number_simulations: int
    number_points: int
    parameters: Optional[Union[int, float, List[float], List[int]]] = None
    starting_point: Optional[Union[float, str, List[float], List[str]]] = None

    @validator("number_simulations")
    def validate_number_simulations(cls, number_simulations):
        if number_simulations < 1:
            raise ValueError(
                "The minimum number of Monte Carlo simulations should be >= 1."
            )

        return number_simulations

    @validator("number_points")
    def validate_number_points(cls, number_points):
        if number_points < 1:
            raise ValueError(
                "The minimum number of points in a single Monte Carlo trace should be >= 1."
            )

        return number_points


class OutputPath(BaseModel):
    output_path: str

    @validator("output_path")
    def validate_output_path(cls, output_path):
        directory_path, file_name = os.path.split(output_path)

        if directory_path and not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
            logging.info(f"Directory {directory_path} created.")

        if file_name and not os.path.exists(output_path):
            try:
                with open(output_path, "x") as f:
                    f.write("")
                logging.info(f"File {output_path} created.")
            except PermissionError:
                raise PermissionError(
                    f"Not enough permissions to write to the {file_name} file"
                )

        return output_path


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

        index_seeds = [i for i in range(orchestration_dimension)]

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

        class WriteToCsvDoFn(beam.DoFn):
            def process(self, element):
                simulations, output_path = element

                with open(output_path, "w") as file:
                    writer = csv.writer(file)

                    for simulation in simulations:
                        writer.writerow(simulation)

        with beam.Pipeline(options=ParallelMCBattery.pipeline_options) as pipeline:
            (
                pipeline
                | "Initialize models workbench..." >> beam.Create(input_collection)
                | "Generate Monte Carlo simulations" >> beam.ParDo(SimulateDoFn())
                | "Write simulations to output files" >> beam.ParDo(WriteToCsvDoFn())
            )

        logging.info("The Monte Carlo simulations completed succesfully.")

    @classmethod
    def handle_validation_battery(self, battery_configs):
        try:
            battery_configs = battery_configs or ParallelMCBattery.battery_configs
            rng = battery_configs["rng"]
            pipeline_options = battery_configs["pipeline_options"]

            battery_configs = BatteryConfigs(
                rng=rng, pipeline_options=pipeline_options
            )
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
    def handle_validation_simulation(self, simulation_configs):
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
    def handle_validation_output(self, output_paths, orchestration_dimension):
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
