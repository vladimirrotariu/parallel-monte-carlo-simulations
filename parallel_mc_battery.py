# flake8: noqa
import os
import time
import logging

import numpy as np
from pydantic import BaseModel, validator, ValidationError
from typing import Optional, Union, List

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions


class BatteryConfigs(BaseModel):
    rng: Optional[str] = None
    pipeline_options: Optional[PipelineOptions] = None

    @validator("rng")
    def validate_rng(cls, rng):
        allowed_rngs = ["PCG64", "Philox", "SFC64", "MT19937"]

        if rng == None:
            rng = "PCG64"

        if rng not in allowed_rngs:
            raise ValueError(
                f"Unsupported RNG choice.\
                      Allowed options: {' ,'.join(allowed_rngs)}"
            )

        return rng


class SimulationConfigs(BaseModel):
    parameters: Union[List[float], List[int]]
    starting_point: float
    number_simulations: int
    number_points: int

    @validator("number_simulations")
    def validate_number_simulations(cls, number_simulations):
        if number_simulations < 1:
            raise ValueError(
                "The minimum number of Monte Carlo simulations\
                                               should be >= 1."
            )

        return number_simulations

    @validator("number_points")
    def validate_number_points(cls, number_points):
        if number_points < 1:
            raise ValueError(
                "The minimum number of points in a single Monte Carlo\
                                              trace should be >= 1."
            )

        return number_points


class OutputPath(BaseModel):
    output_path: str

    @validator("output_path")
    def validate_output_path(cls, output_path):
        directory_path, file_name = os.path.split(output_path)

        if not os.path.isdir(output_path):
            os.makedirs(directory_path, exist_ok=True)
            logging.info(f"Directory {directory_path} created.")

        if not os.path.isfile(output_path):
            with open(output_path, "x") as f:
                f.write("")
                logging.info(f"File {output_path} created.")

        if not os.access(output_path, os.W_OK):
            raise PermissionError(f"Could not write to the {file_name} file")


class ParallelMCBattery:
    """
    Helper class to orchestrate in parallel Monte Carlo simulations
    for an arbitrary number of models,
    with low-level parameter granularity.
    """

    def __init__(self, battery_configs):
        rng = battery_configs["rng"]
        pipeline_options = battery_configs["pipeline_options"]
        battery_configs = ParallelMCBattery.handle_validation_battery(
            rng=rng, pipeline_options=pipeline_options
        )

        type(self).rng = battery_configs.rng
        type(self).pipeline_options = battery_configs.pipeline_options

    def simulate(self, models, simulation_configs, output_paths="."):

        simulation_configs = ParallelMCBattery.handle_validation_simulation(
            simulation_configs
        )

        output_paths = ParallelMCBattery.handle_validation_output(output_paths)

        ParallelMCBattery.output_paths = output_paths

        class SimulateDoFn(beam.DoFn):
            def start_bundle(self):
                logging.info(
                    f"New bundle created at {time.time()},\
                              and sent to workers for parallel simulation..."
                )

            def process(self, element):
                model, simulation_configs, output_path = element

                number_points = simulation_configs["number_points"]
                number_simulations = simulation_configs["number_simulations"]
                starting_point = simulation_configs["starting_point"]
                parameters = simulation_configs["parameters"]
                monte_carlo_traces = []

                for _ in range(number_simulations):
                    monte_carlo_trace = model(
                        parameters,
                        starting_point,
                        number_points,
                        ParallelMCBattery.rng,
                    )
                    monte_carlo_traces.append(monte_carlo_trace)

                yield (monte_carlo_traces, output_path)

    @classmethod
    def handle_validation_battery(self, battery_configs):
        try:
            battery_configs = battery_configs or type(self).battery_configs
            rng = battery_configs["rng"]
            pipeline_options = battery_configs["pipeline_options"]

            battery_configs = BatteryConfigs(rng=rng, pipeline_options=pipeline_options)
            type(self).pipeline_options = battery_configs.mode

            rng_mapping = {
                "PCG64": np.random.PCG64,
                "Philox": np.random.Philox,
                "SFC64": np.random.SFC64,
                "MT19937": np.random.MT19937,
            }

            rng_generator = rng_mapping[battery_configs.rng]

            type(self).rng = np.random.default_rng(rng_generator())
        except ValidationError:
            logging.exception("Validation of battery configuration failed")
            raise

        return battery_configs

    @classmethod
    def handle_validation_simulation(simulation_configs):
        try:
            for simulation_config in simulation_configs:
                parameters = simulation_config["parameters"]
                starting_point = simulation_config["starting_point"]
                number_simulations = simulation_config["number_simulations"]
                number_points = simulation_config["number_points"]
                simulation_config = SimulationConfigs(
                    parameters=parameters,
                    starting_point=starting_point,
                    number_simulations=number_simulations,
                    number_points=number_points,
                )
        except KeyError:
            logging.exception(
                f"Missing parameters\
                               from simulation configuration\
                                  {str(simulation_config)}"
            )
            raise
        except ValidationError:
            logging.exception(
                f"Validation of simulation configurations\
                               failed at {str(simulation_config)}"
            )
            raise

        return simulation_configs

    @classmethod
    def handle_validation_output(self, output_paths):
        try:
            for output_path in output_paths:
                output_path = OutputPath(output_path=output_path)
        except PermissionError:
            logging.exception(f"The file for the path {output_path} cannot be accessed")
            raise

        return output_paths
